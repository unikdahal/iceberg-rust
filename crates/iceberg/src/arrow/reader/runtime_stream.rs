// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Decoding with row-group-local selections and boundary refresh. When a
//! runtime predicate provider is configured, a task is decoded with parquet's
//! push decoder, so each row group keeps its own page and positional-delete
//! selection. At a row-group boundary after a newer publication, the remaining
//! groups are pruned and their page selections and the row filter refreshed
//! before any further column ranges are fetched.
//!
//! Cost: a boundary with an unchanged generation costs one atomic load. Each
//! new generation costs one pass over the remaining row groups (statistics and
//! page index) and one decoder rebuild.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::RecordBatch;
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ParquetRecordBatchReader, RowFilter, RowSelection,
};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::{ParquetPushDecoder, RowGroupSelection};
use parquet::file::metadata::ParquetMetaData;

use super::runtime_predicate::{RuntimePredicates, check_runtime_predicate_columns};
use super::{ArrowFileReader, ArrowReader};
use crate::arrow::ScanMetrics;
use crate::expr::BoundPredicate;
use crate::expr::visitors::row_group_metrics_evaluator::RowGroupMetricsEvaluator;
use crate::scan::{ArrowRecordBatchStream, FileScanTask};
use crate::{Error, Result};

/// Splits a selection over the concatenated rows of `row_groups` (ascending)
/// into one selection per row group. `None` selects every row.
pub(super) fn split_row_selection(
    metadata: &ParquetMetaData,
    row_groups: &[usize],
    mut selection: Option<RowSelection>,
) -> Vec<RowGroupSelection> {
    row_groups
        .iter()
        .map(|&index| {
            let rows = metadata.row_group(index).num_rows() as usize;
            let local = selection
                .as_mut()
                .map(|selection| selection.split_off(rows));
            RowGroupSelection::new(index, local)
        })
        .collect()
}

/// A predicate resolved against the open file.
#[derive(Clone)]
pub(super) struct ResolvedPredicate {
    pub(super) predicate: Arc<BoundPredicate>,
    pub(super) field_ids: HashSet<i32>,
    pub(super) field_id_map: HashMap<i32, usize>,
}

/// What a task needs to refresh its runtime predicate at row-group boundaries.
pub(super) struct BoundaryRefresh {
    pub(super) predicates: Arc<RuntimePredicates>,
    /// The generation the task last looked at.
    pub(super) seen_generation: u64,
    /// The planned and equality-delete predicate.
    pub(super) planned: Option<ResolvedPredicate>,
    /// The runtime predicate in force. Publications only tighten, so a newer
    /// one replaces it, and a later `None` leaves it in place: it never
    /// restores groups, pages or rows already skipped.
    pub(super) runtime: Option<ResolvedPredicate>,
    pub(super) row_selection_enabled: bool,
    pub(super) task: FileScanTask,
    pub(super) use_position_fallback: bool,
    pub(super) metrics: ScanMetrics,
}

pub(super) struct RuntimePrunedStream {
    decoder: Option<ParquetPushDecoder>,
    active_reader: Option<ParquetRecordBatchReader>,
    file_reader: ArrowFileReader,
    metadata: ArrowReaderMetadata,
    /// Selections installed in the decoder for the row groups not yet started,
    /// in file order.
    selections: Vec<RowGroupSelection>,
    refresh: BoundaryRefresh,
}

impl RuntimePrunedStream {
    pub(super) fn new(
        decoder: ParquetPushDecoder,
        file_reader: ArrowFileReader,
        metadata: ArrowReaderMetadata,
        selections: Vec<RowGroupSelection>,
        refresh: BoundaryRefresh,
    ) -> Self {
        Self {
            decoder: Some(decoder),
            active_reader: None,
            file_reader,
            metadata,
            selections,
            refresh,
        }
    }

    fn refresh_at_boundary(&mut self) -> Result<()> {
        let decoder = self
            .decoder
            .as_ref()
            .expect("decoder exists while streaming");
        if !decoder.is_at_row_group_boundary() || decoder.row_groups_remaining() == 0 {
            return Ok(());
        }
        let generation = self.refresh.predicates.generation();
        if generation == self.refresh.seen_generation {
            return Ok(());
        }
        self.refresh.seen_generation = generation;
        // `None`, a failed publication or a predicate already in force keep the
        // current restrictions.
        let Some(predicate) = self.refresh.predicates.current(
            &self.refresh.task.schema_ref(),
            self.refresh.task.case_sensitive(),
            self.refresh.task.data_file_path(),
        ) else {
            return Ok(());
        };
        if self
            .refresh
            .runtime
            .as_ref()
            .is_some_and(|runtime| Arc::ptr_eq(&runtime.predicate, &predicate))
        {
            return Ok(());
        }

        // Bring the local plan in lock-step with the decoder, which may have
        // skipped groups internally (for example an empty static selection).
        let Some(next) = decoder.peek_next_row_group()? else {
            self.selections.clear();
            return Ok(());
        };
        let started = self
            .selections
            .iter()
            .take_while(|selection| selection.row_group_index() != next)
            .count();
        self.selections.drain(..started);

        let (runtime, selections) = match self.restrict_remaining(predicate) {
            Ok(restricted) => restricted,
            Err(error) => {
                tracing::debug!(
                    "Skipping live runtime predicate for {}: {error}",
                    self.refresh.task.data_file_path()
                );
                return Ok(());
            }
        };
        let Some(row_filter) = self.compile_row_filter(&runtime) else {
            return Ok(());
        };
        let pruned = self.selections.len() - selections.len();
        self.selections = selections;
        self.refresh.runtime = Some(runtime);
        self.refresh.metrics.record_runtime_refresh(pruned);
        self.rebuild_decoder(next, row_filter)
    }

    /// Returns the remaining selections narrowed by `predicate`: row groups
    /// whose statistics cannot match are dropped, and the page selections of
    /// the rest are intersected with the predicate's page selections. An error
    /// makes the predicate unusable for this file.
    fn restrict_remaining(
        &self,
        predicate: Arc<BoundPredicate>,
    ) -> Result<(ResolvedPredicate, Vec<RowGroupSelection>)> {
        let parquet_metadata = self.metadata.metadata();
        let task = &self.refresh.task;
        check_runtime_predicate_columns(
            &predicate,
            parquet_metadata.file_metadata().schema_descr(),
            self.metadata.schema(),
            task.schema(),
            self.refresh.use_position_fallback,
        )?;
        let (field_ids, field_id_map) = ArrowReader::build_field_id_set_and_map(
            parquet_metadata.file_metadata().schema_descr(),
            self.metadata.schema(),
            &predicate,
            self.refresh.use_position_fallback,
        )?;
        let mut kept = Vec::with_capacity(self.selections.len());
        for selection in &self.selections {
            if RowGroupMetricsEvaluator::eval(
                &predicate,
                parquet_metadata.row_group(selection.row_group_index()),
                &field_id_map,
                task.schema(),
            )? {
                kept.push(selection.clone());
            }
        }
        if self.refresh.row_selection_enabled && !kept.is_empty() {
            let indices: Vec<usize> = kept
                .iter()
                .map(RowGroupSelection::row_group_index)
                .collect();
            // Page pruning only narrows reads, and the row filter still applies
            // the predicate, so a failure keeps the groups without refinement.
            match ArrowReader::get_row_selection_for_filter_predicate(
                &predicate,
                parquet_metadata,
                &Some(indices.clone()),
                &field_id_map,
                task.schema(),
            ) {
                Ok(Some(pages)) => {
                    let pages = split_row_selection(parquet_metadata, &indices, Some(pages));
                    kept = kept
                        .into_iter()
                        .zip(pages)
                        .map(|(current, runtime)| {
                            let combined = match (current.selection(), runtime.selection()) {
                                (Some(current), Some(runtime)) => {
                                    Some(current.intersection(runtime))
                                }
                                (Some(current), None) => Some(current.clone()),
                                (None, runtime) => runtime.cloned(),
                            };
                            RowGroupSelection::new(current.row_group_index(), combined)
                        })
                        .collect();
                }
                Ok(None) => {}
                Err(error) => tracing::debug!(
                    "Skipping live page pruning for {}: {error}",
                    task.data_file_path()
                ),
            }
        }
        Ok((
            ResolvedPredicate {
                predicate,
                field_ids,
                field_id_map,
            },
            kept,
        ))
    }

    /// Compiles the planned predicate and `runtime` into one row filter. If the
    /// runtime part cannot be compiled the predicate is not adopted.
    fn compile_row_filter(&self, runtime: &ResolvedPredicate) -> Option<RowFilter> {
        let schema_descr = self.metadata.metadata().file_metadata().schema_descr();
        let planned = self.refresh.planned.iter().chain([runtime]);
        let predicates: Vec<_> = planned
            .map(|resolved| {
                (
                    resolved.predicate.as_ref(),
                    &resolved.field_ids,
                    &resolved.field_id_map,
                )
            })
            .collect();
        match ArrowReader::get_arrow_predicate(&predicates, schema_descr) {
            Ok(predicate) => Some(RowFilter::new(vec![predicate])),
            Err(error) => {
                tracing::debug!(
                    "Skipping live row filter for {}: {error}",
                    self.refresh.task.data_file_path()
                );
                None
            }
        }
    }

    /// Rebuilds the decoder once with the current selections and row filter.
    fn rebuild_decoder(&mut self, next: usize, row_filter: RowFilter) -> Result<()> {
        let next_pruned = self
            .selections
            .first()
            .is_none_or(|selection| selection.row_group_index() != next);
        let decoder = self.decoder.take().expect("decoder exists while streaming");
        let mut decoder = decoder
            .into_builder()?
            .with_row_group_selections(self.selections.clone())
            .with_row_filter(row_filter)
            .build()?;
        if next_pruned {
            // Only the next group's ranges can be buffered at a boundary.
            decoder.clear_all_ranges();
        }
        self.decoder = Some(decoder);
        Ok(())
    }

    async fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        loop {
            if let Some(reader) = self.active_reader.as_mut() {
                if let Some(batch) = reader.next() {
                    return Ok(Some(batch?));
                }
                self.active_reader = None;
            }
            self.refresh_at_boundary()?;
            let decoder = self
                .decoder
                .as_mut()
                .expect("decoder exists while streaming");
            match decoder.try_next_reader()? {
                DecodeResult::NeedsData(ranges) => {
                    let bytes = self.file_reader.get_byte_ranges(ranges.clone()).await?;
                    decoder.push_ranges(ranges, bytes)?;
                }
                DecodeResult::Data(reader) => self.active_reader = Some(reader),
                DecodeResult::Finished => return Ok(None),
            }
        }
    }

    pub(super) fn into_stream(self) -> ArrowRecordBatchStream {
        Box::pin(futures::stream::try_unfold(self, |mut state| async move {
            let batch = state.next_batch().await?;
            Ok::<_, Error>(batch.map(|batch| (batch, state)))
        }))
    }
}

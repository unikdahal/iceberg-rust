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

//! Boundary-aware decoding with task-local generations. At a row-group
//! boundary after a newer runtime publication, the remaining row groups are
//! pruned and their page selections and the row filter refreshed, before any
//! further column ranges are fetched.
//!
//! Cost: a boundary with an unchanged generation costs one atomic load. Each
//! accepted publication change costs one pass over the remaining row groups
//! (statistics and page index) and one decoder rebuild, so a file with `R`
//! row groups and `C` observed changes costs `O(R * C)`, independent of how
//! many row groups a stable predicate prunes pages in.

use arrow_array::RecordBatch;
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReader, RowFilter};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::{ParquetPushDecoder, RowGroupSelection};

use super::runtime_predicate::{RuntimePredicateState, check_runtime_predicate_columns};
use super::{ArrowFileReader, ArrowReader};
use crate::arrow::ScanMetrics;
use crate::expr::BoundPredicate;
use crate::expr::visitors::row_group_metrics_evaluator::RowGroupMetricsEvaluator;
use crate::scan::{ArrowRecordBatchStream, FileScanTask};
use crate::{Error, Result};

pub(super) struct RuntimePrunedParquetStream {
    decoder: Option<ParquetPushDecoder>,
    active_reader: Option<ParquetRecordBatchReader>,
    file_reader: ArrowFileReader,
    metadata: ArrowReaderMetadata,
    /// Selections installed in the decoder for the row groups not yet started,
    /// in file order. They only ever narrow: every accepted publication stays
    /// valid for the rest of the scan, so its page selections are intersected
    /// into these and kept even if a later generation is `None`.
    selections: Vec<RowGroupSelection>,
    /// Planned and equality-delete predicate, built once at task open.
    base_predicate: Option<BoundPredicate>,
    row_selection_enabled: bool,
    runtime: RuntimePredicateState,
    task: FileScanTask,
    use_position_fallback: bool,
    metrics: ScanMetrics,
    accepted_runtime: bool,
}

impl RuntimePrunedParquetStream {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        decoder: ParquetPushDecoder,
        file_reader: ArrowFileReader,
        metadata: ArrowReaderMetadata,
        selections: Vec<RowGroupSelection>,
        base_predicate: Option<BoundPredicate>,
        row_selection_enabled: bool,
        runtime: RuntimePredicateState,
        task: FileScanTask,
        use_position_fallback: bool,
        metrics: ScanMetrics,
    ) -> Self {
        debug_assert!(
            selections
                .windows(2)
                .all(|pair| pair[0].row_group_index() < pair[1].row_group_index()),
            "row-group selections must be in strictly increasing file order"
        );
        metrics.record_runtime_live_pruning_task();
        let accepted_runtime = runtime.predicate().is_some();
        Self {
            decoder: Some(decoder),
            active_reader: None,
            file_reader,
            metadata,
            selections,
            base_predicate,
            row_selection_enabled,
            runtime,
            task,
            use_position_fallback,
            metrics,
            accepted_runtime,
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
        let changed = match self
            .runtime
            .refresh_if_changed(self.task.schema_ref(), self.task.case_sensitive())
        {
            Ok(changed) => {
                if changed {
                    self.metrics.record_runtime_predicate_refresh();
                }
                changed
            }
            Err(error) => {
                tracing::debug!(
                    "Skipping live runtime publication for {}: {error}",
                    self.task.data_file_path()
                );
                // The failed publication cleared the advisory predicate.
                true
            }
        };
        if !changed {
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

        if let Some(predicate) = self.runtime.predicate().cloned() {
            let restricted = check_runtime_predicate_columns(
                &predicate,
                self.metadata.metadata().file_metadata().schema_descr(),
                self.metadata.schema(),
                self.task.schema(),
                self.use_position_fallback,
            )
            .and_then(|()| self.restrict_remaining(&predicate));
            match restricted {
                Ok(selections) => {
                    if !self.accepted_runtime {
                        self.metrics.record_runtime_predicate_task();
                        self.accepted_runtime = true;
                    }
                    self.metrics
                        .record_runtime_row_groups_considered(self.selections.len());
                    self.metrics.record_runtime_row_groups_pruned_live(
                        self.selections.len() - selections.len(),
                    );
                    self.selections = selections;
                }
                Err(error) => {
                    tracing::debug!(
                        "Skipping live runtime predicate for {}: {error}",
                        self.task.data_file_path()
                    );
                    self.runtime.reject_current();
                }
            }
        }
        self.rebuild_decoder(next)
    }

    /// Returns the remaining selections narrowed by `predicate`: row groups
    /// whose statistics cannot match are dropped, and the page selections of
    /// the rest are intersected with the predicate's page selections. An error
    /// makes the predicate unusable for this file.
    fn restrict_remaining(&self, predicate: &BoundPredicate) -> Result<Vec<RowGroupSelection>> {
        let (_, field_id_map) = ArrowReader::build_field_id_set_and_map(
            self.metadata.metadata().file_metadata().schema_descr(),
            self.metadata.schema(),
            predicate,
            self.use_position_fallback,
        )?;
        let mut kept = Vec::with_capacity(self.selections.len());
        for selection in &self.selections {
            if RowGroupMetricsEvaluator::eval(
                predicate,
                self.metadata
                    .metadata()
                    .row_group(selection.row_group_index()),
                &field_id_map,
                self.task.schema(),
            )? {
                kept.push(selection.clone());
            }
        }
        if !self.row_selection_enabled || kept.is_empty() {
            return Ok(kept);
        }
        let indices: Vec<usize> = kept
            .iter()
            .map(RowGroupSelection::row_group_index)
            .collect();
        let pages = match ArrowReader::get_row_group_selections_for_filter_predicate(
            Some(predicate),
            &self.metadata,
            &indices,
            self.task.schema(),
            self.use_position_fallback,
        ) {
            Ok(pages) => pages,
            Err(error) => {
                // Row-group pruning stays valid without page refinement.
                tracing::debug!(
                    "Skipping live page pruning for {}: {error}",
                    self.task.data_file_path()
                );
                return Ok(kept);
            }
        };
        Ok(kept
            .into_iter()
            .zip(pages)
            .map(|(current, runtime)| {
                debug_assert_eq!(current.row_group_index(), runtime.row_group_index());
                let combined = match (current.selection(), runtime.selection()) {
                    (Some(current), Some(runtime)) => Some(current.intersection(runtime)),
                    (Some(current), None) => Some(current.clone()),
                    (None, runtime) => runtime.cloned(),
                };
                RowGroupSelection::new(current.row_group_index(), combined)
            })
            .collect())
    }

    /// Rebuilds the decoder once with the current selections and the row
    /// filter for the current publication.
    fn rebuild_decoder(&mut self, next: usize) -> Result<()> {
        // Compile before taking the decoder, so an error cannot lose it.
        let row_filter = self.compile_row_filter()?;
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
        self.metrics.record_runtime_decoder_rebuild();
        Ok(())
    }

    /// Compiles the base predicate AND the current runtime predicate, falling
    /// back to the base predicate when the runtime part cannot be compiled.
    fn compile_row_filter(&mut self) -> Result<RowFilter> {
        let compile = |predicate: Option<&BoundPredicate>| -> Result<RowFilter> {
            let Some(predicate) = predicate else {
                return Ok(RowFilter::new(vec![]));
            };
            let schema_descr = self.metadata.metadata().file_metadata().schema_descr();
            let (ids, map) = ArrowReader::build_field_id_set_and_map(
                schema_descr,
                self.metadata.schema(),
                predicate,
                self.use_position_fallback,
            )?;
            ArrowReader::get_row_filter(predicate, schema_descr, &ids, &map)
        };
        let effective = match (&self.base_predicate, self.runtime.predicate()) {
            (Some(base), Some(runtime)) => Some(base.clone().and(runtime.clone())),
            (Some(base), None) => Some(base.clone()),
            (None, Some(runtime)) => Some(runtime.clone()),
            (None, None) => None,
        };
        match compile(effective.as_ref()) {
            Ok(filter) => Ok(filter),
            Err(error) => {
                tracing::debug!(
                    "Skipping live row filter for {}: {error}",
                    self.task.data_file_path()
                );
                let filter = compile(self.base_predicate.as_ref())?;
                self.runtime.reject_current();
                Ok(filter)
            }
        }
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

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

//! Boundary-aware decoding with task-local generations. The optional local
//! selection path refreshes the next row group's pages and row filter before
//! fetching its column ranges; flattened selections retain task-open pruning.

use std::collections::VecDeque;

use arrow_array::RecordBatch;
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReader};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::ParquetPushDecoder;

use super::runtime_predicate::RuntimePredicateState;
use super::{ArrowFileReader, ArrowReader};
use crate::arrow::ScanMetrics;
use crate::expr::visitors::row_group_metrics_evaluator::RowGroupMetricsEvaluator;
use crate::scan::{ArrowRecordBatchStream, FileScanTask};
use crate::{Error, Result};

pub(super) struct RuntimePrunedParquetStream {
    decoder: Option<ParquetPushDecoder>,
    active_reader: Option<ParquetRecordBatchReader>,
    file_reader: ArrowFileReader,
    metadata: ArrowReaderMetadata,
    remaining: VecDeque<usize>,
    runtime: RuntimePredicateState,
    task: FileScanTask,
    use_position_fallback: bool,
    metrics: ScanMetrics,
    accepted_runtime: bool,
    #[cfg(feature = "runtime-row-group-selections")]
    row_group_selections: Option<Vec<parquet::arrow::push_decoder::RowGroupSelection>>,
    #[cfg(feature = "runtime-row-group-selections")]
    base_predicate: Option<crate::expr::BoundPredicate>,
    #[cfg(feature = "runtime-row-group-selections")]
    row_selection_enabled: bool,
    #[cfg(feature = "runtime-row-group-selections")]
    prepared_next: Option<usize>,
}

impl RuntimePrunedParquetStream {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        decoder: ParquetPushDecoder,
        file_reader: ArrowFileReader,
        metadata: ArrowReaderMetadata,
        remaining: Vec<usize>,
        runtime: RuntimePredicateState,
        task: FileScanTask,
        use_position_fallback: bool,
        metrics: ScanMetrics,
    ) -> Self {
        metrics.record_runtime_live_pruning_task();
        let accepted_runtime = runtime.predicate().is_some();
        Self {
            decoder: Some(decoder),
            active_reader: None,
            file_reader,
            metadata,
            remaining: remaining.into(),
            runtime,
            task,
            use_position_fallback,
            metrics,
            accepted_runtime,
            #[cfg(feature = "runtime-row-group-selections")]
            row_group_selections: None,
            #[cfg(feature = "runtime-row-group-selections")]
            base_predicate: None,
            #[cfg(feature = "runtime-row-group-selections")]
            row_selection_enabled: false,
            #[cfg(feature = "runtime-row-group-selections")]
            prepared_next: None,
        }
    }

    #[cfg(feature = "runtime-row-group-selections")]
    pub(super) fn with_row_group_selections(
        mut self,
        selections: Vec<parquet::arrow::push_decoder::RowGroupSelection>,
        base_predicate: Option<crate::expr::BoundPredicate>,
        row_selection_enabled: bool,
    ) -> Self {
        // Cache static pages and deletes separately from the changing runtime
        // selection. Equality-delete predicate construction remains task-local.
        self.row_group_selections = Some(selections);
        self.base_predicate = base_predicate;
        self.row_selection_enabled = row_selection_enabled;
        self
    }

    fn refresh_at_boundary(&mut self) -> Result<()> {
        let decoder = self
            .decoder
            .as_ref()
            .expect("decoder exists while streaming");
        if !decoder.is_at_row_group_boundary() || decoder.row_groups_remaining() == 0 {
            return Ok(());
        }
        while self.remaining.len() > decoder.row_groups_remaining() {
            self.remaining.pop_front();
        }
        let changed = match self
            .runtime
            .refresh_if_changed(self.task.schema_ref(), self.task.case_sensitive())
        {
            Ok(false) => false,
            Ok(true) => {
                self.metrics.record_runtime_predicate_refresh();
                true
            }
            Err(error) => {
                tracing::debug!(
                    "Skipping live runtime publication for {}: {error}",
                    self.task.data_file_path()
                );
                // The failed publication cleared the advisory predicate.
                // Preserve the static/equality/delete constraints.
                return self.prepare_next_row_group(true);
            }
        };
        if !changed {
            return self.prepare_next_row_group(false);
        }
        let Some(predicate) = self.runtime.predicate() else {
            return self.prepare_next_row_group(true);
        };
        if !self.accepted_runtime {
            self.metrics.record_runtime_predicate_task();
            self.accepted_runtime = true;
        }
        // The base/equality-delete predicate was built once at task open.
        // Reevaluate only remaining row groups on a new runtime generation.
        let keep: Result<Vec<usize>> = (|| {
            let (_, field_id_map) = ArrowReader::build_field_id_set_and_map(
                self.metadata.metadata().file_metadata().schema_descr(),
                self.metadata.schema(),
                predicate,
                self.use_position_fallback,
            )?;
            self.remaining
                .iter()
                .copied()
                .filter_map(|idx| {
                    match RowGroupMetricsEvaluator::eval(
                        predicate,
                        self.metadata.metadata().row_group(idx),
                        &field_id_map,
                        self.task.schema(),
                    ) {
                        Ok(true) => Some(Ok(idx)),
                        Ok(false) => None,
                        Err(error) => Some(Err(error)),
                    }
                })
                .collect()
        })();
        let keep = match keep {
            Ok(keep) => keep,
            Err(error) => {
                tracing::debug!(
                    "Skipping live runtime statistics for {}: {error}",
                    self.task.data_file_path()
                );
                return self.prepare_next_row_group(true);
            }
        };
        let pruned = self.remaining.len() - keep.len();
        self.metrics
            .record_runtime_row_groups_considered(self.remaining.len());
        if pruned != 0 {
            let decoder = self.decoder.take().expect("decoder exists while streaming");
            let builder = decoder.into_builder()?;
            #[cfg(feature = "runtime-row-group-selections")]
            let builder = if let Some(selections) = self.row_group_selections.as_mut() {
                selections
                    .retain(|selection| keep.binary_search(&selection.row_group_index()).is_ok());
                builder.with_row_group_selections(selections.clone())
            } else {
                builder.with_row_groups(keep.clone())
            };
            #[cfg(not(feature = "runtime-row-group-selections"))]
            let builder = builder.with_row_groups(keep.clone());
            self.decoder = Some(builder.build()?);
            self.remaining = keep.into();
            self.metrics.record_runtime_row_groups_pruned_live(pruned);
        }
        self.prepare_next_row_group(true)
    }

    #[cfg(not(feature = "runtime-row-group-selections"))]
    fn prepare_next_row_group(&mut self, _changed: bool) -> Result<()> {
        Ok(())
    }

    #[cfg(feature = "runtime-row-group-selections")]
    fn prepare_next_row_group(&mut self, changed: bool) -> Result<()> {
        use parquet::arrow::arrow_reader::RowFilter;
        use parquet::arrow::push_decoder::RowGroupSelection;

        let Some(base_selections) = self.row_group_selections.as_ref() else {
            return Ok(());
        };
        let Some(&next) = self.remaining.front() else {
            return Ok(());
        };
        if !changed
            && (!self.row_selection_enabled
                || self.prepared_next == Some(next)
                || self.runtime.predicate().is_none())
        {
            return Ok(());
        }
        let mut selections: Vec<_> = base_selections
            .iter()
            .filter(|selection| {
                self.remaining
                    .iter()
                    .any(|&idx| idx == selection.row_group_index())
            })
            .cloned()
            .collect();
        if self.row_selection_enabled
            && let Some(predicate) = self.runtime.predicate()
        {
            // Evaluate pages only for the next group, using its local coordinates.
            // A missing/unsupported page index leaves the static/delete mask intact.
            match ArrowReader::get_row_group_selections_for_filter_predicate(
                Some(predicate),
                &self.metadata,
                &[next],
                self.task.schema(),
                self.use_position_fallback,
            ) {
                Ok(runtime) => {
                    let selection = selections
                        .iter_mut()
                        .find(|selection| selection.row_group_index() == next)
                        .expect("remaining group has its static selection");
                    let combined = match (selection.selection(), runtime[0].selection()) {
                        (Some(base), Some(runtime)) => Some(base.intersection(runtime)),
                        (None, Some(runtime)) => Some(runtime.clone()),
                        (Some(base), None) => Some(base.clone()),
                        (None, None) => None,
                    };
                    *selection = RowGroupSelection::new(next, combined);
                }
                Err(error) => tracing::debug!(
                    "Skipping live page pruning for {}: {error}",
                    self.task.data_file_path()
                ),
            }
        }
        let row_filter = if changed {
            let effective = match (&self.base_predicate, self.runtime.predicate()) {
                (Some(base), Some(runtime)) => Some(base.clone().and(runtime.clone())),
                (Some(base), None) => Some(base.clone()),
                (None, Some(runtime)) => Some(runtime.clone()),
                (None, None) => None,
            };
            let compile = |predicate: Option<&crate::expr::BoundPredicate>| -> Result<RowFilter> {
                let Some(predicate) = predicate else {
                    return Ok(RowFilter::new(vec![]));
                };
                let (ids, map) = ArrowReader::build_field_id_set_and_map(
                    self.metadata.metadata().file_metadata().schema_descr(),
                    self.metadata.schema(),
                    predicate,
                    self.use_position_fallback,
                )?;
                ArrowReader::get_row_filter(
                    predicate,
                    self.metadata.metadata().file_metadata().schema_descr(),
                    &ids,
                    &map,
                )
            };
            Some(match compile(effective.as_ref()) {
                Ok(filter) => filter,
                Err(error) => {
                    tracing::debug!(
                        "Skipping live row filter for {}: {error}",
                        self.task.data_file_path()
                    );
                    compile(self.base_predicate.as_ref())?
                }
            })
        } else {
            None
        };
        let decoder = self.decoder.take().expect("decoder exists while streaming");
        let mut builder = decoder
            .into_builder()?
            .with_row_group_selections(selections);
        if let Some(filter) = row_filter {
            builder = builder.with_row_filter(filter);
        }
        self.decoder = Some(builder.build()?);
        self.prepared_next = Some(next);
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
                DecodeResult::Data(reader) => {
                    // Some static filters can consume empty groups internally.
                    // The remaining count tracks their removal too, avoiding
                    // guessed row offsets or accidental rereading.
                    while self.remaining.len() > decoder.row_groups_remaining() {
                        self.remaining.pop_front();
                    }
                    self.active_reader = Some(reader);
                }
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

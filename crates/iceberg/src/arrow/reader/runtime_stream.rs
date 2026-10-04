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

//! Boundary-aware decoding with task-local generations. At each row-group
//! boundary a newer runtime publication can prune the remaining row groups and
//! refresh the next group's page selection and row filter before any of its
//! column ranges are fetched.

use arrow_array::RecordBatch;
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReader, RowFilter};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::{ParquetPushDecoder, RowGroupSelection};

use super::runtime_predicate::RuntimePredicateState;
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
    /// Static page and positional-delete selections of the row groups not yet
    /// started, in file order. Runtime page selections are never stored here,
    /// so a later publication always starts from the static masks.
    selections: Vec<RowGroupSelection>,
    /// Planned and equality-delete predicate, built once at task open.
    base_predicate: Option<BoundPredicate>,
    row_selection_enabled: bool,
    runtime: RuntimePredicateState,
    task: FileScanTask,
    use_position_fallback: bool,
    metrics: ScanMetrics,
    accepted_runtime: bool,
    /// The next row group whose runtime page selection is already installed.
    prepared_next: Option<usize>,
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
            prepared_next: None,
        }
    }

    fn decoder(&self) -> &ParquetPushDecoder {
        self.decoder
            .as_ref()
            .expect("decoder exists while streaming")
    }

    fn refresh_at_boundary(&mut self) -> Result<()> {
        let decoder = self.decoder();
        if !decoder.is_at_row_group_boundary() || decoder.row_groups_remaining() == 0 {
            return Ok(());
        }
        // The decoder may skip groups internally, for example when a static
        // selection is empty. Keep the local plan in lock-step with it.
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
        if changed && let Some(predicate) = self.runtime.predicate().cloned() {
            if !self.accepted_runtime {
                self.metrics.record_runtime_predicate_task();
                self.accepted_runtime = true;
            }
            match self.matching_row_groups(&predicate) {
                Ok(keep) => self.prune_remaining(keep, next)?,
                Err(error) => {
                    tracing::debug!(
                        "Skipping live runtime statistics for {}: {error}",
                        self.task.data_file_path()
                    );
                    self.runtime.reject_current();
                }
            }
        }
        self.prepare_next_row_group(changed)
    }

    /// Returns the remaining selections whose statistics might match
    /// `predicate`. An error makes the predicate advisory-only for this task.
    fn matching_row_groups(&self, predicate: &BoundPredicate) -> Result<Vec<RowGroupSelection>> {
        let (_, field_id_map) = ArrowReader::build_field_id_set_and_map(
            self.metadata.metadata().file_metadata().schema_descr(),
            self.metadata.schema(),
            predicate,
            self.use_position_fallback,
        )?;
        let mut keep = Vec::with_capacity(self.selections.len());
        for selection in &self.selections {
            if RowGroupMetricsEvaluator::eval(
                predicate,
                self.metadata
                    .metadata()
                    .row_group(selection.row_group_index()),
                &field_id_map,
                self.task.schema(),
            )? {
                keep.push(selection.clone());
            }
        }
        self.metrics
            .record_runtime_row_groups_considered(self.selections.len());
        Ok(keep)
    }

    /// Rebuilds the decoder over `keep` when it removes any remaining group.
    fn prune_remaining(&mut self, keep: Vec<RowGroupSelection>, next: usize) -> Result<()> {
        let pruned = self.selections.len() - keep.len();
        if pruned == 0 {
            return Ok(());
        }
        let next_pruned = keep
            .first()
            .is_none_or(|selection| selection.row_group_index() != next);
        let decoder = self.decoder.take().expect("decoder exists while streaming");
        let mut decoder = decoder
            .into_builder()?
            .with_row_group_selections(keep.clone())
            .build()?;
        if next_pruned {
            // Only the next group's ranges can be buffered at a boundary.
            decoder.clear_all_ranges();
        }
        self.decoder = Some(decoder);
        self.selections = keep;
        self.prepared_next = None;
        self.metrics.record_runtime_row_groups_pruned_live(pruned);
        Ok(())
    }

    /// Installs the next row group's runtime page selection and, after a
    /// publication change, the matching row filter. Rebuilding passes every
    /// remaining selection, so each rebuild is linear in the remaining groups;
    /// rebuilds happen only on a publication change or a page reduction.
    fn prepare_next_row_group(&mut self, changed: bool) -> Result<()> {
        let Some(next) = self
            .selections
            .first()
            .map(RowGroupSelection::row_group_index)
        else {
            return Ok(());
        };
        if !changed
            && (!self.row_selection_enabled
                || self.prepared_next == Some(next)
                || self.runtime.predicate().is_none())
        {
            return Ok(());
        }
        let runtime_selection = match self.runtime.predicate() {
            Some(predicate) if self.row_selection_enabled => {
                // Evaluate only the next group's pages. A stable predicate with no
                // page reduction keeps the existing decoder.
                match ArrowReader::get_row_group_selections_for_filter_predicate(
                    Some(predicate),
                    &self.metadata,
                    &[next],
                    self.task.schema(),
                    self.use_position_fallback,
                ) {
                    Ok(runtime) => runtime[0].selection().cloned(),
                    Err(error) => {
                        tracing::debug!(
                            "Skipping live page pruning for {}: {error}",
                            self.task.data_file_path()
                        );
                        None
                    }
                }
            }
            _ => None,
        };
        if !changed
            && runtime_selection
                .as_ref()
                .is_none_or(|selection| selection.skipped_row_count() == 0)
        {
            self.prepared_next = Some(next);
            return Ok(());
        }
        let mut selections = self.selections.clone();
        if let Some(runtime) = runtime_selection {
            let combined = match selections[0].selection() {
                Some(base) => base.intersection(&runtime),
                None => runtime,
            };
            selections[0] = RowGroupSelection::new(next, Some(combined));
        }
        let row_filter = if changed {
            Some(self.compile_row_filter()?)
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
                DecodeResult::Data(reader) => {
                    // The next boundary drops this group from the unstarted plan.
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

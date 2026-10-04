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

//! Boundary-aware decoding. This first stage only accepts scans without a
//! flattened RowSelection. Selection-bearing tasks keep task-open pruning.

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
        match self
            .runtime
            .refresh_if_changed(self.task.schema_ref(), self.task.case_sensitive())
        {
            Ok(false) => return Ok(()),
            Ok(true) => self.metrics.record_runtime_predicate_refresh(),
            Err(error) => {
                tracing::debug!(
                    "Skipping live runtime publication for {}: {error}",
                    self.task.data_file_path()
                );
                return Ok(());
            }
        }
        let Some(predicate) = self.runtime.predicate() else {
            return Ok(());
        };
        if !self.accepted_runtime {
            self.metrics.record_runtime_predicate_task();
            self.accepted_runtime = true;
        }
        // The base predicate and equality-delete state were built once at task
        // open. They remain installed in the decoder; only runtime statistics
        // are reevaluated over the not-yet-consumed row groups.
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
                return Ok(());
            }
        };
        let pruned = self.remaining.len() - keep.len();
        self.metrics
            .record_runtime_row_groups_considered(self.remaining.len());
        if pruned != 0 {
            let decoder = self.decoder.take().expect("decoder exists while streaming");
            self.decoder = Some(
                decoder
                    .into_builder()?
                    .with_row_groups(keep.clone())
                    .build()?,
            );
            self.remaining = keep.into();
            self.metrics.record_runtime_row_groups_pruned_live(pruned);
        }
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

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

//! Decoding with row-group-local selections. When a runtime predicate provider
//! is configured, a task is decoded with parquet's push decoder, so each row
//! group keeps its own page and positional-delete selection. Removing a group
//! then never renumbers the rows of another.

use arrow_array::RecordBatch;
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, RowSelection};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::{ParquetPushDecoder, RowGroupSelection};
use parquet::file::metadata::ParquetMetaData;

use super::ArrowFileReader;
use crate::scan::ArrowRecordBatchStream;
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

pub(super) struct PushDecodedStream {
    decoder: ParquetPushDecoder,
    active_reader: Option<ParquetRecordBatchReader>,
    file_reader: ArrowFileReader,
}

impl PushDecodedStream {
    pub(super) fn new(decoder: ParquetPushDecoder, file_reader: ArrowFileReader) -> Self {
        Self {
            decoder,
            active_reader: None,
            file_reader,
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
            match self.decoder.try_next_reader()? {
                DecodeResult::NeedsData(ranges) => {
                    let bytes = self.file_reader.get_byte_ranges(ranges.clone()).await?;
                    self.decoder.push_ranges(ranges, bytes)?;
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

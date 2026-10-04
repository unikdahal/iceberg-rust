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

//! Scan metrics and I/O counting for Parquet data file reads.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;

use crate::error::Result;
use crate::io::FileRead;
use crate::scan::ArrowRecordBatchStream;

/// Wraps a [`FileRead`] to count bytes read via a shared atomic counter.
pub(crate) struct CountingFileRead<F: FileRead> {
    inner: F,
    bytes_read: Arc<AtomicU64>,
}

impl<F: FileRead> CountingFileRead<F> {
    pub(crate) fn new(inner: F, bytes_read: Arc<AtomicU64>) -> Self {
        Self { inner, bytes_read }
    }
}

#[async_trait::async_trait]
impl<F: FileRead> FileRead for CountingFileRead<F> {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        debug_assert!(range.end >= range.start);
        self.bytes_read
            .fetch_add(range.end - range.start, Ordering::Relaxed);
        self.inner.read(range).await
    }
}

/// Metrics collected during an Iceberg scan.
#[derive(Clone, Debug)]
pub struct ScanMetrics {
    bytes_read: Arc<AtomicU64>,
    runtime_predicate_tasks: Arc<AtomicU64>,
    runtime_file_tasks_pruned: Arc<AtomicU64>,
    runtime_row_groups_pruned: Arc<AtomicU64>,
}

impl ScanMetrics {
    pub(crate) fn new() -> Self {
        Self {
            bytes_read: Arc::new(AtomicU64::new(0)),
            runtime_predicate_tasks: Arc::new(AtomicU64::new(0)),
            runtime_file_tasks_pruned: Arc::new(AtomicU64::new(0)),
            runtime_row_groups_pruned: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn record_runtime_predicate_task(&self) {
        self.runtime_predicate_tasks.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_runtime_file_task_pruned(&self) {
        self.runtime_file_tasks_pruned
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_runtime_row_groups_pruned(&self, count: usize) {
        self.runtime_row_groups_pruned
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    pub(crate) fn bytes_read_counter(&self) -> &Arc<AtomicU64> {
        &self.bytes_read
    }

    /// Total bytes read from storage during this scan, including data files and delete files.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// Data-file tasks that applied a runtime predicate, including tasks it
    /// rejected before opening their file.
    pub fn runtime_predicate_tasks(&self) -> u64 {
        self.runtime_predicate_tasks.load(Ordering::Relaxed)
    }

    /// Data-file tasks rejected from whole-file statistics before their data
    /// or delete files were read. Byte-range splits count as separate tasks.
    pub fn runtime_file_tasks_pruned(&self) -> u64 {
        self.runtime_file_tasks_pruned.load(Ordering::Relaxed)
    }

    /// Row groups skipped by statistics because of a runtime predicate, beyond
    /// those excluded by the task byte range and the planned and
    /// equality-delete predicates.
    pub fn runtime_row_groups_pruned(&self) -> u64 {
        self.runtime_row_groups_pruned.load(Ordering::Relaxed)
    }
}

/// Result of [`ArrowReader::read`](super::ArrowReader::read), containing the
/// record batch stream and metrics collected during the scan.
pub struct ScanResult {
    stream: ArrowRecordBatchStream,
    metrics: ScanMetrics,
}

impl ScanResult {
    pub(crate) fn new(stream: ArrowRecordBatchStream, metrics: ScanMetrics) -> Self {
        Self { stream, metrics }
    }

    /// Consumes the result, returning only the record batch stream.
    pub fn stream(self) -> ArrowRecordBatchStream {
        self.stream
    }

    /// Returns a reference to the scan metrics.
    pub fn metrics(&self) -> &ScanMetrics {
        &self.metrics
    }
}

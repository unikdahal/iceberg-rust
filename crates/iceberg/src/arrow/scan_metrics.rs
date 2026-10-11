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

/// Why the reader skipped a runtime predicate, or part of it.
#[derive(Clone, Copy)]
pub(crate) enum RuntimeFallback {
    /// The provider failed to return a snapshot.
    Snapshot,
    /// The predicate did not bind to the scanned schema.
    Bind,
    /// The snapshot was older than a generation the reader already saw.
    Stale,
    /// The predicate needs a type or column the reader cannot prune on.
    Unsupported,
    /// Row-group, page-index or row-filter planning failed.
    Planning,
    /// Evaluating the predicate on decoded rows failed.
    RowEvaluation,
}

/// Metrics collected during an Iceberg scan.
#[derive(Clone, Debug)]
pub struct ScanMetrics {
    bytes_read: Arc<AtomicU64>,
    runtime_predicate_tasks: Arc<AtomicU64>,
    runtime_file_tasks_pruned: Arc<AtomicU64>,
    runtime_row_groups_pruned: Arc<AtomicU64>,
    runtime_predicate_refreshes: Arc<AtomicU64>,
    runtime_row_groups_pruned_live: Arc<AtomicU64>,
    runtime_decoder_rebuilds: Arc<AtomicU64>,
    runtime_snapshot_fallbacks: Arc<AtomicU64>,
    runtime_bind_fallbacks: Arc<AtomicU64>,
    runtime_stale_fallbacks: Arc<AtomicU64>,
    runtime_unsupported_fallbacks: Arc<AtomicU64>,
    runtime_planning_fallbacks: Arc<AtomicU64>,
    runtime_row_evaluation_fallbacks: Arc<AtomicU64>,
}

impl ScanMetrics {
    pub(crate) fn new() -> Self {
        Self {
            bytes_read: Arc::new(AtomicU64::new(0)),
            runtime_predicate_tasks: Arc::new(AtomicU64::new(0)),
            runtime_file_tasks_pruned: Arc::new(AtomicU64::new(0)),
            runtime_row_groups_pruned: Arc::new(AtomicU64::new(0)),
            runtime_predicate_refreshes: Arc::new(AtomicU64::new(0)),
            runtime_row_groups_pruned_live: Arc::new(AtomicU64::new(0)),
            runtime_decoder_rebuilds: Arc::new(AtomicU64::new(0)),
            runtime_snapshot_fallbacks: Arc::new(AtomicU64::new(0)),
            runtime_bind_fallbacks: Arc::new(AtomicU64::new(0)),
            runtime_stale_fallbacks: Arc::new(AtomicU64::new(0)),
            runtime_unsupported_fallbacks: Arc::new(AtomicU64::new(0)),
            runtime_planning_fallbacks: Arc::new(AtomicU64::new(0)),
            runtime_row_evaluation_fallbacks: Arc::new(AtomicU64::new(0)),
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

    pub(crate) fn record_runtime_refresh(&self, row_groups_pruned: usize) {
        self.runtime_predicate_refreshes
            .fetch_add(1, Ordering::Relaxed);
        self.runtime_row_groups_pruned_live
            .fetch_add(row_groups_pruned as u64, Ordering::Relaxed);
        self.runtime_row_groups_pruned
            .fetch_add(row_groups_pruned as u64, Ordering::Relaxed);
        self.runtime_decoder_rebuilds
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_runtime_fallback(&self, reason: RuntimeFallback) {
        let counter = match reason {
            RuntimeFallback::Snapshot => &self.runtime_snapshot_fallbacks,
            RuntimeFallback::Bind => &self.runtime_bind_fallbacks,
            RuntimeFallback::Stale => &self.runtime_stale_fallbacks,
            RuntimeFallback::Unsupported => &self.runtime_unsupported_fallbacks,
            RuntimeFallback::Planning => &self.runtime_planning_fallbacks,
            RuntimeFallback::RowEvaluation => &self.runtime_row_evaluation_fallbacks,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn bytes_read_counter(&self) -> &Arc<AtomicU64> {
        &self.bytes_read
    }

    /// Total bytes requested from storage during this scan, including data files and delete
    /// files. Each read is counted when it is issued, so failed reads are included.
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
    /// equality-delete predicates, whether at task start or at a later
    /// row-group boundary.
    pub fn runtime_row_groups_pruned(&self) -> u64 {
        self.runtime_row_groups_pruned.load(Ordering::Relaxed)
    }

    /// Times a task adopted a newer runtime predicate at a row-group boundary.
    pub fn runtime_predicate_refreshes(&self) -> u64 {
        self.runtime_predicate_refreshes.load(Ordering::Relaxed)
    }

    /// The part of [`Self::runtime_row_groups_pruned`] removed by refreshes
    /// while a task was being read, before those row groups were started.
    pub fn runtime_row_groups_pruned_live(&self) -> u64 {
        self.runtime_row_groups_pruned_live.load(Ordering::Relaxed)
    }

    /// Decoder rebuilds. Each refresh costs exactly one, however many row
    /// groups follow it.
    pub fn runtime_decoder_rebuilds(&self) -> u64 {
        self.runtime_decoder_rebuilds.load(Ordering::Relaxed)
    }

    /// Runtime predicate snapshots the provider failed to return. Each failed
    /// snapshot counts once, however many tasks skip its predicate.
    pub fn runtime_snapshot_fallbacks(&self) -> u64 {
        self.runtime_snapshot_fallbacks.load(Ordering::Relaxed)
    }

    /// Runtime predicates that failed to bind to the scanned schema. Each
    /// failed binding counts once, however many tasks skip it.
    pub fn runtime_bind_fallbacks(&self) -> u64 {
        self.runtime_bind_fallbacks.load(Ordering::Relaxed)
    }

    /// Runtime predicate snapshots ignored because they were older than a
    /// generation the reader had already seen.
    pub fn runtime_stale_fallbacks(&self) -> u64 {
        self.runtime_stale_fallbacks.load(Ordering::Relaxed)
    }

    /// Runtime predicates skipped as unsupported: once per binding for a
    /// floating-point comparison or set, and once per task or refresh for a
    /// data file that lacks a referenced column or stores it with an
    /// unsupported type.
    pub fn runtime_unsupported_fallbacks(&self) -> u64 {
        self.runtime_unsupported_fallbacks.load(Ordering::Relaxed)
    }

    /// Runtime planning failures. A row-group statistics or row filter
    /// failure skips the runtime predicate for that task or refresh; a
    /// page-index failure skips only its page pruning.
    pub fn runtime_planning_fallbacks(&self) -> u64 {
        self.runtime_planning_fallbacks.load(Ordering::Relaxed)
    }

    /// Tasks that stopped evaluating the runtime predicate on rows after an
    /// evaluation error. Counted when the task's batches are fully read or
    /// the stream is dropped.
    pub fn runtime_row_evaluation_fallbacks(&self) -> u64 {
        self.runtime_row_evaluation_fallbacks
            .load(Ordering::Relaxed)
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

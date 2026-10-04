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

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::runtime_predicate::RuntimePredicateState;
use super::{ArrowReaderBuilder, RuntimePredicateProvider, RuntimePredicateSnapshot};
use crate::arrow::ScanMetrics;
use crate::expr::{Bind, Predicate, Reference};
use crate::io::FileIO;
use crate::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
use crate::spec::{
    DataContentType, DataFileFormat, Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type,
};
use crate::{Result, Runtime};

#[derive(Debug)]
struct FixedRuntimePredicate {
    predicate: Predicate,
    generation: u64,
    snapshots: AtomicU64,
}

impl FixedRuntimePredicate {
    fn new(predicate: Predicate) -> Self {
        Self {
            predicate,
            generation: 1,
            snapshots: AtomicU64::new(0),
        }
    }

    fn snapshots(&self) -> u64 {
        self.snapshots.load(Ordering::Relaxed)
    }
}

impl RuntimePredicateProvider for FixedRuntimePredicate {
    fn generation(&self) -> u64 {
        self.generation
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
        self.snapshots.fetch_add(1, Ordering::Relaxed);
        Ok(RuntimePredicateSnapshot::new(
            Some(self.predicate.clone()),
            self.generation,
        ))
    }
}

fn iceberg_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap(),
    )
}

struct ChangingRuntimePredicate {
    generation: AtomicU64,
    snapshot: Mutex<RuntimePredicateSnapshot>,
    snapshots: AtomicU64,
    fail: AtomicBool,
}

impl ChangingRuntimePredicate {
    fn new(predicate: Option<Predicate>, generation: u64) -> Self {
        Self {
            generation: AtomicU64::new(generation),
            snapshot: Mutex::new(RuntimePredicateSnapshot::new(predicate, generation)),
            snapshots: AtomicU64::new(0),
            fail: AtomicBool::new(false),
        }
    }

    fn publish(&self, predicate: Option<Predicate>, generation: u64) {
        *self.snapshot.lock().unwrap() = RuntimePredicateSnapshot::new(predicate, generation);
        self.generation.store(generation, Ordering::Release);
    }
}

impl RuntimePredicateProvider for ChangingRuntimePredicate {
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
        self.snapshots.fetch_add(1, Ordering::Relaxed);
        if self.fail.load(Ordering::Acquire) {
            return Err(crate::Error::new(
                crate::ErrorKind::Unexpected,
                "publication failed",
            ));
        }
        Ok(self.snapshot.lock().unwrap().clone())
    }
}

#[test]
fn runtime_predicate_generation_zero_and_unchanged_do_not_resnapshot() {
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let mut state = RuntimePredicateState::new(provider.clone());
    assert!(state.refresh_if_changed(iceberg_schema(), false).unwrap());
    for _ in 0..100 {
        assert!(!state.refresh_if_changed(iceberg_schema(), false).unwrap());
    }
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 1);
    assert!(state.predicate().is_none());
    let predicate = Reference::new("id").greater_than_or_equal_to(Datum::int(100));
    provider.publish(Some(predicate.clone()), 1);
    assert!(state.refresh_if_changed(iceberg_schema(), false).unwrap());
    assert_eq!(
        state.predicate(),
        Some(&predicate.bind(iceberg_schema(), false).unwrap())
    );
    provider.publish(None, 2);
    assert!(state.refresh_if_changed(iceberg_schema(), false).unwrap());
    assert!(state.predicate().is_none());
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 3);
}

#[test]
fn runtime_predicate_caches_the_snapshot_generation_after_a_race() {
    let predicate = Reference::new("id").less_than_or_equal_to(Datum::int(100));
    let provider = Arc::new(ChangingRuntimePredicate::new(Some(predicate), 3));
    // Publication advances between the cheap read and coherent snapshot.
    provider.generation.store(2, Ordering::Release);
    let mut state = RuntimePredicateState::new(provider.clone());
    assert!(state.refresh_if_changed(iceberg_schema(), false).unwrap());
    provider.generation.store(3, Ordering::Release);
    assert!(!state.refresh_if_changed(iceberg_schema(), false).unwrap());
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 1);
}

#[test]
fn runtime_predicate_invalid_binding_is_cached_and_can_recover() {
    let provider = Arc::new(ChangingRuntimePredicate::new(
        Some(Reference::new("ID").equal_to(Datum::int(100))),
        1,
    ));
    let mut state = RuntimePredicateState::new(provider.clone());
    assert!(state.refresh_if_changed(iceberg_schema(), true).is_err());
    assert!(state.predicate().is_none());
    assert!(!state.refresh_if_changed(iceberg_schema(), true).unwrap());
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 1);
    provider.publish(Some(Reference::new("id").equal_to(Datum::int(100))), 2);
    assert!(state.refresh_if_changed(iceberg_schema(), true).unwrap());
    assert!(state.predicate().is_some());
}

#[test]
fn runtime_predicate_snapshot_failure_clears_old_restriction_and_recovers() {
    let predicate = Reference::new("id").equal_to(Datum::int(100));
    let provider = Arc::new(ChangingRuntimePredicate::new(Some(predicate.clone()), 1));
    let mut state = RuntimePredicateState::new(provider.clone());
    assert!(state.refresh_if_changed(iceberg_schema(), false).unwrap());
    provider.fail.store(true, Ordering::Release);
    provider.generation.store(2, Ordering::Release);
    assert!(state.refresh_if_changed(iceberg_schema(), false).is_err());
    assert!(state.predicate().is_none());
    assert!(!state.refresh_if_changed(iceberg_schema(), false).unwrap());
    provider.fail.store(false, Ordering::Release);
    provider.publish(Some(predicate), 3);
    assert!(state.refresh_if_changed(iceberg_schema(), false).unwrap());
    assert!(state.predicate().is_some());
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 3);
}

fn write_three_row_group_file(dir: &str, name: &str) -> String {
    write_row_group_file(dir, name, &[0, 100, 200])
}

fn write_row_group_file(dir: &str, name: &str, bases: &[i32]) -> String {
    let id_field = Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        "1".to_string(),
    )]));
    let payload_field = Field::new("payload", DataType::Utf8, false).with_metadata(HashMap::from(
        [(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())],
    ));
    let arrow_schema = Arc::new(ArrowSchema::new(vec![id_field, payload_field]));

    let file_path = format!("{dir}/{name}");
    let file = File::create(&file_path).unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_max_row_group_row_count(Some(4))
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&arrow_schema), Some(props)).unwrap();

    for &base in bases {
        let ids: Vec<i32> = (base..base + 4).collect();
        let payloads: Vec<String> = (0..4)
            .map(|row| {
                (0..16_384)
                    .map(|offset| {
                        let value = ((base as usize) + row + offset * 17) % 94;
                        char::from_u32(33 + value as u32).unwrap()
                    })
                    .collect()
            })
            .collect();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(payloads)),
        ];
        let batch = RecordBatch::try_new(Arc::clone(&arrow_schema), columns).unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();
    file_path
}

fn start_runtime_scan(
    task: FileScanTask,
    provider: Option<Arc<dyn RuntimePredicateProvider>>,
    row_selection: bool,
    row_groups: bool,
    batch_size: usize,
) -> (crate::scan::ArrowRecordBatchStream, ScanMetrics) {
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_data_file_concurrency_limit(1)
        .with_row_selection_enabled(row_selection)
        .with_row_group_filtering_enabled(row_groups)
        .with_metadata_size_hint(1024)
        .with_batch_size(batch_size);
    if let Some(provider) = provider {
        builder = builder.with_runtime_predicate_provider(provider);
    }
    let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
    let scan = builder.build().read(tasks).unwrap();
    let metrics = scan.metrics().clone();
    (scan.stream(), metrics)
}

#[tokio::test]
async fn runtime_predicate_live_arrival_waits_for_row_group_boundary() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "live.parquet");
    let task = scan_task(path, iceberg_schema(), None);
    let (baseline, baseline_metrics) = start_runtime_scan(task.clone(), None, false, true, 2);
    let baseline: Vec<RecordBatch> = baseline.try_collect().await.unwrap();
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(task, Some(provider.clone()), false, true, 2);
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    assert_eq!(ids(&batches), vec![0, 1]);
    provider.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
        1,
    );
    // RG0 is in flight. Tightening cannot discard its remaining batch.
    batches.push(stream.try_next().await.unwrap().unwrap());
    assert_eq!(ids(&batches), vec![0, 1, 2, 3]);
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 1);
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(ids(&batches), vec![0, 1, 2, 3, 200, 201, 202, 203]);
    assert_eq!(
        ids(&batches).into_iter().max(),
        ids(&baseline).into_iter().max()
    );
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 2);
    assert_eq!(metrics.runtime_predicate_tasks(), 1);
    assert_eq!(metrics.runtime_live_pruning_tasks(), 1);
    assert_eq!(metrics.runtime_predicate_refreshes(), 1);
    assert_eq!(metrics.runtime_row_groups_pruned_live(), 1);
    assert_eq!(metrics.runtime_row_groups_pruned(), 1);
    assert_eq!(metrics.runtime_row_groups_pruned_initial(), 0);
    assert_eq!(metrics.runtime_row_groups_considered(), 2);
    assert!(metrics.bytes_read() < baseline_metrics.bytes_read());
}

#[tokio::test]
async fn runtime_predicate_live_tightening_only_prunes_remaining_groups() {
    let temp = TempDir::new().unwrap();
    let path = write_row_group_file(temp.path().to_str().unwrap(), "tightening.parquet", &[
        200, 0, 300, 100, 400, 350,
    ]);
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(
        scan_task(path, iceberg_schema(), None),
        Some(provider.clone()),
        false,
        true,
        4,
    );
    let mut output = Vec::new();
    for (generation, expected) in [
        (1, vec![200, 201, 202, 203]),
        (2, vec![300, 301, 302, 303]),
        (3, vec![400, 401, 402, 403]),
    ] {
        let batch = stream.try_next().await.unwrap().unwrap();
        assert_eq!(ids(std::slice::from_ref(&batch)), expected);
        let max = *expected.last().unwrap();
        output.push(batch);
        provider.publish(
            Some(Reference::new("id").greater_than(Datum::int(max))),
            generation,
        );
    }
    assert!(stream.try_next().await.unwrap().is_none());
    assert_eq!(ids(&output).into_iter().max(), Some(403));
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 4);
    assert_eq!(metrics.runtime_predicate_refreshes(), 3);
    assert_eq!(metrics.runtime_row_groups_pruned_live(), 3);
    assert_eq!(metrics.runtime_row_groups_considered(), 8);
}

#[tokio::test]
async fn runtime_predicate_concurrent_files_cache_generations_independently() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().to_str().unwrap();
    let task_a = scan_task(
        write_three_row_group_file(dir, "a.parquet"),
        iceberg_schema(),
        None,
    );
    let task_b = scan_task(
        write_three_row_group_file(dir, "b.parquet"),
        iceberg_schema(),
        None,
    );
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut a, metrics_a) = start_runtime_scan(task_a, Some(provider.clone()), false, true, 4);
    let (mut b, metrics_b) = start_runtime_scan(task_b, Some(provider.clone()), false, true, 4);
    let (first_a, first_b) = tokio::join!(a.try_next(), b.try_next());
    assert_eq!(ids(&[first_a.unwrap().unwrap()]), vec![0, 1, 2, 3]);
    assert_eq!(ids(&[first_b.unwrap().unwrap()]), vec![0, 1, 2, 3]);
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 2);
    provider.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
        1,
    );
    let (rest_a, rest_b) = tokio::join!(a.try_collect::<Vec<_>>(), b.try_collect::<Vec<_>>());
    assert_eq!(ids(&rest_a.unwrap()), vec![200, 201, 202, 203]);
    assert_eq!(ids(&rest_b.unwrap()), vec![200, 201, 202, 203]);
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 4);
    for metrics in [metrics_a, metrics_b] {
        assert_eq!(metrics.runtime_predicate_refreshes(), 1);
        assert_eq!(metrics.runtime_row_groups_pruned_live(), 1);
        assert_eq!(metrics.runtime_row_groups_considered(), 2);
    }
}

#[tokio::test]
async fn runtime_predicate_multifile_reader_samples_each_task_once() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().to_str().unwrap();
    let tasks = Box::pin(futures::stream::iter(vec![
        Ok(scan_task(
            write_three_row_group_file(dir, "parallel-a.parquet"),
            iceberg_schema(),
            None,
        )),
        Ok(scan_task(
            write_three_row_group_file(dir, "parallel-b.parquet"),
            iceberg_schema(),
            None,
        )),
    ])) as FileScanTaskStream;
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let scan = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_data_file_concurrency_limit(2)
        .with_batch_size(2)
        .with_runtime_predicate_provider(provider.clone())
        .build()
        .read(tasks)
        .unwrap();
    let metrics = scan.metrics().clone();
    let batches: Vec<RecordBatch> = scan.stream().try_collect().await.unwrap();
    let mut values = ids(&batches);
    values.sort_unstable();
    let mut expected: Vec<i32> = [0, 100, 200]
        .into_iter()
        .flat_map(|base| (base..base + 4).flat_map(|id| [id, id]))
        .collect();
    expected.sort_unstable();
    assert_eq!(values, expected);
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 2);
    assert_eq!(metrics.runtime_live_pruning_tasks(), 2);
    assert_eq!(metrics.runtime_predicate_refreshes(), 0);
}

#[tokio::test]
async fn runtime_predicate_live_unchanged_generation_does_not_resnapshot() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "unchanged.parquet");
    let task = scan_task(path, iceberg_schema(), None);
    let (baseline, baseline_metrics) = start_runtime_scan(task.clone(), None, false, true, 1);
    let baseline: Vec<RecordBatch> = baseline.try_collect().await.unwrap();
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (stream, metrics) = start_runtime_scan(task, Some(provider.clone()), false, true, 1);
    let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
    assert_eq!(ids(&batches).len(), 12);
    assert_eq!(ids(&batches), ids(&baseline));
    assert_eq!(metrics.bytes_read(), baseline_metrics.bytes_read());
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.runtime_predicate_refreshes(), 0);
    assert_eq!(metrics.runtime_live_pruning_tasks(), 1);
}

#[tokio::test]
async fn runtime_predicate_live_refresh_keeps_planned_row_filter() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "planned.parquet");
    let schema = iceberg_schema();
    let planned = Reference::new("id")
        .less_than_or_equal_to(Datum::int(201))
        .bind(Arc::clone(&schema), false)
        .unwrap();
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(
        scan_task(path, schema, Some(planned)),
        Some(provider.clone()),
        false,
        true,
        4,
    );
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    provider.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
        1,
    );
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(ids(&batches), vec![0, 1, 2, 3, 200, 201]);
    assert_eq!(metrics.runtime_row_groups_pruned_live(), 1);
}

#[tokio::test]
async fn runtime_predicate_live_bad_publication_fails_open_and_next_generation_recovers() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "recover.parquet");
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(
        scan_task(path, iceberg_schema(), None),
        Some(provider.clone()),
        false,
        true,
        4,
    );
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    provider.publish(Some(Reference::new("missing").equal_to(Datum::int(100))), 1);
    batches.push(stream.try_next().await.unwrap().unwrap());
    assert_eq!(ids(&batches), vec![0, 1, 2, 3, 100, 101, 102, 103]);
    provider.publish(Some(Predicate::AlwaysFalse), 2);
    assert!(stream.try_next().await.unwrap().is_none());
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 3);
    assert_eq!(metrics.runtime_predicate_refreshes(), 1);
    assert_eq!(metrics.runtime_row_groups_pruned_live(), 1);
}

#[tokio::test]
async fn runtime_predicate_live_refresh_preserves_cached_equality_deletes() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().to_str().unwrap();
    let path = write_three_row_group_file(dir, "equalities-data.parquet");
    let equality_path = format!("{dir}/live-equality.parquet");
    write_delete(&equality_path, vec![field("id", DataType::Int32, 1)], vec![
        Arc::new(Int32Array::from(vec![1, 202])),
    ]);
    let delete = FileScanTaskDeleteFile::builder()
        .with_file_size_in_bytes(std::fs::metadata(&equality_path).unwrap().len())
        .with_file_path(equality_path)
        .with_file_type(DataContentType::EqualityDeletes)
        .with_file_format(DataFileFormat::Parquet)
        .with_partition_spec_id(0)
        .with_equality_ids(Some(vec![1]))
        .build();
    let task = scan_task_with_deletes(path, iceberg_schema(), None, vec![delete]);
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(task, Some(provider.clone()), false, true, 4);
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    assert_eq!(ids(&batches), vec![0, 2, 3]);
    provider.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
        1,
    );
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(ids(&batches), vec![0, 2, 3, 200, 201, 203]);
    assert_eq!(metrics.runtime_row_groups_pruned_live(), 1);
}

#[tokio::test]
async fn runtime_predicate_live_disabled_for_flattened_page_selection() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "pages.parquet");
    let provider = Arc::new(ChangingRuntimePredicate::new(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(0))),
        0,
    ));
    let (mut stream, metrics) = start_runtime_scan(
        scan_task(path, iceberg_schema(), None),
        Some(provider.clone()),
        true,
        true,
        4,
    );
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    provider.publish(Some(Predicate::AlwaysFalse), 1);
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(ids(&batches).len(), 12);
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.runtime_live_pruning_tasks(), 0);
}

#[tokio::test]
async fn runtime_predicate_live_disabled_for_position_delete_selection() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().to_str().unwrap();
    let path = write_three_row_group_file(dir, "positions-data.parquet");
    let positions = format!("{dir}/live-positions.parquet");
    write_delete(
        &positions,
        vec![
            field("file_path", DataType::Utf8, 2_147_483_546),
            field("pos", DataType::Int64, 2_147_483_545),
        ],
        vec![
            Arc::new(StringArray::from(vec![path.as_str()])),
            Arc::new(Int64Array::from(vec![5])),
        ],
    );
    let delete = FileScanTaskDeleteFile::builder()
        .with_file_size_in_bytes(std::fs::metadata(&positions).unwrap().len())
        .with_file_path(positions)
        .with_file_type(DataContentType::PositionDeletes)
        .with_file_format(DataFileFormat::Parquet)
        .with_partition_spec_id(0)
        .build();
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let task = scan_task_with_deletes(path, iceberg_schema(), None, vec![delete]);
    let (mut stream, metrics) = start_runtime_scan(task, Some(provider.clone()), false, true, 4);
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    provider.publish(Some(Predicate::AlwaysFalse), 1);
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(ids(&batches), vec![
        0, 1, 2, 3, 100, 102, 103, 200, 201, 202, 203
    ]);
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.runtime_live_pruning_tasks(), 0);
}

#[tokio::test]
async fn runtime_predicate_live_disabled_when_row_group_filtering_is_disabled() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "disabled.parquet");
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(
        scan_task(path, iceberg_schema(), None),
        Some(provider.clone()),
        false,
        false,
        4,
    );
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    provider.publish(Some(Predicate::AlwaysFalse), 1);
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(ids(&batches).len(), 12);
    assert_eq!(provider.snapshots.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.runtime_live_pruning_tasks(), 0);
}

fn scan_task(
    file_path: String,
    schema: SchemaRef,
    predicate: Option<crate::expr::BoundPredicate>,
) -> FileScanTask {
    scan_task_with_deletes(file_path, schema, predicate, vec![])
}

fn scan_task_with_deletes(
    file_path: String,
    schema: SchemaRef,
    predicate: Option<crate::expr::BoundPredicate>,
    deletes: Vec<FileScanTaskDeleteFile>,
) -> FileScanTask {
    FileScanTask::builder()
        .with_file_size_in_bytes(std::fs::metadata(&file_path).unwrap().len())
        .with_start(0)
        .with_length(0)
        .with_data_file_path(file_path)
        .with_data_file_format(DataFileFormat::Parquet)
        .with_schema(schema)
        .with_project_field_ids(vec![1, 2])
        .with_predicate(predicate)
        .with_deletes(deletes)
        .with_case_sensitive(false)
        .build()
        .unwrap()
}

async fn execute(
    task: FileScanTask,
    provider: Option<Arc<dyn RuntimePredicateProvider>>,
) -> (Vec<RecordBatch>, ScanMetrics) {
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_data_file_concurrency_limit(1)
        .with_row_selection_enabled(true);
    if let Some(provider) = provider {
        builder = builder.with_runtime_predicate_provider(provider);
    }
    let reader = builder.build();
    let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
    let scan = reader.read(tasks).unwrap();
    let metrics = scan.metrics().clone();
    let batches = scan.stream().try_collect().await.unwrap();
    (batches, metrics)
}

fn ids(batches: &[RecordBatch]) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn runtime_predicate_prunes_row_groups_and_bytes() {
    let temp = TempDir::new().unwrap();
    let file_path =
        write_three_row_group_file(temp.path().to_str().unwrap(), "runtime_predicate.parquet");
    let schema = iceberg_schema();
    let task = scan_task(file_path, Arc::clone(&schema), None);

    let (_, baseline_metrics) = execute(task.clone(), None).await;

    let predicate = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(100))
        .and(Reference::new("id").less_than_or_equal_to(Datum::int(103)));
    let provider = Arc::new(FixedRuntimePredicate::new(predicate));
    let (batches, runtime_metrics) = execute(task, Some(provider.clone())).await;

    assert_eq!(ids(&batches), vec![100, 101, 102, 103]);
    assert_eq!(provider.snapshots(), 1);
    assert_eq!(runtime_metrics.runtime_predicate_tasks(), 1);
    assert_eq!(runtime_metrics.runtime_row_groups_pruned(), 2);
    assert!(
        runtime_metrics.bytes_read() < baseline_metrics.bytes_read(),
        "runtime pruning should request fewer bytes: runtime={} baseline={}",
        runtime_metrics.bytes_read(),
        baseline_metrics.bytes_read()
    );
}

#[tokio::test]
async fn runtime_predicate_is_anded_with_task_predicate() {
    let temp = TempDir::new().unwrap();
    let file_path =
        write_three_row_group_file(temp.path().to_str().unwrap(), "runtime_and_static.parquet");
    let schema = iceberg_schema();
    let static_predicate = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(100))
        .bind(Arc::clone(&schema), false)
        .unwrap();
    let task = scan_task(file_path, schema, Some(static_predicate));

    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").less_than_or_equal_to(Datum::int(103)),
    ));
    let (batches, metrics) = execute(task, Some(provider)).await;

    assert_eq!(ids(&batches), vec![100, 101, 102, 103]);
    assert_eq!(metrics.runtime_predicate_tasks(), 1);
    assert_eq!(metrics.runtime_row_groups_pruned(), 1);
}

#[tokio::test]
async fn invalid_runtime_predicate_fails_open() {
    let temp = TempDir::new().unwrap();
    let file_path =
        write_three_row_group_file(temp.path().to_str().unwrap(), "runtime_invalid.parquet");
    let schema = iceberg_schema();
    let task = scan_task(file_path, schema, None);
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("missing").equal_to(Datum::int(1)),
    ));

    let (batches, metrics) = execute(task, Some(provider.clone())).await;

    assert_eq!(ids(&batches), vec![
        0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203
    ]);
    assert_eq!(provider.snapshots(), 1);
    assert_eq!(metrics.runtime_predicate_tasks(), 0);
    assert_eq!(metrics.runtime_row_groups_pruned(), 0);
}

fn write_delete(path: &str, fields: Vec<Field>, arrays: Vec<ArrayRef>) {
    let schema = Arc::new(ArrowSchema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn field(name: &str, data_type: DataType, id: i32) -> Field {
    Field::new(name, data_type, false).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

#[tokio::test]
async fn runtime_pruning_preserves_position_and_equality_deletes() {
    const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: i32 = 2_147_483_546;
    const FIELD_ID_POSITIONAL_DELETE_POS: i32 = 2_147_483_545;

    let temp = TempDir::new().unwrap();
    let dir = temp.path().to_str().unwrap();
    let data_path = write_three_row_group_file(dir, "data.parquet");
    let position_path = format!("{dir}/positions.parquet");
    let equality_path = format!("{dir}/equalities.parquet");
    // Position five means id=101 in the second row group. If selections are
    // incorrectly renumbered after pruning the first group, the wrong row is deleted.
    write_delete(
        &position_path,
        vec![
            field(
                "file_path",
                DataType::Utf8,
                FIELD_ID_POSITIONAL_DELETE_FILE_PATH,
            ),
            field("pos", DataType::Int64, FIELD_ID_POSITIONAL_DELETE_POS),
        ],
        vec![
            Arc::new(StringArray::from(vec![data_path.as_str()])),
            Arc::new(Int64Array::from(vec![5])),
        ],
    );
    write_delete(&equality_path, vec![field("id", DataType::Int32, 1)], vec![
        Arc::new(Int32Array::from(vec![102])),
    ]);
    let delete = |path: String, file_type, equality_ids| {
        FileScanTaskDeleteFile::builder()
            .with_file_size_in_bytes(std::fs::metadata(&path).unwrap().len())
            .with_file_path(path)
            .with_file_type(file_type)
            .with_file_format(DataFileFormat::Parquet)
            .with_partition_spec_id(0)
            .with_equality_ids(equality_ids)
            .build()
    };
    let position = delete(position_path, DataContentType::PositionDeletes, None);
    let equality = delete(
        equality_path,
        DataContentType::EqualityDeletes,
        Some(vec![1]),
    );
    for (deletes, expected) in [
        (vec![position.clone()], vec![100, 102, 103]),
        (vec![equality.clone()], vec![100, 101, 103]),
        (vec![position, equality], vec![100, 103]),
    ] {
        let task = scan_task_with_deletes(data_path.clone(), iceberg_schema(), None, deletes);
        let (baseline, baseline_metrics) = execute(task.clone(), None).await;
        let provider = Arc::new(FixedRuntimePredicate::new(
            Reference::new("id")
                .greater_than_or_equal_to(Datum::int(100))
                .and(Reference::new("id").less_than_or_equal_to(Datum::int(103))),
        ));
        let (runtime, metrics) = execute(task, Some(provider.clone())).await;
        assert_eq!(ids(&runtime), expected);
        assert_eq!(
            ids(&baseline)
                .into_iter()
                .filter(|id| (100..=103).contains(id))
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(provider.snapshots(), 1);
        assert_eq!(metrics.runtime_predicate_tasks(), 1);
        assert_eq!(metrics.runtime_row_groups_pruned(), 2);
        assert!(metrics.bytes_read() < baseline_metrics.bytes_read());
    }
}

struct FailedProvider;
impl RuntimePredicateProvider for FailedProvider {
    fn generation(&self) -> u64 {
        0
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
        Err(crate::Error::new(
            crate::ErrorKind::Unexpected,
            "provider failed",
        ))
    }
}

#[tokio::test]
async fn provider_error_keeps_static_filter() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "fallback.parquet");
    let schema = iceberg_schema();
    let predicate = Reference::new("id")
        .equal_to(Datum::int(101))
        .bind(Arc::clone(&schema), false)
        .unwrap();
    let task = scan_task(path, schema, Some(predicate));
    let (batches, metrics) = execute(task, Some(Arc::new(FailedProvider))).await;
    assert_eq!(ids(&batches), vec![101]);
    assert_eq!(metrics.runtime_predicate_tasks(), 0);
    assert_eq!(metrics.runtime_row_groups_pruned(), 0);
}

#[tokio::test]
async fn runtime_pruning_metrics_respect_task_byte_range() {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "split.parquet");
    let parquet = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let start = 4 + parquet.metadata().row_group(0).compressed_size() as u64;
    let file_size = std::fs::metadata(&path).unwrap().len();
    let schema = iceberg_schema();
    let static_predicate = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(100))
        .bind(Arc::clone(&schema), false)
        .unwrap();
    let task = FileScanTask::builder()
        .with_file_size_in_bytes(file_size)
        .with_start(start)
        .with_length(file_size - start)
        .with_data_file_path(path)
        .with_data_file_format(DataFileFormat::Parquet)
        .with_schema(schema)
        .with_project_field_ids(vec![1, 2])
        .with_predicate(Some(static_predicate))
        .with_case_sensitive(false)
        .build()
        .unwrap();
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").greater_than_or_equal_to(Datum::int(200)),
    ));
    let (batches, metrics) = execute(task, Some(provider)).await;
    assert_eq!(ids(&batches), vec![200, 201, 202, 203]);
    // RG0 belongs to another split; only RG1 is attributed to runtime pruning.
    assert_eq!(metrics.runtime_row_groups_pruned(), 1);
}

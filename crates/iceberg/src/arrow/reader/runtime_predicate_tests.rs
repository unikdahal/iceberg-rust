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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{
    Array, ArrayRef, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::{ArrowReaderBuilder, RuntimePredicateProvider, RuntimePredicateSnapshot};
use crate::arrow::ScanMetrics;
use crate::expr::{Bind, Predicate, Reference};
use crate::io::FileIO;
use crate::scan::{
    ArrowRecordBatchStream, FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream,
};
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

fn write_three_row_group_file(dir: &str, name: &str) -> String {
    write_row_group_file(dir, name, &[0, 100, 200])
}

fn write_row_group_file(dir: &str, name: &str, bases: &[i32]) -> String {
    write_row_group_file_with_page_size(dir, name, bases, 1024)
}

fn write_row_group_file_with_page_size(
    dir: &str,
    name: &str,
    bases: &[i32],
    page_rows: usize,
) -> String {
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
        .set_data_page_row_count_limit(page_rows)
        .set_write_batch_size(page_rows)
        .set_dictionary_enabled(false)
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
    scan_task_with_deletes_and_projection(file_path, schema, predicate, deletes, vec![1, 2])
}

fn scan_task_with_deletes_and_projection(
    file_path: String,
    schema: SchemaRef,
    predicate: Option<crate::expr::BoundPredicate>,
    deletes: Vec<FileScanTaskDeleteFile>,
    project_field_ids: Vec<i32>,
) -> FileScanTask {
    FileScanTask::builder()
        .with_file_size_in_bytes(std::fs::metadata(&file_path).unwrap().len())
        .with_start(0)
        .with_length(0)
        .with_data_file_path(file_path)
        .with_data_file_format(DataFileFormat::Parquet)
        .with_schema(schema)
        .with_project_field_ids(project_field_ids)
        .with_predicate(predicate)
        .with_deletes(deletes)
        .with_case_sensitive(false)
        .build()
        .unwrap()
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

/// Writes one four-row group per base with small payloads.
fn write_groups(path: &str, bases: &[i32], field_ids: bool, key: Option<&[u8]>, bloom: bool) {
    let metadata = |id: &str| {
        if field_ids {
            HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())])
        } else {
            HashMap::new()
        }
    };
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(metadata("1")),
        Field::new("payload", DataType::Utf8, false).with_metadata(metadata("2")),
    ]));
    let mut properties = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_max_row_group_row_count(Some(4))
        .set_bloom_filter_enabled(bloom);
    if let Some(key) = key {
        properties = properties.with_file_encryption_properties(
            parquet::encryption::encrypt::FileEncryptionProperties::builder(key.to_vec())
                .build()
                .unwrap(),
        );
    }
    let mut writer = ArrowWriter::try_new(
        File::create(path).unwrap(),
        Arc::clone(&schema),
        Some(properties.build()),
    )
    .unwrap();
    for &base in bases {
        let ids: Vec<i32> = (base..base + 4).collect();
        let payloads: Vec<String> = ids.iter().map(|id| format!("payload-{id}")).collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(payloads)),
        ])
        .unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();
}

#[derive(Default)]
struct FailedProvider {
    snapshots: AtomicU64,
}

impl RuntimePredicateProvider for FailedProvider {
    fn generation(&self) -> u64 {
        0
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
        self.snapshots.fetch_add(1, Ordering::Relaxed);
        Err(crate::Error::new(
            crate::ErrorKind::Unexpected,
            "publication failed",
        ))
    }
}

async fn execute_tasks(
    tasks: Vec<FileScanTask>,
    provider: Option<Arc<dyn RuntimePredicateProvider>>,
    bloom_filter: bool,
) -> (Vec<RecordBatch>, ScanMetrics) {
    execute_tasks_with_concurrency(tasks, provider, bloom_filter, 1).await
}

async fn execute_tasks_with_concurrency(
    tasks: Vec<FileScanTask>,
    provider: Option<Arc<dyn RuntimePredicateProvider>>,
    bloom_filter: bool,
    concurrency: usize,
) -> (Vec<RecordBatch>, ScanMetrics) {
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_data_file_concurrency_limit(concurrency)
        .with_row_selection_enabled(true)
        .with_bloom_filter_enabled(bloom_filter);
    if let Some(provider) = provider {
        builder = builder.with_runtime_predicate_provider(provider);
    }
    let tasks = Box::pin(futures::stream::iter(tasks.into_iter().map(Ok))) as FileScanTaskStream;
    let scan = builder.build().read(tasks).unwrap();
    let metrics = scan.metrics().clone();
    let batches = scan.stream().try_collect().await.unwrap();
    (batches, metrics)
}

async fn execute(
    task: FileScanTask,
    provider: Option<Arc<dyn RuntimePredicateProvider>>,
) -> (Vec<RecordBatch>, ScanMetrics) {
    execute_tasks(vec![task], provider, false).await
}

fn all_ids() -> Vec<i32> {
    [0, 100, 200]
        .into_iter()
        .flat_map(|base| base..base + 4)
        .collect()
}

#[tokio::test]
async fn runtime_predicate_prunes_row_groups_and_bytes() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "prune.parquet");
    let task = scan_task(path, iceberg_schema(), None);
    let (_, baseline) = execute(task.clone(), None).await;

    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id")
            .greater_than_or_equal_to(Datum::int(100))
            .and(Reference::new("id").less_than_or_equal_to(Datum::int(103))),
    ));
    let (batches, metrics) = execute(task, Some(provider.clone())).await;
    assert_eq!(ids(&batches), vec![100, 101, 102, 103]);
    assert_eq!(provider.snapshots(), 1);
    assert!(
        metrics.bytes_read() < baseline.bytes_read(),
        "runtime={} baseline={}",
        metrics.bytes_read(),
        baseline.bytes_read()
    );
}

#[tokio::test]
async fn runtime_predicate_is_anded_with_task_predicate() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "and.parquet");
    let planned = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(100))
        .bind(iceberg_schema(), false)
        .unwrap();
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").less_than_or_equal_to(Datum::int(103)),
    ));
    let (batches, _) = execute(
        scan_task(path, iceberg_schema(), Some(planned)),
        Some(provider),
    )
    .await;
    assert_eq!(ids(&batches), vec![100, 101, 102, 103]);
}

/// Predicate columns on either side of an output column are fetched together.
/// The bytes read over the gap must be reused by the output stage, rather than
/// fetched a second time. Multiple groups also exercise buffer reclamation.
#[tokio::test]
async fn runtime_predicate_reuses_coalesced_output_ranges() {
    let temp = TempDir::new().unwrap();
    let schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(3, "key", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap(),
    );
    let fields = [
        (1, "id", DataType::Int32),
        (2, "payload", DataType::Utf8),
        (3, "key", DataType::Int32),
    ]
    .into_iter()
    .map(|(id, name, ty)| {
        Field::new(name, ty, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    })
    .collect::<Vec<_>>();
    let arrow_schema = Arc::new(ArrowSchema::new(fields));
    let path = temp.path().join("coalesced.parquet");
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(&path).unwrap(),
        arrow_schema.clone(),
        Some(props),
    )
    .unwrap();
    for base in [0, 100, 200] {
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from_iter_values(base..base + 4)),
            Arc::new(StringArray::from(vec!["x".repeat(65_536); 4])),
            Arc::new(Int32Array::from_iter_values(base..base + 4)),
        ])
        .unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();
    let planned = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(0))
        .bind(schema.clone(), false)
        .unwrap();
    let task = scan_task_with_deletes_and_projection(
        path.to_str().unwrap().to_string(),
        schema,
        Some(planned),
        vec![],
        vec![1, 2, 3],
    );
    let (baseline, baseline_metrics) = execute(task.clone(), None).await;
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("key").greater_than_or_equal_to(Datum::int(0)),
    ));
    let (runtime, runtime_metrics) = execute(task, Some(provider)).await;
    assert_eq!(ids(&runtime), ids(&baseline));
    assert_eq!(runtime_metrics.runtime_predicate_tasks(), 1);
    assert!(
        runtime_metrics.bytes_read() <= baseline_metrics.bytes_read(),
        "runtime={} baseline={}",
        runtime_metrics.bytes_read(),
        baseline_metrics.bytes_read(),
    );
}

#[tokio::test]
async fn runtime_predicate_failures_keep_the_planned_filter() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "fail.parquet");
    let planned = || {
        Some(
            Reference::new("id")
                .equal_to(Datum::int(101))
                .bind(iceberg_schema(), false)
                .unwrap(),
        )
    };
    // A failing provider, and a predicate that cannot be bound.
    for provider in [
        Arc::new(FailedProvider::default()) as Arc<dyn RuntimePredicateProvider>,
        Arc::new(FixedRuntimePredicate::new(
            Reference::new("missing").equal_to(Datum::int(1)),
        )),
    ] {
        let (batches, _) = execute(
            scan_task(path.clone(), iceberg_schema(), planned()),
            Some(provider),
        )
        .await;
        assert_eq!(ids(&batches), vec![101]);
        let (batches, _) = execute(scan_task(path.clone(), iceberg_schema(), None), None).await;
        assert_eq!(ids(&batches), all_ids());
    }
}

#[tokio::test]
async fn runtime_not_predicates_keep_matching_rows() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "not.parquet");
    // RG1 holds 100..=103, so NOT(id < 102) must keep 102 and 103.
    let provider = Arc::new(FixedRuntimePredicate::new(
        !Reference::new("id").less_than(Datum::int(102)),
    ));
    let (batches, _) = execute(scan_task(path, iceberg_schema(), None), Some(provider)).await;
    assert_eq!(ids(&batches), vec![102, 103, 200, 201, 202, 203]);
}

#[tokio::test]
async fn runtime_predicate_on_column_missing_from_file_is_ignored() {
    use crate::spec::Literal;

    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "evolved.parquet");
    // `added` was added after this file was written. Its value in this file is
    // the initial default (7) or null, which physical filters never see.
    for default in [Some(Literal::int(7)), None] {
        let mut added = NestedField::optional(3, "added", Type::Primitive(PrimitiveType::Int));
        if let Some(default) = default.clone() {
            added = added.with_initial_default(default);
        }
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_schema_id(2)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "payload", Type::Primitive(PrimitiveType::String))
                        .into(),
                    added.into(),
                ])
                .build()
                .unwrap(),
        );
        for predicate in [
            Reference::new("added").is_null(),
            Reference::new("added").greater_than_or_equal_to(Datum::int(5)),
            Reference::new("added").less_than(Datum::int(5)),
        ] {
            let provider = Arc::new(FixedRuntimePredicate::new(predicate.clone()));
            let (batches, _) = execute(
                scan_task(path.clone(), Arc::clone(&schema), None),
                Some(provider),
            )
            .await;
            assert_eq!(
                ids(&batches),
                all_ids(),
                "{predicate} with default {default:?}"
            );
        }
    }
}

#[tokio::test]
async fn runtime_predicate_on_promoted_column_is_ignored() {
    let temp = TempDir::new().unwrap();
    // The file stores `id` as INT; the table has since promoted it to BIGINT.
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "promoted.parquet");
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_schema_id(2)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap(),
    );
    // Above i32::MAX: a literal cast down to the physical INT type would overflow.
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").less_than(Datum::long(i64::from(i32::MAX) + 1)),
    ));
    let (batches, _) = execute(scan_task(path, schema, None), Some(provider)).await;
    let values: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(
        values,
        all_ids().into_iter().map(i64::from).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn runtime_predicate_preserves_position_and_equality_deletes() {
    const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: i32 = 2_147_483_546;
    const FIELD_ID_POSITIONAL_DELETE_POS: i32 = 2_147_483_545;

    let temp = TempDir::new().unwrap();
    let dir = temp.path().to_str().unwrap();
    let data_path = write_three_row_group_file(dir, "data.parquet");
    let position_path = format!("{dir}/positions.parquet");
    let equality_path = format!("{dir}/equalities.parquet");
    // Position 5 is id=101 in the second row group. If positions were
    // renumbered after the first group is pruned, the wrong row would go.
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
            .unwrap()
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
        let (runtime, metrics) = execute(task.clone(), Some(provider.clone())).await;
        assert_eq!(ids(&runtime), expected);
        assert_eq!(
            ids(&baseline)
                .into_iter()
                .filter(|id| (100..=103).contains(id))
                .collect::<Vec<_>>(),
            expected
        );
        assert!(metrics.bytes_read() < baseline_metrics.bytes_read());

        // COUNT(*) needs no output columns, but still applies both delete forms
        // and the predicate before counting. Exercise both ordinary and live
        // decoders with the same row groups and original row positions.
        let mut count_task = serde_json::to_value(task).unwrap();
        count_task["project_field_ids"] = serde_json::json!([]);
        let count_task: FileScanTask = serde_json::from_value(count_task).unwrap();
        let (baseline_count, _) = execute(count_task.clone(), None).await;
        assert_eq!(
            baseline_count
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            ids(&baseline).len()
        );
        let (runtime_count, count_metrics) = execute(count_task, Some(provider)).await;
        assert_eq!(
            runtime_count
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            expected.len()
        );
        assert!(
            baseline_count
                .iter()
                .chain(&runtime_count)
                .all(|batch| batch.num_columns() == 0)
        );
        assert!(count_metrics.bytes_read() < metrics.bytes_read());
    }
}

#[tokio::test]
async fn runtime_predicate_reader_preserves_promoted_columns_with_out_of_range_equality_deletes() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().to_str().unwrap();
    let data_path = write_three_row_group_file(dir, "old-int-data.parquet");
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap(),
    );
    let out_of_range = i64::from(i32::MAX) + 1;
    for (index, keys) in [
        vec![out_of_range],
        vec![i64::from(i32::MIN) - 1],
        (out_of_range..out_of_range + 20).collect(),
        std::iter::once(0)
            .chain([100, 200])
            .chain(out_of_range..out_of_range + 20)
            .collect(),
    ]
    .into_iter()
    .enumerate()
    {
        let equality_path = format!("{dir}/long-deletes-{index}.parquet");
        write_delete(&equality_path, vec![field("id", DataType::Int64, 1)], vec![
            Arc::new(Int64Array::from(keys.clone())),
        ]);
        let delete = FileScanTaskDeleteFile::builder()
            .with_file_size_in_bytes(std::fs::metadata(&equality_path).unwrap().len())
            .with_file_path(equality_path)
            .with_file_type(DataContentType::EqualityDeletes)
            .with_file_format(DataFileFormat::Parquet)
            .with_partition_spec_id(0)
            .with_equality_ids(Some(vec![1]))
            .build()
            .unwrap();
        let task = scan_task_with_deletes(data_path.clone(), schema.clone(), None, vec![delete]);
        let expected: Vec<_> = all_ids()
            .into_iter()
            .map(i64::from)
            .filter(|value| !keys.contains(value))
            .collect();
        for provider in [
            None,
            Some(Arc::new(ChangingRuntimePredicate::new(None, 0))
                as Arc<dyn RuntimePredicateProvider>),
        ] {
            let (batches, _) = execute(task.clone(), provider).await;
            let actual: Vec<_> = batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .values()
                        .iter()
                        .copied()
                })
                .collect();
            assert_eq!(actual, expected, "delete keys {keys:?}");
        }
    }
}

#[tokio::test]
async fn runtime_predicate_reader_compares_after_lossless_physical_numeric_promotion() {
    use arrow_array::types::Int8Type;
    use arrow_array::{
        Decimal32Array, Decimal64Array, Decimal256Array, DictionaryArray, Int8Array, Int16Array,
        UInt8Array, UInt16Array, UInt32Array,
    };
    use arrow_buffer::i256;

    let dictionary = DictionaryArray::<Int8Type>::try_new(
        Int8Array::from(vec![Some(0), None, Some(1), Some(2)]),
        Arc::new(Int8Array::from(vec![1, 2, 3])),
    )
    .unwrap();
    let cases: Vec<(&str, ArrayRef, PrimitiveType, Datum)> = vec![
        (
            "int8",
            Arc::new(Int8Array::from(vec![Some(1), None, Some(2), Some(3)])),
            PrimitiveType::Int,
            Datum::int(1000),
        ),
        (
            "int16",
            Arc::new(Int16Array::from(vec![Some(1), None, Some(2), Some(3)])),
            PrimitiveType::Long,
            Datum::long(100_000),
        ),
        (
            "uint8",
            Arc::new(UInt8Array::from(vec![Some(1), None, Some(2), Some(3)])),
            PrimitiveType::Int,
            Datum::int(1000),
        ),
        (
            "uint16",
            Arc::new(UInt16Array::from(vec![Some(1), None, Some(2), Some(3)])),
            PrimitiveType::Int,
            Datum::int(100_000),
        ),
        (
            "uint32",
            Arc::new(UInt32Array::from(vec![
                Some(1),
                None,
                Some(u32::MAX),
                Some(3),
            ])),
            PrimitiveType::Long,
            Datum::long(i64::from(u32::MAX) + 1),
        ),
        (
            "dictionary-int8",
            Arc::new(dictionary),
            PrimitiveType::Int,
            Datum::int(1000),
        ),
        (
            "decimal32",
            Arc::new(
                Decimal32Array::from(vec![Some(1), None, Some(2), Some(3)])
                    .with_precision_and_scale(8, 0)
                    .unwrap(),
            ),
            PrimitiveType::Decimal {
                precision: 12,
                scale: 0,
            },
            Datum::decimal_from_str("10000000000").unwrap(),
        ),
        (
            "decimal64",
            Arc::new(
                Decimal64Array::from(vec![Some(1), None, Some(2), Some(3)])
                    .with_precision_and_scale(15, 0)
                    .unwrap(),
            ),
            PrimitiveType::Decimal {
                precision: 30,
                scale: 0,
            },
            Datum::decimal_from_str("10000000000000000000000000").unwrap(),
        ),
        (
            "decimal256",
            Arc::new(
                Decimal256Array::from(vec![
                    Some(i256::from_i128(1)),
                    None,
                    Some(i256::from_i128(2)),
                    Some(i256::from_i128(3)),
                ])
                .with_precision_and_scale(20, 0)
                .unwrap(),
            ),
            PrimitiveType::Decimal {
                precision: 30,
                scale: 0,
            },
            Datum::decimal_from_str("10000000000000000000000000").unwrap(),
        ),
    ];
    let temp = TempDir::new().unwrap();
    for (name, values, target, beyond_source) in cases {
        let path = temp.path().join(format!("promotion-{name}.parquet"));
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            field("id", DataType::Int32, 1),
            Field::new("value", values.data_type().clone(), true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
            values,
        ])
        .unwrap();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), arrow_schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let target_type = Type::Primitive(target);
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "value", target_type.clone()).into(),
                ])
                .build()
                .unwrap(),
        );
        let datum = |value: i64| match target_type {
            Type::Primitive(PrimitiveType::Decimal { .. }) => {
                Datum::decimal_from_str(value.to_string())
                    .unwrap()
                    .to(&target_type)
                    .unwrap()
            }
            _ => Datum::long(value).to(&target_type).unwrap(),
        };
        let keys = || {
            std::iter::once(datum(1))
                .chain(std::iter::once(beyond_source.clone()))
                .chain((4..20).map(datum))
                .collect::<Vec<_>>()
        };
        for (predicate, expected) in [
            (
                Reference::new("value").less_than(beyond_source.clone()),
                vec![0, 2, 3],
            ),
            (
                Reference::new("value").not_equal_to(beyond_source.clone()),
                vec![0, 2, 3],
            ),
            (Reference::new("value").is_in(keys()), vec![0]),
            (Reference::new("value").is_not_in(keys()), vec![2, 3]),
        ] {
            let planned = predicate.clone().bind(schema.clone(), false).unwrap();
            let task = scan_task(
                path.to_str().unwrap().to_string(),
                schema.clone(),
                Some(planned),
            );
            for provider in [
                None,
                Some(Arc::new(ChangingRuntimePredicate::new(None, 0))
                    as Arc<dyn RuntimePredicateProvider>),
            ] {
                let (batches, _) = execute(task.clone(), provider).await;
                assert_eq!(ids(&batches), expected, "{name}: {predicate}");
            }
        }
    }
}

#[tokio::test]
async fn runtime_predicate_respects_task_byte_range() {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "split.parquet");
    let parquet = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let start = 4 + parquet.metadata().row_group(0).compressed_size() as u64;
    let file_size = std::fs::metadata(&path).unwrap().len();
    let task = FileScanTask::builder()
        .with_file_size_in_bytes(file_size)
        .with_start(start)
        .with_length(file_size - start)
        .with_data_file_path(path)
        .with_data_file_format(DataFileFormat::Parquet)
        .with_schema(iceberg_schema())
        .with_project_field_ids(vec![1, 2])
        .with_case_sensitive(false)
        .build()
        .unwrap();
    // RG0 belongs to another split, so even a predicate matching it reads nothing there.
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id")
            .less_than(Datum::int(4))
            .or(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
    ));
    let (batches, _) = execute(task, Some(provider)).await;
    assert_eq!(ids(&batches), vec![200, 201, 202, 203]);
}

#[tokio::test]
async fn runtime_predicate_with_bloom_filters_keeps_planned_equality() {
    let temp = TempDir::new().unwrap();
    let path = format!("{}/bloom.parquet", temp.path().to_str().unwrap());
    write_groups(&path, &[0, 100, 200], true, None, true);
    let planned = Reference::new("id")
        .equal_to(Datum::int(201))
        .bind(iceberg_schema(), false)
        .unwrap();
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").greater_than_or_equal_to(Datum::int(100)),
    ));
    let (batches, _) = execute_tasks(
        vec![scan_task(path, iceberg_schema(), Some(planned))],
        Some(provider),
        true,
    )
    .await;
    assert_eq!(ids(&batches), vec![201]);
}

/// A provider whose predicate changes when [`Self::publish`] is called.
struct ChangingRuntimePredicate {
    generation: AtomicU64,
    publication: std::sync::Mutex<(u64, Option<Predicate>)>,
    snapshots: AtomicU64,
}

impl ChangingRuntimePredicate {
    fn new(predicate: Option<Predicate>, generation: u64) -> Self {
        Self {
            generation: AtomicU64::new(generation),
            publication: std::sync::Mutex::new((generation, predicate)),
            snapshots: AtomicU64::new(0),
        }
    }

    fn publish(&self, predicate: Option<Predicate>, generation: u64) {
        self.publish_with_gate(predicate, generation, || {});
    }

    fn publish_with_gate(
        &self,
        predicate: Option<Predicate>,
        generation: u64,
        gate: impl FnOnce(),
    ) {
        let mut publication = self.publication.lock().unwrap();
        assert!(generation > publication.0);
        *publication = (generation, predicate);
        gate();
        self.generation.store(generation, Ordering::Release);
    }

    fn snapshots(&self) -> u64 {
        self.snapshots.load(Ordering::Relaxed)
    }
}

impl RuntimePredicateProvider for ChangingRuntimePredicate {
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
        self.snapshots.fetch_add(1, Ordering::Relaxed);
        let publication = self.publication.lock().unwrap();
        Ok(RuntimePredicateSnapshot::new(
            publication.1.clone(),
            publication.0,
        ))
    }
}

#[tokio::test]
async fn runtime_predicate_is_bound_once_per_generation_across_tasks() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().to_str().unwrap();
    let tasks: Vec<_> = ["a.parquet", "b.parquet", "c.parquet"]
        .into_iter()
        .map(|name| {
            scan_task(
                write_three_row_group_file(dir, name),
                iceberg_schema(),
                None,
            )
        })
        .collect();
    let provider = Arc::new(ChangingRuntimePredicate::new(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(100))),
        1,
    ));
    let scan = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_data_file_concurrency_limit(1)
        .with_runtime_predicate_provider(provider.clone())
        .build()
        .read(Box::pin(futures::stream::iter(tasks.into_iter().map(Ok))) as FileScanTaskStream)
        .unwrap();
    let mut stream = scan.stream();
    // The first task reads RG1 and RG2 under generation 1.
    let mut batches = vec![
        stream.try_next().await.unwrap().unwrap(),
        stream.try_next().await.unwrap().unwrap(),
    ];
    assert_eq!(ids(&batches), vec![100, 101, 102, 103, 200, 201, 202, 203]);
    assert_eq!(provider.snapshots(), 1);
    // A tighter generation is picked up by the tasks that start afterwards and
    // bound once for both of them.
    provider.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
        2,
    );
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    let mut expected = vec![100, 101, 102, 103, 200, 201, 202, 203];
    for _ in 0..2 {
        expected.extend([200, 201, 202, 203]);
    }
    assert_eq!(ids(&batches), expected);
    assert_eq!(provider.snapshots(), 2);
}

#[tokio::test]
async fn runtime_predicate_on_a_column_that_is_not_projected() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "filter-only.parquet");
    // Only `payload` is projected; the runtime predicate filters on `id`.
    let task = scan_task_with_deletes_and_projection(path, iceberg_schema(), None, vec![], vec![2]);
    let (baseline, baseline_metrics) = execute(task.clone(), None).await;
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").greater_than_or_equal_to(Datum::int(200)),
    ));
    let (batches, metrics) = execute(task, Some(provider)).await;
    let rows = |batches: &[RecordBatch]| batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    assert_eq!(rows(&baseline), 12);
    assert_eq!(rows(&batches), 4);
    assert!(batches.iter().all(|batch| batch.num_columns() == 1));
    assert!(metrics.bytes_read() < baseline_metrics.bytes_read());
}

#[tokio::test]
async fn runtime_predicate_resolves_files_without_field_ids() {
    use crate::spec::{MappedField, NameMapping};

    let temp = TempDir::new().unwrap();
    let path = format!("{}/no-ids.parquet", temp.path().to_str().unwrap());
    write_groups(&path, &[0, 100, 200], false, None, false);
    let mapping = Arc::new(NameMapping::new(vec![
        MappedField::new(Some(1), vec!["id".to_string()], vec![]),
        MappedField::new(Some(2), vec!["payload".to_string()], vec![]),
    ]));
    // A name mapping assigns ids by name; without one, ids follow column positions.
    for name_mapping in [Some(mapping), None] {
        let task = FileScanTask::builder()
            .with_file_size_in_bytes(std::fs::metadata(&path).unwrap().len())
            .with_start(0)
            .with_length(0)
            .with_data_file_path(path.clone())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(iceberg_schema())
            .with_project_field_ids(vec![1, 2])
            .with_name_mapping(name_mapping)
            .with_case_sensitive(false)
            .build()
            .unwrap();
        let provider = Arc::new(FixedRuntimePredicate::new(
            Reference::new("id").greater_than_or_equal_to(Datum::int(200)),
        ));
        let (batches, _) = execute(task, Some(provider)).await;
        assert_eq!(ids(&batches), vec![200, 201, 202, 203]);
    }
}

#[tokio::test]
async fn runtime_predicate_prunes_pages_within_a_row_group() {
    use parquet::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};

    const PAGE_ROWS: i32 = 1024;
    let temp = TempDir::new().unwrap();
    let path = format!("{}/pages.parquet", temp.path().to_str().unwrap());
    // One row group of four pages. Pages hold many rows so the reader skips
    // unselected pages instead of decoding them under a row mask.
    let schema = Arc::new(ArrowSchema::new(vec![field("id", DataType::Int32, 1)]));
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_data_page_row_count_limit(PAGE_ROWS as usize)
        .set_write_batch_size(PAGE_ROWS as usize)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(&path).unwrap(),
        Arc::clone(&schema),
        Some(props),
    )
    .unwrap();
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(
        Int32Array::from_iter_values(0..4 * PAGE_ROWS),
    )])
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let metadata = ParquetMetaDataReader::new()
        .with_page_index_policy(PageIndexPolicy::Required)
        .parse_and_finish(&File::open(&path).unwrap())
        .unwrap();
    assert_eq!(metadata.num_row_groups(), 1);
    assert_eq!(
        metadata.offset_index().unwrap()[0][0]
            .page_locations()
            .len(),
        4
    );

    let read = |row_selection: bool| {
        let task = scan_task_with_deletes_and_projection(
            path.clone(),
            iceberg_schema(),
            None,
            vec![],
            vec![1],
        );
        async move {
            let provider = Arc::new(FixedRuntimePredicate::new(
                Reference::new("id")
                    .greater_than_or_equal_to(Datum::int(2 * PAGE_ROWS))
                    .and(Reference::new("id").less_than(Datum::int(3 * PAGE_ROWS))),
            ));
            let scan = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
                .with_row_group_filtering_enabled(false)
                .with_row_selection_enabled(row_selection)
                // Keep the skipped page ranges separate; the default 1 MiB
                // coalescing threshold would read them along with selected pages.
                .with_range_coalesce_bytes(0)
                .with_runtime_predicate_provider(provider)
                .build()
                .read(Box::pin(futures::stream::iter([Ok(task)])) as FileScanTaskStream)
                .unwrap();
            let metrics = scan.metrics().clone();
            let batches: Vec<RecordBatch> = scan.stream().try_collect().await.unwrap();
            (ids(&batches), metrics.bytes_read())
        }
    };
    let expected: Vec<i32> = (2 * PAGE_ROWS..3 * PAGE_ROWS).collect();
    // Without page selection the row filter decodes every page.
    let (unpruned_ids, unpruned) = read(false).await;
    let (pruned_ids, pruned) = read(true).await;
    assert_eq!(unpruned_ids, expected);
    assert_eq!(pruned_ids, expected);
    assert!(pruned < unpruned, "pruned={pruned} unpruned={unpruned}");
}

#[tokio::test]
async fn runtime_predicate_that_cannot_be_planned_keeps_the_planned_filter() {
    use arrow_array::StructArray;

    let temp = TempDir::new().unwrap();
    let path = format!("{}/nested.parquet", temp.path().to_str().unwrap());
    let x = field("x", DataType::Int32, 4);
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, 1),
        field("s", DataType::Struct(vec![x.clone()].into()), 3),
    ]));
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(4))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(&path).unwrap(),
        Arc::clone(&arrow_schema),
        Some(props),
    )
    .unwrap();
    for base in [0, 100, 200] {
        let s = StructArray::from(vec![(
            Arc::new(x.clone()),
            Arc::new(Int32Array::from(vec![-1; 4])) as ArrayRef,
        )]);
        let batch = RecordBatch::try_new(Arc::clone(&arrow_schema), vec![
            Arc::new(Int32Array::from((base..base + 4).collect::<Vec<_>>())),
            Arc::new(s),
        ])
        .unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();

    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(
                    3,
                    "s",
                    Type::Struct(crate::spec::StructType::new(vec![
                        NestedField::required(4, "x", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let task = |predicate: Option<crate::expr::BoundPredicate>| {
        scan_task_with_deletes_and_projection(
            path.clone(),
            Arc::clone(&schema),
            predicate,
            vec![],
            vec![1],
        )
    };
    let planned = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(100))
        .bind(Arc::clone(&schema), false)
        .unwrap();
    // The reader cannot build row filters on nested columns, so this passes the
    // column checks but fails planning; applied, it would reject every row.
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("s.x").greater_than_or_equal_to(Datum::int(0)),
    ));
    let (_, baseline) = execute(task(None), None).await;
    let (batches, metrics) = execute(task(Some(planned)), Some(provider)).await;
    assert_eq!(ids(&batches), vec![100, 101, 102, 103, 200, 201, 202, 203]);
    // The planned predicate still prunes RG0.
    assert!(metrics.bytes_read() < baseline.bytes_read());
}

#[test]
fn runtime_page_selection_failure_keeps_the_planned_selection() {
    use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
    use parquet::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};

    use super::ArrowReader;
    use super::runtime_predicate::intersect_page_selection;

    let temp = TempDir::new().unwrap();
    let path = write_row_group_file_with_page_size(
        temp.path().to_str().unwrap(),
        "pages.parquet",
        &[0],
        1,
    );
    let metadata = Arc::new(
        ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Required)
            .parse_and_finish(&File::open(&path).unwrap())
            .unwrap(),
    );
    // The page-index evaluator rejects NOT, which the reader rewrites away
    // before planning, so this yields a real evaluation error.
    let predicate = (!Reference::new("id").less_than(Datum::int(2)))
        .bind(iceberg_schema(), false)
        .unwrap();
    let field_id_map = HashMap::from([(1, 0), (2, 1)]);
    let selection = || {
        ArrowReader::get_row_selection_for_filter_predicate(
            &predicate,
            &metadata,
            &None,
            &field_id_map,
            &iceberg_schema(),
        )
    };
    assert!(selection().is_err());

    let planned = RowSelection::from(vec![RowSelector::skip(1), RowSelector::select(3)]);
    let combined =
        intersect_page_selection(Some(planned.clone()), selection(), true, &path).unwrap();
    assert_eq!(combined, Some(planned));
    assert!(intersect_page_selection(None, selection(), false, &path).is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn runtime_predicate_concurrent_tasks_reuse_success_and_failure() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "concurrent.parquet");
    let tasks: Vec<_> = (0..12)
        .map(|_| scan_task(path.clone(), iceberg_schema(), None))
        .collect();
    let success = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").greater_than_or_equal_to(Datum::int(200)),
    ));
    let absent = Arc::new(ChangingRuntimePredicate::new(None, 1));
    let failure = Arc::new(FailedProvider::default());
    let binding_failure = Arc::new(FixedRuntimePredicate::new(
        Reference::new("missing").equal_to(Datum::int(0)),
    ));
    for (provider, expected) in [
        (success.clone() as Arc<dyn RuntimePredicateProvider>, vec![
            200, 201, 202, 203,
        ]),
        (
            absent.clone() as Arc<dyn RuntimePredicateProvider>,
            all_ids(),
        ),
        (
            failure.clone() as Arc<dyn RuntimePredicateProvider>,
            all_ids(),
        ),
        (
            binding_failure.clone() as Arc<dyn RuntimePredicateProvider>,
            all_ids(),
        ),
    ] {
        let (batches, _) =
            execute_tasks_with_concurrency(tasks.clone(), Some(provider), false, 4).await;
        let mut actual = ids(&batches);
        actual.sort_unstable();
        let mut expected = expected.repeat(tasks.len());
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }
    assert_eq!(success.snapshots(), 1);
    assert_eq!(absent.snapshots(), 1);
    assert_eq!(failure.snapshots.load(Ordering::Relaxed), 1);
    assert_eq!(binding_failure.snapshots(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn runtime_predicate_concurrent_tasks_pick_up_new_generation_without_regression() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "generations.parquet");
    let task = scan_task(path, iceberg_schema(), None);
    let provider = Arc::new(ChangingRuntimePredicate::new(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(100))),
        1,
    ));
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let next = task.clone();
    // Withhold the second wave until the four first-wave tasks have completed.
    // This tests a publication boundary without depending on file-read timing.
    let tasks = futures::stream::iter(vec![task.clone(); 4].into_iter().map(Ok))
        .chain(futures::stream::once(async move {
            release_rx.await.unwrap();
            Ok(next)
        }))
        .chain(futures::stream::iter(vec![task; 3].into_iter().map(Ok)));
    let scan = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_data_file_concurrency_limit(4)
        .with_runtime_predicate_provider(provider.clone())
        .build()
        .read(Box::pin(tasks) as FileScanTaskStream)
        .unwrap();
    let mut stream = scan.stream();
    let mut first_wave = vec![];
    // Each first-wave file produces two four-row batches (RG1 and RG2).
    for _ in 0..8 {
        first_wave.push(stream.try_next().await.unwrap().unwrap());
    }
    let mut actual = ids(&first_wave);
    actual.sort_unstable();
    let mut expected = [100, 101, 102, 103, 200, 201, 202, 203].repeat(4);
    expected.sort_unstable();
    assert_eq!(actual, expected);
    assert_eq!(provider.snapshots(), 1);

    provider.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
        2,
    );
    release_tx.send(()).unwrap();
    let second_wave = stream.try_collect::<Vec<_>>().await.unwrap();
    let mut actual = ids(&second_wave);
    actual.sort_unstable();
    let mut expected = [200, 201, 202, 203].repeat(4);
    expected.sort_unstable();
    assert_eq!(actual, expected);
    assert_eq!(provider.snapshots(), 2);
}

async fn check_promoted_runtime_column(
    path: &str,
    file_type: DataType,
    table_type: PrimitiveType,
    values: ArrayRef,
    predicate: Predicate,
) -> Vec<RecordBatch> {
    write_delete(
        path,
        vec![
            field("id", DataType::Int32, 1),
            field("value", file_type, 3),
        ],
        vec![Arc::new(Int32Array::from(vec![0, 1, 2, 3])), values],
    );
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(3, "value", Type::Primitive(table_type)).into(),
            ])
            .build()
            .unwrap(),
    );
    let planned = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(1))
        .bind(schema.clone(), false)
        .unwrap();
    let task = scan_task_with_deletes_and_projection(
        path.to_string(),
        schema,
        Some(planned),
        vec![],
        vec![1, 3],
    );
    let (baseline, _) = execute(task.clone(), None).await;
    let provider = Arc::new(FixedRuntimePredicate::new(predicate));
    let (batches, _) = execute(task, Some(provider.clone())).await;
    assert_eq!(ids(&batches), vec![1, 2, 3]);
    assert_eq!(
        batches, baseline,
        "promotion must fail open and preserve the planned filter"
    );
    assert_eq!(provider.snapshots(), 1);
    batches
}

#[tokio::test]
async fn runtime_predicate_on_float_promoted_to_double_is_ignored() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("float.parquet");
    // The DOUBLE boundary is between adjacent FLOAT values. Narrowing it
    // changes the comparison even though it lies within the FLOAT range.
    let boundary = 1.0 + f64::from(f32::EPSILON) / 2.0;
    let batches = check_promoted_runtime_column(
        path.to_str().unwrap(),
        DataType::Float32,
        PrimitiveType::Double,
        Arc::new(Float32Array::from(vec![
            1.0,
            1.0,
            1.0 + f32::EPSILON,
            1.0 + f32::EPSILON,
        ])),
        Reference::new("value").less_than(Datum::double(boundary)),
    )
    .await;
    assert_eq!(batches[0].column(1).data_type(), &DataType::Float64);
    let values = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(values.values().as_ref(), &[
        1.0,
        f64::from(1.0 + f32::EPSILON),
        f64::from(1.0 + f32::EPSILON)
    ]);
}

#[tokio::test]
async fn runtime_predicate_on_widened_decimal_precision_is_ignored() {
    let temp = TempDir::new().unwrap();
    // 8 -> 9 can share the same physical width; precision itself must be
    // checked, rather than only the Parquet physical storage type.
    for precision in [9, 12] {
        let path = temp.path().join(format!("decimal-{precision}.parquet"));
        let batches = check_promoted_runtime_column(
            path.to_str().unwrap(),
            DataType::Decimal128(8, 2),
            PrimitiveType::Decimal {
                precision,
                scale: 2,
            },
            Arc::new(
                Decimal128Array::from(vec![123, 456, 789, 1000])
                    .with_precision_and_scale(8, 2)
                    .unwrap(),
            ),
            // Fits the table precision but exceeds the file's precision.
            Reference::new("value").greater_than(Datum::decimal_from_str("1000000.00").unwrap()),
        )
        .await;
        assert_eq!(
            batches[0].column(1).data_type(),
            &DataType::Decimal128(precision as u8, 2)
        );
        let values = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap();
        assert_eq!(values.values().as_ref(), &[456, 789, 1000]);
    }
}

#[tokio::test]
async fn runtime_predicate_row_group_pruning_preserves_projected_positions() {
    use crate::metadata_columns::RESERVED_FIELD_ID_POS;

    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "positions.parquet");
    let task = scan_task_with_deletes_and_projection(path, iceberg_schema(), None, vec![], vec![
        1,
        2,
        RESERVED_FIELD_ID_POS,
    ]);
    let positions = |batches: &[RecordBatch]| {
        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("_pos")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>()
    };
    let (baseline, baseline_metrics) = execute(task.clone(), None).await;
    assert_eq!(positions(&baseline), (0..12).collect::<Vec<i64>>());
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id")
            .greater_than_or_equal_to(Datum::int(100))
            .and(Reference::new("id").less_than_or_equal_to(Datum::int(103))),
    ));
    let (batches, metrics) = execute(task, Some(provider)).await;
    assert_eq!(ids(&batches), vec![100, 101, 102, 103]);
    // RG0 was pruned, but RowNumber must still use RG1's original file ordinals.
    assert_eq!(positions(&batches), vec![4, 5, 6, 7]);
    assert!(metrics.bytes_read() < baseline_metrics.bytes_read());
}

#[tokio::test]
async fn runtime_predicate_shares_bloom_filter_reads_with_planned_predicate() {
    let temp = TempDir::new().unwrap();
    let path = format!("{}/shared-bloom.parquet", temp.path().to_str().unwrap());
    // Each row group spans the probed values, so statistics prune nothing and
    // only the bloom filters can prune row groups 0 and 2.
    let schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, 1),
        field("payload", DataType::Utf8, 2),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_max_row_group_row_count(Some(4))
        .set_bloom_filter_enabled(true)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(&path).unwrap(),
        Arc::clone(&schema),
        Some(props),
    )
    .unwrap();
    for ids in [[0, 500, 1, 501], [2, 502, 3, 503], [4, 504, 5, 505]] {
        let payloads: Vec<String> = ids.iter().map(|id| format!("payload-{id}")).collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(payloads)),
        ])
        .unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();

    let bind = |predicate: Predicate| predicate.bind(iceberg_schema(), false).unwrap();
    let planned = Reference::new("id").equal_to(Datum::int(3));
    let runtime = Reference::new("id").is_in([Datum::int(3), Datum::int(77)]);
    // The same restriction as one planned predicate, which reads each bloom
    // filter once.
    let (expected, expected_metrics) = execute_tasks(
        vec![scan_task(
            path.clone(),
            iceberg_schema(),
            Some(bind(planned.clone().and(runtime.clone()))),
        )],
        None,
        true,
    )
    .await;
    let (batches, metrics) = execute_tasks(
        vec![scan_task(path, iceberg_schema(), Some(bind(planned)))],
        Some(Arc::new(FixedRuntimePredicate::new(runtime))),
        true,
    )
    .await;
    assert_eq!(ids(&expected), vec![3]);
    assert_eq!(ids(&batches), vec![3]);
    // A bloom filter is far larger than the rows read, so a repeated read of
    // the overlapping column would show up here.
    assert_eq!(metrics.bytes_read(), expected_metrics.bytes_read());
}

async fn next_ids(stream: &mut ArrowRecordBatchStream, batches: usize) -> Vec<i32> {
    let mut read = vec![];
    for _ in 0..batches {
        read.push(stream.try_next().await.unwrap().unwrap());
    }
    ids(&read)
}

#[tokio::test]
async fn runtime_predicate_transitions_to_none_and_then_to_a_tighter_predicate() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "transitions.parquet");
    let task = scan_task(path, iceberg_schema(), None);
    let provider = Arc::new(ChangingRuntimePredicate::new(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(100))),
        1,
    ));
    // Tasks are released one at a time, so each is planned after the
    // preceding publication.
    let (tasks, released) = futures::channel::mpsc::unbounded::<FileScanTask>();
    let scan = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_data_file_concurrency_limit(1)
        .with_runtime_predicate_provider(provider.clone())
        .build()
        .read(Box::pin(released.map(Ok)) as FileScanTaskStream)
        .unwrap();
    let mut stream = scan.stream();

    tasks.unbounded_send(task.clone()).unwrap();
    assert_eq!(next_ids(&mut stream, 2).await, vec![
        100, 101, 102, 103, 200, 201, 202, 203
    ]);
    assert_eq!(provider.snapshots(), 1);

    // `None` stops pruning for later tasks; the cached predicate must not be reused.
    provider.publish(None, 2);
    tasks.unbounded_send(task.clone()).unwrap();
    assert_eq!(next_ids(&mut stream, 3).await, all_ids());
    assert_eq!(provider.snapshots(), 2);

    // A later predicate tighter than the first one applies again.
    provider.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
        3,
    );
    tasks.unbounded_send(task.clone()).unwrap();
    assert_eq!(next_ids(&mut stream, 1).await, vec![200, 201, 202, 203]);
    assert_eq!(provider.snapshots(), 3);

    // The unchanged generation is reused, not snapshotted again.
    tasks.unbounded_send(task).unwrap();
    assert_eq!(next_ids(&mut stream, 1).await, vec![200, 201, 202, 203]);
    assert_eq!(provider.snapshots(), 3);

    drop(tasks);
    assert!(stream.try_next().await.unwrap().is_none());
}

fn three_group_file_metrics() -> crate::scan::FileScanTaskMetrics {
    crate::scan::FileScanTaskMetrics::builder()
        .with_record_count(Some(12))
        .with_value_counts(HashMap::new())
        .with_null_value_counts(HashMap::new())
        .with_nan_value_counts(HashMap::new())
        .with_lower_bounds(HashMap::from([(1, Datum::int(0))]))
        .with_upper_bounds(HashMap::from([(1, Datum::int(203))]))
        .build()
}

fn with_file_metrics(
    task: FileScanTask,
    metrics: crate::scan::FileScanTaskMetrics,
) -> FileScanTask {
    // Statistics also survive task serialization, as for distributed engines.
    let mut json = serde_json::to_value(task).unwrap();
    json["file_metrics"] = serde_json::to_value(metrics).unwrap();
    serde_json::from_value(json).unwrap()
}

#[tokio::test]
async fn empty_projection_counts_filtered_rows_without_reading_payload() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "count_rows.parquet");
    let schema = iceberg_schema();
    for planned in [
        None,
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(100))),
        Some(Reference::new("id").less_than(Datum::int(0))),
    ] {
        for live in [false, true] {
            let predicate = planned
                .clone()
                .map(|predicate| predicate.bind(schema.clone(), false).unwrap());
            let provider = live.then(|| {
                Arc::new(FixedRuntimePredicate::new(
                    Reference::new("id").less_than_or_equal_to(Datum::int(101)),
                )) as Arc<dyn RuntimePredicateProvider>
            });
            let projected = scan_task_with_deletes_and_projection(
                path.clone(),
                schema.clone(),
                predicate.clone(),
                vec![],
                vec![1, 2],
            );
            let empty = scan_task_with_deletes_and_projection(
                path.clone(),
                schema.clone(),
                predicate,
                vec![],
                vec![],
            );
            let (reference, reference_metrics) = execute(projected, provider.clone()).await;
            let (counted, count_metrics) = execute(empty, provider).await;
            let expected = ids(&reference).len();
            assert_eq!(
                counted.iter().map(RecordBatch::num_rows).sum::<usize>(),
                expected
            );
            assert!(counted.iter().all(|batch| batch.num_columns() == 0));
            if expected > 0 {
                assert!(count_metrics.bytes_read() < reference_metrics.bytes_read());
            }
        }
    }
}

#[tokio::test]
async fn planned_all_match_filter_does_not_decode_predicate_only_column() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "planned_all.parquet");
    let schema = iceberg_schema();
    let predicate = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(0))
        .bind(schema.clone(), false)
        .unwrap();
    let task =
        scan_task_with_deletes_and_projection(path, schema, Some(predicate), vec![], vec![2]);
    let metrics = crate::scan::FileScanTaskMetrics::builder()
        .with_record_count(Some(12))
        .with_value_counts(HashMap::from([(1, 12)]))
        .with_null_value_counts(HashMap::from([(1, 0)]))
        .with_nan_value_counts(HashMap::new())
        .with_lower_bounds(HashMap::from([(1, Datum::int(0))]))
        .with_upper_bounds(HashMap::from([(1, Datum::int(203))]))
        .build();
    let (unproven, unproven_metrics) = execute(task.clone(), None).await;
    let (proven, proven_metrics) = execute(with_file_metrics(task, metrics), None).await;
    assert_eq!(proven, unproven);
    assert!(
        proven_metrics.bytes_read() < unproven_metrics.bytes_read(),
        "proven={} unproven={}",
        proven_metrics.bytes_read(),
        unproven_metrics.bytes_read(),
    );
}

#[tokio::test]
async fn nan_equality_deletes_with_file_metrics_preserve_empty_projection_counts() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().to_str().unwrap();
    let data_path = format!("{dir}/nan_data.parquet");
    let delete_path = format!("{dir}/nan_delete.parquet");
    let key = field("key", DataType::Float64, 1).with_nullable(true);
    write_delete(&data_path, vec![key.clone()], vec![Arc::new(
        Float64Array::from(vec![Some(f64::NAN), Some(f64::NAN)]),
    )]);
    write_delete(&delete_path, vec![key], vec![Arc::new(Float64Array::from(
        vec![Some(f64::NAN)],
    ))]);
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "key", Type::Primitive(PrimitiveType::Double)).into(),
            ])
            .build()
            .unwrap(),
    );
    let delete = FileScanTaskDeleteFile::builder()
        .with_file_size_in_bytes(std::fs::metadata(&delete_path).unwrap().len())
        .with_file_path(delete_path)
        .with_file_type(DataContentType::EqualityDeletes)
        .with_file_format(DataFileFormat::Parquet)
        .with_partition_spec_id(0)
        .with_equality_ids(Some(vec![1]))
        .build()
        .unwrap();
    let metrics = crate::scan::FileScanTaskMetrics::builder()
        .with_record_count(Some(2))
        .with_value_counts(HashMap::from([(1, 2)]))
        .with_null_value_counts(HashMap::from([(1, 0)]))
        .with_nan_value_counts(HashMap::from([(1, 2)]))
        .with_lower_bounds(HashMap::new())
        .with_upper_bounds(HashMap::new())
        .build();
    for projection in [vec![1], vec![]] {
        let task = scan_task_with_deletes_and_projection(
            data_path.clone(),
            schema.clone(),
            None,
            vec![delete.clone()],
            projection,
        );
        let (unproven, _) = execute(task.clone(), None).await;
        let (proven, _) = execute(with_file_metrics(task, metrics.clone()), None).await;
        assert_eq!(unproven.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        assert_eq!(proven.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    }
}

#[tokio::test]
async fn runtime_predicate_file_rejection_precedes_data_and_delete_io() {
    for delete_type in [
        None,
        Some(DataContentType::PositionDeletes),
        Some(DataContentType::EqualityDeletes),
    ] {
        // Neither the data file nor the delete file exists: any read would fail.
        let deletes = delete_type
            .map(|file_type| {
                FileScanTaskDeleteFile::builder()
                    .with_file_path("/does-not-exist/delete.parquet".to_string())
                    .with_file_size_in_bytes(100)
                    .with_file_type(file_type)
                    .with_file_format(DataFileFormat::Parquet)
                    .with_partition_spec_id(0)
                    .with_equality_ids(
                        (file_type == DataContentType::EqualityDeletes).then_some(vec![1]),
                    )
                    .build()
                    .unwrap()
            })
            .into_iter()
            .collect();
        // A byte-range split: whole-file statistics apply to every split.
        let task = FileScanTask::builder()
            .with_file_size_in_bytes(100)
            .with_start(40)
            .with_length(20)
            .with_data_file_path("/does-not-exist/data.parquet".to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(iceberg_schema())
            .with_project_field_ids(vec![1, 2])
            .with_file_metrics(Some(Arc::new(three_group_file_metrics())))
            .with_deletes(deletes)
            .with_case_sensitive(false)
            .build()
            .unwrap();
        let provider = Arc::new(FixedRuntimePredicate::new(
            Reference::new("id").greater_than(Datum::int(203)),
        ));
        let (batches, metrics) = execute(task, Some(provider.clone())).await;
        assert!(batches.is_empty());
        assert_eq!(metrics.bytes_read(), 0);
        assert_eq!(provider.snapshots(), 1);
    }
}

#[tokio::test]
async fn runtime_predicate_file_statistics_keep_possible_matches_and_fail_open() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "file_stats.parquet");
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").greater_than_or_equal_to(Datum::int(200)),
    ));
    let stats = |lower: Option<Datum>, upper: Option<Datum>, record_count| {
        crate::scan::FileScanTaskMetrics::builder()
            .with_record_count(record_count)
            .with_value_counts(HashMap::new())
            .with_null_value_counts(HashMap::new())
            .with_nan_value_counts(HashMap::new())
            .with_lower_bounds(lower.map(|bound| (1, bound)).into_iter().collect())
            .with_upper_bounds(upper.map(|bound| (1, bound)).into_iter().collect())
            .build()
    };
    for file_metrics in [
        // Bounds that intersect the predicate keep the file.
        three_group_file_metrics(),
        // Missing statistics keep a non-empty file.
        stats(None, None, None),
        stats(None, None, Some(12)),
        // A bound of the wrong type cannot exclude a correctly typed predicate.
        stats(None, Some(Datum::long(0)), Some(12)),
    ] {
        let task = with_file_metrics(
            scan_task(path.clone(), iceberg_schema(), None),
            file_metrics,
        );
        let (batches, metrics) = execute(task, Some(provider.clone())).await;
        assert_eq!(ids(&batches), vec![200, 201, 202, 203]);
        assert!(metrics.bytes_read() > 0);
    }

    // NOT is normalized before statistics evaluation: NOT(id < 102) can match.
    let task = with_file_metrics(
        scan_task(path.clone(), iceberg_schema(), None),
        three_group_file_metrics(),
    );
    let provider = Arc::new(FixedRuntimePredicate::new(
        !Reference::new("id").less_than(Datum::int(102)),
    ));
    let (batches, _) = execute(task, Some(provider)).await;
    assert_eq!(ids(&batches), vec![102, 103, 200, 201, 202, 203]);

    // Without a runtime predicate, statistics never skip a file.
    let task = with_file_metrics(
        scan_task(path, iceberg_schema(), None),
        stats(Some(Datum::int(1000)), Some(Datum::int(2000)), Some(12)),
    );
    let (batches, _) = execute(task, None).await;
    assert_eq!(ids(&batches), all_ids());
}

#[tokio::test]
async fn runtime_predicate_file_pruning_ignores_incompatible_statistics_of_unreferenced_fields() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "unrelated-stats.parquet");
    let mut metrics = three_group_file_metrics();
    // One dropped field and one promoted field: neither participates in `id`.
    metrics.lower_bounds.insert(99, Datum::long(0));
    metrics.upper_bounds.insert(99, Datum::long(10));
    metrics.lower_bounds.insert(2, Datum::int(0));
    metrics.upper_bounds.insert(2, Datum::int(10));
    let task = with_file_metrics(scan_task(path, iceberg_schema(), None), metrics);
    let (batches, metrics) = execute(
        task,
        Some(Arc::new(FixedRuntimePredicate::new(
            Reference::new("id").greater_than(Datum::int(203)),
        ))),
    )
    .await;
    assert!(batches.is_empty());
    assert_eq!(metrics.bytes_read(), 0);
    assert_eq!(metrics.runtime_file_tasks_pruned(), 1);
}

#[tokio::test]
async fn runtime_predicate_metrics_attribute_only_runtime_pruning() {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "metrics.parquet");
    let range_100_103 = || {
        Reference::new("id")
            .greater_than_or_equal_to(Datum::int(100))
            .and(Reference::new("id").less_than_or_equal_to(Datum::int(103)))
    };

    // Without a provider nothing is recorded.
    let (_, metrics) = execute(scan_task(path.clone(), iceberg_schema(), None), None).await;
    assert_eq!(metrics.runtime_predicate_tasks(), 0);
    assert_eq!(metrics.runtime_row_groups_pruned(), 0);

    // The runtime predicate alone removes RG0 and RG2.
    let provider = Arc::new(FixedRuntimePredicate::new(range_100_103()));
    let (_, metrics) = execute(
        scan_task(path.clone(), iceberg_schema(), None),
        Some(provider),
    )
    .await;
    assert_eq!(metrics.runtime_predicate_tasks(), 1);
    assert_eq!(metrics.runtime_row_groups_pruned(), 2);
    assert_eq!(metrics.runtime_file_tasks_pruned(), 0);

    // The planned predicate already removes RG0; only RG2 is the runtime's.
    let planned = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(100))
        .bind(iceberg_schema(), false)
        .unwrap();
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").less_than_or_equal_to(Datum::int(103)),
    ));
    let (_, metrics) = execute(
        scan_task(path.clone(), iceberg_schema(), Some(planned)),
        Some(provider),
    )
    .await;
    assert_eq!(metrics.runtime_row_groups_pruned(), 1);

    // RG0 belongs to another byte-range split and is not counted.
    let parquet = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
    let start = 4 + parquet.metadata().row_group(0).compressed_size() as u64;
    let file_size = std::fs::metadata(&path).unwrap().len();
    let split = FileScanTask::builder()
        .with_file_size_in_bytes(file_size)
        .with_start(start)
        .with_length(file_size - start)
        .with_data_file_path(path.clone())
        .with_data_file_format(DataFileFormat::Parquet)
        .with_schema(iceberg_schema())
        .with_project_field_ids(vec![1, 2])
        .with_case_sensitive(false)
        .build()
        .unwrap();
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").greater_than_or_equal_to(Datum::int(200)),
    ));
    let (_, metrics) = execute(split, Some(provider)).await;
    assert_eq!(metrics.runtime_row_groups_pruned(), 1);

    // An unusable predicate is not counted.
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("missing").equal_to(Datum::int(1)),
    ));
    let (_, metrics) = execute(
        scan_task(path.clone(), iceberg_schema(), None),
        Some(provider),
    )
    .await;
    assert_eq!(metrics.runtime_predicate_tasks(), 0);
    assert_eq!(metrics.runtime_row_groups_pruned(), 0);

    // A task rejected from whole-file statistics is counted before any I/O.
    let task = with_file_metrics(
        scan_task(path, iceberg_schema(), None),
        three_group_file_metrics(),
    );
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").greater_than(Datum::int(203)),
    ));
    let (_, metrics) = execute(task, Some(provider)).await;
    assert_eq!(metrics.runtime_predicate_tasks(), 1);
    assert_eq!(metrics.runtime_file_tasks_pruned(), 1);
    assert_eq!(metrics.bytes_read(), 0);
}

#[tokio::test]
async fn runtime_predicate_that_prunes_every_row_group_reads_no_rows() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "none.parquet");
    let task = scan_task(path, iceberg_schema(), None);
    let (_, baseline) = execute(task.clone(), None).await;
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").greater_than_or_equal_to(Datum::int(1000)),
    ));
    let (batches, metrics) = execute(task, Some(provider)).await;
    assert!(batches.is_empty());
    assert!(metrics.bytes_read() < baseline.bytes_read());
}

fn start_runtime_scan(
    task: FileScanTask,
    provider: Option<Arc<dyn RuntimePredicateProvider>>,
    row_selection: bool,
    row_groups: bool,
    batch_size: usize,
) -> (ArrowRecordBatchStream, ScanMetrics) {
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_data_file_concurrency_limit(1)
        .with_row_selection_enabled(row_selection)
        .with_row_group_filtering_enabled(row_groups)
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
    let _: Vec<RecordBatch> = baseline.try_collect().await.unwrap();
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
    assert_eq!(provider.snapshots(), 1);
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(ids(&batches), vec![0, 1, 2, 3, 200, 201, 202, 203]);
    assert_eq!(provider.snapshots(), 2);
    assert!(metrics.bytes_read() < baseline_metrics.bytes_read());
}

#[tokio::test]
async fn runtime_predicate_live_tightening_only_prunes_remaining_groups() {
    let temp = TempDir::new().unwrap();
    let path = write_row_group_file(temp.path().to_str().unwrap(), "tightening.parquet", &[
        200, 0, 300, 100, 400, 350,
    ]);
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, _) = start_runtime_scan(
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
    assert_eq!(provider.snapshots(), 4);
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
    let (mut stream, _) = start_runtime_scan(
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
}

#[tokio::test]
async fn runtime_predicate_live_bad_publication_fails_open_and_next_generation_recovers() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "recover.parquet");
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, _) = start_runtime_scan(
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
    assert_eq!(provider.snapshots(), 3);
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
        .build()
        .unwrap();
    let task = scan_task_with_deletes(path, iceberg_schema(), None, vec![delete]);
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, _) = start_runtime_scan(task, Some(provider.clone()), false, true, 4);
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    assert_eq!(ids(&batches), vec![0, 2, 3]);
    provider.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
        1,
    );
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(ids(&batches), vec![0, 2, 3, 200, 201, 203]);
}

#[tokio::test]
async fn runtime_predicate_live_removal_preserves_local_page_selections() {
    let temp = TempDir::new().unwrap();
    let path = write_row_group_file_with_page_size(
        temp.path().to_str().unwrap(),
        "local-pages.parquet",
        &[0, 100, 200],
        1,
    );
    // Each row is a page. The static predicate retains only the last two
    // pages of RG0 and the first two pages of RG2, giving distinct local masks.
    let planned = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(2))
        .and(Reference::new("id").less_than_or_equal_to(Datum::int(201)))
        .bind(iceberg_schema(), false)
        .unwrap();
    let task = scan_task(path, iceberg_schema(), Some(planned));
    let (baseline, baseline_metrics) = start_runtime_scan(task.clone(), None, true, true, 1);
    let full = baseline.try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(ids(&full), vec![2, 3, 100, 101, 102, 103, 200, 201]);
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(task, Some(provider.clone()), true, true, 1);
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    assert_eq!(ids(&batches), vec![2]);
    provider.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
        1,
    );
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    // A generation change halfway through RG0 takes effect only after its
    // remaining selected page. Removing RG1 cannot shift RG2's local mask.
    assert_eq!(ids(&batches), vec![2, 3, 200, 201]);
    assert_eq!(provider.snapshots(), 2);
    assert!(metrics.bytes_read() < baseline_metrics.bytes_read());
}

#[tokio::test]
async fn runtime_predicate_live_can_remove_all_remaining_row_groups() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "all-pages.parquet");
    let provider = Arc::new(ChangingRuntimePredicate::new(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(0))),
        0,
    ));
    let (mut stream, _) = start_runtime_scan(
        scan_task(path, iceberg_schema(), None),
        Some(provider.clone()),
        true,
        true,
        4,
    );
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    provider.publish(Some(Predicate::AlwaysFalse), 1);
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(ids(&batches), vec![0, 1, 2, 3]);
}

#[tokio::test]
async fn runtime_predicate_live_refresh_is_off_when_row_group_filtering_is_disabled() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "disabled.parquet");
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, _) = start_runtime_scan(
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
    assert_eq!(provider.snapshots(), 1);
}

#[tokio::test]
async fn runtime_predicate_live_metrics_count_refreshes_and_pruned_groups() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "metrics.parquet");
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(
        scan_task(path, iceberg_schema(), None),
        Some(provider.clone()),
        false,
        true,
        4,
    );
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    assert_eq!(metrics.runtime_predicate_refreshes(), 0);
    provider.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
        1,
    );
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    assert_eq!(ids(&batches), vec![0, 1, 2, 3, 200, 201, 202, 203]);
    assert_eq!(metrics.runtime_predicate_tasks(), 1);
    assert_eq!(metrics.runtime_predicate_refreshes(), 1);
    assert_eq!(metrics.runtime_row_groups_pruned_live(), 1);
    // Nothing was pruned at task start, so the whole total is live pruning.
    assert_eq!(metrics.runtime_row_groups_pruned(), 1);
    assert_eq!(metrics.runtime_decoder_rebuilds(), 1);
}

#[tokio::test]
async fn runtime_predicate_failed_advisory_refresh_counts_no_rebuild() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "failed-refresh.parquet");
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(
        scan_task(path, iceberg_schema(), None),
        Some(provider.clone()),
        false,
        true,
        4,
    );
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    // An advisory predicate that cannot be validated (missing column) fails
    // open: the refresh is skipped and no pruning or decoder rebuild is
    // reported for work that never happened.
    provider.publish(Some(Reference::new("missing").equal_to(Datum::int(100))), 1);
    batches.push(stream.try_next().await.unwrap().unwrap());
    assert_eq!(ids(&batches), vec![0, 1, 2, 3, 100, 101, 102, 103]);
    assert_eq!(metrics.runtime_predicate_refreshes(), 0);
    assert_eq!(metrics.runtime_decoder_rebuilds(), 0);

    // The next usable publication is adopted and counted exactly once.
    provider.publish(Some(Predicate::AlwaysFalse), 2);
    assert!(stream.try_next().await.unwrap().is_none());
    assert_eq!(metrics.runtime_predicate_refreshes(), 1);
    assert_eq!(metrics.runtime_decoder_rebuilds(), 1);
}

/// Writes `groups` four-row groups of `(id, k)` with one page per row, where
/// `k` cycles 0..4 inside every group.
fn write_cycling_groups(path: &str, groups: i32) {
    let schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, 1),
        field("k", DataType::Int32, 2),
    ]));
    let properties = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_max_row_group_row_count(Some(4))
        .set_data_page_row_count_limit(1)
        .set_write_batch_size(1)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(path).unwrap(),
        Arc::clone(&schema),
        Some(properties),
    )
    .unwrap();
    for group in 0..groups {
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![
            Arc::new(Int32Array::from(
                (group * 4..group * 4 + 4).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
        ])
        .unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();
}

#[tokio::test]
async fn runtime_predicate_cost_does_not_grow_with_row_groups_per_boundary() {
    const GROUPS: i32 = 2000;
    let temp = TempDir::new().unwrap();
    let path = format!("{}/many-groups.parquet", temp.path().to_str().unwrap());
    write_cycling_groups(&path, GROUPS);
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "k", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap(),
    );
    let task = || scan_task(path.clone(), Arc::clone(&schema), None);
    // `k >= 2` keeps every group (each spans 0..=3) but removes two pages of
    // each one, the worst case for per-boundary page work.
    let predicate = Reference::new("k").greater_than_or_equal_to(Datum::int(2));

    // Stable generation from task open: no decoder rebuild at any boundary.
    let provider = Arc::new(FixedRuntimePredicate::new(predicate.clone()));
    let (stream, metrics) = start_runtime_scan(task(), Some(provider.clone()), true, true, 4);
    let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
    assert_eq!(ids(&batches).len(), GROUPS as usize * 2);
    assert_eq!(provider.snapshots(), 1);
    assert_eq!(metrics.runtime_decoder_rebuilds(), 0);

    // Each observed publication costs exactly one rebuild, however many
    // boundaries follow it.
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(task(), Some(provider.clone()), true, true, 4);
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    provider.publish(Some(predicate), 1);
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    // The first group was read before the publication.
    assert_eq!(ids(&batches).len(), 4 + (GROUPS as usize - 1) * 2);
    assert_eq!(metrics.runtime_predicate_refreshes(), 1);
    assert_eq!(metrics.runtime_decoder_rebuilds(), 1);
    assert_eq!(provider.snapshots(), 2);
}

#[tokio::test]
async fn runtime_predicate_file_statistics_that_prove_every_row_do_not_change_results() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "all_rows.parquet");
    let metrics = |nulls: Option<u64>| {
        crate::scan::FileScanTaskMetrics::builder()
            .with_record_count(Some(12))
            .with_value_counts(HashMap::new())
            .with_null_value_counts(nulls.map(|count| (1, count)).into_iter().collect())
            .with_nan_value_counts(HashMap::new())
            .with_lower_bounds(HashMap::from([(1, Datum::int(0))]))
            .with_upper_bounds(HashMap::from([(1, Datum::int(203))]))
            .build()
    };
    let predicates = [
        // Every row satisfies these, so the row filter is skipped for the file.
        (
            Reference::new("id").greater_than_or_equal_to(Datum::int(0)),
            all_ids(),
        ),
        (Reference::new("id").less_than(Datum::int(204)), all_ids()),
        (
            Reference::new("id")
                .greater_than(Datum::int(-1))
                .and(Reference::new("id").less_than_or_equal_to(Datum::int(203))),
            all_ids(),
        ),
        // Some rows fail these, so the filter still runs.
        (
            Reference::new("id").greater_than_or_equal_to(Datum::int(100)),
            vec![100, 101, 102, 103, 200, 201, 202, 203],
        ),
        (
            Reference::new("id")
                .greater_than_or_equal_to(Datum::int(0))
                .and(Reference::new("id").less_than(Datum::int(102))),
            vec![0, 1, 2, 3, 100, 101],
        ),
    ];
    // With and without a null count, which is required to skip the filter.
    for nulls in [Some(0), Some(1), None] {
        for (predicate, expected) in &predicates {
            let planned = predicate.clone().bind(iceberg_schema(), false).unwrap();
            let static_task = with_file_metrics(
                scan_task(path.clone(), iceberg_schema(), Some(planned)),
                metrics(nulls),
            );
            let (batches, _) = execute(static_task, None).await;
            assert_eq!(
                &ids(&batches),
                expected,
                "planned {predicate} with nulls {nulls:?}"
            );
            let task = with_file_metrics(
                scan_task(path.clone(), iceberg_schema(), None),
                metrics(nulls),
            );
            let provider = Arc::new(FixedRuntimePredicate::new(predicate.clone()));
            let (batches, _) = execute(task, Some(provider)).await;
            assert_eq!(&ids(&batches), expected, "{predicate} with nulls {nulls:?}");
        }
    }
}

#[tokio::test]
async fn runtime_predicate_negative_predicates_preserve_null_filtering_with_file_metrics() {
    let temp = TempDir::new().unwrap();
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "value", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap(),
    );
    for values in [vec![None, Some(10), None, Some(11)], vec![None; 4]] {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            field("id", DataType::Int32, 1),
            Field::new("value", DataType::Int32, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let path = temp
            .path()
            .join(format!("nulls-{}.parquet", values.iter().flatten().count()));
        let batch = RecordBatch::try_new(Arc::clone(&arrow_schema), vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int32Array::from(values.clone())),
        ])
        .unwrap();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), arrow_schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let metrics = crate::scan::FileScanTaskMetrics::builder()
            .with_record_count(Some(4))
            .with_value_counts(HashMap::from([(1, 4), (2, 4)]))
            .with_null_value_counts(HashMap::from([
                (1, 0),
                (
                    2,
                    values.iter().filter(|value| value.is_none()).count() as u64,
                ),
            ]))
            .with_nan_value_counts(HashMap::new())
            .with_lower_bounds(
                std::iter::once((1, Datum::int(1)))
                    .chain(
                        values
                            .iter()
                            .flatten()
                            .min()
                            .map(|&value| (2, Datum::int(value))),
                    )
                    .collect(),
            )
            .with_upper_bounds(
                std::iter::once((1, Datum::int(4)))
                    .chain(
                        values
                            .iter()
                            .flatten()
                            .max()
                            .map(|&value| (2, Datum::int(value))),
                    )
                    .collect(),
            )
            .build();
        let expected: Vec<_> = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.map(|_| index as i32 + 1))
            .collect();
        let negative = Reference::new("value").not_equal_to(Datum::int(0));
        let predicates = [
            negative.clone(),
            Reference::new("value").is_not_in([Datum::int(0), Datum::int(1)]),
            negative.or(Reference::new("id").less_than(Datum::int(0))),
        ];
        for predicate in predicates {
            let task = scan_task(path.to_str().unwrap().to_string(), schema.clone(), None);
            let static_task = scan_task(
                path.to_str().unwrap().to_string(),
                schema.clone(),
                Some(predicate.clone().bind(schema.clone(), false).unwrap()),
            );
            for static_task in [
                static_task.clone(),
                with_file_metrics(static_task, metrics.clone()),
            ] {
                let (static_batches, _) = execute(static_task, None).await;
                assert_eq!(ids(&static_batches), expected, "static {predicate}");
            }
            for task in [task.clone(), with_file_metrics(task, metrics.clone())] {
                let (batches, _) = execute(
                    task,
                    Some(Arc::new(FixedRuntimePredicate::new(predicate.clone()))),
                )
                .await;
                assert_eq!(ids(&batches), expected, "runtime {predicate}");
            }
        }
    }
}

#[tokio::test]
async fn runtime_predicate_float_comparisons_fail_open_before_statistics_pruning() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("float-semantics.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, 1),
        field("value", DataType::Float64, 2),
    ]));
    let values = vec![-0.0, 0.0, f64::NAN, -f64::NAN];
    let batch = RecordBatch::try_new(Arc::clone(&arrow_schema), vec![
        Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
        Arc::new(Float64Array::from(values)),
    ])
    .unwrap();
    let mut writer =
        ArrowWriter::try_new(File::create(&path).unwrap(), arrow_schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "value", Type::Primitive(PrimitiveType::Double)).into(),
            ])
            .build()
            .unwrap(),
    );
    let planned = Reference::new("id")
        .greater_than(Datum::int(0))
        .bind(schema.clone(), false)
        .unwrap();
    let task = scan_task(path.to_str().unwrap().to_string(), schema, Some(planned));
    let metrics = crate::scan::FileScanTaskMetrics::builder()
        .with_record_count(Some(4))
        .with_value_counts(HashMap::from([(2, 4)]))
        .with_null_value_counts(HashMap::from([(2, 0)]))
        .with_nan_value_counts(HashMap::from([(2, 2)]))
        .with_lower_bounds(HashMap::from([(2, Datum::double(-0.0))]))
        .with_upper_bounds(HashMap::from([(2, Datum::double(0.0))]))
        .build();
    for predicate in [
        Reference::new("value").equal_to(Datum::double(0.0)),
        Reference::new("value").not_equal_to(Datum::double(f64::NAN)),
        Reference::new("value").greater_than(Datum::double(0.0)),
        Reference::new("value").less_than(Datum::double(0.0)),
        Reference::new("value").is_in([Datum::double(0.0), Datum::double(1.0)]),
        Reference::new("value").is_not_in([Datum::double(0.0), Datum::double(1.0)]),
    ] {
        for task in [
            task.clone(),
            with_file_metrics(task.clone(), metrics.clone()),
        ] {
            let (batches, scan_metrics) = execute(
                task,
                Some(Arc::new(FixedRuntimePredicate::new(predicate.clone()))),
            )
            .await;
            assert_eq!(ids(&batches), vec![1, 2, 3], "{predicate}");
            assert_eq!(scan_metrics.runtime_predicate_tasks(), 0);
        }
    }
    // Unary NaN predicates do not compare NaN payloads or signed zero and
    // remain usable on floating-point columns.
    let (batches, _) = execute(
        task,
        Some(Arc::new(FixedRuntimePredicate::new(
            Reference::new("value").is_nan(),
        ))),
    )
    .await;
    assert_eq!(ids(&batches), vec![2, 3]);
}

/// Delegates to a fixed predicate and asks for the largest values first.
struct LargestFirst(FixedRuntimePredicate);

impl RuntimePredicateProvider for LargestFirst {
    fn generation(&self) -> u64 {
        self.0.generation()
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
        self.0.snapshot()
    }

    fn largest_first_column(&self) -> Option<String> {
        Some("id".into())
    }
}

#[tokio::test]
async fn runtime_predicate_preferring_largest_values_reads_row_groups_by_descending_maximum() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "largest_first.parquet");
    let predicate = || Reference::new("id").greater_than_or_equal_to(Datum::int(1));

    let task = scan_task(path.clone(), iceberg_schema(), None);
    let (batches, _) = execute(
        task,
        Some(Arc::new(LargestFirst(FixedRuntimePredicate::new(
            predicate(),
        )))),
    )
    .await;
    assert_eq!(ids(&batches), vec![
        200, 201, 202, 203, 100, 101, 102, 103, 1, 2, 3
    ]);

    // Without the preference the file order stays.
    let task = scan_task(path, iceberg_schema(), None);
    let (batches, _) = execute(
        task,
        Some(Arc::new(FixedRuntimePredicate::new(predicate()))),
    )
    .await;
    assert_eq!(ids(&batches), vec![
        1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203
    ]);
}

/// A changing provider that asks for the largest `id` values first.
struct LargestFirstChanging(Arc<ChangingRuntimePredicate>);

impl RuntimePredicateProvider for LargestFirstChanging {
    fn generation(&self) -> u64 {
        self.0.generation()
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
        self.0.snapshot()
    }

    fn largest_first_column(&self) -> Option<String> {
        Some("id".into())
    }
}

#[tokio::test]
async fn runtime_predicate_live_refresh_keeps_page_selections_with_largest_first_order() {
    let temp = TempDir::new().unwrap();
    // Two rows per page, so a refresh selects pages inside each remaining group.
    let path = write_row_group_file_with_page_size(
        temp.path().to_str().unwrap(),
        "largest_first_pages.parquet",
        &[0, 100, 200],
        2,
    );
    let changing = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, _) = start_runtime_scan(
        scan_task(path, iceberg_schema(), None),
        Some(Arc::new(LargestFirstChanging(Arc::clone(&changing)))),
        true,
        true,
        4,
    );
    let first = stream.try_next().await.unwrap().unwrap();
    assert_eq!(ids(std::slice::from_ref(&first)), vec![200, 201, 202, 203]);
    // The refresh plans pages for the remaining groups, read as 100 then 0.
    changing.publish(
        Some(Reference::new("id").greater_than_or_equal_to(Datum::int(2))),
        1,
    );
    let rest: Vec<RecordBatch> = stream.try_collect().await.unwrap();
    assert_eq!(ids(&rest), vec![100, 101, 102, 103, 2, 3]);
}

#[tokio::test]
async fn runtime_predicate_advisory_with_row_group_pruning_applies_row_filter() {
    let temp = TempDir::new().unwrap();
    // 2 row groups: [0, 1, 2, 3] and [100, 101, 102, 103].
    let path = write_row_group_file_with_page_size(
        temp.path().to_str().unwrap(),
        "rg_prune.parquet",
        &[0, 100],
        1024,
    );
    let task = scan_task(path, iceberg_schema(), None);
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").equal_to(Datum::int(102)),
    ));
    let (batches, metrics) = execute(task, Some(provider)).await;
    assert_eq!(ids(&batches), vec![102]);
    assert_eq!(metrics.runtime_row_groups_pruned(), 1);
}

#[tokio::test]
async fn runtime_predicate_advisory_with_page_pruning_applies_row_filter() {
    let temp = TempDir::new().unwrap();
    // 1 row group with 2 pages: page 0 has [0, 1], page 1 has [2, 3].
    let path = write_row_group_file_with_page_size(
        temp.path().to_str().unwrap(),
        "page_prune.parquet",
        &[0],
        2,
    );
    let task = scan_task(path, iceberg_schema(), None);
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").equal_to(Datum::int(3)),
    ));
    let (batches, metrics) = execute(task, Some(provider)).await;
    assert_eq!(ids(&batches), vec![3]);
    assert_eq!(metrics.runtime_row_groups_pruned(), 0);
}

#[tokio::test]
async fn planned_predicate_without_pruning_still_filters_rows() {
    let temp = TempDir::new().unwrap();
    // 1 row group with 4 rows (ids 0, 1, 2, 3), 1 page.
    let path = write_row_group_file_with_page_size(
        temp.path().to_str().unwrap(),
        "planned_no_prune.parquet",
        &[0],
        1024,
    );
    let schema = iceberg_schema();
    let planned = Reference::new("id")
        .equal_to(Datum::int(2))
        .bind(Arc::clone(&schema), false)
        .unwrap();
    let task = scan_task(path, schema, Some(planned));
    let (batches, _) = execute(task, None).await;
    assert_eq!(ids(&batches), vec![2]);
}

#[test]
fn concurrent_publishers_keep_snapshot_pair_and_generation() {
    use std::sync::mpsc::channel;
    use std::thread;

    use super::runtime_predicate::RuntimePredicates;

    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let cache = Arc::new(RuntimePredicates::new(provider.clone()));
    let predicate =
        |generation| Reference::new("id").greater_than_or_equal_to(Datum::int(generation));
    let (entered_tx, entered_rx) = channel();
    let (release_tx, release_rx) = channel();
    let first_provider = provider.clone();
    let first = thread::spawn(move || {
        first_provider.publish_with_gate(Some(predicate(1)), 1, || {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
    });
    entered_rx.recv().unwrap();
    assert_eq!(provider.generation(), 0);
    assert!(provider.publication.try_lock().is_err());

    let (started_tx, started_rx) = channel();
    let (second_tx, second_rx) = channel();
    let (finish_tx, finish_rx) = channel();
    let second_provider = provider.clone();
    let second = thread::spawn(move || {
        started_tx.send(()).unwrap();
        second_provider.publish_with_gate(Some(predicate(2)), 2, || {
            second_tx.send(()).unwrap();
            finish_rx.recv().unwrap();
        });
    });
    started_rx.recv().unwrap();
    let observer_provider = provider.clone();
    let observer_cache = cache.clone();
    let observer = thread::spawn(move || {
        let snapshot = observer_provider.snapshot().unwrap();
        let generation = snapshot.generation();
        assert!([1, 2].contains(&generation));
        assert_eq!(
            snapshot.into_predicate(),
            Some(predicate(generation as i32)),
        );
        observer_cache.current(&iceberg_schema(), false, "publisher")
    });
    release_tx.send(()).unwrap();
    second_rx.recv().unwrap();
    assert_eq!(provider.generation(), 1);
    assert!(provider.publication.try_lock().is_err());
    finish_tx.send(()).unwrap();
    first.join().unwrap();
    second.join().unwrap();
    let _ = observer.join().unwrap();
    assert_eq!(provider.generation(), 2);
    let snapshot = provider.snapshot().unwrap();
    assert_eq!(snapshot.generation(), 2);
    assert_eq!(snapshot.into_predicate(), Some(predicate(2)));
    let schema = iceberg_schema();
    let latest = cache.current(&schema, false, "publisher").unwrap();
    assert_eq!(*latest, predicate(2).bind(schema.clone(), false).unwrap(),);
    assert!(Arc::ptr_eq(
        &latest,
        &cache.current(&schema, false, "publisher").unwrap(),
    ));
}

fn refresh_metric_counts(metrics: &ScanMetrics) -> [u64; 6] {
    [
        metrics.runtime_predicate_tasks(),
        metrics.runtime_file_tasks_pruned(),
        metrics.runtime_predicate_refreshes(),
        metrics.runtime_decoder_rebuilds(),
        metrics.runtime_row_groups_pruned_live(),
        metrics.runtime_row_groups_pruned(),
    ]
}

async fn check_rebuild_failure(
    stage: super::runtime_stream::test_support::Stage,
    expected_error: &str,
) {
    use super::runtime_stream::test_support::Probe;

    for initially_active in [false, true] {
        let temp = TempDir::new().unwrap();
        let path =
            write_three_row_group_file(temp.path().to_str().unwrap(), "rebuild-failure.parquet");
        let probe = Probe::register(&path);
        let initial =
            initially_active.then(|| Reference::new("id").greater_than_or_equal_to(Datum::int(0)));
        let provider = Arc::new(ChangingRuntimePredicate::new(initial, 0));
        let (mut stream, metrics) = start_runtime_scan(
            scan_task(path, iceberg_schema(), None),
            Some(provider.clone()),
            true,
            true,
            4,
        );
        let first = stream.try_next().await.unwrap().unwrap();
        assert_eq!(ids(&[first]), vec![0, 1, 2, 3]);
        let before = refresh_metric_counts(&metrics);
        probe.fail_at(stage);
        provider.publish(
            Some(Reference::new("id").greater_than_or_equal_to(Datum::int(200))),
            1,
        );
        let error = stream.try_next().await.unwrap_err();
        assert!(error.to_string().contains(expected_error), "{error}");
        assert_eq!(refresh_metric_counts(&metrics), before);
        assert_eq!(&probe.counts()[1..], &[1, 1, 1, 1]);
        assert!(stream.try_next().await.unwrap().is_none());
        assert_eq!(refresh_metric_counts(&metrics), before);
    }
}

#[tokio::test]
async fn live_refresh_rebuild_into_builder_failure_metrics() {
    check_rebuild_failure(
        super::runtime_stream::test_support::Stage::IntoBuilder,
        "into_builder called on a finished decoder",
    )
    .await;
}

#[tokio::test]
async fn live_refresh_rebuild_build_failure_metrics() {
    check_rebuild_failure(
        super::runtime_stream::test_support::Stage::Build,
        "with_row_group_selections cannot be combined",
    )
    .await;
}

const LINEAGE_PAGE_ROWS: i32 = 1024;
const LINEAGE_GROUP_ROWS: i32 = 4 * LINEAGE_PAGE_ROWS;
const LINEAGE_FIRST_ROW_ID: i64 = 1_000_000;

fn lineage_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(3, "k", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap(),
    )
}

fn write_lineage_groups(path: &str, physical_row_id: bool) {
    use parquet::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};

    use crate::metadata_columns::RESERVED_FIELD_ID_ROW_ID;

    let mut fields = vec![
        field("id", DataType::Int32, 1),
        field("payload", DataType::Utf8, 2),
        field("k", DataType::Int32, 3),
    ];
    if physical_row_id {
        fields
            .push(field("_row_id", DataType::Int64, RESERVED_FIELD_ID_ROW_ID).with_nullable(true));
    }
    let schema = Arc::new(ArrowSchema::new(fields));
    let properties = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_max_row_group_row_count(Some(LINEAGE_GROUP_ROWS as usize))
        .set_data_page_row_count_limit(LINEAGE_PAGE_ROWS as usize)
        .set_write_batch_size(LINEAGE_PAGE_ROWS as usize)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(path).unwrap(),
        schema.clone(),
        Some(properties),
    )
    .unwrap();
    for group in 0..5 {
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter_values(
                (0..LINEAGE_GROUP_ROWS).map(|row| group * 10000 + row),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..LINEAGE_GROUP_ROWS).map(|row| format!("{group}-{row}-{}", "x".repeat(64))),
            )),
            Arc::new(Int32Array::from_iter_values(
                (0..LINEAGE_GROUP_ROWS).map(|row| row / LINEAGE_PAGE_ROWS),
            )),
        ];
        if physical_row_id {
            columns.push(Arc::new(Int64Array::from_iter(
                (0..LINEAGE_GROUP_ROWS).map(|row| {
                    let pos = i64::from(group * LINEAGE_GROUP_ROWS + row);
                    (pos % 3 == 0).then_some(9_000_000 + pos)
                }),
            )));
        }
        writer
            .write(&RecordBatch::try_new(schema.clone(), columns).unwrap())
            .unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();
    let metadata = ParquetMetaDataReader::new()
        .with_page_index_policy(PageIndexPolicy::Required)
        .parse_and_finish(&File::open(path).unwrap())
        .unwrap();
    assert_eq!(metadata.num_row_groups(), 5);
    for group in metadata.offset_index().unwrap() {
        assert_eq!(group[0].page_locations().len(), 4);
        assert_eq!(group[2].page_locations().len(), 4);
    }
}

fn lineage_task(path: &str, planned: Option<Predicate>) -> FileScanTask {
    use crate::metadata_columns::{RESERVED_FIELD_ID_POS, RESERVED_FIELD_ID_ROW_ID};

    let schema = lineage_schema();
    FileScanTask::builder()
        .with_file_size_in_bytes(std::fs::metadata(path).unwrap().len())
        .with_start(0)
        .with_length(0)
        .with_data_file_path(path.to_owned())
        .with_data_file_format(DataFileFormat::Parquet)
        .with_schema(schema.clone())
        .with_project_field_ids(vec![1, 2, RESERVED_FIELD_ID_POS, RESERVED_FIELD_ID_ROW_ID])
        .with_first_row_id(Some(LINEAGE_FIRST_ROW_ID))
        .with_predicate(planned.map(|predicate| predicate.bind(schema, false).unwrap()))
        .with_case_sensitive(false)
        .build()
        .unwrap()
}

fn patch_task(task: FileScanTask, patch: serde_json::Value) -> FileScanTask {
    let mut value = serde_json::to_value(task).unwrap();
    for (key, field) in patch.as_object().unwrap() {
        value[key] = field.clone();
    }
    serde_json::from_value(value).unwrap()
}

fn lineage_rows(batches: &[RecordBatch]) -> Vec<(i32, i64, i64)> {
    let mut rows = Vec::new();
    for batch in batches {
        let id = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let pos = batch
            .column_by_name("_pos")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let row_id = batch
            .column_by_name("_row_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(row_id.null_count(), 0);
        for row in 0..batch.num_rows() {
            rows.push((id.value(row), pos.value(row), row_id.value(row)));
        }
    }
    rows
}

fn expected_lineage(id: i32, physical: bool) -> (i32, i64, i64) {
    let pos = i64::from((id / 10000) * LINEAGE_GROUP_ROWS + id % 10000);
    let row_id = if physical && pos % 3 == 0 {
        9_000_000 + pos
    } else {
        LINEAGE_FIRST_ROW_ID + pos
    };
    (id, pos, row_id)
}

fn delete_task(path: String, position: bool) -> FileScanTaskDeleteFile {
    FileScanTaskDeleteFile::builder()
        .with_file_size_in_bytes(std::fs::metadata(&path).unwrap().len())
        .with_file_path(path)
        .with_file_type(if position {
            DataContentType::PositionDeletes
        } else {
            DataContentType::EqualityDeletes
        })
        .with_file_format(DataFileFormat::Parquet)
        .with_partition_spec_id(0)
        .with_equality_ids((!position).then_some(vec![1]))
        .build()
        .unwrap()
}

fn lineage_deletes(path: &str, dir: &str) -> Vec<FileScanTaskDeleteFile> {
    let position_path = format!("{dir}/lineage-position.parquet");
    let equality_path = format!("{dir}/lineage-equality.parquet");
    let mut positions: Vec<i64> = (LINEAGE_GROUP_ROWS..2 * LINEAGE_GROUP_ROWS)
        .map(i64::from)
        .collect();
    for group in [0, 2, 3, 4] {
        positions.push(i64::from(
            group * LINEAGE_GROUP_ROWS + 2 * LINEAGE_PAGE_ROWS + 1,
        ));
    }
    positions.sort_unstable();
    write_delete(
        &position_path,
        vec![
            field("file_path", DataType::Utf8, 2_147_483_546),
            field("pos", DataType::Int64, 2_147_483_545),
        ],
        vec![
            Arc::new(StringArray::from(vec![path; positions.len()])),
            Arc::new(Int64Array::from(positions)),
        ],
    );
    write_delete(&equality_path, vec![field("id", DataType::Int32, 1)], vec![
        Arc::new(Int32Array::from_iter_values(
            (0..5).map(|group| group * 10000 + 2 * LINEAGE_PAGE_ROWS + 2),
        )),
    ]);
    vec![
        delete_task(position_path, true),
        delete_task(equality_path, false),
    ]
}

#[tokio::test]
async fn live_refresh_preserves_deletes_and_absolute_lineage_differential() {
    for physical in [false, true] {
        let temp = TempDir::new().unwrap();
        let dir = temp.path().to_str().unwrap();
        let path = format!("{dir}/lineage.parquet");
        write_lineage_groups(&path, physical);
        let planned = Reference::new("k").not_equal_to(Datum::int(3));
        let task = patch_task(
            lineage_task(&path, Some(planned)),
            serde_json::json!({
                "deletes": lineage_deletes(&path, dir),
            }),
        );
        let (baseline, _) = execute(task.clone(), None).await;
        let baseline_rows = lineage_rows(&baseline);
        let expected_baseline: Vec<_> = (0..5)
            .filter(|group| *group != 1)
            .flat_map(|group| {
                (0..3 * LINEAGE_PAGE_ROWS)
                    .filter(|row| {
                        ![2 * LINEAGE_PAGE_ROWS + 1, 2 * LINEAGE_PAGE_ROWS + 2].contains(row)
                    })
                    .map(move |row| expected_lineage(group * 10000 + row, physical))
            })
            .collect();
        assert_eq!(baseline_rows, expected_baseline);
        for largest_first in [false, true] {
            let changing = Arc::new(ChangingRuntimePredicate::new(None, 0));
            let provider: Arc<dyn RuntimePredicateProvider> = if largest_first {
                Arc::new(LargestFirstChanging(changing.clone()))
            } else {
                changing.clone()
            };
            let (mut stream, metrics) =
                start_runtime_scan(task.clone(), Some(provider), true, true, 128);
            let first = stream.try_next().await.unwrap().unwrap();
            let first_group = if largest_first { 4 } else { 0 };
            assert_eq!(
                lineage_rows(std::slice::from_ref(&first)),
                (0..128)
                    .map(|row| expected_lineage(first_group * 10000 + row, physical,))
                    .collect::<Vec<_>>(),
            );
            changing.publish(
                Some(
                    Reference::new("k")
                        .greater_than_or_equal_to(Datum::int(2))
                        .and(
                            Reference::new("id")
                                .less_than(Datum::int(20000))
                                .or(Reference::new("id")
                                    .greater_than_or_equal_to(Datum::int(30000))),
                        ),
                ),
                1,
            );
            let mut batches = vec![first];
            batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
            let mut expected: Vec<_> = baseline_rows
                .iter()
                .copied()
                .filter(|(id, _, _)| {
                    id / 10000 == first_group
                        || (id % 10000 >= 2 * LINEAGE_PAGE_ROWS && !(20000..30000).contains(id))
                })
                .collect();
            if largest_first {
                expected.sort_by_key(|(id, _, _)| (std::cmp::Reverse(id / 10000), id % 10000));
            }
            assert_eq!(lineage_rows(&batches), expected);
            assert_eq!(metrics.runtime_predicate_refreshes(), 1);
            assert_eq!(metrics.runtime_decoder_rebuilds(), 1);
            assert_eq!(metrics.runtime_row_groups_pruned_live(), 1);
            assert_eq!(changing.snapshots(), 2);
        }
    }
}

fn start_uncoalesced_scan(
    task: FileScanTask,
    provider: Option<Arc<dyn RuntimePredicateProvider>>,
    pages: bool,
) -> (ArrowRecordBatchStream, ScanMetrics) {
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_range_coalesce_bytes(0)
        .with_row_selection_enabled(pages)
        .with_batch_size(LINEAGE_GROUP_ROWS as usize);
    if let Some(provider) = provider {
        builder = builder.with_runtime_predicate_provider(provider);
    }
    let tasks = Box::pin(futures::stream::iter([Ok(task)])) as FileScanTaskStream;
    let scan = builder.build().read(tasks).unwrap();
    let metrics = scan.metrics().clone();
    (scan.stream(), metrics)
}

fn lineage_all_match_metrics() -> crate::scan::FileScanTaskMetrics {
    crate::scan::FileScanTaskMetrics::builder()
        .with_record_count(Some(5 * LINEAGE_GROUP_ROWS as u64))
        .with_value_counts(HashMap::from([(3, 5 * LINEAGE_GROUP_ROWS as u64)]))
        .with_null_value_counts(HashMap::from([(3, 0)]))
        .with_lower_bounds(HashMap::from([(3, Datum::int(0))]))
        .with_upper_bounds(HashMap::from([(3, Datum::int(3))]))
        .build()
}

fn lineage_middle_split(task: FileScanTask) -> FileScanTask {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let parquet = SerializedFileReader::new(File::open(task.data_file_path()).unwrap()).unwrap();
    let metadata = parquet.metadata();
    let start = 4 + metadata.row_group(0).compressed_size() as u64;
    let length = metadata.row_group(1).compressed_size() as u64;
    patch_task(
        task,
        serde_json::json!({
            "start": start, "length": length,
        }),
    )
}

async fn check_runtime_page_lineage(split: bool) {
    for physical in [false, true] {
        let temp = TempDir::new().unwrap();
        let path = format!("{}/pages.parquet", temp.path().display());
        write_lineage_groups(&path, physical);
        let mut task = lineage_task(&path, None);
        if split {
            task = lineage_middle_split(task);
        }
        let predicate = Reference::new("k")
            .equal_to(Datum::int(0))
            .or(Reference::new("k").equal_to(Datum::int(3)));
        let provider = Arc::new(FixedRuntimePredicate::new(predicate));
        let (reference, reference_metrics) =
            start_uncoalesced_scan(task.clone(), Some(provider.clone()), false);
        let reference: Vec<RecordBatch> = reference.try_collect().await.unwrap();
        let (pruned, metrics) = start_uncoalesced_scan(task.clone(), Some(provider), true);
        let pruned: Vec<RecordBatch> = pruned.try_collect().await.unwrap();
        let groups: Vec<i32> = if split { vec![1] } else { (0..5).collect() };
        let expected: Vec<_> = groups
            .into_iter()
            .flat_map(|group| {
                (0..LINEAGE_GROUP_ROWS)
                    .filter(|row| *row < LINEAGE_PAGE_ROWS || *row >= 3 * LINEAGE_PAGE_ROWS)
                    .map(move |row| expected_lineage(group * 10000 + row, physical))
            })
            .collect();
        assert_eq!(lineage_rows(&reference), expected);
        assert_eq!(lineage_rows(&pruned), expected);
        assert!(metrics.bytes_read() < reference_metrics.bytes_read());
        if split {
            let (whole, whole_metrics) =
                start_uncoalesced_scan(lineage_task(&path, None), None, false);
            let _: Vec<RecordBatch> = whole.try_collect().await.unwrap();
            assert!(metrics.bytes_read() < whole_metrics.bytes_read());
        }
    }
}

#[tokio::test]
async fn runtime_page_pruning_preserves_pos_and_row_id() {
    check_runtime_page_lineage(false).await;
}

#[tokio::test]
async fn runtime_page_pruning_within_byte_range_split() {
    check_runtime_page_lineage(true).await;
}

async fn check_all_match_lineage(split: bool) {
    for physical in [false, true] {
        let temp = TempDir::new().unwrap();
        let path = format!("{}/all-match.parquet", temp.path().display());
        write_lineage_groups(&path, physical);
        let mut task = lineage_task(
            &path,
            Some(Reference::new("k").greater_than_or_equal_to(Datum::int(0))),
        );
        if split {
            task = lineage_middle_split(task);
        }
        let (reference, reference_metrics) = start_uncoalesced_scan(task.clone(), None, true);
        let reference: Vec<RecordBatch> = reference.try_collect().await.unwrap();
        let (shortcut, shortcut_metrics) = start_uncoalesced_scan(
            with_file_metrics(task, lineage_all_match_metrics()),
            None,
            true,
        );
        let shortcut: Vec<RecordBatch> = shortcut.try_collect().await.unwrap();
        let groups: Vec<i32> = if split { vec![1] } else { (0..5).collect() };
        let expected: Vec<_> = groups
            .into_iter()
            .flat_map(|group| {
                (0..LINEAGE_GROUP_ROWS)
                    .map(move |row| expected_lineage(group * 10000 + row, physical))
            })
            .collect();
        assert_eq!(lineage_rows(&reference), expected);
        assert_eq!(lineage_rows(&shortcut), expected);
        assert_eq!(shortcut, reference);
        assert!(shortcut_metrics.bytes_read() < reference_metrics.bytes_read(),);
    }
}

#[tokio::test]
async fn file_always_matches_preserves_pos_and_row_id() {
    check_all_match_lineage(false).await;
}

#[tokio::test]
async fn all_match_shortcut_on_byte_range_split() {
    check_all_match_lineage(true).await;
}

#[tokio::test]
async fn unchanged_generation_checks_do_not_replan() {
    use super::runtime_stream::test_support::Probe;

    for disabled in [false, true] {
        let temp = TempDir::new().unwrap();
        let dir = temp.path().to_str().unwrap();
        let path = format!("{dir}/unchanged.parquet");
        write_lineage_groups(&path, false);
        let probe = Probe::register(&path);
        let task = patch_task(
            lineage_task(&path, None),
            serde_json::json!({
                "deletes": lineage_deletes(&path, dir),
            }),
        );
        let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
        let (mut stream, metrics) =
            start_runtime_scan(task, Some(provider.clone()), true, true, 128);
        let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
        assert_eq!(&probe.counts()[1..], &[0, 0, 0, 0]);
        provider.publish(
            Some(Reference::new("k").greater_than_or_equal_to(Datum::int(2))),
            1,
        );
        // Drain the active group; none of its batches samples again.
        while ids(&batches).last().unwrap() < &10000 {
            let batch = stream.try_next().await.unwrap().unwrap();
            let next_group = ids(std::slice::from_ref(&batch))[0] >= 10000;
            batches.push(batch);
            if next_group {
                break;
            }
            assert_eq!(&probe.counts()[1..], &[0, 0, 0, 0]);
        }
        assert_eq!(ids(&batches).last().unwrap() / 10000, 2);
        let after_refresh = probe.counts();
        assert_eq!(&after_refresh[1..], &[1, 1, 1, 1]);
        assert_eq!(provider.snapshots(), 2);
        let batch = stream.try_next().await.unwrap().unwrap();
        batches.push(batch);
        assert_eq!(probe.counts(), after_refresh);
        if disabled {
            probe.disable_runtime();
            provider.publish(Some(Predicate::AlwaysFalse), 2);
        }
        batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
        let expected: Vec<_> = (0..5)
            .filter(|group| *group != 1)
            .flat_map(|group| {
                (0..LINEAGE_GROUP_ROWS)
                    .filter(move |row| {
                        (group == 0 || *row >= 2 * LINEAGE_PAGE_ROWS)
                            && ![2 * LINEAGE_PAGE_ROWS + 1, 2 * LINEAGE_PAGE_ROWS + 2].contains(row)
                    })
                    .map(move |row| expected_lineage(group * 10000 + row, false))
            })
            .collect();
        assert_eq!(lineage_rows(&batches), expected);
        let final_counts = probe.counts();
        assert_eq!(&final_counts[1..], &[1, 1, 1, 1]);
        if disabled {
            assert_eq!(final_counts, after_refresh);
        } else {
            assert!(final_counts[0] > after_refresh[0]);
        }
        assert_eq!(provider.snapshots(), 2);
        assert_eq!(metrics.runtime_decoder_rebuilds(), 1);
    }
}

struct OrderingHint {
    column: Option<&'static str>,
}

impl RuntimePredicateProvider for OrderingHint {
    fn generation(&self) -> u64 {
        0
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
        Ok(RuntimePredicateSnapshot::new(None, 0))
    }

    fn largest_first_column(&self) -> Option<String> {
        self.column.map(str::to_owned)
    }
}

fn write_ordering_fallback_file(path: &str, field_ids: bool, statistics: bool) -> SchemaRef {
    use parquet::file::properties::EnabledStatistics;

    let mut fields = vec![
        field("id", DataType::Int32, 1),
        field("payload", DataType::Utf8, 2),
        field("d", DataType::Decimal128(30, 2), 3),
        field("f", DataType::Float64, 4),
    ];
    if !field_ids {
        for field in &mut fields {
            *field = field.clone().with_metadata(HashMap::new());
        }
    }
    let schema = Arc::new(ArrowSchema::new(fields));
    let properties = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .set_statistics_enabled(if statistics {
            EnabledStatistics::Page
        } else {
            EnabledStatistics::None
        })
        .set_max_row_group_row_count(Some(4))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(path).unwrap(),
        schema.clone(),
        Some(properties),
    )
    .unwrap();
    for base in [0, 100, 200] {
        let values: Vec<i32> = (base..base + 4).collect();
        let decimal =
            Decimal128Array::from_iter_values(values.iter().map(|value| i128::from(*value)))
                .with_precision_and_scale(30, 2)
                .unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(Int32Array::from(values.clone())),
            Arc::new(StringArray::from_iter_values(
                values.iter().map(|value| format!("{value:03}")),
            )),
            Arc::new(decimal),
            Arc::new(Float64Array::from_iter_values(
                values.iter().map(|value| f64::from(*value)),
            )),
        ])
        .unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(
                    3,
                    "d",
                    Type::Primitive(PrimitiveType::Decimal {
                        precision: 30,
                        scale: 2,
                    }),
                )
                .into(),
                NestedField::required(4, "f", Type::Primitive(PrimitiveType::Double)).into(),
            ])
            .build()
            .unwrap(),
    )
}

#[tokio::test]
async fn largest_first_fallback_contract() {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    for (hint, field_ids, statistics, row_groups) in [
        (Some("payload"), true, true, true),
        (Some("d"), true, true, true),
        (Some("f"), true, true, true),
        (Some("missing"), true, true, true),
        (None, true, true, true),
        (Some("id"), true, false, true),
        (Some("id"), false, true, true),
        (Some("id"), true, true, false),
    ] {
        let temp = TempDir::new().unwrap();
        let path = format!("{}/fallback.parquet", temp.path().display());
        let schema = write_ordering_fallback_file(&path, field_ids, statistics);
        let parquet = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
        assert_eq!(
            parquet
                .metadata()
                .file_metadata()
                .schema_descr()
                .column(2)
                .physical_type(),
            parquet::basic::Type::FIXED_LEN_BYTE_ARRAY,
        );
        if !statistics {
            assert!(
                parquet
                    .metadata()
                    .row_group(0)
                    .column(0)
                    .statistics()
                    .is_none()
            );
        }
        let task = scan_task(path, schema, None);
        let (reference, _) = start_runtime_scan(task.clone(), None, true, row_groups, 2);
        let reference: Vec<RecordBatch> = reference.try_collect().await.unwrap();
        let (hinted, metrics) = start_runtime_scan(
            task,
            Some(Arc::new(OrderingHint { column: hint })),
            true,
            row_groups,
            2,
        );
        let hinted: Vec<RecordBatch> = hinted.try_collect().await.unwrap();
        assert_eq!(ids(&hinted), all_ids());
        assert_eq!(hinted, reference);
        assert_eq!(metrics.runtime_decoder_rebuilds(), 0);
    }
}

#[tokio::test]
#[ignore = "timing test run explicitly by fork CI"]
async fn runtime_performance_boundary_publications() {
    use std::time::Instant;

    let temp = TempDir::new().unwrap();
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "k", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap(),
    );
    for groups in [2000, 4000] {
        let path = temp.path().join(format!("timing-{groups}.parquet"));
        let path = path.to_str().unwrap();
        write_cycling_groups(path, groups);
        for mode in [
            "no-provider",
            "none",
            "stable",
            "one",
            "churn",
            "tightening",
        ] {
            let mut samples = Vec::new();
            for _ in 0..3 {
                let predicate = Reference::new("k").greater_than_or_equal_to(Datum::int(2));
                let initial =
                    matches!(mode, "stable" | "churn" | "tightening").then_some(predicate.clone());
                let provider = Arc::new(ChangingRuntimePredicate::new(initial, 0));
                let runtime = (mode != "no-provider")
                    .then_some(provider.clone() as Arc<dyn RuntimePredicateProvider>);
                let task = scan_task(path.to_string(), schema.clone(), None);
                let started = Instant::now();
                let (mut stream, metrics) = start_runtime_scan(task, runtime, true, true, 4);
                let mut rows = 0;
                let mut boundaries = 0;
                while let Some(batch) = stream.try_next().await.unwrap() {
                    rows += batch.num_rows();
                    boundaries += 1;
                    if boundaries < groups
                        && (mode == "churn"
                            || mode == "tightening"
                            || (mode == "one" && boundaries == 1))
                    {
                        let published = if mode == "tightening" {
                            predicate.clone().and(
                                Reference::new("id")
                                    .greater_than_or_equal_to(Datum::int(boundaries * 4)),
                            )
                        } else {
                            predicate.clone()
                        };
                        provider.publish(Some(published), boundaries as u64);
                    }
                }
                samples.push(started.elapsed().as_secs_f64() * 1000.0);
                let expected_rows = match mode {
                    "no-provider" | "none" => groups as usize * 4,
                    "one" => groups as usize * 2 + 2,
                    _ => groups as usize * 2,
                };
                assert_eq!(rows, expected_rows);
                assert_eq!(boundaries, groups);
                println!(
                    "R3 boundary groups={groups} mode={mode} ms={:.3} rebuilds={} snapshots={}",
                    samples.last().unwrap(),
                    metrics.runtime_decoder_rebuilds(),
                    provider.snapshots(),
                );
            }
            samples.sort_by(f64::total_cmp);
            println!("R3 median groups={groups} mode={mode} ms={:.3}", samples[1]);
        }
    }
}

#[tokio::test]
#[ignore = "timing test run explicitly by fork CI"]
async fn runtime_performance_promoted_integer() {
    use std::time::Instant;

    let temp = TempDir::new().unwrap();
    let path = temp.path().join("promoted-timing.parquet");
    let path = path.to_str().unwrap();
    let bases: Vec<_> = (0..2000).map(|group| group * 4).collect();
    write_groups(path, &bases, true, None, false);
    for primitive in [PrimitiveType::Int, PrimitiveType::Long] {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(primitive.clone())).into(),
                    NestedField::required(2, "payload", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .unwrap(),
        );
        let predicate = Reference::new("id").greater_than_or_equal_to(match primitive {
            PrimitiveType::Int => Datum::int(4000),
            _ => Datum::long(4000),
        });
        let mut samples = Vec::new();
        for runtime in [false, true] {
            samples.clear();
            for _ in 0..3 {
                let task = scan_task(path.to_string(), schema.clone(), None);
                let provider = runtime
                    .then_some(Arc::new(FixedRuntimePredicate::new(predicate.clone()))
                        as Arc<dyn RuntimePredicateProvider>);
                let started = Instant::now();
                let (batches, metrics) = execute(task, provider).await;
                samples.push(started.elapsed().as_secs_f64() * 1000.0);
                let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
                assert_eq!(rows, if runtime { 4000 } else { 8000 });
                println!(
                    "R3 promotion table={primitive} runtime={runtime} ms={:.3} rows={rows} bytes={} groups_pruned={}",
                    samples.last().unwrap(),
                    metrics.bytes_read(),
                    metrics.runtime_row_groups_pruned(),
                );
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "R3 promotion median table={primitive} runtime={runtime} ms={:.3}",
                samples[1]
            );
        }
    }
}

#[tokio::test]
async fn runtime_int_to_long_operator_matrix() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("int-long-matrix.parquet");
    let path = path.to_str().unwrap();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, 1),
        Field::new("value", DataType::Int32, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]));
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .set_data_page_row_count_limit(1)
        .set_write_batch_size(1)
        .set_bloom_filter_enabled(true)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(path).unwrap(),
        arrow_schema.clone(),
        Some(properties),
    )
    .unwrap();
    let values = [
        Some(i32::MIN),
        None,
        Some(-1),
        Some(0),
        Some(1),
        Some(i32::MAX),
    ];
    for (group, values) in values.chunks(2).enumerate() {
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(vec![
                group as i32 * 2,
                group as i32 * 2 + 1,
            ])),
            Arc::new(Int32Array::from(values.to_vec())),
        ])
        .unwrap();
        writer.write(&batch).unwrap();
        writer.flush().unwrap();
    }
    writer.close().unwrap();
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "value", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap(),
    );
    let below = Datum::long(i64::from(i32::MIN) - 1);
    let above = Datum::long(i64::from(i32::MAX) + 1);
    for (predicate, expected) in [
        (Reference::new("value").less_than(below.clone()), vec![]),
        (Reference::new("value").less_than(above.clone()), vec![
            0, 2, 3, 4, 5,
        ]),
        (
            Reference::new("value").less_than_or_equal_to(Datum::long(0)),
            vec![0, 2, 3],
        ),
        (Reference::new("value").greater_than(Datum::long(0)), vec![
            4, 5,
        ]),
        (
            Reference::new("value").greater_than_or_equal_to(above.clone()),
            vec![],
        ),
        (Reference::new("value").equal_to(Datum::long(1)), vec![4]),
        (Reference::new("value").not_equal_to(above.clone()), vec![
            0, 2, 3, 4, 5,
        ]),
        (
            Reference::new("value").is_in([below.clone(), Datum::long(-1), above.clone()]),
            vec![2],
        ),
        (
            Reference::new("value").is_not_in([below, Datum::long(-1), above]),
            vec![0, 3, 4, 5],
        ),
        (Reference::new("value").is_null(), vec![1]),
        (Reference::new("value").is_not_null(), vec![0, 2, 3, 4, 5]),
    ] {
        let planned = predicate.clone().bind(schema.clone(), false).unwrap();
        let task = scan_task(path.to_string(), schema.clone(), Some(planned));
        let (baseline, _) = execute_tasks(vec![task], None, true).await;
        assert_eq!(ids(&baseline), expected, "planned: {predicate}");
        for bloom in [false, true] {
            let task = scan_task(path.to_string(), schema.clone(), None);
            let provider = Arc::new(FixedRuntimePredicate::new(predicate.clone()));
            let (batches, metrics) = execute_tasks(vec![task], Some(provider), bloom).await;
            assert_eq!(
                ids(&batches),
                expected,
                "runtime: {predicate}, bloom={bloom}"
            );
            assert_eq!(metrics.runtime_predicate_tasks(), 1);
        }
    }
}

#[tokio::test]
async fn runtime_publication_budget_retains_mandatory_filters_and_deletes() {
    const GROUPS: i32 = 256;
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("budget.parquet");
    let path = path.to_str().unwrap();
    write_cycling_groups(path, GROUPS);
    let position_path = temp.path().join("budget-position.parquet");
    let position_path = position_path.to_str().unwrap();
    write_delete(
        position_path,
        vec![
            field("file_path", DataType::Utf8, 2_147_483_546),
            field("pos", DataType::Int64, 2_147_483_545),
        ],
        vec![
            Arc::new(StringArray::from(vec![path])),
            Arc::new(Int64Array::from(vec![502])),
        ],
    );
    let equality_path = temp.path().join("budget-equality.parquet");
    let equality_path = equality_path.to_str().unwrap();
    write_delete(equality_path, vec![field("id", DataType::Int32, 1)], vec![
        Arc::new(Int32Array::from(vec![402])),
    ]);
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "k", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap(),
    );
    let planned = Reference::new("k")
        .less_than_or_equal_to(Datum::int(2))
        .bind(schema.clone(), false)
        .unwrap();
    let task = scan_task_with_deletes_and_projection(
        path.to_string(),
        schema,
        Some(planned),
        vec![
            delete_task(position_path.to_string(), true),
            delete_task(equality_path.to_string(), false),
        ],
        vec![1, 2, crate::metadata_columns::RESERVED_FIELD_ID_POS],
    );
    let predicate = Reference::new("k").greater_than_or_equal_to(Datum::int(2));
    let provider = Arc::new(ChangingRuntimePredicate::new(Some(predicate.clone()), 0));
    let (mut stream, metrics) = start_runtime_scan(task, Some(provider.clone()), true, true, 4);
    let mut batches = Vec::new();
    for boundary in 1_i32..=12 {
        batches.push(stream.try_next().await.unwrap().unwrap());
        provider.publish(
            Some(
                predicate
                    .clone()
                    .and(Reference::new("id").greater_than_or_equal_to(Datum::int(boundary * 4))),
            ),
            boundary as u64,
        );
    }
    let rebuilds = metrics.runtime_decoder_rebuilds();
    assert!(rebuilds > 0 && rebuilds <= 9);
    provider.publish(Some(Predicate::AlwaysFalse), 13);
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    let expected: Vec<_> = (0..GROUPS)
        .map(|group| group * 4 + 2)
        .filter(|id| ![402, 502].contains(id))
        .collect();
    assert_eq!(ids(&batches), expected);
    let positions: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(
        positions,
        expected.into_iter().map(i64::from).collect::<Vec<_>>()
    );
    assert_eq!(metrics.runtime_decoder_rebuilds(), rebuilds);
    assert_eq!(metrics.runtime_predicate_refreshes(), rebuilds);
    assert!(provider.snapshots() < 13);
}

#[tokio::test]
async fn runtime_int_to_long_live_publication_prunes_groups_and_pages() {
    let temp = TempDir::new().unwrap();
    let path = write_three_row_group_file(temp.path().to_str().unwrap(), "live-long.parquet");
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "payload", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap(),
    );
    let task = scan_task(path, schema, None);
    let provider = Arc::new(ChangingRuntimePredicate::new(None, 0));
    let (mut stream, metrics) = start_runtime_scan(task, Some(provider.clone()), true, true, 4);
    let mut batches = vec![stream.try_next().await.unwrap().unwrap()];
    provider.publish(Some(Reference::new("id").greater_than(Datum::long(200))), 1);
    batches.extend(stream.try_collect::<Vec<_>>().await.unwrap());
    let values: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(values, vec![0, 1, 2, 3, 201, 202, 203]);
    assert_eq!(metrics.runtime_row_groups_pruned_live(), 1);
    assert_eq!(metrics.runtime_decoder_rebuilds(), 1);
}

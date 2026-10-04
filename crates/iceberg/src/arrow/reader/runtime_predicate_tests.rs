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

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

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

struct FailedProvider;

impl RuntimePredicateProvider for FailedProvider {
    fn generation(&self) -> u64 {
        0
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
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
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_data_file_concurrency_limit(1)
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

#[tokio::test]
async fn runtime_predicate_is_sampled_once_per_task() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().to_str().unwrap();
    let tasks = vec![
        scan_task(
            write_three_row_group_file(dir, "a.parquet"),
            iceberg_schema(),
            None,
        ),
        scan_task(
            write_three_row_group_file(dir, "b.parquet"),
            iceberg_schema(),
            None,
        ),
    ];
    let provider = Arc::new(FixedRuntimePredicate::new(
        Reference::new("id").greater_than_or_equal_to(Datum::int(200)),
    ));
    let (batches, _) = execute_tasks(tasks, Some(provider.clone()), false).await;
    assert_eq!(ids(&batches), vec![200, 201, 202, 203, 200, 201, 202, 203]);
    assert_eq!(provider.snapshots(), 2);
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
        Arc::new(FailedProvider) as Arc<dyn RuntimePredicateProvider>,
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
        let (runtime, metrics) = execute(task, Some(provider)).await;
        assert_eq!(ids(&runtime), expected);
        assert_eq!(
            ids(&baseline)
                .into_iter()
                .filter(|id| (100..=103).contains(id))
                .collect::<Vec<_>>(),
            expected
        );
        assert!(metrics.bytes_read() < baseline_metrics.bytes_read());
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

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

    for base in [0_i32, 100, 200] {
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

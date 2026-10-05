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

//! Opt-in end-to-end benchmark of runtime/planned row-filter ordering, kept on the
//! benchmark branch. Not part of the upstream change.

use std::fs::File;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::{ArrowReaderBuilder, RuntimePredicateProvider, RuntimePredicateSnapshot};
use crate::expr::{Bind, Predicate, Reference};
use crate::io::FileIO;
use crate::scan::{FileScanTask, FileScanTaskStream};
use crate::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type};
use crate::{Result, Runtime};

const ROWS: usize = 2_000_000;
const ROW_GROUP_ROWS: usize = 100_000;

struct Fixed(Predicate);

impl RuntimePredicateProvider for Fixed {
    fn generation(&self) -> u64 {
        1
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
        Ok(RuntimePredicateSnapshot::new(Some(self.0.clone()), 1))
    }
}

fn schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "a", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(3, "b", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(4, "payload", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap(),
    )
}

fn field(name: &str, data_type: DataType, id: i32) -> Field {
    Field::new(name, data_type, false).with_metadata(
        [(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]
            .into_iter()
            .collect(),
    )
}

/// Writes `ROWS` rows whose `a` and `b` are independent uniform values in 0..1000.
fn write_file(path: &str) {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, 1),
        field("a", DataType::Int64, 2),
        field("b", DataType::Int64, 3),
        field("payload", DataType::Utf8, 4),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(path).unwrap(),
        Arc::clone(&arrow_schema),
        Some(props),
    )
    .unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) % 1000) as i64
    };
    for start in (0..ROWS).step_by(ROW_GROUP_ROWS) {
        let ids: Vec<i32> = (start..start + ROW_GROUP_ROWS).map(|i| i as i32).collect();
        let a: Vec<i64> = (0..ROW_GROUP_ROWS).map(|_| next()).collect();
        let b: Vec<i64> = (0..ROW_GROUP_ROWS).map(|_| next()).collect();
        let payload: Vec<String> = ids.iter().map(|id| format!("payload-{id:020}")).collect();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Int64Array::from(a)),
            Arc::new(Int64Array::from(b)),
            Arc::new(StringArray::from(payload)),
        ];
        writer
            .write(&RecordBatch::try_new(Arc::clone(&arrow_schema), columns).unwrap())
            .unwrap();
    }
    writer.close().unwrap();
}

async fn scan(path: &str, planned_a_below: i64, runtime: Option<Predicate>) -> (usize, Duration) {
    let planned = Reference::new("a")
        .less_than(Datum::long(planned_a_below))
        .bind(schema(), false)
        .unwrap();
    let task = FileScanTask::builder()
        .with_file_size_in_bytes(std::fs::metadata(path).unwrap().len())
        .with_start(0)
        .with_length(0)
        .with_data_file_path(path.to_string())
        .with_data_file_format(DataFileFormat::Parquet)
        .with_schema(schema())
        .with_project_field_ids(vec![1])
        .with_predicate(Some(planned))
        .with_case_sensitive(false)
        .build()
        .unwrap();
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
        .with_data_file_concurrency_limit(1)
        .with_row_group_filtering_enabled(false);
    if let Some(runtime) = runtime {
        builder = builder.with_runtime_predicate_provider(Arc::new(Fixed(runtime)));
    }
    let start = Instant::now();
    let tasks = Box::pin(futures::stream::iter([Ok(task)])) as FileScanTaskStream;
    let batches: Vec<RecordBatch> = builder
        .build()
        .read(tasks)
        .unwrap()
        .stream()
        .try_collect()
        .await
        .unwrap();
    let elapsed = start.elapsed();
    (batches.iter().map(RecordBatch::num_rows).sum(), elapsed)
}

fn summarize(mut samples: Vec<Duration>) -> (f64, f64) {
    samples.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    (ms(samples[samples.len() / 2]), ms(samples[0]))
}

#[tokio::test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture"]
async fn filter_order_benchmark() {
    let repetitions: usize = std::env::var("ICEBERG_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let temp = TempDir::new().unwrap();
    let path = format!("{}/bench.parquet", temp.path().to_str().unwrap());
    write_file(&path);
    println!(
        "rows={ROWS} row_group_rows={ROW_GROUP_ROWS} repetitions={repetitions}; a,b uniform 0..1000; projection: id only; row-group filtering off"
    );
    println!(
        "| planned | runtime | columns | planned-only ms | planned + runtime ms | ratio | rows (planned-only / with runtime) |"
    );
    println!("| --- | --- | --- | ---: | ---: | ---: | --- |");
    for planned_below in [50_i64, 800] {
        for (column, columns) in [("a", "overlap"), ("b", "disjoint")] {
            for runtime_below in [10_i64, 500, 990] {
                let runtime = Reference::new(column).less_than(Datum::long(runtime_below));
                // Warm the page cache and allocator.
                scan(&path, planned_below, None).await;

                let mut baseline = vec![];
                let mut rows_baseline = 0;
                for _ in 0..repetitions {
                    let (rows, elapsed) = scan(&path, planned_below, None).await;
                    rows_baseline = rows;
                    baseline.push(elapsed);
                }
                let mut with_runtime = vec![];
                let mut rows_with_runtime = 0;
                for _ in 0..repetitions {
                    let (rows, elapsed) = scan(&path, planned_below, Some(runtime.clone())).await;
                    rows_with_runtime = rows;
                    with_runtime.push(elapsed);
                }
                let (baseline_ms, _) = summarize(baseline);
                let (runtime_ms, _) = summarize(with_runtime);
                println!(
                    "| a<{planned_below} | {column}<{runtime_below} | {columns} | {baseline_ms:.1} | {runtime_ms:.1} | {:.2} | {rows_baseline} / {rows_with_runtime} |",
                    runtime_ms / baseline_ms
                );
            }
        }
    }
}

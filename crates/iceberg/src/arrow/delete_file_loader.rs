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

use std::sync::Arc;

use arrow_array::{Array, Int64Array, StringArray};
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use roaring::RoaringTreemap;

use crate::arrow::ArrowReader;
use crate::arrow::reader::ParquetReadOptions;
use crate::arrow::record_batch_transformer::RecordBatchTransformerBuilder;
use crate::arrow::scan_metrics::ScanMetrics;
use crate::io::FileIO;
use crate::scan::{ArrowRecordBatchStream, FileScanTaskDeleteFile};
use crate::spec::{DataContentType, DataFileFormat, Schema, SchemaRef};
use crate::{Error, ErrorKind, Result};

/// Delete File Loader
#[allow(unused)]
#[async_trait::async_trait]
pub trait DeleteFileLoader {
    /// Read the delete file referred to in the task
    ///
    /// Returns the contents of the delete file as a RecordBatch stream. Applies schema evolution.
    async fn read_delete_file(
        &self,
        task: &FileScanTaskDeleteFile,
        schema: SchemaRef,
    ) -> Result<ArrowRecordBatchStream>;
}

#[derive(Clone, Debug)]
pub(crate) struct BasicDeleteFileLoader {
    file_io: FileIO,
    scan_metrics: ScanMetrics,
}

#[allow(unused_variables)]
impl BasicDeleteFileLoader {
    pub fn new(file_io: FileIO, scan_metrics: ScanMetrics) -> Self {
        BasicDeleteFileLoader {
            file_io,
            scan_metrics,
        }
    }

    pub(crate) fn file_io(&self) -> &FileIO {
        &self.file_io
    }

    /// Loads a RecordBatchStream for a given datafile.
    pub(crate) async fn parquet_to_batch_stream(
        &self,
        data_file_path: &str,
        file_size_in_bytes: u64,
        key_metadata: Option<&[u8]>,
    ) -> Result<ArrowRecordBatchStream> {
        /*
           Essentially a super-cut-down ArrowReader. We can't use ArrowReader directly
           as that introduces a circular dependency.
        */
        let parquet_read_options = ParquetReadOptions::builder().build();

        let (parquet_file_reader, arrow_metadata) = ArrowReader::open_parquet_file(
            data_file_path,
            &self.file_io,
            file_size_in_bytes,
            parquet_read_options,
            self.scan_metrics.bytes_read_counter(),
            key_metadata,
        )
        .await?;

        let record_batch_stream =
            ParquetRecordBatchStreamBuilder::new_with_metadata(parquet_file_reader, arrow_metadata)
                .build()?
                .map_err(|e| Error::new(ErrorKind::Unexpected, format!("{e}")));

        Ok(Box::pin(record_batch_stream) as ArrowRecordBatchStream)
    }

    /// Evolves the schema of the RecordBatches from an equality delete file.
    ///
    /// Per the [Iceberg spec](https://iceberg.apache.org/spec/#equality-delete-files),
    /// only evolves the specified `equality_ids` columns, not all table columns.
    pub(crate) async fn evolve_schema(
        record_batch_stream: ArrowRecordBatchStream,
        target_schema: Arc<Schema>,
        equality_ids: &[i32],
    ) -> Result<ArrowRecordBatchStream> {
        let mut record_batch_transformer =
            RecordBatchTransformerBuilder::new(target_schema.clone(), equality_ids).build();

        let record_batch_stream = record_batch_stream.map(move |record_batch| {
            record_batch.and_then(|record_batch| {
                record_batch_transformer.process_record_batch(record_batch)
            })
        });

        Ok(Box::pin(record_batch_stream) as ArrowRecordBatchStream)
    }
}

#[async_trait::async_trait]
impl DeleteFileLoader for BasicDeleteFileLoader {
    async fn read_delete_file(
        &self,
        task: &FileScanTaskDeleteFile,
        schema: SchemaRef,
    ) -> Result<ArrowRecordBatchStream> {
        let raw_batch_stream = self
            .parquet_to_batch_stream(
                &task.file_path,
                task.file_size_in_bytes,
                task.key_metadata.as_deref(),
            )
            .await?;

        // For equality deletes, only evolve the equality_ids columns.
        // For positional deletes (equality_ids is None), use all field IDs.
        let field_ids = match &task.equality_ids {
            Some(ids) => ids.clone(),
            None => schema.field_id_to_name_map().keys().cloned().collect(),
        };

        Self::evolve_schema(raw_batch_stream, schema, &field_ids).await
    }
}

/// An in-memory index of row positions from one file-scoped position-delete file.
///
/// The representation is intentionally hidden so callers do not depend on the bitmap
/// implementation used by Iceberg Rust.
#[derive(Debug)]
pub struct PositionDeleteIndex {
    positions: RoaringTreemap,
}

impl PositionDeleteIndex {
    fn new() -> Self {
        Self {
            positions: RoaringTreemap::new(),
        }
    }

    /// Returns the number of unique deleted row positions.
    pub fn len(&self) -> u64 {
        self.positions.len()
    }

    /// Returns whether the index contains no deleted row positions.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Returns whether `position` is present in the index.
    pub fn contains(&self, position: u64) -> bool {
        self.positions.contains(position)
    }

    /// Iterates deleted row positions in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.positions.iter()
    }
}

/// Loads an index from a single file-scoped V2 Parquet position-delete file.
///
/// This uses the same Parquet opening and encryption path as Iceberg scans. The caller supplies
/// the target data file. If the manifest entry carries `referenced_data_file`, it must match the
/// caller-provided path; the physical rows are also validated so metadata/content disagreement is
/// reported as corrupt input.
#[derive(Clone, Debug)]
pub struct PositionDeleteIndexLoader {
    basic_loader: BasicDeleteFileLoader,
}

impl PositionDeleteIndexLoader {
    /// Creates a loader for the given Iceberg `FileIO`.
    pub fn new(file_io: FileIO) -> Self {
        Self {
            basic_loader: BasicDeleteFileLoader::new(file_io, ScanMetrics::new()),
        }
    }

    fn reserved_field_index(
        schema: &arrow_schema::Schema,
        field_id: i32,
        logical_name: &str,
        delete_file_path: &str,
    ) -> Result<usize> {
        let mut matches = schema.fields().iter().enumerate().filter_map(|(index, field)| {
            field
                .metadata()
                .get(PARQUET_FIELD_ID_META_KEY)
                .and_then(|id| id.parse::<i32>().ok())
                .filter(|id| *id == field_id)
                .map(|_| index)
        });

        let Some(index) = matches.next() else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Position-delete file {delete_file_path} has no `{logical_name}` column with reserved field id {field_id}"
                ),
            ));
        };
        if matches.next().is_some() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Position-delete file {delete_file_path} has multiple columns with reserved field id {field_id}"
                ),
            ));
        }

        Ok(index)
    }

    fn position_delete_columns(
        schema: &arrow_schema::Schema,
        delete_file_path: &str,
    ) -> Result<(usize, usize)> {
        Ok((
            Self::reserved_field_index(
                schema,
                crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH,
                "file_path",
                delete_file_path,
            )?,
            Self::reserved_field_index(
                schema,
                crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_POS,
                "pos",
                delete_file_path,
            )?,
        ))
    }

    /// Reads and validates all positions in `delete_file` for exactly `expected_data_file`.
    ///
    /// Duplicate physical rows collapse to one position in the returned index, matching
    /// Iceberg's bitmap-based position-delete handling. The manifest `record_count`, when
    /// present, is validated against physical rows before de-duplication.
    pub async fn load_file_scoped_positions(
        &self,
        delete_file: &FileScanTaskDeleteFile,
        expected_data_file: &str,
    ) -> Result<PositionDeleteIndex> {
        if delete_file.file_type != DataContentType::PositionDeletes
            || delete_file.file_format != DataFileFormat::Parquet
            || delete_file.equality_ids.is_some()
        {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "Expected a V2 Parquet position-delete file, got {:?}/{:?} at {}",
                    delete_file.file_type, delete_file.file_format, delete_file.file_path
                ),
            ));
        }

        if delete_file.content_offset.is_some() || delete_file.content_size_in_bytes.is_some() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "V2 Parquet position-delete file {} must not carry deletion-vector content coordinates",
                    delete_file.file_path
                ),
            ));
        }

        if let Some(referenced_data_file) = delete_file.referenced_data_file.as_deref()
            && referenced_data_file != expected_data_file
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Position-delete file {} references {referenced_data_file}, expected {expected_data_file}",
                    delete_file.file_path
                ),
            ));
        }

        let mut batches = self
            .basic_loader
            .parquet_to_batch_stream(
                &delete_file.file_path,
                delete_file.file_size_in_bytes,
                delete_file.key_metadata.as_deref(),
            )
            .await?;

        let mut index = PositionDeleteIndex::new();
        let mut column_indexes = None;
        let mut rows_read = 0u64;

        while let Some(batch) = batches.try_next().await? {
            let (path_index, position_index) = match column_indexes {
                Some(indexes) => indexes,
                None => {
                    let indexes =
                        Self::position_delete_columns(batch.schema().as_ref(), &delete_file.file_path)?;
                    column_indexes = Some(indexes);
                    indexes
                }
            };

            let paths = batch
                .column(path_index)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Position-delete file {} has a non-Utf8 file_path column",
                            delete_file.file_path
                        ),
                    )
                })?;
            let row_positions = batch
                .column(position_index)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Position-delete file {} has a non-Int64 pos column",
                            delete_file.file_path
                        ),
                    )
                })?;
            if paths.null_count() != 0 || row_positions.null_count() != 0 {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("Position-delete file {} contains nulls", delete_file.file_path),
                ));
            }

            rows_read += batch.num_rows() as u64;
            for row in 0..batch.num_rows() {
                let data_file = paths.value(row);
                if data_file != expected_data_file {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "File-scoped position-delete file {} contains target {data_file}, expected {expected_data_file}",
                            delete_file.file_path
                        ),
                    ));
                }

                let position = row_positions.value(row);
                if position < 0 {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Position-delete file {} contains a negative row position {position}",
                            delete_file.file_path
                        ),
                    ));
                }
                index.positions.insert(position as u64);
            }
        }

        if let Some(expected_count) = delete_file.record_count
            && rows_read != expected_count
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Position-delete file {} contains {rows_read} rows, expected {expected_count} from record_count",
                    delete_file.file_path
                ),
            ));
        }

        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::ArrowWriter;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::delete_filter::tests::setup;
    use crate::arrow::test_utils::write_encrypted_parquet;

    fn write_plain_parquet(path: &str, batch: &RecordBatch) {
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
    }

    fn position_delete_batch(paths: Vec<&str>, positions: Vec<i64>) -> RecordBatch {
        let schema = crate::arrow::delete_filter::tests::create_pos_del_schema();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(paths)),
                Arc::new(Int64Array::from(positions)),
            ],
        )
        .unwrap()
    }

    fn position_delete_task(
        path: &str,
        record_count: Option<u64>,
        referenced_data_file: Option<&str>,
    ) -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile {
            file_path: path.to_string(),
            file_size_in_bytes: std::fs::metadata(path).unwrap().len(),
            file_type: DataContentType::PositionDeletes,
            file_format: DataFileFormat::Parquet,
            partition_spec_id: 0,
            equality_ids: None,
            referenced_data_file: referenced_data_file.map(str::to_string),
            content_offset: None,
            content_size_in_bytes: None,
            record_count,
            key_metadata: None,
        }
    }

    #[tokio::test]
    async fn test_position_delete_index_loader_reads_valid_file_and_deduplicates() {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join("pos-delete.parquet");
        let path = path.to_str().unwrap();
        let batch = position_delete_batch(
            vec!["data.parquet", "data.parquet", "data.parquet"],
            vec![5, 1, 5],
        );
        write_plain_parquet(path, &batch);

        let task = position_delete_task(path, Some(3), Some("data.parquet"));
        let loader = PositionDeleteIndexLoader::new(FileIO::new_with_fs());
        let index = loader
            .load_file_scoped_positions(&task, "data.parquet")
            .await
            .unwrap();

        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
        assert!(index.contains(1));
        assert!(index.contains(5));
        assert_eq!(index.iter().collect::<Vec<_>>(), vec![1, 5]);
    }

    #[tokio::test]
    async fn test_position_delete_index_loader_accepts_additional_columns() {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join("pos-delete-with-row.parquet");
        let path = path.to_str().unwrap();

        let base_schema = crate::arrow::delete_filter::tests::create_pos_del_schema();
        let schema = Arc::new(ArrowSchema::new(vec![
            base_schema.field(0).clone(),
            base_schema.field(1).clone(),
            Field::new("row_payload", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["data.parquet"])),
                Arc::new(Int64Array::from(vec![7i64])),
                Arc::new(Int64Array::from(vec![Some(42i64)])),
            ],
        )
        .unwrap();
        write_plain_parquet(path, &batch);

        let task = position_delete_task(path, Some(1), Some("data.parquet"));
        let index = PositionDeleteIndexLoader::new(FileIO::new_with_fs())
            .load_file_scoped_positions(&task, "data.parquet")
            .await
            .unwrap();

        assert_eq!(index.iter().collect::<Vec<_>>(), vec![7]);
    }

    #[tokio::test]
    async fn test_position_delete_index_loader_requires_reserved_field_ids() {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join("pos-delete-bad-id.parquet");
        let path = path.to_str().unwrap();

        let base_schema = crate::arrow::delete_filter::tests::create_pos_del_schema();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false),
            base_schema.field(1).clone(),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["data.parquet"])),
                Arc::new(Int64Array::from(vec![1i64])),
            ],
        )
        .unwrap();
        write_plain_parquet(path, &batch);

        let task = position_delete_task(path, Some(1), Some("data.parquet"));
        let err = PositionDeleteIndexLoader::new(FileIO::new_with_fs())
            .load_file_scoped_positions(&task, "data.parquet")
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(err.message().contains("reserved field id"));
    }

    #[tokio::test]
    async fn test_position_delete_index_loader_rejects_manifest_target_mismatch() {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join("pos-delete.parquet");
        let path = path.to_str().unwrap();
        write_plain_parquet(
            path,
            &position_delete_batch(vec!["data.parquet"], vec![1]),
        );

        let task = position_delete_task(path, Some(1), Some("other.parquet"));
        let err = PositionDeleteIndexLoader::new(FileIO::new_with_fs())
            .load_file_scoped_positions(&task, "data.parquet")
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(err.message().contains("references other.parquet"));
    }

    #[tokio::test]
    async fn test_position_delete_index_loader_rejects_row_target_mismatch() {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join("pos-delete.parquet");
        let path = path.to_str().unwrap();
        write_plain_parquet(
            path,
            &position_delete_batch(vec!["other.parquet"], vec![1]),
        );

        let task = position_delete_task(path, Some(1), None);
        let err = PositionDeleteIndexLoader::new(FileIO::new_with_fs())
            .load_file_scoped_positions(&task, "data.parquet")
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(err.message().contains("contains target other.parquet"));
    }

    #[tokio::test]
    async fn test_position_delete_index_loader_rejects_negative_position() {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join("pos-delete.parquet");
        let path = path.to_str().unwrap();
        write_plain_parquet(
            path,
            &position_delete_batch(vec!["data.parquet"], vec![-1]),
        );

        let task = position_delete_task(path, Some(1), Some("data.parquet"));
        let err = PositionDeleteIndexLoader::new(FileIO::new_with_fs())
            .load_file_scoped_positions(&task, "data.parquet")
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(err.message().contains("negative row position"));
    }

    #[tokio::test]
    async fn test_position_delete_index_loader_validates_record_count() {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join("pos-delete.parquet");
        let path = path.to_str().unwrap();
        write_plain_parquet(
            path,
            &position_delete_batch(
                vec!["data.parquet", "data.parquet"],
                vec![1, 2],
            ),
        );

        let task = position_delete_task(path, Some(3), Some("data.parquet"));
        let err = PositionDeleteIndexLoader::new(FileIO::new_with_fs())
            .load_file_scoped_positions(&task, "data.parquet")
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(err.message().contains("expected 3 from record_count"));
    }

    #[tokio::test]
    async fn test_position_delete_index_loader_rejects_dv_coordinates() {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join("pos-delete.parquet");
        let path = path.to_str().unwrap();
        write_plain_parquet(
            path,
            &position_delete_batch(vec!["data.parquet"], vec![1]),
        );

        let mut task = position_delete_task(path, Some(1), Some("data.parquet"));
        task.content_offset = Some(0);
        task.content_size_in_bytes = Some(10);

        let err = PositionDeleteIndexLoader::new(FileIO::new_with_fs())
            .load_file_scoped_positions(&task, "data.parquet")
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(err.message().contains("deletion-vector content coordinates"));
    }

    #[tokio::test]
    async fn test_basic_delete_file_loader_read_delete_file() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let scan_metrics = ScanMetrics::new();
        let delete_file_loader = BasicDeleteFileLoader::new(file_io.clone(), scan_metrics);

        let file_scan_tasks = setup(table_location);

        let result = delete_file_loader
            .read_delete_file(
                &file_scan_tasks[0].deletes()[0],
                file_scan_tasks[0].schema_ref(),
            )
            .await
            .unwrap();

        let result = result.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_read_encrypted_positional_delete_file() {
        use std::sync::Arc;

        use arrow_array::{Int64Array, RecordBatch, StringArray};

        use crate::arrow::delete_filter::tests::create_pos_del_schema;
        use crate::encryption::StandardKeyMetadata;
        use crate::scan::FileScanTaskDeleteFile;
        use crate::spec::{DataContentType, DataFileFormat};

        let encryption_key = b"0123456789abcdef";
        let aad_prefix = b"aad_prefix";

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap();
        let file_io = FileIO::new_with_fs();

        let positional_delete_schema = create_pos_del_schema();
        let file_path_col = Arc::new(StringArray::from_iter_values(vec!["data.parquet"; 4]));
        let pos_col = Arc::new(Int64Array::from(vec![0i64, 1, 5, 10]));
        let batch = RecordBatch::try_new(positional_delete_schema.clone(), vec![
            file_path_col,
            pos_col,
        ])
        .unwrap();

        let del_path = format!("{table_location}/encrypted-pos-del.parquet");
        write_encrypted_parquet(&del_path, &batch, encryption_key, Some(aad_prefix));

        let key_metadata = StandardKeyMetadata::try_new(encryption_key)
            .unwrap()
            .with_aad_prefix(aad_prefix)
            .encode()
            .unwrap();

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    crate::spec::NestedField::required(
                        2147483546,
                        "file_path",
                        crate::spec::Type::Primitive(crate::spec::PrimitiveType::String),
                    )
                    .into(),
                    crate::spec::NestedField::required(
                        2147483545,
                        "pos",
                        crate::spec::Type::Primitive(crate::spec::PrimitiveType::Long),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let task = FileScanTaskDeleteFile {
            file_path: del_path.clone(),
            file_size_in_bytes: std::fs::metadata(&del_path).unwrap().len(),
            file_type: DataContentType::PositionDeletes,
            file_format: DataFileFormat::Parquet,
            partition_spec_id: 0,
            equality_ids: None,
            key_metadata: Some(Box::from(key_metadata.as_ref())),
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };

        let scan_metrics = ScanMetrics::new();
        let delete_file_loader = BasicDeleteFileLoader::new(file_io.clone(), scan_metrics);

        let result = delete_file_loader
            .read_delete_file(&task, schema)
            .await
            .unwrap();

        let batches: Vec<_> = result.try_collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 4);

        let index = PositionDeleteIndexLoader::new(file_io)
            .load_file_scoped_positions(&task, "data.parquet")
            .await
            .unwrap();
        assert_eq!(index.iter().collect::<Vec<_>>(), vec![0, 1, 5, 10]);
    }

    #[tokio::test]
    async fn test_read_encrypted_equality_delete_file() {
        use std::collections::HashMap;
        use std::sync::Arc;

        use arrow_array::{Int64Array, RecordBatch};
        use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

        use crate::encryption::StandardKeyMetadata;
        use crate::scan::FileScanTaskDeleteFile;
        use crate::spec::{DataContentType, DataFileFormat};

        let encryption_key = b"0123456789abcdef";
        let aad_prefix = b"my-table-uuid!!";

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap();
        let file_io = FileIO::new_with_fs();

        let arrow_schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Int64, false).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
            ),
        ]));

        let id_col = Arc::new(Int64Array::from(vec![100i64, 200, 300]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![id_col]).unwrap();

        let del_path = format!("{table_location}/encrypted-eq-del.parquet");
        write_encrypted_parquet(&del_path, &batch, encryption_key, Some(aad_prefix));

        let key_metadata = StandardKeyMetadata::try_new(encryption_key)
            .unwrap()
            .with_aad_prefix(aad_prefix)
            .encode()
            .unwrap();

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    crate::spec::NestedField::required(
                        1,
                        "id",
                        crate::spec::Type::Primitive(crate::spec::PrimitiveType::Long),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let task = FileScanTaskDeleteFile {
            file_path: del_path.clone(),
            file_size_in_bytes: std::fs::metadata(&del_path).unwrap().len(),
            file_type: DataContentType::EqualityDeletes,
            file_format: DataFileFormat::Parquet,
            partition_spec_id: 0,
            equality_ids: Some(vec![1]),
            key_metadata: Some(Box::from(key_metadata.as_ref())),
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };

        let scan_metrics = ScanMetrics::new();
        let delete_file_loader = BasicDeleteFileLoader::new(file_io, scan_metrics);

        let result = delete_file_loader
            .read_delete_file(&task, schema)
            .await
            .unwrap();

        let batches: Vec<_> = result.try_collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
    }
}

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

//! This module provide `DeltaWriter`, which composes a data writer, a position delete
//! writer, and an equality delete writer to support row-level `INSERT`/`DELETE`/`UPDATE`
//! within a single write task (the analogue of Java's `BaseTaskWriter.RowDataDeltaWriter`).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_cast::display::array_value_to_string;

use crate::arrow::schema_to_arrow_schema;
use crate::spec::{DataFile, PartitionKey};
use crate::writer::base_writer::equality_delete_writer::EqualityDeleteWriterConfig;
use crate::writer::base_writer::position_delete_writer::position_delete_schema;
use crate::writer::{CurrentFileStatus, IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Builder for `DeltaWriter`.
///
/// Composes three independently configured [`IcebergWriterBuilder`]s: one for inserted rows
/// (written in full), one for position deletes, and one for equality deletes. Any writer
/// builder can be used for each role (e.g. a plain [`DataFileWriterBuilder`][crate::writer::base_writer::data_file_writer::DataFileWriterBuilder],
/// or one wrapped in a partitioning writer), as long as the data writer's output also
/// implements [`CurrentFileStatus`] so `DeltaWriter` can record where each inserted row landed.
#[derive(Debug)]
pub struct DeltaWriterBuilder<D, PD, ED>
where
    D: IcebergWriterBuilder,
    D::R: CurrentFileStatus,
    PD: IcebergWriterBuilder,
    ED: IcebergWriterBuilder,
{
    data_writer_builder: D,
    position_delete_writer_builder: PD,
    equality_delete_writer_builder: ED,
    equality_config: EqualityDeleteWriterConfig,
}

impl<D, PD, ED> DeltaWriterBuilder<D, PD, ED>
where
    D: IcebergWriterBuilder,
    D::R: CurrentFileStatus,
    PD: IcebergWriterBuilder,
    ED: IcebergWriterBuilder,
{
    /// Create a new `DeltaWriterBuilder`.
    ///
    /// `equality_config` determines which columns identify a row for the purposes of both
    /// equality deletes and the in-task insert index; it must be built from the same schema
    /// and equality ids used to configure `equality_delete_writer_builder`'s delete schema.
    pub fn new(
        data_writer_builder: D,
        position_delete_writer_builder: PD,
        equality_delete_writer_builder: ED,
        equality_config: EqualityDeleteWriterConfig,
    ) -> Self {
        Self {
            data_writer_builder,
            position_delete_writer_builder,
            equality_delete_writer_builder,
            equality_config,
        }
    }

    /// Build a `DeltaWriter` for the given partition.
    ///
    /// Not implemented as [`IcebergWriterBuilder`] because `DeltaWriter` exposes
    /// `insert`/`delete`/`update`, not the single-method [`IcebergWriter`] contract.
    pub async fn build(
        &self,
        partition_key: Option<PartitionKey>,
    ) -> Result<DeltaWriter<D, PD, ED>> {
        Ok(DeltaWriter {
            data_writer: self
                .data_writer_builder
                .build(partition_key.clone())
                .await?,
            position_delete_writer: self
                .position_delete_writer_builder
                .build(partition_key.clone())
                .await?,
            equality_delete_writer: self
                .equality_delete_writer_builder
                .build(partition_key)
                .await?,
            equality_config: self.equality_config.clone(),
            insert_index: HashMap::new(),
            closed: false,
        })
    }
}

/// Writer supporting row-level `insert`/`delete`/`update` within one write task.
///
/// A row deleted after being inserted through the *same* `DeltaWriter` instance is recorded
/// as a position delete against the file and offset it was just written to, rather than an
/// equality delete; this mirrors Java's `BaseTaskWriter.RowDataDeltaWriter` and avoids
/// emitting equality deletes for insert/delete pairs that never leave the task. Deletes for
/// rows not seen as inserts in this task fall back to equality deletes.
///
/// `insert`, `delete`, and `update` each take exactly one row per call: `DeltaWriter`
/// determines an inserted row's file and position from the underlying writer's state
/// immediately after writing it, which is only well-defined one row at a time.
#[derive(Debug)]
pub struct DeltaWriter<D, PD, ED>
where
    D: IcebergWriterBuilder,
    D::R: CurrentFileStatus,
    PD: IcebergWriterBuilder,
    ED: IcebergWriterBuilder,
{
    data_writer: D::R,
    position_delete_writer: PD::R,
    equality_delete_writer: ED::R,
    equality_config: EqualityDeleteWriterConfig,
    /// Identifier key (see [`Self::identifier_key`]) -> (file_path, pos) for every row
    /// inserted so far in this task that hasn't since been deleted.
    insert_index: HashMap<String, (String, i64)>,
    closed: bool,
}

impl<D, PD, ED> DeltaWriter<D, PD, ED>
where
    D: IcebergWriterBuilder,
    D::R: CurrentFileStatus,
    PD: IcebergWriterBuilder,
    ED: IcebergWriterBuilder,
{
    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Delta writer has been closed.",
            ));
        }
        Ok(())
    }

    fn ensure_single_row(batch: &RecordBatch, method: &str) -> Result<()> {
        if batch.num_rows() != 1 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "DeltaWriter::{method} expects exactly one row per call, got {}",
                    batch.num_rows()
                ),
            ));
        }
        Ok(())
    }

    /// Build a string key identifying a row from its (already validated single-row)
    /// equality delete columns. NUL-delimited so that no combination of identifier values
    /// can collide with a different combination, as long as no value itself contains a NUL
    /// byte, which holds for the primitive, non-floating types equality ids are restricted
    /// to.
    fn identifier_key(&self, row: RecordBatch) -> Result<String> {
        let projected = self.equality_config.project_batch(row)?;
        let mut key = String::new();
        for column in projected.columns() {
            if column.is_null(0) {
                key.push_str("\u{0}N\u{0}");
            } else {
                let value = array_value_to_string(column.as_ref(), 0).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Failed to stringify identifier column: {e}"),
                    )
                })?;
                write!(key, "\u{0}{value}\u{0}").unwrap();
            }
        }
        Ok(key)
    }

    fn position_delete_row(file_path: &str, pos: i64) -> Result<RecordBatch> {
        let arrow_schema = Arc::new(schema_to_arrow_schema(&position_delete_schema())?);
        RecordBatch::try_new(arrow_schema, vec![
            Arc::new(StringArray::from(vec![file_path.to_string()])),
            Arc::new(Int64Array::from(vec![pos])),
        ])
        .map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to build position delete row: {e}"),
            )
        })
    }

    /// Insert one row, shaped like the table's full original schema.
    pub async fn insert(&mut self, row: RecordBatch) -> Result<()> {
        self.ensure_open()?;
        Self::ensure_single_row(&row, "insert")?;

        let key = self.identifier_key(row.clone())?;
        self.data_writer.write(row).await?;
        // Valid immediately after the write above: the row just landed in whichever file
        // and at whichever offset the underlying writer's state now reflects, even if this
        // write triggered a roll to a new file.
        let pos = self.data_writer.current_row_num() as i64 - 1;
        let file_path = self.data_writer.current_file_path();
        self.insert_index.insert(key, (file_path, pos));
        Ok(())
    }

    /// Delete one row, identified by an `identifier` batch shaped like the table's full
    /// original schema (only the configured equality delete columns are used).
    pub async fn delete(&mut self, identifier: RecordBatch) -> Result<()> {
        self.ensure_open()?;
        Self::ensure_single_row(&identifier, "delete")?;

        let key = self.identifier_key(identifier.clone())?;
        if let Some((file_path, pos)) = self.insert_index.remove(&key) {
            let pos_row = Self::position_delete_row(&file_path, pos)?;
            self.position_delete_writer.write(pos_row).await
        } else {
            self.equality_delete_writer.write(identifier).await
        }
    }

    /// Update one row: delete the row identified by `identifier`, then insert `row`.
    pub async fn update(&mut self, identifier: RecordBatch, row: RecordBatch) -> Result<()> {
        self.delete(identifier).await?;
        self.insert(row).await
    }

    /// Close the writer, returning every data file, position delete file, and equality
    /// delete file produced.
    pub async fn close(&mut self) -> Result<Vec<DataFile>> {
        self.ensure_open()?;
        self.closed = true;

        let mut files = self.data_writer.close().await?;
        files.extend(self.position_delete_writer.close().await?);
        files.extend(self.equality_delete_writer.close().await?);
        Ok(files)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::arrow_schema_to_schema;
    use crate::io::FileIO;
    use crate::spec::{
        DataContentType, DataFileFormat, NestedField, PrimitiveType, Schema, SchemaRef, Type,
    };
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::base_writer::equality_delete_writer::EqualityDeleteFileWriterBuilder;
    use crate::writer::base_writer::sorted_position_delete_writer::{
        SortedPositionDeleteWriterBuilder, SortedPositionDeleteWriterConfig,
    };
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;

    fn original_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn row_batch(schema: &Schema, id: i32, name: &str) -> RecordBatch {
        let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());
        RecordBatch::try_new(arrow_schema, vec![
            Arc::new(arrow_array::Int32Array::from(vec![id])),
            Arc::new(StringArray::from(vec![name])),
        ])
        .unwrap()
    }

    type TestDeltaWriterBuilder = DeltaWriterBuilder<
        DataFileWriterBuilder<
            ParquetWriterBuilder,
            DefaultLocationGenerator,
            DefaultFileNameGenerator,
        >,
        SortedPositionDeleteWriterBuilder<
            ParquetWriterBuilder,
            DefaultLocationGenerator,
            DefaultFileNameGenerator,
        >,
        EqualityDeleteFileWriterBuilder<
            ParquetWriterBuilder,
            DefaultLocationGenerator,
            DefaultFileNameGenerator,
        >,
    >;

    fn setup(file_prefix: &str) -> (TempDir, FileIO, SchemaRef, TestDeltaWriterBuilder) {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let schema = original_schema();

        let data_pb =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone());
        let data_rolling = RollingFileWriterBuilder::new_with_default_file_size(
            data_pb,
            file_io.clone(),
            location_gen.clone(),
            DefaultFileNameGenerator::new(
                format!("{file_prefix}_data"),
                None,
                DataFileFormat::Parquet,
            ),
        );
        let data_writer_builder = DataFileWriterBuilder::new(data_rolling);

        let pos_pb = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(position_delete_schema()),
        );
        let pos_rolling = RollingFileWriterBuilder::new_with_default_file_size(
            pos_pb,
            file_io.clone(),
            location_gen.clone(),
            DefaultFileNameGenerator::new(
                format!("{file_prefix}_pos_delete"),
                None,
                DataFileFormat::Parquet,
            ),
        );
        let position_delete_writer_builder = SortedPositionDeleteWriterBuilder::new(
            pos_rolling,
            SortedPositionDeleteWriterConfig::new(1024).unwrap(),
        );

        let equality_config = EqualityDeleteWriterConfig::new(vec![1], schema.clone()).unwrap();
        let eq_delete_schema =
            arrow_schema_to_schema(equality_config.projected_arrow_schema_ref()).unwrap();
        let eq_pb = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(eq_delete_schema),
        );
        let eq_rolling = RollingFileWriterBuilder::new_with_default_file_size(
            eq_pb,
            file_io.clone(),
            location_gen,
            DefaultFileNameGenerator::new(
                format!("{file_prefix}_eq_delete"),
                None,
                DataFileFormat::Parquet,
            ),
        );
        let equality_delete_writer_builder =
            EqualityDeleteFileWriterBuilder::new(eq_rolling, equality_config.clone());

        let builder = DeltaWriterBuilder::new(
            data_writer_builder,
            position_delete_writer_builder,
            equality_delete_writer_builder,
            equality_config,
        );

        (temp_dir, file_io, schema, builder)
    }

    #[tokio::test]
    async fn test_insert_then_delete_same_task_becomes_position_delete() -> Result<()> {
        let (_temp_dir, _file_io, schema, builder) = setup("test_insert_then_delete_same_task");
        let mut writer = builder.build(None).await?;

        writer.insert(row_batch(&schema, 1, "a")).await?;
        writer.delete(row_batch(&schema, 1, "a")).await?;

        let data_files = writer.close().await?;
        let contents: Vec<_> = data_files.iter().map(|f| f.content_type()).collect();
        assert_eq!(contents.len(), 2, "{contents:?}");
        assert!(contents.contains(&DataContentType::Data));
        assert!(contents.contains(&DataContentType::PositionDeletes));
        assert!(!contents.contains(&DataContentType::EqualityDeletes));

        let pos_delete_file = data_files
            .iter()
            .find(|f| f.content_type() == DataContentType::PositionDeletes)
            .unwrap();
        assert_eq!(pos_delete_file.record_count(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_delete_without_prior_insert_becomes_equality_delete() -> Result<()> {
        let (_temp_dir, _file_io, schema, builder) = setup("test_delete_without_prior_insert");
        let mut writer = builder.build(None).await?;

        writer.delete(row_batch(&schema, 5, "z")).await?;

        let data_files = writer.close().await?;
        assert_eq!(data_files.len(), 1);
        assert_eq!(
            data_files[0].content_type(),
            DataContentType::EqualityDeletes
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_update_deletes_old_row_and_inserts_new() -> Result<()> {
        let (_temp_dir, _file_io, schema, builder) = setup("test_update");
        let mut writer = builder.build(None).await?;

        writer.insert(row_batch(&schema, 1, "a")).await?;
        writer
            .update(row_batch(&schema, 1, "a"), row_batch(&schema, 1, "b"))
            .await?;

        let data_files = writer.close().await?;
        let data_file = data_files
            .iter()
            .find(|f| f.content_type() == DataContentType::Data)
            .unwrap();
        assert_eq!(data_file.record_count(), 2);
        let pos_delete_file = data_files
            .iter()
            .find(|f| f.content_type() == DataContentType::PositionDeletes)
            .unwrap();
        assert_eq!(pos_delete_file.record_count(), 1);
        assert!(
            !data_files
                .iter()
                .any(|f| f.content_type() == DataContentType::EqualityDeletes)
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_rejects_multi_row_batch() -> Result<()> {
        let (_temp_dir, _file_io, schema, builder) = setup("test_multi_row");
        let mut writer = builder.build(None).await?;

        let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let two_rows = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(arrow_array::Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ])
        .unwrap();

        let err = writer.insert(two_rows).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DataInvalid);

        Ok(())
    }

    #[tokio::test]
    async fn test_rejects_reuse_after_close() -> Result<()> {
        let (_temp_dir, _file_io, schema, builder) = setup("test_reuse_after_close");
        let mut writer = builder.build(None).await?;

        writer.insert(row_batch(&schema, 1, "a")).await?;
        let _ = writer.close().await?;

        let err = writer.insert(row_batch(&schema, 2, "b")).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);

        Ok(())
    }

    /// End-to-end check that an insert-then-delete pair resolved to a position delete is
    /// correctly interpreted by iceberg-rust's own read path: the reader must mark exactly
    /// the inserted row's position as deleted on the file DeltaWriter actually wrote it to.
    #[tokio::test]
    async fn test_delta_writer_position_delete_output_round_trips_through_reader() -> Result<()> {
        use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
        use crate::runtime::Runtime;
        use crate::scan::{FileScanTask, FileScanTaskDeleteFile};

        let (_temp_dir, file_io, schema, builder) = setup("test_delta_writer_pos_delete_roundtrip");
        let mut writer = builder.build(None).await?;

        writer.insert(row_batch(&schema, 1, "a")).await?;
        writer.delete(row_batch(&schema, 1, "a")).await?;
        let data_files = writer.close().await?;

        let data_file = data_files
            .iter()
            .find(|f| f.content_type() == DataContentType::Data)
            .unwrap();
        let pos_delete_file = data_files
            .iter()
            .find(|f| f.content_type() == DataContentType::PositionDeletes)
            .unwrap();

        let delete_task = FileScanTaskDeleteFile::builder()
            .with_file_path(pos_delete_file.file_path().to_string())
            .with_file_size_in_bytes(pos_delete_file.file_size_in_bytes())
            .with_file_type(DataContentType::PositionDeletes)
            .with_partition_spec_id(0)
            .build();

        let scan_task = FileScanTask::builder()
            .with_file_size_in_bytes(data_file.file_size_in_bytes())
            .with_start(0)
            .with_length(data_file.file_size_in_bytes())
            .with_data_file_path(data_file.file_path().to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema.clone())
            .with_project_field_ids(vec![])
            .with_deletes(vec![delete_task])
            .with_case_sensitive(false)
            .build();

        let delete_file_loader = CachingDeleteFileLoader::new(file_io, 10, Runtime::current());
        let delete_filter = delete_file_loader
            .load_deletes(&scan_task.deletes, scan_task.schema_ref())
            .await
            .unwrap()?;
        let delete_vector = delete_filter.get_delete_vector(&scan_task).unwrap();
        let positions: Vec<u64> = delete_vector.lock().unwrap().iter().collect();
        assert_eq!(positions, vec![0]);

        Ok(())
    }

    /// End-to-end check that a delete with no matching in-task insert, which DeltaWriter falls
    /// back to writing as an equality delete, is correctly parsed back into an equivalent
    /// exclusion predicate by iceberg-rust's own read path.
    #[tokio::test]
    async fn test_delta_writer_equality_delete_output_round_trips_through_reader() -> Result<()> {
        use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
        use crate::runtime::Runtime;
        use crate::scan::{FileScanTask, FileScanTaskDeleteFile};

        let (_temp_dir, file_io, schema, builder) = setup("test_delta_writer_eq_delete_roundtrip");
        let mut writer = builder.build(None).await?;

        writer.delete(row_batch(&schema, 5, "z")).await?;
        let data_files = writer.close().await?;
        assert_eq!(data_files.len(), 1);
        let eq_delete_file = &data_files[0];
        assert_eq!(
            eq_delete_file.content_type(),
            DataContentType::EqualityDeletes
        );

        let delete_task = FileScanTaskDeleteFile::builder()
            .with_file_path(eq_delete_file.file_path().to_string())
            .with_file_size_in_bytes(eq_delete_file.file_size_in_bytes())
            .with_file_type(DataContentType::EqualityDeletes)
            .with_partition_spec_id(0)
            .with_equality_ids(Some(vec![1]))
            .build();

        // The referenced data file need not exist: with only an equality delete attached, the
        // reader resolves a predicate without ever opening the data file itself.
        let scan_task = FileScanTask::builder()
            .with_file_size_in_bytes(0)
            .with_start(0)
            .with_length(0)
            .with_data_file_path("unread.parquet".to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema)
            .with_project_field_ids(vec![])
            .with_deletes(vec![delete_task])
            .with_case_sensitive(false)
            .build();

        let delete_file_loader = CachingDeleteFileLoader::new(file_io, 10, Runtime::current());
        let delete_filter = delete_file_loader
            .load_deletes(&scan_task.deletes, scan_task.schema_ref())
            .await
            .unwrap()?;
        let predicate = delete_filter
            .get_equality_delete_predicate_for_delete_file_path(eq_delete_file.file_path())
            .await
            .expect("equality delete predicate should have loaded");
        assert_eq!(predicate.to_string(), "(id IS NULL) OR (id != 5)");

        Ok(())
    }
}

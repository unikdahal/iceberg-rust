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

//! This module provide `PositionDeleteFileWriter`.

use arrow_array::{RecordBatch, StringArray};

use crate::metadata_columns::{delete_file_path_field, delete_file_pos_field};
use crate::spec::{DataContentType, DataFile, PartitionKey, Schema};
use crate::writer::file_writer::FileWriterBuilder;
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::rolling_writer::{RollingFileWriter, RollingFileWriterBuilder};
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Returns the fixed Iceberg schema used by position delete files: `file_path` (required
/// string, field id [`RESERVED_FIELD_ID_DELETE_FILE_PATH`][crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH])
/// followed by `pos` (required long, field id [`RESERVED_FIELD_ID_DELETE_FILE_POS`][crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_POS]).
///
/// Column order matters: readers (see `caching_delete_file_loader`) assume `file_path` is
/// column 0 and `pos` is column 1.
pub fn position_delete_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            delete_file_path_field().clone(),
            delete_file_pos_field().clone(),
        ])
        .build()
        .expect("position delete schema is statically valid")
}

/// Builder for `PositionDeleteFileWriter`.
#[derive(Debug)]
pub struct PositionDeleteFileWriterBuilder<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    inner: RollingFileWriterBuilder<B, L, F>,
}

impl<B, L, F> PositionDeleteFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Create a new `PositionDeleteFileWriterBuilder` using a `RollingFileWriterBuilder`.
    ///
    /// The `RollingFileWriterBuilder` must be constructed with a file writer configured for
    /// [`position_delete_schema`].
    pub fn new(inner: RollingFileWriterBuilder<B, L, F>) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriterBuilder for PositionDeleteFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    type R = PositionDeleteFileWriter<B, L, F>;

    async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R> {
        Ok(PositionDeleteFileWriter {
            inner: Some(self.inner.build()),
            partition_key,
            referenced_data_file: None,
            saw_multiple_referenced_files: false,
        })
    }
}

/// Writer used to write position delete files.
///
/// Rows must be projected to the fixed [`position_delete_schema`] (`file_path`, `pos`) before
/// being passed to [`write`][IcebergWriter::write]. Iceberg requires position delete files to
/// be sorted by `(file_path, pos)` within each file; this writer does not sort, it writes rows
/// in the order given.
#[derive(Debug)]
pub struct PositionDeleteFileWriter<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    inner: Option<RollingFileWriter<B, L, F>>,
    partition_key: Option<PartitionKey>,
    /// The single data file every delete written so far refers to, if there is one.
    referenced_data_file: Option<String>,
    /// Set once a batch has been observed to reference more than one data file, at which
    /// point `referenced_data_file` is no longer meaningful and is left unset on output files.
    saw_multiple_referenced_files: bool,
}

impl<B, L, F> PositionDeleteFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Track whether every row written so far references the same data file, so `close` can
    /// populate `DataFile.referenced_data_file` when it is safe to do so.
    ///
    /// This is best-effort: it is purely a read-side optimisation (see #2936), so a batch
    /// whose column 0 isn't a plain `StringArray` (column order per [`position_delete_schema`])
    /// just disables the optimisation rather than failing the write.
    fn track_referenced_data_file(&mut self, batch: &RecordBatch) {
        if self.saw_multiple_referenced_files {
            return;
        }

        let Some(file_paths) = batch.column(0).as_any().downcast_ref::<StringArray>() else {
            self.saw_multiple_referenced_files = true;
            self.referenced_data_file = None;
            return;
        };

        for file_path in file_paths.iter().flatten() {
            match &self.referenced_data_file {
                None => self.referenced_data_file = Some(file_path.to_string()),
                Some(existing) if existing != file_path => {
                    self.saw_multiple_referenced_files = true;
                    self.referenced_data_file = None;
                    break;
                }
                _ => {}
            }
        }
    }
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriter for PositionDeleteFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        self.track_referenced_data_file(&batch);
        if let Some(writer) = self.inner.as_mut() {
            writer.write(&self.partition_key, &batch).await
        } else {
            Err(Error::new(
                ErrorKind::Unexpected,
                "Position delete inner writer has been closed.",
            ))
        }
    }

    async fn close(&mut self) -> Result<Vec<DataFile>> {
        if let Some(writer) = self.inner.take() {
            let referenced_data_file = self.referenced_data_file.take();
            writer
                .close()
                .await?
                .into_iter()
                .map(|mut res| {
                    res.content(DataContentType::PositionDeletes);
                    // Position deletes must be sorted by (file_path, pos), not by a table sort
                    // order; the spec requires sort_order_id to be left null for them.
                    if let Some(pk) = self.partition_key.as_ref() {
                        res.partition(pk.data().clone());
                        res.partition_spec_id(pk.spec().spec_id());
                    }
                    if let Some(path) = referenced_data_file.clone() {
                        res.referenced_data_file(Some(path));
                    }
                    res.build().map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Failed to build data file: {e}"),
                        )
                    })
                })
                .collect()
        } else {
            Err(Error::new(
                ErrorKind::Unexpected,
                "Position delete inner writer has been closed.",
            ))
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch};
    use itertools::Itertools;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::schema_to_arrow_schema;
    use crate::io::FileIO;
    use crate::metadata_columns::{
        RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
    };
    use crate::spec::{DataFileFormat, Literal, PartitionSpec, Struct};
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;

    fn setup(
        file_prefix: &str,
    ) -> (
        TempDir,
        FileIO,
        RollingFileWriterBuilder<
            ParquetWriterBuilder,
            DefaultLocationGenerator,
            DefaultFileNameGenerator,
        >,
    ) {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new(file_prefix.to_string(), None, DataFileFormat::Parquet);

        let pb = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(position_delete_schema()),
        );
        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            pb,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );

        (temp_dir, file_io, rolling_writer_builder)
    }

    fn position_delete_batch(paths: Vec<&str>, positions: Vec<i64>) -> RecordBatch {
        let arrow_schema = schema_to_arrow_schema(&position_delete_schema()).unwrap();
        RecordBatch::try_new(Arc::new(arrow_schema), vec![
            Arc::new(StringArray::from(paths)),
            Arc::new(Int64Array::from(positions)),
        ])
        .unwrap()
    }

    #[tokio::test]
    async fn test_position_delete_writer() -> Result<()> {
        let (_temp_dir, file_io, rolling_writer_builder) = setup("test_position_delete");

        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_writer_builder)
            .build(None)
            .await?;

        let batch = position_delete_batch(vec!["data_1.parquet", "data_1.parquet"], vec![1, 3]);
        writer.write(batch.clone()).await?;
        let data_files = writer.close().await?;
        assert_eq!(data_files.len(), 1);

        let data_file = &data_files[0];
        assert_eq!(data_file.file_format, DataFileFormat::Parquet);
        assert_eq!(data_file.content_type(), DataContentType::PositionDeletes);
        assert_eq!(data_file.partition, Struct::empty());
        assert_eq!(data_file.sort_order_id(), None);
        assert_eq!(
            data_file.referenced_data_file(),
            Some("data_1.parquet".to_string())
        );

        // check written content
        let input_file = file_io.new_input(data_file.file_path.clone())?;
        let input_content = input_file.read().await?;
        let reader_builder = ParquetRecordBatchReaderBuilder::try_new(input_content)?;
        let metadata = reader_builder.metadata().clone();
        let field_ids: Vec<i32> = metadata
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .map(|col| col.self_type().get_basic_info().id())
            .collect();
        assert_eq!(field_ids, vec![
            RESERVED_FIELD_ID_DELETE_FILE_PATH,
            RESERVED_FIELD_ID_DELETE_FILE_POS
        ]);

        let reader = reader_builder.build()?;
        let batches = reader.map(|b| b.unwrap()).collect_vec();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], batch);

        Ok(())
    }

    #[tokio::test]
    async fn test_position_delete_writer_with_partition() -> Result<()> {
        let (_temp_dir, _file_io, rolling_writer_builder) =
            setup("test_position_delete_partitioned");

        let schema = Arc::new(position_delete_schema());
        let partition_value = Struct::from_iter([Some(Literal::string("US"))]);
        let partition_spec = Arc::new(PartitionSpec::builder(schema.clone()).build()?);
        let partition_key = PartitionKey::new(
            partition_spec.as_ref().clone(),
            schema,
            partition_value.clone(),
        );

        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_writer_builder)
            .build(Some(partition_key))
            .await?;

        writer
            .write(position_delete_batch(vec!["data_1.parquet"], vec![0]))
            .await?;
        let data_files = writer.close().await?;
        assert_eq!(data_files.len(), 1);
        assert_eq!(data_files[0].partition, partition_value);
        assert_eq!(data_files[0].partition_spec_id, partition_spec.spec_id());

        Ok(())
    }

    #[tokio::test]
    async fn test_position_delete_writer_multiple_referenced_files() -> Result<()> {
        let (_temp_dir, _file_io, rolling_writer_builder) =
            setup("test_position_delete_multi_file");

        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_writer_builder)
            .build(None)
            .await?;

        writer
            .write(position_delete_batch(
                vec!["data_1.parquet", "data_2.parquet"],
                vec![0, 0],
            ))
            .await?;
        let data_files = writer.close().await?;
        assert_eq!(data_files.len(), 1);
        assert_eq!(data_files[0].referenced_data_file(), None);

        Ok(())
    }

    #[tokio::test]
    async fn test_position_delete_writer_empty_close_error() -> Result<()> {
        let (_temp_dir, _file_io, rolling_writer_builder) = setup("test_position_delete_closed");

        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_writer_builder)
            .build(None)
            .await?;
        writer
            .write(position_delete_batch(vec!["data_1.parquet"], vec![0]))
            .await?;
        let _ = writer.close().await?;

        let err = writer
            .write(position_delete_batch(vec!["data_1.parquet"], vec![1]))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);

        Ok(())
    }
}

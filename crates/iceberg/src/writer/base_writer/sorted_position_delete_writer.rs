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

//! This module provide `SortedPositionDeleteWriter`.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};

use crate::arrow::schema_to_arrow_schema;
use crate::spec::{DataFile, PartitionKey};
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, position_delete_schema,
};
use crate::writer::file_writer::FileWriterBuilder;
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Config for `SortedPositionDeleteWriter`.
#[derive(Debug, Clone)]
pub struct SortedPositionDeleteWriterConfig {
    /// Number of buffered delete rows after which a sorted run is flushed to a new file.
    max_records_per_file: usize,
}

impl SortedPositionDeleteWriterConfig {
    /// Create a new `SortedPositionDeleteWriterConfig`.
    ///
    /// `max_records_per_file` bounds the number of `(file_path, pos)` rows buffered in memory
    /// before they are sorted and spilled to a new position delete file. Iceberg allows a
    /// position delete to be split across multiple files as long as each file is individually
    /// sorted by `(file_path, pos)`, so this bound trades file count for memory.
    pub fn new(max_records_per_file: usize) -> Result<Self> {
        if max_records_per_file == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "max_records_per_file must be greater than zero.",
            ));
        }
        Ok(Self {
            max_records_per_file,
        })
    }
}

/// Builder for `SortedPositionDeleteWriter`.
#[derive(Debug)]
pub struct SortedPositionDeleteWriterBuilder<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    inner: RollingFileWriterBuilder<B, L, F>,
    config: SortedPositionDeleteWriterConfig,
}

impl<B, L, F> SortedPositionDeleteWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Create a new `SortedPositionDeleteWriterBuilder` using a `RollingFileWriterBuilder`.
    ///
    /// The `RollingFileWriterBuilder` must be constructed with a file writer configured for
    /// [`position_delete_schema`].
    pub fn new(
        inner: RollingFileWriterBuilder<B, L, F>,
        config: SortedPositionDeleteWriterConfig,
    ) -> Self {
        Self { inner, config }
    }
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriterBuilder for SortedPositionDeleteWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    type R = SortedPositionDeleteWriter<B, L, F>;

    async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R> {
        Ok(SortedPositionDeleteWriter {
            inner: self.inner.clone(),
            partition_key,
            max_records_per_file: self.config.max_records_per_file,
            buffer: Vec::new(),
            data_files: Vec::new(),
            closed: false,
        })
    }
}

/// Writer that buffers `(file_path, pos)` position deletes, sorts them, and spills sorted runs
/// to files written through the wrapped [`PositionDeleteFileWriterBuilder`].
///
/// Position delete files must be sorted by `(file_path, pos)` within each file. This writer
/// buffers rows in memory up to `max_records_per_file`, sorts the buffer, and writes it out as
/// one file; if more rows follow, a new sorted file is started. Iceberg permits a position
/// delete to span multiple files this way, since each file only needs to be sorted internally.
#[derive(Debug)]
pub struct SortedPositionDeleteWriter<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    inner: RollingFileWriterBuilder<B, L, F>,
    partition_key: Option<PartitionKey>,
    max_records_per_file: usize,
    buffer: Vec<(String, i64)>,
    data_files: Vec<DataFile>,
    closed: bool,
}

impl<B, L, F> SortedPositionDeleteWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Sort the buffered rows and write them out as a single new position delete file.
    async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let mut rows = std::mem::take(&mut self.buffer);
        rows.sort_unstable();

        let arrow_schema = Arc::new(schema_to_arrow_schema(&position_delete_schema())?);
        let file_paths = StringArray::from(
            rows.iter()
                .map(|(path, _)| path.as_str())
                .collect::<Vec<_>>(),
        );
        let positions = Int64Array::from(rows.iter().map(|(_, pos)| *pos).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(file_paths),
            Arc::new(positions),
        ])
        .map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to build position delete batch: {e}"),
            )
        })?;

        let writer_builder = PositionDeleteFileWriterBuilder::new(self.inner.clone());
        let mut writer = writer_builder.build(self.partition_key.clone()).await?;
        writer.write(batch).await?;
        self.data_files.extend(writer.close().await?);

        Ok(())
    }

    /// Buffer one batch of `(file_path, pos)` rows, flushing a sorted run if the buffer has
    /// grown past `max_records_per_file`.
    fn buffer_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let file_paths = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Could not downcast file_path column to StringArray",
                )
            })?;
        let positions = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Could not downcast pos column to Int64Array",
                )
            })?;

        self.buffer.reserve(batch.num_rows());
        for (file_path, pos) in file_paths.iter().zip(positions.iter()) {
            let (Some(file_path), Some(pos)) = (file_path, pos) else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "file_path and pos must not be null in position delete rows",
                ));
            };
            self.buffer.push((file_path.to_string(), pos));
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriter for SortedPositionDeleteWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        if self.closed {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Sorted position delete writer has been closed.",
            ));
        }

        self.buffer_batch(&batch)?;

        if self.buffer.len() >= self.max_records_per_file {
            self.flush().await?;
        }

        Ok(())
    }

    async fn close(&mut self) -> Result<Vec<DataFile>> {
        if self.closed {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Sorted position delete writer has been closed.",
            ));
        }
        self.closed = true;

        self.flush().await?;

        Ok(std::mem::take(&mut self.data_files))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::io::FileIO;
    use crate::spec::{DataContentType, DataFileFormat, Struct};
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };

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

    async fn read_all_rows(file_io: &FileIO, data_file: &DataFile) -> Vec<(String, i64)> {
        let input_file = file_io.new_input(data_file.file_path.clone()).unwrap();
        let input_content = input_file.read().await.unwrap();
        let reader_builder = ParquetRecordBatchReaderBuilder::try_new(input_content).unwrap();
        let reader = reader_builder.build().unwrap();
        let mut rows = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let file_paths = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let positions = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for (path, pos) in file_paths.iter().zip(positions.iter()) {
                rows.push((path.unwrap().to_string(), pos.unwrap()));
            }
        }
        rows
    }

    #[tokio::test]
    async fn test_sorted_position_delete_writer_sorts_within_one_file() -> Result<()> {
        let (_temp_dir, file_io, rolling_writer_builder) = setup("test_sorted_pos_delete");
        let config = SortedPositionDeleteWriterConfig::new(1024)?;

        let mut writer = SortedPositionDeleteWriterBuilder::new(rolling_writer_builder, config)
            .build(None)
            .await?;

        writer
            .write(position_delete_batch(
                vec!["b.parquet", "a.parquet", "a.parquet"],
                vec![5, 10, 1],
            ))
            .await?;
        let data_files = writer.close().await?;
        assert_eq!(data_files.len(), 1);
        assert_eq!(
            data_files[0].content_type(),
            DataContentType::PositionDeletes
        );
        assert_eq!(data_files[0].partition, Struct::empty());

        let rows = read_all_rows(&file_io, &data_files[0]).await;
        assert_eq!(rows, vec![
            ("a.parquet".to_string(), 1),
            ("a.parquet".to_string(), 10),
            ("b.parquet".to_string(), 5),
        ]);

        Ok(())
    }

    #[tokio::test]
    async fn test_sorted_position_delete_writer_flushes_on_threshold() -> Result<()> {
        let (_temp_dir, file_io, rolling_writer_builder) = setup("test_sorted_pos_delete_flush");
        let config = SortedPositionDeleteWriterConfig::new(2)?;

        let mut writer = SortedPositionDeleteWriterBuilder::new(rolling_writer_builder, config)
            .build(None)
            .await?;

        // First batch hits the threshold exactly and should flush immediately.
        writer
            .write(position_delete_batch(vec!["a.parquet", "a.parquet"], vec![
                2, 1,
            ]))
            .await?;
        // Second batch is flushed on close as a second, separate sorted file.
        writer
            .write(position_delete_batch(vec!["a.parquet"], vec![3]))
            .await?;
        let data_files = writer.close().await?;
        assert_eq!(data_files.len(), 2);

        let mut first_file_rows = read_all_rows(&file_io, &data_files[0]).await;
        first_file_rows.sort();
        assert_eq!(first_file_rows, vec![
            ("a.parquet".to_string(), 1),
            ("a.parquet".to_string(), 2)
        ]);
        let second_file_rows = read_all_rows(&file_io, &data_files[1]).await;
        assert_eq!(second_file_rows, vec![("a.parquet".to_string(), 3)]);

        Ok(())
    }

    #[tokio::test]
    async fn test_sorted_position_delete_writer_empty_close() -> Result<()> {
        let (_temp_dir, _file_io, rolling_writer_builder) = setup("test_sorted_pos_delete_empty");
        let config = SortedPositionDeleteWriterConfig::new(1024)?;

        let mut writer = SortedPositionDeleteWriterBuilder::new(rolling_writer_builder, config)
            .build(None)
            .await?;
        let data_files = writer.close().await?;
        assert!(data_files.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_sorted_position_delete_writer_rejects_reuse_after_close() -> Result<()> {
        let (_temp_dir, _file_io, rolling_writer_builder) = setup("test_sorted_pos_delete_closed");
        let config = SortedPositionDeleteWriterConfig::new(1024)?;

        let mut writer = SortedPositionDeleteWriterBuilder::new(rolling_writer_builder, config)
            .build(None)
            .await?;
        writer
            .write(position_delete_batch(vec!["a.parquet"], vec![0]))
            .await?;
        let _ = writer.close().await?;

        let err = writer
            .write(position_delete_batch(vec!["a.parquet"], vec![1]))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);

        let err = writer.close().await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);

        Ok(())
    }

    #[test]
    fn test_sorted_position_delete_writer_config_rejects_zero() {
        let err = SortedPositionDeleteWriterConfig::new(0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    #[tokio::test]
    async fn test_sorted_position_delete_writer_with_partition() -> Result<()> {
        use crate::spec::{Literal, PartitionSpec, Schema};

        let (_temp_dir, _file_io, rolling_writer_builder) =
            setup("test_sorted_pos_delete_partitioned");
        let config = SortedPositionDeleteWriterConfig::new(1024)?;

        let schema = Arc::new(Schema::builder().build().unwrap());
        let partition_value = Struct::from_iter([Some(Literal::string("US"))]);
        let partition_spec = Arc::new(PartitionSpec::builder(schema.clone()).build()?);
        let partition_key = PartitionKey::new(
            partition_spec.as_ref().clone(),
            schema,
            partition_value.clone(),
        );

        let mut writer = SortedPositionDeleteWriterBuilder::new(rolling_writer_builder, config)
            .build(Some(partition_key))
            .await?;
        writer
            .write(position_delete_batch(vec!["a.parquet"], vec![0]))
            .await?;
        let data_files = writer.close().await?;
        assert_eq!(data_files.len(), 1);
        assert_eq!(data_files[0].partition, partition_value);

        Ok(())
    }

    /// End-to-end check that multiple sorted-run files produced by this writer are correctly
    /// unioned by iceberg-rust's own read path (`CachingDeleteFileLoader`). A low threshold
    /// forces the same data file's deletes to be split across three separate delete files,
    /// each fed with unsorted rows; the reader must still recover every position.
    #[tokio::test]
    async fn test_sorted_position_delete_writer_output_round_trips_through_reader() -> Result<()> {
        use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
        use crate::runtime::Runtime;
        use crate::scan::{FileScanTask, FileScanTaskDeleteFile};
        use crate::spec::Schema;

        let (temp_dir, file_io, rolling_writer_builder) = setup("test_sorted_pos_delete_roundtrip");
        let config = SortedPositionDeleteWriterConfig::new(2)?;
        let mut writer = SortedPositionDeleteWriterBuilder::new(rolling_writer_builder, config)
            .build(None)
            .await?;

        let data_file_path = format!("{}/data-1.parquet", temp_dir.path().to_str().unwrap());
        // Threshold of 2: each of these two writes flushes immediately as its own sorted
        // run; the third row stays buffered until close() flushes a third file.
        writer
            .write(position_delete_batch(
                vec![&data_file_path, &data_file_path],
                vec![5, 2],
            ))
            .await?;
        writer
            .write(position_delete_batch(
                vec![&data_file_path, &data_file_path],
                vec![9, 1],
            ))
            .await?;
        writer
            .write(position_delete_batch(vec![&data_file_path], vec![7]))
            .await?;
        let data_files = writer.close().await?;
        assert_eq!(data_files.len(), 3, "expected three separate sorted runs");

        let delete_tasks = data_files
            .iter()
            .map(|f| {
                FileScanTaskDeleteFile::builder()
                    .with_file_path(f.file_path().to_string())
                    .with_file_size_in_bytes(f.file_size_in_bytes())
                    .with_file_type(DataContentType::PositionDeletes)
                    .with_partition_spec_id(0)
                    .build()
            })
            .collect::<Vec<_>>();

        let scan_task = FileScanTask::builder()
            .with_file_size_in_bytes(0)
            .with_start(0)
            .with_length(0)
            .with_data_file_path(data_file_path)
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(Arc::new(Schema::builder().build()?))
            .with_project_field_ids(vec![])
            .with_deletes(delete_tasks)
            .with_case_sensitive(false)
            .build();

        let delete_file_loader = CachingDeleteFileLoader::new(file_io, 10, Runtime::current());
        let delete_filter = delete_file_loader
            .load_deletes(&scan_task.deletes, scan_task.schema_ref())
            .await
            .unwrap()?;
        let delete_vector = delete_filter.get_delete_vector(&scan_task).unwrap();
        let mut positions: Vec<u64> = delete_vector.lock().unwrap().iter().collect();
        positions.sort_unstable();
        assert_eq!(positions, vec![1, 2, 5, 7, 9]);

        Ok(())
    }
}

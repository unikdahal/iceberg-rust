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

//! A sorting position-delete writer for unordered row-level delete input.
//!
//! The writer retains one entry per unique (file_path, position) pair, then emits rows in
//! lexical path order and ascending position order. Its memory use is O(unique paths + unique
//! positions); spilling is intentionally left to a future implementation.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::builder::{Int64Builder, StringBuilder};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::SchemaRef as ArrowSchemaRef;
use roaring::RoaringTreemap;

use crate::arrow::schema_to_arrow_schema;
use crate::error::invalid_data;
use crate::spec::{DataFile, PartitionKey};
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriter, PositionDeleteFileWriterBuilder, position_delete_schema,
    validate_position_delete_batch,
};
use crate::writer::file_writer::FileWriterBuilder;
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Number of sorted position-delete rows sent to the underlying writer per Arrow batch.
const DEFAULT_FLUSH_ROWS: usize = 8192;

/// Builder for SortingPositionOnlyDeleteWriter.
pub struct SortingPositionOnlyDeleteWriterBuilder<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    inner: RollingFileWriterBuilder<B, L, F>,
    flush_rows: usize,
}

impl<B, L, F> SortingPositionOnlyDeleteWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Wraps a rolling position-delete writer with sorting and de-duplication.
    ///
    /// The rolling writer must use the schema returned by position_delete_schema.
    pub fn new(inner: RollingFileWriterBuilder<B, L, F>) -> Self {
        Self {
            inner,
            flush_rows: DEFAULT_FLUSH_ROWS,
        }
    }

    /// Sets the maximum number of sorted position-delete rows sent per Arrow batch.
    ///
    /// This only controls buffering at the Arrow boundary; it does not change file rolling or
    /// duplicate semantics.
    pub(crate) fn with_flush_rows(mut self, flush_rows: usize) -> Self {
        self.flush_rows = flush_rows;
        self
    }
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriterBuilder for SortingPositionOnlyDeleteWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    type R = SortingPositionOnlyDeleteWriter<B, L, F>;

    async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R> {
        if self.flush_rows == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Sorting position-only delete writer flush_rows must be greater than zero.",
            ));
        }
        let inner = PositionDeleteFileWriterBuilder::new(self.inner.clone())
            .build(partition_key)
            .await?;
        Ok(SortingPositionOnlyDeleteWriter {
            inner: Some(inner),
            positions: BTreeMap::new(),
            flush_rows: self.flush_rows,
            closed: false,
        })
    }
}

/// Buffers unordered position deletes, then writes sorted and de-duplicated delete records.
///
/// Repeated `(file_path, position)` pairs are idempotent and are emitted once, matching the
/// bitmap-backed behavior of Iceberg-Java's sorting position-delete writer.
///
/// The in-memory index is O(unique paths + unique positions). Each emitted Arrow batch is bounded
/// to `flush_rows` records, while the path map retains all unique positions until close so
/// duplicates are removed even when they arrive in different batches.
pub struct SortingPositionOnlyDeleteWriter<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    inner: Option<PositionDeleteFileWriter<B, L, F>>,
    positions: BTreeMap<String, RoaringTreemap>,
    flush_rows: usize,
    closed: bool,
}

impl<B, L, F> SortingPositionOnlyDeleteWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Adds one delete record.
    ///
    /// Negative row positions are invalid. Re-adding the same path and position is a no-op.
    pub fn write_delete(&mut self, file_path: impl Into<String>, position: i64) -> Result<()> {
        self.ensure_open()?;
        if position < 0 {
            return Err(invalid_data!(
                "Position delete row position must be non-negative, got {position}."
            ));
        }
        self.positions
            .entry(file_path.into())
            .or_default()
            .insert(position as u64);
        Ok(())
    }

    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Sorting position-only delete writer is already closed.",
            ));
        }
        Ok(())
    }

    async fn flush_batch(
        inner: &mut PositionDeleteFileWriter<B, L, F>,
        schema: &ArrowSchemaRef,
        paths: &mut StringBuilder,
        positions: &mut Int64Builder,
        buffered_rows: usize,
    ) -> Result<()> {
        if buffered_rows == 0 {
            return Ok(());
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(paths.finish()) as ArrayRef,
                Arc::new(positions.finish()) as ArrayRef,
            ],
        )
        .map_err(|e| invalid_data!("Failed to build position-delete batch: {e}"))?;
        inner.write(batch).await
    }
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriter for SortingPositionOnlyDeleteWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        self.ensure_open()?;
        validate_position_delete_batch(&batch)?;
        let paths = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| invalid_data!("Position-delete file_path must be a StringArray."))?;
        let positions = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| invalid_data!("Position-delete pos must be an Int64Array."))?;
        if paths.null_count() != 0 || positions.null_count() != 0 {
            return Err(invalid_data!(
                "Position-delete file_path and pos values must not be null."
            ));
        }
        if positions.values().iter().any(|position| *position < 0) {
            return Err(invalid_data!(
                "Position-delete row positions must be non-negative."
            ));
        }

        // Validate the complete batch before changing writer state.
        for index in 0..batch.num_rows() {
            let path = paths.value(index);
            let position = positions.value(index) as u64;
            if let Some(path_positions) = self.positions.get_mut(path) {
                path_positions.insert(position);
            } else {
                let mut path_positions = RoaringTreemap::new();
                path_positions.insert(position);
                self.positions.insert(path.to_owned(), path_positions);
            }
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<Vec<DataFile>> {
        self.ensure_open()?;
        let schema = Arc::new(schema_to_arrow_schema(&position_delete_schema())?);
        self.closed = true;
        let positions_by_path = std::mem::take(&mut self.positions);
        let flush_rows = self.flush_rows;
        let mut inner = self.inner.take().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Sorting position-only delete writer is already closed.",
            )
        })?;

        let write_result = async {
            let mut paths = StringBuilder::new();
            let mut positions = Int64Builder::new();
            let mut buffered_rows = 0usize;

            // BTreeMap<String, _> yields the same scalar-value path ordering used by
            // Iceberg-Java's Comparators.charSequences(). Each bitmap yields positions in
            // ascending order, so the output is already sorted by (file_path, pos).
            for (path, path_positions) in positions_by_path {
                let mut path_positions = path_positions.iter().peekable();
                while path_positions.peek().is_some() {
                    let remaining = flush_rows - buffered_rows;
                    let mut appended = 0usize;
                    while appended < remaining {
                        let Some(position) = path_positions.next() else {
                            break;
                        };
                        positions.append_value(i64::try_from(position).map_err(|_| {
                            invalid_data!("Position delete row position exceeds i64::MAX.")
                        })?);
                        appended += 1;
                    }
                    paths.append_value_n(path.as_str(), appended);
                    buffered_rows += appended;

                    if buffered_rows == flush_rows {
                        Self::flush_batch(
                            &mut inner,
                            &schema,
                            &mut paths,
                            &mut positions,
                            buffered_rows,
                        )
                        .await?;
                        buffered_rows = 0;
                    }
                }
            }

            Self::flush_batch(
                &mut inner,
                &schema,
                &mut paths,
                &mut positions,
                buffered_rows,
            )
            .await
        }
        .await;

        if let Err(err) = write_result {
            // Finalize any underlying file handle even when materializing/writing sorted
            // deletes fails. The partially written output must not be returned to the caller.
            let _ = inner.close().await;
            return Err(err);
        }

        inner.close().await
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::delete_file_loader::PositionDeleteIndexLoader;
    use crate::arrow::schema_to_arrow_schema;
    use crate::io::FileIO;
    use crate::scan::FileScanTaskDeleteFile;
    use crate::spec::{
        DataContentType, DataFileFormat, Manifest, ManifestWriterBuilder, NestedField,
        PartitionSpec, PrimitiveType, Schema, Type,
    };
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;

    fn setup(
        file_prefix: &str,
        target_file_size: usize,
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
        let location_generator = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_generator =
            DefaultFileNameGenerator::new(file_prefix.to_string(), None, DataFileFormat::Parquet);
        let parquet_writer = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            position_delete_schema(),
        );
        let rolling_writer = RollingFileWriterBuilder::new(
            parquet_writer,
            target_file_size,
            file_io.clone(),
            location_generator,
            file_name_generator,
        );
        (temp_dir, file_io, rolling_writer)
    }

    fn delete_batch(paths: Vec<&str>, row_positions: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(schema_to_arrow_schema(&position_delete_schema()).unwrap());
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(paths)),
                Arc::new(Int64Array::from(row_positions)),
            ],
        )
        .unwrap()
    }

    async fn read_rows(file_io: &FileIO, file: &DataFile) -> Vec<(String, i64)> {
        let input = file_io
            .new_input(file.file_path.clone())
            .unwrap()
            .read()
            .await
            .unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(input)
            .unwrap()
            .build()
            .unwrap();
        let mut rows = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let paths = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let positions = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            rows.extend(
                paths
                    .iter()
                    .zip(positions.iter())
                    .map(|(path, position)| (path.unwrap().to_owned(), position.unwrap())),
            );
        }
        rows
    }

    #[tokio::test]
    async fn sorts_and_deduplicates_across_batches() -> Result<()> {
        let (_temp_dir, file_io, rolling_writer) = setup("sorted_pos_delete", usize::MAX);
        let mut writer =
            SortingPositionOnlyDeleteWriterBuilder::new(rolling_writer).build(None).await?;
        writer
            .write(delete_batch(vec!["b.parquet", "a.parquet"], vec![8, 9]))
            .await?;
        writer
            .write(delete_batch(vec!["a.parquet", "a.parquet", "b.parquet"], vec![
                2, 9, 1,
            ]))
            .await?;

        let files = writer.close().await?;
        assert_eq!(files.len(), 1);
        let rows = read_rows(&file_io, &files[0]).await;
        assert_eq!(rows, vec![
            ("a.parquet".to_string(), 2),
            ("a.parquet".to_string(), 9),
            ("b.parquet".to_string(), 1),
            ("b.parquet".to_string(), 8),
        ]);
        Ok(())
    }

    #[tokio::test]
    async fn writer_output_round_trips_through_file_scoped_loader() -> Result<()> {
        let (temp_dir, file_io, rolling_writer) = setup("roundtrip_pos_delete", usize::MAX);
        let mut writer =
            SortingPositionOnlyDeleteWriterBuilder::new(rolling_writer).build(None).await?;

        writer
            .write(delete_batch(
                vec!["data.parquet", "data.parquet", "data.parquet"],
                vec![5, 1, 5],
            ))
            .await?;

        let files = writer.close().await?;
        assert_eq!(files.len(), 1);
        let file = &files[0];
        assert_eq!(file.content_type(), DataContentType::PositionDeletes);
        assert_eq!(file.file_format(), DataFileFormat::Parquet);
        assert_eq!(file.record_count(), 2);

        // Carry the writer-produced DataFile through an actual V2 delete manifest so this
        // regression covers the same metadata serialization boundary used by table commits.
        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                )
                .into()])
                .build()?,
        );
        let partition_spec = PartitionSpec::builder(table_schema.clone())
            .with_spec_id(0)
            .build()?;
        let manifest_path = temp_dir.path().join("roundtrip-delete-manifest.avro");
        let output = file_io.new_output(manifest_path.to_str().unwrap())?;
        let mut manifest_writer =
            ManifestWriterBuilder::new(output, Some(1), table_schema, partition_spec)
                .build_v2_deletes();
        manifest_writer.add_file(file.clone(), 0)?;
        manifest_writer.write_manifest_file().await?;

        let manifest = Manifest::parse_avro(&fs::read(manifest_path).unwrap())?;
        assert_eq!(manifest.entries().len(), 1);
        let manifest_file = manifest.entries()[0].data_file();
        assert_eq!(
            manifest_file.content_type(),
            DataContentType::PositionDeletes
        );
        assert_eq!(manifest_file.record_count(), 2);

        let task = FileScanTaskDeleteFile::builder()
            .with_file_path(manifest_file.file_path().to_string())
            .with_file_size_in_bytes(manifest_file.file_size_in_bytes())
            .with_file_type(manifest_file.content_type())
            .with_file_format(manifest_file.file_format())
            .with_partition_spec_id(0)
            .with_equality_ids(manifest_file.equality_ids())
            .with_referenced_data_file(manifest_file.referenced_data_file())
            .with_content_offset(manifest_file.content_offset())
            .with_content_size_in_bytes(manifest_file.content_size_in_bytes())
            .with_record_count(Some(manifest_file.record_count()))
            .with_key_metadata(manifest_file.key_metadata().map(Box::from))
            .build();

        let index = PositionDeleteIndexLoader::new(file_io)
            .load_file_scoped_positions(&task, "data.parquet")
            .await?;

        assert_eq!(index.iter().collect::<Vec<_>>(), vec![1, 5]);
        Ok(())
    }

    #[tokio::test]
    async fn close_without_rows_returns_no_files() -> Result<()> {
        let (_temp_dir, _file_io, rolling_writer) = setup("empty_pos_delete", usize::MAX);
        let mut writer =
            SortingPositionOnlyDeleteWriterBuilder::new(rolling_writer).build(None).await?;
        assert!(writer.close().await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn rejects_negative_positions_before_mutating_the_batch() -> Result<()> {
        let (_temp_dir, _file_io, rolling_writer) = setup("negative_pos_delete", usize::MAX);
        let mut writer =
            SortingPositionOnlyDeleteWriterBuilder::new(rolling_writer).build(None).await?;
        assert!(writer
            .write(delete_batch(vec!["a.parquet", "b.parquet"], vec![1, -1]))
            .await
            .is_err());
        assert!(writer.close().await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn rolling_writer_metadata_is_returned_for_every_sorted_batch() -> Result<()> {
        const FLUSH_ROWS: usize = 64;
        let (_temp_dir, file_io, rolling_writer) = setup("rolling_pos_delete", 1);
        let mut writer = SortingPositionOnlyDeleteWriterBuilder::new(rolling_writer)
            .with_flush_rows(FLUSH_ROWS)
            .build(None)
            .await?;
        for position in 0..(FLUSH_ROWS as i64 + 1) {
            writer.write_delete("a.parquet", position)?;
        }

        let files = writer.close().await?;
        assert!(files.len() >= 2);
        let mut rows = Vec::new();
        for file in &files {
            rows.extend(read_rows(&file_io, file).await);
        }
        assert_eq!(rows.len(), FLUSH_ROWS + 1);
        assert!(rows.windows(2).all(|pair| pair[0].1 < pair[1].1));
        Ok(())
    }

    #[tokio::test]
    async fn matches_iceberg_java_char_sequence_order_and_accepts_large_positions() -> Result<()> {
        let (_temp_dir, file_io, rolling_writer) = setup("lexical_pos_delete", usize::MAX);
        let mut writer =
            SortingPositionOnlyDeleteWriterBuilder::new(rolling_writer).build(None).await?;
        writer.write_delete("\u{e000}.parquet", 0)?;
        writer.write_delete("\u{10000}.parquet", i64::MAX)?;

        let files = writer.close().await?;
        let rows = read_rows(&file_io, &files[0]).await;
        assert_eq!(rows, vec![
            ("\u{e000}.parquet".to_string(), 0),
            ("\u{10000}.parquet".to_string(), i64::MAX),
        ]);
        Ok(())
    }

    #[tokio::test]
    async fn rejects_zero_flush_rows() -> Result<()> {
        let (_temp_dir, _file_io, rolling_writer) = setup("zero_flush_rows", usize::MAX);
        let err = SortingPositionOnlyDeleteWriterBuilder::new(rolling_writer)
            .with_flush_rows(0)
            .build(None)
            .await
            .err()
            .expect("expected invalid flush_rows error");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        Ok(())
    }

    #[tokio::test]
    async fn close_is_one_shot() -> Result<()> {
        let (_temp_dir, _file_io, rolling_writer) = setup("one_shot_pos_delete", usize::MAX);
        let mut writer =
            SortingPositionOnlyDeleteWriterBuilder::new(rolling_writer).build(None).await?;
        assert!(writer.close().await?.is_empty());
        assert!(writer.close().await.is_err());
        Ok(())
    }
}

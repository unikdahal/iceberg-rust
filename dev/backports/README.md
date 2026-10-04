# Parquet row-group-local selection backport

This patch carries the generic Arrow 60 row-group-local selection API onto
Arrow/Parquet 59.3.0, based on Arrow commit
`f90e061326bd821a7af09281d9e92de6f3b603d9`.
DataFusion 55.1 requires Arrow 59.2, so upgrading only Parquet to 60 would
introduce incompatible Arrow types.

The patch adds `RowGroupSelection`, builder configuration, and an independent
selection queue for each row group. Existing global selections keep their
current semantics. The existing push decoder, its buffered ranges, projection,
row filter, and remaining offset/limit budget are retained when rebuilding at
a row-group boundary. Codec and Arrow 60 page-index changes are excluded.

Relevant Arrow 60 tests cover local offsets, reordered and duplicate row
groups, bitmap and RLE selections, empty selections, row filters, offset/limit,
invalid group indices and lengths, conflicting builder options, and rebuilding
with unconsumed local selections.

Parquet's manifest uses the registry dependencies from the published 59.3.0
crate. This prevents a git-patched Parquet dependency from pulling a second
copy of the Arrow types used by DataFusion and Iceberg.

The fork-only validation workflow applies the patch to an exact read-only
Arrow checkout and runs decoder and reader regression tests. Publishing the
dependency itself requires an accessible `unikdahal/arrow-rs` fork. This branch
does not change Iceberg's production dependency or enable selection-bearing
live scans yet.

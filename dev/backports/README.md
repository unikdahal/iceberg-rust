<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Parquet row-group-local selection backport

The pinned Arrow fork carries the generic Arrow 60 row-group-local selection API onto
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

The production reader now pins the tested backport from
`unikdahal/arrow-rs` at `fa20c8b77ff3d613c8b638f4f686f5316eb0138a`.
The root Cargo patch is required in every consuming workspace: Cargo patches
are not transitive. The backport source is maintained once in that Arrow fork; this branch does
not duplicate its patch text. The local-selection feature is enabled by default;
`--no-default-features` still tests the flattened-selection fallback.

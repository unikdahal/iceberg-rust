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

The pinned Arrow fork carries the Arrow 60 adaptive push-decoder API onto
Arrow/Parquet 59.3.0: `RowGroupSelection`, `with_row_group_selections`,
`is_at_row_group_boundary`, `row_groups_remaining`, `peek_next_row_group`,
`clear_all_ranges` and `into_builder`. Its public surface matches Arrow 60.0.0.
DataFusion 55.1 requires Arrow 59.2, so upgrading only Parquet to 60 would
introduce incompatible Arrow types.

The reader's live runtime pruning requires this API unconditionally. On a workspace
that uses Arrow 60, delete the root `[patch.crates-io]` entry and this file; no
source changes are needed.

Existing global selections keep their current semantics. Rebuilding at a
row-group boundary retains the push decoder's buffered ranges, projection,
row filter, and remaining offset/limit budget. Codec and Arrow 60 page-index
changes are excluded.

Parquet's manifest uses the registry dependencies from the published 59.3.0
crate. This prevents a git-patched Parquet dependency from pulling a second
copy of the Arrow types used by DataFusion and Iceberg.

The production reader pins the tested backport from `unikdahal/arrow-rs` at
`fa20c8b77ff3d613c8b638f4f686f5316eb0138a`. Cargo patches are not transitive,
so every consuming workspace must carry the same root patch.

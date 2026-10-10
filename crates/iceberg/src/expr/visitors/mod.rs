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

pub(crate) mod bloom_filter_evaluator;
pub(crate) mod bound_predicate_visitor;
pub(crate) mod expression_evaluator;
pub(crate) mod inclusive_metrics_evaluator;
pub(crate) mod inclusive_projection;
pub(crate) mod manifest_evaluator;
pub(crate) mod page_index_evaluator;
pub(crate) mod predicate_visitor;
pub(crate) mod rewrite_not;
pub(crate) mod row_group_metrics_evaluator;
pub(crate) mod strict_metrics_evaluator;
pub(crate) mod strict_projection;

use std::collections::HashMap;

use crate::spec::{DataFile, Datum};

/// Borrowed whole-file column statistics, keyed by Iceberg field ID.
///
/// The statistics evaluator only reads these maps, so callers that hold them
/// outside a [`DataFile`] can be evaluated without building one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FileMetrics<'a> {
    /// Number of records in the file, when known.
    pub(crate) record_count: Option<u64>,
    /// Number of values, including nulls and NaNs.
    pub(crate) value_counts: &'a HashMap<i32, u64>,
    /// Number of null values.
    pub(crate) null_value_counts: &'a HashMap<i32, u64>,
    /// Number of NaN values.
    pub(crate) nan_value_counts: &'a HashMap<i32, u64>,
    /// Inclusive lower bounds.
    pub(crate) lower_bounds: &'a HashMap<i32, Datum>,
    /// Inclusive upper bounds.
    pub(crate) upper_bounds: &'a HashMap<i32, Datum>,
}

impl<'a> From<&'a DataFile> for FileMetrics<'a> {
    fn from(data_file: &'a DataFile) -> Self {
        Self {
            record_count: Some(data_file.record_count),
            value_counts: &data_file.value_counts,
            null_value_counts: &data_file.null_value_counts,
            nan_value_counts: &data_file.nan_value_counts,
            lower_bounds: &data_file.lower_bounds,
            upper_bounds: &data_file.upper_bounds,
        }
    }
}

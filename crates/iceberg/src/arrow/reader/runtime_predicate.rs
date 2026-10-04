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

//! Execution-time predicates published to the Arrow reader.

use std::sync::{Arc, Mutex};

use arrow_schema::SchemaRef as ArrowSchemaRef;
use parquet::arrow::arrow_reader::RowSelection;
use parquet::basic::Type as PhysicalType;
use parquet::schema::types::SchemaDescriptor;

use super::ArrowReader;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::spec::{PrimitiveType, Schema, SchemaRef};
use crate::{Error, ErrorKind, Result};

/// An immutable view of a runtime predicate at one point in execution.
#[derive(Debug)]
pub struct RuntimePredicateSnapshot {
    predicate: Option<Predicate>,
    generation: u64,
}

impl RuntimePredicateSnapshot {
    /// Creates a runtime predicate snapshot. A `None` predicate adds no
    /// restriction and does not undo pruning already applied.
    pub fn new(predicate: Option<Predicate>, generation: u64) -> Self {
        Self {
            predicate,
            generation,
        }
    }

    /// Returns the publication generation this predicate belongs to.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Consumes the snapshot, returning its predicate without cloning it.
    pub fn into_predicate(self) -> Option<Predicate> {
        self.predicate
    }
}

/// Supplies execution-time predicates, such as a completed join build's key
/// range or a TopK/MIN/MAX bound, to the Arrow reader.
///
/// Contract:
/// * A publication must be safe to AND with the planned predicate for every
///   row read after it. Later generations may only tighten, or be `None`.
/// * Generations increase on every change and become visible only after
///   their predicate.
/// * Failures are advisory, never fail a scan, and need not be retried.
pub trait RuntimePredicateProvider: Send + Sync {
    /// Returns the current publication generation. Called for every task, so
    /// keep it cheap, for example an atomic load with acquire ordering.
    fn generation(&self) -> u64;

    /// Returns the current runtime predicate snapshot.
    fn snapshot(&self) -> Result<RuntimePredicateSnapshot>;
}

/// Caches a provider's bound predicate per generation, schema and case policy
/// across a scan's tasks. Failures are cached as `None` for their generation.
pub(super) struct RuntimePredicates {
    provider: Arc<dyn RuntimePredicateProvider>,
    cached: Mutex<Option<CachedPredicate>>,
}

struct CachedPredicate {
    generation: u64,
    schema: SchemaRef,
    case_sensitive: bool,
    predicate: Option<Arc<BoundPredicate>>,
}

impl CachedPredicate {
    fn matches(&self, generation: u64, schema: &SchemaRef, case_sensitive: bool) -> bool {
        self.generation == generation
            && self.case_sensitive == case_sensitive
            && (Arc::ptr_eq(&self.schema, schema) || *self.schema == **schema)
    }
}

impl RuntimePredicates {
    pub(super) fn new(provider: Arc<dyn RuntimePredicateProvider>) -> Self {
        Self {
            provider,
            cached: Mutex::new(None),
        }
    }

    /// Returns the current predicate bound to `schema`, with NOT pushed to the
    /// leaves because statistics evaluators cannot negate a "might match".
    pub(super) fn current(
        &self,
        schema: &SchemaRef,
        case_sensitive: bool,
        data_file_path: &str,
    ) -> Option<Arc<BoundPredicate>> {
        let generation = self.provider.generation();
        if let Some(cached) = self.cached.lock().unwrap().as_ref()
            && cached.matches(generation, schema, case_sensitive)
        {
            return cached.predicate.clone();
        }

        let (generation, predicate) = match self.provider.snapshot() {
            Ok(snapshot) => {
                let generation = snapshot.generation();
                let bound = snapshot
                    .into_predicate()
                    .map(|predicate| predicate.rewrite_not().bind(schema.clone(), case_sensitive))
                    .transpose();
                match bound {
                    Ok(bound) => (generation, bound.map(Arc::new)),
                    Err(error) => {
                        tracing::debug!("Skipping runtime predicate for {data_file_path}: {error}");
                        (generation, None)
                    }
                }
            }
            Err(error) => {
                tracing::debug!("Skipping runtime predicate for {data_file_path}: {error}");
                (generation, None)
            }
        };
        let result = predicate.clone();
        *self.cached.lock().unwrap() = Some(CachedPredicate {
            generation,
            schema: schema.clone(),
            case_sensitive,
            predicate,
        });
        result
    }
}

/// Fails if `predicate` references a column the file lacks (its default is
/// invisible to physical filters) or stores with a promoted type (a literal
/// cast to the narrower type can overflow and reject every row).
pub(super) fn check_runtime_predicate_columns(
    predicate: &BoundPredicate,
    parquet_schema: &SchemaDescriptor,
    arrow_schema: &ArrowSchemaRef,
    table_schema: &Schema,
    use_position_fallback: bool,
) -> Result<()> {
    let (field_ids, field_id_map) = ArrowReader::build_field_id_set_and_map(
        parquet_schema,
        arrow_schema,
        predicate,
        use_position_fallback,
    )?;
    for field_id in field_ids {
        let unusable = |reason: &str| {
            Error::new(
                ErrorKind::FeatureUnsupported,
                format!("Runtime predicate field {field_id} {reason}"),
            )
        };
        let column = *field_id_map
            .get(&field_id)
            .ok_or_else(|| unusable("is not stored in this file"))?;
        let field = table_schema
            .field_by_id(field_id)
            .ok_or_else(|| unusable("is not in the table schema"))?;
        let expected = field
            .field_type
            .as_primitive_type()
            .ok_or_else(|| unusable("is not a primitive column"))?;
        let descriptor = parquet_schema.column(column);
        let same_type = match expected {
            PrimitiveType::Int => descriptor.physical_type() == PhysicalType::INT32,
            PrimitiveType::Long => descriptor.physical_type() == PhysicalType::INT64,
            PrimitiveType::Float => descriptor.physical_type() == PhysicalType::FLOAT,
            PrimitiveType::Double => descriptor.physical_type() == PhysicalType::DOUBLE,
            PrimitiveType::Decimal { precision, scale } => {
                i64::from(descriptor.type_precision()) == i64::from(*precision)
                    && i64::from(descriptor.type_scale()) == i64::from(*scale)
            }
            _ => true,
        };
        if !same_type {
            return Err(unusable("is stored with a promoted physical type"));
        }
    }
    Ok(())
}

/// Intersects a predicate's page-index selection with `current`, where `None`
/// selects every row. An advisory predicate's failure leaves `current` as is.
pub(super) fn intersect_page_selection(
    current: Option<RowSelection>,
    selection: Result<Option<RowSelection>>,
    advisory: bool,
    data_file_path: &str,
) -> Result<Option<RowSelection>> {
    let selection = match selection {
        Ok(selection) => selection,
        Err(error) if advisory => {
            tracing::debug!("Skipping runtime page pruning for {data_file_path}: {error}");
            None
        }
        Err(error) => return Err(error),
    };
    Ok(match (current, selection) {
        (Some(current), Some(selection)) => Some(current.intersection(&selection)),
        (current, selection) => current.or(selection),
    })
}

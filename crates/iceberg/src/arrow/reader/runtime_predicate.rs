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

use arrow_schema::SchemaRef as ArrowSchemaRef;
use parquet::basic::Type as PhysicalType;
use parquet::schema::types::SchemaDescriptor;

use super::ArrowReader;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::spec::{PrimitiveType, Schema, SchemaRef};
use crate::{Error, ErrorKind, Result};

/// An immutable view of a runtime predicate at one point in execution.
#[derive(Clone, Debug)]
pub struct RuntimePredicateSnapshot {
    predicate: Option<Predicate>,
    generation: u64,
}

impl RuntimePredicateSnapshot {
    /// Creates a runtime predicate snapshot.
    pub fn new(predicate: Option<Predicate>, generation: u64) -> Self {
        Self {
            predicate,
            generation,
        }
    }

    /// Returns the predicate to apply, or `None` when no useful restriction is available.
    pub fn predicate(&self) -> Option<&Predicate> {
        self.predicate.as_ref()
    }

    /// Returns the publication generation this predicate belongs to.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Supplies execution-time predicates to an Arrow reader.
///
/// Execution engines learn restrictions while a query runs, for example the
/// key range of a completed hash-join build side, an improving TopK threshold
/// or a running MIN/MAX bound. A provider publishes such restrictions as
/// Iceberg predicates; the reader samples a snapshot when a data-file task
/// starts and may sample again later in the task.
///
/// # Contract
///
/// * Safety: a published predicate must be safe to AND with the task's planned
///   predicate for every row the scan reads *after* the publication, for the
///   rest of the scan. A reader may keep using an earlier snapshot after newer
///   generations are published, so a later generation may only be equally or
///   more restrictive than an earlier one, or `None`. Never publish a
///   provisional predicate that a later generation would need to widen.
///   `None` only stops applying an additional restriction to rows read later;
///   it never restores data a reader already skipped.
/// * Generations: generations must increase whenever the predicate changes,
///   including changes to or from `None`. Publish the predicate before its
///   generation becomes visible, and pair each snapshot's predicate with its
///   own publication generation.
/// * Failure: errors from [`Self::snapshot`] and predicates that cannot be bound
///   or applied to a file are advisory. The reader then keeps the planned and
///   delete predicates and does not fail the scan. A failed snapshot is cached
///   for its generation: the reader only tries again after the generation
///   changes, so report a transient failure by publishing a new generation.
///
/// # Example
///
/// ```
/// use std::sync::Mutex;
/// use std::sync::atomic::{AtomicU64, Ordering};
///
/// use iceberg::Result;
/// use iceberg::arrow::{RuntimePredicateProvider, RuntimePredicateSnapshot};
/// use iceberg::expr::{Predicate, Reference};
/// use iceberg::spec::Datum;
///
/// /// Publishes a lower bound that only ever tightens.
/// #[derive(Default)]
/// struct LowerBound {
///     generation: AtomicU64,
///     snapshot: Mutex<Option<RuntimePredicateSnapshot>>,
/// }
///
/// impl LowerBound {
///     fn publish(&self, bound: i64) {
///         let mut snapshot = self.snapshot.lock().unwrap();
///         let generation = self.generation.load(Ordering::Relaxed) + 1;
///         let predicate = Reference::new("id").greater_than_or_equal_to(Datum::long(bound));
///         // Publish the predicate before its generation becomes visible.
///         *snapshot = Some(RuntimePredicateSnapshot::new(Some(predicate), generation));
///         self.generation.store(generation, Ordering::Release);
///     }
/// }
///
/// impl RuntimePredicateProvider for LowerBound {
///     fn generation(&self) -> u64 {
///         self.generation.load(Ordering::Acquire)
///     }
///
///     fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
///         Ok(self
///             .snapshot
///             .lock()
///             .unwrap()
///             .clone()
///             .unwrap_or_else(|| RuntimePredicateSnapshot::new(None, 0)))
///     }
/// }
///
/// let provider = LowerBound::default();
/// assert!(provider.snapshot()?.predicate().is_none());
/// provider.publish(100);
/// let snapshot = provider.snapshot()?;
/// assert_eq!(snapshot.generation(), provider.generation());
/// assert!(snapshot.predicate().is_some());
/// # Ok::<(), iceberg::Error>(())
/// ```
pub trait RuntimePredicateProvider: Send + Sync {
    /// Returns the current publication generation without cloning the predicate.
    ///
    /// Readers call this on their hot path. Implementations should use a cheap
    /// synchronized read, such as an atomic load with acquire ordering.
    fn generation(&self) -> u64;

    /// Returns the current runtime predicate snapshot.
    fn snapshot(&self) -> Result<RuntimePredicateSnapshot>;
}

/// Samples and binds a provider's current predicate for one task.
///
/// Like planned filters, NOT is pushed down to the leaves before binding:
/// statistics evaluators cannot negate a "might match" result. Provider and
/// binding failures are advisory and yield `None`.
pub(super) fn sample_runtime_predicate(
    provider: &dyn RuntimePredicateProvider,
    schema: SchemaRef,
    case_sensitive: bool,
    data_file_path: &str,
) -> Option<BoundPredicate> {
    let bound = provider.snapshot().and_then(|snapshot| {
        snapshot
            .predicate()
            .map(|predicate| predicate.clone().rewrite_not().bind(schema, case_sensitive))
            .transpose()
    });
    match bound {
        Ok(predicate) => predicate,
        Err(error) => {
            tracing::debug!("Skipping runtime predicate for {data_file_path}: {error}");
            None
        }
    }
}

/// Checks that a file stores every column `predicate` references with the
/// table's own physical type, so that physical pruning and row filtering see
/// exactly the values the reader returns.
///
/// A column missing from the file is read as its initial default or null,
/// which physical filters never see. A promoted column (for example INT read
/// as BIGINT) would have literals cast to the narrower physical type, where an
/// out-of-range literal becomes null and silently rejects every row. Either
/// case makes the runtime predicate unusable for the file.
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

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

use crate::Result;
use crate::expr::Predicate;

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

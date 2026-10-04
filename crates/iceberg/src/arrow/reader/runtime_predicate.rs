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

use std::sync::Arc;

use crate::Result;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::spec::SchemaRef;

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

    /// Returns the predicate to apply, or None when no useful restriction is available.
    pub fn predicate(&self) -> Option<&Predicate> {
        self.predicate.as_ref()
    }

    /// Returns a monotonic version supplied by the producer.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Supplies execution-time predicates to an Arrow reader.
///
/// A snapshot is requested when a data-file task begins processing, and may be
/// refreshed between row groups. Implementations must only return predicates
/// that are safe to AND with the task's planned predicate for unread rows.
/// Generations must increase when the predicate changes, including changes to
/// or from `None`. Publish the predicate before its generation becomes visible.
/// Each snapshot must pair a predicate with its own publication generation.
pub trait RuntimePredicateProvider: Send + Sync {
    /// Returns the current publication generation without cloning the predicate.
    ///
    /// This is called on the reader's hot path. Implementations should use a
    /// cheap synchronized read, such as an atomic load with acquire ordering.
    fn generation(&self) -> u64;

    /// Returns the current runtime predicate snapshot.
    fn snapshot(&self) -> Result<RuntimePredicateSnapshot>;
}

/// One task's bound runtime predicate. The task schema and case policy must stay
/// fixed for the lifetime of this state.
pub(super) struct RuntimePredicateState {
    provider: Arc<dyn RuntimePredicateProvider>,
    generation: Option<u64>,
    predicate: Option<BoundPredicate>,
}

impl RuntimePredicateState {
    pub(super) fn new(provider: Arc<dyn RuntimePredicateProvider>) -> Self {
        Self {
            provider,
            generation: None,
            predicate: None,
        }
    }

    pub(super) fn predicate(&self) -> Option<&BoundPredicate> {
        self.predicate.as_ref()
    }

    /// Snapshot and bind only on first use or after a publication change.
    /// An advisory provider/binding failure clears the cached restriction; the
    /// caller retains the planned predicate and delete processing.
    pub(super) fn refresh_if_changed(
        &mut self,
        schema: SchemaRef,
        case_sensitive: bool,
    ) -> Result<bool> {
        let observed = self.provider.generation();
        if self.generation == Some(observed) {
            return Ok(false);
        }

        // Cache failed publications too, avoiding repeated snapshots or binds
        // for a permanently invalid predicate. A new generation can recover.
        self.predicate = None;
        self.generation = Some(observed);
        let snapshot = self.provider.snapshot()?;
        self.generation = Some(snapshot.generation());
        self.predicate = snapshot
            .predicate()
            .map(|predicate| predicate.bind(schema, case_sensitive))
            .transpose()?;
        Ok(true)
    }
}

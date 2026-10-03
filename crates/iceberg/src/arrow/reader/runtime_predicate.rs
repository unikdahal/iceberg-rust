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
/// A snapshot is requested when a data-file task begins processing. Implementations
/// must only return predicates that are safe to AND with the task's planned predicate.
pub trait RuntimePredicateProvider: Send + Sync {
    /// Returns the current runtime predicate snapshot.
    fn snapshot(&self) -> Result<RuntimePredicateSnapshot>;
}

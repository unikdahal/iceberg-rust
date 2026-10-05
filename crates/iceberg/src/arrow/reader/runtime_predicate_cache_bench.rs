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

//! Opt-in benchmark of the runtime predicate cache under concurrent readers and
//! publications, kept on the benchmark branch. Compares the production cache
//! (one mutex held through refresh) with a read-mostly `RwLock` variant that
//! refreshes under a separate single-flight mutex. Not part of the upstream change.

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::runtime_predicate::RuntimePredicates;
use super::{RuntimePredicateProvider, RuntimePredicateSnapshot as Snapshot};
use crate::Result;
use crate::expr::{Bind, BoundPredicate, Predicate, Reference};
use crate::spec::{Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type};

trait Cache: Send + Sync {
    fn current(&self, schema: &SchemaRef) -> Option<Arc<BoundPredicate>>;
}

impl Cache for RuntimePredicates {
    fn current(&self, schema: &SchemaRef) -> Option<Arc<BoundPredicate>> {
        RuntimePredicates::current(self, schema, false, "bench")
    }
}

struct Cached {
    generation: u64,
    schema: SchemaRef,
    predicate: Option<Arc<BoundPredicate>>,
}

/// Read-mostly alternative: hits take a read lock, refreshes are single-flight
/// under a separate mutex and hold the write lock only to publish.
struct RwCache {
    provider: Arc<dyn RuntimePredicateProvider>,
    cached: RwLock<Option<Cached>>,
    refresh: Mutex<()>,
}

impl RwCache {
    fn lookup(&self, generation: u64, schema: &SchemaRef) -> Option<Option<Arc<BoundPredicate>>> {
        self.cached
            .read()
            .unwrap()
            .as_ref()
            .filter(|c| c.generation >= generation && Arc::ptr_eq(&c.schema, schema))
            .map(|c| c.predicate.clone())
    }
}

impl Cache for RwCache {
    fn current(&self, schema: &SchemaRef) -> Option<Arc<BoundPredicate>> {
        if let Some(predicate) = self.lookup(self.provider.generation(), schema) {
            return predicate;
        }
        let _refresh = self.refresh.lock().unwrap();
        let generation = self.provider.generation();
        if let Some(predicate) = self.lookup(generation, schema) {
            return predicate;
        }
        let (generation, predicate) = match self.provider.snapshot() {
            Ok(snapshot) if snapshot.generation() < generation => (generation, None),
            Ok(snapshot) => {
                let generation = snapshot.generation();
                let bound = snapshot
                    .into_predicate()
                    .map(|p| p.rewrite_not().bind(schema.clone(), false))
                    .transpose();
                match bound {
                    Ok(bound) => (generation, bound.map(Arc::new)),
                    Err(_) => (generation, None),
                }
            }
            Err(_) => (generation, None),
        };
        *self.cached.write().unwrap() = Some(Cached {
            generation,
            schema: schema.clone(),
            predicate: predicate.clone(),
        });
        predicate
    }
}

/// A provider whose publisher swaps in a new predicate and bumps the generation
/// while holding the predicate lock, so snapshots are self-consistent.
struct Churn {
    generation: AtomicU64,
    predicate: Mutex<Predicate>,
}

impl Churn {
    fn publish(&self, predicate: Predicate) {
        let mut guard = self.predicate.lock().unwrap();
        *guard = predicate;
        self.generation.fetch_add(1, Ordering::AcqRel);
    }
}

impl RuntimePredicateProvider for Churn {
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn snapshot(&self) -> Result<Snapshot> {
        let guard = self.predicate.lock().unwrap();
        Ok(Snapshot::new(
            Some(guard.clone()),
            self.generation.load(Ordering::Acquire),
        ))
    }
}

fn predicate(literals: i64, salt: i64) -> Predicate {
    if literals == 1 {
        Reference::new("id").greater_than(Datum::long(salt))
    } else {
        Reference::new("id").is_in((0..literals).map(|v| Datum::long(v + salt)))
    }
}

fn schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap(),
    )
}

#[derive(Default, Clone)]
struct Latencies {
    calls: u64,
    buckets: [u64; 6],
    max_ns: u64,
}

impl Latencies {
    fn record(&mut self, ns: u64) {
        self.calls += 1;
        self.max_ns = self.max_ns.max(ns);
        let bucket = match ns {
            0..1_000 => 0,
            1_000..10_000 => 1,
            10_000..100_000 => 2,
            100_000..1_000_000 => 3,
            1_000_000..10_000_000 => 4,
            _ => 5,
        };
        self.buckets[bucket] += 1;
    }
}

fn run(
    name: &str,
    make: impl Fn(Arc<Churn>) -> Arc<dyn Cache>,
    threads: usize,
    literals: i64,
    interval: Option<Duration>,
    window: Duration,
) {
    let provider = Arc::new(Churn {
        generation: AtomicU64::new(1),
        predicate: Mutex::new(predicate(literals, 0)),
    });
    let cache = make(provider.clone());
    let schema = schema();
    // Warm the first generation so that only publications cause refreshes.
    black_box(cache.current(&schema));
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(threads + 1));
    let readers: Vec<_> = (0..threads)
        .map(|_| {
            let (cache, schema, stop, barrier) =
                (cache.clone(), schema.clone(), stop.clone(), barrier.clone());
            std::thread::spawn(move || {
                let mut latencies = Latencies::default();
                barrier.wait();
                while !stop.load(Ordering::Relaxed) {
                    let start = Instant::now();
                    black_box(cache.current(&schema));
                    latencies.record(start.elapsed().as_nanos() as u64);
                }
                latencies
            })
        })
        .collect();
    barrier.wait();
    let begin = Instant::now();
    let mut publications = 0_u64;
    while begin.elapsed() < window {
        match interval {
            Some(interval) => {
                publications += 1;
                provider.publish(predicate(literals, publications as i64));
                std::thread::sleep(interval);
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
    stop.store(true, Ordering::Relaxed);
    let elapsed = begin.elapsed();
    let mut total = Latencies::default();
    for reader in readers {
        let l = reader.join().unwrap();
        total.calls += l.calls;
        total.max_ns = total.max_ns.max(l.max_ns);
        for (sum, v) in total.buckets.iter_mut().zip(l.buckets) {
            *sum += v;
        }
    }
    let share = |n: u64| 100.0 * n as f64 / total.calls as f64;
    println!(
        "| {name} | {threads} | {literals} | {} | {publications} | {:.2} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {} |",
        interval.map_or("none".to_string(), |i| format!("{}ms", i.as_millis())),
        total.calls as f64 / elapsed.as_secs_f64() / 1e6,
        share(
            total.buckets[1]
                + total.buckets[2]
                + total.buckets[3]
                + total.buckets[4]
                + total.buckets[5]
        ),
        share(total.buckets[2] + total.buckets[3] + total.buckets[4] + total.buckets[5]),
        share(total.buckets[3] + total.buckets[4] + total.buckets[5]),
        share(total.buckets[4] + total.buckets[5]),
        share(total.buckets[5]),
        total.max_ns / 1000,
    );
}

#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture --test-threads=1"]
fn runtime_predicate_cache_benchmark() {
    let seconds: f64 = std::env::var("ICEBERG_BENCH_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);
    let window = Duration::from_secs_f64(seconds);

    // Cost of one refresh: waiters are blocked this long under the production cache.
    for literals in [1_i64, 10_000, 100_000] {
        let provider = Arc::new(Churn {
            generation: AtomicU64::new(1),
            predicate: Mutex::new(predicate(literals, 0)),
        });
        let cache = RuntimePredicates::new(provider.clone());
        let schema = schema();
        let mut total = Duration::ZERO;
        for salt in 1..=20 {
            provider.publish(predicate(literals, salt));
            let start = Instant::now();
            black_box(Cache::current(&cache, &schema));
            total += start.elapsed();
        }
        println!(
            "refresh (snapshot + rewrite + bind), {literals} literals: {:.3} ms",
            total.as_secs_f64() * 1000.0 / 20.0
        );
    }

    println!(
        "| cache | threads | literals | publish every | publications | M calls/s | % calls >=1us | >=10us | >=100us | >=1ms | >=10ms | max us |"
    );
    println!("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    let makers: [(&str, fn(Arc<Churn>) -> Arc<dyn Cache>); 2] = [
        ("mutex-through-refresh (production)", |provider| {
            Arc::new(RuntimePredicates::new(provider))
        }),
        ("rwlock + single-flight", |provider| {
            Arc::new(RwCache {
                provider,
                cached: RwLock::new(None),
                refresh: Mutex::new(()),
            })
        }),
    ];
    for threads in [1, 4, 8] {
        for (literals, interval) in [
            (1_i64, None),
            (1, Some(Duration::from_millis(1))),
            (10_000, Some(Duration::from_millis(20))),
            (100_000, Some(Duration::from_millis(100))),
        ] {
            for (name, make) in &makers {
                run(name, make, threads, literals, interval, window);
            }
        }
    }
}

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

//! Opt-in stable-generation benchmark, kept on the benchmark branch.
//! Measures the production cache against the original hot path and an unlocked
//! generation-check/Arc-clone control. This is task-boundary cache overhead,
//! not end-to-end scan throughput or a predicate-evaluation benchmark.

use std::hint::black_box;
use std::sync::Barrier;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::*;
use crate::expr::Reference;
use crate::spec::{Datum, NestedField, Type};

struct Provider {
    generation: AtomicU64,
    snapshots: AtomicU64,
    restricted: bool,
}

impl RuntimePredicateProvider for Provider {
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn snapshot(&self) -> Result<RuntimePredicateSnapshot> {
        self.snapshots.fetch_add(1, Ordering::Relaxed);
        Ok(RuntimePredicateSnapshot::new(
            self.restricted
                .then(|| Reference::new("id").greater_than(Datum::long(0))),
            self.generation(),
        ))
    }
}

/// Original cache implementation copied from 505e988b, including its miss
/// path. Keeping the full implementation avoids comparing an easily inlined
/// hit-only baseline with the larger production function. Only warmed hits are
/// exercised during measurement; baseline and production use the same bound Arc.
struct OriginalCachedPredicate {
    generation: u64,
    schema: SchemaRef,
    case_sensitive: bool,
    predicate: Option<Arc<BoundPredicate>>,
}

impl OriginalCachedPredicate {
    fn matches(&self, generation: u64, schema: &SchemaRef, case_sensitive: bool) -> bool {
        self.generation == generation
            && self.case_sensitive == case_sensitive
            && (Arc::ptr_eq(&self.schema, schema) || *self.schema == **schema)
    }
}

struct OriginalCache {
    provider: Arc<dyn RuntimePredicateProvider>,
    cached: Mutex<Option<OriginalCachedPredicate>>,
}

impl OriginalCache {
    fn current(
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
        *self.cached.lock().unwrap() = Some(OriginalCachedPredicate {
            generation,
            schema: schema.clone(),
            case_sensitive,
            predicate,
        });
        result
    }
}

/// Reuses OS threads across repetitions. The ready barrier excludes thread
/// startup from timing; the start/done barriers add a small fixed per-run cost.
/// Each result crosses black_box and is dropped, so the loop includes Arc drops
/// and cannot optimize the calls away. Records elapsed wall time, not CPU time.
fn measure<F>(threads: usize, total_calls: usize, repetitions: usize, read: F) -> Vec<f64>
where F: Fn() -> Option<Arc<BoundPredicate>> + Sync {
    let ready = Barrier::new(threads + 1);
    let start = Barrier::new(threads + 1);
    let done = Barrier::new(threads + 1);
    let calls_per_thread = total_calls / threads;
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let read = &read;
            let ready = &ready;
            let start = &start;
            let done = &done;
            scope.spawn(move || {
                for _ in 0..repetitions + 1 {
                    ready.wait();
                    start.wait();
                    for _ in 0..calls_per_thread {
                        drop(black_box(read()));
                    }
                    done.wait();
                }
            });
        }
        let mut times = Vec::with_capacity(repetitions);
        // First iteration warms the instruction/data caches and is discarded.
        for repetition in 0..repetitions + 1 {
            ready.wait();
            let began = Instant::now();
            start.wait();
            done.wait();
            if repetition != 0 {
                times.push(began.elapsed().as_nanos() as f64 / (calls_per_thread * threads) as f64);
            }
        }
        times
    })
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[test]
#[ignore = "opt-in optimized contention benchmark; run with --release --ignored --nocapture"]
fn stable_generation_contention_benchmark() {
    if cfg!(debug_assertions) {
        panic!("run this benchmark with --release");
    }
    let total_calls = std::env::var("ICEBERG_RUNTIME_BENCH_CALLS")
        .map_or(10_000_000, |value| value.parse::<usize>().unwrap());
    let repetitions = std::env::var("ICEBERG_RUNTIME_BENCH_REPETITIONS")
        .map_or(9, |value| value.parse::<usize>().unwrap());
    assert!(total_calls >= 8 && total_calls.is_multiple_of(8));
    assert!(repetitions >= 3 && repetitions % 2 == 1);
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap(),
    );
    println!(
        "case,threads,path,median_wall_ns_per_call,min_wall_ns_per_call,max_wall_ns_per_call,calls_per_sample,repetitions"
    );
    for restricted in [false, true] {
        let provider = Arc::new(Provider {
            generation: AtomicU64::new(1),
            snapshots: AtomicU64::new(0),
            restricted,
        });
        let control_provider: Arc<dyn RuntimePredicateProvider> = provider.clone();
        let cache = RuntimePredicates::new(provider.clone());
        let bound = cache.current(&schema, false, "benchmark.parquet");
        let original = OriginalCache {
            provider: provider.clone(),
            cached: Mutex::new(Some(OriginalCachedPredicate {
                generation: 1,
                schema: schema.clone(),
                case_sensitive: false,
                predicate: bound.clone(),
            })),
        };
        let case = if restricted { "bound" } else { "none" };
        for threads in [1, 2, 4, 8] {
            let mut samples = [Vec::new(), Vec::new(), Vec::new()];
            // Rotate the three implementations each round to reduce temporal
            // bias (CPU frequency, contention, and VM scheduling). Each round
            // discards a warm-up and keeps one sample per implementation.
            for round in 0..repetitions {
                for offset in 0..3 {
                    let path = (round + offset) % 3;
                    let time = match path {
                        0 => measure(threads, total_calls, 1, || {
                            black_box(control_provider.generation());
                            bound.clone()
                        }),
                        1 => measure(threads, total_calls, 1, || {
                            original.current(black_box(&schema), false, "benchmark.parquet")
                        }),
                        2 => measure(threads, total_calls, 1, || {
                            cache.current(black_box(&schema), false, "benchmark.parquet")
                        }),
                        _ => unreachable!(),
                    };
                    samples[path].extend(time);
                }
            }
            for (path, times) in ["unlocked_control", "original", "serialized"]
                .into_iter()
                .zip(&samples)
            {
                for (round, time) in times.iter().enumerate() {
                    println!("sample,{case},{threads},{path},{round},{time:.2}");
                }
                println!(
                    "{case},{threads},{path},{:.2},{:.2},{:.2},{total_calls},{repetitions}",
                    median(times),
                    times.iter().copied().fold(f64::INFINITY, f64::min),
                    times.iter().copied().fold(0.0, f64::max)
                );
            }
            println!(
                "ratio,{case},{threads},{:.4}",
                median(&samples[2]) / median(&samples[1])
            );
            // Stable generation must stay warm across every measured call.
            assert_eq!(provider.snapshots.load(Ordering::Relaxed), 1);
        }
    }
}

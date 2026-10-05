<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Stable-generation runtime predicate contention

This benchmark lives only on `bench/runtime-predicate-contention`. It does not change the provider API or the production implementation. It calls the actual production `RuntimePredicates::current()` from an ignored test, so no visibility changes are needed.

Run from the repository root:

```sh
cargo test --release -p iceberg --lib stable_generation_contention_benchmark -- --ignored --nocapture --test-threads=1
```

The comparison includes:

- **serialized:** the fixed production cache, which reads generation and snapshots/binds while holding one mutex;
- **original:** the complete original cache implementation from `505e988b`, with generation checked outside the same mutex. Its miss path is retained to avoid unfair inlining differences, but only warmed hits are timed;
- **unlocked control:** the generation load and bound `Arc` clone/drop without cache key checks or a mutex. This is a lower bound, not a proposed replacement.

The provider stays at generation 1. Both a bound predicate and a `None` snapshot are measured, and the benchmark asserts exactly one snapshot for each workload after every thread-count experiment. The schema is shared by `Arc`, as in a scan using a common table schema. No snapshot/binding work or file I/O is timed.

Each sample makes 10,000,000 calls in total across 1, 2, 4, or 8 worker threads. Threads synchronize at start/end barriers; thread creation is excluded. One warm-up sample is discarded before each timed sample. Nine samples per implementation are collected, rotating implementation order to reduce temporal bias. Every result crosses `black_box` and is dropped, including the `Arc` reference-count cost.

Set `ICEBERG_RUNTIME_BENCH_CALLS` (divisible by eight) and `ICEBERG_RUNTIME_BENCH_REPETITIONS` (odd, at least three) to adjust run length without recompilation.

The output reports median, minimum, and maximum **elapsed wall nanoseconds divided by total calls**, plus all raw samples. At multiple threads this is the reciprocal of aggregate throughput, not individual call latency. The start/end barrier overhead is amortized over ten million calls. The benchmark is deliberately an extreme repeated-cache-lookup workload; it does not measure end-to-end scan performance.

## Environment and results

Measured 2026-10-04 on Linux 6.18.44, AMD EPYC 9V74, with three visible vCPUs and a cgroup quota of two CPUs (`cpu.max = 200000 100000`). The 4/8-thread experiments are oversubscribed. Compiler: `rustc 1.97.0-nightly (e8e4541ff 2026-04-15)`, pinned by `nightly-2026-04-16`. Cargo release optimization (`opt-level=3`, debug info disabled), two build jobs; no other local build ran during measurement. The production code under test is PR head `dfa99edaa9714d1044859459bbef508f321f1545`.

Median aggregate wall ns/call; percentages compare serialized with original. The unlocked control is not an alternative cache implementation.

| Snapshot | Threads | Original | Serialized | Change | Unlocked control |
| --- | ---: | ---: | ---: | ---: | ---: |
| bound | 1 | 9.79 | 9.22 | -5.8% | 3.79 |
| bound | 2 | 37.61 | 36.51 | -2.9% | 7.32 |
| bound | 4 | 44.39 | 45.40 | +2.3% | 12.55 |
| bound | 8 | 53.25 | 33.35 | -37.4% | 11.37 |
| none | 1 | 5.48 | 5.03 | -8.2% | 2.23 |
| none | 2 | 15.85 | 20.78 | +31.1% | 1.14 |
| none | 4 | 37.67 | 34.60 | -8.1% | 1.13 |
| none | 8 | 33.93 | 34.13 | +0.6% | 1.00 |

Uncontended bound-predicate lookup is 9.22 ns versus 9.79 ns for the original implementation. At 2/4 threads, the bound-predicate medians change by -2.9%/+2.3%. There is no consistent stable-generation regression across thread counts in this run. The 8-thread bound-predicate result looks faster, but scheduling variation and the CPU quota prevent treating it as a general speedup.

The cached-`None` 2-thread median is higher (20.78 versus 15.85 ns, +31.1%, an absolute 4.93 ns per aggregate call), with overlapping ranges of 10.76–28.22 versus 11.40–28.29 ns. The 4/8-thread `None` medians change by -8.1%/+0.6%. This anomalous result is included rather than discarded; this machine does not establish a precise high-contention overhead bound. The existing mutex and shared Arc reference-count costs remain visible compared with the unlocked control.

The reader checks the cache once when a data-file task starts, not for each row or page. These synthetic lookup measurements do not establish an end-to-end query throughput change. The corrected cache guarantees should be retained; these results do not justify changing the provider API or introducing another synchronization design.

The optimized benchmark completed in 94.38 seconds with all assertions passing. The benchmark code also passed release-mode Clippy with warnings denied and formatting checks. Both providers remained at generation 1 and each took exactly one snapshot across all measurements. The harness adds no public API, dependencies, or default test execution; it is ignored unless explicitly selected.

[All summary statistics](runtime-predicate-contention-summary.csv) and [all 216 raw samples](runtime-predicate-contention-samples.csv) are retained for inspection and future comparisons.

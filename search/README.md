# Search Extension for Persistent SafeGraphs

## Installation & Usage

Enabled by the `search` Cargo feature, which pulls in NumKong for dense kernels, StringZilla for sequence kernels, and ForkUnion for structured parallelism.

```rust
let table = GraphTable::from_bytes(bytes)?;
let mut searcher = DotSearcherAny::new(table.value_dtype(), expansion_factor, ExecutionScope::CPUs(0), Global)?;
let mut matches = Matrix::<u64>::new(query_count, k)?;
let mut distances = Matrix::<f64>::new(query_count, k)?;
let mut counts = vec![0u64; query_count];
searcher.search(query_bytes, &table, &mut matches, &mut distances, &mut counts)?;
```

## Design Principles

__Two functors per metric family.__
A `Scanner` is exact brute force: it scores every query against every node in one batched kernel call.
A `Searcher` is approximate: it drives greedy best-first walks over the file's frozen graph, one `Walker` per query.
Both are stateful objects that own their resources and reusable scratch; neither is a free function.

__One uniform call shape.__
Every functor exposes exactly one operation, and both operations share one signature:

```rust
scan(queries, table, matches: &mut Matrix<u64>, distances: &mut Matrix<f64>, counts: &mut [u64])
search(queries, table, matches: &mut Matrix<u64>, distances: &mut Matrix<f64>, counts: &mut [u64])
```

The output matrices are `query_count × k`, so `k` and the query count are read off their shapes and are never separate arguments.
`counts[q]` reports how many results query `q` found; unfilled `matches` slots hold a `u64::MAX` sentinel.

__Queries are a per-family generic, never coerced.__
`MetricLike::Queries` names the natural batch type of each family: an `nk::MatrixView<T>` for dense and binary vectors, a byte tape for variable-length sequences.
The erased `*Any` layer accepts raw query bytes in the table's dtype and builds the typed view internally.
The `QueryBatchLike` trait — a count plus per-index access — is the only thing the shared batch engine asks of a query type.

__Construction is configuration; the table is a call-time argument.__
Every functor has exactly one constructor, taking only what does not depend on any particular file: the metric's own configuration (costs, the dtype string), the walk breadth `expansion_factor` for searchers, the `ExecutionScope`, and — wherever the functor owns `allocator_api2` scratch — the allocator, always last and never defaulted away behind a `new`/`new_in` pair.
The `GraphTable` first appears at the `scan`/`search` call, and everything derived from it — node count, neighbor index width — is resolved there, per call.
One functor therefore serves many tables: scratch grows to fit the largest table seen and repeat calls allocate nothing.

__Parallelism is owned, never exposed.__
`ExecutionScope` — `Serial`, `CPUs(n)`, or `GPU(index)` — is the only knob that crosses the API, and no pool, topology, scheduler, or device type appears in any signature.
Each family opens the scope into whatever its backend calls an executor: the NumKong families spawn a ForkUnion thread pool, the StringZilla ones a `DeviceScope`, and a CPU-only family answers `GPU` with an error at construction rather than pretending.
A scanner parallelizes inside a single call, fanning the packed `queries × nodes` kernel across its executor.
A NumKong searcher parallelizes across queries — one `TraversalContext` per pool thread, work-stealing whole walks, since walk lengths vary with the component they land in — while a StringZilla searcher runs whole walks on a single context and parallelizes inside each scoring call, where its engines already drive every worker the scope names.
With `Serial` the fan-out runs inline on the caller — the sequential path is the same code path.
One functor drives one device; spanning several GPUs means one functor per device and sharding tables across them, a layer above this crate.

__Owners are allocator-generic; parallel code sees only views.__
`GraphTable<Allocator>` owns the mapping, but the search layer speaks `GraphTableView` — a borrowed, `Copy + Send + Sync` window of the mapped bytes plus the `Copy` column descriptors.
A fold's tasks capture the view directly, so no `unsafe` shareability assertion, no per-family re-slicing of columns, and no allocator generic survives past the public `scan`/`search` entry, which converts once with `table.view()`.

__Native precision inside, wide at the boundary.__
Scoring accumulates in each scalar's native output type — `i32` for `i8` dots, `u32` for Hamming, `f32` for the halves and minifloats.
Node indices travel at the table's native width, `u32` or `u64`, so a `u32` graph never pays for `u64` frontier cells.
Widening to the erased `u64` node and `f64` distance happens once, when results are written to the caller's matrices.

__Three dispatch layers, each doing one job.__
Typed leaves are generic over the scalar `T` and the index width `I`; one marker type per metric replaces a struct per dtype.
A width resolver per family holds the executor and one worker set, settling `I` from the first searched table.
The `*Any` enums dispatch on the header dtype string alone, one arm per scalar, sorted by dynamic range: `f64, f32, bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, i4x2, u8, u4x2, u1x8`.
The sequence family has no dtype axis — every value is `utf8` bytes — so it skips the `*Any` layer entirely: byte-level Levenshtein, its codepoint-level UTF-8 variant, and the alignment scores are separate metrics the cross-family router in `dispatch` routes between.

## Constraints

Stable Rust, `panic = "abort"`, and no panics in library code: every allocation is `try_reserve`, every cast is checked, every failure is an `Error` variant.
No `macro_rules!` — dispatch tables are written out by hand.
Heap allocation is minimized: lazy iterators are consumed in place, and per-call scratch lives in the functor, not the call.
Reads from the file are zero-copy views over the memory map, alignment- and width-checked.
Backend types stay in their pillar: `search.rs` names neither NumKong nor StringZilla, dense files never import StringZilla, sequence files never import NumKong.

## Code Organization

- `search.rs` — the backend-free contract: `MetricLike`, `ScannerLike`, `WalkerLike`, `QueryBatchLike`, `ExecutionScope`, the greedy `walk`, `TraversalContext`, and the owned output `Matrix`.
- `search/batch.rs` — the shared batch engine: `batch_scan`, the pooled `batch_search`, and the single-worker `batch_search_serial`, plus the width-resolver `Workers` lanes every family reuses.
- `search/dense.rs` — Dot, Euclidean, and Angular families over every NumKong scalar.
- `search/binary.rs` — Hamming and Jaccard over packed `u1x8` bits.
- `search/sequence.rs` — one generic `SequenceScanner`/`SequenceWalker`/`SequenceSearcher` body over an `EngineLike` seam, instantiated four ways: byte-level Levenshtein and its codepoint-level UTF-8 variant (per-edit `LevenshteinCosts`, smaller-is-better), and Needleman-Wunsch and Smith-Waterman alignment scores (class-matrix `AlignmentCosts`, larger-is-better).
- `search/geospatial.rs` — Haversine and Vincenty over `[latitude, longitude]` pairs in radians, meters out, nearest-first: one generic body over a `KernelLike` seam, the scan fanning per-query broadcasts across the pool since the kernels have no packed cross-product.
- `search/maxsim.rs` — late-interaction MaxSim over ragged `[*, depth]` multi-vectors in `f32`/`f16`/`bf16`, nearest-first: the kernel sums one-minus-best-cosine per query token, so an identical pair scores zero. Queries pack once per batch and become the metric's own `Query` type; nodes pack into reusable staging slots — per staged node in the walk, per pool worker in the scan, which fans over nodes to fill contiguous tile rows.
- `search/dispatch.rs` — the cross-family router over a file's declared metric; deliberately empty until the families' erased entry points settle.

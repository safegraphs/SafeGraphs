# Search Extension for Persistent SafeGraphs

## Installation & Usage

Enabled by the `search` Cargo feature, which pulls in NumKong for dense kernels, StringZilla for sequence kernels, and ForkUnion for structured parallelism.

```rust
let table = Table::parse(bytes)?;
let mut searcher = DotSearcherAny::new(table.value_dtype(), expansion_factor, threads)?;
let mut matches = Matrix::<u64>::new(query_count, k)?;
let mut distances = Matrix::<f64>::new(query_count, k)?;
let mut counts = vec![0u64; query_count];
searcher.search(query_bytes, &table, &mut matches, &mut distances, &mut counts)?;
```

## Design Principles

__Two functors per metric family.__
A `Scanner` is exact brute force: it scores every query against every node in one batched kernel call.
A `Searcher` is approximate: it runs a greedy best-first walk over the file's frozen graph, one walk per query.
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
`ScoringLike::Queries` names the natural batch type of each family: an `nk::MatrixView<T>` for dense and binary vectors, a byte tape for variable-length sequences.
The erased `*Any` layer accepts raw query bytes in the table's dtype and builds the typed view internally.
The `QueryBatch` trait — a count plus per-index access — is the only thing the shared batch engine asks of a query type.

__Construction is configuration; the table is a call-time argument.__
A constructor takes only what does not depend on any particular file: the value dtype string, the walk breadth `expansion_factor` for searchers, and `threads`.
The `Table` first appears at the `scan`/`search` call, and everything derived from it — node count, neighbor index width — is resolved there, per call.
One functor therefore serves many tables: scratch grows to fit the largest table seen and repeat calls allocate nothing.

__Parallelism is owned, never exposed.__
Each functor builds and owns one ForkUnion thread pool; `threads` is the only knob that crosses the API, and no pool, topology, or scheduler type appears in any signature.
A scanner parallelizes inside a single call, fanning the packed `queries × nodes` kernel across its pool.
A searcher parallelizes across queries: it keeps one `TraversalContext` per pool thread and work-steals whole walks, since walk lengths vary with the component they land in.
With `threads = 1` the fan-out runs inline on the caller — the sequential path is the same code path.

__Native precision inside, wide at the boundary.__
Scoring accumulates in each scalar's native output type — `i32` for `i8` dots, `u32` for Hamming, `f32` for the halves and minifloats.
Node indices travel at the table's native width, `u32` or `u64`, so a `u32` graph never pays for `u64` frontier cells.
Widening to the erased `u64` node and `f64` distance happens once, when results are written to the caller's matrices.

__Three dispatch layers, each doing one job.__
Typed leaves are generic over the scalar `T` and the index width `I`; one marker type per metric replaces a struct per dtype.
A width resolver per family holds the pool and one worker set, settling `I` from the first searched table.
The `*Any` enums dispatch on the header dtype string alone, one arm per scalar, sorted by dynamic range: `f64, f32, bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, i4x2, u8, u4x2, u1x8`.

## Constraints

Stable Rust, `panic = "abort"`, and no panics in library code: every allocation is `try_reserve`, every cast is checked, every failure is an `Error` variant.
No `macro_rules!` — dispatch tables are written out by hand.
Heap allocation is minimized: lazy iterators are consumed in place, and per-call scratch lives in the functor, not the call.
Reads from the file are zero-copy views over the memory map, alignment- and width-checked.
Backend types stay in their pillar: `search.rs` names neither NumKong nor StringZilla, dense files never import StringZilla, sequence files never import NumKong.

## Code Organization

- `search.rs` — the backend-free contract: `ScoringLike`, `ScannerLike`, `SearcherLike`, `QueryBatch`, the greedy `walk`, `TraversalContext`, and the owned output `Matrix`.
- `search/dense.rs` — Dot, Euclidean, and Angular families over every NumKong scalar, plus the shared batch engine `batch_scan`/`batch_search` the other families reuse.
- `search/binary.rs` — Hamming and Jaccard over packed `u1x8` bits.
- `search/sequence.rs` — Levenshtein over StringZilla byte tapes.
- `search/geospatial.rs` — Haversine and Vincenty over `[lat, lon]` pairs; design notes, not yet implemented.
- `search/maxsim.rs` — late-interaction MaxSim; design notes, not yet implemented.
- `search/dispatch.rs` — the cross-family router over a file's declared metric; deliberately empty until the families' erased entry points settle.

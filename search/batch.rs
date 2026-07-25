//! The shared batch engine: shaped top-k over any scoring family, brute-force
//! or by walk. Names no NumKong and no StringZilla type — only ForkUnion, for
//! the cross-query fan-out — so every family reuses it without importing a
//! sibling's backend. It moved out of `dense` once four of the five families
//! were reaching into it; a module named after one family is the wrong home
//! for all of them.
//!
//! The vocabulary here is the uniform call shape: `matches`/`distances` are
//! `queries × k` [`Matrix`] outputs carrying both counts, `counts[q]` reports
//! how many results query `q` found, and an unfilled `matches` slot holds a
//! `u64::MAX` sentinel. [`batch_scan`] ranks a scanner's full tile;
//! [`batch_search`] fans whole walks across a pool, one query per task; and
//! [`batch_search_serial`] runs the same walks on a single worker, for a
//! family whose parallelism lives inside the kernel rather than across
//! queries.

use super::{
    ExecutionScope, Matrix, MetricLike, QueryBatchLike, Ranking, ScannerLike, SearcherLike,
};
use crate::{Error, TableView, Widen};
use allocator_api2::alloc::{Allocator as AllocatorLike, Global};
use allocator_api2::vec::Vec as AVec;
use forkunion as fu;
use std::sync::OnceLock;

/// Rank column `col` of a row-major score tile (`rows = nodes`, `cols = queries`)
/// top-k, writing row ids widened to `u64` into `nodes_out` and scores widened to
/// `f64` into `scores_out`, best-first per the [`Ranking`] direction. Unfilled
/// slots take a `u64::MAX` sentinel.
/// Returns the count found. Widening is monotonic, so ranking the widened scores is
/// the native order. The shared tail of every batch scan.
pub(crate) fn select_top_k<Score: Copy + Default + Widen>(
    tile: &Matrix<Score, impl AllocatorLike + Clone>,
    col: usize,
    rows: usize,
    ranking: Ranking,
    nodes_out: &mut [u64],
    scores_out: &mut [f64],
) -> Result<usize, Error> {
    let k = nodes_out.len();
    let mut filled = 0usize;
    for row in 0..rows {
        // A missing cell means the tile was reshaped for a different scan;
        // fail the batch rather than silently under-filling `counts`, matching
        // `rank_tile`'s handling of the same mismatch.
        let score = tile.get(row, col).ok_or(Error::Inconsistent)?;
        let score = score.widen();
        if filled < k {
            insert_pair(nodes_out, scores_out, filled, row as u64, score, ranking);
            filled += 1;
        } else if ranking.outranks(score, scores_out[k - 1]) {
            insert_pair(nodes_out, scores_out, k - 1, row as u64, score, ranking);
        }
    }
    for slot in nodes_out.iter_mut().skip(filled) {
        *slot = u64::MAX;
    }
    Ok(filled)
}

/// Insert `(node, score)` into the sorted prefix `..=end` of two parallel arrays,
/// shifting weaker entries right so the prefix stays best-first.
fn insert_pair(
    nodes: &mut [u64],
    scores: &mut [f64],
    end: usize,
    node: u64,
    score: f64,
    ranking: Ranking,
) {
    let mut position = end;
    while position > 0 && ranking.outranks(score, scores[position - 1]) {
        nodes[position] = nodes[position - 1];
        scores[position] = scores[position - 1];
        position -= 1;
    }
    nodes[position] = node;
    scores[position] = score;
}

/// Validate the `matches`/`distances`/`counts` shapes against a `query_count`, and
/// return `k` — read off `matches.cols()`. The shaped batch carries its own count,
/// so neither `k` nor the query count is an argument to `scan`/`search`.
fn batch_shape(
    query_count: usize,
    matches: &Matrix<u64>,
    distances: &Matrix<f64>,
    counts: &[u64],
) -> Result<usize, Error> {
    let k = matches.cols();
    if k == 0
        || matches.rows() != query_count
        || distances.rows() != query_count
        || distances.cols() != k
        || counts.len() != query_count
    {
        return Err(Error::Inconsistent);
    }
    Ok(k)
}

/// Batch brute-force top-k for any scanner: score every (query, node) into the
/// caller's reusable `tile`, then rank each query's column into
/// `matches`/`distances`, best-first per `ranking`, reporting the count found per
/// query. The metric-typed `scan` methods are one-line delegations to this. The
/// tile is grown, never shrunk, so repeat batches allocate nothing once warm.
// Eight arguments is the honest fan-in here: five are the caller-owned outputs
// and scratch that deliberately live outside this function so batches reuse them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_scan<Metric, Scanner>(
    scanner: &mut Scanner,
    queries: Metric::Queries<'_>,
    table: TableView<'_>,
    ranking: Ranking,
    tile: &mut Matrix<Metric::Score, impl AllocatorLike + Clone>,
    matches: &mut Matrix<u64>,
    distances: &mut Matrix<f64>,
    counts: &mut [u64],
) -> Result<(), Error>
where
    Metric: MetricLike,
    Metric::Score: Default + Widen,
    for<'a> Metric::Queries<'a>: QueryBatchLike<Metric>,
    Scanner: ScannerLike<Metric>,
{
    let query_count = queries.count();
    let nodes = table.len();
    batch_shape(query_count, matches, distances, counts)?;
    tile.reshape(nodes, query_count)?;
    scanner.scan(queries, table, tile)?;
    for (query, count) in counts.iter_mut().enumerate() {
        *count = select_top_k(
            tile,
            query,
            nodes,
            ranking,
            matches.row_mut(query).ok_or(Error::Inconsistent)?,
            distances.row_mut(query).ok_or(Error::Inconsistent)?,
        )? as u64;
    }
    Ok(())
}

/// Open an [`ExecutionScope`] as a ForkUnion pool, for the families whose kernels
/// run on CPU threads. The topology is a construction-time local, so no ForkUnion
/// type ever crosses the API. A GPU scope is refused here: a family whose backend
/// can drive one opens the scope itself, and one that cannot must say so rather
/// than silently running on cores.
pub(crate) fn open_pool(scope: ExecutionScope) -> Result<fu::ThreadPool, Error> {
    let topology = fu::Topology::new().map_err(|_| Error::Alloc)?;
    let threads = match scope {
        ExecutionScope::Serial => 1,
        ExecutionScope::Cpus(0) => topology.logical_cores_count().max(1),
        ExecutionScope::Cpus(cpus) => cpus,
        ExecutionScope::Gpu(_) => return Err(Error::WrongType),
    };
    fu::ThreadPool::try_spawn(&topology, threads).map_err(|_| Error::Alloc)
}

/// Build `count` searchers from a fallible constructor, reserving up front so the
/// whole set of per-thread contexts is either allocated or the batch fails cleanly.
pub(crate) fn build_workers<Searcher, Allocator: AllocatorLike>(
    count: usize,
    alloc: Allocator,
    mut make: impl FnMut() -> Result<Searcher, Error>,
) -> Result<AVec<Searcher, Allocator>, Error> {
    let mut workers = AVec::new_in(alloc);
    workers.try_reserve_exact(count).map_err(|_| Error::Alloc)?;
    for _ in 0..count {
        workers.push(make()?);
    }
    Ok(workers)
}

/// A width resolver's worker lanes: traversal-scratch leaves at the neighbor
/// width of the tables being searched — one per pool thread for a fanned-out
/// family, a single leaf for a kernel-parallel one. Construction sees no table,
/// so the lane starts [`Unresolved`](Workers::Unresolved); the first searched
/// table settles it, and a table of the other width rebuilds it — cheap, since
/// a leaf's scratch is empty until its first walk.
pub(crate) enum Workers<Narrow, Wide, Allocator: AllocatorLike = Global> {
    Unresolved,
    U32(AVec<Narrow, Allocator>),
    U64(AVec<Wide, Allocator>),
}

/// A read-only reference shareable across a fold's threads.
///
/// SAFETY: holders only read through it, and the referent is a local held alive
/// for the entire blocking fold — the one `unsafe impl` below is the single place
/// this crate asserts cross-thread shareability instead of scattering pointer
/// casts at every capture site.
pub(crate) struct ShareRead<'a, Referent: ?Sized>(pub(crate) &'a Referent);
unsafe impl<Referent: ?Sized> Sync for ShareRead<'_, Referent> {}
unsafe impl<Referent: ?Sized> Send for ShareRead<'_, Referent> {}

impl<'a, Referent: ?Sized> ShareRead<'a, Referent> {
    /// Accessor rather than a public field, so a closure captures the wrapper —
    /// which is the `Sync` assertion — and not the bare inner reference, which
    /// edition-2021 disjoint capture would otherwise pluck out.
    pub(crate) fn read(&self) -> &'a Referent {
        self.0
    }
}

/// The three caller-owned output buffers, held as raw `SafePtr`s so many threads can
/// fill disjoint query rows at once. `k` is the row width; row `query_index` and slot
/// `query_index` of `counts` belong to one walk alone.
struct Sink {
    matches: fu::SafePtr<u64>,
    distances: fu::SafePtr<f64>,
    counts: fu::SafePtr<u64>,
    k: usize,
}

/// One query's walk and drain, called on a worker's own context and writing that
/// query's row of the outputs — the shared body of every fan-out task, parallel or
/// serial. `matches` and `distances` are written through `SafePtr`, disjoint by
/// `query_index` and so safe to touch from many threads at once; a walk short of `k`
/// leaves a `u64::MAX` sentinel and reports its true count.
fn walk_one<Metric, Searcher>(
    worker: &mut Searcher,
    queries: &Metric::Queries<'_>,
    query_index: usize,
    table: TableView<'_>,
    sink: &Sink,
) -> Result<(), Error>
where
    Metric: MetricLike,
    Metric::Score: Widen,
    for<'a> Metric::Queries<'a>: QueryBatchLike<Metric>,
    Searcher: SearcherLike<Metric>,
{
    let query = queries.query(query_index).ok_or(Error::Inconsistent)?;
    worker.search(query, table)?;
    let base = query_index * sink.k;
    let found = worker
        .context()
        .drain(Searcher::compare, sink.k, |slot, node, score| {
            *sink.matches.get_mut_at(base + slot) = node.into();
            *sink.distances.get_mut_at(base + slot) = score.widen();
        });
    for slot in found..sink.k {
        *sink.matches.get_mut_at(base + slot) = u64::MAX;
    }
    *sink.counts.get_mut_at(query_index) = found as u64;
    Ok(())
}

/// Wrap the three output buffers as a [`Sink`], the form [`walk_one`] writes through.
fn sink_over(matches: &mut Matrix<u64>, distances: &mut Matrix<f64>, counts: &mut [u64]) -> Sink {
    Sink {
        matches: fu::SafePtr::new(matches.as_mut_slice().as_mut_ptr()),
        distances: fu::SafePtr::new(distances.as_mut_slice().as_mut_ptr()),
        counts: fu::SafePtr::new(counts.as_mut_ptr()),
        k: matches.cols(),
    }
}

/// Batch top-k by graph walk, fanned across the pool: each thread runs whole
/// independent walks on its own traversal context, draining best-first into its
/// query's row of `matches`/`distances`; `counts` is the number found, `< k` when a
/// walk exhausts a small component. The schedule steals work per query, since a
/// walk's length varies with the component it lands in. With a one-thread pool the
/// fan-out runs inline on the caller, so this is the sole batch-search path for a
/// fanned-out family, parallel or not.
///
/// `workers` holds one leaf per pool thread — for a dense metric that is exactly a
/// pool of contexts, the leaf being nothing but its scratch. The query batch and the
/// table are read-only and shared; each query's output row is written by one thread
/// alone, so `SafePtr` and [`ShareRead`] carry them across
/// the boundary. The first error any walk raises becomes the batch's.
pub(crate) fn batch_search<Metric, Searcher>(
    pool: &mut fu::ThreadPool,
    workers: &mut [Searcher],
    queries: Metric::Queries<'_>,
    table: TableView<'_>,
    matches: &mut Matrix<u64>,
    distances: &mut Matrix<f64>,
    counts: &mut [u64],
) -> Result<(), Error>
where
    Metric: MetricLike,
    Metric::Score: Widen,
    for<'a> Metric::Queries<'a>: QueryBatchLike<Metric>,
    Searcher: SearcherLike<Metric> + Send,
{
    let query_count = queries.count();
    batch_shape(query_count, matches, distances, counts)?;

    let sink = sink_over(matches, distances, counts);
    // The query batch is immutable for the whole blocking fold and only read by
    // the tasks; its backend view types declare no `Sync`, so it rides the
    // read-only wrapper. The table needs nothing: a [`TableView`] is `Copy` and
    // `Sync` by construction, so the closure just captures it.
    let queries_shared = ShareRead(&queries);
    let failure = OnceLock::new();

    fu::fold_with_scratch(
        pool,
        fu::ParallelRange::new(0..query_count),
        fu::DynamicScheduler,
        workers,
        |worker, query_index, _prong| {
            if let Err(error) = walk_one::<Metric, Searcher>(
                worker,
                queries_shared.read(),
                query_index,
                table,
                &sink,
            ) {
                let _ = failure.set(error);
            }
        },
    );

    match failure.into_inner() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Batch top-k by walk on a single worker, for a family whose parallelism lives
/// inside the kernel rather than across queries — a StringZilla engine already
/// drives every CPU core, or a GPU, from one call, so a thread-per-query fan-out
/// on top would only oversubscribe. Same body as [`batch_search`]'s task, minus
/// the fan-out — both go through [`walk_one`], so the two paths cannot drift in
/// how they rank, sentinel, or count.
pub(crate) fn batch_search_serial<Metric, Searcher>(
    worker: &mut Searcher,
    queries: Metric::Queries<'_>,
    table: TableView<'_>,
    matches: &mut Matrix<u64>,
    distances: &mut Matrix<f64>,
    counts: &mut [u64],
) -> Result<(), Error>
where
    Metric: MetricLike,
    Metric::Score: Widen,
    for<'a> Metric::Queries<'a>: QueryBatchLike<Metric>,
    Searcher: SearcherLike<Metric>,
{
    let query_count = queries.count();
    batch_shape(query_count, matches, distances, counts)?;
    let sink = sink_over(matches, distances, counts);
    for query_index in 0..query_count {
        walk_one::<Metric, Searcher>(worker, &queries, query_index, table, &sink)?;
    }
    Ok(())
}

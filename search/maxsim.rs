//! MaxSim (ColBERT late interaction) over NumKong. No StringZilla type here.
//!
//! A node's value is a multi-vector: a ragged `[tokens, depth]` matrix, one per
//! node, in `f32`/`f16`/`bf16` — the three scalars NumKong's `maxsim.h` packs.
//! Its logical [`Shape`] is `Matrix(0, depth)`: the `0` marks the __token axis as
//! the one that resizes per node__, while `depth` is fixed across the file, so the
//! byte layout is irregular and each node's token count is recovered from its row
//! extent ÷ `depth`. That is the whole reason this family waited for [`Shape`].
//!
//! The score is NumKong's late interaction — `Σᵢ minⱼ angular(qᵢ, dⱼ)`, an
//! __angular distance sum__ (`maxsim/serial.h` accumulates `1 - cosine`, clamped
//! at zero), so smaller is better and the family ranks nearest-first like
//! Euclidean, Geospatial, and Shapes. The Rust wrapper's "max cosine" wording
//! describes the same argmin, not the returned value's direction.
//!
//! Scoring is a __whole-multi-vector reduction__ per cell, like Shapes: the scan
//! packs every node once per call, then fans queries across the pool with one
//! packed-query scratch per worker, exactly the schedule Geospatial uses.

use super::batch::{batch_scan, batch_search, build_workers, open_pool, ShareRead, Workers};
use super::{
    walk, ExecutionScope, Matrix, MetricLike, QueryBatchLike, Ranking, ScannerLike,
    TraversalContext, WalkerLike,
};
use crate::search::dense::matrix_view;
use crate::{cast, Error, GraphTable, GraphTableView, IndexLike, OffsetDType, Shape};
use allocator_api2::alloc::Allocator as TableAllocatorLike;
use core::alloc::Allocator as AllocatorLike;
use core::cmp::Ordering;
use forkunion as fu;
use numkong as nk;
use std::alloc::Global;
use std::sync::OnceLock;

// region: Metric

/// A MaxSim metric over one packable scalar. `depth` is a property of the file,
/// not the type, so one metric spans every multi-vector width.
pub struct MaxSimMetric<Scalar>(core::marker::PhantomData<fn() -> Scalar>);

impl<Scalar: nk::MaxSim + nk::StorageElement + Copy + Default + 'static> MetricLike
    for MaxSimMetric<Scalar>
{
    /// One query multi-vector, flat `tokens * depth` scalars.
    type Query<'a> = &'a [Scalar];
    type Queries<'a> = MaxSimBatch<'a, Scalar>;
    /// NumKong accumulates the angular-distance sum in `f64` regardless of the
    /// packed scalar, so every dtype scores in one width.
    type Score = f64;
}

/// A batch of ragged query multi-vectors: every query's scalars concatenated,
/// plus a token-count prefix sum. `token_offsets` has `count + 1` entries, so
/// query `i` owns tokens `[token_offsets[i], token_offsets[i + 1])` and therefore
/// scalars from `token_offsets[i] * depth`. This mirrors the irregular value
/// column's own bytes-plus-offsets model rather than inventing a second one.
pub struct MaxSimBatch<'a, Scalar> {
    scalars: &'a [Scalar],
    token_offsets: &'a [usize],
    depth: usize,
}

impl<'a, Scalar: Copy> MaxSimBatch<'a, Scalar> {
    /// Borrow `scalars` as `token_offsets.len() - 1` ragged multi-vectors of
    /// width `depth`, or [`Error::Inconsistent`] if the offsets do not describe
    /// exactly that many scalars, or do not ascend.
    pub fn new(
        scalars: &'a [Scalar],
        token_offsets: &'a [usize],
        depth: usize,
    ) -> Result<Self, Error> {
        let Some(&total) = token_offsets.last() else {
            return Err(Error::Inconsistent);
        };
        if depth == 0 || total.checked_mul(depth) != Some(scalars.len()) {
            return Err(Error::Inconsistent);
        }
        if token_offsets.windows(2).any(|pair| pair[1] < pair[0]) {
            return Err(Error::Inconsistent);
        }
        Ok(MaxSimBatch {
            scalars,
            token_offsets,
            depth,
        })
    }

    pub fn depth(&self) -> usize {
        self.depth
    }
}

impl<Scalar> QueryBatchLike<MaxSimMetric<Scalar>> for MaxSimBatch<'_, Scalar>
where
    Scalar: nk::MaxSim + nk::StorageElement + Copy + Default + 'static,
{
    fn count(&self) -> usize {
        self.token_offsets.len().saturating_sub(1)
    }

    fn query(&self, index: usize) -> Option<&[Scalar]> {
        let start = *self.token_offsets.get(index)? * self.depth;
        let end = *self.token_offsets.get(index + 1)? * self.depth;
        self.scalars.get(start..end)
    }
}

/// The fixed token width of a MaxSim value column — the `depth` of every
/// multi-vector in the file. The single shape check every entry point repeats:
/// the token axis must be the ragged one, so only `Matrix(0, depth)` qualifies.
fn multi_vector_depth(table: GraphTableView<'_>) -> Result<usize, Error> {
    match table.value_shape() {
        Some(Shape::Matrix(0, depth)) if depth != 0 => Ok(depth),
        _ => Err(Error::WrongType),
    }
}

/// Pack one node's ragged multi-vector out of its raw value bytes, into `packed`.
/// The token count is the row's scalar count ÷ `depth` — the recovery the ragged
/// [`Shape`] exists to license.
fn pack_multi_vector<Scalar, Allocator>(
    bytes: &[u8],
    depth: usize,
    packed: &mut nk::MaxSimPackedMatrix<Scalar, Allocator>,
) -> Result<(), Error>
where
    Scalar: nk::MaxSim + nk::StorageElement,
    Allocator: AllocatorLike + Clone,
{
    let scalars: &[Scalar] = cast(bytes).ok_or(Error::WrongType)?;
    if depth == 0 || !scalars.len().is_multiple_of(depth) {
        return Err(Error::Inconsistent);
    }
    let view = matrix_view(scalars, depth)?;
    packed.pack_into(&view).map_err(|_| Error::Alloc)
}

// endregion: Metric

// region: Scanner

/// The batch half of the MaxSim metric: owns a thread pool, one packed form per
/// query since queries are the small side, one packed-node scratch per pool
/// thread rather than one per node, and the reusable score tile.
pub struct MaxSimScanner<Scalar: nk::MaxSim, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    packed_queries: Vec<nk::MaxSimPackedMatrix<Scalar, Allocator>>,
    workers: Vec<nk::MaxSimPackedMatrix<Scalar, Allocator>>,
    tile: Matrix<f64, Allocator>,
    alloc: Allocator,
}

impl<Scalar, Allocator> MaxSimScanner<Scalar, Allocator>
where
    Scalar: nk::MaxSim,
    Allocator: AllocatorLike + Clone,
{
    /// A scanner over the workers `scope` names, its packing scratch in `alloc`,
    /// uniform with [`DotScanner::new`]. NumKong's MaxSim kernels are CPU-only,
    /// so a GPU scope is refused.
    ///
    /// [`DotScanner::new`]: crate::search::dense::DotScanner::new
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        let _ = nk::Capabilities::cpu_enabled().configure_thread();
        Ok(MaxSimScanner {
            pool: open_pool(scope)?,
            packed_queries: Vec::new(),
            workers: Vec::new(),
            tile: Matrix::empty_in(alloc.clone()),
            alloc,
        })
    }
}

impl<Scalar, Allocator> MaxSimScanner<Scalar, Allocator>
where
    Scalar: nk::MaxSim + nk::StorageElement + Copy + Default + Send + Sync + 'static,
    <Scalar as nk::MaxSim>::Score: Into<f64>,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    /// Batch brute-force top-k, nearest-first — a late-interaction angular
    /// distance is smaller-is-better. `matches`/`distances` are `queries × k`;
    /// their shape carries both `k` and the query count.
    pub fn scan(
        &mut self,
        queries: MaxSimBatch<'_, Scalar>,
        table: &GraphTable<impl TableAllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let mut tile = std::mem::replace(&mut self.tile, Matrix::empty_in(self.alloc.clone()));
        let outcome = batch_scan::<MaxSimMetric<Scalar>, _>(
            self,
            queries,
            table,
            Ranking::Nearest,
            &mut tile,
            matches,
            distances,
            counts,
        );
        self.tile = tile;
        outcome
    }
}

impl<Scalar, Allocator> ScannerLike<MaxSimMetric<Scalar>> for MaxSimScanner<Scalar, Allocator>
where
    Scalar: nk::MaxSim + nk::StorageElement + Copy + Default + Send + Sync + 'static,
    <Scalar as nk::MaxSim>::Score: Into<f64>,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    fn scan(
        &mut self,
        queries: MaxSimBatch<'_, Scalar>,
        table: GraphTableView<'_>,
        scores: &mut Matrix<f64, impl AllocatorLike + Clone>,
    ) -> Result<(), Error> {
        let depth = multi_vector_depth(table)?;
        let node_count = table.len();
        let query_count = queries.count();
        if queries.depth() != depth || query_count != scores.cols() || node_count != scores.rows() {
            return Err(Error::Inconsistent);
        }

        // Pack the QUERY side once. Queries are the small side of a scan — tens,
        // against a table of millions — so these are the packed forms worth
        // holding, and every node reuses all of them. Buffers are repacked in
        // place across calls, so a warm scanner allocates nothing here.
        grow_packed(&mut self.packed_queries, query_count, &self.alloc)?;
        for (query_index, packed) in self.packed_queries.iter_mut().enumerate() {
            let query = <MaxSimBatch<'_, Scalar> as QueryBatchLike<MaxSimMetric<Scalar>>>::query(
                &queries,
                query_index,
            )
            .ok_or(Error::Inconsistent)?;
            let view = matrix_view(query, depth)?;
            packed.pack_into(&view).map_err(|_| Error::Alloc)?;
        }

        // One packed-node scratch per pool thread — the node side is streamed,
        // never materialized. Packing every node up front would cost
        // `nodes × packed_size`, gigabytes on a real table, to save nothing: a
        // node is read exactly once per scan either way.
        let threads = self.pool.threads_count().max(1);
        grow_packed(&mut self.workers, threads, &self.alloc)?;

        // Fan NODES across the pool: the tile is node-major, so each task owns
        // its node's contiguous row by exclusive borrow — the borrow checker
        // proves the disjointness, no raw pointer needed. Fanning over nodes
        // rather than queries also keeps the scan parallel for a single query.
        let mut rows: Vec<&mut [f64]> = Vec::new();
        rows.try_reserve_exact(node_count)
            .map_err(|_| Error::Alloc)?;
        rows.extend(scores.as_mut_slice().chunks_mut(query_count.max(1)));
        let table_shared = ShareRead(&table);
        let packed_queries = &self.packed_queries;
        let failure = OnceLock::new();

        fu::fold_with_scratch(
            &mut self.pool,
            fu::ParallelSliceMut::new(&mut rows),
            fu::DynamicScheduler,
            self.workers.as_mut_slice(),
            |packed_node, row, task, _thread| {
                let outcome = scan_one_node::<Scalar, Allocator>(
                    packed_node,
                    *table_shared.read(),
                    task,
                    depth,
                    packed_queries,
                    row,
                );
                if let Err(error) = outcome {
                    let _ = failure.set(error);
                }
            },
        )
        .map_err(|_| Error::Alloc)?;

        match failure.into_inner() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// Grow `packed` to hold `count` reusable packed matrices, never shrinking the
/// buffers already there — the decode-loop reuse contract NumKong's own
/// `try_reserve`/`pack_into` pair is built for. Fallible throughout, so an
/// out-of-memory scan returns [`Error::Alloc`] instead of aborting on a
/// `Vec::push`.
fn grow_packed<Scalar, Allocator>(
    packed: &mut Vec<nk::MaxSimPackedMatrix<Scalar, Allocator>>,
    count: usize,
    alloc: &Allocator,
) -> Result<(), Error>
where
    Scalar: nk::MaxSim,
    Allocator: AllocatorLike + Clone,
{
    if packed.len() < count {
        packed
            .try_reserve(count - packed.len())
            .map_err(|_| Error::Alloc)?;
        while packed.len() < count {
            packed.push(nk::MaxSimPackedMatrix::empty_in(alloc.clone()));
        }
    }
    packed.truncate(count);
    Ok(())
}

/// One node's pack and score against every packed query — the body of the
/// scan's fan-out task, handed the node's contiguous tile row by exclusive
/// borrow. The node is packed into this thread's scratch, scored, and then
/// overwritten by the next node the thread takes.
fn scan_one_node<Scalar, Allocator>(
    packed_node: &mut nk::MaxSimPackedMatrix<Scalar, Allocator>,
    table: GraphTableView<'_>,
    node: usize,
    depth: usize,
    packed_queries: &[nk::MaxSimPackedMatrix<Scalar, Allocator>],
    row: &mut [f64],
) -> Result<(), Error>
where
    Scalar: nk::MaxSim + nk::StorageElement + Copy + Default + 'static,
    <Scalar as nk::MaxSim>::Score: Into<f64>,
    Allocator: AllocatorLike + Clone,
{
    // The value column is ragged, so the node's tokens come through the table's
    // own per-node accessor rather than a strided flat slice.
    pack_multi_vector(table.value(node), depth, packed_node)?;
    for (cell, packed_query) in row.iter_mut().zip(packed_queries) {
        *cell = packed_query
            .score(packed_node)
            .map_err(|_| Error::Inconsistent)?
            .into();
    }
    Ok(())
}

// endregion: Scanner

// region: Searcher

/// The traversal half of the MaxSim metric: one query at a fixed node width,
/// packing the query once per walk and scoring each staged node against it.
pub struct MaxSimWalker<Scalar: nk::MaxSim, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, f64, Allocator>,
    packed_query: nk::MaxSimPackedMatrix<Scalar, Allocator>,
    packed_node: nk::MaxSimPackedMatrix<Scalar, Allocator>,
}

impl<Scalar, Index, Allocator> MaxSimWalker<Scalar, Index, Allocator>
where
    Scalar: nk::MaxSim,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
{
    /// A searcher walking with breadth `expansion_factor`, its scratch in
    /// `alloc`, uniform with [`DotWalker::new`].
    ///
    /// [`DotWalker::new`]: crate::search::dense::DotWalker::new
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        let _ = nk::Capabilities::cpu_enabled().configure_thread();
        Ok(MaxSimWalker {
            context: TraversalContext::new_in(expansion_factor, alloc.clone())?,
            packed_query: nk::MaxSimPackedMatrix::empty_in(alloc.clone()),
            packed_node: nk::MaxSimPackedMatrix::empty_in(alloc),
        })
    }

    /// The traversal scratch, for draining results after a walk.
    pub fn context(&mut self) -> &mut TraversalContext<Index, f64, Allocator> {
        &mut self.context
    }
}

impl<Scalar, Index, Allocator> WalkerLike<MaxSimMetric<Scalar>>
    for MaxSimWalker<Scalar, Index, Allocator>
where
    Scalar: nk::MaxSim + nk::StorageElement + Copy + Default + 'static,
    <Scalar as nk::MaxSim>::Score: Into<f64>,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        query: &[Scalar],
        table: GraphTableView<'_>,
        nodes: &[Index],
        out: &mut [f64],
    ) -> Result<(), Error> {
        if nodes.len() != out.len() {
            return Err(Error::Inconsistent);
        }
        let depth = multi_vector_depth(table)?;
        if depth == 0 || !query.len().is_multiple_of(depth) {
            return Err(Error::Inconsistent);
        }
        let view = matrix_view(query, depth)?;
        self.packed_query
            .pack_into(&view)
            .map_err(|_| Error::Alloc)?;

        for (slot, &node) in out.iter_mut().zip(nodes) {
            let index: usize = node.try_into().map_err(|_| Error::TooLarge)?;
            pack_multi_vector(table.value(index), depth, &mut self.packed_node)?;
            *slot = self
                .packed_query
                .score(&self.packed_node)
                .map_err(|_| Error::Inconsistent)?
                .into();
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, f64, Allocator> {
        &mut self.context
    }

    fn compare(a: f64, b: f64) -> Ordering {
        Ranking::Nearest.compare(a, b)
    }

    fn search(&mut self, query: &[Scalar], table: GraphTableView<'_>) -> Result<(), Error> {
        walk::<MaxSimMetric<Scalar>, Self>(self, query, table)
    }
}

/// Batched MaxSim walks, resolving the node width from the first table
/// searched and holding one [`MaxSimWalker`] per pool thread.
pub struct MaxSimSearcher<Scalar: nk::MaxSim, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers: Workers<
        MaxSimWalker<Scalar, u32, Allocator>,
        MaxSimWalker<Scalar, u64, Allocator>,
        Allocator,
    >,
    alloc: Allocator,
}

impl<Scalar, Allocator> MaxSimSearcher<Scalar, Allocator>
where
    Scalar: nk::MaxSim + nk::StorageElement + Copy + Default + Send + Sync + 'static,
    <Scalar as nk::MaxSim>::Score: Into<f64>,
    Allocator: AllocatorLike + Clone + Send,
{
    /// A searcher walking with breadth `expansion_factor` across the workers
    /// `scope` names, uniform with [`DotSearcher::new`].
    ///
    /// [`DotSearcher::new`]: crate::search::dense::DotSearcher::new
    pub fn new(
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(MaxSimSearcher {
            pool: open_pool(scope)?,
            expansion_factor,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, fanned across the pool. Uniform with
    /// [`MaxSimScanner::scan`].
    pub fn search(
        &mut self,
        queries: MaxSimBatch<'_, Scalar>,
        table: &GraphTable<impl TableAllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let threads = self.pool.threads_count();
        let expansion_factor = self.expansion_factor;
        match table.neighbors_dtype() {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(threads, alloc.clone(), || {
                        MaxSimWalker::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<MaxSimMetric<Scalar>, _>(
                    &mut self.pool,
                    workers.as_mut_slice(),
                    queries,
                    table,
                    matches,
                    distances,
                    counts,
                )
            }
            OffsetDType::U64 => {
                if !matches!(self.workers, Workers::U64(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U64(build_workers(threads, alloc.clone(), || {
                        MaxSimWalker::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<MaxSimMetric<Scalar>, _>(
                    &mut self.pool,
                    workers.as_mut_slice(),
                    queries,
                    table,
                    matches,
                    distances,
                    counts,
                )
            }
        }
    }
}

// endregion: Searcher

// region: Runtime dtype dispatch

/// A MaxSim scanner chosen from a file's value dtype string — multi-vectors come
/// in the three scalars NumKong's `maxsim.h` packs.
pub enum MaxSimScannerAny<Allocator: AllocatorLike = Global> {
    F32(MaxSimScanner<f32, Allocator>),
    F16(MaxSimScanner<nk::f16, Allocator>),
    Bf16(MaxSimScanner<nk::bf16, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send + Sync> MaxSimScannerAny<Allocator> {
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "f32" => MaxSimScannerAny::F32(MaxSimScanner::new(scope, alloc)?),
            "f16" => MaxSimScannerAny::F16(MaxSimScanner::new(scope, alloc)?),
            "bf16" => MaxSimScannerAny::Bf16(MaxSimScanner::new(scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    /// Scan raw query bytes plus their token-offset prefix sum — the ragged input
    /// shape the router hands every multi-vector family.
    pub fn scan(
        &mut self,
        query_bytes: &[u8],
        token_offsets: &[usize],
        table: &GraphTable<impl TableAllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let depth = multi_vector_depth(table.view())?;
        match self {
            MaxSimScannerAny::F32(scanner) => scanner.scan(
                MaxSimBatch::new(
                    cast(query_bytes).ok_or(Error::WrongType)?,
                    token_offsets,
                    depth,
                )?,
                table,
                matches,
                distances,
                counts,
            ),
            MaxSimScannerAny::F16(scanner) => scanner.scan(
                MaxSimBatch::new(
                    cast(query_bytes).ok_or(Error::WrongType)?,
                    token_offsets,
                    depth,
                )?,
                table,
                matches,
                distances,
                counts,
            ),
            MaxSimScannerAny::Bf16(scanner) => scanner.scan(
                MaxSimBatch::new(
                    cast(query_bytes).ok_or(Error::WrongType)?,
                    token_offsets,
                    depth,
                )?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// A MaxSim searcher chosen from a file's value dtype string — the walk twin of
/// [`MaxSimScannerAny`].
pub enum MaxSimSearcherAny<Allocator: AllocatorLike = Global> {
    F32(MaxSimSearcher<f32, Allocator>),
    F16(MaxSimSearcher<nk::f16, Allocator>),
    Bf16(MaxSimSearcher<nk::bf16, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send + Sync> MaxSimSearcherAny<Allocator> {
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f32" => MaxSimSearcherAny::F32(MaxSimSearcher::new(expansion_factor, scope, alloc)?),
            "f16" => MaxSimSearcherAny::F16(MaxSimSearcher::new(expansion_factor, scope, alloc)?),
            "bf16" => MaxSimSearcherAny::Bf16(MaxSimSearcher::new(expansion_factor, scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn search(
        &mut self,
        query_bytes: &[u8],
        token_offsets: &[usize],
        table: &GraphTable<impl TableAllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let depth = multi_vector_depth(table.view())?;
        match self {
            MaxSimSearcherAny::F32(search) => search.search(
                MaxSimBatch::new(
                    cast(query_bytes).ok_or(Error::WrongType)?,
                    token_offsets,
                    depth,
                )?,
                table,
                matches,
                distances,
                counts,
            ),
            MaxSimSearcherAny::F16(search) => search.search(
                MaxSimBatch::new(
                    cast(query_bytes).ok_or(Error::WrongType)?,
                    token_offsets,
                    depth,
                )?,
                table,
                matches,
                distances,
                counts,
            ),
            MaxSimSearcherAny::Bf16(search) => search.search(
                MaxSimBatch::new(
                    cast(query_bytes).ok_or(Error::WrongType)?,
                    token_offsets,
                    depth,
                )?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

// endregion: Runtime dtype dispatch

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::testing::outputs;
    use crate::{GraphLike, NeighborList, Shape};

    /// Three ragged multi-vectors at `depth = 2`, so the value column is
    /// irregular and its shape is `Matrix(0, 2)` — the ragged token axis this
    /// family exists to read. Node 0 holds `[[1,0],[0,1]]`, node 1 the single
    /// token `[[1,0]]`, node 2 the opposites `[[-1,0],[0,-1]]`.
    struct MultiVectors {
        values: [Vec<f32>; 3],
    }

    fn multi_vectors() -> MultiVectors {
        MultiVectors {
            values: [
                vec![1.0, 0.0, 0.0, 1.0],
                vec![1.0, 0.0],
                vec![-1.0, 0.0, 0.0, -1.0],
            ],
        }
    }

    impl GraphLike for MultiVectors {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "f32"
        }
        fn value_shape(&self) -> Option<Shape> {
            Some(Shape::Matrix(0, 2))
        }
        fn neighbors_degree(&self) -> Option<usize> {
            Some(2)
        }
        fn key(&self, i: usize) -> &[u8] {
            const KEYS: [[u8; 8]; 3] = [
                (10u64).to_le_bytes(),
                (20u64).to_le_bytes(),
                (30u64).to_le_bytes(),
            ];
            &KEYS[i]
        }
        fn value(&self, i: usize) -> &[u8] {
            let row = &self.values[i];
            // SAFETY: reinterpret the contiguous `f32` tokens as their bytes.
            unsafe {
                core::slice::from_raw_parts(
                    row.as_ptr() as *const u8,
                    core::mem::size_of_val(row.as_slice()),
                )
            }
        }
        fn neighbors(&self, i: usize) -> NeighborList<'_> {
            const ADJACENCY: [[u8; 8]; 3] = [
                [1, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ];
            NeighborList::new(&ADJACENCY[i], OffsetDType::U32)
        }
    }

    /// The ragged shape survives a build/parse round-trip — the property that
    /// unblocked this family. A regular column could not express it.
    #[test]
    fn ragged_shape_round_trips() {
        let table = crate::GraphTable::build(&multi_vectors(), None).unwrap();
        assert_eq!(table.value_shape(), Some(Shape::Matrix(0, 2)));
    }

    /// The query `[[1,0]]` matches nodes 0 and 1 exactly (angular 0) and is
    /// orthogonal-or-opposite to node 2, which must therefore rank last.
    #[test]
    fn scan_ranks_the_opposed_multi_vector_last() {
        let table = crate::GraphTable::build(&multi_vectors(), None).unwrap();
        let mut scanner = MaxSimScanner::<f32>::new(ExecutionScope::Serial, Global).unwrap();

        let scalars = [1.0f32, 0.0];
        let offsets = [0usize, 1];
        let queries = MaxSimBatch::new(&scalars, &offsets, 2).unwrap();
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
        scanner
            .scan(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();

        assert_eq!(counts, [3]);
        let row = distances.row(0).unwrap();
        assert_eq!(matches.row(0).unwrap()[2], 2);
        assert!(row[0] <= row[1] && row[1] < row[2]);
        assert!(row[0].abs() < 1e-5);
    }

    #[test]
    fn dtype_dispatch_covers_the_packable_scalars() {
        for dtype in ["f32", "f16", "bf16"] {
            assert!(MaxSimScannerAny::new(dtype, ExecutionScope::Serial, Global).is_ok());
        }
        assert!(MaxSimScannerAny::new("f64", ExecutionScope::Serial, Global).is_err());
    }
}

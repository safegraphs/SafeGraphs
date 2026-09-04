//! Geospatial metrics over NumKong. No StringZilla type appears here.
//!
//! A node's value is a two-wide `[latitude, longitude]` pair in `f32` or `f64`,
//! stored in __radians__ — the kernels' own convention, adopted rather than
//! converted — and every score is a distance in __meters__, so the family ranks
//! nearest-first like Euclidean. Two metrics share one generic body through the
//! [`KernelLike`] seam: [`HaversineKernel`], the spherical great-circle
//! distance, and [`VincentyKernel`], the WGS84 geodesic, ~20× more accurate at
//! long range.
//!
//! The kernels are element-wise over parallel coordinate slices — there is no
//! packed cross-product analogue of `DotsPackedMatrix` — so the scan builds the
//! product itself: the node column is deinterleaved once per call into
//! latitude/longitude slices, and each query is broadcast against them in one
//! kernel call, filling that query's tile column. Queries fan across the owned
//! pool with per-worker broadcast scratch, the same schedule the batch walk
//! uses.

use super::batch::{batch_scan, batch_search, build_workers, open_pool, ShareRead, Workers};
use super::{
    walk, ExecutionScope, Matrix, MetricLike, QueryBatchLike, Ranking, ScannerLike,
    TraversalContext, WalkerLike,
};
use crate::search::dense::view_from_bytes;
use crate::{cast, Error, GraphTable, GraphTableView, IndexLike, OffsetDType, ScoreLike};
use allocator_api2::alloc::{Allocator as AllocatorLike, Global};
use allocator_api2::vec::Vec as AVec;
use core::cmp::Ordering;
use forkunion as fu;
use numkong as nk;
use std::sync::OnceLock;

// region: Kernel seam

/// The seam between the shared geospatial machinery and one NumKong kernel: a
/// batch of pairwise distances over parallel coordinate slices, radians in,
/// meters out. Implemented by unit marker types, so [`GeospatialScanner`] and
/// friends span both metrics the way `DotMetric<Value>` spans the dense scalars.
///
/// [`DotMetric<Value>`]: crate::search::dense::DotMetric
pub trait KernelLike<Coordinate>: 'static {
    /// One distance per element, all five slices the same length. `None` is a
    /// length mismatch, which the callers turn into [`Error::Inconsistent`].
    fn distances(
        first_latitudes: &[Coordinate],
        first_longitudes: &[Coordinate],
        second_latitudes: &[Coordinate],
        second_longitudes: &[Coordinate],
        out: &mut [Coordinate],
    ) -> Option<()>;
}

/// Spherical great-circle distance — fast, within ~0.5% on Earth.
pub struct HaversineKernel;

impl<Coordinate: nk::Haversine> KernelLike<Coordinate> for HaversineKernel {
    fn distances(
        first_latitudes: &[Coordinate],
        first_longitudes: &[Coordinate],
        second_latitudes: &[Coordinate],
        second_longitudes: &[Coordinate],
        out: &mut [Coordinate],
    ) -> Option<()> {
        Coordinate::haversine(
            first_latitudes,
            first_longitudes,
            second_latitudes,
            second_longitudes,
            out,
        )
    }
}

/// WGS84 ellipsoid geodesic by Vincenty's iteration — slower, survey-grade.
pub struct VincentyKernel;

impl<Coordinate: nk::Vincenty> KernelLike<Coordinate> for VincentyKernel {
    fn distances(
        first_latitudes: &[Coordinate],
        first_longitudes: &[Coordinate],
        second_latitudes: &[Coordinate],
        second_longitudes: &[Coordinate],
        out: &mut [Coordinate],
    ) -> Option<()> {
        Coordinate::vincenty(
            first_latitudes,
            first_longitudes,
            second_latitudes,
            second_longitudes,
            out,
        )
    }
}

// endregion: Kernel seam

/// A geospatial metric's type vocabulary, one marker per kernel and scalar. A
/// query is one `[latitude, longitude]` row and a batch is a `queries × 2`
/// matrix view, so the dense family's [`QueryBatchLike`] impl serves unchanged.
pub struct GeospatialMetric<Kernel, Coordinate>(
    core::marker::PhantomData<(fn() -> Kernel, Coordinate)>,
);

impl<Kernel: KernelLike<Coordinate>, Coordinate: Copy + Default + 'static> MetricLike
    for GeospatialMetric<Kernel, Coordinate>
{
    type Query<'a> = &'a [Coordinate];
    type Queries<'a> = nk::MatrixView<'a, Coordinate>;
    type Score = Coordinate;
}

/// Deinterleave the node column into two coordinate slices, with the shape
/// checks every entry point repeats: the value column must be exactly two
/// scalars wide, or the table is not geospatial at all. Returns the node count.
fn coordinate_columns<Coordinate: nk::StorageElement + Copy>(
    table: GraphTableView<'_>,
    node_latitudes: &mut AVec<Coordinate, impl AllocatorLike>,
    node_longitudes: &mut AVec<Coordinate, impl AllocatorLike>,
) -> Result<usize, Error> {
    if table.value_width() != Some(2) {
        return Err(Error::WrongType);
    }
    let scalars: &[Coordinate] = cast(table.value_buffer()).ok_or(Error::WrongType)?;
    let node_count = table.len();
    if scalars.len() != node_count * 2 {
        return Err(Error::Inconsistent);
    }
    node_latitudes.clear();
    node_longitudes.clear();
    node_latitudes
        .try_reserve(node_count)
        .map_err(|_| Error::Alloc)?;
    node_longitudes
        .try_reserve(node_count)
        .map_err(|_| Error::Alloc)?;
    for pair in scalars.chunks_exact(2) {
        node_latitudes.push(pair[0]);
        node_longitudes.push(pair[1]);
    }
    Ok(node_count)
}

/// One worker's broadcast scratch: a query's coordinate repeated across the
/// node count, and the kernel's output row.
struct BroadcastScratch<Coordinate, Allocator: AllocatorLike> {
    query_latitudes: AVec<Coordinate, Allocator>,
    query_longitudes: AVec<Coordinate, Allocator>,
    distances: AVec<Coordinate, Allocator>,
}

impl<Coordinate: Copy + Default, Allocator: AllocatorLike + Clone>
    BroadcastScratch<Coordinate, Allocator>
{
    fn new_in(alloc: Allocator) -> Self {
        BroadcastScratch {
            query_latitudes: AVec::new_in(alloc.clone()),
            query_longitudes: AVec::new_in(alloc.clone()),
            distances: AVec::new_in(alloc),
        }
    }

    /// Fill the broadcast slices with one query's coordinate, `count` wide.
    fn broadcast(
        &mut self,
        latitude: Coordinate,
        longitude: Coordinate,
        count: usize,
    ) -> Result<(), Error> {
        self.query_latitudes.clear();
        self.query_longitudes.clear();
        self.distances.clear();
        self.query_latitudes
            .try_reserve(count)
            .map_err(|_| Error::Alloc)?;
        self.query_longitudes
            .try_reserve(count)
            .map_err(|_| Error::Alloc)?;
        self.distances
            .try_reserve(count)
            .map_err(|_| Error::Alloc)?;
        self.query_latitudes.resize(count, latitude);
        self.query_longitudes.resize(count, longitude);
        self.distances.resize(count, Coordinate::default());
        Ok(())
    }
}

/// The batch half of any geospatial metric: owns a thread pool, the
/// deinterleaved node columns, and one broadcast scratch per worker. There is
/// no packed form to reuse, so every scan deinterleaves the mapped column
/// afresh — one linear pass, dwarfed by the trigonometry behind it.
pub struct GeospatialScanner<Kernel, Coordinate, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    node_latitudes: AVec<Coordinate, Allocator>,
    node_longitudes: AVec<Coordinate, Allocator>,
    workers: AVec<BroadcastScratch<Coordinate, Allocator>, Allocator>,
    tile: Matrix<Coordinate, Allocator>,
    alloc: Allocator,
    kernel: core::marker::PhantomData<fn() -> Kernel>,
}

impl<Kernel, Coordinate, Allocator> GeospatialScanner<Kernel, Coordinate, Allocator>
where
    Kernel: KernelLike<Coordinate>,
    Coordinate: Copy + Default,
    Allocator: AllocatorLike + Clone,
{
    /// A scanner over the workers `scope` names, its scratch in `alloc`,
    /// uniform with [`DotScanner::new`]. NumKong is CPU-only here, so a GPU
    /// scope is refused.
    ///
    /// [`DotScanner::new`]: crate::search::dense::DotScanner::new
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(GeospatialScanner {
            pool: open_pool(scope)?,
            node_latitudes: AVec::new_in(alloc.clone()),
            node_longitudes: AVec::new_in(alloc.clone()),
            workers: AVec::new_in(alloc.clone()),
            tile: Matrix::empty_in(alloc.clone()),
            alloc,
            kernel: core::marker::PhantomData,
        })
    }
}

impl<Kernel, Coordinate, Allocator> GeospatialScanner<Kernel, Coordinate, Allocator>
where
    Kernel: KernelLike<Coordinate>,
    Coordinate:
        nk::StorageElement + Copy + Default + PartialOrd + ScoreLike + Send + Sync + 'static,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    /// Batch brute-force top-k, nearest-first — a great-circle distance is
    /// smaller-is-better. `matches`/`distances` are `queries × k`; their shape
    /// carries both `k` and the query count, uniform with every dense scanner.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Coordinate>,
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let mut tile = std::mem::replace(&mut self.tile, Matrix::empty_in(self.alloc.clone()));
        let outcome = batch_scan::<GeospatialMetric<Kernel, Coordinate>, _>(
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

impl<Kernel, Coordinate, Allocator> ScannerLike<GeospatialMetric<Kernel, Coordinate>>
    for GeospatialScanner<Kernel, Coordinate, Allocator>
where
    Kernel: KernelLike<Coordinate>,
    Coordinate: nk::StorageElement + Copy + Default + Send + Sync + 'static,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Coordinate>,
        table: GraphTableView<'_>,
        scores: &mut Matrix<Coordinate, impl AllocatorLike + Clone>,
    ) -> Result<(), Error> {
        let node_count =
            coordinate_columns(table, &mut self.node_latitudes, &mut self.node_longitudes)?;
        let query_count = queries.shape().first().copied().unwrap_or(0);
        if queries.shape().get(1) != Some(&2)
            || query_count != scores.cols()
            || node_count != scores.rows()
        {
            return Err(Error::Inconsistent);
        }

        // One broadcast scratch per pool thread, grown on demand and reused.
        let threads = self.pool.threads_count();
        while self.workers.len() < threads {
            self.workers
                .push(BroadcastScratch::new_in(self.alloc.clone()));
        }

        // Fan node CHUNKS across the pool: each task owns a contiguous block of
        // tile rows by exclusive borrow — the borrow checker proves the
        // disjointness, so the fan-out carries no raw tile pointer. Per query
        // the task runs one kernel call over its whole node sub-range, so the
        // batch length stays a chunk, not a query count: a single-query scan
        // still feeds the SIMD lanes full spans. Four chunks per worker give
        // the dynamic schedule slack to balance.
        let chunk_nodes = node_count.div_ceil(threads.max(1) * 4).max(1);
        let mut row_chunks: Vec<&mut [Coordinate]> = Vec::new();
        row_chunks
            .try_reserve_exact(node_count.div_ceil(chunk_nodes))
            .map_err(|_| Error::Alloc)?;
        row_chunks.extend(scores.as_mut_slice().chunks_mut(chunk_nodes * query_count));
        let (node_latitudes, node_longitudes) = (&self.node_latitudes, &self.node_longitudes);
        let queries_shared = ShareRead(&queries);
        let failure = OnceLock::new();

        fu::fold_with_scratch(
            &mut self.pool,
            fu::ParallelSliceMut::new(&mut row_chunks),
            fu::DynamicScheduler,
            self.workers.as_mut_slice(),
            |worker, chunk, prong| {
                let queries = queries_shared.read();
                let start = prong.task_index * chunk_nodes;
                let outcome = scan_node_chunk::<Kernel, Coordinate>(
                    worker,
                    queries,
                    &node_latitudes[start..],
                    &node_longitudes[start..],
                    chunk,
                    query_count,
                );
                if let Err(error) = outcome {
                    let _ = failure.set(error);
                }
            },
        );

        match failure.into_inner() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// One node chunk's whole block of tile rows — the body of the scan's fan-out
/// task. Per query: broadcast the query across the chunk, one kernel call over
/// the chunk's node sub-range, then write that query's column of the owned
/// block — strided, but exclusively borrowed, so every write is safe.
fn scan_node_chunk<Kernel, Coordinate>(
    worker: &mut BroadcastScratch<Coordinate, impl AllocatorLike + Clone>,
    queries: &nk::MatrixView<'_, Coordinate>,
    node_latitudes: &[Coordinate],
    node_longitudes: &[Coordinate],
    chunk: &mut [Coordinate],
    query_count: usize,
) -> Result<(), Error>
where
    Kernel: KernelLike<Coordinate>,
    Coordinate: nk::StorageElement + Copy + Default + 'static,
{
    let chunk_nodes = chunk.len() / query_count.max(1);
    let node_latitudes = &node_latitudes[..chunk_nodes];
    let node_longitudes = &node_longitudes[..chunk_nodes];
    for query_index in 0..query_count {
        let query = <nk::MatrixView<'_, Coordinate> as QueryBatchLike<
            GeospatialMetric<Kernel, Coordinate>,
        >>::query(queries, query_index)
        .ok_or(Error::Inconsistent)?;
        worker.broadcast(query[0], query[1], chunk_nodes)?;
        Kernel::distances(
            &worker.query_latitudes,
            &worker.query_longitudes,
            node_latitudes,
            node_longitudes,
            &mut worker.distances,
        )
        .ok_or(Error::Inconsistent)?;
        for (node, &distance) in worker.distances.iter().enumerate() {
            chunk[node * query_count + query_index] = distance;
        }
    }
    Ok(())
}

/// The traversal half of the geospatial metric: one query at a fixed node width,
/// gathering a staged expansion's scattered coordinate pairs into one kernel call.
pub struct GeospatialWalker<Kernel, Coordinate, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, Coordinate, Allocator>,
    staged_latitudes: AVec<Coordinate, Allocator>,
    staged_longitudes: AVec<Coordinate, Allocator>,
    broadcast: BroadcastScratch<Coordinate, Allocator>,
    kernel: core::marker::PhantomData<fn() -> Kernel>,
}

impl<Kernel, Coordinate, Index, Allocator> GeospatialWalker<Kernel, Coordinate, Index, Allocator>
where
    Kernel: KernelLike<Coordinate>,
    Coordinate: Copy + Default,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
{
    /// A searcher walking with breadth `expansion_factor`, its scratch in
    /// `alloc`, uniform with [`DotWalker::new`].
    ///
    /// [`DotWalker::new`]: crate::search::dense::DotWalker::new
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        Ok(GeospatialWalker {
            context: TraversalContext::new_in(expansion_factor, alloc.clone())?,
            staged_latitudes: AVec::new_in(alloc.clone()),
            staged_longitudes: AVec::new_in(alloc.clone()),
            broadcast: BroadcastScratch::new_in(alloc),
            kernel: core::marker::PhantomData,
        })
    }

    /// The traversal scratch, for draining results after a walk.
    pub fn context(&mut self) -> &mut TraversalContext<Index, Coordinate, Allocator> {
        &mut self.context
    }
}

impl<Kernel, Coordinate, Index, Allocator> WalkerLike<GeospatialMetric<Kernel, Coordinate>>
    for GeospatialWalker<Kernel, Coordinate, Index, Allocator>
where
    Kernel: KernelLike<Coordinate>,
    Coordinate: nk::StorageElement + Copy + Default + PartialOrd + 'static,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        query: &[Coordinate],
        table: GraphTableView<'_>,
        nodes: &[Index],
        out: &mut [Coordinate],
    ) -> Result<(), Error> {
        if nodes.len() != out.len() || query.len() != 2 {
            return Err(Error::Inconsistent);
        }
        if nodes.is_empty() {
            return Ok(());
        }

        self.staged_latitudes.clear();
        self.staged_longitudes.clear();
        self.staged_latitudes
            .try_reserve(nodes.len())
            .map_err(|_| Error::Alloc)?;
        self.staged_longitudes
            .try_reserve(nodes.len())
            .map_err(|_| Error::Alloc)?;
        for &node in nodes {
            let index: usize = node.try_into().map_err(|_| Error::TooLarge)?;
            let pair: &[Coordinate] = cast(table.value(index)).ok_or(Error::WrongType)?;
            if pair.len() != 2 {
                return Err(Error::Inconsistent);
            }
            self.staged_latitudes.push(pair[0]);
            self.staged_longitudes.push(pair[1]);
        }

        self.broadcast.broadcast(query[0], query[1], nodes.len())?;
        Kernel::distances(
            &self.broadcast.query_latitudes,
            &self.broadcast.query_longitudes,
            &self.staged_latitudes,
            &self.staged_longitudes,
            &mut self.broadcast.distances,
        )
        .ok_or(Error::Inconsistent)?;
        out.copy_from_slice(&self.broadcast.distances);
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, Coordinate, Allocator> {
        &mut self.context
    }

    fn compare(a: Coordinate, b: Coordinate) -> Ordering {
        Ranking::Nearest.compare(a, b)
    }

    fn search(&mut self, query: &[Coordinate], table: GraphTableView<'_>) -> Result<(), Error> {
        walk::<GeospatialMetric<Kernel, Coordinate>, Self>(self, query, table)
    }
}

/// Batched geospatial walks, resolving the node width from the first table
/// searched and holding one [`GeospatialWalker`] per pool thread.
pub struct GeospatialSearcher<Kernel, Coordinate, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers: Workers<
        GeospatialWalker<Kernel, Coordinate, u32, Allocator>,
        GeospatialWalker<Kernel, Coordinate, u64, Allocator>,
        Allocator,
    >,
    alloc: Allocator,
}

impl<Kernel, Coordinate, Allocator> GeospatialSearcher<Kernel, Coordinate, Allocator>
where
    Kernel: KernelLike<Coordinate>,
    Coordinate:
        nk::StorageElement + Copy + Default + PartialOrd + ScoreLike + Send + Sync + 'static,
    Allocator: AllocatorLike + Clone + Send,
{
    /// A searcher walking with breadth `expansion_factor` across the workers
    /// `scope` names, its scratch in `alloc`, uniform with [`DotSearcher::new`].
    ///
    /// [`DotSearcher::new`]: crate::search::dense::DotSearcher::new
    pub fn new(
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(GeospatialSearcher {
            pool: open_pool(scope)?,
            expansion_factor,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, fanned across the pool. Uniform with
    /// [`GeospatialScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, Coordinate>,
        table: &GraphTable<impl AllocatorLike>,
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
                        GeospatialWalker::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<GeospatialMetric<Kernel, Coordinate>, _>(
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
                        GeospatialWalker::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<GeospatialMetric<Kernel, Coordinate>, _>(
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

// region: Metric aliases

/// Spherical great-circle distance, nearest-first, meters.
pub type HaversineMetric<Coordinate> = GeospatialMetric<HaversineKernel, Coordinate>;
/// Exact brute force over the Haversine distance.
pub type HaversineScanner<Coordinate, Allocator = Global> =
    GeospatialScanner<HaversineKernel, Coordinate, Allocator>;
/// One query's greedy walk under the Haversine distance.
pub type HaversineWalker<Coordinate, Index = u32, Allocator = Global> =
    GeospatialWalker<HaversineKernel, Coordinate, Index, Allocator>;
/// Batched walks under the Haversine distance.
pub type HaversineSearcher<Coordinate, Allocator = Global> =
    GeospatialSearcher<HaversineKernel, Coordinate, Allocator>;

/// WGS84 geodesic distance, nearest-first, meters.
pub type VincentyMetric<Coordinate> = GeospatialMetric<VincentyKernel, Coordinate>;
/// Exact brute force over the Vincenty distance.
pub type VincentyScanner<Coordinate, Allocator = Global> =
    GeospatialScanner<VincentyKernel, Coordinate, Allocator>;
/// One query's greedy walk under the Vincenty distance.
pub type VincentyWalker<Coordinate, Index = u32, Allocator = Global> =
    GeospatialWalker<VincentyKernel, Coordinate, Index, Allocator>;
/// Batched walks under the Vincenty distance.
pub type VincentySearcher<Coordinate, Allocator = Global> =
    GeospatialSearcher<VincentyKernel, Coordinate, Allocator>;

// endregion: Metric aliases

// region: Runtime dtype dispatch

/// A Haversine scanner chosen from a file's value dtype string — coordinates
/// come in `f64` or `f32` alone, so the family's dtype axis is two arms.
pub enum HaversineScannerAny<Allocator: AllocatorLike = Global> {
    F64(HaversineScanner<f64, Allocator>),
    F32(HaversineScanner<f32, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send + Sync> HaversineScannerAny<Allocator> {
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => HaversineScannerAny::F64(HaversineScanner::new(scope, alloc)?),
            "f32" => HaversineScannerAny::F32(HaversineScanner::new(scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            HaversineScannerAny::F64(_) => "f64",
            HaversineScannerAny::F32(_) => "f32",
        }
    }

    /// Batch brute-force top-k over a query batch given as raw bytes in the
    /// table's dtype — `queries × 2` coordinates, radians.
    pub fn scan(
        &mut self,
        queries: &[u8],
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            HaversineScannerAny::F64(scanner) => scanner.scan(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
            HaversineScannerAny::F32(scanner) => scanner.scan(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// The batch-walk twin of [`HaversineScannerAny`].
pub enum HaversineSearcherAny<Allocator: AllocatorLike = Global> {
    F64(HaversineSearcher<f64, Allocator>),
    F32(HaversineSearcher<f32, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send> HaversineSearcherAny<Allocator> {
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => {
                HaversineSearcherAny::F64(HaversineSearcher::new(expansion_factor, scope, alloc)?)
            }
            "f32" => {
                HaversineSearcherAny::F32(HaversineSearcher::new(expansion_factor, scope, alloc)?)
            }
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            HaversineSearcherAny::F64(_) => "f64",
            HaversineSearcherAny::F32(_) => "f32",
        }
    }

    pub fn search(
        &mut self,
        queries: &[u8],
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            HaversineSearcherAny::F64(search) => search.search(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
            HaversineSearcherAny::F32(search) => search.search(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// The Vincenty twin of [`HaversineScannerAny`].
pub enum VincentyScannerAny<Allocator: AllocatorLike = Global> {
    F64(VincentyScanner<f64, Allocator>),
    F32(VincentyScanner<f32, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send + Sync> VincentyScannerAny<Allocator> {
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => VincentyScannerAny::F64(VincentyScanner::new(scope, alloc)?),
            "f32" => VincentyScannerAny::F32(VincentyScanner::new(scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            VincentyScannerAny::F64(_) => "f64",
            VincentyScannerAny::F32(_) => "f32",
        }
    }

    pub fn scan(
        &mut self,
        queries: &[u8],
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            VincentyScannerAny::F64(scanner) => scanner.scan(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
            VincentyScannerAny::F32(scanner) => scanner.scan(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// The batch-walk twin of [`VincentyScannerAny`].
pub enum VincentySearcherAny<Allocator: AllocatorLike = Global> {
    F64(VincentySearcher<f64, Allocator>),
    F32(VincentySearcher<f32, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send> VincentySearcherAny<Allocator> {
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => {
                VincentySearcherAny::F64(VincentySearcher::new(expansion_factor, scope, alloc)?)
            }
            "f32" => {
                VincentySearcherAny::F32(VincentySearcher::new(expansion_factor, scope, alloc)?)
            }
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            VincentySearcherAny::F64(_) => "f64",
            VincentySearcherAny::F32(_) => "f32",
        }
    }

    pub fn search(
        &mut self,
        queries: &[u8],
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            VincentySearcherAny::F64(search) => search.search(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
            VincentySearcherAny::F32(search) => search.search(
                view_from_bytes(queries, table.view())?,
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

    /// Three cities on a fully connected triangle, coordinates in radians:
    /// London, Paris, New York. Every metric below queries from Berlin, which
    /// sits ~878 km from Paris, ~930 km from London, and an ocean from New
    /// York — so nearest-first is `[1, 0, 2]` under both kernels.
    struct Cities {
        values: [[f64; 2]; 3],
    }

    fn cities() -> Cities {
        Cities {
            values: [
                [51.5074f64.to_radians(), (-0.1278f64).to_radians()], // London
                [48.8566f64.to_radians(), 2.3522f64.to_radians()],    // Paris
                [40.7128f64.to_radians(), (-74.0060f64).to_radians()], // New York
            ],
        }
    }

    fn berlin() -> [f64; 2] {
        [52.5200f64.to_radians(), 13.4050f64.to_radians()]
    }

    impl GraphLike for Cities {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "f64"
        }
        fn value_shape(&self) -> Option<Shape> {
            Some(Shape::Vector(2))
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
            // SAFETY: reinterpret the contiguous `[f64; 2]` as its bytes.
            unsafe {
                core::slice::from_raw_parts(row.as_ptr() as *const u8, core::mem::size_of_val(row))
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

    #[test]
    fn haversine_scan_ranks_nearest_city() {
        let table = crate::GraphTable::build(&cities(), None).unwrap();
        let mut scanner = HaversineScanner::<f64>::new(ExecutionScope::Serial, Global).unwrap();

        let query = berlin();
        let queries =
            nk::Matrix::<f64>::try_from_scalars(&[query[0] as f32, query[1] as f32], &[1, 2])
                .unwrap();
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
        scanner
            .scan(
                queries.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[1, 0, 2]);
        // Meters, ascending, and the Atlantic dwarfs the Channel.
        let row = distances.row(0).unwrap();
        assert!(row[0] < row[1] && row[1] < row[2]);
        assert!(row[0] > 700_000.0 && row[1] < 1_100_000.0);
        assert!(row[2] > 5_000_000.0);
    }

    #[test]
    fn haversine_walk_agrees_with_scan() {
        let table = crate::GraphTable::build(&cities(), None).unwrap();
        let mut searcher =
            HaversineSearcher::<f64>::new(3, ExecutionScope::Serial, Global).unwrap();

        let query = berlin();
        let queries =
            nk::Matrix::<f64>::try_from_scalars(&[query[0] as f32, query[1] as f32], &[1, 2])
                .unwrap();
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
        searcher
            .search(
                queries.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[1, 0, 2]);
    }

    #[test]
    fn vincenty_agrees_on_ordering() {
        // The geodesic refines the sphere's distances without reordering three
        // cities this far apart — and its Berlin-Paris figure lands within a
        // percent of the true ~878 km.
        let table = crate::GraphTable::build(&cities(), None).unwrap();
        let mut scanner = VincentyScanner::<f64>::new(ExecutionScope::Serial, Global).unwrap();

        let query = berlin();
        let queries =
            nk::Matrix::<f64>::try_from_scalars(&[query[0] as f32, query[1] as f32], &[1, 2])
                .unwrap();
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
        scanner
            .scan(
                queries.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[1, 0, 2]);
        let paris = distances.row(0).unwrap()[0];
        assert!((paris - 878_000.0).abs() < 10_000.0, "got {paris}");
    }

    #[test]
    fn parallel_scan_matches_serial() {
        // Eight queries across four workers must land the exact rows the serial
        // path lands: each worker owns its query's strided tile column.
        let table = crate::GraphTable::build(&cities(), None).unwrap();
        let mut coordinates = Vec::new();
        for city in [
            (52.5200, 13.4050),
            (41.9028, 12.4964),
            (55.7558, 37.6173),
            (35.6762, 139.6503),
            (37.7749, -122.4194),
            (48.2082, 16.3738),
            (59.3293, 18.0686),
            (40.4168, -3.7038),
        ] {
            coordinates.push((city.0 as f64).to_radians() as f32);
            coordinates.push((city.1 as f64).to_radians() as f32);
        }
        let queries = nk::Matrix::<f64>::try_from_scalars(&coordinates, &[8, 2]).unwrap();

        let run = |scope: ExecutionScope| {
            let mut scanner = HaversineScanner::<f64>::new(scope, Global).unwrap();
            let (mut matches, mut distances, mut counts) = outputs(8, 3);
            scanner
                .scan(
                    queries.view(),
                    &table,
                    &mut matches,
                    &mut distances,
                    &mut counts,
                )
                .unwrap();
            (matches, distances, counts)
        };

        let (serial_matches, serial_distances, serial_counts) = run(ExecutionScope::Serial);
        let (parallel_matches, parallel_distances, parallel_counts) = run(ExecutionScope::CPUs(4));
        assert_eq!(serial_counts, parallel_counts);
        for query in 0..8 {
            assert_eq!(
                serial_matches.row(query).unwrap(),
                parallel_matches.row(query).unwrap(),
                "query {query} matches"
            );
            assert_eq!(
                serial_distances.row(query).unwrap(),
                parallel_distances.row(query).unwrap(),
                "query {query} distances"
            );
        }
    }

    #[test]
    fn dispatch_resolves_dtype() {
        let table = crate::GraphTable::build(&cities(), None).unwrap();
        let query = berlin();
        // SAFETY: reinterpret the `[f64; 2]` query as the raw bytes the erased
        // layer accepts.
        let bytes = unsafe {
            core::slice::from_raw_parts(query.as_ptr() as *const u8, core::mem::size_of_val(&query))
        };
        let (mut matches, mut distances, mut counts) = outputs(1, 2);

        let mut scanner =
            HaversineScannerAny::new(table.value_dtype(), ExecutionScope::Serial, Global).unwrap();
        assert_eq!(scanner.dtype(), "f64");
        scanner
            .scan(bytes, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[1, 0]);

        let mut searcher =
            VincentySearcherAny::new(table.value_dtype(), 3, ExecutionScope::Serial, Global)
                .unwrap();
        assert_eq!(searcher.dtype(), "f64");
        searcher
            .search(bytes, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[1, 0]);
    }

    #[test]
    fn scan_refuses_a_non_pair_column() {
        // A `[latitude, longitude]` family over anything but a two-wide column
        // is a type error, not a garbage distance — here a three-wide column.
        struct Triples;
        impl GraphLike for Triples {
            fn len(&self) -> usize {
                1
            }
            fn key_dtype(&self) -> &str {
                "u64"
            }
            fn value_dtype(&self) -> &str {
                "f64"
            }
            fn value_shape(&self) -> Option<Shape> {
                Some(Shape::Vector(3))
            }
            fn neighbors_degree(&self) -> Option<usize> {
                Some(0)
            }
            fn key(&self, _i: usize) -> &[u8] {
                &[0; 8]
            }
            fn value(&self, _i: usize) -> &[u8] {
                const ROW: [u8; 24] = [0; 24];
                &ROW
            }
            fn neighbors(&self, _i: usize) -> NeighborList<'_> {
                NeighborList::new(&[], OffsetDType::U32)
            }
        }

        let table = crate::GraphTable::build(&Triples, None).unwrap();
        let mut scanner = HaversineScanner::<f64>::new(ExecutionScope::Serial, Global).unwrap();
        let queries = nk::Matrix::<f64>::try_from_scalars(&[0.0f32, 0.0], &[1, 2]).unwrap();
        let (mut matches, mut distances, mut counts) = outputs(1, 1);
        assert!(matches!(
            scanner.scan(
                queries.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts
            ),
            Err(Error::WrongType)
        ));
    }
}

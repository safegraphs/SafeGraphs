//! Binary-vector metrics over NumKong: Hamming and Jaccard distance over packed
//! bits. A node's value is a `u1x8` bit string, eight lanes to the byte, and both
//! metrics are smaller-is-better distances, so this reads like the smaller-is-better
//! half of `dense` — an [`AngularScanner`]-shaped packed scan and a greedy walk —
//! only the kernel and the scalar differ.
//!
//! [`nk::Hammings`] and [`nk::Jaccards`], the packed scan kernels, ship for `u1x8`
//! alone; the single-pair kernels reach further — `u8` Hamming over whole-byte
//! elements, `u16`/`u32` weighted Jaccard over count vectors — and those dtypes
//! get pairwise-scanned arms through [`SetPairScanner`], scoring with the very
//! kernel the walk uses, so scan and walk agree on every dtype either way.
//!
//! The packing-aware `matrix_view` and `value_view` scan primitives are reused
//! from `dense`, and the shaped `batch_scan` and `batch_search` top-k from the
//! shared `batch` engine; only the metric bodies live here.
//!
//! [`AngularScanner`]: crate::search::dense::AngularScanner

use super::batch::{batch_scan, batch_search, build_workers, open_pool, Workers};
use super::{
    walk, ExecutionScope, Matrix, MetricLike, Ranking, ScannerLike, TraversalContext, WalkerLike,
};
use crate::search::dense::{value_view, view_from_bytes};
use crate::{cast, Error, GraphTable, GraphTableView, IndexLike, OffsetDType, ScoreLike};
use allocator_api2::alloc::{Allocator as AllocatorLike, Global};
use core::cmp::Ordering;
use forkunion as fu;
use numkong as nk;

// region: Hamming

/// The batch half of the Hamming metric: owns a thread pool, packs the query side
/// once per call, and streams the value side from the map through a popcount kernel.
pub struct HammingScanner<Value: nk::Hammings + nk::Hamming, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<Value, Allocator>,
    tile: Matrix<<Value as nk::Hamming>::Output, Allocator>,
    alloc: Allocator,
}

impl<Value: nk::Hammings + nk::Hamming, Allocator: AllocatorLike + Clone>
    HammingScanner<Value, Allocator>
{
    /// A scanner over the workers `scope` names, its packed query in `alloc`,
    /// uniform with [`DotScanner::new`].
    ///
    /// [`DotScanner::new`]: crate::search::dense::DotScanner::new
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(HammingScanner {
            pool: open_pool(scope)?,
            packed: nk::DotsPackedMatrix::empty_in(alloc.clone()),
            tile: Matrix::empty_in(alloc.clone()),
            alloc,
        })
    }
}

impl<Value, Allocator> HammingScanner<Value, Allocator>
where
    Allocator: AllocatorLike + Clone,
    Value: nk::Hamming + nk::Hammings + 'static,
    <Value as nk::Hamming>::Output: Copy + Default + PartialOrd + ScoreLike,
    HammingScanner<Value, Allocator>: ScannerLike<HammingMetric<Value>>,
{
    /// Batch brute-force top-k, nearest-first — Hamming is a smaller-is-better
    /// distance. `matches`/`distances` carry `k` and the query count.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let mut tile = std::mem::replace(&mut self.tile, Matrix::empty_in(self.alloc.clone()));
        let outcome = batch_scan::<HammingMetric<Value>, _>(
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

/// The traversal half of the Hamming metric: one query at a fixed node width,
/// scoring a staged node set one bit string at a time.
pub struct HammingWalker<Value: nk::Hamming, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, <Value as nk::Hamming>::Output, Allocator>,
}

impl<Value, Index, Allocator> HammingWalker<Value, Index, Allocator>
where
    Value: nk::Hamming,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Hamming>::Output: Copy + Default,
{
    /// A searcher walking with breadth `expansion_factor`, its scratch in
    /// `alloc`, uniform with [`DotWalker::new`].
    ///
    /// [`DotWalker::new`]: crate::search::dense::DotWalker::new
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(HammingWalker {
            context: TraversalContext::new_in(expansion_factor, alloc)?,
        })
    }

    pub fn context(
        &mut self,
    ) -> &mut TraversalContext<Index, <Value as nk::Hamming>::Output, Allocator> {
        &mut self.context
    }
}

/// Hamming distance — the count of differing bits — over a packed-bit scalar, in
/// `u32`. Smaller is nearer, so it ranks like a Euclidean distance.
pub struct HammingMetric<Value>(core::marker::PhantomData<Value>);

impl<Value: nk::Hamming + 'static> MetricLike for HammingMetric<Value>
where
    <Value as nk::Hamming>::Output: Copy + Default,
{
    type Query<'a> = &'a [Value];
    type Queries<'a> = nk::MatrixView<'a, Value>;
    type Score = <Value as nk::Hamming>::Output;
}

impl<Metric, Value, Allocator> ScannerLike<Metric> for HammingScanner<Value, Allocator>
where
    Allocator: AllocatorLike + Clone,
    Value: nk::Hammings + nk::Hamming + Clone + Send + Sync + 'static,
    Metric: MetricLike<Score = u32> + ?Sized,
    for<'a> Metric::Queries<'a>: Into<nk::MatrixView<'a, Value>>,
{
    fn scan(
        &mut self,
        queries: Metric::Queries<'_>,
        table: GraphTableView<'_>,
        scores: &mut Matrix<Metric::Score, impl AllocatorLike + Clone>,
    ) -> Result<(), Error> {
        // The batch is already a shaped `nodes-agnostic` matrix; its width must match
        // the table's value column, and its row count the tile's query count.
        let width = table.value_width().ok_or(Error::Inconsistent)?;
        let query_view: nk::MatrixView<Value> = queries.into();
        let values = value_view::<Value>(table, width)?;

        let (query_count, node_count) = (query_view.shape()[0], values.shape()[0]);
        if query_view.shape()[1] != width
            || query_count != scores.cols()
            || node_count != scores.rows()
        {
            return Err(Error::Inconsistent);
        }

        self.packed
            .try_pack_parallel_into(&query_view, &mut self.pool)
            .map_err(|_| Error::Inconsistent)?;

        // Node-major, contiguous rows, as every dense scan; only the kernel differs.
        let element = core::mem::size_of::<Metric::Score>() as isize;
        let mut tile = unsafe {
            nk::MatrixSpan::<Metric::Score>::from_raw_parts(
                scores.as_mut_ptr(),
                &[node_count, query_count],
                &[query_count as isize * element, element],
            )
        };
        nk::HammingsPackedParallelOps::try_hammings_packed_parallel_into(
            &values,
            &self.packed,
            &mut tile,
            &mut self.pool,
        )
        .map_err(|_| Error::Inconsistent)
    }
}

impl<Metric, Value, Index, Allocator> WalkerLike<Metric> for HammingWalker<Value, Index, Allocator>
where
    Value: nk::Hamming + 'static,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Hamming>::Output: Copy + Default + PartialOrd,
    Metric: MetricLike<Score = <Value as nk::Hamming>::Output> + ?Sized,
    for<'a> Metric::Query<'a>: Into<&'a [Value]>,
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        query: Metric::Query<'_>,
        table: GraphTableView<'_>,
        nodes: &[Index],
        out: &mut [Metric::Score],
    ) -> Result<(), Error> {
        if nodes.len() != out.len() {
            return Err(Error::Inconsistent);
        }
        let query: &[Value] = query.into();
        for (slot, &node) in out.iter_mut().zip(nodes) {
            let index: usize = node.try_into().map_err(|_| Error::TooLarge)?;
            let value: &[Value] = cast(table.value(index)).ok_or(Error::WrongType)?;
            *slot = <Value as nk::Hamming>::hamming(query, value).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, Metric::Score, Allocator> {
        &mut self.context
    }

    fn compare(a: Metric::Score, b: Metric::Score) -> Ordering {
        Ranking::Nearest.compare(a, b)
    }

    fn search(&mut self, query: Metric::Query<'_>, table: GraphTableView<'_>) -> Result<(), Error> {
        walk::<Metric, Self>(self, query, table)
    }
}

/// Batched Hamming walks, resolving the node width from the first table
/// searched and holding one [`HammingWalker`] per pool thread.
pub struct HammingSearcher<Value: nk::Hamming, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers: Workers<
        HammingWalker<Value, u32, Allocator>,
        HammingWalker<Value, u64, Allocator>,
        Allocator,
    >,
    alloc: Allocator,
}

impl<Value, Allocator> HammingSearcher<Value, Allocator>
where
    Value: nk::Hamming + 'static,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Hamming>::Output: Copy + Default + PartialOrd,
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
        Ok(HammingSearcher {
            pool: open_pool(scope)?,
            expansion_factor,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, fanned across the pool. Uniform with [`HammingScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <Value as nk::Hamming>::Output: ScoreLike,
        HammingWalker<Value, u32, Allocator>: WalkerLike<HammingMetric<Value>, Index = u32> + Send,
        HammingWalker<Value, u64, Allocator>: WalkerLike<HammingMetric<Value>, Index = u64> + Send,
    {
        let table = table.view();
        let threads = self.pool.threads_count();
        let expansion_factor = self.expansion_factor;
        match table.neighbors_dtype() {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(threads, alloc.clone(), || {
                        HammingWalker::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<HammingMetric<Value>, _>(
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
                        HammingWalker::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<HammingMetric<Value>, _>(
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

/// The Hamming twin of [`DotScannerAny`]: runtime dtype dispatch, `u1x8` the one arm
/// with a packed Hamming kernel. The typed scanner owns its tile, so unlike the dot
/// dispatcher this needs no scratch of its own.
///
/// [`DotScannerAny`]: crate::search::dense::DotScannerAny
// region: Pairwise set scanner

/// The seam for the set dtypes NumKong ships __no packed kernel__ for — `u8`
/// Hamming and the `u16`/`u32` weighted Jaccard: one distance per pair, the
/// very kernel the walk uses, so a pairwise scan and a walk agree exactly.
pub trait SetKernelLike<Value>: 'static {
    /// The kernel's native output: `u32` for Hamming counts, `f32` for Jaccard.
    type Output: Copy + Default + PartialOrd + ScoreLike + Send + Sync;

    /// The distance between two equal-length rows. `None` is a length
    /// mismatch, which the callers turn into [`Error::Inconsistent`].
    fn distance(query: &[Value], node: &[Value]) -> Option<Self::Output>;
}

/// Element-wise Hamming — the count of differing elements.
pub struct HammingKernel;

impl<Value: nk::Hamming + 'static> SetKernelLike<Value> for HammingKernel
where
    <Value as nk::Hamming>::Output: Copy + Default + PartialOrd + ScoreLike + Send + Sync,
{
    type Output = <Value as nk::Hamming>::Output;

    fn distance(query: &[Value], node: &[Value]) -> Option<Self::Output> {
        <Value as nk::Hamming>::hamming(query, node)
    }
}

/// Weighted Jaccard over count vectors — zero if and only if identical.
pub struct JaccardKernel;

impl<Value: nk::Jaccard + 'static> SetKernelLike<Value> for JaccardKernel
where
    <Value as nk::Jaccard>::Output: Copy + Default + PartialOrd + ScoreLike + Send + Sync,
{
    type Output = <Value as nk::Jaccard>::Output;

    fn distance(query: &[Value], node: &[Value]) -> Option<Self::Output> {
        <Value as nk::Jaccard>::jaccard(query, node)
    }
}

/// The batch half for the pairwise set dtypes: no packed form exists, so the
/// scan fans __nodes__ across the owned pool, each task exclusively borrowing
/// its node's contiguous tile row — the same safe shape as the mesh scan — and
/// scoring every query with the walk's own single-pair kernel.
pub struct SetPairScanner<Kernel: SetKernelLike<Value>, Value, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    tile: Matrix<<Kernel as SetKernelLike<Value>>::Output, Allocator>,
    alloc: Allocator,
    kernel: core::marker::PhantomData<fn() -> Kernel>,
}

impl<Kernel, Value, Allocator> SetPairScanner<Kernel, Value, Allocator>
where
    Kernel: SetKernelLike<Value>,
    Allocator: AllocatorLike + Clone,
{
    /// A scanner over the workers `scope` names, its tile in `alloc`, uniform
    /// with [`DotScanner::new`].
    ///
    /// [`DotScanner::new`]: crate::search::dense::DotScanner::new
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(SetPairScanner {
            pool: open_pool(scope)?,
            tile: Matrix::empty_in(alloc.clone()),
            alloc,
            kernel: core::marker::PhantomData,
        })
    }
}

/// Fill a node-major tile with pairwise set distances — the shared body of
/// both kernels' `ScannerLike` impls.
fn scan_pairwise_tile<Kernel, Value>(
    pool: &mut fu::ThreadPool,
    queries: nk::MatrixView<'_, Value>,
    table: GraphTableView<'_>,
    scores: &mut Matrix<Kernel::Output, impl AllocatorLike + Clone>,
) -> Result<(), Error>
where
    Kernel: SetKernelLike<Value>,
    Value: nk::StorageElement + Copy + Default + Send + Sync + 'static,
{
    use super::batch::ShareRead;
    use std::sync::OnceLock;

    let width = table.value_width().ok_or(Error::WrongType)?;
    let node_values: &[Value] = cast(table.value_buffer()).ok_or(Error::WrongType)?;
    let node_count = table.len();
    if node_values.len() != node_count * width {
        return Err(Error::Inconsistent);
    }
    let query_count = queries.shape().first().copied().unwrap_or(0);
    if queries.shape().get(1) != Some(&width)
        || query_count != scores.cols()
        || node_count != scores.rows()
    {
        return Err(Error::Inconsistent);
    }

    let mut rows: Vec<&mut [Kernel::Output]> = Vec::new();
    rows.try_reserve_exact(node_count)
        .map_err(|_| Error::Alloc)?;
    rows.extend(scores.as_mut_slice().chunks_mut(query_count.max(1)));
    let queries_shared = ShareRead(&queries);
    let failure = OnceLock::new();

    fu::ParallelSliceMut::new(&mut rows).for_each_dynamic(pool, |row, prong| {
        let queries = queries_shared.read();
        let node = &node_values[prong.task_index * width..prong.task_index * width + width];
        for (query_index, cell) in row.iter_mut().enumerate() {
            let Some(storage) = queries.as_packed_slice() else {
                let _ = failure.set(Error::Inconsistent);
                return;
            };
            let query = &storage[query_index * width..query_index * width + width];
            match Kernel::distance(query, node) {
                Some(distance) => *cell = distance,
                None => {
                    let _ = failure.set(Error::Inconsistent);
                    return;
                }
            }
        }
    });

    match failure.into_inner() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

impl<Metric, Value, Allocator> ScannerLike<Metric>
    for SetPairScanner<HammingKernel, Value, Allocator>
where
    Value: nk::Hamming + nk::StorageElement + Copy + Default + Send + Sync + 'static,
    <Value as nk::Hamming>::Output: Copy + Default + PartialOrd + ScoreLike + Send + Sync,
    Metric: MetricLike<Score = <Value as nk::Hamming>::Output> + ?Sized,
    for<'a> Metric::Queries<'a>: Into<nk::MatrixView<'a, Value>>,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    fn scan(
        &mut self,
        queries: Metric::Queries<'_>,
        table: GraphTableView<'_>,
        scores: &mut Matrix<Metric::Score, impl AllocatorLike + Clone>,
    ) -> Result<(), Error> {
        scan_pairwise_tile::<HammingKernel, Value>(&mut self.pool, queries.into(), table, scores)
    }
}

impl<Metric, Value, Allocator> ScannerLike<Metric>
    for SetPairScanner<JaccardKernel, Value, Allocator>
where
    Value: nk::Jaccard + nk::StorageElement + Copy + Default + Send + Sync + 'static,
    <Value as nk::Jaccard>::Output: Copy + Default + PartialOrd + ScoreLike + Send + Sync,
    Metric: MetricLike<Score = <Value as nk::Jaccard>::Output> + ?Sized,
    for<'a> Metric::Queries<'a>: Into<nk::MatrixView<'a, Value>>,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    fn scan(
        &mut self,
        queries: Metric::Queries<'_>,
        table: GraphTableView<'_>,
        scores: &mut Matrix<Metric::Score, impl AllocatorLike + Clone>,
    ) -> Result<(), Error> {
        scan_pairwise_tile::<JaccardKernel, Value>(&mut self.pool, queries.into(), table, scores)
    }
}

impl<Value, Allocator> SetPairScanner<HammingKernel, Value, Allocator>
where
    Value: nk::Hamming + nk::StorageElement + Copy + Default + Send + Sync + 'static,
    <Value as nk::Hamming>::Output: Copy + Default + PartialOrd + ScoreLike + Send + Sync,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    /// Batch brute-force top-k, nearest-first, uniform with
    /// [`HammingScanner::scan`].
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let mut tile = std::mem::replace(&mut self.tile, Matrix::empty_in(self.alloc.clone()));
        let outcome = batch_scan::<HammingMetric<Value>, _>(
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

impl<Value, Allocator> SetPairScanner<JaccardKernel, Value, Allocator>
where
    Value: nk::Jaccard + nk::StorageElement + Copy + Default + Send + Sync + 'static,
    <Value as nk::Jaccard>::Output: Copy + Default + PartialOrd + ScoreLike + Send + Sync,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    /// Batch brute-force top-k, nearest-first, uniform with
    /// [`JaccardScanner::scan`].
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let mut tile = std::mem::replace(&mut self.tile, Matrix::empty_in(self.alloc.clone()));
        let outcome = batch_scan::<JaccardMetric<Value>, _>(
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

/// Element-wise Hamming over whole-byte elements, the pairwise-scanned arm.
pub type HammingPairScanner<Value, Allocator = Global> =
    SetPairScanner<HammingKernel, Value, Allocator>;
/// Weighted Jaccard over count vectors, the pairwise-scanned arms.
pub type JaccardPairScanner<Value, Allocator = Global> =
    SetPairScanner<JaccardKernel, Value, Allocator>;

// endregion: Pairwise set scanner

/// A Hamming scanner chosen at runtime from a file's value dtype: `u1x8` through
/// the packed kernel, the wider set dtypes pairwise.
pub enum HammingScannerAny<Allocator: AllocatorLike = Global> {
    U1x8(HammingScanner<nk::u1x8, Allocator>),
    /// `u8` ships no packed kernel, so its arm scans pairwise — exact, just
    /// not through `DotsPackedMatrix`.
    U8(HammingPairScanner<u8, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send + Sync> HammingScannerAny<Allocator> {
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => HammingScannerAny::U1x8(HammingScanner::new(scope, alloc)?),
            "u8" => HammingScannerAny::U8(HammingPairScanner::new(scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            HammingScannerAny::U1x8(_) => "u1x8",
            HammingScannerAny::U8(_) => "u8",
        }
    }

    /// Batch brute-force top-k over a query batch given as raw bytes in the table's
    /// dtype, cast to the packed bit scalar and forwarded to the typed scanner.
    pub fn scan(
        &mut self,
        queries: &[u8],
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            HammingScannerAny::U1x8(scanner) => scanner.scan(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
            HammingScannerAny::U8(scanner) => scanner.scan(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// The Hamming twin of [`DotSearcherAny`]: runtime dispatch over a batch walk.
///
/// [`DotSearcherAny`]: crate::search::dense::DotSearcherAny
pub enum HammingSearcherAny<Allocator: AllocatorLike = Global> {
    U1x8(HammingSearcher<nk::u1x8, Allocator>),
    U8(HammingSearcher<u8, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send> HammingSearcherAny<Allocator> {
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => {
                HammingSearcherAny::U1x8(HammingSearcher::new(expansion_factor, scope, alloc)?)
            }
            "u8" => HammingSearcherAny::U8(HammingSearcher::new(expansion_factor, scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            HammingSearcherAny::U1x8(_) => "u1x8",
            HammingSearcherAny::U8(_) => "u8",
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
            HammingSearcherAny::U1x8(search) => search.search(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
            HammingSearcherAny::U8(search) => search.search(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

// endregion: Hamming

// region: Jaccard

/// The batch half of the Jaccard metric, the [`HammingScanner`] with the
/// intersection-over-union kernel and an `f32` result.
pub struct JaccardScanner<Value: nk::Jaccards + nk::Jaccard, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<Value, Allocator>,
    tile: Matrix<<Value as nk::Jaccard>::Output, Allocator>,
    alloc: Allocator,
}

impl<Value: nk::Jaccards + nk::Jaccard, Allocator: AllocatorLike + Clone>
    JaccardScanner<Value, Allocator>
{
    /// A scanner over the workers `scope` names, its packed query in `alloc`,
    /// uniform with [`DotScanner::new`].
    ///
    /// [`DotScanner::new`]: crate::search::dense::DotScanner::new
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(JaccardScanner {
            pool: open_pool(scope)?,
            packed: nk::DotsPackedMatrix::empty_in(alloc.clone()),
            tile: Matrix::empty_in(alloc.clone()),
            alloc,
        })
    }
}

impl<Value, Allocator> JaccardScanner<Value, Allocator>
where
    Allocator: AllocatorLike + Clone,
    Value: nk::Jaccard + nk::Jaccards + 'static,
    <Value as nk::Jaccard>::Output: Copy + Default + PartialOrd + ScoreLike,
    JaccardScanner<Value, Allocator>: ScannerLike<JaccardMetric<Value>>,
{
    /// Batch brute-force top-k, nearest-first — Jaccard is a smaller-is-better
    /// distance. `matches`/`distances` carry `k` and the query count.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let mut tile = std::mem::replace(&mut self.tile, Matrix::empty_in(self.alloc.clone()));
        let outcome = batch_scan::<JaccardMetric<Value>, _>(
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

/// The traversal half of the Jaccard metric: one query at a fixed node width,
/// scoring a staged node set one bit string at a time.
pub struct JaccardWalker<Value: nk::Jaccard, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, <Value as nk::Jaccard>::Output, Allocator>,
}

impl<Value, Index, Allocator> JaccardWalker<Value, Index, Allocator>
where
    Value: nk::Jaccard,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Jaccard>::Output: Copy + Default,
{
    /// A searcher walking with breadth `expansion_factor`, its scratch in
    /// `alloc`, uniform with [`DotWalker::new`].
    ///
    /// [`DotWalker::new`]: crate::search::dense::DotWalker::new
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(JaccardWalker {
            context: TraversalContext::new_in(expansion_factor, alloc)?,
        })
    }

    pub fn context(
        &mut self,
    ) -> &mut TraversalContext<Index, <Value as nk::Jaccard>::Output, Allocator> {
        &mut self.context
    }
}

/// Jaccard distance — one minus intersection over union — over a packed-bit scalar,
/// in `f32`. Smaller is nearer, ranking like Hamming.
pub struct JaccardMetric<Value>(core::marker::PhantomData<Value>);

impl<Value: nk::Jaccard + 'static> MetricLike for JaccardMetric<Value>
where
    <Value as nk::Jaccard>::Output: Copy + Default,
{
    type Query<'a> = &'a [Value];
    type Queries<'a> = nk::MatrixView<'a, Value>;
    type Score = <Value as nk::Jaccard>::Output;
}

impl<Metric, Value, Allocator> ScannerLike<Metric> for JaccardScanner<Value, Allocator>
where
    Allocator: AllocatorLike + Clone,
    Value: nk::Jaccards + nk::Jaccard + Clone + Send + Sync + 'static,
    <Value as nk::Jaccards>::JaccardResult: Send + Sync,
    Metric: MetricLike<Score = <Value as nk::Jaccards>::JaccardResult> + ?Sized,
    for<'a> Metric::Queries<'a>: Into<nk::MatrixView<'a, Value>>,
{
    fn scan(
        &mut self,
        queries: Metric::Queries<'_>,
        table: GraphTableView<'_>,
        scores: &mut Matrix<Metric::Score, impl AllocatorLike + Clone>,
    ) -> Result<(), Error> {
        // The batch is already a shaped `nodes-agnostic` matrix; its width must match
        // the table's value column, and its row count the tile's query count.
        let width = table.value_width().ok_or(Error::Inconsistent)?;
        let query_view: nk::MatrixView<Value> = queries.into();
        let values = value_view::<Value>(table, width)?;

        let (query_count, node_count) = (query_view.shape()[0], values.shape()[0]);
        if query_view.shape()[1] != width
            || query_count != scores.cols()
            || node_count != scores.rows()
        {
            return Err(Error::Inconsistent);
        }

        self.packed
            .try_pack_parallel_into(&query_view, &mut self.pool)
            .map_err(|_| Error::Inconsistent)?;

        let element = core::mem::size_of::<Metric::Score>() as isize;
        let mut tile = unsafe {
            nk::MatrixSpan::<Metric::Score>::from_raw_parts(
                scores.as_mut_ptr(),
                &[node_count, query_count],
                &[query_count as isize * element, element],
            )
        };
        nk::JaccardsPackedParallelOps::try_jaccards_packed_parallel_into(
            &values,
            &self.packed,
            &mut tile,
            &mut self.pool,
        )
        .map_err(|_| Error::Inconsistent)
    }
}

impl<Metric, Value, Index, Allocator> WalkerLike<Metric> for JaccardWalker<Value, Index, Allocator>
where
    Value: nk::Jaccard + 'static,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Jaccard>::Output: Copy + Default + PartialOrd,
    Metric: MetricLike<Score = <Value as nk::Jaccard>::Output> + ?Sized,
    for<'a> Metric::Query<'a>: Into<&'a [Value]>,
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        query: Metric::Query<'_>,
        table: GraphTableView<'_>,
        nodes: &[Index],
        out: &mut [Metric::Score],
    ) -> Result<(), Error> {
        if nodes.len() != out.len() {
            return Err(Error::Inconsistent);
        }
        let query: &[Value] = query.into();
        for (slot, &node) in out.iter_mut().zip(nodes) {
            let index: usize = node.try_into().map_err(|_| Error::TooLarge)?;
            let value: &[Value] = cast(table.value(index)).ok_or(Error::WrongType)?;
            *slot = <Value as nk::Jaccard>::jaccard(query, value).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, Metric::Score, Allocator> {
        &mut self.context
    }

    fn compare(a: Metric::Score, b: Metric::Score) -> Ordering {
        Ranking::Nearest.compare(a, b)
    }

    fn search(&mut self, query: Metric::Query<'_>, table: GraphTableView<'_>) -> Result<(), Error> {
        walk::<Metric, Self>(self, query, table)
    }
}

/// Batched Jaccard walks, resolving the node width from the first table
/// searched and holding one [`JaccardWalker`] per pool thread.
pub struct JaccardSearcher<Value: nk::Jaccard, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers: Workers<
        JaccardWalker<Value, u32, Allocator>,
        JaccardWalker<Value, u64, Allocator>,
        Allocator,
    >,
    alloc: Allocator,
}

impl<Value, Allocator> JaccardSearcher<Value, Allocator>
where
    Value: nk::Jaccard + 'static,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Jaccard>::Output: Copy + Default + PartialOrd,
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
        Ok(JaccardSearcher {
            pool: open_pool(scope)?,
            expansion_factor,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, fanned across the pool. Uniform with [`JaccardScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <Value as nk::Jaccard>::Output: ScoreLike,
        JaccardWalker<Value, u32, Allocator>: WalkerLike<JaccardMetric<Value>, Index = u32> + Send,
        JaccardWalker<Value, u64, Allocator>: WalkerLike<JaccardMetric<Value>, Index = u64> + Send,
    {
        let table = table.view();
        let threads = self.pool.threads_count();
        let expansion_factor = self.expansion_factor;
        match table.neighbors_dtype() {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(threads, alloc.clone(), || {
                        JaccardWalker::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<JaccardMetric<Value>, _>(
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
                        JaccardWalker::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<JaccardMetric<Value>, _>(
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

/// The Jaccard twin of [`HammingScannerAny`]: runtime dispatch, `u1x8` the one arm.
pub enum JaccardScannerAny<Allocator: AllocatorLike = Global> {
    U1x8(JaccardScanner<nk::u1x8, Allocator>),
    /// The weighted count-vector arms scan pairwise — no packed kernel ships.
    U16(JaccardPairScanner<u16, Allocator>),
    U32(JaccardPairScanner<u32, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send + Sync> JaccardScannerAny<Allocator> {
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => JaccardScannerAny::U1x8(JaccardScanner::new(scope, alloc)?),
            "u16" => JaccardScannerAny::U16(JaccardPairScanner::new(scope, alloc)?),
            "u32" => JaccardScannerAny::U32(JaccardPairScanner::new(scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            JaccardScannerAny::U1x8(_) => "u1x8",
            JaccardScannerAny::U16(_) => "u16",
            JaccardScannerAny::U32(_) => "u32",
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
            JaccardScannerAny::U1x8(scanner) => scanner.scan(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
            JaccardScannerAny::U16(scanner) => scanner.scan(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
            JaccardScannerAny::U32(scanner) => scanner.scan(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// The Jaccard twin of [`HammingSearcherAny`]: runtime dispatch over a batch walk.
pub enum JaccardSearcherAny<Allocator: AllocatorLike = Global> {
    U1x8(JaccardSearcher<nk::u1x8, Allocator>),
    U16(JaccardSearcher<u16, Allocator>),
    U32(JaccardSearcher<u32, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send> JaccardSearcherAny<Allocator> {
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => {
                JaccardSearcherAny::U1x8(JaccardSearcher::new(expansion_factor, scope, alloc)?)
            }
            "u16" => JaccardSearcherAny::U16(JaccardSearcher::new(expansion_factor, scope, alloc)?),
            "u32" => JaccardSearcherAny::U32(JaccardSearcher::new(expansion_factor, scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            JaccardSearcherAny::U1x8(_) => "u1x8",
            JaccardSearcherAny::U16(_) => "u16",
            JaccardSearcherAny::U32(_) => "u32",
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
            JaccardSearcherAny::U1x8(search) => search.search(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
            JaccardSearcherAny::U16(search) => search.search(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
            JaccardSearcherAny::U32(search) => search.search(
                view_from_bytes(queries, table.view())?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

// endregion: Jaccard

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::testing::assert_scores_close;
    use crate::{GraphLike, NeighborList, OffsetDType, Shape};

    /// Three nested 8-bit nodes `11111111 ⊃ 11110000 ⊃ 10000000`, packed as `u1x8`,
    /// fully connected at degree two. Against the query `11111110`, node 0 is nearest
    /// and node 2 farthest under both bit distances, so the order is `[0, 1, 2]`.
    struct BinaryRing {
        values: nk::Matrix<nk::u1x8>,
        degree: usize,
    }

    const BINARY_ROWS: [f32; 24] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // node 0
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, // node 1
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // node 2
    ];
    const BINARY_QUERY: [f32; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];

    fn binary_ring(degree: usize) -> BinaryRing {
        BinaryRing {
            values: nk::Matrix::try_from_scalars(&BINARY_ROWS, &[3, 8]).unwrap(),
            degree,
        }
    }

    impl GraphLike for BinaryRing {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "u1x8"
        }
        fn value_shape(&self) -> Option<Shape> {
            Some(Shape::Vector(self.values.shape()[1]))
        }
        fn neighbors_degree(&self) -> Option<usize> {
            Some(self.degree)
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
            let storage = self.values.as_slice();
            let per_row = storage.len() / self.values.shape()[0];
            let row = &storage[i * per_row..(i + 1) * per_row];
            // SAFETY: reinterpret the contiguous packed row as its bytes.
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
            let bytes = if self.degree == 0 {
                &ADJACENCY[i][..0]
            } else {
                &ADJACENCY[i][..]
            };
            NeighborList::new(bytes, OffsetDType::U32)
        }
    }

    /// The binary query as its one-row `u1x8` batch: `11111110`, seven bits set.
    fn binary_query() -> nk::Matrix<nk::u1x8> {
        nk::Matrix::<nk::u1x8>::try_from_scalars(&BINARY_QUERY, &[1, 8]).unwrap()
    }

    #[test]
    fn hamming_scan_and_search_agree() {
        // Popcount of XOR against `11111110`: node 0 differs in 1 bit, node 1 in 3,
        // node 2 in 6 — so the batch ranks `[0,1,2]`, distances `[1,3,6]`, and the
        // scan and the walk agree exactly. `k` and the query count read off `matches`.
        let table = GraphTable::build(&binary_ring(2), None).unwrap();
        let query = binary_query();

        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut scanner = HammingScanner::<nk::u1x8>::new(ExecutionScope::Serial, Global).unwrap();
        scanner
            .scan(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[0, 1, 2]);
        assert_scores_close(distances.row(0).unwrap(), &[1.0, 3.0, 6.0]);

        let mut searcher =
            HammingSearcher::<nk::u1x8>::new(3, ExecutionScope::Serial, Global).unwrap();
        let mut walked = Matrix::<u64>::new(1, 3).unwrap();
        let mut walked_scores = Matrix::<f64>::new(1, 3).unwrap();
        searcher
            .search(
                query.view(),
                &table,
                &mut walked,
                &mut walked_scores,
                &mut counts,
            )
            .unwrap();
        assert_eq!(walked.row(0).unwrap(), &[0, 1, 2]);
        assert_scores_close(walked_scores.row(0).unwrap(), &[1.0, 3.0, 6.0]);
    }

    #[test]
    fn jaccard_scan_and_search_agree() {
        // One minus intersection over union: node 0 is `1 - 7/8`, the others farther,
        // so the ranking is `[0,1,2]`, and scan and walk agree.
        let table = GraphTable::build(&binary_ring(2), None).unwrap();
        let query = binary_query();

        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut scanner = JaccardScanner::<nk::u1x8>::new(ExecutionScope::Serial, Global).unwrap();
        scanner
            .scan(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1, 2]);
        assert_scores_close(&distances.row(0).unwrap()[..1], &[0.125]);

        let mut searcher =
            JaccardSearcher::<nk::u1x8>::new(3, ExecutionScope::Serial, Global).unwrap();
        let mut walked = Matrix::<u64>::new(1, 3).unwrap();
        let mut walked_scores = Matrix::<f64>::new(1, 3).unwrap();
        searcher
            .search(
                query.view(),
                &table,
                &mut walked,
                &mut walked_scores,
                &mut counts,
            )
            .unwrap();
        assert_eq!(walked.row(0).unwrap(), &[0, 1, 2]);
        assert_scores_close(&walked_scores.row(0).unwrap()[..1], &[0.125]);
    }

    /// Three `u8` count vectors on the triangle: node 1 equals the query, node
    /// 0 differs in one element, node 2 in all four — covering the pairwise
    /// `u8` Hamming arm and the `u32` weighted Jaccard arm, whose scans reuse
    /// the walk's own single-pair kernels.
    struct Counts;

    const COUNT_ROWS: [[u8; 4]; 3] = [[3, 1, 4, 1], [3, 1, 4, 2], [9, 9, 9, 9]];
    const COUNT_QUERY: [u8; 4] = [3, 1, 4, 2];

    impl GraphLike for Counts {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "u8"
        }
        fn value_shape(&self) -> Option<crate::Shape> {
            Some(crate::Shape::Vector(4))
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
            &COUNT_ROWS[i]
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
    fn pairwise_arms_scan_and_walk_agree() {
        let table = GraphTable::build(&Counts, None).unwrap();
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        // The erased path exercises the raw-bytes entry the callers use.
        let mut scanner =
            HammingScannerAny::new(table.value_dtype(), ExecutionScope::Serial, Global).unwrap();
        assert_eq!(scanner.dtype(), "u8");
        scanner
            .scan(
                &COUNT_QUERY,
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[1, 0, 2]);
        assert_scores_close(distances.row(0).unwrap(), &[0.0, 1.0, 4.0]);

        let mut searcher =
            HammingSearcherAny::new(table.value_dtype(), 3, ExecutionScope::Serial, Global)
                .unwrap();
        searcher
            .search(
                &COUNT_QUERY,
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[1, 0, 2]);
        assert_scores_close(distances.row(0).unwrap(), &[0.0, 1.0, 4.0]);
    }

    /// The same triangle as `u32` counts, for the weighted Jaccard arms.
    struct WideCounts;

    const WIDE_ROWS: [[u32; 4]; 3] = [[3, 1, 4, 1], [3, 1, 4, 2], [9, 9, 9, 9]];
    const WIDE_QUERY: [u32; 4] = [3, 1, 4, 2];

    impl GraphLike for WideCounts {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "u32"
        }
        fn value_shape(&self) -> Option<crate::Shape> {
            Some(crate::Shape::Vector(4))
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
            let row = &WIDE_ROWS[i];
            // SAFETY: reinterpret the contiguous `[u32; 4]` as its bytes.
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
    fn weighted_jaccard_u32_scan_and_walk_agree() {
        // Weighted Jaccard against `[3,1,4,2]`: node 1 is identical — distance
        // zero — node 0 near, node 2 far. Scan and walk agree exactly.
        let table = GraphTable::build(&WideCounts, None).unwrap();
        let query_bytes = {
            // SAFETY: reinterpret the contiguous `[u32; 4]` query as its bytes.
            unsafe {
                core::slice::from_raw_parts(
                    WIDE_QUERY.as_ptr() as *const u8,
                    core::mem::size_of_val(&WIDE_QUERY),
                )
            }
        };
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut scanner =
            JaccardScannerAny::new(table.value_dtype(), ExecutionScope::Serial, Global).unwrap();
        assert_eq!(scanner.dtype(), "u32");
        scanner
            .scan(
                query_bytes,
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[1, 0, 2]);
        let scanned = distances.row(0).unwrap().to_vec();
        assert!(
            scanned[0].abs() < 1e-6,
            "identical vectors diverge: {scanned:?}"
        );

        let mut searcher =
            JaccardSearcherAny::new(table.value_dtype(), 3, ExecutionScope::Serial, Global)
                .unwrap();
        searcher
            .search(
                query_bytes,
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[1, 0, 2]);
        assert_scores_close(distances.row(0).unwrap(), &scanned[..]);
    }

    #[test]
    fn dispatch_scans_and_walks_bits() {
        // The erased runtime path: open a bit table, dispatch on its `u1x8` header,
        // and rank nearest-first through both a batch scan and a batch walk.
        let table = GraphTable::build(&binary_ring(2), None).unwrap();
        let query: [u8; 1] = [0b0111_1111];

        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut scanner =
            HammingScannerAny::new(table.value_dtype(), ExecutionScope::Serial, Global).unwrap();
        assert_eq!(scanner.dtype(), "u1x8");
        scanner
            .scan(&query, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1, 2]);

        let mut searcher =
            JaccardSearcherAny::new(table.value_dtype(), 3, ExecutionScope::Serial, Global)
                .unwrap();
        assert_eq!(searcher.dtype(), "u1x8");
        searcher
            .search(&query, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1, 2]);
    }
}

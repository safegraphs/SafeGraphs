//! Dense-vector metrics over NumKong. No StringZilla type appears here.
//!
//! Each metric is a generic descriptor — [`DotMetric<Value>`] over any scalar `T` —
//! that names a scalar and its score. One [`DotScanner`] and [`DotSearcher`] body,
//! generic over that scalar, serves every dtype; the searcher is also generic over
//! the table's neighbor width, so a `u32` graph traverses in `u32`. [`DotSearch`]
//! resolves that width from a table for a caller who has not named it.
//!
//! Choosing the scalar at runtime is a dense concern, so it lives here too:
//! [`DotScannerAny`] and [`DotSearcherAny`] route a header dtype string to the
//! matching functor. The cross-metric router in `dispatch` sits above these and
//! never learns a dtype.
//!
//! Scores land in the scalar's native accumulator, inferred from NumKong rather
//! than chosen here: `i32` for `i8`, `f64` for `f32`, `f32` for `bf16`. The value
//! column streams zero-copy from the memory map and is never packed or widened;
//! only the small query side is packed.

use super::batch::{batch_scan, batch_search, build_workers, open_pool, Workers};
use super::{
    walk, ExecutionScope, Matrix, MetricLike, QueryBatchLike, Ranking, ScannerLike, SearcherLike,
    TraversalContext,
};
use crate::{cast, Error, GraphLike, IndexLike, OffsetDType, Table, Widen};
use allocator_api2::alloc::{Allocator as AllocatorLike, Global};
use core::cmp::Ordering;
use forkunion as fu;
use numkong as nk;

/// A zero-copy `rows × width` view over caller-owned scalars, where `width` is a
/// count of logical elements. For a sub-byte scalar like `i4x2` or `u1x8` that is
/// wider than one storage value, so `rows` holds `width / dimensions_per_value`
/// storage elements per row and the row stride packs them tight — the same shape
/// arithmetic NumKong's own tensors use. `pub(crate)` so the sibling binary-metric
/// module builds its packed views the same way.
pub(crate) fn matrix_view<Value>(
    rows: &[Value],
    width: usize,
) -> Result<nk::MatrixView<'_, Value>, Error>
where
    Value: nk::StorageElement,
{
    let dimensions_per_value = Value::dimensions_per_value();
    if width == 0 || !width.is_multiple_of(dimensions_per_value) {
        return Err(Error::Inconsistent);
    }
    let storage_per_row = width / dimensions_per_value;
    if storage_per_row == 0 || rows.is_empty() || !rows.len().is_multiple_of(storage_per_row) {
        return Err(Error::Inconsistent);
    }
    let count = rows.len() / storage_per_row;
    let size = core::mem::size_of::<Value>() as isize;
    // SAFETY: `rows` is `count * storage_per_row` contiguous `T`. The logical shape
    // `[count, width]` with byte strides describes exactly that extent — the
    // innermost stride is one storage element, the row stride its packed width —
    // and the view borrows `rows` for `'_`.
    Ok(unsafe {
        nk::MatrixView::<Value>::from_raw_parts(
            rows.as_ptr(),
            &[count, width],
            &[storage_per_row as isize * size, size],
        )
    })
}

/// The table's value column as a zero-copy `nodes × width` matrix, `width` in
/// logical elements. The byte buffer reinterprets as packed storage scalars, which
/// [`matrix_view`] then unpacks by shape.
pub(crate) fn value_view<Value>(
    table: &Table<impl AllocatorLike>,
    width: usize,
) -> Result<nk::MatrixView<'_, Value>, Error>
where
    Value: nk::StorageElement,
{
    let scalars: &[Value] = cast(table.value_buffer()).ok_or(Error::WrongType)?;
    matrix_view(scalars, width)
}

/// The batch half of every dense metric: owns a thread pool, packs the query
/// side once per call, and streams the value side straight from the memory map.
pub struct DotScanner<Value: nk::Dots + nk::Dot, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<Value, Allocator>,
    tile: Matrix<<Value as nk::Dot>::Output>,
}

impl<Value: nk::Dots + nk::Dot, Allocator: AllocatorLike> DotScanner<Value, Allocator> {
    /// A scanner over the workers `scope` names, the caller included, on a pool
    /// it builds and owns, its packed query in `alloc` — NumKong bridges every
    /// `allocator_api2` allocator into its own trait, so no adapter appears
    /// here. NumKong is CPU-only, so a GPU scope is refused.
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        // NumKong's own workers claim AMX tile state themselves, but the calling
        // thread also participates in a dispatch, so it must claim it too — the
        // packed integer kernels trap without it.
        nk::capabilities::configure_thread();
        Ok(DotScanner {
            pool: open_pool(scope)?,
            packed: nk::DotsPackedMatrix::empty_in(alloc),
            tile: Matrix::empty(),
        })
    }
}

/// A dense query batch — a `nodes × width` [`nk::MatrixView`] — as a [`QueryBatchLike`]:
/// its row count is the query count, and each row is one query slice. This is what
/// lets one generic [`batch_search`] walk a dense batch or a sequence tape alike.
impl<Metric, Value> QueryBatchLike<Metric> for nk::MatrixView<'_, Value>
where
    Value: nk::StorageElement + Clone,
    Metric: MetricLike + ?Sized,
    for<'a> Metric::Query<'a>: From<&'a [Value]>,
{
    fn count(&self) -> usize {
        self.shape().first().copied().unwrap_or(0)
    }

    fn query(&self, index: usize) -> Option<Metric::Query<'_>> {
        let storage_per_row = self.shape().get(1)? / Value::dimensions_per_value();
        let storage = self.as_packed_slice()?;
        let start = index.checked_mul(storage_per_row)?;
        let row = storage.get(start..start.checked_add(storage_per_row)?)?;
        Some(row.into())
    }
}

impl<Value, Allocator> DotScanner<Value, Allocator>
where
    Allocator: AllocatorLike,
    Value: nk::Dot + nk::Dots + 'static,
    <Value as nk::Dot>::Output: Copy + Default + PartialOrd + Widen,
    DotScanner<Value, Allocator>: ScannerLike<DotMetric<Value>>,
{
    /// Batch brute-force top-k. `matches`/`distances` are `queries × k`; their shape
    /// carries both `k` and the query count. Larger dot ranks first.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let mut tile = std::mem::take(&mut self.tile);
        let outcome = batch_scan::<DotMetric<Value>, _>(
            self,
            queries,
            table,
            Ranking::Farthest,
            &mut tile,
            matches,
            distances,
            counts,
        );
        self.tile = tile;
        outcome
    }
}

/// The traversal half of every dense metric: owns the reusable scratch and scores
/// a node set one vector at a time, which is what a handful of neighbors wants.
/// The index type `I` matches the table's neighbor width, so the frontier stays
/// compact; `DotSearch` resolves it, but a caller who knows their table can name
/// it directly, `I` defaulting to `u32`.
pub struct DotSearcher<Value: nk::Dot, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, <Value as nk::Dot>::Output, Allocator>,
}

impl<Value, Index, Allocator> DotSearcher<Value, Index, Allocator>
where
    Value: nk::Dot,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Dot>::Output: Copy + Default,
{
    /// A searcher walking with breadth `expansion_factor`, its scratch in
    /// `alloc`. It carries no table — the scratch grows to fit each searched
    /// graph at walk time — and no scope: a leaf never executes a batch, its
    /// per-expansion scoring runs inline on whichever thread walks it.
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(DotSearcher {
            context: TraversalContext::new_in(expansion_factor, alloc)?,
        })
    }

    /// The traversal scratch, for draining results after a walk. Inherent so a
    /// caller need not name the metric to reach it, unlike the trait method it
    /// shadows.
    pub fn context(
        &mut self,
    ) -> &mut TraversalContext<Index, <Value as nk::Dot>::Output, Allocator> {
        &mut self.context
    }
}

/// Inner-product similarity over any NumKong scalar, accumulated in that scalar's
/// native dot output — `i32` for `i8`, `u32` for `u8`, `f64` for `f32`, `f32` for
/// every half and minifloat. One generic marker replaces a struct per dtype: the
/// scalar is the only thing that ever varied, so `DotMetric<i8>` and
/// `DotMetric<nk::e4m3>` are the whole vocabulary, and a new dtype adds no code.
pub struct DotMetric<Value>(core::marker::PhantomData<Value>);

impl<Value: nk::Dot + 'static> MetricLike for DotMetric<Value>
where
    <Value as nk::Dot>::Output: Copy + Default,
{
    type Query<'a> = &'a [Value];
    type Queries<'a> = nk::MatrixView<'a, Value>;
    type Score = <Value as nk::Dot>::Output;
}

impl<Metric, Value, Allocator> ScannerLike<Metric> for DotScanner<Value, Allocator>
where
    Allocator: AllocatorLike,
    Value: nk::Dots
        + nk::Dot<Output = <Value as nk::Dots>::Accumulator>
        + Clone
        + Send
        + Sync
        + 'static,
    <Value as nk::Dots>::Accumulator: Send + Sync,
    Metric: MetricLike<Score = <Value as nk::Dots>::Accumulator> + ?Sized,
    for<'a> Metric::Queries<'a>: Into<nk::MatrixView<'a, Value>>,
{
    fn scan(
        &mut self,
        queries: Metric::Queries<'_>,
        table: &Table<impl AllocatorLike>,
        scores: &mut Matrix<Metric::Score>,
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

        // NumKong computes `values × queriesᵀ`, which is exactly the tile's own
        // `nodes × queries` shape, so the kernel writes straight into the
        // caller's buffer. Its rows must be contiguous — a strided view is
        // rejected outright — which is why the tile is node-major.
        let element = core::mem::size_of::<Metric::Score>() as isize;
        let mut tile = unsafe {
            nk::MatrixSpan::<Metric::Score>::from_raw_parts(
                scores.as_mut_ptr(),
                &[node_count, query_count],
                &[query_count as isize * element, element],
            )
        };
        nk::DotsPackedParallelOps::try_dots_packed_parallel_into(
            &values,
            &self.packed,
            &mut tile,
            &mut self.pool,
        )
        .map_err(|_| Error::Inconsistent)
    }
}

impl<Metric, Value, Index, Allocator> SearcherLike<Metric> for DotSearcher<Value, Index, Allocator>
where
    Value: nk::Dot + 'static,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Dot>::Output: Copy + Default + PartialOrd,
    Metric: MetricLike<Score = <Value as nk::Dot>::Output> + ?Sized,
    for<'a> Metric::Query<'a>: Into<&'a [Value]>,
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        query: Metric::Query<'_>,
        table: &Table<impl AllocatorLike>,
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
            *slot = <Value as nk::Dot>::dot(query, value).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, Metric::Score, Allocator> {
        &mut self.context
    }

    fn compare(a: Metric::Score, b: Metric::Score) -> Ordering {
        Ranking::Farthest.compare(a, b)
    }

    fn search(
        &mut self,
        query: Metric::Query<'_>,
        table: &Table<impl AllocatorLike>,
    ) -> Result<(), Error> {
        walk::<Metric, Self>(self, query, table)
    }
}

/// A dot searcher whose neighbor width is settled by the tables it searches, not
/// named by the caller. Construction is configuration alone — breadth, scope —
/// and builds the pool; the first searched table picks the `u32` or `u64` worker
/// lane, one traversal context per pool thread, and a later table of the other
/// width rebuilds it. For a dense metric a leaf is nothing but its scratch, so the
/// lane is exactly a pool of contexts. A dispatcher over value dtypes builds one of
/// these per dtype and never branches on width itself; the query fan-out lives in
/// [`batch_search`].
pub struct DotSearch<Value: nk::Dot, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers: Workers<DotSearcher<Value, u32, Allocator>, DotSearcher<Value, u64, Allocator>>,
    alloc: Allocator,
}

impl<Value, Allocator> DotSearch<Value, Allocator>
where
    Value: nk::Dot + 'static,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Dot>::Output: Copy + Default + PartialOrd,
{
    /// A searcher walking with breadth `expansion_factor` across the workers
    /// `scope` names, over a pool it builds and owns, its scratch in `alloc`.
    /// No table is seen here: the neighbor-width worker lane settles on the
    /// first searched table.
    pub fn new(
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(DotSearch {
            pool: open_pool(scope)?,
            expansion_factor,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, fanned across the pool. Uniform with [`DotScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <Value as nk::Dot>::Output: Widen,
        DotSearcher<Value, u32, Allocator>: SearcherLike<DotMetric<Value>, Index = u32> + Send,
        DotSearcher<Value, u64, Allocator>: SearcherLike<DotMetric<Value>, Index = u64> + Send,
    {
        let threads = self.pool.threads_count();
        let expansion_factor = self.expansion_factor;
        match table.neighbors_column().dtype {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(threads, || {
                        DotSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<DotMetric<Value>, _>(
                    &mut self.pool,
                    workers,
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
                    self.workers = Workers::U64(build_workers(threads, || {
                        DotSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<DotMetric<Value>, _>(
                    &mut self.pool,
                    workers,
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

/// The batch half of every Euclidean metric. Same shape as [`DotScanner`] — a
/// pool and a reused packed query — because `Euclideans` packs through the very
/// same [`nk::DotsPackedMatrix`]; only the kernel it calls differs.
pub struct EuclideanScanner<Value: nk::Dots + nk::Euclidean, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<Value, Allocator>,
    tile: Matrix<<Value as nk::Euclidean>::EuclideanOutput>,
}

impl<Value: nk::Dots + nk::Euclidean, Allocator: AllocatorLike> EuclideanScanner<Value, Allocator> {
    /// A scanner over the workers `scope` names, its packed query in `alloc`,
    /// uniform with [`DotScanner::new`].
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(EuclideanScanner {
            pool: open_pool(scope)?,
            packed: nk::DotsPackedMatrix::empty_in(alloc),
            tile: Matrix::empty(),
        })
    }
}

impl<Value, Allocator> EuclideanScanner<Value, Allocator>
where
    Allocator: AllocatorLike,
    Value: nk::Euclidean + nk::Euclideans + 'static,
    <Value as nk::Euclidean>::EuclideanOutput: Copy + Default + PartialOrd + Widen,
    EuclideanScanner<Value, Allocator>: ScannerLike<EuclideanMetric<Value>>,
{
    /// Batch brute-force top-k, nearest-first — L2 is smaller-is-better.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let mut tile = std::mem::take(&mut self.tile);
        let outcome = batch_scan::<EuclideanMetric<Value>, _>(
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

/// The traversal half of every Euclidean metric.
pub struct EuclideanSearcher<Value: nk::Euclidean, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, <Value as nk::Euclidean>::EuclideanOutput, Allocator>,
}

impl<Value, Index, Allocator> EuclideanSearcher<Value, Index, Allocator>
where
    Value: nk::Euclidean,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Euclidean>::EuclideanOutput: Copy + Default,
{
    /// A searcher walking with breadth `expansion_factor`, its scratch in
    /// `alloc`, uniform with [`DotSearcher::new`].
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(EuclideanSearcher {
            context: TraversalContext::new_in(expansion_factor, alloc)?,
        })
    }

    pub fn context(
        &mut self,
    ) -> &mut TraversalContext<Index, <Value as nk::Euclidean>::EuclideanOutput, Allocator> {
        &mut self.context
    }
}

/// L2 distance over any NumKong scalar, in its native Euclidean output. Smaller is
/// nearer, so it ranks the reverse of a dot — the same contract, only the
/// comparator flips. Like [`DotMetric`], one generic marker spans every dtype.
pub struct EuclideanMetric<Value>(core::marker::PhantomData<Value>);

impl<Value: nk::Euclidean + 'static> MetricLike for EuclideanMetric<Value>
where
    <Value as nk::Euclidean>::EuclideanOutput: Copy + Default,
{
    type Query<'a> = &'a [Value];
    type Queries<'a> = nk::MatrixView<'a, Value>;
    type Score = <Value as nk::Euclidean>::EuclideanOutput;
}

impl<Metric, Value, Allocator> ScannerLike<Metric> for EuclideanScanner<Value, Allocator>
where
    Allocator: AllocatorLike,
    Value: nk::Euclideans + nk::Euclidean + nk::Dots + Clone + Send + Sync + 'static,
    <Value as nk::Euclideans>::SpatialResult: Send + Sync,
    Metric: MetricLike<Score = <Value as nk::Euclideans>::SpatialResult> + ?Sized,
    for<'a> Metric::Queries<'a>: Into<nk::MatrixView<'a, Value>>,
{
    fn scan(
        &mut self,
        queries: Metric::Queries<'_>,
        table: &Table<impl AllocatorLike>,
        scores: &mut Matrix<Metric::Score>,
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

        // Node-major, contiguous rows — exactly as the dot scan, since the output
        // shape is the same `nodes × queries`; only the kernel differs.
        let element = core::mem::size_of::<Metric::Score>() as isize;
        let mut tile = unsafe {
            nk::MatrixSpan::<Metric::Score>::from_raw_parts(
                scores.as_mut_ptr(),
                &[node_count, query_count],
                &[query_count as isize * element, element],
            )
        };
        nk::EuclideansPackedParallelOps::try_euclideans_packed_parallel_into(
            &values,
            &self.packed,
            &mut tile,
            &mut self.pool,
        )
        .map_err(|_| Error::Inconsistent)
    }
}

impl<Metric, Value, Index, Allocator> SearcherLike<Metric>
    for EuclideanSearcher<Value, Index, Allocator>
where
    Value: nk::Euclidean + 'static,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Euclidean>::EuclideanOutput: Copy + Default + PartialOrd,
    Metric: MetricLike<Score = <Value as nk::Euclidean>::EuclideanOutput> + ?Sized,
    for<'a> Metric::Query<'a>: Into<&'a [Value]>,
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        query: Metric::Query<'_>,
        table: &Table<impl AllocatorLike>,
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
            // The true distance, not the squared one, so its type and value match
            // the batch `Euclideans` kernel the scan uses — scan and walk agree.
            *slot = <Value as nk::Euclidean>::euclidean(query, value).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, Metric::Score, Allocator> {
        &mut self.context
    }

    fn compare(a: Metric::Score, b: Metric::Score) -> Ordering {
        Ranking::Nearest.compare(a, b)
    }

    fn search(
        &mut self,
        query: Metric::Query<'_>,
        table: &Table<impl AllocatorLike>,
    ) -> Result<(), Error> {
        walk::<Metric, Self>(self, query, table)
    }
}

/// A Euclidean searcher whose neighbor width is settled by the tables it searches,
/// the distance twin of [`DotSearch`]: a pool plus one context per thread.
pub struct EuclideanSearch<Value: nk::Euclidean, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers:
        Workers<EuclideanSearcher<Value, u32, Allocator>, EuclideanSearcher<Value, u64, Allocator>>,
    alloc: Allocator,
}

impl<Value, Allocator> EuclideanSearch<Value, Allocator>
where
    Value: nk::Euclidean + 'static,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Euclidean>::EuclideanOutput: Copy + Default + PartialOrd,
{
    /// A searcher walking with breadth `expansion_factor` across the workers
    /// `scope` names, its scratch in `alloc`, uniform with [`DotSearch::new`].
    pub fn new(
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(EuclideanSearch {
            pool: open_pool(scope)?,
            expansion_factor,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, fanned across the pool. Uniform with
    /// [`EuclideanScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <Value as nk::Euclidean>::EuclideanOutput: Widen,
        EuclideanSearcher<Value, u32, Allocator>:
            SearcherLike<EuclideanMetric<Value>, Index = u32> + Send,
        EuclideanSearcher<Value, u64, Allocator>:
            SearcherLike<EuclideanMetric<Value>, Index = u64> + Send,
    {
        let threads = self.pool.threads_count();
        let expansion_factor = self.expansion_factor;
        match table.neighbors_column().dtype {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(threads, || {
                        EuclideanSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<EuclideanMetric<Value>, _>(
                    &mut self.pool,
                    workers,
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
                    self.workers = Workers::U64(build_workers(threads, || {
                        EuclideanSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<EuclideanMetric<Value>, _>(
                    &mut self.pool,
                    workers,
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

/// The batch half of every angular metric. Structurally identical to
/// [`EuclideanScanner`] — both pack through [`nk::DotsPackedMatrix`] and are a
/// smaller-is-better distance — differing only in the kernel it calls.
pub struct AngularScanner<Value: nk::Dots + nk::Angular, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<Value, Allocator>,
    tile: Matrix<<Value as nk::Angular>::Output>,
}

impl<Value: nk::Dots + nk::Angular, Allocator: AllocatorLike> AngularScanner<Value, Allocator> {
    /// A scanner over the workers `scope` names, its packed query in `alloc`,
    /// uniform with [`DotScanner::new`].
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(AngularScanner {
            pool: open_pool(scope)?,
            packed: nk::DotsPackedMatrix::empty_in(alloc),
            tile: Matrix::empty(),
        })
    }
}

impl<Value, Allocator> AngularScanner<Value, Allocator>
where
    Allocator: AllocatorLike,
    Value: nk::Angular + nk::Angulars + 'static,
    <Value as nk::Angular>::Output: Copy + Default + PartialOrd + Widen,
    AngularScanner<Value, Allocator>: ScannerLike<AngularMetric<Value>>,
{
    /// Batch brute-force top-k, nearest-first — cosine distance is smaller-is-better.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let mut tile = std::mem::take(&mut self.tile);
        let outcome = batch_scan::<AngularMetric<Value>, _>(
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

/// The traversal half of every angular metric.
pub struct AngularSearcher<Value: nk::Angular, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, <Value as nk::Angular>::Output, Allocator>,
}

impl<Value, Index, Allocator> AngularSearcher<Value, Index, Allocator>
where
    Value: nk::Angular,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Angular>::Output: Copy + Default,
{
    /// A searcher walking with breadth `expansion_factor`, its scratch in
    /// `alloc`, uniform with [`DotSearcher::new`].
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(AngularSearcher {
            context: TraversalContext::new_in(expansion_factor, alloc)?,
        })
    }

    pub fn context(
        &mut self,
    ) -> &mut TraversalContext<Index, <Value as nk::Angular>::Output, Allocator> {
        &mut self.context
    }
}

/// Angular (cosine) distance over any NumKong scalar, in its native angular
/// output. Smaller is nearer, so it ranks like a Euclidean distance, only the
/// kernel differs. One generic marker spans every dtype, as [`DotMetric`] does.
pub struct AngularMetric<Value>(core::marker::PhantomData<Value>);

impl<Value: nk::Angular + 'static> MetricLike for AngularMetric<Value>
where
    <Value as nk::Angular>::Output: Copy + Default,
{
    type Query<'a> = &'a [Value];
    type Queries<'a> = nk::MatrixView<'a, Value>;
    type Score = <Value as nk::Angular>::Output;
}

impl<Metric, Value, Allocator> ScannerLike<Metric> for AngularScanner<Value, Allocator>
where
    Allocator: AllocatorLike,
    Value: nk::Angulars + nk::Angular + nk::Dots + Clone + Send + Sync + 'static,
    <Value as nk::Angulars>::SpatialResult: Send + Sync,
    Metric: MetricLike<Score = <Value as nk::Angulars>::SpatialResult> + ?Sized,
    for<'a> Metric::Queries<'a>: Into<nk::MatrixView<'a, Value>>,
{
    fn scan(
        &mut self,
        queries: Metric::Queries<'_>,
        table: &Table<impl AllocatorLike>,
        scores: &mut Matrix<Metric::Score>,
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
        nk::AngularsPackedParallelOps::try_angulars_packed_parallel_into(
            &values,
            &self.packed,
            &mut tile,
            &mut self.pool,
        )
        .map_err(|_| Error::Inconsistent)
    }
}

impl<Metric, Value, Index, Allocator> SearcherLike<Metric>
    for AngularSearcher<Value, Index, Allocator>
where
    Value: nk::Angular + 'static,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Angular>::Output: Copy + Default + PartialOrd,
    Metric: MetricLike<Score = <Value as nk::Angular>::Output> + ?Sized,
    for<'a> Metric::Query<'a>: Into<&'a [Value]>,
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        query: Metric::Query<'_>,
        table: &Table<impl AllocatorLike>,
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
            *slot = <Value as nk::Angular>::angular(query, value).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, Metric::Score, Allocator> {
        &mut self.context
    }

    fn compare(a: Metric::Score, b: Metric::Score) -> Ordering {
        Ranking::Nearest.compare(a, b)
    }

    fn search(
        &mut self,
        query: Metric::Query<'_>,
        table: &Table<impl AllocatorLike>,
    ) -> Result<(), Error> {
        walk::<Metric, Self>(self, query, table)
    }
}

/// An angular searcher whose neighbor width is settled by the tables it searches,
/// the cosine twin of [`EuclideanSearch`]: a pool plus one context per thread.
pub struct AngularSearch<Value: nk::Angular, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers:
        Workers<AngularSearcher<Value, u32, Allocator>, AngularSearcher<Value, u64, Allocator>>,
    alloc: Allocator,
}

impl<Value, Allocator> AngularSearch<Value, Allocator>
where
    Value: nk::Angular + 'static,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Angular>::Output: Copy + Default + PartialOrd,
{
    /// A searcher walking with breadth `expansion_factor` across the workers
    /// `scope` names, its scratch in `alloc`, uniform with [`DotSearch::new`].
    pub fn new(
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(AngularSearch {
            pool: open_pool(scope)?,
            expansion_factor,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, fanned across the pool. Uniform with
    /// [`AngularScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <Value as nk::Angular>::Output: Widen,
        AngularSearcher<Value, u32, Allocator>:
            SearcherLike<AngularMetric<Value>, Index = u32> + Send,
        AngularSearcher<Value, u64, Allocator>:
            SearcherLike<AngularMetric<Value>, Index = u64> + Send,
    {
        let threads = self.pool.threads_count();
        let expansion_factor = self.expansion_factor;
        match table.neighbors_column().dtype {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(threads, || {
                        AngularSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<AngularMetric<Value>, _>(
                    &mut self.pool,
                    workers,
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
                    self.workers = Workers::U64(build_workers(threads, || {
                        AngularSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<AngularMetric<Value>, _>(
                    &mut self.pool,
                    workers,
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

/// Reinterpret a query byte batch as a `nodes-agnostic` [`nk::MatrixView`] of the
/// table's dtype `T`, `width` from the value column — the shaped batch each erased
/// `*Any` hands to its typed scanner or searcher.
pub(crate) fn view_from_bytes<'q, Value: nk::StorageElement>(
    queries: &'q [u8],
    table: &Table<impl AllocatorLike>,
) -> Result<nk::MatrixView<'q, Value>, Error> {
    let width = table.value_width().ok_or(Error::Inconsistent)?;
    matrix_view(cast(queries).ok_or(Error::WrongType)?, width)
}

/// A dot scanner chosen at runtime from a file's value dtype string — the
/// within-family dispatch, so the choice of scalar stays a dense concern rather than
/// leaking into the cross-metric router. Each arm is a whole [`DotScanner`], fully
/// typed; only the result crosses out, node widened to `u64` and score to `f64`. The
/// typed scanner owns its own tile, so the dispatcher holds no scratch. Arms are
/// ordered by dynamic range, widest float to tightest int.
pub enum DotScannerAny<Allocator: AllocatorLike = Global> {
    F64(DotScanner<f64, Allocator>),
    F32(DotScanner<f32, Allocator>),
    Bf16(DotScanner<nk::bf16, Allocator>),
    F16(DotScanner<nk::f16, Allocator>),
    E5M2(DotScanner<nk::e5m2, Allocator>),
    E4M3(DotScanner<nk::e4m3, Allocator>),
    E3M2(DotScanner<nk::e3m2, Allocator>),
    E2M3(DotScanner<nk::e2m3, Allocator>),
    I8(DotScanner<i8, Allocator>),
    I4x2(DotScanner<nk::i4x2, Allocator>),
    U8(DotScanner<u8, Allocator>),
    U4x2(DotScanner<nk::u4x2, Allocator>),
    U1x8(DotScanner<nk::u1x8, Allocator>),
}

impl<Allocator: AllocatorLike + Clone> DotScannerAny<Allocator> {
    /// A scanner for `dtype` over the workers `scope` names. [`Error::UnknownDType`] if
    /// no arm covers it — including a dtype the format understands but this metric
    /// has not been instantiated for.
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => DotScannerAny::F64(DotScanner::new(scope, alloc.clone())?),
            "f32" => DotScannerAny::F32(DotScanner::new(scope, alloc.clone())?),
            "bf16" => DotScannerAny::Bf16(DotScanner::new(scope, alloc.clone())?),
            "f16" => DotScannerAny::F16(DotScanner::new(scope, alloc.clone())?),
            "e5m2" => DotScannerAny::E5M2(DotScanner::new(scope, alloc.clone())?),
            "e4m3" => DotScannerAny::E4M3(DotScanner::new(scope, alloc.clone())?),
            "e3m2" => DotScannerAny::E3M2(DotScanner::new(scope, alloc.clone())?),
            "e2m3" => DotScannerAny::E2M3(DotScanner::new(scope, alloc.clone())?),
            "i8" => DotScannerAny::I8(DotScanner::new(scope, alloc.clone())?),
            "i4x2" => DotScannerAny::I4x2(DotScanner::new(scope, alloc.clone())?),
            "u8" => DotScannerAny::U8(DotScanner::new(scope, alloc.clone())?),
            "u4x2" => DotScannerAny::U4x2(DotScanner::new(scope, alloc.clone())?),
            "u1x8" => DotScannerAny::U1x8(DotScanner::new(scope, alloc.clone())?),
            _ => return Err(Error::UnknownDType),
        })
    }

    /// The header dtype this scanner was built for.
    pub fn dtype(&self) -> &'static str {
        match self {
            DotScannerAny::F64(_) => "f64",
            DotScannerAny::F32(_) => "f32",
            DotScannerAny::Bf16(_) => "bf16",
            DotScannerAny::F16(_) => "f16",
            DotScannerAny::E5M2(_) => "e5m2",
            DotScannerAny::E4M3(_) => "e4m3",
            DotScannerAny::E3M2(_) => "e3m2",
            DotScannerAny::E2M3(_) => "e2m3",
            DotScannerAny::I8(_) => "i8",
            DotScannerAny::I4x2(_) => "i4x2",
            DotScannerAny::U8(_) => "u8",
            DotScannerAny::U4x2(_) => "u4x2",
            DotScannerAny::U1x8(_) => "u1x8",
        }
    }

    /// Batch brute-force top-k over a query batch given as raw bytes in the table's
    /// dtype, reinterpreted as the scalar's shaped view and forwarded to the typed
    /// scanner. `matches`/`distances` are `queries × k`, carrying `k` and the count.
    pub fn scan(
        &mut self,
        queries: &[u8],
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            DotScannerAny::F64(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::F32(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::Bf16(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::F16(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::E5M2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::E4M3(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::E3M2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::E2M3(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::I8(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::I4x2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::U8(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::U4x2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotScannerAny::U1x8(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// A dot searcher chosen from a file's value dtype string. As with
/// [`DotScannerAny`], the branch is on the value dtype alone; each arm's
/// [`DotSearch`] resolves the neighbor width from the table, so a `u32` graph
/// walks in `u32` cells and only the final results widen to `u64`.
pub enum DotSearcherAny<Allocator: AllocatorLike = Global> {
    F64(DotSearch<f64, Allocator>),
    F32(DotSearch<f32, Allocator>),
    Bf16(DotSearch<nk::bf16, Allocator>),
    F16(DotSearch<nk::f16, Allocator>),
    E5M2(DotSearch<nk::e5m2, Allocator>),
    E4M3(DotSearch<nk::e4m3, Allocator>),
    E3M2(DotSearch<nk::e3m2, Allocator>),
    E2M3(DotSearch<nk::e2m3, Allocator>),
    I8(DotSearch<i8, Allocator>),
    I4x2(DotSearch<nk::i4x2, Allocator>),
    U8(DotSearch<u8, Allocator>),
    U4x2(DotSearch<nk::u4x2, Allocator>),
    U1x8(DotSearch<nk::u1x8, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send> DotSearcherAny<Allocator> {
    /// A searcher for `dtype`, walking with breadth `expansion_factor` across
    /// the workers `scope` names, its scratch in `alloc`. No table is seen here: the neighbor width is settled
    /// inside [`DotSearch`] by the first searched table, never branched on here.
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => DotSearcherAny::F64(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "f32" => DotSearcherAny::F32(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "bf16" => DotSearcherAny::Bf16(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "f16" => DotSearcherAny::F16(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "e5m2" => DotSearcherAny::E5M2(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "e4m3" => DotSearcherAny::E4M3(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "e3m2" => DotSearcherAny::E3M2(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "e2m3" => DotSearcherAny::E2M3(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "i8" => DotSearcherAny::I8(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "i4x2" => DotSearcherAny::I4x2(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "u8" => DotSearcherAny::U8(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "u4x2" => DotSearcherAny::U4x2(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            "u1x8" => DotSearcherAny::U1x8(DotSearch::new(expansion_factor, scope, alloc.clone())?),
            _ => return Err(Error::UnknownDType),
        })
    }

    /// The header dtype this searcher was built for.
    pub fn dtype(&self) -> &'static str {
        match self {
            DotSearcherAny::F64(_) => "f64",
            DotSearcherAny::F32(_) => "f32",
            DotSearcherAny::Bf16(_) => "bf16",
            DotSearcherAny::F16(_) => "f16",
            DotSearcherAny::E5M2(_) => "e5m2",
            DotSearcherAny::E4M3(_) => "e4m3",
            DotSearcherAny::E3M2(_) => "e3m2",
            DotSearcherAny::E2M3(_) => "e2m3",
            DotSearcherAny::I8(_) => "i8",
            DotSearcherAny::I4x2(_) => "i4x2",
            DotSearcherAny::U8(_) => "u8",
            DotSearcherAny::U4x2(_) => "u4x2",
            DotSearcherAny::U1x8(_) => "u1x8",
        }
    }

    /// Batch top-k by walk, over a query batch given as raw bytes in the table's
    /// dtype, forwarded to the width-resolved leaf. `matches`/`distances` carry `k`
    /// and the query count.
    pub fn search(
        &mut self,
        queries: &[u8],
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            DotSearcherAny::F64(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::F32(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::Bf16(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::F16(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::E5M2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::E4M3(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::E3M2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::E2M3(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::I8(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::I4x2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::U8(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::U4x2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            DotSearcherAny::U1x8(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// The Euclidean twin of [`DotScannerAny`]: the same runtime dtype dispatch for the
/// smaller-is-better L2 metric. No `u1x8` arm — L2 is not a bit metric.
pub enum EuclideanScannerAny<Allocator: AllocatorLike = Global> {
    F64(EuclideanScanner<f64, Allocator>),
    F32(EuclideanScanner<f32, Allocator>),
    Bf16(EuclideanScanner<nk::bf16, Allocator>),
    F16(EuclideanScanner<nk::f16, Allocator>),
    E5M2(EuclideanScanner<nk::e5m2, Allocator>),
    E4M3(EuclideanScanner<nk::e4m3, Allocator>),
    E3M2(EuclideanScanner<nk::e3m2, Allocator>),
    E2M3(EuclideanScanner<nk::e2m3, Allocator>),
    I8(EuclideanScanner<i8, Allocator>),
    I4x2(EuclideanScanner<nk::i4x2, Allocator>),
    U8(EuclideanScanner<u8, Allocator>),
    U4x2(EuclideanScanner<nk::u4x2, Allocator>),
}

impl<Allocator: AllocatorLike + Clone> EuclideanScannerAny<Allocator> {
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => EuclideanScannerAny::F64(EuclideanScanner::new(scope, alloc.clone())?),
            "f32" => EuclideanScannerAny::F32(EuclideanScanner::new(scope, alloc.clone())?),
            "bf16" => EuclideanScannerAny::Bf16(EuclideanScanner::new(scope, alloc.clone())?),
            "f16" => EuclideanScannerAny::F16(EuclideanScanner::new(scope, alloc.clone())?),
            "e5m2" => EuclideanScannerAny::E5M2(EuclideanScanner::new(scope, alloc.clone())?),
            "e4m3" => EuclideanScannerAny::E4M3(EuclideanScanner::new(scope, alloc.clone())?),
            "e3m2" => EuclideanScannerAny::E3M2(EuclideanScanner::new(scope, alloc.clone())?),
            "e2m3" => EuclideanScannerAny::E2M3(EuclideanScanner::new(scope, alloc.clone())?),
            "i8" => EuclideanScannerAny::I8(EuclideanScanner::new(scope, alloc.clone())?),
            "i4x2" => EuclideanScannerAny::I4x2(EuclideanScanner::new(scope, alloc.clone())?),
            "u8" => EuclideanScannerAny::U8(EuclideanScanner::new(scope, alloc.clone())?),
            "u4x2" => EuclideanScannerAny::U4x2(EuclideanScanner::new(scope, alloc.clone())?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            EuclideanScannerAny::F64(_) => "f64",
            EuclideanScannerAny::F32(_) => "f32",
            EuclideanScannerAny::Bf16(_) => "bf16",
            EuclideanScannerAny::F16(_) => "f16",
            EuclideanScannerAny::E5M2(_) => "e5m2",
            EuclideanScannerAny::E4M3(_) => "e4m3",
            EuclideanScannerAny::E3M2(_) => "e3m2",
            EuclideanScannerAny::E2M3(_) => "e2m3",
            EuclideanScannerAny::I8(_) => "i8",
            EuclideanScannerAny::I4x2(_) => "i4x2",
            EuclideanScannerAny::U8(_) => "u8",
            EuclideanScannerAny::U4x2(_) => "u4x2",
        }
    }

    pub fn scan(
        &mut self,
        queries: &[u8],
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            EuclideanScannerAny::F64(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::F32(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::Bf16(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::F16(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::E5M2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::E4M3(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::E3M2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::E2M3(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::I8(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::I4x2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::U8(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanScannerAny::U4x2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// The Euclidean twin of [`DotSearcherAny`]: runtime dtype dispatch over a walk,
/// nearest-first. The neighbor width resolves inside each [`EuclideanSearch`].
pub enum EuclideanSearcherAny<Allocator: AllocatorLike = Global> {
    F64(EuclideanSearch<f64, Allocator>),
    F32(EuclideanSearch<f32, Allocator>),
    Bf16(EuclideanSearch<nk::bf16, Allocator>),
    F16(EuclideanSearch<nk::f16, Allocator>),
    E5M2(EuclideanSearch<nk::e5m2, Allocator>),
    E4M3(EuclideanSearch<nk::e4m3, Allocator>),
    E3M2(EuclideanSearch<nk::e3m2, Allocator>),
    E2M3(EuclideanSearch<nk::e2m3, Allocator>),
    I8(EuclideanSearch<i8, Allocator>),
    I4x2(EuclideanSearch<nk::i4x2, Allocator>),
    U8(EuclideanSearch<u8, Allocator>),
    U4x2(EuclideanSearch<nk::u4x2, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send> EuclideanSearcherAny<Allocator> {
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => EuclideanSearcherAny::F64(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "f32" => EuclideanSearcherAny::F32(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "bf16" => EuclideanSearcherAny::Bf16(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "f16" => EuclideanSearcherAny::F16(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "e5m2" => EuclideanSearcherAny::E5M2(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "e4m3" => EuclideanSearcherAny::E4M3(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "e3m2" => EuclideanSearcherAny::E3M2(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "e2m3" => EuclideanSearcherAny::E2M3(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "i8" => EuclideanSearcherAny::I8(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "i4x2" => EuclideanSearcherAny::I4x2(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "u8" => EuclideanSearcherAny::U8(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "u4x2" => EuclideanSearcherAny::U4x2(EuclideanSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            EuclideanSearcherAny::F64(_) => "f64",
            EuclideanSearcherAny::F32(_) => "f32",
            EuclideanSearcherAny::Bf16(_) => "bf16",
            EuclideanSearcherAny::F16(_) => "f16",
            EuclideanSearcherAny::E5M2(_) => "e5m2",
            EuclideanSearcherAny::E4M3(_) => "e4m3",
            EuclideanSearcherAny::E3M2(_) => "e3m2",
            EuclideanSearcherAny::E2M3(_) => "e2m3",
            EuclideanSearcherAny::I8(_) => "i8",
            EuclideanSearcherAny::I4x2(_) => "i4x2",
            EuclideanSearcherAny::U8(_) => "u8",
            EuclideanSearcherAny::U4x2(_) => "u4x2",
        }
    }

    pub fn search(
        &mut self,
        queries: &[u8],
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            EuclideanSearcherAny::F64(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::F32(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::Bf16(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::F16(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::E5M2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::E4M3(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::E3M2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::E2M3(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::I8(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::I4x2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::U8(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            EuclideanSearcherAny::U4x2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// The Angular twin of [`DotScannerAny`]: runtime dtype dispatch for cosine
/// distance. No `u1x8` arm.
pub enum AngularScannerAny<Allocator: AllocatorLike = Global> {
    F64(AngularScanner<f64, Allocator>),
    F32(AngularScanner<f32, Allocator>),
    Bf16(AngularScanner<nk::bf16, Allocator>),
    F16(AngularScanner<nk::f16, Allocator>),
    E5M2(AngularScanner<nk::e5m2, Allocator>),
    E4M3(AngularScanner<nk::e4m3, Allocator>),
    E3M2(AngularScanner<nk::e3m2, Allocator>),
    E2M3(AngularScanner<nk::e2m3, Allocator>),
    I8(AngularScanner<i8, Allocator>),
    I4x2(AngularScanner<nk::i4x2, Allocator>),
    U8(AngularScanner<u8, Allocator>),
    U4x2(AngularScanner<nk::u4x2, Allocator>),
}

impl<Allocator: AllocatorLike + Clone> AngularScannerAny<Allocator> {
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => AngularScannerAny::F64(AngularScanner::new(scope, alloc.clone())?),
            "f32" => AngularScannerAny::F32(AngularScanner::new(scope, alloc.clone())?),
            "bf16" => AngularScannerAny::Bf16(AngularScanner::new(scope, alloc.clone())?),
            "f16" => AngularScannerAny::F16(AngularScanner::new(scope, alloc.clone())?),
            "e5m2" => AngularScannerAny::E5M2(AngularScanner::new(scope, alloc.clone())?),
            "e4m3" => AngularScannerAny::E4M3(AngularScanner::new(scope, alloc.clone())?),
            "e3m2" => AngularScannerAny::E3M2(AngularScanner::new(scope, alloc.clone())?),
            "e2m3" => AngularScannerAny::E2M3(AngularScanner::new(scope, alloc.clone())?),
            "i8" => AngularScannerAny::I8(AngularScanner::new(scope, alloc.clone())?),
            "i4x2" => AngularScannerAny::I4x2(AngularScanner::new(scope, alloc.clone())?),
            "u8" => AngularScannerAny::U8(AngularScanner::new(scope, alloc.clone())?),
            "u4x2" => AngularScannerAny::U4x2(AngularScanner::new(scope, alloc.clone())?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            AngularScannerAny::F64(_) => "f64",
            AngularScannerAny::F32(_) => "f32",
            AngularScannerAny::Bf16(_) => "bf16",
            AngularScannerAny::F16(_) => "f16",
            AngularScannerAny::E5M2(_) => "e5m2",
            AngularScannerAny::E4M3(_) => "e4m3",
            AngularScannerAny::E3M2(_) => "e3m2",
            AngularScannerAny::E2M3(_) => "e2m3",
            AngularScannerAny::I8(_) => "i8",
            AngularScannerAny::I4x2(_) => "i4x2",
            AngularScannerAny::U8(_) => "u8",
            AngularScannerAny::U4x2(_) => "u4x2",
        }
    }

    pub fn scan(
        &mut self,
        queries: &[u8],
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            AngularScannerAny::F64(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::F32(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::Bf16(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::F16(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::E5M2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::E4M3(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::E3M2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::E2M3(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::I8(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::I4x2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::U8(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularScannerAny::U4x2(s) => s.scan(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// The Angular twin of [`DotSearcherAny`]: runtime dtype dispatch over a walk,
/// nearest-first, width resolved inside each [`AngularSearch`].
pub enum AngularSearcherAny<Allocator: AllocatorLike = Global> {
    F64(AngularSearch<f64, Allocator>),
    F32(AngularSearch<f32, Allocator>),
    Bf16(AngularSearch<nk::bf16, Allocator>),
    F16(AngularSearch<nk::f16, Allocator>),
    E5M2(AngularSearch<nk::e5m2, Allocator>),
    E4M3(AngularSearch<nk::e4m3, Allocator>),
    E3M2(AngularSearch<nk::e3m2, Allocator>),
    E2M3(AngularSearch<nk::e2m3, Allocator>),
    I8(AngularSearch<i8, Allocator>),
    I4x2(AngularSearch<nk::i4x2, Allocator>),
    U8(AngularSearch<u8, Allocator>),
    U4x2(AngularSearch<nk::u4x2, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send> AngularSearcherAny<Allocator> {
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => {
                AngularSearcherAny::F64(AngularSearch::new(expansion_factor, scope, alloc.clone())?)
            }
            "f32" => {
                AngularSearcherAny::F32(AngularSearch::new(expansion_factor, scope, alloc.clone())?)
            }
            "bf16" => AngularSearcherAny::Bf16(AngularSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "f16" => {
                AngularSearcherAny::F16(AngularSearch::new(expansion_factor, scope, alloc.clone())?)
            }
            "e5m2" => AngularSearcherAny::E5M2(AngularSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "e4m3" => AngularSearcherAny::E4M3(AngularSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "e3m2" => AngularSearcherAny::E3M2(AngularSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "e2m3" => AngularSearcherAny::E2M3(AngularSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "i8" => {
                AngularSearcherAny::I8(AngularSearch::new(expansion_factor, scope, alloc.clone())?)
            }
            "i4x2" => AngularSearcherAny::I4x2(AngularSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            "u8" => {
                AngularSearcherAny::U8(AngularSearch::new(expansion_factor, scope, alloc.clone())?)
            }
            "u4x2" => AngularSearcherAny::U4x2(AngularSearch::new(
                expansion_factor,
                scope,
                alloc.clone(),
            )?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            AngularSearcherAny::F64(_) => "f64",
            AngularSearcherAny::F32(_) => "f32",
            AngularSearcherAny::Bf16(_) => "bf16",
            AngularSearcherAny::F16(_) => "f16",
            AngularSearcherAny::E5M2(_) => "e5m2",
            AngularSearcherAny::E4M3(_) => "e4m3",
            AngularSearcherAny::E3M2(_) => "e3m2",
            AngularSearcherAny::E2M3(_) => "e2m3",
            AngularSearcherAny::I8(_) => "i8",
            AngularSearcherAny::I4x2(_) => "i4x2",
            AngularSearcherAny::U8(_) => "u8",
            AngularSearcherAny::U4x2(_) => "u4x2",
        }
    }

    pub fn search(
        &mut self,
        queries: &[u8],
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            AngularSearcherAny::F64(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::F32(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::Bf16(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::F16(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::E5M2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::E4M3(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::E3M2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::E2M3(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::I8(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::I4x2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::U8(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
            AngularSearcherAny::U4x2(s) => s.search(
                view_from_bytes(queries, table)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NeighborList, OffsetDType, Shape};

    /// Three nodes whose values are held in dtype `T` in a NumKong matrix, so one
    /// fixture serves every scalar — float, minifloat, integer, nibble, or packed
    /// bit. [`try_from_scalars`] packs the logical `f32` rows into `T` in whatever
    /// layout the scalar demands, so a sub-byte `i4x2` or `u1x8` builds the same way
    /// a byte type does, and the matrix carries its own logical width. Fully
    /// connected at degree two, so a walk from any landmark reaches all three; degree
    /// zero gives an edgeless graph.
    ///
    /// [`try_from_scalars`]: nk::Matrix::try_from_scalars
    struct Ring<Value: nk::FloatConvertible> {
        dtype: &'static str,
        values: nk::Matrix<Value>,
        degree: usize,
    }

    /// The dense ring `[1,2] [3,4] [5,6]` and its query `[3,5]`, three 2-D rows for
    /// every metric a real number can score.
    const RING_ROWS: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const RING_QUERY: [f32; 2] = [3.0, 5.0];

    /// A binary ring for the bit type `u1x8` and the set metrics: three 8-bit rows
    /// nested `11111111 ⊃ 11110000 ⊃ 10000000` and a query `11111110`, so node 0 is
    /// nearest and node 2 farthest under dot, Hamming, and Jaccard alike.
    const BINARY_ROWS: [f32; 24] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // node 0
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, // node 1
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // node 2
    ];
    const BINARY_QUERY: [f32; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];

    fn ring<Value: nk::FloatConvertible>(
        dtype: &'static str,
        rows: &[f32],
        width: usize,
        degree: usize,
    ) -> Ring<Value> {
        let node_count = rows.len() / width;
        Ring {
            dtype,
            values: nk::Matrix::try_from_scalars(rows, &[node_count, width]).unwrap(),
            degree,
        }
    }

    fn ring_i8() -> Ring<i8> {
        ring::<i8>("i8", &RING_ROWS, 2, 2)
    }

    impl<Value: nk::FloatConvertible> GraphLike for Ring<Value> {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            self.dtype
        }
        fn value_shape(&self) -> Option<Shape> {
            // The logical column count the matrix already knows.
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
            // Row `i`'s packed storage scalars, sliced from the matrix's own storage
            // slice — `as_slice` already returns storage units, so a byte type gets
            // `width` scalars and a sub-byte one their packed fraction. `row()` is
            // avoided: it is unsound for packed scalars.
            let storage = self.values.as_slice();
            let per_row = storage.len() / self.values.shape()[0];
            let row = &storage[i * per_row..(i + 1) * per_row];
            // SAFETY: reinterpret the contiguous packed row as the little-endian
            // bytes the writer concatenates into the value column.
            unsafe {
                core::slice::from_raw_parts(row.as_ptr() as *const u8, core::mem::size_of_val(row))
            }
        }
        fn neighbors(&self, i: usize) -> NeighborList<'_> {
            // Each node links to the other two, as little-endian `u32` indices.
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

    /// Build the ring `rows` in `dtype`, run `query` through both a scan and a walk,
    /// and assert each ranks the nodes in `order`. Ranking uses the searcher's own
    /// comparator, so a smaller-is-better metric passes the same helper with no
    /// change; scan and walk agreeing is the guard that batched expansion scored a
    /// whole neighbor list the same as a full sweep. One line per `(metric, dtype)`
    /// extends coverage — the point of collapsing the descriptors into generics.
    /// Build the ring `rows` in `dtype` and its one-row query, run both a batch scan
    /// and a batch walk, and assert each ranks the nodes `order` in `matches`. Scan
    /// and walk agreeing is the guard that the packed sweep and the scalar walk score
    /// identically. `T`'s [`nk::Matrix::view`] is the shaped [`MetricLike::Queries`].
    fn ring_table<Value: nk::FloatConvertible>(
        dtype: &'static str,
        rows: &[f32],
        query: &[f32],
    ) -> (Table, nk::Matrix<Value>) {
        let width = query.len();
        let node_count = rows.len() / width;
        let values = nk::Matrix::<Value>::try_from_scalars(rows, &[node_count, width]).unwrap();
        let table = Table::build(
            &Ring {
                dtype,
                values,
                degree: 2,
            },
            None,
        )
        .unwrap();
        (
            table,
            nk::Matrix::<Value>::try_from_scalars(query, &[1, width]).unwrap(),
        )
    }

    fn dot_covers<Value>(dtype: &'static str, rows: &[f32], query: &[f32], order: [u64; 3])
    where
        Value: nk::FloatConvertible + nk::Dot + nk::Dots + Clone + Send + Sync + 'static,
        Value: nk::Dot<Output = <Value as nk::Dots>::Accumulator>,
        <Value as nk::Dot>::Output: Copy + Default + PartialOrd + Widen + Send + Sync,
        DotSearcher<Value, u32>: SearcherLike<DotMetric<Value>, Index = u32> + Send,
        DotSearcher<Value, u64>: SearcherLike<DotMetric<Value>, Index = u64> + Send,
    {
        let (table, query) = ring_table::<Value>(dtype, rows, query);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        DotScanner::<Value>::new(ExecutionScope::Serial, Global)
            .unwrap()
            .scan(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &order, "{dtype} scan");
        DotSearch::<Value>::new(3, ExecutionScope::Serial, Global)
            .unwrap()
            .search(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &order, "{dtype} walk");
    }

    fn euclidean_covers<Value>(dtype: &'static str, rows: &[f32], query: &[f32], order: [u64; 3])
    where
        Value:
            nk::FloatConvertible + nk::Euclidean + nk::Euclideans + Clone + Send + Sync + 'static,
        Value: nk::Euclidean<EuclideanOutput = <Value as nk::Euclideans>::SpatialResult>,
        <Value as nk::Euclidean>::EuclideanOutput:
            Copy + Default + PartialOrd + Widen + Send + Sync,
        EuclideanSearcher<Value, u32>: SearcherLike<EuclideanMetric<Value>, Index = u32> + Send,
        EuclideanSearcher<Value, u64>: SearcherLike<EuclideanMetric<Value>, Index = u64> + Send,
    {
        let (table, query) = ring_table::<Value>(dtype, rows, query);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        EuclideanScanner::<Value>::new(ExecutionScope::Serial, Global)
            .unwrap()
            .scan(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &order, "{dtype} scan");
        EuclideanSearch::<Value>::new(3, ExecutionScope::Serial, Global)
            .unwrap()
            .search(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &order, "{dtype} walk");
    }

    fn angular_covers<Value>(dtype: &'static str, rows: &[f32], query: &[f32], order: [u64; 3])
    where
        Value: nk::FloatConvertible + nk::Angular + nk::Angulars + Clone + Send + Sync + 'static,
        Value: nk::Angular<Output = <Value as nk::Angulars>::SpatialResult>,
        <Value as nk::Angular>::Output: Copy + Default + PartialOrd + Widen + Send + Sync,
        AngularSearcher<Value, u32>: SearcherLike<AngularMetric<Value>, Index = u32> + Send,
        AngularSearcher<Value, u64>: SearcherLike<AngularMetric<Value>, Index = u64> + Send,
    {
        let (table, query) = ring_table::<Value>(dtype, rows, query);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        AngularScanner::<Value>::new(ExecutionScope::Serial, Global)
            .unwrap()
            .scan(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &order, "{dtype} scan");
        AngularSearch::<Value>::new(3, ExecutionScope::Serial, Global)
            .unwrap()
            .search(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &order, "{dtype} walk");
    }

    #[test]
    fn dot_ranks_and_walks_every_dtype() {
        // The dot of `[3,5]` against the ring is `[13, 29, 45]`, so nodes rank
        // `[2, 1, 0]` in every real dtype. One line each, sorted by dynamic range —
        // widest float to tightest int — over the whole NumKong scalar zoo.
        let dense = [2u64, 1, 0];
        dot_covers::<f64>("f64", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<f32>("f32", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::bf16>("bf16", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::f16>("f16", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::e5m2>("e5m2", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::e4m3>("e4m3", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::e3m2>("e3m2", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::e2m3>("e2m3", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<i8>("i8", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::i4x2>("i4x2", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<u8>("u8", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::u4x2>("u4x2", &RING_ROWS, &RING_QUERY, dense);
        // The bit type takes the binary ring: dot is `|A ∩ B|`, larger first.
        dot_covers::<nk::u1x8>("u1x8", &BINARY_ROWS, &BINARY_QUERY, [0, 1, 2]);
    }

    #[test]
    fn euclidean_ranks_and_walks_every_dtype() {
        // Smaller-is-better: `[3,5]` sits nearest node 1, then 2, then 0, so the same
        // helper — ranking by the searcher's reversed comparator — yields `[1, 2, 0]`,
        // a different order from the dot over the very same fixture. No `u1x8`: L2 is
        // not a bit metric.
        let near = [1u64, 2, 0];
        euclidean_covers::<f64>("f64", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<f32>("f32", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::bf16>("bf16", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::f16>("f16", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::e5m2>("e5m2", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::e4m3>("e4m3", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::e3m2>("e3m2", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::e2m3>("e2m3", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<i8>("i8", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::i4x2>("i4x2", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<u8>("u8", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::u4x2>("u4x2", &RING_ROWS, &RING_QUERY, near);
    }

    #[test]
    fn angular_ranks_and_walks_every_dtype() {
        // Cosine distance: `[3,5]` is most aligned with node 0, least with node 2, so
        // nearest-first is `[0, 1, 2]` — a third distinct order from one fixture,
        // proving the metric, not the data, decides the ranking.
        let aligned = [0u64, 1, 2];
        angular_covers::<f64>("f64", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<f32>("f32", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::bf16>("bf16", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::f16>("f16", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::e5m2>("e5m2", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::e4m3>("e4m3", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::e3m2>("e3m2", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::e2m3>("e2m3", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<i8>("i8", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::i4x2>("i4x2", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<u8>("u8", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::u4x2>("u4x2", &RING_ROWS, &RING_QUERY, aligned);
    }

    #[test]
    fn batch_scan_and_search_fill_matrices() {
        // The proposed uniform batch shape: one query row, top-3 out, `k` and the
        // query count read off the `matches`/`distances` matrices, `counts` reporting
        // how many each query found. Query `[3,5]` dots to `[13,29,45]`, so both the
        // brute-force scan and the graph walk rank nodes `[2,1,0]`, scores `[45,29,13]`.
        let table = Table::build(&ring_i8(), None).unwrap();
        let queries = i8_query(&[3.0, 5.0]);

        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut scanner: DotScanner<i8> = DotScanner::new(ExecutionScope::Serial, Global).unwrap();
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
        assert_eq!(matches.row(0).unwrap(), &[2, 1, 0]);
        assert_eq!(distances.row(0).unwrap(), &[45.0, 29.0, 13.0]);

        let mut searcher = DotSearch::<i8>::new(3, ExecutionScope::Serial, Global).unwrap();
        let mut walked = Matrix::<u64>::new(1, 3).unwrap();
        let mut walked_scores = Matrix::<f64>::new(1, 3).unwrap();
        searcher
            .search(
                queries.view(),
                &table,
                &mut walked,
                &mut walked_scores,
                &mut counts,
            )
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(walked.row(0).unwrap(), &[2, 1, 0]);
        assert_eq!(walked_scores.row(0).unwrap(), &[45.0, 29.0, 13.0]);
    }

    /// A one-row `i8` query batch as a shaped matrix, for the behavior tests.
    fn i8_query(elems: &[f32]) -> nk::Matrix<i8> {
        nk::Matrix::<i8>::try_from_scalars(elems, &[1, elems.len()]).unwrap()
    }

    #[test]
    fn scan_over_a_materialized_file() {
        // The core promise: build an index, serialize it to the on-disk image, parse
        // that image back, and scan the parsed table — not the in-memory fixture.
        let built = Table::build(&ring_i8(), None).unwrap();
        let image = built.as_bytes().to_vec();
        let table = Table::from_bytes(&image).unwrap();

        let mut scanner = DotScanner::<i8>::new(ExecutionScope::Cpus(2), Global).unwrap();
        let query = i8_query(&[3.0, 4.0]);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        scanner
            .scan(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        // Dots against [3,4]: node0=11, node1=25, node2=39 → ranked [2,1,0].
        assert_eq!(matches.row(0).unwrap(), &[2, 1, 0]);
        assert_eq!(distances.row(0).unwrap(), &[39.0, 25.0, 11.0]);
    }

    #[test]
    fn euclidean_scan_pins_exact_distances() {
        // The distances themselves, not just the ranking: node 1 sits at `√8`, node 0
        // and 2 farther, confirming the kernel returns real L2 distance, not squared.
        let table = Table::build(&ring_i8(), None).unwrap();
        let mut scanner = EuclideanScanner::<i8>::new(ExecutionScope::Serial, Global).unwrap();
        let query = i8_query(&[1.0, 2.0]);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        scanner
            .scan(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        // Query equals node 0: distance 0 there, `√8` and `√32` to nodes 1 and 2.
        assert_eq!(matches.row(0).unwrap(), &[0, 1, 2]);
        assert_eq!(distances.row(0).unwrap()[0], 0.0);
        assert!((distances.row(0).unwrap()[1] - 8f64.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn batch_walks_many_queries() {
        // A three-query batch through one searcher — the shaped input carries the
        // count, and `matches`/`distances` come back `queries × k`, one row each.
        let table = Table::build(&ring_i8(), None).unwrap();
        let queries =
            nk::Matrix::<i8>::try_from_scalars(&[3.0, 4.0, 5.0, 6.0, 1.0, 2.0], &[3, 2]).unwrap();
        let mut searcher = DotSearch::<i8>::new(3, ExecutionScope::Serial, Global).unwrap();
        let mut matches = Matrix::<u64>::new(3, 2).unwrap();
        let mut distances = Matrix::<f64>::new(3, 2).unwrap();
        let mut counts = [0u64; 3];
        searcher
            .search(
                queries.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(counts, [2, 2, 2]);
        // q0·[3,4]→39,25; q1·[5,6]→61,39; q2·[1,2]→17,11; every walk reaches [2,1].
        assert_eq!(matches.row(0).unwrap(), &[2, 1]);
        assert_eq!(matches.row(1).unwrap(), &[2, 1]);
        assert_eq!(matches.row(2).unwrap(), &[2, 1]);
        assert_eq!(distances.row(0).unwrap(), &[39.0, 25.0]);
        assert_eq!(distances.row(1).unwrap(), &[61.0, 39.0]);
    }

    #[test]
    fn parallel_search_matches_single_thread() {
        // The fan-out is only correct if a four-thread run lands the exact same top-k
        // as a one-thread run: the dynamic schedule hands each query to whichever
        // thread is free, and each must write its own row of the outputs with no
        // crosstalk. Eight varied queries make the schedule interleave.
        let table = Table::build(&ring_i8(), None).unwrap();
        let scalars: [f32; 16] = [
            3.0, 5.0, 1.0, 4.0, 5.0, 2.0, 2.0, 6.0, 4.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ];
        let queries = nk::Matrix::<i8>::try_from_scalars(&scalars, &[8, 2]).unwrap();

        let run = |scope: ExecutionScope| {
            let mut searcher = DotSearch::<i8>::new(3, scope, Global).unwrap();
            let mut matches = Matrix::<u64>::new(8, 3).unwrap();
            let mut distances = Matrix::<f64>::new(8, 3).unwrap();
            let mut counts = [0u64; 8];
            searcher
                .search(
                    queries.view(),
                    &table,
                    &mut matches,
                    &mut distances,
                    &mut counts,
                )
                .unwrap();
            (matches, distances, counts)
        };

        let (sequential_matches, sequential_distances, sequential_counts) =
            run(ExecutionScope::Serial);
        let (parallel_matches, parallel_distances, parallel_counts) = run(ExecutionScope::Cpus(4));
        assert_eq!(sequential_counts, parallel_counts);
        for query in 0..8 {
            assert_eq!(
                sequential_matches.row(query).unwrap(),
                parallel_matches.row(query).unwrap(),
                "query {query} matches"
            );
            assert_eq!(
                sequential_distances.row(query).unwrap(),
                parallel_distances.row(query).unwrap(),
                "query {query} distances"
            );
        }
    }

    /// Two disconnected triangles. A single entry point reaches only its own
    /// component, so finding the global best requires the starvation restart to
    /// re-seed into the other half — the behavior fixed roots could never give.
    struct TwoTriangles;
    impl GraphLike for TwoTriangles {
        fn len(&self) -> usize {
            6
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "i8"
        }
        fn value_shape(&self) -> Option<Shape> {
            Some(Shape::Vector(2))
        }
        fn neighbors_degree(&self) -> Option<usize> {
            Some(2)
        }
        fn key(&self, i: usize) -> &[u8] {
            const KEYS: [[u8; 8]; 6] = [
                (0u64).to_le_bytes(),
                (1u64).to_le_bytes(),
                (2u64).to_le_bytes(),
                (3u64).to_le_bytes(),
                (4u64).to_le_bytes(),
                (5u64).to_le_bytes(),
            ];
            &KEYS[i]
        }
        fn value(&self, i: usize) -> &[u8] {
            // Ascending dot with [1,1]: node 5 is the global best, in the far
            // component from node 0.
            const VALUES: [[u8; 2]; 6] = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]];
            &VALUES[i]
        }
        fn neighbors(&self, i: usize) -> NeighborList<'_> {
            // {0,1,2} and {3,4,5}, each a triangle, no edge between the two.
            const ADJACENCY: [[u8; 8]; 6] = [
                [1, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [4, 0, 0, 0, 5, 0, 0, 0],
                [3, 0, 0, 0, 5, 0, 0, 0],
                [3, 0, 0, 0, 4, 0, 0, 0],
            ];
            NeighborList::new(&ADJACENCY[i], OffsetDType::U32)
        }
    }

    #[test]
    fn walk_restarts_across_components() {
        let table = Table::build(&TwoTriangles, None).unwrap();
        // ef spans the whole graph, so the walk must keep restarting until every node
        // is visited — and only then can it name the true top-3.
        let mut searcher = DotSearch::<i8>::new(6, ExecutionScope::Serial, Global).unwrap();
        let query = i8_query(&[1.0, 1.0]);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        searcher
            .search(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(counts, [3]);
        // Dots against [1,1]: node i scores 2*(i+1); node 5 = 12 is best, and it lives
        // in the component no fixed root of node 0 would reach.
        assert_eq!(matches.row(0).unwrap(), &[5, 4, 3]);
        assert_eq!(distances.row(0).unwrap(), &[12.0, 10.0, 8.0]);
    }

    #[test]
    fn one_searcher_serves_many_tables() {
        // The construction-is-configuration contract: a searcher built with no table
        // in sight serves a three-node ring, then a six-node graph, the scratch
        // growing to fit each at walk time and the width lane settling on first use.
        let ring = Table::build(&ring_i8(), None).unwrap();
        let triangles = Table::build(&TwoTriangles, None).unwrap();
        let mut searcher = DotSearch::<i8>::new(6, ExecutionScope::Serial, Global).unwrap();
        let query = i8_query(&[1.0, 1.0]);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        searcher
            .search(
                query.view(),
                &ring,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        // Ring dots against [1,1]: [3, 7, 11] — larger first.
        assert_eq!(matches.row(0).unwrap(), &[2, 1, 0]);

        searcher
            .search(
                query.view(),
                &triangles,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        // The bigger graph's global best, reachable only after the scratch grew.
        assert_eq!(matches.row(0).unwrap(), &[5, 4, 3]);
    }

    #[test]
    fn scan_f32_accumulates_in_f64() {
        // The same ring in `f32`, degree zero so the fixture also covers an edgeless
        // graph; the tile lands exactly in the `f64` accumulator.
        let table = Table::build(&ring::<f32>("f32", &RING_ROWS, 2, 0), None).unwrap();
        let mut scanner = DotScanner::<f32>::new(ExecutionScope::Cpus(2), Global).unwrap();
        let query = nk::Matrix::<f32>::try_from_scalars(&[3.0, 4.0], &[1, 2]).unwrap();
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        scanner
            .scan(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        // Dots against [3,4]: 11,25,39 → ranked [2,1,0] with exact `f64` distances.
        assert_eq!(matches.row(0).unwrap(), &[2, 1, 0]);
        assert_eq!(distances.row(0).unwrap(), &[39.0, 25.0, 11.0]);
    }

    #[test]
    fn scan_rejects_a_mismatched_output() {
        // The output matrices carry the query count; a `matches` claiming two queries
        // against a one-query batch is inconsistent, and the scan says so.
        let table = Table::build(&ring_i8(), None).unwrap();
        let mut scanner = DotScanner::<i8>::new(ExecutionScope::Serial, Global).unwrap();
        let query = i8_query(&[3.0, 4.0]);
        let mut matches = Matrix::<u64>::new(2, 3).unwrap();
        let mut distances = Matrix::<f64>::new(2, 3).unwrap();
        let mut counts = [0u64; 2];
        assert!(matches!(
            scanner.scan(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts
            ),
            Err(Error::Inconsistent)
        ));
    }

    #[test]
    fn neighbor_list_reads_its_native_width() {
        // A `NeighborList` yields a native `&[I]` only at its own width; the wrong
        // width is `None`, so a searcher can never reinterpret node indices. The
        // materializer minimizes small graphs to `u32`, so both widths are checked
        // here at the source rather than through a built table.
        let u32_bytes: [u8; 8] = [1, 0, 0, 0, 2, 0, 0, 0];
        let list = NeighborList::new(&u32_bytes, OffsetDType::U32);
        assert_eq!(list.as_slice::<u32>().unwrap(), &[1u32, 2]);
        assert!(list.as_slice::<u64>().is_none());

        let u64_bytes: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
        let wide = NeighborList::new(&u64_bytes, OffsetDType::U64);
        assert_eq!(wide.as_slice::<u64>().unwrap(), &[1u64, 2]);
        assert!(wide.as_slice::<u32>().is_none());
    }

    #[test]
    fn u64_leaf_walks() {
        // The width is a real parameter, not a fixed `u32`: a `DotSearcher<i8, u64>` is
        // constructible and runs the per-query walk. This table's neighbors are `u32`,
        // so reading them at `u64` is a genuine width mismatch — [`Error::WrongType`] —
        // exactly the check that proves `I` is honored at the neighbor read. A true
        // `u64`-neighbor table needs a graph past four billion nodes to build.
        let table = Table::build(&ring_i8(), None).unwrap();
        let mut searcher: DotSearcher<i8, u64> = DotSearcher::new(3, Global).unwrap();
        let query: [i8; 2] = [3, 4];
        assert!(matches!(
            <DotSearcher<i8, u64> as SearcherLike<DotMetric<i8>>>::search(
                &mut searcher,
                &query,
                &table,
            ),
            Err(Error::WrongType)
        ));
    }

    #[test]
    fn dispatcher_resolves_dtype_and_width() {
        // A caller who opened the file blind: dtype from the header, neighbor width
        // from the column, both resolved inside the searcher, not at the call site.
        let table = Table::build(&ring_i8(), None).unwrap();
        let mut searcher =
            DotSearcherAny::new(table.value_dtype(), 3, ExecutionScope::Serial, Global).unwrap();
        assert_eq!(searcher.dtype(), "i8");
        let query: [i8; 2] = [3, 4];
        let bytes = unsafe { core::slice::from_raw_parts(query.as_ptr() as *const u8, 2) };
        let mut matches = Matrix::<u64>::new(1, 2).unwrap();
        let mut distances = Matrix::<f64>::new(1, 2).unwrap();
        let mut counts = [0u64; 1];
        searcher
            .search(bytes, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[2, 1]);
        assert_eq!(distances.row(0).unwrap(), &[39.0, 25.0]);
    }

    #[test]
    fn scan_dispatches_on_header_dtype() {
        // The caller never names i8: it opens bytes and reads the dtype string.
        let image = Table::build(&ring_i8(), None).unwrap().as_bytes().to_vec();
        let table = Table::from_bytes(&image).unwrap();

        let mut scanner =
            DotScannerAny::new(table.value_dtype(), ExecutionScope::Serial, Global).unwrap();
        assert_eq!(scanner.dtype(), "i8");

        let query: [i8; 2] = [3, 4];
        let bytes = unsafe { core::slice::from_raw_parts(query.as_ptr() as *const u8, 2) };
        let mut matches = Matrix::<u64>::new(1, 2).unwrap();
        let mut distances = Matrix::<f64>::new(1, 2).unwrap();
        let mut counts = [0u64; 1];
        scanner
            .scan(bytes, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        // i8 dots widened to f64: node2=39, node1=25 — same order as the typed path.
        assert_eq!(matches.row(0).unwrap(), &[2, 1]);
        assert_eq!(distances.row(0).unwrap(), &[39.0, 25.0]);
    }

    #[test]
    fn dispatch_rejects_an_uninstantiated_dtype() {
        // `u16` is a dtype the format understands but Dot has no kernel for, so the
        // router rejects it rather than silently mismatching a scalar.
        assert!(matches!(
            DotScannerAny::new("u16", ExecutionScope::Serial, Global),
            Err(Error::UnknownDType)
        ));
    }

    #[test]
    fn euclidean_and_angular_dispatch_rank_nearest() {
        // The erased runtime path for the two distance metrics: dispatch on the header
        // dtype and rank nearest-first, the scan for Euclidean and the walk for
        // Angular, from the same `[3,5]` query.
        let table = Table::build(&ring_i8(), None).unwrap();
        let query: [i8; 2] = [3, 5];
        let bytes = unsafe { core::slice::from_raw_parts(query.as_ptr() as *const u8, 2) };
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut euclidean =
            EuclideanScannerAny::new(table.value_dtype(), ExecutionScope::Serial, Global).unwrap();
        assert_eq!(euclidean.dtype(), "i8");
        euclidean
            .scan(bytes, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[1, 2, 0]);

        let mut angular =
            AngularSearcherAny::new(table.value_dtype(), 3, ExecutionScope::Serial, Global)
                .unwrap();
        assert_eq!(angular.dtype(), "i8");
        angular
            .search(bytes, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1, 2]);
    }

    /// A three-node ring whose fixed degree-three adjacency pads every row's third
    /// slot with an out-of-range sentinel, the value `99`. A walk must skip it; had
    /// it not, scoring node `99` would read past the value column and fail.
    struct SentinelRing;
    impl GraphLike for SentinelRing {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "i8"
        }
        fn value_shape(&self) -> Option<Shape> {
            Some(Shape::Vector(2))
        }
        fn neighbors_degree(&self) -> Option<usize> {
            Some(3)
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
            const VALUES: [[u8; 2]; 3] = [[1, 2], [3, 4], [5, 6]];
            &VALUES[i]
        }
        fn neighbors(&self, i: usize) -> NeighborList<'_> {
            // The other two nodes, then a sentinel `99` well past the node count.
            const ADJACENCY: [[u8; 12]; 3] = [
                [1, 0, 0, 0, 2, 0, 0, 0, 99, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 99, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 99, 0, 0, 0],
            ];
            NeighborList::new(&ADJACENCY[i], OffsetDType::U32)
        }
    }

    #[test]
    fn walk_skips_neighbor_sentinels() {
        let table = Table::build(&SentinelRing, None).unwrap();
        let mut searcher = DotSearch::<i8>::new(3, ExecutionScope::Serial, Global).unwrap();
        let query = i8_query(&[3.0, 4.0]);
        let mut matches = Matrix::<u64>::new(1, 2).unwrap();
        let mut distances = Matrix::<f64>::new(1, 2).unwrap();
        let mut counts = [0u64; 1];
        searcher
            .search(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        // The same top-k as the unpadded ring: `99` was skipped, never scored.
        assert_eq!(matches.row(0).unwrap(), &[2, 1]);
        assert_eq!(distances.row(0).unwrap(), &[39.0, 25.0]);
    }
}

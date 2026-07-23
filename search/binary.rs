//! Binary-vector metrics over NumKong: Hamming and Jaccard distance over packed
//! bits. A node's value is a `u1x8` bit string, eight lanes to the byte, and both
//! metrics are smaller-is-better distances, so this reads like the smaller-is-better
//! half of `dense` — an [`AngularScanner`]-shaped packed scan and a greedy walk —
//! only the kernel and the scalar differ.
//!
//! [`nk::Hammings`] and [`nk::Jaccards`], the packed scan kernels, ship for `u1x8`
//! alone, and a scan and a walk must agree, so each family instantiates the one bit
//! type. The runtime dispatchers still take the full `*Any` shape of the dense
//! families, so a bit type added upstream is one arm here.
//!
//! The packing-aware [`matrix_view`]/[`value_view`] scan primitive is reused from
//! `dense`, and the shaped [`batch_scan`]/[`batch_search`] top-k from the shared
//! `batch` engine; only the metric bodies live here.
//!
//! [`AngularScanner`]: crate::search::dense::AngularScanner
//! [`batch_scan`]: crate::search::batch::batch_scan
//! [`batch_search`]: crate::search::batch::batch_search

use super::batch::{batch_scan, batch_search, build_workers, open_pool, Workers};
use super::{
    walk, ExecutionScope, Matrix, MetricLike, Ranking, ScannerLike, SearcherLike, TraversalContext,
};
use crate::search::dense::{value_view, view_from_bytes};
use crate::{cast, Error, GraphLike, IndexLike, OffsetDType, Table, Widen};
use allocator_api2::alloc::{Allocator as AllocatorLike, Global};
use core::cmp::Ordering;
use forkunion as fu;
use numkong as nk;

/// Best-first for a distance: `Greater` means `a` is the nearer, so a smaller score
/// outranks a larger one. Both bit metrics rank this way.
fn smaller_first<Score: PartialOrd>(a: Score, b: Score) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Less)
}

// region: Hamming

/// The batch half of the Hamming metric. Structurally the [`AngularScanner`] of the
/// bit world: it packs the query side through [`nk::DotsPackedMatrix`] and streams
/// the value side from the map, differing only in the popcount-of-XOR kernel.
///
/// [`AngularScanner`]: crate::search::dense::AngularScanner
pub struct HammingScanner<Value: nk::Hammings + nk::Hamming, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<Value, Allocator>,
    tile: Matrix<<Value as nk::Hamming>::Output>,
}

impl<Value: nk::Hammings + nk::Hamming, Allocator: AllocatorLike> HammingScanner<Value, Allocator> {
    /// A scanner over the workers `scope` names, its packed query in `alloc`,
    /// uniform with [`DotScanner::new`].
    ///
    /// [`DotScanner::new`]: crate::search::dense::DotScanner::new
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(HammingScanner {
            pool: open_pool(scope)?,
            packed: nk::DotsPackedMatrix::empty_in(alloc),
            tile: Matrix::empty(),
        })
    }
}

impl<Value, Allocator> HammingScanner<Value, Allocator>
where
    Allocator: AllocatorLike,
    Value: nk::Hamming + nk::Hammings + 'static,
    <Value as nk::Hamming>::Output: Copy + Default + PartialOrd + Widen,
    HammingScanner<Value, Allocator>: ScannerLike<HammingMetric<Value>>,
{
    /// Batch brute-force top-k, nearest-first — Hamming is a smaller-is-better
    /// distance. `matches`/`distances` carry `k` and the query count.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let mut tile = std::mem::take(&mut self.tile);
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

/// The traversal half of the Hamming metric.
pub struct HammingSearcher<Value: nk::Hamming, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, <Value as nk::Hamming>::Output, Allocator>,
}

impl<Value, Index, Allocator> HammingSearcher<Value, Index, Allocator>
where
    Value: nk::Hamming,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Hamming>::Output: Copy + Default,
{
    /// A searcher walking with breadth `expansion_factor`, its scratch in
    /// `alloc`, uniform with [`DotSearcher::new`].
    ///
    /// [`DotSearcher::new`]: crate::search::dense::DotSearcher::new
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(HammingSearcher {
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
/// `u32`. Smaller is nearer, so it ranks like a Euclidean distance. One generic
/// marker spans every bit type, as [`DotMetric`] does for the reals.
///
/// [`DotMetric`]: crate::search::dense::DotMetric
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
    Allocator: AllocatorLike,
    Value: nk::Hammings + nk::Hamming + Clone + Send + Sync + 'static,
    Metric: MetricLike<Score = u32> + ?Sized,
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
        let width = table.value_shape().ok_or(Error::Inconsistent)?;
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

impl<Metric, Value, Index, Allocator> SearcherLike<Metric>
    for HammingSearcher<Value, Index, Allocator>
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
            *slot = <Value as nk::Hamming>::hamming(query, value).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, Metric::Score, Allocator> {
        &mut self.context
    }

    fn compare(a: Metric::Score, b: Metric::Score) -> Ordering {
        smaller_first(a, b)
    }

    fn search(
        &mut self,
        query: Metric::Query<'_>,
        table: &Table<impl AllocatorLike>,
    ) -> Result<(), Error> {
        walk::<Metric, Self>(self, query, table)
    }
}

/// A Hamming searcher whose neighbor width is settled by the tables it searches,
/// the bit twin of [`AngularSearch`].
///
/// [`AngularSearch`]: crate::search::dense::AngularSearch
pub struct HammingSearch<Value: nk::Hamming, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers:
        Workers<HammingSearcher<Value, u32, Allocator>, HammingSearcher<Value, u64, Allocator>>,
    alloc: Allocator,
}

impl<Value, Allocator> HammingSearch<Value, Allocator>
where
    Value: nk::Hamming + 'static,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Hamming>::Output: Copy + Default + PartialOrd,
{
    /// A searcher walking with breadth `expansion_factor` across the workers
    /// `scope` names, its scratch in `alloc`, uniform with [`DotSearch::new`].
    ///
    /// [`DotSearch::new`]: crate::search::dense::DotSearch::new
    pub fn new(
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(HammingSearch {
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
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <Value as nk::Hamming>::Output: Widen,
        HammingSearcher<Value, u32, Allocator>:
            SearcherLike<HammingMetric<Value>, Index = u32> + Send,
        HammingSearcher<Value, u64, Allocator>:
            SearcherLike<HammingMetric<Value>, Index = u64> + Send,
    {
        let threads = self.pool.threads_count();
        let expansion_factor = self.expansion_factor;
        match table.neighbors_column().dtype {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(threads, || {
                        HammingSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<HammingMetric<Value>, _>(
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
                        HammingSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<HammingMetric<Value>, _>(
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

/// The Hamming twin of [`DotScannerAny`]: runtime dtype dispatch, `u1x8` the one arm
/// with a packed Hamming kernel. The typed scanner owns its tile, so unlike the dot
/// dispatcher this needs no scratch of its own.
///
/// [`DotScannerAny`]: crate::search::dense::DotScannerAny
pub enum HammingScannerAny<Allocator: AllocatorLike = Global> {
    U1x8(HammingScanner<nk::u1x8, Allocator>),
}

impl<Allocator: AllocatorLike + Clone> HammingScannerAny<Allocator> {
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => HammingScannerAny::U1x8(HammingScanner::new(scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            HammingScannerAny::U1x8(_) => "u1x8",
        }
    }

    /// Batch brute-force top-k over a query batch given as raw bytes in the table's
    /// dtype, cast to the packed bit scalar and forwarded to the typed scanner.
    pub fn scan(
        &mut self,
        queries: &[u8],
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            HammingScannerAny::U1x8(scanner) => scanner.scan(
                view_from_bytes(queries, table)?,
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
    U1x8(HammingSearch<nk::u1x8, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send> HammingSearcherAny<Allocator> {
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => HammingSearcherAny::U1x8(HammingSearch::new(expansion_factor, scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            HammingSearcherAny::U1x8(_) => "u1x8",
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
            HammingSearcherAny::U1x8(search) => search.search(
                view_from_bytes(queries, table)?,
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
    tile: Matrix<<Value as nk::Jaccard>::Output>,
}

impl<Value: nk::Jaccards + nk::Jaccard, Allocator: AllocatorLike> JaccardScanner<Value, Allocator> {
    /// A scanner over the workers `scope` names, its packed query in `alloc`,
    /// uniform with [`DotScanner::new`].
    ///
    /// [`DotScanner::new`]: crate::search::dense::DotScanner::new
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(JaccardScanner {
            pool: open_pool(scope)?,
            packed: nk::DotsPackedMatrix::empty_in(alloc),
            tile: Matrix::empty(),
        })
    }
}

impl<Value, Allocator> JaccardScanner<Value, Allocator>
where
    Allocator: AllocatorLike,
    Value: nk::Jaccard + nk::Jaccards + 'static,
    <Value as nk::Jaccard>::Output: Copy + Default + PartialOrd + Widen,
    JaccardScanner<Value, Allocator>: ScannerLike<JaccardMetric<Value>>,
{
    /// Batch brute-force top-k, nearest-first — Jaccard is a smaller-is-better
    /// distance. `matches`/`distances` carry `k` and the query count.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Value>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let mut tile = std::mem::take(&mut self.tile);
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

/// The traversal half of the Jaccard metric.
pub struct JaccardSearcher<Value: nk::Jaccard, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, <Value as nk::Jaccard>::Output, Allocator>,
}

impl<Value, Index, Allocator> JaccardSearcher<Value, Index, Allocator>
where
    Value: nk::Jaccard,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Jaccard>::Output: Copy + Default,
{
    /// A searcher walking with breadth `expansion_factor`, its scratch in
    /// `alloc`, uniform with [`DotSearcher::new`].
    ///
    /// [`DotSearcher::new`]: crate::search::dense::DotSearcher::new
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(JaccardSearcher {
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
    Allocator: AllocatorLike,
    Value: nk::Jaccards + nk::Jaccard + Clone + Send + Sync + 'static,
    <Value as nk::Jaccards>::JaccardResult: Send + Sync,
    Metric: MetricLike<Score = <Value as nk::Jaccards>::JaccardResult> + ?Sized,
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
        let width = table.value_shape().ok_or(Error::Inconsistent)?;
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

impl<Metric, Value, Index, Allocator> SearcherLike<Metric>
    for JaccardSearcher<Value, Index, Allocator>
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
            *slot = <Value as nk::Jaccard>::jaccard(query, value).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, Metric::Score, Allocator> {
        &mut self.context
    }

    fn compare(a: Metric::Score, b: Metric::Score) -> Ordering {
        smaller_first(a, b)
    }

    fn search(
        &mut self,
        query: Metric::Query<'_>,
        table: &Table<impl AllocatorLike>,
    ) -> Result<(), Error> {
        walk::<Metric, Self>(self, query, table)
    }
}

/// A Jaccard searcher whose neighbor width is settled by the tables it searches.
pub struct JaccardSearch<Value: nk::Jaccard, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers:
        Workers<JaccardSearcher<Value, u32, Allocator>, JaccardSearcher<Value, u64, Allocator>>,
    alloc: Allocator,
}

impl<Value, Allocator> JaccardSearch<Value, Allocator>
where
    Value: nk::Jaccard + 'static,
    Allocator: AllocatorLike + Clone,
    <Value as nk::Jaccard>::Output: Copy + Default + PartialOrd,
{
    /// A searcher walking with breadth `expansion_factor` across the workers
    /// `scope` names, its scratch in `alloc`, uniform with [`DotSearch::new`].
    ///
    /// [`DotSearch::new`]: crate::search::dense::DotSearch::new
    pub fn new(
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(JaccardSearch {
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
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <Value as nk::Jaccard>::Output: Widen,
        JaccardSearcher<Value, u32, Allocator>:
            SearcherLike<JaccardMetric<Value>, Index = u32> + Send,
        JaccardSearcher<Value, u64, Allocator>:
            SearcherLike<JaccardMetric<Value>, Index = u64> + Send,
    {
        let threads = self.pool.threads_count();
        let expansion_factor = self.expansion_factor;
        match table.neighbors_column().dtype {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(threads, || {
                        JaccardSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<JaccardMetric<Value>, _>(
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
                        JaccardSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<JaccardMetric<Value>, _>(
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

/// The Jaccard twin of [`HammingScannerAny`]: runtime dispatch, `u1x8` the one arm.
pub enum JaccardScannerAny<Allocator: AllocatorLike = Global> {
    U1x8(JaccardScanner<nk::u1x8, Allocator>),
}

impl<Allocator: AllocatorLike + Clone> JaccardScannerAny<Allocator> {
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => JaccardScannerAny::U1x8(JaccardScanner::new(scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            JaccardScannerAny::U1x8(_) => "u1x8",
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
            JaccardScannerAny::U1x8(scanner) => scanner.scan(
                view_from_bytes(queries, table)?,
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
    U1x8(JaccardSearch<nk::u1x8, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send> JaccardSearcherAny<Allocator> {
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => JaccardSearcherAny::U1x8(JaccardSearch::new(expansion_factor, scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            JaccardSearcherAny::U1x8(_) => "u1x8",
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
            JaccardSearcherAny::U1x8(search) => search.search(
                view_from_bytes(queries, table)?,
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
    use crate::{NeighborList, OffsetDType};

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
        fn value_shape(&self) -> Option<usize> {
            Some(self.values.shape()[1])
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
        let table = Table::build(&binary_ring(2), None).unwrap();
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
        assert_eq!(distances.row(0).unwrap(), &[1.0, 3.0, 6.0]);

        let mut searcher =
            HammingSearch::<nk::u1x8>::new(3, ExecutionScope::Serial, Global).unwrap();
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
        assert_eq!(walked_scores.row(0).unwrap(), &[1.0, 3.0, 6.0]);
    }

    #[test]
    fn jaccard_scan_and_search_agree() {
        // One minus intersection over union: node 0 is `1 - 7/8`, the others farther,
        // so the ranking is `[0,1,2]`, and scan and walk agree.
        let table = Table::build(&binary_ring(2), None).unwrap();
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
        assert_eq!(distances.row(0).unwrap()[0], 0.125);

        let mut searcher =
            JaccardSearch::<nk::u1x8>::new(3, ExecutionScope::Serial, Global).unwrap();
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
        assert_eq!(walked_scores.row(0).unwrap()[0], 0.125);
    }

    #[test]
    fn dispatch_scans_and_walks_bits() {
        // The erased runtime path: open a bit table, dispatch on its `u1x8` header,
        // and rank nearest-first through both a batch scan and a batch walk.
        let table = Table::build(&binary_ring(2), None).unwrap();
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

//! Dense-vector metrics over NumKong. No StringZilla type appears here.
//!
//! Each metric is a generic descriptor — [`DotScoring<T>`] over any scalar `T` —
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

use super::{
    walk, Matrix, QueryBatch, ScannerLike, Scores2D, ScoringLike, SearcherLike, TraversalContext,
};
use crate::{cast, Error, GraphLike, OffsetDType, OffsetScalar, Table, Widen};
use allocator_api2::alloc::{Allocator, Global};
use core::cmp::Ordering;
use forkunion as fu;
use numkong as nk;

/// A zero-copy `rows × width` view over caller-owned scalars, where `width` is a
/// count of logical elements. For a sub-byte scalar like `i4x2` or `u1x8` that is
/// wider than one storage value, so `rows` holds `width / dimensions_per_value`
/// storage elements per row and the row stride packs them tight — the same shape
/// arithmetic NumKong's own tensors use. `pub(crate)` so the sibling binary-metric
/// module builds its packed views the same way.
pub(crate) fn matrix_view<T>(rows: &[T], width: usize) -> Result<nk::MatrixView<'_, T>, Error>
where
    T: nk::StorageElement,
{
    let dimensions_per_value = T::dimensions_per_value();
    if width == 0 || !width.is_multiple_of(dimensions_per_value) {
        return Err(Error::Inconsistent);
    }
    let storage_per_row = width / dimensions_per_value;
    if storage_per_row == 0 || rows.is_empty() || !rows.len().is_multiple_of(storage_per_row) {
        return Err(Error::Inconsistent);
    }
    let count = rows.len() / storage_per_row;
    let size = core::mem::size_of::<T>() as isize;
    // SAFETY: `rows` is `count * storage_per_row` contiguous `T`. The logical shape
    // `[count, width]` with byte strides describes exactly that extent — the
    // innermost stride is one storage element, the row stride its packed width —
    // and the view borrows `rows` for `'_`.
    Ok(unsafe {
        nk::MatrixView::<T>::from_raw_parts(
            rows.as_ptr(),
            &[count, width],
            &[storage_per_row as isize * size, size],
        )
    })
}

/// The table's value column as a zero-copy `nodes × width` matrix, `width` in
/// logical elements. The byte buffer reinterprets as packed storage scalars, which
/// [`matrix_view`] then unpacks by shape.
pub(crate) fn value_view<T>(
    table: &Table<impl Allocator>,
    width: usize,
) -> Result<nk::MatrixView<'_, T>, Error>
where
    T: nk::StorageElement,
{
    let scalars: &[T] = cast(table.value_buffer()).ok_or(Error::WrongType)?;
    matrix_view(scalars, width)
}

/// The batch half of every dense metric: owns a thread pool, packs the query
/// side once per call, and streams the value side straight from the memory map.
pub struct DotScanner<T: nk::Dots> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<T>,
}

impl<T: nk::Dots> DotScanner<T> {
    /// A scanner driving `threads` workers, the caller included. `Topology` probes
    /// the machine and is neither cheap nor cloneable, so the driver builds one
    /// and lends it to every scanner it makes.
    pub fn new(topology: &fu::Topology, threads: usize) -> Result<Self, Error> {
        // NumKong's own workers claim AMX tile state themselves, but the calling
        // thread also participates in a dispatch, so it must claim it too — the
        // packed integer kernels trap without it.
        nk::capabilities::configure_thread();
        Ok(DotScanner {
            pool: fu::ThreadPool::try_spawn(topology, threads.max(1)).map_err(|_| Error::Alloc)?,
            packed: nk::DotsPackedMatrix::empty_in(nk::Global),
        })
    }
}

/// Rank column `query` of a node-major score tile top-k, writing node ids widened to
/// `u64` into `nodes_out` and scores widened to `f64` into `scores_out`, best-first;
/// `nearest` picks smaller-is-better. Unfilled slots take a `u64::MAX` sentinel.
/// Returns the count found. Widening is monotonic, so ranking the widened scores is
/// the native order. The shared tail of every batch scan, dense or binary.
pub(crate) fn select_top_k<S: Copy + Widen>(
    tile: &Scores2D<'_, S>,
    query: usize,
    node_count: usize,
    nearest: bool,
    nodes_out: &mut [u64],
    scores_out: &mut [f64],
) -> usize {
    let k = nodes_out.len();
    let mut filled = 0usize;
    for node in 0..node_count {
        let Some(score) = tile.get(node, query) else {
            continue;
        };
        let score = score.widen();
        if filled < k {
            insert_pair(nodes_out, scores_out, filled, node as u64, score, nearest);
            filled += 1;
        } else if outranks(score, scores_out[k - 1], nearest) {
            insert_pair(nodes_out, scores_out, k - 1, node as u64, score, nearest);
        }
    }
    for slot in nodes_out.iter_mut().skip(filled) {
        *slot = u64::MAX;
    }
    filled
}

/// Insert `(node, score)` into the sorted prefix `..=end` of two parallel arrays,
/// shifting weaker entries right so the prefix stays best-first.
fn insert_pair(
    nodes: &mut [u64],
    scores: &mut [f64],
    end: usize,
    node: u64,
    score: f64,
    nearest: bool,
) {
    let mut i = end;
    while i > 0 && outranks(score, scores[i - 1], nearest) {
        nodes[i] = nodes[i - 1];
        scores[i] = scores[i - 1];
        i -= 1;
    }
    nodes[i] = node;
    scores[i] = score;
}

/// Validate a batch's shapes against the query bytes and the table, returning the
/// logical query count, the node count, and the per-query storage stride. `matches`
/// and `distances` are `query_count × k`, so `k` and the count are read off them.
/// Shared by every metric's batch scan and search, dense or binary.
/// A dense query batch — a `nodes × width` [`nk::MatrixView`] — as a [`QueryBatch`]:
/// its row count is the query count, and each row is one query slice. This is what
/// lets one generic [`batch_search`] walk a dense batch or a sequence tape alike.
impl<M, T> QueryBatch<M> for nk::MatrixView<'_, T>
where
    T: nk::StorageElement + Clone,
    M: ScoringLike + ?Sized,
    for<'a> M::Query<'a>: From<&'a [T]>,
{
    fn count(&self) -> usize {
        self.shape().first().copied().unwrap_or(0)
    }

    fn query(&self, index: usize) -> Option<M::Query<'_>> {
        let storage_per_row = self.shape().get(1)? / T::dimensions_per_value();
        let storage = self.as_packed_slice()?;
        let start = index.checked_mul(storage_per_row)?;
        let row = storage.get(start..start.checked_add(storage_per_row)?)?;
        Some(row.into())
    }
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

/// Batch brute-force top-k for any dense scanner: score every (query, node) into a
/// scratch tile, then rank each query's column into `matches`/`distances`, best-first
/// — `nearest` for a distance — reporting the count found per query. The metric-typed
/// `scan` methods are one-line delegations to this.
pub(crate) fn batch_scan<M, Sc>(
    scanner: &mut Sc,
    queries: M::Queries<'_>,
    table: &Table<impl Allocator>,
    nearest: bool,
    matches: &mut Matrix<u64>,
    distances: &mut Matrix<f64>,
    counts: &mut [u64],
) -> Result<(), Error>
where
    M: ScoringLike,
    M::Score: Widen,
    for<'a> M::Queries<'a>: QueryBatch<M>,
    Sc: ScannerLike<M>,
{
    let query_count = queries.count();
    let nodes = table.len();
    batch_shape(query_count, matches, distances, counts)?;
    let cells = nodes.checked_mul(query_count).ok_or(Error::TooLarge)?;
    let mut tile: Vec<M::Score> = Vec::new();
    tile.try_reserve_exact(cells).map_err(|_| Error::Alloc)?;
    tile.resize(cells, Default::default());
    let mut scores = Scores2D::new(&mut tile, nodes, query_count)?;
    <Sc as ScannerLike<M>>::scan(scanner, queries, table, &mut scores)?;
    for (query, count) in counts.iter_mut().enumerate() {
        *count = select_top_k(
            &scores,
            query,
            nodes,
            nearest,
            matches.row_mut(query).ok_or(Error::Inconsistent)?,
            distances.row_mut(query).ok_or(Error::Inconsistent)?,
        ) as u64;
    }
    Ok(())
}

/// Batch top-k by graph walk for any searcher, one query at a time — the pool
/// fan-out over queries is the next step. The searcher's own comparator drains each
/// walk best-first into `matches`/`distances`, widening node to `u64` and score to
/// `f64`; `counts` is the number found, `< k` when a walk exhausts a small component.
pub(crate) fn batch_search<M, Se>(
    searcher: &mut Se,
    queries: M::Queries<'_>,
    table: &Table<impl Allocator>,
    matches: &mut Matrix<u64>,
    distances: &mut Matrix<f64>,
    counts: &mut [u64],
) -> Result<(), Error>
where
    M: ScoringLike,
    M::Score: Widen,
    for<'a> M::Queries<'a>: QueryBatch<M>,
    Se: SearcherLike<M>,
{
    let query_count = queries.count();
    let k = batch_shape(query_count, matches, distances, counts)?;
    for (query, count) in counts.iter_mut().enumerate() {
        let one = queries.query(query).ok_or(Error::Inconsistent)?;
        <Se as SearcherLike<M>>::search(searcher, one, table)?;
        let nodes_out = matches.row_mut(query).ok_or(Error::Inconsistent)?;
        let scores_out = distances.row_mut(query).ok_or(Error::Inconsistent)?;
        let found = <Se as SearcherLike<M>>::context(searcher).drain(
            <Se as SearcherLike<M>>::compare,
            k,
            |slot, node, score| {
                nodes_out[slot] = node.into();
                scores_out[slot] = score.widen();
            },
        );
        for slot in nodes_out.iter_mut().skip(found) {
            *slot = u64::MAX;
        }
        *count = found as u64;
    }
    Ok(())
}

impl<T> DotScanner<T>
where
    T: nk::Dot + nk::Dots + 'static,
    <T as nk::Dot>::Output: Copy + Default + PartialOrd + Widen,
    DotScanner<T>: ScannerLike<DotScoring<T>>,
{
    /// Batch brute-force top-k. `matches`/`distances` are `queries × k`; their shape
    /// carries both `k` and the query count. Larger dot ranks first.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        batch_scan::<DotScoring<T>, _>(self, queries, table, false, matches, distances, counts)
    }
}

/// The traversal half of every dense metric: owns the reusable scratch and scores
/// a node set one vector at a time, which is what a handful of neighbors wants.
/// The index type `I` matches the table's neighbor width, so the frontier stays
/// compact; `DotSearch` resolves it, but a caller who knows their table can name
/// it directly, `I` defaulting to `u32`.
pub struct DotSearcher<T: nk::Dot, I = u32, A: Allocator = Global> {
    context: TraversalContext<I, <T as nk::Dot>::Output, A>,
}

impl<T, I> DotSearcher<T, I, Global>
where
    T: nk::Dot,
    I: OffsetScalar,
    <T as nk::Dot>::Output: Copy + Default,
{
    /// A searcher for a graph of `nodes` nodes, walking with breadth
    /// `expansion_factor`.
    pub fn new(nodes: usize, expansion_factor: usize) -> Result<Self, Error> {
        DotSearcher::new_in(nodes, expansion_factor, Global)
    }
}

impl<T, I, A> DotSearcher<T, I, A>
where
    T: nk::Dot,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Dot>::Output: Copy + Default,
{
    pub fn new_in(nodes: usize, expansion_factor: usize, alloc: A) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(DotSearcher {
            context: TraversalContext::new_in(nodes, expansion_factor, alloc)?,
        })
    }

    /// The traversal scratch, for draining results after a walk. Inherent so a
    /// caller need not name the metric to reach it, unlike the trait method it
    /// shadows.
    pub fn context(&mut self) -> &mut TraversalContext<I, <T as nk::Dot>::Output, A> {
        &mut self.context
    }
}

impl<T, I, A> DotSearcher<T, I, A>
where
    T: nk::Dot + 'static,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Dot>::Output: Copy + Default + PartialOrd + Widen,
    DotSearcher<T, I, A>: SearcherLike<DotScoring<T>, Index = I>,
{
    /// Batch top-k by graph walk. Uniform with [`DotScanner::scan`]; `matches`/
    /// `distances` carry `k` and the query count, `counts` the number found per query.
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        batch_search::<DotScoring<T>, _>(self, queries, table, matches, distances, counts)
    }
}

/// Higher score ranks first, and a NaN sinks rather than poisoning the order.
fn larger_first<S: PartialOrd>(a: S, b: S) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Less)
}

/// Inner-product similarity over any NumKong scalar, accumulated in that scalar's
/// native dot output — `i32` for `i8`, `u32` for `u8`, `f64` for `f32`, `f32` for
/// every half and minifloat. One generic marker replaces a struct per dtype: the
/// scalar is the only thing that ever varied, so `DotScoring<i8>` and
/// `DotScoring<nk::e4m3>` are the whole vocabulary, and a new dtype adds no code.
pub struct DotScoring<T>(core::marker::PhantomData<T>);

impl<T: nk::Dot + 'static> ScoringLike for DotScoring<T>
where
    <T as nk::Dot>::Output: Copy + Default,
{
    type Query<'a> = &'a [T];
    type Queries<'a> = nk::MatrixView<'a, T>;
    type Score = <T as nk::Dot>::Output;
}

impl<M, T> ScannerLike<M> for DotScanner<T>
where
    T: nk::Dots + nk::Dot<Output = <T as nk::Dots>::Accumulator> + Clone + Send + Sync + 'static,
    <T as nk::Dots>::Accumulator: Send + Sync,
    M: ScoringLike<Score = <T as nk::Dots>::Accumulator> + ?Sized,
    for<'a> M::Queries<'a>: Into<nk::MatrixView<'a, T>>,
{
    fn scan(
        &mut self,
        queries: M::Queries<'_>,
        table: &Table<impl Allocator>,
        scores: &mut Scores2D<'_, M::Score>,
    ) -> Result<(), Error> {
        // The batch is already a shaped `nodes-agnostic` matrix; its width must match
        // the table's value column, and its row count the tile's query count.
        let width = table.value_shape().ok_or(Error::Inconsistent)?;
        let query_view: nk::MatrixView<T> = queries.into();
        let values = value_view::<T>(table, width)?;

        let (query_count, node_count) = (query_view.shape()[0], values.shape()[0]);
        if query_view.shape()[1] != width
            || query_count != scores.queries()
            || node_count != scores.nodes()
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
        let element = core::mem::size_of::<M::Score>() as isize;
        let mut tile = unsafe {
            nk::MatrixSpan::<M::Score>::from_raw_parts(
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

impl<M, T, I, A> SearcherLike<M> for DotSearcher<T, I, A>
where
    T: nk::Dot + 'static,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Dot>::Output: Copy + Default + PartialOrd,
    M: ScoringLike<Score = <T as nk::Dot>::Output> + ?Sized,
    for<'a> M::Query<'a>: Into<&'a [T]>,
{
    type Index = I;
    type Alloc = A;

    fn score_nodes(
        &mut self,
        query: M::Query<'_>,
        table: &Table<impl Allocator>,
        nodes: &[I],
        out: &mut [M::Score],
    ) -> Result<(), Error> {
        if nodes.len() != out.len() {
            return Err(Error::Inconsistent);
        }
        let query: &[T] = query.into();
        for (slot, &node) in out.iter_mut().zip(nodes) {
            let index: usize = node.try_into().map_err(|_| Error::TooLarge)?;
            let value: &[T] = cast(table.value(index)).ok_or(Error::WrongType)?;
            *slot = <T as nk::Dot>::dot(query, value).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<I, M::Score, A> {
        &mut self.context
    }

    fn compare(a: M::Score, b: M::Score) -> Ordering {
        larger_first(a, b)
    }

    fn search(&mut self, query: M::Query<'_>, table: &Table<impl Allocator>) -> Result<(), Error> {
        walk::<M, Self>(self, query, table)
    }
}

/// A dot searcher whose neighbor width is resolved from the table, not named by
/// the caller — the two arms hold the `u32` and `u64` leaves. A dispatcher over
/// value dtypes builds one of these per dtype and never branches on width itself.
pub enum DotSearch<T: nk::Dot, A: Allocator = Global> {
    U32(DotSearcher<T, u32, A>),
    U64(DotSearcher<T, u64, A>),
}

impl<T, A> DotSearch<T, A>
where
    T: nk::Dot + 'static,
    A: Allocator + Clone,
    <T as nk::Dot>::Output: Copy + Default + PartialOrd,
{
    /// Read the neighbor width from `table` and build the matching leaf.
    pub fn for_table(
        table: &Table<impl Allocator>,
        expansion_factor: usize,
        alloc: A,
    ) -> Result<Self, Error> {
        Ok(match table.neighbors_column().dtype {
            OffsetDType::U32 => {
                DotSearch::U32(DotSearcher::new_in(table.len(), expansion_factor, alloc)?)
            }
            OffsetDType::U64 => {
                DotSearch::U64(DotSearcher::new_in(table.len(), expansion_factor, alloc)?)
            }
        })
    }

    /// Batch top-k by walk, forwarding to the width-resolved leaf. Uniform with
    /// [`DotScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <T as nk::Dot>::Output: Widen,
        DotSearcher<T, u32, A>: SearcherLike<DotScoring<T>, Index = u32>,
        DotSearcher<T, u64, A>: SearcherLike<DotScoring<T>, Index = u64>,
    {
        match self {
            DotSearch::U32(leaf) => leaf.search(queries, table, matches, distances, counts),
            DotSearch::U64(leaf) => leaf.search(queries, table, matches, distances, counts),
        }
    }
}

/// Smaller distance ranks first; a NaN sinks rather than poisoning the order.
fn smaller_first<S: PartialOrd>(a: S, b: S) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Less)
}

/// The batch half of every Euclidean metric. Same shape as [`DotScanner`] — a
/// pool and a reused packed query — because `Euclideans` packs through the very
/// same [`nk::DotsPackedMatrix`]; only the kernel it calls differs.
pub struct EuclideanScanner<T: nk::Dots> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<T>,
}

impl<T: nk::Dots> EuclideanScanner<T> {
    pub fn new(topology: &fu::Topology, threads: usize) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(EuclideanScanner {
            pool: fu::ThreadPool::try_spawn(topology, threads.max(1)).map_err(|_| Error::Alloc)?,
            packed: nk::DotsPackedMatrix::empty_in(nk::Global),
        })
    }
}

impl<T> EuclideanScanner<T>
where
    T: nk::Euclidean + nk::Euclideans + 'static,
    <T as nk::Euclidean>::EuclideanOutput: Copy + Default + PartialOrd + Widen,
    EuclideanScanner<T>: ScannerLike<EuclideanScoring<T>>,
{
    /// Batch brute-force top-k, nearest-first — L2 is smaller-is-better.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        batch_scan::<EuclideanScoring<T>, _>(self, queries, table, true, matches, distances, counts)
    }
}

/// The traversal half of every Euclidean metric.
pub struct EuclideanSearcher<T: nk::Euclidean, I = u32, A: Allocator = Global> {
    context: TraversalContext<I, <T as nk::Euclidean>::EuclideanOutput, A>,
}

impl<T, I> EuclideanSearcher<T, I, Global>
where
    T: nk::Euclidean,
    I: OffsetScalar,
    <T as nk::Euclidean>::EuclideanOutput: Copy + Default,
{
    pub fn new(nodes: usize, expansion_factor: usize) -> Result<Self, Error> {
        EuclideanSearcher::new_in(nodes, expansion_factor, Global)
    }
}

impl<T, I, A> EuclideanSearcher<T, I, A>
where
    T: nk::Euclidean,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Euclidean>::EuclideanOutput: Copy + Default,
{
    pub fn new_in(nodes: usize, expansion_factor: usize, alloc: A) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(EuclideanSearcher {
            context: TraversalContext::new_in(nodes, expansion_factor, alloc)?,
        })
    }

    pub fn context(
        &mut self,
    ) -> &mut TraversalContext<I, <T as nk::Euclidean>::EuclideanOutput, A> {
        &mut self.context
    }
}

impl<T, I, A> EuclideanSearcher<T, I, A>
where
    T: nk::Euclidean + 'static,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Euclidean>::EuclideanOutput: Copy + Default + PartialOrd + Widen,
    EuclideanSearcher<T, I, A>: SearcherLike<EuclideanScoring<T>, Index = I>,
{
    /// Batch top-k by graph walk, uniform with [`EuclideanScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        batch_search::<EuclideanScoring<T>, _>(self, queries, table, matches, distances, counts)
    }
}

/// L2 distance over any NumKong scalar, in its native Euclidean output. Smaller is
/// nearer, so it ranks the reverse of a dot — the same contract, only the
/// comparator flips. Like [`DotScoring`], one generic marker spans every dtype.
pub struct EuclideanScoring<T>(core::marker::PhantomData<T>);

impl<T: nk::Euclidean + 'static> ScoringLike for EuclideanScoring<T>
where
    <T as nk::Euclidean>::EuclideanOutput: Copy + Default,
{
    type Query<'a> = &'a [T];
    type Queries<'a> = nk::MatrixView<'a, T>;
    type Score = <T as nk::Euclidean>::EuclideanOutput;
}

impl<M, T> ScannerLike<M> for EuclideanScanner<T>
where
    T: nk::Euclideans + Clone + Send + Sync + 'static,
    <T as nk::Euclideans>::SpatialResult: Send + Sync,
    M: ScoringLike<Score = <T as nk::Euclideans>::SpatialResult> + ?Sized,
    for<'a> M::Queries<'a>: Into<nk::MatrixView<'a, T>>,
{
    fn scan(
        &mut self,
        queries: M::Queries<'_>,
        table: &Table<impl Allocator>,
        scores: &mut Scores2D<'_, M::Score>,
    ) -> Result<(), Error> {
        // The batch is already a shaped `nodes-agnostic` matrix; its width must match
        // the table's value column, and its row count the tile's query count.
        let width = table.value_shape().ok_or(Error::Inconsistent)?;
        let query_view: nk::MatrixView<T> = queries.into();
        let values = value_view::<T>(table, width)?;

        let (query_count, node_count) = (query_view.shape()[0], values.shape()[0]);
        if query_view.shape()[1] != width
            || query_count != scores.queries()
            || node_count != scores.nodes()
        {
            return Err(Error::Inconsistent);
        }

        self.packed
            .try_pack_parallel_into(&query_view, &mut self.pool)
            .map_err(|_| Error::Inconsistent)?;

        // Node-major, contiguous rows — exactly as the dot scan, since the output
        // shape is the same `nodes × queries`; only the kernel differs.
        let element = core::mem::size_of::<M::Score>() as isize;
        let mut tile = unsafe {
            nk::MatrixSpan::<M::Score>::from_raw_parts(
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

impl<M, T, I, A> SearcherLike<M> for EuclideanSearcher<T, I, A>
where
    T: nk::Euclidean + 'static,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Euclidean>::EuclideanOutput: Copy + Default + PartialOrd,
    M: ScoringLike<Score = <T as nk::Euclidean>::EuclideanOutput> + ?Sized,
    for<'a> M::Query<'a>: Into<&'a [T]>,
{
    type Index = I;
    type Alloc = A;

    fn score_nodes(
        &mut self,
        query: M::Query<'_>,
        table: &Table<impl Allocator>,
        nodes: &[I],
        out: &mut [M::Score],
    ) -> Result<(), Error> {
        if nodes.len() != out.len() {
            return Err(Error::Inconsistent);
        }
        let query: &[T] = query.into();
        for (slot, &node) in out.iter_mut().zip(nodes) {
            let index: usize = node.try_into().map_err(|_| Error::TooLarge)?;
            let value: &[T] = cast(table.value(index)).ok_or(Error::WrongType)?;
            // The true distance, not the squared one, so its type and value match
            // the batch `Euclideans` kernel the scan uses — scan and walk agree.
            *slot = <T as nk::Euclidean>::euclidean(query, value).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<I, M::Score, A> {
        &mut self.context
    }

    fn compare(a: M::Score, b: M::Score) -> Ordering {
        smaller_first(a, b)
    }

    fn search(&mut self, query: M::Query<'_>, table: &Table<impl Allocator>) -> Result<(), Error> {
        walk::<M, Self>(self, query, table)
    }
}

/// A Euclidean searcher whose neighbor width is resolved from the table, the
/// distance twin of [`DotSearch`].
pub enum EuclideanSearch<T: nk::Euclidean, A: Allocator = Global> {
    U32(EuclideanSearcher<T, u32, A>),
    U64(EuclideanSearcher<T, u64, A>),
}

impl<T, A> EuclideanSearch<T, A>
where
    T: nk::Euclidean + 'static,
    A: Allocator + Clone,
    <T as nk::Euclidean>::EuclideanOutput: Copy + Default + PartialOrd,
{
    pub fn for_table(
        table: &Table<impl Allocator>,
        expansion_factor: usize,
        alloc: A,
    ) -> Result<Self, Error> {
        Ok(match table.neighbors_column().dtype {
            OffsetDType::U32 => EuclideanSearch::U32(EuclideanSearcher::new_in(
                table.len(),
                expansion_factor,
                alloc,
            )?),
            OffsetDType::U64 => EuclideanSearch::U64(EuclideanSearcher::new_in(
                table.len(),
                expansion_factor,
                alloc,
            )?),
        })
    }

    /// Batch top-k by walk, forwarding to the width-resolved leaf. Uniform with
    /// [`EuclideanScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <T as nk::Euclidean>::EuclideanOutput: Widen,
        EuclideanSearcher<T, u32, A>: SearcherLike<EuclideanScoring<T>, Index = u32>,
        EuclideanSearcher<T, u64, A>: SearcherLike<EuclideanScoring<T>, Index = u64>,
    {
        match self {
            EuclideanSearch::U32(leaf) => leaf.search(queries, table, matches, distances, counts),
            EuclideanSearch::U64(leaf) => leaf.search(queries, table, matches, distances, counts),
        }
    }
}

/// The batch half of every angular metric. Structurally identical to
/// [`EuclideanScanner`] — both pack through [`nk::DotsPackedMatrix`] and are a
/// smaller-is-better distance — differing only in the kernel it calls.
pub struct AngularScanner<T: nk::Dots> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<T>,
}

impl<T: nk::Dots> AngularScanner<T> {
    pub fn new(topology: &fu::Topology, threads: usize) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(AngularScanner {
            pool: fu::ThreadPool::try_spawn(topology, threads.max(1)).map_err(|_| Error::Alloc)?,
            packed: nk::DotsPackedMatrix::empty_in(nk::Global),
        })
    }
}

impl<T> AngularScanner<T>
where
    T: nk::Angular + nk::Angulars + 'static,
    <T as nk::Angular>::Output: Copy + Default + PartialOrd + Widen,
    AngularScanner<T>: ScannerLike<AngularScoring<T>>,
{
    /// Batch brute-force top-k, nearest-first — cosine distance is smaller-is-better.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        batch_scan::<AngularScoring<T>, _>(self, queries, table, true, matches, distances, counts)
    }
}

/// The traversal half of every angular metric.
pub struct AngularSearcher<T: nk::Angular, I = u32, A: Allocator = Global> {
    context: TraversalContext<I, <T as nk::Angular>::Output, A>,
}

impl<T, I> AngularSearcher<T, I, Global>
where
    T: nk::Angular,
    I: OffsetScalar,
    <T as nk::Angular>::Output: Copy + Default,
{
    pub fn new(nodes: usize, expansion_factor: usize) -> Result<Self, Error> {
        AngularSearcher::new_in(nodes, expansion_factor, Global)
    }
}

impl<T, I, A> AngularSearcher<T, I, A>
where
    T: nk::Angular,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Angular>::Output: Copy + Default,
{
    pub fn new_in(nodes: usize, expansion_factor: usize, alloc: A) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(AngularSearcher {
            context: TraversalContext::new_in(nodes, expansion_factor, alloc)?,
        })
    }

    pub fn context(&mut self) -> &mut TraversalContext<I, <T as nk::Angular>::Output, A> {
        &mut self.context
    }
}

impl<T, I, A> AngularSearcher<T, I, A>
where
    T: nk::Angular + 'static,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Angular>::Output: Copy + Default + PartialOrd + Widen,
    AngularSearcher<T, I, A>: SearcherLike<AngularScoring<T>, Index = I>,
{
    /// Batch top-k by graph walk, uniform with [`AngularScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        batch_search::<AngularScoring<T>, _>(self, queries, table, matches, distances, counts)
    }
}

/// Angular (cosine) distance over any NumKong scalar, in its native angular
/// output. Smaller is nearer, so it ranks like a Euclidean distance, only the
/// kernel differs. One generic marker spans every dtype, as [`DotScoring`] does.
pub struct AngularScoring<T>(core::marker::PhantomData<T>);

impl<T: nk::Angular + 'static> ScoringLike for AngularScoring<T>
where
    <T as nk::Angular>::Output: Copy + Default,
{
    type Query<'a> = &'a [T];
    type Queries<'a> = nk::MatrixView<'a, T>;
    type Score = <T as nk::Angular>::Output;
}

impl<M, T> ScannerLike<M> for AngularScanner<T>
where
    T: nk::Angulars + Clone + Send + Sync + 'static,
    <T as nk::Angulars>::SpatialResult: Send + Sync,
    M: ScoringLike<Score = <T as nk::Angulars>::SpatialResult> + ?Sized,
    for<'a> M::Queries<'a>: Into<nk::MatrixView<'a, T>>,
{
    fn scan(
        &mut self,
        queries: M::Queries<'_>,
        table: &Table<impl Allocator>,
        scores: &mut Scores2D<'_, M::Score>,
    ) -> Result<(), Error> {
        // The batch is already a shaped `nodes-agnostic` matrix; its width must match
        // the table's value column, and its row count the tile's query count.
        let width = table.value_shape().ok_or(Error::Inconsistent)?;
        let query_view: nk::MatrixView<T> = queries.into();
        let values = value_view::<T>(table, width)?;

        let (query_count, node_count) = (query_view.shape()[0], values.shape()[0]);
        if query_view.shape()[1] != width
            || query_count != scores.queries()
            || node_count != scores.nodes()
        {
            return Err(Error::Inconsistent);
        }

        self.packed
            .try_pack_parallel_into(&query_view, &mut self.pool)
            .map_err(|_| Error::Inconsistent)?;

        // Node-major, contiguous rows, as every dense scan; only the kernel differs.
        let element = core::mem::size_of::<M::Score>() as isize;
        let mut tile = unsafe {
            nk::MatrixSpan::<M::Score>::from_raw_parts(
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

impl<M, T, I, A> SearcherLike<M> for AngularSearcher<T, I, A>
where
    T: nk::Angular + 'static,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Angular>::Output: Copy + Default + PartialOrd,
    M: ScoringLike<Score = <T as nk::Angular>::Output> + ?Sized,
    for<'a> M::Query<'a>: Into<&'a [T]>,
{
    type Index = I;
    type Alloc = A;

    fn score_nodes(
        &mut self,
        query: M::Query<'_>,
        table: &Table<impl Allocator>,
        nodes: &[I],
        out: &mut [M::Score],
    ) -> Result<(), Error> {
        if nodes.len() != out.len() {
            return Err(Error::Inconsistent);
        }
        let query: &[T] = query.into();
        for (slot, &node) in out.iter_mut().zip(nodes) {
            let index: usize = node.try_into().map_err(|_| Error::TooLarge)?;
            let value: &[T] = cast(table.value(index)).ok_or(Error::WrongType)?;
            *slot = <T as nk::Angular>::angular(query, value).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<I, M::Score, A> {
        &mut self.context
    }

    fn compare(a: M::Score, b: M::Score) -> Ordering {
        smaller_first(a, b)
    }

    fn search(&mut self, query: M::Query<'_>, table: &Table<impl Allocator>) -> Result<(), Error> {
        walk::<M, Self>(self, query, table)
    }
}

/// An angular searcher whose neighbor width is resolved from the table, the
/// cosine twin of [`EuclideanSearch`].
pub enum AngularSearch<T: nk::Angular, A: Allocator = Global> {
    U32(AngularSearcher<T, u32, A>),
    U64(AngularSearcher<T, u64, A>),
}

impl<T, A> AngularSearch<T, A>
where
    T: nk::Angular + 'static,
    A: Allocator + Clone,
    <T as nk::Angular>::Output: Copy + Default + PartialOrd,
{
    pub fn for_table(
        table: &Table<impl Allocator>,
        expansion_factor: usize,
        alloc: A,
    ) -> Result<Self, Error> {
        Ok(match table.neighbors_column().dtype {
            OffsetDType::U32 => AngularSearch::U32(AngularSearcher::new_in(
                table.len(),
                expansion_factor,
                alloc,
            )?),
            OffsetDType::U64 => AngularSearch::U64(AngularSearcher::new_in(
                table.len(),
                expansion_factor,
                alloc,
            )?),
        })
    }

    /// Batch top-k by walk, forwarding to the width-resolved leaf. Uniform with
    /// [`AngularScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <T as nk::Angular>::Output: Widen,
        AngularSearcher<T, u32, A>: SearcherLike<AngularScoring<T>, Index = u32>,
        AngularSearcher<T, u64, A>: SearcherLike<AngularScoring<T>, Index = u64>,
    {
        match self {
            AngularSearch::U32(leaf) => leaf.search(queries, table, matches, distances, counts),
            AngularSearch::U64(leaf) => leaf.search(queries, table, matches, distances, counts),
        }
    }
}

/// `a` outranks `b` when it is larger, or smaller for a `nearest` distance metric.
fn outranks(a: f64, b: f64, nearest: bool) -> bool {
    if nearest {
        a < b
    } else {
        a > b
    }
}

/// Reinterpret a query byte batch as a `nodes-agnostic` [`nk::MatrixView`] of the
/// table's dtype `T`, `width` from the value column — the shaped batch each erased
/// `*Any` hands to its typed scanner or searcher.
pub(crate) fn view_from_bytes<'q, T: nk::StorageElement>(
    queries: &'q [u8],
    table: &Table<impl Allocator>,
) -> Result<nk::MatrixView<'q, T>, Error> {
    let width = table.value_shape().ok_or(Error::Inconsistent)?;
    matrix_view(cast(queries).ok_or(Error::WrongType)?, width)
}

/// A dot scanner chosen at runtime from a file's value dtype string — the
/// within-family dispatch, so the choice of scalar stays a dense concern rather than
/// leaking into the cross-metric router. Each arm is a whole [`DotScanner`], fully
/// typed; only the result crosses out, node widened to `u64` and score to `f64`. The
/// typed scanner owns its own tile, so the dispatcher holds no scratch. Arms are
/// ordered by dynamic range, widest float to tightest int.
pub enum DotScannerAny {
    F64(DotScanner<f64>),
    F32(DotScanner<f32>),
    Bf16(DotScanner<nk::bf16>),
    F16(DotScanner<nk::f16>),
    E5M2(DotScanner<nk::e5m2>),
    E4M3(DotScanner<nk::e4m3>),
    E3M2(DotScanner<nk::e3m2>),
    E2M3(DotScanner<nk::e2m3>),
    I8(DotScanner<i8>),
    I4x2(DotScanner<nk::i4x2>),
    U8(DotScanner<u8>),
    U4x2(DotScanner<nk::u4x2>),
    U1x8(DotScanner<nk::u1x8>),
}

impl DotScannerAny {
    /// A scanner for `dtype`, driving `threads` workers. [`Error::UnknownDType`] if
    /// no arm covers it — including a dtype the format understands but this metric
    /// has not been instantiated for.
    pub fn for_dtype(dtype: &str, topology: &fu::Topology, threads: usize) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => DotScannerAny::F64(DotScanner::new(topology, threads)?),
            "f32" => DotScannerAny::F32(DotScanner::new(topology, threads)?),
            "bf16" => DotScannerAny::Bf16(DotScanner::new(topology, threads)?),
            "f16" => DotScannerAny::F16(DotScanner::new(topology, threads)?),
            "e5m2" => DotScannerAny::E5M2(DotScanner::new(topology, threads)?),
            "e4m3" => DotScannerAny::E4M3(DotScanner::new(topology, threads)?),
            "e3m2" => DotScannerAny::E3M2(DotScanner::new(topology, threads)?),
            "e2m3" => DotScannerAny::E2M3(DotScanner::new(topology, threads)?),
            "i8" => DotScannerAny::I8(DotScanner::new(topology, threads)?),
            "i4x2" => DotScannerAny::I4x2(DotScanner::new(topology, threads)?),
            "u8" => DotScannerAny::U8(DotScanner::new(topology, threads)?),
            "u4x2" => DotScannerAny::U4x2(DotScanner::new(topology, threads)?),
            "u1x8" => DotScannerAny::U1x8(DotScanner::new(topology, threads)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    /// A scanner matched to an open table — its value dtype and thread pool.
    pub fn for_table(
        table: &Table<impl Allocator>,
        topology: &fu::Topology,
        threads: usize,
    ) -> Result<Self, Error> {
        Self::for_dtype(table.value_dtype(), topology, threads)
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
        table: &Table<impl Allocator>,
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
pub enum DotSearcherAny {
    F64(DotSearch<f64>),
    F32(DotSearch<f32>),
    Bf16(DotSearch<nk::bf16>),
    F16(DotSearch<nk::f16>),
    E5M2(DotSearch<nk::e5m2>),
    E4M3(DotSearch<nk::e4m3>),
    E3M2(DotSearch<nk::e3m2>),
    E2M3(DotSearch<nk::e2m3>),
    I8(DotSearch<i8>),
    I4x2(DotSearch<nk::i4x2>),
    U8(DotSearch<u8>),
    U4x2(DotSearch<nk::u4x2>),
    U1x8(DotSearch<nk::u1x8>),
}

impl DotSearcherAny {
    /// A searcher for `table`'s value dtype, walking with breadth
    /// `expansion_factor`. The neighbor width is read from the table inside
    /// [`DotSearch::for_table`], not branched on here.
    pub fn for_dtype(
        dtype: &str,
        table: &Table<impl Allocator>,
        expansion_factor: usize,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => DotSearcherAny::F64(DotSearch::for_table(table, expansion_factor, Global)?),
            "f32" => DotSearcherAny::F32(DotSearch::for_table(table, expansion_factor, Global)?),
            "bf16" => DotSearcherAny::Bf16(DotSearch::for_table(table, expansion_factor, Global)?),
            "f16" => DotSearcherAny::F16(DotSearch::for_table(table, expansion_factor, Global)?),
            "e5m2" => DotSearcherAny::E5M2(DotSearch::for_table(table, expansion_factor, Global)?),
            "e4m3" => DotSearcherAny::E4M3(DotSearch::for_table(table, expansion_factor, Global)?),
            "e3m2" => DotSearcherAny::E3M2(DotSearch::for_table(table, expansion_factor, Global)?),
            "e2m3" => DotSearcherAny::E2M3(DotSearch::for_table(table, expansion_factor, Global)?),
            "i8" => DotSearcherAny::I8(DotSearch::for_table(table, expansion_factor, Global)?),
            "i4x2" => DotSearcherAny::I4x2(DotSearch::for_table(table, expansion_factor, Global)?),
            "u8" => DotSearcherAny::U8(DotSearch::for_table(table, expansion_factor, Global)?),
            "u4x2" => DotSearcherAny::U4x2(DotSearch::for_table(table, expansion_factor, Global)?),
            "u1x8" => DotSearcherAny::U1x8(DotSearch::for_table(table, expansion_factor, Global)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    /// A searcher matched to an open table — its value dtype and neighbor width.
    pub fn for_table(
        table: &Table<impl Allocator>,
        expansion_factor: usize,
    ) -> Result<Self, Error> {
        Self::for_dtype(table.value_dtype(), table, expansion_factor)
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
        table: &Table<impl Allocator>,
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
pub enum EuclideanScannerAny {
    F64(EuclideanScanner<f64>),
    F32(EuclideanScanner<f32>),
    Bf16(EuclideanScanner<nk::bf16>),
    F16(EuclideanScanner<nk::f16>),
    E5M2(EuclideanScanner<nk::e5m2>),
    E4M3(EuclideanScanner<nk::e4m3>),
    E3M2(EuclideanScanner<nk::e3m2>),
    E2M3(EuclideanScanner<nk::e2m3>),
    I8(EuclideanScanner<i8>),
    I4x2(EuclideanScanner<nk::i4x2>),
    U8(EuclideanScanner<u8>),
    U4x2(EuclideanScanner<nk::u4x2>),
}

impl EuclideanScannerAny {
    pub fn for_dtype(dtype: &str, topology: &fu::Topology, threads: usize) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => EuclideanScannerAny::F64(EuclideanScanner::new(topology, threads)?),
            "f32" => EuclideanScannerAny::F32(EuclideanScanner::new(topology, threads)?),
            "bf16" => EuclideanScannerAny::Bf16(EuclideanScanner::new(topology, threads)?),
            "f16" => EuclideanScannerAny::F16(EuclideanScanner::new(topology, threads)?),
            "e5m2" => EuclideanScannerAny::E5M2(EuclideanScanner::new(topology, threads)?),
            "e4m3" => EuclideanScannerAny::E4M3(EuclideanScanner::new(topology, threads)?),
            "e3m2" => EuclideanScannerAny::E3M2(EuclideanScanner::new(topology, threads)?),
            "e2m3" => EuclideanScannerAny::E2M3(EuclideanScanner::new(topology, threads)?),
            "i8" => EuclideanScannerAny::I8(EuclideanScanner::new(topology, threads)?),
            "i4x2" => EuclideanScannerAny::I4x2(EuclideanScanner::new(topology, threads)?),
            "u8" => EuclideanScannerAny::U8(EuclideanScanner::new(topology, threads)?),
            "u4x2" => EuclideanScannerAny::U4x2(EuclideanScanner::new(topology, threads)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn for_table(
        table: &Table<impl Allocator>,
        topology: &fu::Topology,
        threads: usize,
    ) -> Result<Self, Error> {
        Self::for_dtype(table.value_dtype(), topology, threads)
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
        table: &Table<impl Allocator>,
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
pub enum EuclideanSearcherAny {
    F64(EuclideanSearch<f64>),
    F32(EuclideanSearch<f32>),
    Bf16(EuclideanSearch<nk::bf16>),
    F16(EuclideanSearch<nk::f16>),
    E5M2(EuclideanSearch<nk::e5m2>),
    E4M3(EuclideanSearch<nk::e4m3>),
    E3M2(EuclideanSearch<nk::e3m2>),
    E2M3(EuclideanSearch<nk::e2m3>),
    I8(EuclideanSearch<i8>),
    I4x2(EuclideanSearch<nk::i4x2>),
    U8(EuclideanSearch<u8>),
    U4x2(EuclideanSearch<nk::u4x2>),
}

impl EuclideanSearcherAny {
    pub fn for_dtype(
        dtype: &str,
        table: &Table<impl Allocator>,
        expansion_factor: usize,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => EuclideanSearcherAny::F64(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "f32" => EuclideanSearcherAny::F32(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "bf16" => EuclideanSearcherAny::Bf16(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "f16" => EuclideanSearcherAny::F16(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "e5m2" => EuclideanSearcherAny::E5M2(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "e4m3" => EuclideanSearcherAny::E4M3(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "e3m2" => EuclideanSearcherAny::E3M2(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "e2m3" => EuclideanSearcherAny::E2M3(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "i8" => EuclideanSearcherAny::I8(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "i4x2" => EuclideanSearcherAny::I4x2(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "u8" => EuclideanSearcherAny::U8(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            "u4x2" => EuclideanSearcherAny::U4x2(EuclideanSearch::for_table(
                table,
                expansion_factor,
                Global,
            )?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn for_table(
        table: &Table<impl Allocator>,
        expansion_factor: usize,
    ) -> Result<Self, Error> {
        Self::for_dtype(table.value_dtype(), table, expansion_factor)
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
        table: &Table<impl Allocator>,
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
pub enum AngularScannerAny {
    F64(AngularScanner<f64>),
    F32(AngularScanner<f32>),
    Bf16(AngularScanner<nk::bf16>),
    F16(AngularScanner<nk::f16>),
    E5M2(AngularScanner<nk::e5m2>),
    E4M3(AngularScanner<nk::e4m3>),
    E3M2(AngularScanner<nk::e3m2>),
    E2M3(AngularScanner<nk::e2m3>),
    I8(AngularScanner<i8>),
    I4x2(AngularScanner<nk::i4x2>),
    U8(AngularScanner<u8>),
    U4x2(AngularScanner<nk::u4x2>),
}

impl AngularScannerAny {
    pub fn for_dtype(dtype: &str, topology: &fu::Topology, threads: usize) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => AngularScannerAny::F64(AngularScanner::new(topology, threads)?),
            "f32" => AngularScannerAny::F32(AngularScanner::new(topology, threads)?),
            "bf16" => AngularScannerAny::Bf16(AngularScanner::new(topology, threads)?),
            "f16" => AngularScannerAny::F16(AngularScanner::new(topology, threads)?),
            "e5m2" => AngularScannerAny::E5M2(AngularScanner::new(topology, threads)?),
            "e4m3" => AngularScannerAny::E4M3(AngularScanner::new(topology, threads)?),
            "e3m2" => AngularScannerAny::E3M2(AngularScanner::new(topology, threads)?),
            "e2m3" => AngularScannerAny::E2M3(AngularScanner::new(topology, threads)?),
            "i8" => AngularScannerAny::I8(AngularScanner::new(topology, threads)?),
            "i4x2" => AngularScannerAny::I4x2(AngularScanner::new(topology, threads)?),
            "u8" => AngularScannerAny::U8(AngularScanner::new(topology, threads)?),
            "u4x2" => AngularScannerAny::U4x2(AngularScanner::new(topology, threads)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn for_table(
        table: &Table<impl Allocator>,
        topology: &fu::Topology,
        threads: usize,
    ) -> Result<Self, Error> {
        Self::for_dtype(table.value_dtype(), topology, threads)
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
        table: &Table<impl Allocator>,
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
pub enum AngularSearcherAny {
    F64(AngularSearch<f64>),
    F32(AngularSearch<f32>),
    Bf16(AngularSearch<nk::bf16>),
    F16(AngularSearch<nk::f16>),
    E5M2(AngularSearch<nk::e5m2>),
    E4M3(AngularSearch<nk::e4m3>),
    E3M2(AngularSearch<nk::e3m2>),
    E2M3(AngularSearch<nk::e2m3>),
    I8(AngularSearch<i8>),
    I4x2(AngularSearch<nk::i4x2>),
    U8(AngularSearch<u8>),
    U4x2(AngularSearch<nk::u4x2>),
}

impl AngularSearcherAny {
    pub fn for_dtype(
        dtype: &str,
        table: &Table<impl Allocator>,
        expansion_factor: usize,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => {
                AngularSearcherAny::F64(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "f32" => {
                AngularSearcherAny::F32(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "bf16" => {
                AngularSearcherAny::Bf16(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "f16" => {
                AngularSearcherAny::F16(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "e5m2" => {
                AngularSearcherAny::E5M2(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "e4m3" => {
                AngularSearcherAny::E4M3(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "e3m2" => {
                AngularSearcherAny::E3M2(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "e2m3" => {
                AngularSearcherAny::E2M3(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "i8" => {
                AngularSearcherAny::I8(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "i4x2" => {
                AngularSearcherAny::I4x2(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "u8" => {
                AngularSearcherAny::U8(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            "u4x2" => {
                AngularSearcherAny::U4x2(AngularSearch::for_table(table, expansion_factor, Global)?)
            }
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn for_table(
        table: &Table<impl Allocator>,
        expansion_factor: usize,
    ) -> Result<Self, Error> {
        Self::for_dtype(table.value_dtype(), table, expansion_factor)
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
        table: &Table<impl Allocator>,
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
    use crate::{NeighborList, OffsetDType};

    /// Three nodes whose values are held in dtype `T` in a NumKong matrix, so one
    /// fixture serves every scalar — float, minifloat, integer, nibble, or packed
    /// bit. [`try_from_scalars`] packs the logical `f32` rows into `T` in whatever
    /// layout the scalar demands, so a sub-byte `i4x2` or `u1x8` builds the same way
    /// a byte type does, and the matrix carries its own logical width. Fully
    /// connected at degree two, so a walk from any landmark reaches all three; degree
    /// zero gives an edgeless graph.
    ///
    /// [`try_from_scalars`]: nk::Matrix::try_from_scalars
    struct Ring<T: nk::FloatConvertible> {
        dtype: &'static str,
        values: nk::Matrix<T>,
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

    fn ring<T: nk::FloatConvertible>(
        dtype: &'static str,
        rows: &[f32],
        width: usize,
        degree: usize,
    ) -> Ring<T> {
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

    impl<T: nk::FloatConvertible> GraphLike for Ring<T> {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            self.dtype
        }
        fn value_shape(&self) -> Option<usize> {
            // The logical column count the matrix already knows.
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
    /// identically. `T`'s [`nk::Matrix::view`] is the shaped [`ScoringLike::Queries`].
    fn ring_table<T: nk::FloatConvertible>(
        dtype: &'static str,
        rows: &[f32],
        query: &[f32],
    ) -> (Table, nk::Matrix<T>) {
        let width = query.len();
        let node_count = rows.len() / width;
        let values = nk::Matrix::<T>::try_from_scalars(rows, &[node_count, width]).unwrap();
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
            nk::Matrix::<T>::try_from_scalars(query, &[1, width]).unwrap(),
        )
    }

    fn dot_covers<T>(
        topology: &fu::Topology,
        dtype: &'static str,
        rows: &[f32],
        query: &[f32],
        order: [u64; 3],
    ) where
        T: nk::FloatConvertible + nk::Dot + nk::Dots + Clone + Send + Sync + 'static,
        T: nk::Dot<Output = <T as nk::Dots>::Accumulator>,
        <T as nk::Dot>::Output: Copy + Default + PartialOrd + Widen + Send + Sync,
        DotSearcher<T>: SearcherLike<DotScoring<T>, Index = u32>,
    {
        let (table, query) = ring_table::<T>(dtype, rows, query);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        DotScanner::<T>::new(topology, 1)
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
        DotSearcher::<T>::new(3, 3)
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

    fn euclidean_covers<T>(
        topology: &fu::Topology,
        dtype: &'static str,
        rows: &[f32],
        query: &[f32],
        order: [u64; 3],
    ) where
        T: nk::FloatConvertible + nk::Euclidean + nk::Euclideans + Clone + Send + Sync + 'static,
        T: nk::Euclidean<EuclideanOutput = <T as nk::Euclideans>::SpatialResult>,
        <T as nk::Euclidean>::EuclideanOutput: Copy + Default + PartialOrd + Widen + Send + Sync,
        EuclideanSearcher<T>: SearcherLike<EuclideanScoring<T>, Index = u32>,
    {
        let (table, query) = ring_table::<T>(dtype, rows, query);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        EuclideanScanner::<T>::new(topology, 1)
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
        EuclideanSearcher::<T>::new(3, 3)
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

    fn angular_covers<T>(
        topology: &fu::Topology,
        dtype: &'static str,
        rows: &[f32],
        query: &[f32],
        order: [u64; 3],
    ) where
        T: nk::FloatConvertible + nk::Angular + nk::Angulars + Clone + Send + Sync + 'static,
        T: nk::Angular<Output = <T as nk::Angulars>::SpatialResult>,
        <T as nk::Angular>::Output: Copy + Default + PartialOrd + Widen + Send + Sync,
        AngularSearcher<T>: SearcherLike<AngularScoring<T>, Index = u32>,
    {
        let (table, query) = ring_table::<T>(dtype, rows, query);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        AngularScanner::<T>::new(topology, 1)
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
        AngularSearcher::<T>::new(3, 3)
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
        let topology = fu::Topology::new().unwrap();
        let dense = [2u64, 1, 0];
        dot_covers::<f64>(&topology, "f64", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<f32>(&topology, "f32", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::bf16>(&topology, "bf16", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::f16>(&topology, "f16", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::e5m2>(&topology, "e5m2", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::e4m3>(&topology, "e4m3", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::e3m2>(&topology, "e3m2", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::e2m3>(&topology, "e2m3", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<i8>(&topology, "i8", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::i4x2>(&topology, "i4x2", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<u8>(&topology, "u8", &RING_ROWS, &RING_QUERY, dense);
        dot_covers::<nk::u4x2>(&topology, "u4x2", &RING_ROWS, &RING_QUERY, dense);
        // The bit type takes the binary ring: dot is `|A ∩ B|`, larger first.
        dot_covers::<nk::u1x8>(&topology, "u1x8", &BINARY_ROWS, &BINARY_QUERY, [0, 1, 2]);
    }

    #[test]
    fn euclidean_ranks_and_walks_every_dtype() {
        // Smaller-is-better: `[3,5]` sits nearest node 1, then 2, then 0, so the same
        // helper — ranking by the searcher's reversed comparator — yields `[1, 2, 0]`,
        // a different order from the dot over the very same fixture. No `u1x8`: L2 is
        // not a bit metric.
        let topology = fu::Topology::new().unwrap();
        let near = [1u64, 2, 0];
        euclidean_covers::<f64>(&topology, "f64", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<f32>(&topology, "f32", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::bf16>(&topology, "bf16", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::f16>(&topology, "f16", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::e5m2>(&topology, "e5m2", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::e4m3>(&topology, "e4m3", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::e3m2>(&topology, "e3m2", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::e2m3>(&topology, "e2m3", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<i8>(&topology, "i8", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::i4x2>(&topology, "i4x2", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<u8>(&topology, "u8", &RING_ROWS, &RING_QUERY, near);
        euclidean_covers::<nk::u4x2>(&topology, "u4x2", &RING_ROWS, &RING_QUERY, near);
    }

    #[test]
    fn angular_ranks_and_walks_every_dtype() {
        // Cosine distance: `[3,5]` is most aligned with node 0, least with node 2, so
        // nearest-first is `[0, 1, 2]` — a third distinct order from one fixture,
        // proving the metric, not the data, decides the ranking.
        let topology = fu::Topology::new().unwrap();
        let aligned = [0u64, 1, 2];
        angular_covers::<f64>(&topology, "f64", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<f32>(&topology, "f32", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::bf16>(&topology, "bf16", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::f16>(&topology, "f16", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::e5m2>(&topology, "e5m2", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::e4m3>(&topology, "e4m3", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::e3m2>(&topology, "e3m2", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::e2m3>(&topology, "e2m3", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<i8>(&topology, "i8", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::i4x2>(&topology, "i4x2", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<u8>(&topology, "u8", &RING_ROWS, &RING_QUERY, aligned);
        angular_covers::<nk::u4x2>(&topology, "u4x2", &RING_ROWS, &RING_QUERY, aligned);
    }

    #[test]
    fn batch_scan_and_search_fill_matrices() {
        // The proposed uniform batch shape: one query row, top-3 out, `k` and the
        // query count read off the `matches`/`distances` matrices, `counts` reporting
        // how many each query found. Query `[3,5]` dots to `[13,29,45]`, so both the
        // brute-force scan and the graph walk rank nodes `[2,1,0]`, scores `[45,29,13]`.
        let table = Table::build(&ring_i8(), None).unwrap();
        let topology = fu::Topology::new().unwrap();
        let queries = i8_query(&[3.0, 5.0]);

        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut scanner: DotScanner<i8> = DotScanner::new(&topology, 1).unwrap();
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

        let mut searcher: DotSearcher<i8> = DotSearcher::new(table.len(), 3).unwrap();
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

        let topology = fu::Topology::new().unwrap();
        let mut scanner = DotScanner::<i8>::new(&topology, 2).unwrap();
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
        let topology = fu::Topology::new().unwrap();
        let mut scanner = EuclideanScanner::<i8>::new(&topology, 1).unwrap();
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
        let mut searcher = DotSearcher::<i8>::new(table.len(), 3).unwrap();
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
        fn value_shape(&self) -> Option<usize> {
            Some(2)
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
        let mut searcher = DotSearcher::<i8>::new(table.len(), 6).unwrap();
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
    fn scan_f32_accumulates_in_f64() {
        // The same ring in `f32`, degree zero so the fixture also covers an edgeless
        // graph; the tile lands exactly in the `f64` accumulator.
        let table = Table::build(&ring::<f32>("f32", &RING_ROWS, 2, 0), None).unwrap();
        let topology = fu::Topology::new().unwrap();
        let mut scanner = DotScanner::<f32>::new(&topology, 2).unwrap();
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
        let topology = fu::Topology::new().unwrap();
        let mut scanner = DotScanner::<i8>::new(&topology, 1).unwrap();
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
        // constructible and runs the batch path. This table's neighbors are `u32`, so
        // reading them at `u64` is a genuine width mismatch — [`Error::WrongType`] —
        // which is exactly the check that proves `I` is honored at the neighbor read. A
        // true `u64`-neighbor table needs a graph past four billion nodes to build.
        let table = Table::build(&ring_i8(), None).unwrap();
        let mut searcher: DotSearcher<i8, u64> = DotSearcher::new(table.len(), 3).unwrap();
        let query = i8_query(&[3.0, 4.0]);
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];
        assert!(matches!(
            searcher.search(
                query.view(),
                &table,
                &mut matches,
                &mut distances,
                &mut counts
            ),
            Err(Error::WrongType)
        ));
    }

    #[test]
    fn dispatcher_resolves_dtype_and_width() {
        // A caller who opened the file blind: dtype from the header, neighbor width
        // from the column, both resolved inside the searcher, not at the call site.
        let table = Table::build(&ring_i8(), None).unwrap();
        let mut searcher = DotSearcherAny::for_table(&table, 3).unwrap();
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

        let topology = fu::Topology::new().unwrap();
        let mut scanner = DotScannerAny::for_table(&table, &topology, 1).unwrap();
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
        let topology = fu::Topology::new().unwrap();
        assert!(matches!(
            DotScannerAny::for_dtype("u16", &topology, 1),
            Err(Error::UnknownDType)
        ));
    }

    #[test]
    fn euclidean_and_angular_dispatch_rank_nearest() {
        // The erased runtime path for the two distance metrics: dispatch on the header
        // dtype and rank nearest-first, the scan for Euclidean and the walk for
        // Angular, from the same `[3,5]` query.
        let table = Table::build(&ring_i8(), None).unwrap();
        let topology = fu::Topology::new().unwrap();
        let query: [i8; 2] = [3, 5];
        let bytes = unsafe { core::slice::from_raw_parts(query.as_ptr() as *const u8, 2) };
        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut euclidean = EuclideanScannerAny::for_table(&table, &topology, 1).unwrap();
        assert_eq!(euclidean.dtype(), "i8");
        euclidean
            .scan(bytes, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[1, 2, 0]);

        let mut angular = AngularSearcherAny::for_table(&table, 3).unwrap();
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
        fn value_shape(&self) -> Option<usize> {
            Some(2)
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
        let mut searcher = DotSearcher::<i8>::new(table.len(), 3).unwrap();
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

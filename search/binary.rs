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
//! The dense-tensor plumbing — the packing-aware [`matrix_view`]/[`value_view`] scan
//! primitive and the shared [`batch_scan`]/[`batch_search`] top-k — is reused from
//! `dense` rather than copied; only the metric bodies live here.
//!
//! [`AngularScanner`]: crate::search::dense::AngularScanner
//! [`batch_scan`]: crate::search::dense::batch_scan
//! [`batch_search`]: crate::search::dense::batch_search

use super::{walk, Matrix, ScannerLike, Scores2D, ScoringLike, SearcherLike, TraversalContext};
use crate::search::dense::{batch_scan, batch_search, value_view, view_from_bytes};
use crate::{cast, Error, GraphLike, OffsetDType, OffsetScalar, Table, Widen};
use allocator_api2::alloc::{Allocator, Global};
use core::cmp::Ordering;
use forkunion as fu;
use numkong as nk;

/// Best-first for a distance: `Greater` means `a` is the nearer, so a smaller score
/// outranks a larger one. Both bit metrics rank this way.
fn smaller_first<S: PartialOrd>(a: S, b: S) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Less)
}

// region: Hamming

/// The batch half of the Hamming metric. Structurally the [`AngularScanner`] of the
/// bit world: it packs the query side through [`nk::DotsPackedMatrix`] and streams
/// the value side from the map, differing only in the popcount-of-XOR kernel.
///
/// [`AngularScanner`]: crate::search::dense::AngularScanner
pub struct HammingScanner<T: nk::Hammings> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<T>,
}

impl<T: nk::Hammings> HammingScanner<T> {
    pub fn new(topology: &fu::Topology, threads: usize) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(HammingScanner {
            pool: fu::ThreadPool::try_spawn(topology, threads.max(1)).map_err(|_| Error::Alloc)?,
            packed: nk::DotsPackedMatrix::empty_in(nk::Global),
        })
    }
}

impl<T> HammingScanner<T>
where
    T: nk::Hamming + nk::Hammings + 'static,
    <T as nk::Hamming>::Output: Copy + Default + PartialOrd + Widen,
    HammingScanner<T>: ScannerLike<HammingScoring<T>>,
{
    /// Batch brute-force top-k, nearest-first — Hamming is a smaller-is-better
    /// distance. `matches`/`distances` carry `k` and the query count.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        batch_scan::<HammingScoring<T>, _>(self, queries, table, true, matches, distances, counts)
    }
}

/// The traversal half of the Hamming metric.
pub struct HammingSearcher<T: nk::Hamming, I = u32, A: Allocator = Global> {
    context: TraversalContext<I, <T as nk::Hamming>::Output, A>,
}

impl<T, I> HammingSearcher<T, I, Global>
where
    T: nk::Hamming,
    I: OffsetScalar,
    <T as nk::Hamming>::Output: Copy + Default,
{
    pub fn new(nodes: usize, expansion_factor: usize) -> Result<Self, Error> {
        HammingSearcher::new_in(nodes, expansion_factor, Global)
    }
}

impl<T, I, A> HammingSearcher<T, I, A>
where
    T: nk::Hamming,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Hamming>::Output: Copy + Default,
{
    pub fn new_in(nodes: usize, expansion_factor: usize, alloc: A) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(HammingSearcher {
            context: TraversalContext::new_in(nodes, expansion_factor, alloc)?,
        })
    }

    pub fn context(&mut self) -> &mut TraversalContext<I, <T as nk::Hamming>::Output, A> {
        &mut self.context
    }
}

impl<T, I, A> HammingSearcher<T, I, A>
where
    T: nk::Hamming + 'static,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Hamming>::Output: Copy + Default + PartialOrd + Widen,
    HammingSearcher<T, I, A>: SearcherLike<HammingScoring<T>, Index = I>,
{
    /// Batch top-k by graph walk, uniform with [`HammingScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        batch_search::<HammingScoring<T>, _>(self, queries, table, matches, distances, counts)
    }
}

/// Hamming distance — the count of differing bits — over a packed-bit scalar, in
/// `u32`. Smaller is nearer, so it ranks like a Euclidean distance. One generic
/// marker spans every bit type, as [`DotScoring`] does for the reals.
///
/// [`DotScoring`]: crate::search::dense::DotScoring
pub struct HammingScoring<T>(core::marker::PhantomData<T>);

impl<T: nk::Hamming + 'static> ScoringLike for HammingScoring<T>
where
    <T as nk::Hamming>::Output: Copy + Default,
{
    type Query<'a> = &'a [T];
    type Queries<'a> = nk::MatrixView<'a, T>;
    type Score = <T as nk::Hamming>::Output;
}

impl<M, T> ScannerLike<M> for HammingScanner<T>
where
    T: nk::Hammings + Clone + Send + Sync + 'static,
    M: ScoringLike<Score = u32> + ?Sized,
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
        nk::HammingsPackedParallelOps::try_hammings_packed_parallel_into(
            &values,
            &self.packed,
            &mut tile,
            &mut self.pool,
        )
        .map_err(|_| Error::Inconsistent)
    }
}

impl<M, T, I, A> SearcherLike<M> for HammingSearcher<T, I, A>
where
    T: nk::Hamming + 'static,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Hamming>::Output: Copy + Default + PartialOrd,
    M: ScoringLike<Score = <T as nk::Hamming>::Output> + ?Sized,
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
            *slot = <T as nk::Hamming>::hamming(query, value).ok_or(Error::Inconsistent)?;
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

/// A Hamming searcher whose neighbor width is resolved from the table, the bit twin
/// of [`AngularSearch`].
///
/// [`AngularSearch`]: crate::search::dense::AngularSearch
pub enum HammingSearch<T: nk::Hamming, A: Allocator = Global> {
    U32(HammingSearcher<T, u32, A>),
    U64(HammingSearcher<T, u64, A>),
}

impl<T, A> HammingSearch<T, A>
where
    T: nk::Hamming + 'static,
    A: Allocator + Clone,
    <T as nk::Hamming>::Output: Copy + Default + PartialOrd,
{
    pub fn for_table(
        table: &Table<impl Allocator>,
        expansion_factor: usize,
        alloc: A,
    ) -> Result<Self, Error> {
        Ok(match table.neighbors_column().dtype {
            OffsetDType::U32 => HammingSearch::U32(HammingSearcher::new_in(
                table.len(),
                expansion_factor,
                alloc,
            )?),
            OffsetDType::U64 => HammingSearch::U64(HammingSearcher::new_in(
                table.len(),
                expansion_factor,
                alloc,
            )?),
        })
    }

    /// Batch top-k by walk, forwarding to the width-resolved leaf. Uniform with
    /// [`HammingScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <T as nk::Hamming>::Output: Widen,
        HammingSearcher<T, u32, A>: SearcherLike<HammingScoring<T>, Index = u32>,
        HammingSearcher<T, u64, A>: SearcherLike<HammingScoring<T>, Index = u64>,
    {
        match self {
            HammingSearch::U32(leaf) => leaf.search(queries, table, matches, distances, counts),
            HammingSearch::U64(leaf) => leaf.search(queries, table, matches, distances, counts),
        }
    }
}

/// The Hamming twin of [`DotScannerAny`]: runtime dtype dispatch, `u1x8` the one arm
/// with a packed Hamming kernel. The typed scanner owns its tile, so unlike the dot
/// dispatcher this needs no scratch of its own.
///
/// [`DotScannerAny`]: crate::search::dense::DotScannerAny
pub enum HammingScannerAny {
    U1x8(HammingScanner<nk::u1x8>),
}

impl HammingScannerAny {
    pub fn for_dtype(dtype: &str, topology: &fu::Topology, threads: usize) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => HammingScannerAny::U1x8(HammingScanner::new(topology, threads)?),
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
            HammingScannerAny::U1x8(_) => "u1x8",
        }
    }

    /// Batch brute-force top-k over a query batch given as raw bytes in the table's
    /// dtype, cast to the packed bit scalar and forwarded to the typed scanner.
    pub fn scan(
        &mut self,
        queries: &[u8],
        table: &Table<impl Allocator>,
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
pub enum HammingSearcherAny {
    U1x8(HammingSearch<nk::u1x8>),
}

impl HammingSearcherAny {
    pub fn for_dtype(
        dtype: &str,
        table: &Table<impl Allocator>,
        expansion_factor: usize,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => {
                HammingSearcherAny::U1x8(HammingSearch::for_table(table, expansion_factor, Global)?)
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
            HammingSearcherAny::U1x8(_) => "u1x8",
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
pub struct JaccardScanner<T: nk::Jaccards> {
    pool: fu::ThreadPool,
    packed: nk::DotsPackedMatrix<T>,
}

impl<T: nk::Jaccards> JaccardScanner<T> {
    pub fn new(topology: &fu::Topology, threads: usize) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(JaccardScanner {
            pool: fu::ThreadPool::try_spawn(topology, threads.max(1)).map_err(|_| Error::Alloc)?,
            packed: nk::DotsPackedMatrix::empty_in(nk::Global),
        })
    }
}

impl<T> JaccardScanner<T>
where
    T: nk::Jaccard + nk::Jaccards + 'static,
    <T as nk::Jaccard>::Output: Copy + Default + PartialOrd + Widen,
    JaccardScanner<T>: ScannerLike<JaccardScoring<T>>,
{
    /// Batch brute-force top-k, nearest-first — Jaccard is a smaller-is-better
    /// distance. `matches`/`distances` carry `k` and the query count.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        batch_scan::<JaccardScoring<T>, _>(self, queries, table, true, matches, distances, counts)
    }
}

/// The traversal half of the Jaccard metric.
pub struct JaccardSearcher<T: nk::Jaccard, I = u32, A: Allocator = Global> {
    context: TraversalContext<I, <T as nk::Jaccard>::Output, A>,
}

impl<T, I> JaccardSearcher<T, I, Global>
where
    T: nk::Jaccard,
    I: OffsetScalar,
    <T as nk::Jaccard>::Output: Copy + Default,
{
    pub fn new(nodes: usize, expansion_factor: usize) -> Result<Self, Error> {
        JaccardSearcher::new_in(nodes, expansion_factor, Global)
    }
}

impl<T, I, A> JaccardSearcher<T, I, A>
where
    T: nk::Jaccard,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Jaccard>::Output: Copy + Default,
{
    pub fn new_in(nodes: usize, expansion_factor: usize, alloc: A) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(JaccardSearcher {
            context: TraversalContext::new_in(nodes, expansion_factor, alloc)?,
        })
    }

    pub fn context(&mut self) -> &mut TraversalContext<I, <T as nk::Jaccard>::Output, A> {
        &mut self.context
    }
}

impl<T, I, A> JaccardSearcher<T, I, A>
where
    T: nk::Jaccard + 'static,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Jaccard>::Output: Copy + Default + PartialOrd + Widen,
    JaccardSearcher<T, I, A>: SearcherLike<JaccardScoring<T>, Index = I>,
{
    /// Batch top-k by graph walk, uniform with [`JaccardScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        batch_search::<JaccardScoring<T>, _>(self, queries, table, matches, distances, counts)
    }
}

/// Jaccard distance — one minus intersection over union — over a packed-bit scalar,
/// in `f32`. Smaller is nearer, ranking like Hamming.
pub struct JaccardScoring<T>(core::marker::PhantomData<T>);

impl<T: nk::Jaccard + 'static> ScoringLike for JaccardScoring<T>
where
    <T as nk::Jaccard>::Output: Copy + Default,
{
    type Query<'a> = &'a [T];
    type Queries<'a> = nk::MatrixView<'a, T>;
    type Score = <T as nk::Jaccard>::Output;
}

impl<M, T> ScannerLike<M> for JaccardScanner<T>
where
    T: nk::Jaccards + Clone + Send + Sync + 'static,
    <T as nk::Jaccards>::JaccardResult: Send + Sync,
    M: ScoringLike<Score = <T as nk::Jaccards>::JaccardResult> + ?Sized,
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

        let element = core::mem::size_of::<M::Score>() as isize;
        let mut tile = unsafe {
            nk::MatrixSpan::<M::Score>::from_raw_parts(
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

impl<M, T, I, A> SearcherLike<M> for JaccardSearcher<T, I, A>
where
    T: nk::Jaccard + 'static,
    I: OffsetScalar,
    A: Allocator + Clone,
    <T as nk::Jaccard>::Output: Copy + Default + PartialOrd,
    M: ScoringLike<Score = <T as nk::Jaccard>::Output> + ?Sized,
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
            *slot = <T as nk::Jaccard>::jaccard(query, value).ok_or(Error::Inconsistent)?;
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

/// A Jaccard searcher whose neighbor width is resolved from the table.
pub enum JaccardSearch<T: nk::Jaccard, A: Allocator = Global> {
    U32(JaccardSearcher<T, u32, A>),
    U64(JaccardSearcher<T, u64, A>),
}

impl<T, A> JaccardSearch<T, A>
where
    T: nk::Jaccard + 'static,
    A: Allocator + Clone,
    <T as nk::Jaccard>::Output: Copy + Default + PartialOrd,
{
    pub fn for_table(
        table: &Table<impl Allocator>,
        expansion_factor: usize,
        alloc: A,
    ) -> Result<Self, Error> {
        Ok(match table.neighbors_column().dtype {
            OffsetDType::U32 => JaccardSearch::U32(JaccardSearcher::new_in(
                table.len(),
                expansion_factor,
                alloc,
            )?),
            OffsetDType::U64 => JaccardSearch::U64(JaccardSearcher::new_in(
                table.len(),
                expansion_factor,
                alloc,
            )?),
        })
    }

    /// Batch top-k by walk, forwarding to the width-resolved leaf. Uniform with
    /// [`JaccardScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, T>,
        table: &Table<impl Allocator>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error>
    where
        <T as nk::Jaccard>::Output: Widen,
        JaccardSearcher<T, u32, A>: SearcherLike<JaccardScoring<T>, Index = u32>,
        JaccardSearcher<T, u64, A>: SearcherLike<JaccardScoring<T>, Index = u64>,
    {
        match self {
            JaccardSearch::U32(leaf) => leaf.search(queries, table, matches, distances, counts),
            JaccardSearch::U64(leaf) => leaf.search(queries, table, matches, distances, counts),
        }
    }
}

/// The Jaccard twin of [`HammingScannerAny`]: runtime dispatch, `u1x8` the one arm.
pub enum JaccardScannerAny {
    U1x8(JaccardScanner<nk::u1x8>),
}

impl JaccardScannerAny {
    pub fn for_dtype(dtype: &str, topology: &fu::Topology, threads: usize) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => JaccardScannerAny::U1x8(JaccardScanner::new(topology, threads)?),
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
            JaccardScannerAny::U1x8(_) => "u1x8",
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
pub enum JaccardSearcherAny {
    U1x8(JaccardSearch<nk::u1x8>),
}

impl JaccardSearcherAny {
    pub fn for_dtype(
        dtype: &str,
        table: &Table<impl Allocator>,
        expansion_factor: usize,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "u1x8" => {
                JaccardSearcherAny::U1x8(JaccardSearch::for_table(table, expansion_factor, Global)?)
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
            JaccardSearcherAny::U1x8(_) => "u1x8",
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
        let topology = fu::Topology::new().unwrap();
        let table = Table::build(&binary_ring(2), None).unwrap();
        let query = binary_query();

        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut scanner = HammingScanner::<nk::u1x8>::new(&topology, 1).unwrap();
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

        let mut searcher = HammingSearcher::<nk::u1x8>::new(table.len(), 3).unwrap();
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
        let topology = fu::Topology::new().unwrap();
        let table = Table::build(&binary_ring(2), None).unwrap();
        let query = binary_query();

        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut scanner = JaccardScanner::<nk::u1x8>::new(&topology, 1).unwrap();
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

        let mut searcher = JaccardSearcher::<nk::u1x8>::new(table.len(), 3).unwrap();
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
        let topology = fu::Topology::new().unwrap();

        let mut matches = Matrix::<u64>::new(1, 3).unwrap();
        let mut distances = Matrix::<f64>::new(1, 3).unwrap();
        let mut counts = [0u64; 1];

        let mut scanner = HammingScannerAny::for_table(&table, &topology, 1).unwrap();
        assert_eq!(scanner.dtype(), "u1x8");
        scanner
            .scan(&query, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1, 2]);

        let mut searcher = JaccardSearcherAny::for_table(&table, 3).unwrap();
        assert_eq!(searcher.dtype(), "u1x8");
        searcher
            .search(&query, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1, 2]);
    }
}

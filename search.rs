//! Search over a [`crate::Table`]: a brute-force scan and an A* traversal.
//!
//! This module has two layers. The reusable primitives here — [`MinQueue`],
//! [`VisitedSet`], [`Context`], and [`Match`] — are allocator-aware and
//! dependency-free, and compile in every build. The scan/A* engine and the
//! NumKong / StringZilla scoring kernels live behind the `search` feature.
//!
//! The primitives are meant to be reused: one [`Context`] per thread, reset
//! between queries instead of reallocated, and outputs written into a
//! caller-preallocated slice of [`Match`].

use allocator_api2::alloc::{Allocator, Global};
use allocator_api2::vec::Vec as AVec;

use crate::Error;

/// A scored result: a node index of any width paired with a score of any type.
/// The caller preallocates `[Match<I, S>]` and thereby fixes both widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Match<I, S> {
    pub node: I,
    pub score: S,
}

/// A search-frontier entry: a node paired with its priority score, generic over
/// both the node-index width `I` and the score `S`. The engine pops the best
/// candidate first, ordered by the metric's [`ScoringLike::cmp`], so no total
/// order — and no encoded `i64` sort key — is baked into the type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate<I, S> {
    pub node: I,
    pub score: S,
}

/// A reusable binary heap ordered by a caller-supplied comparator, where
/// `before(a, b)` is true when `a` should be popped before `b`. Passing the
/// comparator per call keeps the heap free of any `Ord` bound, so it can order
/// scores like `f64` that have no total order. `clear` keeps the allocation for
/// the next query.
pub struct MinQueue<T, A: Allocator = Global> {
    heap: AVec<T, A>,
}

impl<T, A: Allocator + Clone> MinQueue<T, A> {
    pub fn new_in(alloc: A) -> Self {
        MinQueue {
            heap: AVec::new_in(alloc),
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Empty the heap, keeping its capacity for reuse.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// The highest-priority element without removing it.
    pub fn peek(&self) -> Option<&T> {
        self.heap.first()
    }

    /// Push `item`, sifting it up while it should precede its parent.
    pub fn push(&mut self, item: T, before: impl Fn(&T, &T) -> bool) -> Result<(), Error> {
        self.heap.try_reserve(1).map_err(|_| Error::Alloc)?;
        self.heap.push(item);
        let mut i = self.heap.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if before(&self.heap[i], &self.heap[parent]) {
                self.heap.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Pop the highest-priority element — the one no sibling precedes.
    pub fn pop(&mut self, before: impl Fn(&T, &T) -> bool) -> Option<T> {
        let count = self.heap.len();
        if count == 0 {
            return None;
        }
        self.heap.swap(0, count - 1);
        let best = self.heap.pop();
        let len = self.heap.len();
        let mut i = 0;
        loop {
            let (left, right) = (2 * i + 1, 2 * i + 2);
            let mut next = i;
            if left < len && before(&self.heap[left], &self.heap[next]) {
                next = left;
            }
            if right < len && before(&self.heap[right], &self.heap[next]) {
                next = right;
            }
            if next == i {
                break;
            }
            self.heap.swap(i, next);
            i = next;
        }
        best
    }
}

/// A reusable visited set over a fixed node range. Version-stamped, so `reset`
/// is O(1): it bumps a generation counter instead of clearing the array.
pub struct VisitedSet<A: Allocator = Global> {
    stamps: AVec<u32, A>,
    generation: u32,
}

impl<A: Allocator + Clone> VisitedSet<A> {
    pub fn new_in(capacity: usize, alloc: A) -> Result<Self, Error> {
        let mut stamps = AVec::new_in(alloc);
        stamps
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Alloc)?;
        stamps.resize(capacity, 0);
        Ok(VisitedSet {
            stamps,
            generation: 1,
        })
    }

    pub fn capacity(&self) -> usize {
        self.stamps.len()
    }

    /// Forget every mark. O(1) unless the generation counter wraps.
    pub fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            for stamp in self.stamps.iter_mut() {
                *stamp = 0;
            }
            self.generation = 1;
        }
    }

    /// Mark `node`, returning true if it was not already marked this generation.
    pub fn visit(&mut self, node: usize) -> bool {
        match self.stamps.get_mut(node) {
            Some(stamp) if *stamp != self.generation => {
                *stamp = self.generation;
                true
            }
            _ => false,
        }
    }

    pub fn contains(&self, node: usize) -> bool {
        self.stamps
            .get(node)
            .is_some_and(|stamp| *stamp == self.generation)
    }
}

/// A per-thread bundle of reusable traversal scratch. Hold one per worker thread
/// and [`reset`](TraversalContext::reset) it between queries rather than
/// reallocating.
///
/// Generic over the node-index width `I` (`u32`, `u64`, …) and the metric's
/// score type `S` (`f32`, `i32`, `f64`, …).
pub struct TraversalContext<A: Allocator = Global, I = u32, S = f32> {
    /// A* candidate frontier, ordered best-first for expansion.
    frontier: MinQueue<Candidate<I, S>, A>,
    /// A* bounded best-so-far set, ordered worst-first for eviction.
    results: MinQueue<Candidate<I, S>, A>,
    /// Closed set for A*.
    visited: VisitedSet<A>,
    /// Backing for the rank-2 cross-product score tile used by the scan.
    scores: AVec<S, A>,
}

impl<A: Allocator + Clone, I, S> TraversalContext<A, I, S> {
    /// A context sized for a graph of `nodes` nodes, allocating in `alloc`.
    pub fn new_in(nodes: usize, alloc: A) -> Result<Self, Error> {
        Ok(TraversalContext {
            frontier: MinQueue::new_in(alloc.clone()),
            results: MinQueue::new_in(alloc.clone()),
            visited: VisitedSet::new_in(nodes, alloc.clone())?,
            scores: AVec::new_in(alloc),
        })
    }

    /// Clear every buffer for the next query, keeping the allocations.
    pub fn reset(&mut self) {
        self.frontier.clear();
        self.results.clear();
        self.visited.reset();
        self.scores.clear();
    }
}

impl TraversalContext<Global, u32, f32> {
    /// A context sized for `nodes` nodes on the global heap, with `u32` node
    /// indices and `f32` scores.
    pub fn new(nodes: usize) -> Result<Self, Error> {
        TraversalContext::new_in(nodes, Global)
    }
}

// The scoring engine. Everything above compiles in every build; everything
// below is gated on `search`. Each metric family lives in its own submodule,
// grouped by how it reads a node's value.
#[cfg(feature = "search")]
pub mod dense;
#[cfg(feature = "search")]
pub mod geospatial;
#[cfg(feature = "search")]
pub mod maxsim;
#[cfg(feature = "search")]
pub mod sequence;

#[cfg(feature = "search")]
pub use dense::ScoringDot;

#[cfg(feature = "search")]
use crate::{GraphLike, Table};
#[cfg(feature = "search")]
use numkong as nk;

/// The SafeGraphs header dtype string for a NumKong scalar. A local trait over
/// foreign scalars — legal by the orphan rule, leaving the sealed `Dots` set
/// untouched — so a metric can assert a table's runtime dtype matches the
/// query's compile-time scalar `T`. The table below is the authoritative dtype
/// universe search understands; a runtime `&str → T` dispatcher will read from
/// the same list.
#[cfg(feature = "search")]
pub trait DTypeName: nk::Dots {
    const DTYPE: &'static str;
}

#[cfg(feature = "search")]
macro_rules! dtype_names {
    ($($scalar:ty => $name:literal),+ $(,)?) => {
        $(
            impl DTypeName for $scalar {
                const DTYPE: &'static str = $name;
            }
        )+
    };
}

#[cfg(feature = "search")]
dtype_names! {
    f32 => "f32",
    f64 => "f64",
    nk::f16 => "f16",
    nk::bf16 => "bf16",
    i8 => "i8",
    u8 => "u8",
    nk::e4m3 => "e4m3",
    nk::e5m2 => "e5m2",
}

/// A scoring metric together with its reusable traversal scratch — the object a
/// caller scores with. A metric OVERRIDES the three kernels ([`cross`], [`score`],
/// [`compare`]) and exposes its stored [`TraversalContext`] via [`context`]; on
/// top of those it inherits [`scan`] (brute force) and [`search`] (A*). `Score`
/// is the metric's native accumulator (`f64` for an `f32` dot, `i32` for `i8`),
/// never a blanket-widened type.
///
/// [`cross`]: ScoringLike::cross
/// [`score`]: ScoringLike::score
/// [`compare`]: ScoringLike::compare
/// [`context`]: ScoringLike::context
/// [`scan`]: ScoringLike::scan
/// [`search`]: ScoringLike::search
#[cfg(feature = "search")]
pub trait ScoringLike<T: nk::Dots> {
    /// The metric's native score type.
    type Score: nk::StorageElement;
    /// The allocator behind the scorer's [`TraversalContext`].
    type Alloc: Allocator + Clone;
    /// The node-index width of results and the frontier.
    type Index: Copy + TryFrom<usize> + TryInto<usize>;

    /// Batched: fill `matches[node][query]` for a packed query batch scored
    /// against a whole value column in one cross-product (NumKong `DotsPacked`).
    /// `&mut self` so a metric may drive its own executor (e.g. a thread pool).
    fn cross(
        &mut self,
        queries: &nk::PackedMatrix<T>,
        values: &nk::MatrixView<'_, T>,
        matches: &mut nk::MatrixSpan<'_, Self::Score>,
    ) -> Result<(), Error>;

    /// Vector-vector: score one query row against one node's value (NumKong
    /// `nk::Dot`), or `None` if the two are not the same width.
    fn score(&self, query: &[T], value: &[T]) -> Option<Self::Score>;

    /// Best-first ordering: `Greater` means `a` outranks `b`.
    fn compare(a: Self::Score, b: Self::Score) -> core::cmp::Ordering;

    /// The scorer's reusable traversal scratch.
    fn context(&mut self) -> &mut TraversalContext<Self::Alloc, Self::Index, Self::Score>;

    /// Brute-force top-`k` over every node, for an eager-packed query batch.
    ///
    /// `queries` is packed once (and reused across shards) by e.g.
    /// [`ScoringDot::prepare`]. `out` is a caller-preallocated `queries × k`
    /// slice, row-major, so `out[query * k + rank]` is the `rank`-th best node for
    /// that query.
    fn scan(
        &mut self,
        queries: &nk::PackedMatrix<T>,
        table: &Table<impl Allocator>,
        k: usize,
        out: &mut [Match<Self::Index, Self::Score>],
    ) -> Result<(), Error>
    where
        T: DTypeName,
    {
        if T::DTYPE != table.value_dtype() {
            return Err(Error::WrongType);
        }
        let nodes = table.len();
        let dimension = table.value_shape().ok_or(Error::Inconsistent)?;
        let queries_count = queries.dims().0;
        if k == 0 || out.len() != queries_count * k {
            return Err(Error::Inconsistent);
        }
        let values = value_view::<T>(table, nodes, dimension)?;

        // Size the reusable `nodes × queries` score tile, then fill it in place.
        let cells = nodes.checked_mul(queries_count).ok_or(Error::TooLarge)?;
        {
            let scratch = self.context();
            scratch.scores.clear();
            scratch.scores.try_reserve(cells).map_err(|_| Error::Alloc)?;
            scratch.scores.resize(cells, Self::Score::default());
        }
        {
            let element_bytes = core::mem::size_of::<Self::Score>() as isize;
            let tile_ptr = self.context().scores.as_mut_ptr();
            let mut tile = unsafe {
                nk::MatrixSpan::<Self::Score>::from_raw_parts(
                    tile_ptr,
                    &[nodes, queries_count],
                    &[queries_count as isize * element_bytes, element_bytes],
                )
            };
            self.cross(queries, &values, &mut tile)?;
        }

        // Rank each query's column (stride `queries_count`) into its slot of `out`.
        let compare = |a: Self::Score, b: Self::Score| Self::compare(a, b);
        let scores = &self.context().scores;
        for query in 0..queries_count {
            let ranked = &mut out[query * k..query * k + k];
            let mut filled = 0usize;
            for node in 0..nodes {
                let score = scores[node * queries_count + query];
                if filled < k {
                    insert_at(ranked, filled, node, score, &compare)?;
                    filled += 1;
                } else if Self::compare(score, ranked[k - 1].score) == core::cmp::Ordering::Greater {
                    insert_at(ranked, k - 1, node, score, &compare)?;
                }
            }
        }
        Ok(())
    }

    /// Greedy/A* top-`k` for one query, walking the neighbor graph from `roots`.
    ///
    /// `ef` (≥ `k`) is the search breadth — the bounded best-so-far set kept while
    /// walking. `out` (length `k`) is filled best-first; if fewer than `k` nodes
    /// are reached, the trailing slots are left untouched.
    fn search(
        &mut self,
        query: &[T],
        table: &Table<impl Allocator>,
        roots: &[Self::Index],
        ef: usize,
        k: usize,
        out: &mut [Match<Self::Index, Self::Score>],
    ) -> Result<(), Error>
    where
        T: DTypeName,
    {
        if T::DTYPE != table.value_dtype() {
            return Err(Error::WrongType);
        }
        if k == 0 || ef < k || out.len() != k {
            return Err(Error::Inconsistent);
        }
        let nodes = table.len();
        let toward_best =
            |a: &Candidate<Self::Index, Self::Score>, b: &Candidate<Self::Index, Self::Score>| {
                Self::compare(a.score, b.score) == core::cmp::Ordering::Greater
            };
        let toward_worst =
            |a: &Candidate<Self::Index, Self::Score>, b: &Candidate<Self::Index, Self::Score>| {
                Self::compare(a.score, b.score) == core::cmp::Ordering::Less
            };

        self.context().reset();

        // Seed the frontier and result set with the entry points.
        for &root in roots {
            let index: usize = root.try_into().map_err(|_| Error::Inconsistent)?;
            if index >= nodes || !self.context().visited.visit(index) {
                continue;
            }
            let score = self.score(query, value_at::<T>(table, index)?).ok_or(Error::Inconsistent)?;
            let candidate = Candidate { node: root, score };
            self.context().frontier.push(candidate, toward_best)?;
            self.context().results.push(candidate, toward_worst)?;
            if self.context().results.len() > ef {
                self.context().results.pop(toward_worst);
            }
        }

        // Expand the best candidate until none can improve the result set.
        while let Some(candidate) = self.context().frontier.pop(toward_best) {
            if self.saturated(candidate.score, ef) {
                break;
            }
            let node: usize = candidate.node.try_into().map_err(|_| Error::Inconsistent)?;
            let neighbors = table.neighbors(node);
            for neighbor in neighbors.iter() {
                let index = neighbor as usize;
                if index >= nodes || !self.context().visited.visit(index) {
                    continue;
                }
                let score = self.score(query, value_at::<T>(table, index)?).ok_or(Error::Inconsistent)?;
                if self.saturated(score, ef) {
                    continue;
                }
                let node_index = Self::Index::try_from(index).map_err(|_| Error::TooLarge)?;
                let candidate = Candidate { node: node_index, score };
                self.context().frontier.push(candidate, toward_best)?;
                self.context().results.push(candidate, toward_worst)?;
                if self.context().results.len() > ef {
                    self.context().results.pop(toward_worst);
                }
            }
        }

        // Drain the `k` best from the worst-first result set, best-first into `out`.
        let found = self.context().results.len();
        let keep = found.min(k);
        for _ in 0..found.saturating_sub(keep) {
            self.context().results.pop(toward_worst);
        }
        for slot in (0..keep).rev() {
            let candidate = self
                .context()
                .results
                .pop(toward_worst)
                .ok_or(Error::Inconsistent)?;
            out[slot] = Match {
                node: candidate.node,
                score: candidate.score,
            };
        }
        Ok(())
    }

    /// True when `score` cannot enter a full (`>= ef`) result set — the walk's
    /// admission and termination test.
    fn saturated(&mut self, score: Self::Score, ef: usize) -> bool {
        let results = &self.context().results;
        results.len() >= ef
            && match results.peek() {
                Some(worst) => Self::compare(score, worst.score) != core::cmp::Ordering::Greater,
                None => false,
            }
    }
}

/// Insert node `index` with `score` into the sorted prefix `out[..=end]`,
/// shifting weaker entries right so the prefix stays best-first. Widening the
/// `0..len` index into `I` fails with [`Error::TooLarge`] if it does not fit.
#[cfg(feature = "search")]
fn insert_at<I, S>(
    out: &mut [Match<I, S>],
    end: usize,
    index: usize,
    score: S,
    compare: &impl Fn(S, S) -> core::cmp::Ordering,
) -> Result<(), Error>
where
    I: Copy + TryFrom<usize>,
    S: Copy,
{
    let node = I::try_from(index).map_err(|_| Error::TooLarge)?;
    let mut i = end;
    while i > 0 && compare(score, out[i - 1].score) == core::cmp::Ordering::Greater {
        out[i] = out[i - 1];
        i -= 1;
    }
    out[i] = Match { node, score };
    Ok(())
}

/// A zero-copy `nodes × dimension` view of a table's value column, straight from
/// the memory map, for the batched scan.
#[cfg(feature = "search")]
fn value_view<T: DTypeName>(
    table: &Table<impl Allocator>,
    nodes: usize,
    dimension: usize,
) -> Result<nk::MatrixView<'_, T>, Error> {
    let scalars: &[T] = crate::cast(table.value_buffer()).ok_or(Error::WrongType)?;
    if scalars.len() != nodes * dimension {
        return Err(Error::Inconsistent);
    }
    let size = core::mem::size_of::<T>() as isize;
    // SAFETY: the value buffer is dtype-aligned and `nodes * dimension` long, so
    // the shape and unit strides describe exactly its extent, borrowed for `'_`.
    Ok(unsafe {
        nk::MatrixView::<T>::from_raw_parts(
            scalars.as_ptr(),
            &[nodes, dimension],
            &[dimension as isize * size, size],
        )
    })
}

/// One node's value reinterpreted as `&[T]` in place, for the vector-vector walk.
#[cfg(feature = "search")]
fn value_at<T: DTypeName>(table: &Table<impl Allocator>, node: usize) -> Result<&[T], Error> {
    crate::cast(table.value(node)).ok_or(Error::WrongType)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Order smallest-first, the comparator every heap test below reuses.
    fn ascending(a: &i32, b: &i32) -> bool {
        a < b
    }

    #[test]
    fn min_queue_pops_in_order() {
        let mut queue: MinQueue<i32> = MinQueue::new_in(Global);
        for x in [5, 1, 4, 2, 3, 0] {
            queue.push(x, ascending).unwrap();
        }
        let mut out = Vec::new();
        while let Some(x) = queue.pop(ascending) {
            out.push(x);
        }
        assert_eq!(out, vec![0, 1, 2, 3, 4, 5]);
        assert!(queue.is_empty());
    }

    #[test]
    fn min_queue_reuses_after_clear() {
        let mut queue: MinQueue<i32> = MinQueue::new_in(Global);
        queue.push(9, ascending).unwrap();
        queue.clear();
        assert!(queue.is_empty());
        queue.push(2, ascending).unwrap();
        queue.push(1, ascending).unwrap();
        assert_eq!(queue.pop(ascending), Some(1));
    }

    #[test]
    fn candidate_orders_by_score() {
        // Frontier of `u32` nodes with `i32` scores, popped lowest-score first.
        let mut queue: MinQueue<Candidate<u32, i32>> = MinQueue::new_in(Global);
        let lower = |a: &Candidate<u32, i32>, b: &Candidate<u32, i32>| a.score < b.score;
        queue.push(Candidate { node: 0, score: 30 }, lower).unwrap();
        queue.push(Candidate { node: 1, score: 10 }, lower).unwrap();
        queue.push(Candidate { node: 2, score: 20 }, lower).unwrap();
        assert_eq!(queue.pop(lower).unwrap().node, 1);
        assert_eq!(queue.pop(lower).unwrap().node, 2);
        assert_eq!(queue.pop(lower).unwrap().node, 0);
    }

    #[test]
    fn visited_marks_once_per_generation() {
        let mut visited = VisitedSet::new_in(4, Global).unwrap();
        assert!(visited.visit(2));
        assert!(!visited.visit(2));
        assert!(visited.contains(2));
        assert!(!visited.contains(0));
        assert!(!visited.visit(9)); // out of range

        visited.reset();
        assert!(!visited.contains(2));
        assert!(visited.visit(2));
    }

    #[test]
    fn context_resets() {
        let mut ctx: TraversalContext = TraversalContext::new(8).unwrap();
        ctx.frontier
            .push(Candidate { node: 0, score: 1.0 }, |a, b| a.score < b.score)
            .unwrap();
        assert!(ctx.visited.visit(3));
        ctx.reset();
        assert!(ctx.frontier.is_empty());
        assert!(!ctx.visited.contains(3));
    }
}

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

/// A per-thread bundle of reusable search scratch. Hold one per worker thread
/// and [`reset`](Context::reset) it between queries rather than reallocating.
///
/// Generic over the node-index width `I` (`u32`, `u64`, …) and the metric's
/// score type `S` (`f32`, `i32`, `f64`, …).
pub struct Context<A: Allocator = Global, I = u32, S = f32> {
    /// A* candidate frontier, ordered by the metric's direction.
    frontier: MinQueue<Candidate<I, S>, A>,
    /// Closed set for A*.
    visited: VisitedSet<A>,
    /// Backing for the rank-2 cross-product score tile.
    scores: AVec<S, A>,
    /// Neighbor values gathered contiguous for one cross-product call.
    gather: AVec<u8, A>,
}

impl<A: Allocator + Clone, I, S> Context<A, I, S> {
    /// A context sized for a graph of `nodes` nodes, allocating in `alloc`.
    pub fn new_in(nodes: usize, alloc: A) -> Result<Self, Error> {
        Ok(Context {
            frontier: MinQueue::new_in(alloc.clone()),
            visited: VisitedSet::new_in(nodes, alloc.clone())?,
            scores: AVec::new_in(alloc.clone()),
            gather: AVec::new_in(alloc),
        })
    }

    /// Clear every buffer for the next query, keeping the allocations.
    pub fn reset(&mut self) {
        self.frontier.clear();
        self.visited.reset();
        self.scores.clear();
        self.gather.clear();
    }
}

impl Context<Global, u32, f32> {
    /// A context sized for `nodes` nodes on the global heap, with `u32` node
    /// indices and `f32` scores.
    pub fn new(nodes: usize) -> Result<Self, Error> {
        Context::new_in(nodes, Global)
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
pub use dense::Dot;

#[cfg(feature = "search")]
use crate::{GraphLike, Table};
#[cfg(feature = "search")]
use numkong::MatrixSpan;

/// A scoring kernel: fills a rank-2 score tile `out[i][j] = score(query i,
/// node j)` in one batched pass, then ranks candidates by [`cmp`].
///
/// The value side is always the [`Table`]; only the query representation varies,
/// so it is an associated [`Query`](ScoringLike::Query) type — a dense matrix for
/// vector metrics, byte strings for sequence metrics, multi-vectors for MaxSim.
/// Batching is deliberate: NumKong and StringZilla both prefer whole batches, so
/// the engine hands the kernel every query and every node at once.
///
/// [`cmp`]: ScoringLike::cmp
#[cfg(feature = "search")]
pub trait ScoringLike {
    /// How a batch of queries is handed to this metric.
    type Query<'a>;
    /// The score the metric produces (`f64`, `i64`, …).
    type Score: Copy + Default;

    /// The number of queries in `queries` — the engine needs it to size the tile.
    fn query_count(queries: &Self::Query<'_>) -> usize;

    /// Fill `out[i][j]` with the score of query `i` against node `j`'s value.
    fn cross<A: Allocator>(
        &self,
        table: &Table<A>,
        queries: Self::Query<'_>,
        out: &mut MatrixSpan<'_, Self::Score>,
    ) -> Result<(), Error>;

    /// Best-first ordering: `Greater` means `a` outranks `b`.
    fn cmp(&self, a: Self::Score, b: Self::Score) -> core::cmp::Ordering;
}

#[cfg(feature = "search")]
impl<A: Allocator + Clone> Table<A> {
    /// Brute-force top-`k` over every node, for a batch of queries.
    ///
    /// `out` is a caller-preallocated `q × k` slice, filled row-major so
    /// `out[i * k + r]` is the `r`-th best match for query `i`. The `ctx` scratch
    /// is reused across calls.
    pub fn scan<I: Copy + TryFrom<usize>, M: ScoringLike>(
        &self,
        queries: M::Query<'_>,
        metric: &M,
        k: usize,
        ctx: &mut Context<A, I, M::Score>,
        out: &mut [Match<I, M::Score>],
    ) -> Result<(), Error> {
        let nodes = self.len();
        let queries_count = M::query_count(&queries);
        if k == 0 || out.len() != queries_count * k {
            return Err(Error::Inconsistent);
        }

        // Size the reusable score tile to `queries_count × nodes`, without
        // reallocating inside the reserved capacity.
        let cells = queries_count.checked_mul(nodes).ok_or(Error::TooLarge)?;
        ctx.scores.clear();
        ctx.scores.try_reserve(cells).map_err(|_| Error::Alloc)?;
        ctx.scores.resize(cells, M::Score::default());
        {
            let elem = core::mem::size_of::<M::Score>() as isize;
            let mut tile = unsafe {
                MatrixSpan::<M::Score>::from_raw_parts(
                    ctx.scores.as_mut_ptr(),
                    &[queries_count, nodes],
                    &[nodes as isize * elem, elem],
                )
            };
            metric.cross(self, queries, &mut tile)?;
        }

        // Rank each query's row into its slot of the output.
        for qi in 0..queries_count {
            let row = &ctx.scores[qi * nodes..qi * nodes + nodes];
            top_k(row, metric, &mut out[qi * k..qi * k + k])?;
        }
        Ok(())
    }
}

/// Fill `out` with the best entries of `row`, best-first, by insertion into the
/// sorted output. `out.len()` is the retained count `k`.
#[cfg(feature = "search")]
fn top_k<I: Copy + TryFrom<usize>, M: ScoringLike>(
    row: &[M::Score],
    metric: &M,
    out: &mut [Match<I, M::Score>],
) -> Result<(), Error> {
    use core::cmp::Ordering;
    let k = out.len();
    let mut filled = 0usize;
    for (node, &score) in row.iter().enumerate() {
        if filled < k {
            insert_at(out, filled, node, score, metric)?;
            filled += 1;
        } else if metric.cmp(score, out[k - 1].score) == Ordering::Greater {
            insert_at(out, k - 1, node, score, metric)?;
        }
    }
    Ok(())
}

/// Insert node `index` with `score` into the sorted prefix `out[..=end]`,
/// shifting weaker entries right so the prefix stays best-first. Widening the
/// `0..len` index into `I` fails with [`Error::TooLarge`] if it does not fit.
#[cfg(feature = "search")]
fn insert_at<I: Copy + TryFrom<usize>, M: ScoringLike>(
    out: &mut [Match<I, M::Score>],
    end: usize,
    index: usize,
    score: M::Score,
    metric: &M,
) -> Result<(), Error> {
    let node = I::try_from(index).map_err(|_| Error::TooLarge)?;
    let mut i = end;
    while i > 0 && metric.cmp(score, out[i - 1].score) == core::cmp::Ordering::Greater {
        out[i] = out[i - 1];
        i -= 1;
    }
    out[i] = Match { node, score };
    Ok(())
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
        let mut ctx: Context = Context::new(8).unwrap();
        ctx.frontier
            .push(Candidate { node: 0, score: 1.0 }, |a, b| a.score < b.score)
            .unwrap();
        assert!(ctx.visited.visit(3));
        ctx.reset();
        assert!(ctx.frontier.is_empty());
        assert!(!ctx.visited.contains(3));
    }
}

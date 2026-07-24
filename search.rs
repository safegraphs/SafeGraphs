//! Search over a [`crate::Table`]: shared traversal machinery, and a contract
//! every searcher follows.
//!
//! This module deliberately contains no scoring. It provides the pieces a metric
//! reuses — [`MinQueue`], [`VisitedSet`], [`TraversalContext`], [`Matrix`],
//! [`rank_tile`], and the [`walk`] itself — and the vocabulary that lets a caller
//! treat two metrics alike. Everything that touches a kernel lives in a family
//! module, so `dense` never mentions StringZilla and `sequence` never mentions
//! NumKong.
//!
//! A metric is described by [`MetricLike`], which declares types and no
//! behavior, and is worked by two functors it names. A [`ScannerLike`] owns an
//! executor — a thread pool or a device — and scores a whole batch; there are few
//! of them. A [`SearcherLike`] owns a [`TraversalContext`] and walks the graph for
//! one query; there is one per worker thread, which is what makes its `&mut self`
//! free. Nothing here is inherited: shared code is called, never derived.

use allocator_api2::alloc::{Allocator as AllocatorLike, Global};
use allocator_api2::vec::Vec as AVec;
use core::cmp::Ordering;

use crate::{Error, IndexLike};

/// Where a functor's kernels run, and how wide — the single parallelism knob
/// that crosses the API. Every functor's one constructor takes one of these;
/// no pool, topology, scheduler, or device handle appears in any signature.
///
/// Each family opens the scope into whatever its backend calls an executor: the
/// NumKong families spawn a ForkUnion thread pool, the StringZilla ones a
/// `DeviceScope`, and that conversion is the only place a backend type appears.
/// A family whose kernels are CPU-only answers [`Gpu`](ExecutionScope::Gpu)
/// with [`Error::WrongType`] at construction rather than pretending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionScope {
    /// The caller's thread alone — the nil state. Nothing is spawned, and the
    /// sequential path is the same code path as the parallel one.
    Serial,
    /// CPU threads, the caller included; zero means every logical core the
    /// machine reports.
    Cpus(usize),
    /// One GPU, by zero-based device index. Fanning across several GPUs is not
    /// a functor's business: build one functor per device and shard tables
    /// across them.
    Gpu(usize),
}

/// Which end of a score axis wins: `Nearest` for a distance, where smaller is
/// better, `Farthest` for a similarity. Carried instead of a bare `bool` so a
/// call site reads its direction: `batch_scan(.., Ranking::Nearest, ..)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ranking {
    Nearest,
    Farthest,
}

impl Ranking {
    /// Does `a` outrank `b` under this direction?
    pub fn outranks(self, a: f64, b: f64) -> bool {
        match self {
            Ranking::Nearest => a < b,
            Ranking::Farthest => a > b,
        }
    }

    /// Order two native scores best-first under this direction — the comparator
    /// every family's `SearcherLike::compare` and `rank_tile` hand to the pool.
    /// Generic over the metric's own `Score`, so it runs on unwidened values; a
    /// `NaN` sinks (`Ordering::Less`) rather than poisoning the order. This is the
    /// one home for what was a per-file `smaller_first`/`larger_first`.
    pub fn compare<Score: PartialOrd>(self, a: Score, b: Score) -> Ordering {
        let ordered = match self {
            Ranking::Nearest => b.partial_cmp(&a),
            Ranking::Farthest => a.partial_cmp(&b),
        };
        ordered.unwrap_or(Ordering::Less)
    }
}

/// A scored result. The node is the table's own index type `I`, `u32` or `u64`,
/// so a result carries no more width than the graph declares; it is widened to
/// `u64` only when it leaves a walk for an untyped caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Match<Index, Score> {
    pub node: Index,
    pub score: Score,
}

/// A frontier entry: a node paired with the score that orders it, in the table's
/// native index type `I`, so `Candidate<u32, f32>` is a compact eight bytes.
/// Ordering comes from the metric's own comparator, so no total order — and no
/// encoded sort key — is baked into the type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate<Index, Score> {
    pub node: Index,
    pub score: Score,
}

/// A reusable binary heap ordered by a caller-supplied comparator, where
/// `before(a, b)` is true when `a` should be popped before `b`. Passing the
/// comparator per call keeps the heap free of any `Ord` bound, so it can order
/// scores like `f64` that have no total order. `clear` keeps the allocation for
/// the next query.
pub struct MinQueue<Item, Allocator: AllocatorLike = Global> {
    heap: AVec<Item, Allocator>,
}

impl<Item, Allocator: AllocatorLike + Clone> MinQueue<Item, Allocator> {
    pub fn new_in(alloc: Allocator) -> Self {
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
    pub fn peek(&self) -> Option<&Item> {
        self.heap.first()
    }

    /// Push `item`, sifting it up while it should precede its parent.
    pub fn push(&mut self, item: Item, before: impl Fn(&Item, &Item) -> bool) -> Result<(), Error> {
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
    pub fn pop(&mut self, before: impl Fn(&Item, &Item) -> bool) -> Option<Item> {
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
pub struct VisitedSet<Allocator: AllocatorLike = Global> {
    stamps: AVec<u32, Allocator>,
    generation: u32,
    /// How many nodes are marked this generation. The stamped array cannot report
    /// this in O(1), and the landmark sampler needs it to know when the whole
    /// graph has been drawn.
    marked: usize,
}

impl<Allocator: AllocatorLike + Clone> VisitedSet<Allocator> {
    pub fn new_in(capacity: usize, alloc: Allocator) -> Result<Self, Error> {
        let mut stamps = AVec::new_in(alloc);
        stamps
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Alloc)?;
        stamps.resize(capacity, 0);
        Ok(VisitedSet {
            stamps,
            generation: 1,
            marked: 0,
        })
    }

    pub fn capacity(&self) -> usize {
        self.stamps.len()
    }

    /// Grow to cover `nodes`, keeping the current generation's marks: new slots are
    /// zero-stamped, which no live generation matches, so they read as unvisited.
    /// Never shrinks — a set follows its context from table to table.
    pub fn reserve(&mut self, nodes: usize) -> Result<(), Error> {
        let current = self.stamps.len();
        if nodes <= current {
            return Ok(());
        }
        self.stamps
            .try_reserve_exact(nodes - current)
            .map_err(|_| Error::Alloc)?;
        self.stamps.resize(nodes, 0);
        Ok(())
    }

    /// How many distinct nodes are marked this generation.
    pub fn len(&self) -> usize {
        self.marked
    }

    pub fn is_empty(&self) -> bool {
        self.marked == 0
    }

    /// Forget every mark. O(1) unless the generation counter wraps.
    pub fn reset(&mut self) {
        self.marked = 0;
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
                self.marked += 1;
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

/// A SplitMix64 generator, used to pick a walk's random entry points. Cheap,
/// stateless beyond one `u64`, and deterministic given its seed — the same
/// generator, and the same `bounded` reduction, that the USearch volatile index
/// draws landmarks with, so the two agree.
struct SplitMix {
    state: u64,
}

impl SplitMix {
    fn new(seed: u64) -> Self {
        SplitMix { state: seed }
    }

    /// One SplitMix64 step.
    fn unbounded(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// An index in `[0, bound)` by Lemire's multiply-shift — one multiply, no
    /// division. Full 64-bit, through a `u128` product, so it addresses the whole
    /// node range of a `u64` graph rather than capping at four billion.
    fn bounded(&mut self, bound: u64) -> u64 {
        ((self.unbounded() as u128 * bound as u128) >> 64) as u64
    }
}

/// A per-thread bundle of reusable traversal scratch, owned by a
/// [`SearcherLike`]. Hold one per worker thread and let the walk reset it between
/// queries rather than reallocating.
///
/// Generic over the table's native index type `I` and the metric's score `S`, so
/// the frontier and results stay exactly as wide as the graph — a `u32` node and
/// an `f32` score is an eight-byte [`Candidate`].
pub struct TraversalContext<Index, Score, Allocator: AllocatorLike = Global> {
    /// Candidate frontier, ordered best-first for expansion.
    frontier: MinQueue<Candidate<Index, Score>, Allocator>,
    /// Bounded best-so-far set, ordered worst-first so eviction is a pop.
    results: MinQueue<Candidate<Index, Score>, Allocator>,
    /// Closed set.
    visited: VisitedSet<Allocator>,
    /// Nodes staged for one batched expansion, and the scores that come back.
    /// Lent out wholesale for the duration of a walk, so the searcher can be
    /// borrowed mutably while they are in use.
    neighbors: AVec<Index, Allocator>,
    scores: AVec<Score, Allocator>,
    /// Search breadth, at least `k`. A per-searcher tuning knob rather than a
    /// per-query argument, since one searcher is one thread's worth of policy.
    expansion_factor: usize,
    /// Landmark generator. Its state persists across queries — `reset` does not
    /// touch it — so successive queries on one context start from different
    /// entry points, a stochastic multi-start rather than a fixed one.
    rng: SplitMix,
    /// Kept so a lent-out buffer can be replaced by an empty one of the same
    /// allocator; `AVec<_, Allocator>` has no `Default` to `take` through.
    alloc: Allocator,
}

impl<Index: IndexLike, Score: Copy + Default, Allocator: AllocatorLike + Clone>
    TraversalContext<Index, Score, Allocator>
{
    /// A context searching with breadth `expansion_factor`. It knows no table:
    /// [`walk`] grows the scratch to each searched graph through
    /// [`reserve`](TraversalContext::reserve), so one context serves many tables. The
    /// landmark generator is seeded to a fixed constant, so a single-threaded run
    /// is reproducible; a driver spreading work across threads seeds each context
    /// differently with [`seed`](TraversalContext::seed).
    pub fn new_in(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        if expansion_factor == 0 {
            return Err(Error::Inconsistent);
        }
        Ok(TraversalContext {
            frontier: MinQueue::new_in(alloc.clone()),
            results: MinQueue::new_in(alloc.clone()),
            visited: VisitedSet::new_in(0, alloc.clone())?,
            neighbors: AVec::new_in(alloc.clone()),
            scores: AVec::new_in(alloc.clone()),
            expansion_factor,
            rng: SplitMix::new(0),
            alloc,
        })
    }

    /// Reseed the landmark generator. A driver gives each per-thread context a
    /// distinct seed so their walks explore different entry points; the results
    /// are still deterministic given the seeds.
    pub fn seed(&mut self, seed: u64) {
        self.rng = SplitMix::new(seed);
    }

    /// The largest node count this context has grown to fit.
    pub fn nodes(&self) -> usize {
        self.visited.capacity()
    }

    /// Grow the scratch to cover a graph of `nodes` nodes — a no-op once sized.
    /// [`walk`] calls this at the start of every query, which is what lets one
    /// context follow its searcher from table to table.
    pub fn reserve(&mut self, nodes: usize) -> Result<(), Error> {
        self.visited.reserve(nodes)
    }

    pub fn expansion_factor(&self) -> usize {
        self.expansion_factor
    }

    /// Clear every buffer for the next query, keeping the allocations.
    pub fn reset(&mut self) {
        self.frontier.clear();
        self.results.clear();
        self.visited.reset();
        self.neighbors.clear();
        self.scores.clear();
    }

    /// Move the staging buffers out for the duration of one walk, so scoring can
    /// borrow the searcher mutably while holding them. Always paired with
    /// [`restore_staging`](TraversalContext::restore_staging), including on the
    /// error path, or their capacity is lost.
    fn lend_staging(&mut self) -> (AVec<Index, Allocator>, AVec<Score, Allocator>) {
        (
            core::mem::replace(&mut self.neighbors, AVec::new_in(self.alloc.clone())),
            core::mem::replace(&mut self.scores, AVec::new_in(self.alloc.clone())),
        )
    }

    fn restore_staging(
        &mut self,
        neighbors: AVec<Index, Allocator>,
        scores: AVec<Score, Allocator>,
    ) {
        self.neighbors = neighbors;
        self.scores = scores;
    }

    /// Drain the best-first results, calling `write(slot, node, score)` for each of
    /// the up-to-`count` best, and report how many. The writer decides the output
    /// shape, so one drain fills a native `Match<Index, Score>` and, at the erased
    /// boundary, a widened one — with no intermediate buffer either way.
    pub fn drain(
        &mut self,
        compare: impl Fn(Score, Score) -> Ordering,
        count: usize,
        mut write: impl FnMut(usize, Index, Score),
    ) -> usize {
        let toward_worst = |a: &Candidate<Index, Score>, b: &Candidate<Index, Score>| {
            compare(a.score, b.score) == Ordering::Less
        };
        let found = self.results.len();
        let keep = found.min(count);
        // The set is ordered worst-first, so drop the excess before filling from
        // the back — what remains is exactly the best `keep`.
        for _ in 0..found.saturating_sub(keep) {
            self.results.pop(toward_worst);
        }
        for slot in (0..keep).rev() {
            match self.results.pop(toward_worst) {
                Some(candidate) => write(slot, candidate.node, candidate.score),
                None => return keep,
            }
        }
        keep
    }

    /// Drain into a native `Match<Index, Score>` slice, best-first. Trailing slots are
    /// left untouched.
    pub fn drain_into(
        &mut self,
        compare: impl Fn(Score, Score) -> Ordering,
        out: &mut [Match<Index, Score>],
    ) -> usize {
        self.drain(compare, out.len(), |slot, node, score| {
            out[slot] = Match { node, score }
        })
    }
}

impl<Index: IndexLike, Score: Copy + Default> TraversalContext<Index, Score, Global> {
    /// A context on the global heap.
    pub fn new(expansion_factor: usize) -> Result<Self, Error> {
        TraversalContext::new_in(expansion_factor, Global)
    }
}

/// A dense, owned, row-major `rows × cols` matrix — the crate's only 2D container.
/// Allocation is fallible and never panics. The shape it carries is what lets a
/// batch search read `k` and the query count off the output instead of taking them
/// as arguments, and it grows-and-reuses across calls. Deliberately NumKong-free,
/// so it holds a batch's `matches` and `distances` for a StringZilla sequence
/// search as readily as a NumKong dense one.
///
/// It also serves as a scanner's reusable score tile, with `rows = nodes` and
/// `cols = queries`. Node-major reads backwards next to "top-`k` per query", and
/// it is deliberate: this is the layout NumKong's cross-product writes natively,
/// and its kernels reject a strided output outright, so the dense scan fills it
/// with no copy and no transpose. A sequence metric has to copy out of its device
/// matrix regardless, so it transposes inside that copy for free. Ranking then
/// strides across the tile, one cache line per row per column, dwarfed by the
/// work of producing the scores.
pub struct Matrix<Cell, Allocator: AllocatorLike = Global> {
    data: AVec<Cell, Allocator>,
    rows: usize,
    cols: usize,
}

impl<Cell: Copy + Default> Matrix<Cell, Global> {
    pub fn new(rows: usize, cols: usize) -> Result<Self, Error> {
        Self::new_in(rows, cols, Global)
    }
}

impl<Cell, Allocator: AllocatorLike> Matrix<Cell, Allocator> {
    /// An empty matrix awaiting its first [`reshape`](Matrix::reshape) —
    /// allocation-free, so a scanner can hold one as a field from construction.
    pub fn empty_in(alloc: Allocator) -> Self {
        Matrix {
            data: AVec::new_in(alloc),
            rows: 0,
            cols: 0,
        }
    }
}

impl<Cell> Matrix<Cell, Global> {
    pub fn empty() -> Self {
        Self::empty_in(Global)
    }
}

/// The empty matrix, so scratch tiles can be `mem::take`n around a call that
/// borrows their owner mutably at the same time.
impl<Cell> Default for Matrix<Cell, Global> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<Cell: Copy + Default, Allocator: AllocatorLike + Clone> Matrix<Cell, Allocator> {
    pub fn new_in(rows: usize, cols: usize, alloc: Allocator) -> Result<Self, Error> {
        let mut matrix = Matrix {
            data: AVec::new_in(alloc),
            rows: 0,
            cols: 0,
        };
        matrix.reshape(rows, cols)?;
        Ok(matrix)
    }

    /// Grow the backing to `rows * cols` if needed — never shrink — and adopt that
    /// shape, so a scanner reuses one tile across batches with no per-call alloc.
    pub fn reshape(&mut self, rows: usize, cols: usize) -> Result<(), Error> {
        let cells = rows.checked_mul(cols).ok_or(Error::TooLarge)?;
        if self.data.len() < cells {
            let extra = cells - self.data.len();
            self.data.try_reserve(extra).map_err(|_| Error::Alloc)?;
            self.data.resize(cells, Cell::default());
        }
        (self.rows, self.cols) = (rows, cols);
        Ok(())
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, i: usize) -> Option<&[Cell]> {
        (i < self.rows).then(|| &self.data[i * self.cols..i * self.cols + self.cols])
    }

    pub fn row_mut(&mut self, i: usize) -> Option<&mut [Cell]> {
        (i < self.rows).then(|| &mut self.data[i * self.cols..i * self.cols + self.cols])
    }

    /// The `rows * cols` cells in row-major order, the whole live extent.
    pub fn as_slice(&self) -> &[Cell] {
        &self.data[..self.rows * self.cols]
    }

    pub fn as_mut_slice(&mut self) -> &mut [Cell] {
        &mut self.data[..self.rows * self.cols]
    }

    /// The cell at `(row, col)`, or `None` out of range. Bounds are checked on
    /// both axes, so a `col` past the row's end cannot alias into the next row.
    pub fn get(&self, row: usize, col: usize) -> Option<Cell> {
        (row < self.rows && col < self.cols).then(|| self.data[row * self.cols + col])
    }

    /// A mutable reference to the cell at `(row, col)`, or `None` out of range.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Cell> {
        (row < self.rows && col < self.cols).then(|| &mut self.data[row * self.cols + col])
    }

    /// The raw base pointer, for kernels that fill the matrix wholesale. The
    /// live extent is `rows() * cols()` cells, row-major.
    pub fn as_mut_ptr(&mut self) -> *mut Cell {
        self.data.as_mut_ptr()
    }
}

/// A batch of queries a walk consumes one at a time: it reports how many it holds
/// and yields each as the metric's [`Query`](MetricLike::Query). A dense batch is a
/// `nk::MatrixView` yielding rows; a sequence batch is a `StringTape` view yielding
/// byte strings. This is the seam that lets one batch `search` walk either without
/// assuming a `&[T]`.
pub trait QueryBatchLike<Metric: MetricLike + ?Sized> {
    /// How many queries the batch holds.
    fn count(&self) -> usize;
    /// The query at `index`, or `None` if out of range or the batch is malformed.
    fn query(&self, index: usize) -> Option<Metric::Query<'_>>;
}

/// A metric: the vocabulary of its types, nothing more. It names how a query and
/// a query batch are shaped and what a score is; the functors that do the work —
/// a [`ScannerLike`] and a [`SearcherLike`] — reference it, so it declares no
/// method and no index, which is what keeps a dense metric and a sequence metric
/// from sharing a line of implementation.
pub trait MetricLike {
    /// One query. A rank-1 vector view for a dense metric, a byte slice for a
    /// sequence one. `Copy` because a walk scores with the same query on every
    /// expansion, and a cheap view is all any backend needs here.
    type Query<'a>: Copy;
    /// A batch of queries. A packed matrix view, or a tape of variable-length
    /// strings — there is no requirement that a batch have a fixed width.
    type Queries<'a>;
    /// The metric's native score, inferred from its backend rather than chosen:
    /// `<i8 as nk::Dot>::Output` is `i32`, an edit distance is `usize`.
    type Score: Copy + Default;
}

/// The batch half of a metric: score every query against every node.
pub trait ScannerLike<Metric: MetricLike + ?Sized> {
    /// Fill `scores` with one score per (query, node) pair. Ranking is not this
    /// method's business; [`rank_tile`] does that afterwards.
    fn scan(
        &mut self,
        queries: Metric::Queries<'_>,
        table: crate::TableView<'_>,
        scores: &mut Matrix<Metric::Score, impl AllocatorLike + Clone>,
    ) -> Result<(), Error>;
}

/// The traversal half of a metric: walk the neighbor graph for one query.
pub trait SearcherLike<Metric: MetricLike + ?Sized> {
    /// The node-index width this searcher traverses in, resolved from the table it
    /// was built for. Its scratch, frontier, and results are all this wide, so a
    /// `u32` graph never pays for `u64` cells. Not on [`MetricLike`]: it describes
    /// the table, not the metric.
    type Index: IndexLike;

    /// The allocator behind the context.
    type Alloc: AllocatorLike + Clone;

    /// The walk's only scoring primitive: one query against an arbitrary set of
    /// nodes, writing `nodes.len()` scores into `out`.
    ///
    /// It is batched rather than per-node because StringZilla ships no
    /// single-pair similarity function at all — every sequence metric is a batch
    /// engine — and because a 1×N call is what both backends, and every GPU,
    /// actually want.
    fn score_nodes(
        &mut self,
        query: Metric::Query<'_>,
        table: crate::TableView<'_>,
        nodes: &[Self::Index],
        out: &mut [Metric::Score],
    ) -> Result<(), Error>;

    /// The reusable traversal scratch this searcher owns.
    fn context(&mut self) -> &mut TraversalContext<Self::Index, Metric::Score, Self::Alloc>;

    /// Best-first ordering: `Greater` means `a` outranks `b`. A smaller-is-better
    /// metric reverses it here and needs no change anywhere else.
    fn compare(a: Metric::Score, b: Metric::Score) -> Ordering;

    /// Walk toward `query` from randomly sampled entry points, leaving the result
    /// set in the context. Every implementation is one line — a call to [`walk`] —
    /// because shared code is called here, never inherited.
    fn search(
        &mut self,
        query: Metric::Query<'_>,
        table: crate::TableView<'_>,
    ) -> Result<(), Error>;
}

/// The greedy best-first walk, as a free function every searcher delegates to.
///
/// It seeds from randomly sampled landmarks rather than caller-supplied roots: a
/// stochastic multi-start, seeded with about as many nodes as the graph's degree,
/// re-seeded whenever the frontier drains before the result pool fills. This is
/// the volatile-index traversal (`inner_volatile_index.hpp`), minus everything a
/// frozen file makes unnecessary — no watermark, no locks, no publication checks,
/// since every node in a `.safegraphs` file is readable and immovable.
///
/// One iteration expands one node and scores its entire unvisited neighbor list
/// in a single [`score_nodes`](SearcherLike::score_nodes) call, which is the shape
/// both backends want and the only shape a sequence metric can serve at all. A
/// neighbor id outside `0..table.len()` is a sentinel and is skipped, so a
/// fixed-degree adjacency row may be padded rather than trimmed to length.
/// Results are left in the searcher's context; drain them with
/// [`TraversalContext::drain_into`].
///
/// [`Error::WrongType`] and friends propagate from the metric untouched.
pub fn walk<Metric, Score>(
    searcher: &mut Score,
    query: Metric::Query<'_>,
    table: crate::TableView<'_>,
) -> Result<(), Error>
where
    Metric: MetricLike + ?Sized,
    Score: SearcherLike<Metric> + ?Sized,
{
    searcher.context().reserve(table.len())?;
    searcher.context().reset();
    let (mut staged, mut scores) = searcher.context().lend_staging();
    let outcome = walk_staged::<Metric, Score>(searcher, query, table, &mut staged, &mut scores);
    searcher.context().restore_staging(staged, scores);
    outcome
}

/// The body of [`walk`], with the staging buffers held as locals so the searcher
/// stays mutably borrowable while they are in use.
fn walk_staged<Metric, Score>(
    searcher: &mut Score,
    query: Metric::Query<'_>,
    table: crate::TableView<'_>,
    staged: &mut AVec<Score::Index, impl AllocatorLike>,
    scores: &mut AVec<Metric::Score, impl AllocatorLike>,
) -> Result<(), Error>
where
    Metric: MetricLike + ?Sized,
    Score: SearcherLike<Metric> + ?Sized,
{
    let nodes = table.len();
    if nodes == 0 {
        return Ok(());
    }
    let expansion_factor = searcher.context().expansion_factor();
    // Seed with about as many landmarks as the graph's mean degree — its
    // connectivity — so a walk rarely rides one entry point's local minimum to the
    // exclusion of others. At least one, so even an edgeless graph makes progress.
    let landmarks = table.mean_degree().max(1);
    let toward_best = |a: &Candidate<Score::Index, Metric::Score>,
                       b: &Candidate<Score::Index, Metric::Score>| {
        Score::compare(a.score, b.score) == Ordering::Greater
    };
    let toward_worst = |a: &Candidate<Score::Index, Metric::Score>,
                        b: &Candidate<Score::Index, Metric::Score>| {
        Score::compare(a.score, b.score) == Ordering::Less
    };

    // First seed, scored in one batch like any other expansion.
    sample_landmarks(searcher, nodes, landmarks, staged);
    admit(
        searcher,
        query,
        table,
        staged,
        scores,
        expansion_factor,
        &toward_best,
        &toward_worst,
    )?;

    loop {
        let candidate = match searcher.context().frontier.pop(toward_best) {
            Some(candidate) => candidate,
            None => {
                // A drained frontier before a full pool means this seed's reachable
                // set is exhausted — a sparse or disconnected graph — so restart
                // the descent from fresh landmarks. Self-limiting: both `visited`
                // and `results` only grow, so restarts cannot recur forever.
                if searcher.context().results.len() >= expansion_factor {
                    break;
                }
                if !sample_landmarks(searcher, nodes, landmarks, staged) {
                    break;
                }
                admit(
                    searcher,
                    query,
                    table,
                    staged,
                    scores,
                    expansion_factor,
                    &toward_best,
                    &toward_worst,
                )?;
                continue;
            }
        };
        if saturated::<Metric, Score>(searcher, candidate.score, expansion_factor) {
            break;
        }
        let index: usize = candidate.node.try_into().map_err(|_| Error::TooLarge)?;
        // The neighbor bytes are read as the native index type with no widening;
        // the searcher's `Index` was resolved from this table, so the width agrees.
        let list = table.neighbors(index);
        let neighbors = list.as_slice::<Score::Index>().ok_or(Error::WrongType)?;
        stage_unvisited(searcher, neighbors.iter().copied(), nodes, staged)?;
        if staged.is_empty() {
            continue;
        }
        admit(
            searcher,
            query,
            table,
            staged,
            scores,
            expansion_factor,
            &toward_best,
            &toward_worst,
        )?;
    }
    Ok(())
}

/// Draw up to `count` distinct unvisited nodes into `staged`, marking each
/// visited. A uniform draw that lands on a visited node probes forward cyclically
/// until it finds a free one, so every call makes strict progress and needs no
/// retry budget. Returns false when the whole graph is already visited, which is
/// the walk's signal that no restart can add anything.
fn sample_landmarks<Metric, Score>(
    searcher: &mut Score,
    nodes: usize,
    count: usize,
    staged: &mut AVec<Score::Index, impl AllocatorLike>,
) -> bool
where
    Metric: MetricLike + ?Sized,
    Score: SearcherLike<Metric> + ?Sized,
{
    staged.clear();
    let bound = nodes as u64;
    while staged.len() < count && searcher.context().visited.len() < nodes {
        let mut id = searcher.context().rng.bounded(bound) as usize;
        while searcher.context().visited.contains(id) {
            id = if id + 1 == nodes { 0 } else { id + 1 };
        }
        // `id < nodes`, and the searcher was sized for this table, so it fits; a
        // failure is treated like a failed reserve below, seeding fewer.
        let Ok(node) = Score::Index::try_from(id) else {
            break;
        };
        searcher.context().visited.visit(id);
        if staged.try_reserve(1).is_err() {
            break;
        }
        staged.push(node);
    }
    !staged.is_empty()
}

/// Collect the not-yet-visited members of `candidates` into `staged`, marking
/// each as visited so no node is ever scored twice.
///
/// Any candidate outside `0..nodes` is a sentinel and is skipped, by the format's
/// convention: a neighbor id at or past the node count names no node, so a
/// producer may pad a fixed-degree adjacency row with such a value — the node
/// count itself, or the id type's maximum — instead of storing a variable-length
/// list. The reader needs no distinguished sentinel value; every out-of-range id
/// is one.
fn stage_unvisited<Metric, Score>(
    searcher: &mut Score,
    candidates: impl Iterator<Item = Score::Index>,
    nodes: usize,
    staged: &mut AVec<Score::Index, impl AllocatorLike>,
) -> Result<(), Error>
where
    Metric: MetricLike + ?Sized,
    Score: SearcherLike<Metric> + ?Sized,
{
    staged.clear();
    for node in candidates {
        // Out of range, whether the id overflows `usize` or merely names no node:
        // a sentinel, skipped.
        let Ok(index) = TryInto::<usize>::try_into(node) else {
            continue;
        };
        if index >= nodes {
            continue;
        }
        if searcher.context().visited.visit(index) {
            staged.try_reserve(1).map_err(|_| Error::Alloc)?;
            staged.push(node);
        }
    }
    Ok(())
}

/// Score every staged node in one call, then admit each into the frontier and the
/// bounded result set.
#[allow(clippy::too_many_arguments)]
fn admit<Metric, Score>(
    searcher: &mut Score,
    query: Metric::Query<'_>,
    table: crate::TableView<'_>,
    staged: &AVec<Score::Index, impl AllocatorLike>,
    scores: &mut AVec<Metric::Score, impl AllocatorLike>,
    expansion_factor: usize,
    toward_best: &impl Fn(
        &Candidate<Score::Index, Metric::Score>,
        &Candidate<Score::Index, Metric::Score>,
    ) -> bool,
    toward_worst: &impl Fn(
        &Candidate<Score::Index, Metric::Score>,
        &Candidate<Score::Index, Metric::Score>,
    ) -> bool,
) -> Result<(), Error>
where
    Metric: MetricLike + ?Sized,
    Score: SearcherLike<Metric> + ?Sized,
{
    if staged.is_empty() {
        return Ok(());
    }
    scores.clear();
    scores.try_reserve(staged.len()).map_err(|_| Error::Alloc)?;
    scores.resize(staged.len(), Metric::Score::default());
    searcher.score_nodes(query, table, staged, scores)?;

    for (&node, &score) in staged.iter().zip(scores.iter()) {
        if saturated::<Metric, Score>(searcher, score, expansion_factor) {
            continue;
        }
        let candidate = Candidate { node, score };
        searcher.context().frontier.push(candidate, toward_best)?;
        searcher.context().results.push(candidate, toward_worst)?;
        if searcher.context().results.len() > expansion_factor {
            searcher.context().results.pop(toward_worst);
        }
    }
    Ok(())
}

/// True when `score` cannot enter a full result set — the walk's admission and
/// termination test.
fn saturated<Metric, Score>(
    searcher: &mut Score,
    score: Metric::Score,
    expansion_factor: usize,
) -> bool
where
    Metric: MetricLike + ?Sized,
    Score: SearcherLike<Metric> + ?Sized,
{
    let results = &searcher.context().results;
    results.len() >= expansion_factor
        && match results.peek() {
            Some(worst) => Score::compare(score, worst.score) != Ordering::Greater,
            None => false,
        }
}

/// Rank a score tile into `queries × k` matches, row-major and best-first per
/// query, so `out[query * k + rank]` is that query's `rank`-th best node. The
/// shared ranking every [`ScannerLike::scan`] is finished with.
pub fn rank_tile<Index: IndexLike, Score: Copy + Default>(
    scores: &Matrix<Score, impl AllocatorLike + Clone>,
    k: usize,
    compare: impl Fn(Score, Score) -> Ordering,
    out: &mut [Match<Index, Score>],
) -> Result<(), Error> {
    if k == 0 || out.len() != scores.cols() * k {
        return Err(Error::Inconsistent);
    }
    for query in 0..scores.cols() {
        let ranked = &mut out[query * k..query * k + k];
        let mut filled = 0usize;
        for node in 0..scores.rows() {
            let score = scores.get(node, query).ok_or(Error::Inconsistent)?;
            // The node position becomes `I` only when it wins a slot, so a losing
            // node costs no conversion.
            if filled < k {
                let node = Index::try_from(node).map_err(|_| Error::TooLarge)?;
                insert_at(ranked, filled, node, score, &compare);
                filled += 1;
            } else if compare(score, ranked[k - 1].score) == Ordering::Greater {
                let node = Index::try_from(node).map_err(|_| Error::TooLarge)?;
                insert_at(ranked, k - 1, node, score, &compare);
            }
        }
    }
    Ok(())
}

/// Insert `node` with `score` into the sorted prefix `out[..=end]`, shifting
/// weaker entries right so the prefix stays best-first.
fn insert_at<Index: Copy, Score: Copy>(
    out: &mut [Match<Index, Score>],
    end: usize,
    node: Index,
    score: Score,
    compare: &impl Fn(Score, Score) -> Ordering,
) {
    let mut i = end;
    while i > 0 && compare(score, out[i - 1].score) == Ordering::Greater {
        out[i] = out[i - 1];
        i -= 1;
    }
    out[i] = Match { node, score };
}

// The shared batch engine: shaped top-k over any scoring family, brute-force or
// by walk. Backend-free except for ForkUnion, which carries the cross-query
// fan-out, so every family reuses it without importing a sibling's backend.
#[cfg(feature = "search")]
pub(crate) mod batch;

// Each metric family lives in its own module, grouped by how it reads a node's
// value, and shares no code path with any other.
#[cfg(feature = "search")]
pub mod binary;
#[cfg(feature = "search")]
pub mod dense;
#[cfg(feature = "search")]
pub mod geospatial;
#[cfg(feature = "search")]
pub mod maxsim;
#[cfg(feature = "search")]
pub mod mesh;
#[cfg(feature = "search")]
pub mod sequence;

// Runtime dtype dispatch: pick a concrete scorer from a file's header string,
// the counterpart to the compile-time descriptors in `dense`.
#[cfg(feature = "search")]
pub mod dispatch;

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
        let mut context: TraversalContext<u32, f32> = TraversalContext::new(4).unwrap();
        context.reserve(8).unwrap();
        context
            .frontier
            .push(
                Candidate {
                    node: 0,
                    score: 1.0,
                },
                |a, b| a.score < b.score,
            )
            .unwrap();
        assert!(context.visited.visit(3));
        context.reset();
        assert!(context.frontier.is_empty());
        assert!(!context.visited.contains(3));
    }

    #[test]
    fn context_rejects_zero_breadth() {
        assert!(matches!(
            TraversalContext::<u32, f32>::new(0),
            Err(Error::Inconsistent)
        ));
    }

    /// The whole point of carrying the native index: a `u32` node and an `f32`
    /// score is an eight-byte frontier entry, not the sixteen a fixed `u64` cost.
    #[test]
    fn candidate_pack_is_compact() {
        assert_eq!(core::mem::size_of::<Candidate<u32, f32>>(), 8);
        assert_eq!(core::mem::size_of::<Match<u32, f32>>(), 8);
    }

    #[test]
    fn matrix_checks_its_shape() {
        let tile = Matrix::<f32>::new(3, 2).unwrap();
        assert_eq!((tile.rows(), tile.cols()), (3, 2));
        assert_eq!(tile.get(2, 1), Some(0.0));
        assert_eq!(tile.get(3, 0), None);
        assert_eq!(tile.get(0, 2), None);
    }

    /// Ranking is direction-agnostic: the same tile ranked with a reversed
    /// comparator yields the reversed order. This is what lets a smaller-is-better
    /// sequence metric reuse `rank_tile` untouched.
    #[test]
    fn rank_tile_honors_both_directions() {
        // Two queries, three nodes, node-major: node 0 scores [1, 9], node 1
        // [3, 7], node 2 [2, 8]. So query 0 sees [1, 3, 2] and query 1 [9, 7, 8].
        let cells = [1.0f32, 9.0, 3.0, 7.0, 2.0, 8.0];
        let mut tile = Matrix::<f32>::new(3, 2).unwrap();
        tile.as_mut_slice().copy_from_slice(&cells);
        let mut out = [Match {
            node: 0u32,
            score: 0.0f32,
        }; 4];

        let larger = |a: f32, b: f32| a.partial_cmp(&b).unwrap_or(Ordering::Less);
        rank_tile(&tile, 2, larger, &mut out).unwrap();
        assert_eq!(
            out[0],
            Match {
                node: 1,
                score: 3.0
            }
        ); // query 0 best
        assert_eq!(
            out[1],
            Match {
                node: 2,
                score: 2.0
            }
        );
        assert_eq!(
            out[2],
            Match {
                node: 0,
                score: 9.0
            }
        ); // query 1 best
        assert_eq!(
            out[3],
            Match {
                node: 2,
                score: 8.0
            }
        );

        let smaller = |a: f32, b: f32| b.partial_cmp(&a).unwrap_or(Ordering::Less);
        rank_tile(&tile, 2, smaller, &mut out).unwrap();
        assert_eq!(
            out[0],
            Match {
                node: 0,
                score: 1.0
            }
        );
        assert_eq!(
            out[1],
            Match {
                node: 2,
                score: 2.0
            }
        );
        assert_eq!(
            out[2],
            Match {
                node: 1,
                score: 7.0
            }
        );
        assert_eq!(
            out[3],
            Match {
                node: 2,
                score: 8.0
            }
        );
    }

    /// The three backend surfaces the two metric families are built on must all
    /// resolve, and — the part that silently breaks — the `stringtape` types we
    /// hand StringZilla must come from the same crate StringZilla itself was
    /// compiled against. If the dependency graph ever splits again, this stops
    /// compiling rather than failing much later at the zero-copy handoff.
    #[cfg(feature = "search")]
    #[test]
    fn backend_surfaces_resolve() {
        use forkunion as fu;
        use stringzilla::szs;

        // NumKong's pool-taking kernels exist only under its `parallel` feature.
        let topology = fu::Topology::new().expect("probe machine topology");
        let pool = fu::ThreadPool::try_spawn(&topology, 2).expect("spawn workers");
        assert!(pool.threads_count() >= 1);

        // StringZilla's batch engines exist only under its `cpus` feature.
        let device = szs::DeviceScope::cpu_cores(1).expect("cpu device");
        assert!(!device.is_gpu());

        // A score matrix we allocate ourselves, which needs `try_allocate` to be
        // public — otherwise `compute_into` can only reuse a matrix `compute`
        // returned, and no buffer can be sized ahead of time.
        let scores = szs::UnifiedMat::<usize>::try_allocate(1, 3).expect("unified matrix");
        assert_eq!(scores.dimensions(), (1, 3));

        // The Arrow-shaped handoff: our mmap'd offsets and bytes, borrowed with
        // no copy, in the exact type `AnyBytesTape` carries.
        let bytes = b"alphabetagamma";
        let offsets: [u32; 4] = [0, 5, 9, 14];
        let tape = unsafe { stringtape::BytesTapeView::<u32>::from_raw_parts(bytes, &offsets) };
        assert_eq!(tape.len(), 3);
        assert_eq!(tape.get(1), Some(&b"beta"[..]));
        let _: szs::AnyBytesTape = szs::AnyBytesTape::View32(tape);
    }
}

//! Sequence metrics over StringZilla. No NumKong type appears here.
//!
//! A node's value is a variable-length byte string — DNA, RNA, protein, or text —
//! and a query batch is a tape of them rather than a matrix, so nothing in this
//! family has a width. The edit distances rank smaller-first and the alignment
//! scores larger-first, and the machinery absorbs both directions through the
//! engine's own comparator and nothing else.
//!
//! Each functor opens its [`ExecutionScope`] as a [`DeviceScope`], StringZilla's
//! executor, and that conversion is the only place the backend's device type
//! appears: `Serial` and `Cpus` run the very same engines on CPU threads, `Gpu`
//! on a GPU. The scope is also why this family builds no ForkUnion pool — a
//! StringZilla engine already drives every worker the scope names from one batch
//! call, so the cross-query fan-out is [`batch_search_serial`], whole walks on a
//! single traversal context, and the parallelism lives inside each scoring call.
//!
//! Each metric is its own scanner/searcher pair rather than an arm of a dtype
//! dispatcher: every sequence value is `utf8` bytes, so there is no dtype axis
//! to erase, and the four metrics — byte-level Levenshtein, its codepoint-level
//! UTF-8 variant, and the Needleman-Wunsch and Smith-Waterman alignment scores —
//! are simply different [`EngineLike`] instantiations of one generic body, the
//! way `DotMetric<Value>` spans the dense scalars. The cross-family router in
//! `dispatch` picks between them.
//!
//! [`DeviceScope`]: stringzilla::szs::DeviceScope

use super::batch::{batch_scan, batch_search_serial, build_workers, Workers};
use super::{
    walk, ExecutionScope, Matrix, MetricLike, QueryBatchLike, Ranking, ScannerLike, SearcherLike,
    TraversalContext,
};
use crate::{Error, GraphLike, IndexLike, OffsetDType, Table, Widen};
use allocator_api2::alloc::{Allocator as AllocatorLike, Global};
use core::cmp::Ordering;
use stringtape::{BytesTapeView, CharsTapeView};
use stringzilla::szs;

// region: Engine seam

/// The seam between the shared sequence machinery and one StringZilla engine:
/// its native score, its configuration, its ranking direction, and how to build
/// it and score one byte-tape pair. Implemented for the `szs` engine types
/// themselves, so [`SequenceScanner<Engine>`] and friends span every metric the way
/// `DotMetric<Value>` spans the dense scalars — one generic body, no macro.
pub trait EngineLike: Sized {
    /// The engine's native score: `usize` for an edit distance, `isize` for an
    /// alignment score, which may be negative.
    type Score: Copy + Default + Ord + Widen;

    /// The engine's configuration — costs, substitution classes — everything
    /// its constructor needs beyond the device.
    type Costs: Clone;

    /// True when smaller ranks first: an edit distance. An alignment score
    /// ranks the other way.
    /// Which end of the score axis wins for this engine — [`Ranking::Nearest`]
    /// for an edit distance, [`Ranking::Farthest`] for a similarity.
    const RANKING: Ranking;

    /// Build the engine on `device` with `costs`.
    fn build(device: &szs::DeviceScope, costs: &Self::Costs) -> Result<Self, Error>;

    /// Score every (query, candidate) pair into `matrix`, query-major. Both
    /// tapes carry the same offset width; the callers guarantee it.
    fn compute_into(
        &self,
        device: &szs::DeviceScope,
        queries: szs::AnyBytesTape<'_>,
        candidates: szs::AnyBytesTape<'_>,
        matrix: &mut szs::UnifiedMat<Self::Score>,
    ) -> Result<(), Error>;
}

/// Per-edit costs for the two Levenshtein engines. The default is unit cost —
/// every edit one — and a caller scoring biological sequences, where a gap and
/// a substitution are not worth the same, names its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevenshteinCosts {
    pub match_cost: i8,
    pub mismatch_cost: i8,
    pub open_cost: i8,
    pub extend_cost: i8,
}

impl Default for LevenshteinCosts {
    fn default() -> Self {
        LevenshteinCosts {
            match_cost: 0,
            mismatch_cost: 1,
            open_cost: 1,
            extend_cost: 1,
        }
    }
}

/// Costs for the two alignment engines: a 256-entry map from byte to one of 32
/// character classes, a 32×32 substitution-score matrix between classes, and
/// affine gap penalties. The compact class form is StringZilla's own — DNA,
/// RNA, and protein alphabets all fit 32 classes with room to spare.
///
/// The default maps each byte to its value modulo 32 and scores +2 for a class
/// match, -1 for a mismatch, -2/-1 for gap open/extend. Bytes 32 apart share a
/// class under it — an ASCII letter matches its other case — so it is a demo
/// alphabet, not a substitution model; real work names its own.
#[derive(Clone)]
pub struct AlignmentCosts {
    pub byte_to_class: [u8; 256],
    pub class_substitution_costs: [[i8; 32]; 32],
    pub open_cost: i8,
    pub extend_cost: i8,
}

impl Default for AlignmentCosts {
    fn default() -> Self {
        let mut byte_to_class = [0u8; 256];
        for (byte, class) in byte_to_class.iter_mut().enumerate() {
            *class = (byte % 32) as u8;
        }
        let mut class_substitution_costs = [[-1i8; 32]; 32];
        for (class, row) in class_substitution_costs.iter_mut().enumerate() {
            row[class] = 2;
        }
        AlignmentCosts {
            byte_to_class,
            class_substitution_costs,
            open_cost: -2,
            extend_cost: -1,
        }
    }
}

impl EngineLike for szs::LevenshteinDistances {
    type Score = usize;
    type Costs = LevenshteinCosts;
    const RANKING: Ranking = Ranking::Nearest;

    fn build(device: &szs::DeviceScope, costs: &LevenshteinCosts) -> Result<Self, Error> {
        szs::LevenshteinDistances::new(
            device,
            costs.match_cost,
            costs.mismatch_cost,
            costs.open_cost,
            costs.extend_cost,
        )
        .map_err(|_| Error::Alloc)
    }

    fn compute_into(
        &self,
        device: &szs::DeviceScope,
        queries: szs::AnyBytesTape<'_>,
        candidates: szs::AnyBytesTape<'_>,
        matrix: &mut szs::UnifiedMat<usize>,
    ) -> Result<(), Error> {
        self.compute_into(device, queries, Some(candidates), matrix)
            .map_err(|_| Error::Inconsistent)
    }
}

impl EngineLike for szs::LevenshteinDistancesUtf8 {
    type Score = usize;
    type Costs = LevenshteinCosts;
    const RANKING: Ranking = Ranking::Nearest;

    fn build(device: &szs::DeviceScope, costs: &LevenshteinCosts) -> Result<Self, Error> {
        szs::LevenshteinDistancesUtf8::new(
            device,
            costs.match_cost,
            costs.mismatch_cost,
            costs.open_cost,
            costs.extend_cost,
        )
        .map_err(|_| Error::Alloc)
    }

    /// The one engine whose tapes are codepoints rather than bytes: both sides
    /// are validated as UTF-8 and re-viewed as character tapes, offsets reused
    /// as-is since they index bytes either way. Invalid UTF-8 anywhere is
    /// [`Error::WrongType`] — the column claimed `utf8` and lied.
    fn compute_into(
        &self,
        device: &szs::DeviceScope,
        queries: szs::AnyBytesTape<'_>,
        candidates: szs::AnyBytesTape<'_>,
        matrix: &mut szs::UnifiedMat<usize>,
    ) -> Result<(), Error> {
        let queries = chars_tape(&queries)?;
        let candidates = chars_tape(&candidates)?;
        self.compute_into(device, queries, Some(candidates), matrix)
            .map_err(|_| Error::Inconsistent)
    }
}

impl EngineLike for szs::NeedlemanWunschScores {
    type Score = isize;
    type Costs = AlignmentCosts;
    const RANKING: Ranking = Ranking::Farthest;

    fn build(device: &szs::DeviceScope, costs: &AlignmentCosts) -> Result<Self, Error> {
        szs::NeedlemanWunschScores::new(
            device,
            &costs.byte_to_class,
            &costs.class_substitution_costs,
            costs.open_cost,
            costs.extend_cost,
        )
        .map_err(|_| Error::Alloc)
    }

    fn compute_into(
        &self,
        device: &szs::DeviceScope,
        queries: szs::AnyBytesTape<'_>,
        candidates: szs::AnyBytesTape<'_>,
        matrix: &mut szs::UnifiedMat<isize>,
    ) -> Result<(), Error> {
        self.compute_into(device, queries, Some(candidates), matrix)
            .map_err(|_| Error::Inconsistent)
    }
}

impl EngineLike for szs::SmithWatermanScores {
    type Score = isize;
    type Costs = AlignmentCosts;
    const RANKING: Ranking = Ranking::Farthest;

    fn build(device: &szs::DeviceScope, costs: &AlignmentCosts) -> Result<Self, Error> {
        szs::SmithWatermanScores::new(
            device,
            &costs.byte_to_class,
            &costs.class_substitution_costs,
            costs.open_cost,
            costs.extend_cost,
        )
        .map_err(|_| Error::Alloc)
    }

    fn compute_into(
        &self,
        device: &szs::DeviceScope,
        queries: szs::AnyBytesTape<'_>,
        candidates: szs::AnyBytesTape<'_>,
        matrix: &mut szs::UnifiedMat<isize>,
    ) -> Result<(), Error> {
        self.compute_into(device, queries, Some(candidates), matrix)
            .map_err(|_| Error::Inconsistent)
    }
}

// endregion: Engine seam

// region: Tape plumbing

/// Open an [`ExecutionScope`] as the [`szs::DeviceScope`] every StringZilla
/// engine executes on — the sequence twin of the dense families' `open_pool`,
/// and the one place the device type is named. `Cpus(0)` means every logical CPU, by
/// StringZilla's own convention, matching the pool's.
fn open_device(scope: ExecutionScope) -> Result<szs::DeviceScope, Error> {
    match scope {
        ExecutionScope::Serial => szs::DeviceScope::cpu_cores(1),
        ExecutionScope::Cpus(cpus) => szs::DeviceScope::cpu_cores(cpus),
        ExecutionScope::Gpu(index) => szs::DeviceScope::gpu_device(index),
    }
    .map_err(|_| Error::Alloc)
}

/// Borrow a table's variable-length value column as a tape, with no copy: the
/// prefix sums and the concatenated bytes are already exactly what StringZilla
/// wants. Fails only on a regular column, which has no offsets to borrow.
fn value_tape(table: &Table<impl AllocatorLike>) -> Result<szs::AnyBytesTape<'_>, Error> {
    let (offset_bytes, dtype) = table.value_offsets().ok_or(Error::WrongType)?;
    let values = table.value_buffer();
    let expected = table.len() + 1;
    // SAFETY, in both arms: the offsets are a validated prefix sum over the value
    // buffer, and both are borrowed from the same table for `'_`.
    Ok(match dtype {
        OffsetDType::U32 => {
            let offsets: &[u32] = crate::cast(offset_bytes).ok_or(Error::WrongType)?;
            if offsets.len() != expected {
                return Err(Error::Inconsistent);
            }
            szs::AnyBytesTape::View32(unsafe {
                BytesTapeView::<u32>::from_raw_parts(values, offsets)
            })
        }
        OffsetDType::U64 => {
            let offsets: &[u64] = crate::cast(offset_bytes).ok_or(Error::WrongType)?;
            if offsets.len() != expected {
                return Err(Error::Inconsistent);
            }
            szs::AnyBytesTape::View64(unsafe {
                BytesTapeView::<u64>::from_raw_parts(values, offsets)
            })
        }
    })
}

/// How many strings a tape holds, whichever offset width it carries.
fn tape_len(tape: &szs::AnyBytesTape<'_>) -> usize {
    match tape {
        szs::AnyBytesTape::View32(view) => view.len(),
        szs::AnyBytesTape::View64(view) => view.len(),
        szs::AnyBytesTape::Tape32(tape) => tape.len(),
        szs::AnyBytesTape::Tape64(tape) => tape.len(),
    }
}

/// A tape's borrowed raw parts, whichever offset width and ownership it carries,
/// for rebuilding it at the other width or as a character tape.
enum TapeParts<'t> {
    Narrow { bytes: &'t [u8], offsets: &'t [u32] },
    Wide { bytes: &'t [u8], offsets: &'t [u64] },
}

/// Borrow `tape`'s bytes and prefix sums. An empty offsets buffer is refused:
/// even a zero-string tape carries the leading zero.
fn tape_parts<'t>(tape: &'t szs::AnyBytesTape<'_>) -> Result<TapeParts<'t>, Error> {
    fn slices<'t, Offset>(
        parts: stringtape::RawParts<Offset>,
    ) -> Result<(&'t [u8], &'t [Offset]), Error>
    where
        Offset: stringtape::OffsetType,
    {
        if parts.offsets_ptr.is_null() {
            return Err(Error::Inconsistent);
        }
        // SAFETY: the parts came off a tape borrowed for `'t`, whose invariants
        // promise `data_len` valid bytes and `items_count + 1` offsets.
        Ok(unsafe {
            (
                if parts.data_ptr.is_null() {
                    &[]
                } else {
                    core::slice::from_raw_parts(parts.data_ptr, parts.data_len)
                },
                core::slice::from_raw_parts(parts.offsets_ptr, parts.items_count + 1),
            )
        })
    }
    Ok(match tape {
        szs::AnyBytesTape::View32(view) => {
            let (bytes, offsets) = slices(view.as_raw_parts())?;
            TapeParts::Narrow { bytes, offsets }
        }
        szs::AnyBytesTape::Tape32(tape) => {
            let (bytes, offsets) = slices(tape.as_raw_parts())?;
            TapeParts::Narrow { bytes, offsets }
        }
        szs::AnyBytesTape::View64(view) => {
            let (bytes, offsets) = slices(view.as_raw_parts())?;
            TapeParts::Wide { bytes, offsets }
        }
        szs::AnyBytesTape::Tape64(tape) => {
            let (bytes, offsets) = slices(tape.as_raw_parts())?;
            TapeParts::Wide { bytes, offsets }
        }
    })
}

/// Re-view a byte tape as a character tape for the UTF-8 engine, validating the
/// bytes first — the offsets carry over untouched, since they index bytes in
/// both views. Validation is per call and linear in the bytes, which the
/// quadratic alignment behind it dwarfs.
fn chars_tape<'t>(tape: &'t szs::AnyBytesTape<'_>) -> Result<szs::AnyCharsTape<'t>, Error> {
    Ok(match tape_parts(tape)? {
        TapeParts::Narrow { bytes, offsets } => {
            core::str::from_utf8(bytes).map_err(|_| Error::WrongType)?;
            // SAFETY: same buffer and prefix sum as the byte view, now proven UTF-8.
            szs::AnyCharsTape::View32(unsafe {
                CharsTapeView::<u32>::from_raw_parts(bytes, offsets)
            })
        }
        TapeParts::Wide { bytes, offsets } => {
            core::str::from_utf8(bytes).map_err(|_| Error::WrongType)?;
            // SAFETY: as above.
            szs::AnyCharsTape::View64(unsafe {
                CharsTapeView::<u64>::from_raw_parts(bytes, offsets)
            })
        }
    })
}

/// A score matrix of exactly this shape, reusing `slot`'s allocation when the
/// shape is unchanged — the engines reject any other size.
fn matrix_for<Score>(
    slot: &mut Option<szs::UnifiedMat<Score>>,
    queries: usize,
    nodes: usize,
) -> Result<&mut szs::UnifiedMat<Score>, Error>
where
    Score: Copy + Default,
{
    let fits = matches!(&*slot, Some(matrix) if matrix.dimensions() == (queries, nodes));
    if !fits {
        *slot =
            Some(szs::UnifiedMat::<Score>::try_allocate(queries, nodes).map_err(|_| Error::Alloc)?);
    }
    slot.as_mut().ok_or(Error::Alloc)
}

// endregion: Tape plumbing

// region: Generic functors

/// A sequence metric's type vocabulary, one marker per engine — the sequence
/// twin of `DotMetric<Value>`.
pub struct SequenceMetric<Engine>(core::marker::PhantomData<Engine>);

impl<Engine: EngineLike> MetricLike for SequenceMetric<Engine> {
    type Query<'a> = &'a [u8];
    type Queries<'a> = szs::AnyBytesTape<'a>;
    type Score = Engine::Score;
}

/// A byte tape as a [`QueryBatchLike`]: its string count is the query count, and
/// each string is one query. The seam that lets the shared batch engine walk a
/// sequence tape exactly as it walks a dense matrix.
impl<Metric> QueryBatchLike<Metric> for szs::AnyBytesTape<'_>
where
    Metric: MetricLike + ?Sized,
    for<'a> Metric::Query<'a>: From<&'a [u8]>,
{
    fn count(&self) -> usize {
        tape_len(self)
    }

    fn query(&self, index: usize) -> Option<Metric::Query<'_>> {
        let bytes = match self {
            szs::AnyBytesTape::View32(view) => view.get(index),
            szs::AnyBytesTape::View64(view) => view.get(index),
            szs::AnyBytesTape::Tape32(tape) => tape.get(index),
            szs::AnyBytesTape::Tape64(tape) => tape.get(index),
        }?;
        Some(bytes.into())
    }
}

/// The batch half of any sequence metric, owning its device and engine.
pub struct SequenceScanner<Engine: EngineLike> {
    device: szs::DeviceScope,
    engine: Engine,
    /// Unified memory the engine writes into, reallocated only when the batch
    /// shape changes. It cannot be the crate's tile: the engine owns this
    /// layout, and it is query-major where the tile is node-major.
    matrix: Option<szs::UnifiedMat<Engine::Score>>,
    /// The crate-side node-major score tile, reused across batches.
    tile: Matrix<Engine::Score>,
    /// Offset scratch for restating a query tape at the table's own offset
    /// width, since the engines refuse a mixed-width pair. Only the prefix sums
    /// are rebuilt; the query bytes are never copied.
    offsets_narrow: Vec<u32>,
    offsets_wide: Vec<u64>,
}

impl<Engine: EngineLike> SequenceScanner<Engine> {
    /// A scanner with `costs`, running on whatever `scope` names — the same
    /// engine serves CPU cores and a GPU alike, which is why the device is
    /// owned here rather than exposed. No allocator crosses this signature:
    /// the engine's score matrix lives in StringZilla's own unified memory,
    /// which is not an `allocator_api2` allocator.
    pub fn new(costs: Engine::Costs, scope: ExecutionScope) -> Result<Self, Error> {
        let device = open_device(scope)?;
        let engine = Engine::build(&device, &costs)?;
        Ok(SequenceScanner {
            device,
            engine,
            matrix: None,
            tile: Matrix::empty(),
            offsets_narrow: Vec::new(),
            offsets_wide: Vec::new(),
        })
    }

    /// Batch brute-force top-k, best-first in the engine's own direction —
    /// nearest for an edit distance, highest for an alignment score.
    /// `matches`/`distances` are `queries × k`; their shape carries both `k`
    /// and the query count, uniform with every dense scanner.
    pub fn scan(
        &mut self,
        queries: szs::AnyBytesTape<'_>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let mut tile = std::mem::take(&mut self.tile);
        let outcome = batch_scan::<SequenceMetric<Engine>, _>(
            self,
            queries,
            table,
            Engine::RANKING,
            &mut tile,
            matches,
            distances,
            counts,
        );
        self.tile = tile;
        outcome
    }
}

impl<Engine: EngineLike> ScannerLike<SequenceMetric<Engine>> for SequenceScanner<Engine> {
    fn scan(
        &mut self,
        queries: szs::AnyBytesTape<'_>,
        table: &Table<impl AllocatorLike>,
        scores: &mut Matrix<Engine::Score>,
    ) -> Result<(), Error> {
        let candidates = value_tape(table)?;
        let (query_count, node_count) = (tape_len(&queries), tape_len(&candidates));
        if query_count != scores.cols() || node_count != scores.rows() {
            return Err(Error::Inconsistent);
        }

        // The engines refuse a mixed-width pair, and the table's width is the
        // file's to pick — so when the query tape disagrees, its prefix sums are
        // restated at the table's width in owned scratch, the bytes untouched.
        // A caller never has to know which width a file chose.
        let table_wide = matches!(candidates, szs::AnyBytesTape::View64(_));
        let query_wide = matches!(
            queries,
            szs::AnyBytesTape::View64(_) | szs::AnyBytesTape::Tape64(_)
        );
        let queries = if table_wide == query_wide {
            queries
        } else if table_wide {
            let TapeParts::Narrow { bytes, offsets } = tape_parts(&queries)? else {
                return Err(Error::Inconsistent);
            };
            self.offsets_wide.clear();
            self.offsets_wide
                .try_reserve_exact(offsets.len())
                .map_err(|_| Error::Alloc)?;
            self.offsets_wide
                .extend(offsets.iter().map(|&offset| offset as u64));
            // SAFETY: the widened offsets are the same validated prefix sum over
            // the same bytes, and both outlive the compute call below.
            szs::AnyBytesTape::View64(unsafe {
                BytesTapeView::<u64>::from_raw_parts(bytes, &self.offsets_wide)
            })
        } else {
            let TapeParts::Wide { bytes, offsets } = tape_parts(&queries)? else {
                return Err(Error::Inconsistent);
            };
            self.offsets_narrow.clear();
            self.offsets_narrow
                .try_reserve_exact(offsets.len())
                .map_err(|_| Error::Alloc)?;
            for &offset in offsets {
                self.offsets_narrow
                    .push(u32::try_from(offset).map_err(|_| Error::TooLarge)?);
            }
            // SAFETY: as above — narrowing is checked, so the prefix sum survives.
            szs::AnyBytesTape::View32(unsafe {
                BytesTapeView::<u32>::from_raw_parts(bytes, &self.offsets_narrow)
            })
        };

        let matrix = matrix_for(&mut self.matrix, query_count, node_count)?;
        self.engine
            .compute_into(&self.device, queries, candidates, matrix)?;

        // The engine's matrix is query-major and the tile is node-major, so this
        // copy transposes. It is the one pass the crate-owned tile costs this
        // family, and it is cheap beside the alignment work that produced it.
        for query in 0..query_count {
            for node in 0..node_count {
                *scores.get_mut(node, query).ok_or(Error::Inconsistent)? = matrix[(query, node)];
            }
        }
        Ok(())
    }
}

/// The traversal half of any sequence metric, owning the reusable scratch.
///
/// It owns a device and engine of its own because StringZilla ships no
/// single-pair scoring function — scoring even one neighbor is a batch call — so
/// a searcher cannot borrow a scanner's.
pub struct SequenceSearcher<Engine: EngineLike, Index = u32, Allocator: AllocatorLike = Global> {
    device: szs::DeviceScope,
    engine: Engine,
    context: TraversalContext<Index, Engine::Score, Allocator>,
    /// One expansion's node values, gathered contiguously with their prefix sums
    /// so they can be handed over as a tape. Reused between expansions; the
    /// gather itself is unavoidable, since staged nodes are scattered.
    staged_bytes: Vec<u8>,
    staged_offsets: Vec<u32>,
    matrix: Option<szs::UnifiedMat<Engine::Score>>,
}

impl<Engine: EngineLike, Index: IndexLike, Allocator: AllocatorLike + Clone>
    SequenceSearcher<Engine, Index, Allocator>
{
    /// A searcher with `costs`, walking with breadth `expansion_factor` on
    /// whatever `scope` names, its traversal scratch in `alloc`. It carries no
    /// table: the scratch grows to fit each searched graph at walk time.
    pub fn new(
        costs: Engine::Costs,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        let device = open_device(scope)?;
        let engine = Engine::build(&device, &costs)?;
        Ok(SequenceSearcher {
            device,
            engine,
            context: TraversalContext::new_in(expansion_factor, alloc)?,
            staged_bytes: Vec::new(),
            staged_offsets: Vec::new(),
            matrix: None,
        })
    }

    /// The traversal scratch, for draining results after a walk.
    pub fn context(&mut self) -> &mut TraversalContext<Index, Engine::Score, Allocator> {
        &mut self.context
    }
}

impl<Engine: EngineLike, Index: IndexLike, Allocator: AllocatorLike + Clone>
    SearcherLike<SequenceMetric<Engine>> for SequenceSearcher<Engine, Index, Allocator>
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        query: &[u8],
        table: &Table<impl AllocatorLike>,
        nodes: &[Index],
        out: &mut [Engine::Score],
    ) -> Result<(), Error> {
        if nodes.len() != out.len() {
            return Err(Error::Inconsistent);
        }
        if nodes.is_empty() {
            return Ok(());
        }

        // Gather the scattered node values into one contiguous tape. There is no
        // avoiding the copy — staged nodes are arbitrary indices, not a run — but
        // the buffers are reused, so a steady state does not allocate.
        self.staged_bytes.clear();
        self.staged_offsets.clear();
        self.staged_offsets.push(0);
        for &node in nodes {
            let index: usize = node.try_into().map_err(|_| Error::TooLarge)?;
            self.staged_bytes.extend_from_slice(table.value(index));
            let end = u32::try_from(self.staged_bytes.len()).map_err(|_| Error::TooLarge)?;
            self.staged_offsets.push(end);
        }

        let query_offsets: [u32; 2] = [0, u32::try_from(query.len()).map_err(|_| Error::TooLarge)?];
        // SAFETY: each pair is a prefix sum over the buffer beside it, and both
        // outlive the call below.
        let query_tape = unsafe { BytesTapeView::<u32>::from_raw_parts(query, &query_offsets) };
        let staged_tape = unsafe {
            BytesTapeView::<u32>::from_raw_parts(&self.staged_bytes, &self.staged_offsets)
        };

        let matrix = matrix_for(&mut self.matrix, 1, nodes.len())?;
        self.engine.compute_into(
            &self.device,
            szs::AnyBytesTape::View32(query_tape),
            szs::AnyBytesTape::View32(staged_tape),
            matrix,
        )?;

        for (node, slot) in out.iter_mut().enumerate() {
            *slot = matrix[(0, node)];
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, Engine::Score, Allocator> {
        &mut self.context
    }

    fn compare(a: Engine::Score, b: Engine::Score) -> Ordering {
        if matches!(Engine::RANKING, Ranking::Nearest) {
            b.cmp(&a)
        } else {
            a.cmp(&b)
        }
    }

    fn search(&mut self, query: &[u8], table: &Table<impl AllocatorLike>) -> Result<(), Error> {
        walk::<SequenceMetric<Engine>, Self>(self, query, table)
    }
}

/// A sequence searcher whose neighbor width is settled by the tables it
/// searches, not named by the caller — the sequence twin of the dense families'
/// width resolvers. Its lanes hold a single leaf rather than one per pool
/// thread, because this family fans out inside each StringZilla scoring call,
/// not across queries; the batch path is [`batch_search_serial`].
pub struct SequenceSearch<Engine: EngineLike, Allocator: AllocatorLike = Global> {
    costs: Engine::Costs,
    expansion_factor: usize,
    scope: ExecutionScope,
    workers:
        Workers<SequenceSearcher<Engine, u32, Allocator>, SequenceSearcher<Engine, u64, Allocator>>,
    alloc: Allocator,
}

impl<Engine: EngineLike, Allocator: AllocatorLike + Clone> SequenceSearch<Engine, Allocator> {
    /// A searcher with `costs`, walking with breadth `expansion_factor` on
    /// whatever `scope` names, its scratch in `alloc`. No table is seen here:
    /// the neighbor-width lane settles on the first searched table.
    pub fn new(
        costs: Engine::Costs,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(SequenceSearch {
            costs,
            expansion_factor,
            scope,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, uniform with [`SequenceScanner::scan`].
    pub fn search(
        &mut self,
        queries: szs::AnyBytesTape<'_>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let (costs, expansion_factor, scope) = (&self.costs, self.expansion_factor, self.scope);
        match table.neighbors_column().dtype {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(1, || {
                        SequenceSearcher::new(costs.clone(), expansion_factor, scope, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                let worker = workers.first_mut().ok_or(Error::Inconsistent)?;
                batch_search_serial::<SequenceMetric<Engine>, _>(
                    worker, queries, table, matches, distances, counts,
                )
            }
            OffsetDType::U64 => {
                if !matches!(self.workers, Workers::U64(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U64(build_workers(1, || {
                        SequenceSearcher::new(costs.clone(), expansion_factor, scope, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                let worker = workers.first_mut().ok_or(Error::Inconsistent)?;
                batch_search_serial::<SequenceMetric<Engine>, _>(
                    worker, queries, table, matches, distances, counts,
                )
            }
        }
    }
}

// endregion: Generic functors

// region: Metric aliases

/// Byte-level Levenshtein edit distance, smaller-is-better.
pub type LevenshteinMetric = SequenceMetric<szs::LevenshteinDistances>;
pub type LevenshteinScanner = SequenceScanner<szs::LevenshteinDistances>;
pub type LevenshteinSearcher<Index = u32, Allocator = Global> =
    SequenceSearcher<szs::LevenshteinDistances, Index, Allocator>;
pub type LevenshteinSearch<Allocator = Global> =
    SequenceSearch<szs::LevenshteinDistances, Allocator>;

/// Codepoint-level Levenshtein over validated UTF-8, smaller-is-better: one
/// multibyte character is one edit, not two.
pub type LevenshteinUtf8Metric = SequenceMetric<szs::LevenshteinDistancesUtf8>;
pub type LevenshteinUtf8Scanner = SequenceScanner<szs::LevenshteinDistancesUtf8>;
pub type LevenshteinUtf8Searcher<Index = u32, Allocator = Global> =
    SequenceSearcher<szs::LevenshteinDistancesUtf8, Index, Allocator>;
pub type LevenshteinUtf8Search<Allocator = Global> =
    SequenceSearch<szs::LevenshteinDistancesUtf8, Allocator>;

/// Needleman-Wunsch global alignment score, larger-is-better and possibly
/// negative.
pub type NeedlemanWunschMetric = SequenceMetric<szs::NeedlemanWunschScores>;
pub type NeedlemanWunschScanner = SequenceScanner<szs::NeedlemanWunschScores>;
pub type NeedlemanWunschSearcher<Index = u32, Allocator = Global> =
    SequenceSearcher<szs::NeedlemanWunschScores, Index, Allocator>;
pub type NeedlemanWunschSearch<Allocator = Global> =
    SequenceSearch<szs::NeedlemanWunschScores, Allocator>;

/// Smith-Waterman local alignment score, larger-is-better and never negative.
pub type SmithWatermanMetric = SequenceMetric<szs::SmithWatermanScores>;
pub type SmithWatermanScanner = SequenceScanner<szs::SmithWatermanScores>;
pub type SmithWatermanSearcher<Index = u32, Allocator = Global> =
    SequenceSearcher<szs::SmithWatermanScores, Index, Allocator>;
pub type SmithWatermanSearch<Allocator = Global> =
    SequenceSearch<szs::SmithWatermanScores, Allocator>;

// endregion: Metric aliases

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NeighborList;

    /// Three words on a fully connected triangle. Values are variable-length, so
    /// the column is irregular and has no `value_shape` at all — the property no
    /// dense-shaped API could express.
    struct Words;

    const WORDS: [&[u8]; 3] = [b"kitten", b"sitting", b"mitten"];

    impl GraphLike for Words {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "utf8"
        }
        fn value_shape(&self) -> Option<usize> {
            None
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
            WORDS[i]
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

    /// One-query tape over `bytes`, deliberately at the narrow offset width:
    /// `Table::build` writes `u64` offsets, so every scan below also proves the
    /// scanner restates the width itself rather than pushing it on the caller.
    fn query_tape<'q>(bytes: &'q [u8], offsets: &'q mut [u32; 2]) -> szs::AnyBytesTape<'q> {
        *offsets = [0, u32::try_from(bytes.len()).unwrap()];
        szs::AnyBytesTape::View32(unsafe { BytesTapeView::<u32>::from_raw_parts(bytes, offsets) })
    }

    /// `queries × k` outputs plus counts, the shaped-batch boilerplate.
    fn outputs(queries: usize, k: usize) -> (Matrix<u64>, Matrix<f64>, Vec<u64>) {
        (
            Matrix::<u64>::new(queries, k).unwrap(),
            Matrix::<f64>::new(queries, k).unwrap(),
            vec![0u64; queries],
        )
    }

    #[test]
    fn scan_ranks_nearest_first() {
        let table = crate::Table::build(&Words, None).unwrap();
        let mut scanner =
            LevenshteinScanner::new(LevenshteinCosts::default(), ExecutionScope::Serial).unwrap();

        // "kitten": distance 0 to itself, 3 to "sitting", 1 to "mitten" — and the
        // ranking must put the SMALLEST distance first, unlike every dot metric.
        let mut offsets = [0u32; 2];
        let queries = query_tape(b"kitten", &mut offsets);
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
        scanner
            .scan(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[0, 2, 1]);
        assert_eq!(distances.row(0).unwrap(), &[0.0, 1.0, 3.0]);
    }

    #[test]
    fn walk_agrees_with_scan() {
        // The batched-frontier walk must reach the same ordering as a full scan,
        // with a metric whose scores run the opposite direction to the dense one.
        let table = crate::Table::build(&Words, None).unwrap();
        let mut searcher = LevenshteinSearch::new(
            LevenshteinCosts::default(),
            3,
            ExecutionScope::Serial,
            Global,
        )
        .unwrap();

        let mut offsets = [0u32; 2];
        let queries = query_tape(b"kitten", &mut offsets);
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
        searcher
            .search(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[0, 2, 1]);
        assert_eq!(distances.row(0).unwrap(), &[0.0, 1.0, 3.0]);
    }

    #[test]
    fn batch_walks_many_queries() {
        // A two-query tape through one searcher: the tape carries the count, each
        // query fills its own row, and the rows disagree — proof the loop indexes
        // the tape rather than rescoring the first string.
        let table = crate::Table::build(&Words, None).unwrap();
        let mut searcher = LevenshteinSearch::new(
            LevenshteinCosts::default(),
            3,
            ExecutionScope::Serial,
            Global,
        )
        .unwrap();

        let bytes = b"kittenmitten";
        let offsets: [u32; 3] = [0, 6, 12];
        let queries = szs::AnyBytesTape::View32(unsafe {
            BytesTapeView::<u32>::from_raw_parts(bytes, &offsets)
        });
        let (mut matches, mut distances, mut counts) = outputs(2, 2);
        searcher
            .search(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(counts, [2, 2]);
        // "kitten" sits on node 0, one edit from "mitten"; "mitten" the reverse.
        assert_eq!(matches.row(0).unwrap(), &[0, 2]);
        assert_eq!(distances.row(0).unwrap(), &[0.0, 1.0]);
        assert_eq!(matches.row(1).unwrap(), &[2, 0]);
        assert_eq!(distances.row(1).unwrap(), &[0.0, 1.0]);
    }

    #[test]
    fn needleman_wunsch_ranks_highest_score_first() {
        // The alignment scores flip the direction: the LARGEST score is the best
        // match, exercised through the same generic body that ranks Levenshtein
        // smallest-first. The self-alignment of "kitten" under the default costs
        // is unambiguous — six class matches, no gaps — so its score is pinned.
        let table = crate::Table::build(&Words, None).unwrap();
        let mut scanner =
            NeedlemanWunschScanner::new(AlignmentCosts::default(), ExecutionScope::Serial).unwrap();

        let mut offsets = [0u32; 2];
        let queries = query_tape(b"kitten", &mut offsets);
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
        scanner
            .scan(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[0, 2, 1]);
        assert_eq!(distances.row(0).unwrap()[0], 12.0);

        // The walk agrees with the scan, ordering and scores both.
        let mut searcher = NeedlemanWunschSearch::new(
            AlignmentCosts::default(),
            3,
            ExecutionScope::Serial,
            Global,
        )
        .unwrap();
        let mut offsets = [0u32; 2];
        let queries = query_tape(b"kitten", &mut offsets);
        let (mut walked, mut walked_scores, mut walked_counts) = outputs(1, 3);
        searcher
            .search(
                queries,
                &table,
                &mut walked,
                &mut walked_scores,
                &mut walked_counts,
            )
            .unwrap();
        assert_eq!(walked.row(0).unwrap(), matches.row(0).unwrap());
        assert_eq!(walked_scores.row(0).unwrap(), distances.row(0).unwrap());
    }

    #[test]
    fn smith_waterman_scores_local_alignment() {
        // Local alignment: "kitt" scores its full length against "kitten" alone
        // — eight, four class matches — while the other two words only contain
        // the shorter run "itt", so node 0 outranks them and ties keep node
        // order. A local score never goes negative.
        let table = crate::Table::build(&Words, None).unwrap();
        let mut scanner =
            SmithWatermanScanner::new(AlignmentCosts::default(), ExecutionScope::Serial).unwrap();

        let mut offsets = [0u32; 2];
        let queries = query_tape(b"kitt", &mut offsets);
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
        scanner
            .scan(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[0, 1, 2]);
        assert_eq!(distances.row(0).unwrap(), &[8.0, 6.0, 6.0]);
    }

    /// Three greetings whose first two differ by one CODEPOINT but two bytes —
    /// the case that splits the byte-level engine from the UTF-8 one.
    struct Greetings;

    const GREETINGS: [&[u8]; 3] = ["héllo".as_bytes(), b"hello", b"world"];

    impl GraphLike for Greetings {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "utf8"
        }
        fn value_shape(&self) -> Option<usize> {
            None
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
            GREETINGS[i]
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
    fn utf8_levenshtein_counts_codepoints_not_bytes() {
        let table = crate::Table::build(&Greetings, None).unwrap();

        // Byte-level: "é" is two bytes, so "héllo" sits two edits from "hello".
        let mut byte_scanner =
            LevenshteinScanner::new(LevenshteinCosts::default(), ExecutionScope::Serial).unwrap();
        let mut offsets = [0u32; 2];
        let queries = query_tape("héllo".as_bytes(), &mut offsets);
        let (mut matches, mut distances, mut counts) = outputs(1, 2);
        byte_scanner
            .scan(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1]);
        assert_eq!(distances.row(0).unwrap(), &[0.0, 2.0]);

        // Codepoint-level: the very same pair is ONE edit, and the walk agrees.
        let mut utf8_scanner =
            LevenshteinUtf8Scanner::new(LevenshteinCosts::default(), ExecutionScope::Serial)
                .unwrap();
        let mut offsets = [0u32; 2];
        let queries = query_tape("héllo".as_bytes(), &mut offsets);
        utf8_scanner
            .scan(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1]);
        assert_eq!(distances.row(0).unwrap(), &[0.0, 1.0]);

        let mut searcher = LevenshteinUtf8Search::new(
            LevenshteinCosts::default(),
            3,
            ExecutionScope::Serial,
            Global,
        )
        .unwrap();
        let mut offsets = [0u32; 2];
        let queries = query_tape("héllo".as_bytes(), &mut offsets);
        searcher
            .search(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1]);
        assert_eq!(distances.row(0).unwrap(), &[0.0, 1.0]);
    }

    #[test]
    fn utf8_scan_refuses_invalid_bytes() {
        // The column claimed `utf8`; a query that is not is a type error, not a
        // garbage distance.
        let table = crate::Table::build(&Greetings, None).unwrap();
        let mut scanner =
            LevenshteinUtf8Scanner::new(LevenshteinCosts::default(), ExecutionScope::Serial)
                .unwrap();
        let mut offsets = [0u32; 2];
        let queries = query_tape(&[0xFF, 0xFE], &mut offsets);
        let (mut matches, mut distances, mut counts) = outputs(1, 2);
        assert!(matches!(
            scanner.scan(queries, &table, &mut matches, &mut distances, &mut counts),
            Err(Error::WrongType)
        ));
    }

    #[test]
    fn scan_refuses_a_regular_column() {
        // A fixed-width column has no offsets, so there is no tape to borrow.
        struct Fixed;
        impl GraphLike for Fixed {
            fn len(&self) -> usize {
                1
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
                Some(0)
            }
            fn key(&self, _i: usize) -> &[u8] {
                &[0; 8]
            }
            fn value(&self, _i: usize) -> &[u8] {
                &[1, 2]
            }
            fn neighbors(&self, _i: usize) -> NeighborList<'_> {
                NeighborList::new(&[], OffsetDType::U32)
            }
        }

        let table = crate::Table::build(&Fixed, None).unwrap();
        let mut scanner =
            LevenshteinScanner::new(LevenshteinCosts::default(), ExecutionScope::Serial).unwrap();
        let mut offsets = [0u32; 2];
        let queries = query_tape(b"a", &mut offsets);
        let (mut matches, mut distances, mut counts) = outputs(1, 1);
        assert!(matches!(
            scanner.scan(queries, &table, &mut matches, &mut distances, &mut counts),
            Err(Error::WrongType)
        ));
    }
}

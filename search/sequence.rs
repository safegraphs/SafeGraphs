//! Sequence metrics over StringZilla. No NumKong type appears here.
//!
//! A node's value is a variable-length byte string — DNA, RNA, protein, or text —
//! and a query batch is a column of them rather than a matrix, so nothing in this
//! family has a width. Both metrics are Levenshtein edit distances under unit
//! costs, ranked smaller-first: one counts bytes, the other UTF-8 runes, where a
//! multibyte character is one edit and each ill-formed subpart reads as one
//! `U+FFFD`. A [`sz::LevenshteinSymbol`] value picks between them, so the family
//! has no dtype axis and no per-metric type; the router in `dispatch` holds one
//! arm per metric name.
//!
//! A StringZilla engine prepares a batch of queries once and scores any number of
//! candidate batches against it, on one CPU thread. So this family opens its
//! [`ExecutionScope`] as a ForkUnion pool, like the NumKong families: the scan
//! fans node chunks across it, each task preparing its own engine over the
//! queries, and the walk fans whole queries, one walker per thread. A GPU scope is
//! refused, since a device engine reads only memory its device reaches, and a
//! mapped file is not that.

use super::batch::{batch_scan, batch_search, build_workers, open_pool, Workers};
use super::{
    walk, ExecutionScope, Matrix, MetricLike, QueryBatchLike, Ranking, ScannerLike,
    TraversalContext, WalkerLike,
};
use crate::{Error, GraphTable, GraphTableView, IndexLike, OffsetDType};
use allocator_api2::alloc::Allocator as TableAllocatorLike;
use core::alloc::Allocator as AllocatorLike;
use core::cmp::Ordering;
use forkunion as fu;
use std::alloc::Global;
use std::sync::OnceLock;
use stringzilla as sz;

// region: Metric

/// The sequence metric's type vocabulary — the sequence twin of
/// `DotMetric<Value>`, with nothing to be generic over.
pub struct SequenceMetric;

impl MetricLike for SequenceMetric {
    type Query<'a> = &'a [u8];
    type Queries<'a> = sz::StringsView<'a>;
    type Score = usize;
}

/// A StringZilla column as a [`QueryBatchLike`]: its string count is the query
/// count, and each string is one query. The seam that lets the shared batch
/// engine walk a column of strings exactly as it walks a dense matrix.
impl QueryBatchLike<SequenceMetric> for sz::StringsView<'_> {
    fn count(&self) -> usize {
        self.len()
    }

    fn query(&self, index: usize) -> Option<&[u8]> {
        self.get(index)
    }
}

/// Borrow nodes `[begin, end)` of a table's variable-length value column as a
/// StringZilla column, with no copy: the prefix sums and the concatenated bytes
/// are already an Arrow tape. Fails on a regular column, which has no offsets to
/// borrow, and on any column that is not `utf8`, since a row start counts
/// elements and a tape slices bytes — the two agree only at one byte per element.
fn value_strings(
    table: GraphTableView<'_>,
    begin: usize,
    end: usize,
) -> Result<sz::StringsView<'_>, Error> {
    if table.value_dtype() != "utf8" {
        return Err(Error::WrongType);
    }
    let (offset_bytes, dtype) = table.value_offsets().ok_or(Error::WrongType)?;
    let values = table.value_buffer();
    match dtype {
        OffsetDType::U32 => {
            let offsets: &[u32] = crate::cast(offset_bytes).ok_or(Error::WrongType)?;
            let offsets = offsets.get(begin..=end).ok_or(Error::Inconsistent)?;
            sz::StringsView::from_u32_tape(values, offsets)
        }
        OffsetDType::U64 => {
            let offsets: &[u64] = crate::cast(offset_bytes).ok_or(Error::WrongType)?;
            let offsets = offsets.get(begin..=end).ok_or(Error::Inconsistent)?;
            sz::StringsView::from_u64_tape(values, offsets)
        }
    }
    .map_err(|_| Error::Inconsistent)
}

/// Prepare `queries` and score them against `candidates` into the query-major
/// `distances`, on the CPU thread that calls it.
fn levenshtein_distances<Query: AsRef<[u8]>>(
    queries: &[Query],
    symbol: sz::LevenshteinSymbol,
    candidates: sz::StringsView<'_>,
    distances: &mut [usize],
) -> Result<(), Error> {
    let stream = sz::Stream::default(sz::Capabilities::cpu_enabled());
    let mut engine =
        sz::LevenshteinEngine::new(queries, symbol, &stream).map_err(|_| Error::Alloc)?;
    let stride = candidates.len();
    stream
        .scope(|scope| engine.distances(scope, candidates, None, distances, stride))
        .map_err(|_| Error::Inconsistent)
}

// endregion: Metric

// region: Scanner

/// The batch half of the sequence metric: owns a thread pool, one query-major
/// distance block per pool thread, and the reusable node-major score tile.
pub struct SequenceScanner<Allocator: AllocatorLike = Global> {
    symbol: sz::LevenshteinSymbol,
    pool: fu::ThreadPool,
    workers: Vec<Vec<usize, Allocator>>,
    tile: Matrix<usize, Allocator>,
    alloc: Allocator,
}

impl<Allocator: AllocatorLike + Clone + Send + Sync> SequenceScanner<Allocator> {
    /// A scanner counting `symbol`s over the workers `scope` names, its scratch
    /// in `alloc`, uniform with [`DotScanner::new`].
    ///
    /// [`DotScanner::new`]: crate::search::dense::DotScanner::new
    pub fn new(
        symbol: sz::LevenshteinSymbol,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(SequenceScanner {
            symbol,
            pool: open_pool(scope)?,
            workers: Vec::new(),
            tile: Matrix::empty_in(alloc.clone()),
            alloc,
        })
    }

    /// Batch brute-force top-k, nearest-first — an edit distance is
    /// smaller-is-better. `matches`/`distances` are `queries × k`; their shape
    /// carries both `k` and the query count, uniform with every dense scanner.
    pub fn scan(
        &mut self,
        queries: sz::StringsView<'_>,
        table: &GraphTable<impl TableAllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let mut tile = std::mem::replace(&mut self.tile, Matrix::empty_in(self.alloc.clone()));
        let outcome = batch_scan::<SequenceMetric, _>(
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

impl<Allocator: AllocatorLike + Clone + Send + Sync> ScannerLike<SequenceMetric>
    for SequenceScanner<Allocator>
{
    fn scan(
        &mut self,
        queries: sz::StringsView<'_>,
        table: GraphTableView<'_>,
        scores: &mut Matrix<usize, impl AllocatorLike + Clone>,
    ) -> Result<(), Error> {
        let (query_count, node_count) = (queries.len(), table.len());
        if query_count != scores.cols() || node_count != scores.rows() {
            return Err(Error::Inconsistent);
        }
        // Checked once up front, so a column that is no tape fails the same way
        // whether or not any task runs.
        value_strings(table, 0, node_count)?;

        let mut prepared: Vec<&[u8]> = Vec::new();
        prepared
            .try_reserve_exact(query_count)
            .map_err(|_| Error::Alloc)?;
        prepared.extend(queries.iter());

        let threads = self.pool.threads_count();
        while self.workers.len() < threads {
            self.workers.push(Vec::new_in(self.alloc.clone()));
        }

        // Fan node CHUNKS across the pool, each task owning a contiguous block of
        // tile rows. A task prepares its own engine, which is neither `Send` nor
        // reusable across threads, so four chunks per worker keep that cost a
        // rounding error while leaving the dynamic schedule slack to balance.
        let chunk_nodes = node_count.div_ceil(threads.max(1) * 4).max(1);
        let mut row_chunks: Vec<&mut [usize]> = Vec::new();
        row_chunks
            .try_reserve_exact(node_count.div_ceil(chunk_nodes))
            .map_err(|_| Error::Alloc)?;
        row_chunks.extend(
            scores
                .as_mut_slice()
                .chunks_mut(chunk_nodes * query_count.max(1)),
        );
        let (symbol, prepared) = (self.symbol, prepared.as_slice());
        let failure = OnceLock::new();

        fu::fold_with_scratch(
            &mut self.pool,
            fu::ParallelSliceMut::new(&mut row_chunks),
            fu::DynamicScheduler,
            self.workers.as_mut_slice(),
            |block, rows, task, _thread| {
                let outcome =
                    scan_node_chunk(block, prepared, symbol, table, task * chunk_nodes, rows);
                if let Err(error) = outcome {
                    let _ = failure.set(error);
                }
            },
        )
        .map_err(|_| Error::Alloc)?;

        match failure.into_inner() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// One chunk of nodes against every query — the body of the scan's fan-out task,
/// handed the chunk's contiguous tile rows by exclusive borrow. The engine writes
/// query-major and the tile is node-major, so the block is transposed into place.
fn scan_node_chunk(
    block: &mut Vec<usize, impl AllocatorLike>,
    queries: &[&[u8]],
    symbol: sz::LevenshteinSymbol,
    table: GraphTableView<'_>,
    begin: usize,
    rows: &mut [usize],
) -> Result<(), Error> {
    let query_count = queries.len();
    let nodes = rows.len() / query_count;
    let candidates = value_strings(table, begin, begin + nodes)?;
    block.clear();
    block
        .try_reserve(query_count * nodes)
        .map_err(|_| Error::Alloc)?;
    block.resize(query_count * nodes, 0);
    levenshtein_distances(queries, symbol, candidates, block)?;
    for (node, row) in rows.chunks_mut(query_count).enumerate() {
        for (query, cell) in row.iter_mut().enumerate() {
            *cell = block[query * nodes + node];
        }
    }
    Ok(())
}

// endregion: Scanner

// region: Searcher

/// The traversal half of the sequence metric: one query at a fixed node width,
/// gathering a staged expansion's values into a tape, since StringZilla scores
/// even one neighbor as a batch.
pub struct SequenceWalker<Index = u32, Allocator: AllocatorLike = Global> {
    symbol: sz::LevenshteinSymbol,
    context: TraversalContext<Index, usize, Allocator>,
    staged_bytes: Vec<u8, Allocator>,
    staged_offsets: Vec<u32, Allocator>,
}

impl<Index: IndexLike, Allocator: AllocatorLike + Clone> SequenceWalker<Index, Allocator> {
    /// A walker counting `symbol`s with breadth `expansion_factor`, its scratch
    /// in `alloc`, uniform with [`DotWalker::new`].
    ///
    /// [`DotWalker::new`]: crate::search::dense::DotWalker::new
    pub fn new(
        symbol: sz::LevenshteinSymbol,
        expansion_factor: usize,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(SequenceWalker {
            symbol,
            context: TraversalContext::new_in(expansion_factor, alloc.clone())?,
            staged_bytes: Vec::new_in(alloc.clone()),
            staged_offsets: Vec::new_in(alloc),
        })
    }

    /// The traversal scratch, for draining results after a walk.
    pub fn context(&mut self) -> &mut TraversalContext<Index, usize, Allocator> {
        &mut self.context
    }
}

impl<Index: IndexLike, Allocator: AllocatorLike + Clone> WalkerLike<SequenceMetric>
    for SequenceWalker<Index, Allocator>
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        query: &[u8],
        table: GraphTableView<'_>,
        nodes: &[Index],
        out: &mut [usize],
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
        let candidates = sz::StringsView::from_u32_tape(&self.staged_bytes, &self.staged_offsets)
            .map_err(|_| Error::Inconsistent)?;
        levenshtein_distances(&[query], self.symbol, candidates, out)
    }

    fn context(&mut self) -> &mut TraversalContext<Index, usize, Allocator> {
        &mut self.context
    }

    fn compare(a: usize, b: usize) -> Ordering {
        Ranking::Nearest.compare(a, b)
    }

    fn search(&mut self, query: &[u8], table: GraphTableView<'_>) -> Result<(), Error> {
        walk::<SequenceMetric, Self>(self, query, table)
    }
}

/// Batched sequence walks, resolving the node width from the first table
/// searched and holding one [`SequenceWalker`] per pool thread.
pub struct SequenceSearcher<Allocator: AllocatorLike = Global> {
    symbol: sz::LevenshteinSymbol,
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers: Workers<SequenceWalker<u32, Allocator>, SequenceWalker<u64, Allocator>, Allocator>,
    alloc: Allocator,
}

impl<Allocator: AllocatorLike + Clone + Send> SequenceSearcher<Allocator> {
    /// A searcher counting `symbol`s, walking with breadth `expansion_factor`
    /// across the workers `scope` names, uniform with [`DotSearcher::new`].
    ///
    /// [`DotSearcher::new`]: crate::search::dense::DotSearcher::new
    pub fn new(
        symbol: sz::LevenshteinSymbol,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(SequenceSearcher {
            symbol,
            pool: open_pool(scope)?,
            expansion_factor,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, fanned across the pool. Uniform with
    /// [`SequenceScanner::scan`].
    pub fn search(
        &mut self,
        queries: sz::StringsView<'_>,
        table: &GraphTable<impl TableAllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let threads = self.pool.threads_count();
        let (symbol, expansion_factor) = (self.symbol, self.expansion_factor);
        match table.neighbors_dtype() {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(threads, alloc.clone(), || {
                        SequenceWalker::new(symbol, expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<SequenceMetric, _>(
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
                        SequenceWalker::new(symbol, expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<SequenceMetric, _>(
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

// endregion: Searcher

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::testing::{assert_scores_close, outputs};
    use crate::{GraphLike, NeighborList, Shape};

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
        fn value_shape(&self) -> Option<Shape> {
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

    fn bytes_scanner() -> SequenceScanner {
        SequenceScanner::new(sz::LevenshteinSymbol::Bytes, ExecutionScope::Serial, Global).unwrap()
    }

    #[test]
    fn scan_ranks_nearest_first() {
        let table = crate::GraphTable::build(&Words, None).unwrap();

        // "kitten": distance 0 to itself, 3 to "sitting", 1 to "mitten" — and the
        // ranking must put the SMALLEST distance first, unlike every dot metric.
        // Two threads split three nodes, so the chunked transpose is exercised too.
        let mut scanner = SequenceScanner::new(
            sz::LevenshteinSymbol::Bytes,
            ExecutionScope::CPUs(2),
            Global,
        )
        .unwrap();
        let queries = sz::StringsView::from_slices(&["kitten"]);
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
        scanner
            .scan(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[0, 2, 1]);
        assert_scores_close(distances.row(0).unwrap(), &[0.0, 1.0, 3.0]);
    }

    #[test]
    fn walk_agrees_with_scan() {
        // The batched-frontier walk must reach the same ordering as a full scan,
        // with a metric whose scores run the opposite direction to the dense one.
        let table = crate::GraphTable::build(&Words, None).unwrap();
        let mut searcher = SequenceSearcher::new(
            sz::LevenshteinSymbol::Bytes,
            3,
            ExecutionScope::Serial,
            Global,
        )
        .unwrap();

        let queries = sz::StringsView::from_slices(&["kitten"]);
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
        searcher
            .search(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(counts, [3]);
        assert_eq!(matches.row(0).unwrap(), &[0, 2, 1]);
        assert_scores_close(distances.row(0).unwrap(), &[0.0, 1.0, 3.0]);
    }

    #[test]
    fn batch_walks_many_queries() {
        // A two-query column through one searcher: the column carries the count,
        // each query fills its own row, and the rows disagree — proof the loop
        // indexes the column rather than rescoring the first string.
        let table = crate::GraphTable::build(&Words, None).unwrap();
        let mut searcher = SequenceSearcher::new(
            sz::LevenshteinSymbol::Bytes,
            3,
            ExecutionScope::Serial,
            Global,
        )
        .unwrap();

        let offsets: [u32; 3] = [0, 6, 12];
        let queries = sz::StringsView::from_u32_tape(b"kittenmitten", &offsets).unwrap();
        let (mut matches, mut distances, mut counts) = outputs(2, 2);
        searcher
            .search(queries, &table, &mut matches, &mut distances, &mut counts)
            .unwrap();
        assert_eq!(counts, [2, 2]);
        // "kitten" sits on node 0, one edit from "mitten"; "mitten" the reverse.
        assert_eq!(matches.row(0).unwrap(), &[0, 2]);
        assert_scores_close(distances.row(0).unwrap(), &[0.0, 1.0]);
        assert_eq!(matches.row(1).unwrap(), &[2, 0]);
        assert_eq!(distances.row(1).unwrap(), &[0.0, 1.0]);
    }

    /// Three greetings whose first two differ by one RUNE but two bytes — the
    /// case that splits the byte-level metric from the UTF-8 one.
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
        fn value_shape(&self) -> Option<Shape> {
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
    fn utf8_levenshtein_counts_runes_not_bytes() {
        let table = crate::GraphTable::build(&Greetings, None).unwrap();
        let queries = ["héllo".as_bytes()];

        // Byte-level: "é" is two bytes, so "héllo" sits two edits from "hello".
        let mut byte_scanner = bytes_scanner();
        let (mut matches, mut distances, mut counts) = outputs(1, 2);
        byte_scanner
            .scan(
                sz::StringsView::from_slices(&queries),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1]);
        assert_scores_close(distances.row(0).unwrap(), &[0.0, 2.0]);

        // Rune-level: the very same pair is ONE edit, and the walk agrees.
        let mut rune_scanner =
            SequenceScanner::new(sz::LevenshteinSymbol::Runes, ExecutionScope::Serial, Global)
                .unwrap();
        rune_scanner
            .scan(
                sz::StringsView::from_slices(&queries),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1]);
        assert_scores_close(distances.row(0).unwrap(), &[0.0, 1.0]);

        let mut searcher = SequenceSearcher::new(
            sz::LevenshteinSymbol::Runes,
            3,
            ExecutionScope::Serial,
            Global,
        )
        .unwrap();
        searcher
            .search(
                sz::StringsView::from_slices(&queries),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0, 1]);
        assert_scores_close(distances.row(0).unwrap(), &[0.0, 1.0]);
    }

    /// An ill-formed byte is one `U+FFFD` rune, not an error: "h\xFFllo" is one
    /// substitution from "héllo" counted in runes, and two counted in bytes.
    #[test]
    fn utf8_scan_reads_an_ill_formed_byte_as_one_rune() {
        let table = crate::GraphTable::build(&Greetings, None).unwrap();
        let queries = [&b"h\xFFllo"[..]];
        let mut scanner =
            SequenceScanner::new(sz::LevenshteinSymbol::Runes, ExecutionScope::Serial, Global)
                .unwrap();
        let (mut matches, mut distances, mut counts) = outputs(1, 1);
        scanner
            .scan(
                sz::StringsView::from_slices(&queries),
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap(), &[0]);
        assert_scores_close(distances.row(0).unwrap(), &[1.0]);
    }

    /// A one-node graph whose value column the caller shapes, for the two ways a
    /// column can fail to be a tape.
    struct OneNode {
        dtype: &'static str,
        shape: Option<Shape>,
    }

    impl GraphLike for OneNode {
        fn len(&self) -> usize {
            1
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            self.dtype
        }
        fn value_shape(&self) -> Option<Shape> {
            self.shape
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

    /// Scanning `graph` must fail with [`Error::WrongType`], whichever guard
    /// catches it.
    fn expect_wrong_type(graph: &OneNode) {
        let table = crate::GraphTable::build(graph, None).unwrap();
        let mut scanner = bytes_scanner();
        let (mut matches, mut distances, mut counts) = outputs(1, 1);
        assert!(matches!(
            scanner.scan(
                sz::StringsView::from_slices(&["a"]),
                &table,
                &mut matches,
                &mut distances,
                &mut counts
            ),
            Err(Error::WrongType)
        ));
    }

    /// A fixed-width column has no offsets, so there is no tape to borrow.
    #[test]
    fn scan_refuses_a_regular_column() {
        expect_wrong_type(&OneNode {
            dtype: "utf8",
            shape: Some(Shape::Vector(2)),
        });
    }

    /// A ragged column of anything but bytes has row starts counting elements
    /// where a tape slices bytes, so the two disagree.
    #[test]
    fn scan_refuses_a_non_utf8_column() {
        expect_wrong_type(&OneNode {
            dtype: "u16",
            shape: None,
        });
    }
}

//! Shape distances between point clouds over NumKong. No StringZilla type appears here.
//!
//! A node's value is a point cloud of any size: interleaved `[x0,y0,z0, x1,y1,z1, ...]`
//! coordinates in `f32` or `f16`, the two scalars NumKong's `Shapes` trait packs. Its
//! logical [`Shape`] is `Matrix(0, 3)`, a ragged point axis whose prefix sum counts
//! scalars, three per point, exactly what USearch writes; a fixed `Matrix(points, 3)`
//! column reads the same way. A query is another cloud of any size, and no point
//! correspondence is assumed between the two.
//!
//! Two distances share one generic body through the [`ShapeDistanceLike`] seam:
//!
//! - [`ChamferDistance`] — the mean nearest-partner distance both ways, halved.
//! - [`HausdorffDistance`] — the worst nearest-partner distance either way.
//!
//! Both are invariant to rotation, reflection, and translation, and smaller is better,
//! so the family ranks nearest-first like Euclidean and Geospatial. Distances are `f32`
//! for `f16` clouds and `f64` for `f32` ones.
//!
//! Packing a cloud, centering it and rotating it onto its principal axes, is the
//! expensive step. So a scan packs every query once and the node clouds one tile at a
//! time, scoring the tile's nodes as rows against the queries as columns, which is the
//! node-major score tile itself. A walk packs its query once and each hop's neighbors.

use super::batch::{batch_scan, batch_search, build_workers, open_pool, Workers};
use super::{
    walk, ExecutionScope, Matrix, MetricLike, QueryBatchLike, Ranking, ScannerLike,
    TraversalContext, WalkerLike,
};
use crate::{cast, Error, GraphTable, GraphTableView, IndexLike, OffsetDType, ScoreLike, Shape};
use allocator_api2::alloc::Allocator as TableAllocatorLike;
use core::alloc::Allocator as AllocatorLike;
use core::cmp::Ordering;
use forkunion as fu;
use numkong::{self as nk};
use std::alloc::Global;

/// Node clouds a scan packs at once: bounds the packs held in memory while leaving
/// every pool thread many rows per many-to-many call.
const NODES_PER_TILE: usize = 4096;

// region: Distance seam

/// The seam between the shared shapes machinery and one NumKong many-to-many kernel:
/// the distance from every row cloud to every column cloud, row-major.
pub trait ShapeDistanceLike: 'static {
    /// Fill `results` with `rows × columns` distances on the calling thread.
    fn distances<Scalar, RowsAllocator, ColumnsAllocator>(
        rows: &nk::ShapesPackedMatrix<Scalar, RowsAllocator>,
        columns: &nk::ShapesPackedMatrix<Scalar, ColumnsAllocator>,
        results: &mut [Scalar::Distance],
    ) -> Result<(), Error>
    where
        Scalar: nk::Shapes,
        RowsAllocator: AllocatorLike,
        ColumnsAllocator: AllocatorLike;

    /// [`distances`](ShapeDistanceLike::distances) with the rows split across `pool`.
    fn distances_parallel<Scalar, RowsAllocator, ColumnsAllocator>(
        rows: &nk::ShapesPackedMatrix<Scalar, RowsAllocator>,
        columns: &nk::ShapesPackedMatrix<Scalar, ColumnsAllocator>,
        results: &mut [Scalar::Distance],
        pool: &mut fu::ThreadPool,
    ) -> Result<(), Error>
    where
        Scalar: nk::Shapes + Sync,
        RowsAllocator: AllocatorLike,
        ColumnsAllocator: AllocatorLike;
}

/// Chamfer: the mean nearest-partner distance both ways, halved.
pub struct ChamferDistance;
/// Hausdorff: the worst nearest-partner distance either way.
pub struct HausdorffDistance;

impl ShapeDistanceLike for ChamferDistance {
    fn distances<Scalar, RowsAllocator, ColumnsAllocator>(
        rows: &nk::ShapesPackedMatrix<Scalar, RowsAllocator>,
        columns: &nk::ShapesPackedMatrix<Scalar, ColumnsAllocator>,
        results: &mut [Scalar::Distance],
    ) -> Result<(), Error>
    where
        Scalar: nk::Shapes,
        RowsAllocator: AllocatorLike,
        ColumnsAllocator: AllocatorLike,
    {
        rows.chamfers(columns, results, None, Scalar::UNBOUNDED)
            .map_err(|_| Error::Inconsistent)
    }

    fn distances_parallel<Scalar, RowsAllocator, ColumnsAllocator>(
        rows: &nk::ShapesPackedMatrix<Scalar, RowsAllocator>,
        columns: &nk::ShapesPackedMatrix<Scalar, ColumnsAllocator>,
        results: &mut [Scalar::Distance],
        pool: &mut fu::ThreadPool,
    ) -> Result<(), Error>
    where
        Scalar: nk::Shapes + Sync,
        RowsAllocator: AllocatorLike,
        ColumnsAllocator: AllocatorLike,
    {
        rows.chamfers_parallel(columns, results, None, Scalar::UNBOUNDED, pool)
            .map_err(|_| Error::Inconsistent)
    }
}

impl ShapeDistanceLike for HausdorffDistance {
    fn distances<Scalar, RowsAllocator, ColumnsAllocator>(
        rows: &nk::ShapesPackedMatrix<Scalar, RowsAllocator>,
        columns: &nk::ShapesPackedMatrix<Scalar, ColumnsAllocator>,
        results: &mut [Scalar::Distance],
    ) -> Result<(), Error>
    where
        Scalar: nk::Shapes,
        RowsAllocator: AllocatorLike,
        ColumnsAllocator: AllocatorLike,
    {
        rows.hausdorffs(columns, results, None, Scalar::UNBOUNDED)
            .map_err(|_| Error::Inconsistent)
    }

    fn distances_parallel<Scalar, RowsAllocator, ColumnsAllocator>(
        rows: &nk::ShapesPackedMatrix<Scalar, RowsAllocator>,
        columns: &nk::ShapesPackedMatrix<Scalar, ColumnsAllocator>,
        results: &mut [Scalar::Distance],
        pool: &mut fu::ThreadPool,
    ) -> Result<(), Error>
    where
        Scalar: nk::Shapes + Sync,
        RowsAllocator: AllocatorLike,
        ColumnsAllocator: AllocatorLike,
    {
        rows.hausdorffs_parallel(columns, results, None, Scalar::UNBOUNDED, pool)
            .map_err(|_| Error::Inconsistent)
    }
}

// endregion: Distance seam

// region: Metric

/// A shape metric over one packable scalar and one distance kind.
pub struct ShapeMetric<Scalar, Kind>(core::marker::PhantomData<fn() -> (Scalar, Kind)>);

impl<Scalar, Kind> MetricLike for ShapeMetric<Scalar, Kind>
where
    Scalar: nk::Shapes + Copy + 'static,
    Scalar::Distance: ScoreLike + PartialOrd,
    Kind: ShapeDistanceLike,
{
    /// One query cloud, flat `points * 3` scalars.
    type Query<'a> = &'a [Scalar];
    type Queries<'a> = CloudBatch<'a, Scalar>;
    type Score = Scalar::Distance;
}

/// A batch of ragged query clouds: every query's scalars concatenated, plus a
/// prefix sum of `count + 1` entries counting scalars, so query `i` owns scalars
/// `[offsets[i], offsets[i + 1])`. The value column's own bytes-plus-offsets model.
pub struct CloudBatch<'a, Scalar> {
    scalars: &'a [Scalar],
    offsets: &'a [u64],
}

impl<'a, Scalar> CloudBatch<'a, Scalar> {
    /// Borrow `scalars` as `offsets.len() - 1` clouds, or [`Error::Inconsistent`]
    /// if the offsets do not ascend through whole points to exactly `scalars.len()`.
    pub fn new(scalars: &'a [Scalar], offsets: &'a [u64]) -> Result<Self, Error> {
        if offsets.last().map(|&end| end as usize) != Some(scalars.len())
            || offsets.windows(2).any(|pair| pair[1] < pair[0])
            || offsets.iter().any(|&offset| offset % 3 != 0)
        {
            return Err(Error::Inconsistent);
        }
        Ok(CloudBatch { scalars, offsets })
    }
}

impl<Scalar, Kind> QueryBatchLike<ShapeMetric<Scalar, Kind>> for CloudBatch<'_, Scalar>
where
    Scalar: nk::Shapes + Copy + 'static,
    Scalar::Distance: ScoreLike + PartialOrd,
    Kind: ShapeDistanceLike,
{
    fn count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn query(&self, index: usize) -> Option<&[Scalar]> {
        let start = usize::try_from(*self.offsets.get(index)?).ok()?;
        let end = usize::try_from(*self.offsets.get(index + 1)?).ok()?;
        self.scalars.get(start..end)
    }
}

/// The single shape check every entry point repeats: three coordinates per point,
/// with the point axis ragged or fixed.
fn cloud_column(table: GraphTableView<'_>) -> Result<(), Error> {
    match table.value_shape() {
        Some(Shape::Matrix(_, 3)) => Ok(()),
        _ => Err(Error::WrongType),
    }
}

/// Reshape a flat interleaved cloud into points, or [`Error::Inconsistent`] when a
/// coordinate is left over.
fn points<Scalar>(flat: &[Scalar]) -> Result<&[[Scalar; 3]], Error> {
    match flat.as_chunks::<3>() {
        (points, []) => Ok(points),
        _ => Err(Error::Inconsistent),
    }
}

/// Node `node`'s cloud, read in place from the mapped value column.
fn node_cloud<Scalar: Copy>(
    table: GraphTableView<'_>,
    node: usize,
) -> Result<&[[Scalar; 3]], Error> {
    points(cast(table.value(node)).ok_or(Error::WrongType)?)
}

// endregion: Metric

// region: Scanner

/// The batch half of every shape metric: owns a thread pool and the reusable
/// node-major score tile.
pub struct ShapeScanner<Scalar: nk::Shapes, Kind, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    tile: Matrix<Scalar::Distance, Allocator>,
    alloc: Allocator,
    kind: core::marker::PhantomData<fn() -> Kind>,
}

impl<Scalar, Kind, Allocator> ShapeScanner<Scalar, Kind, Allocator>
where
    Scalar: nk::Shapes + Copy + Sync + 'static,
    Scalar::Distance: ScoreLike + PartialOrd,
    Kind: ShapeDistanceLike,
    Allocator: AllocatorLike + Clone,
{
    /// A scanner over the workers `scope` names, its packs in `alloc`, uniform with
    /// [`DotScanner::new`]. NumKong's shape kernels are CPU-only, so a GPU scope is
    /// refused.
    ///
    /// [`DotScanner::new`]: crate::search::dense::DotScanner::new
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(ShapeScanner {
            pool: open_pool(scope)?,
            tile: Matrix::empty_in(alloc.clone()),
            alloc,
            kind: core::marker::PhantomData,
        })
    }

    /// Batch brute-force top-k, nearest-first. `matches`/`distances` are
    /// `queries × k`; their shape carries both `k` and the query count.
    pub fn scan(
        &mut self,
        queries: CloudBatch<'_, Scalar>,
        table: &GraphTable<impl TableAllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let mut tile = std::mem::replace(&mut self.tile, Matrix::empty_in(self.alloc.clone()));
        let outcome = batch_scan::<ShapeMetric<Scalar, Kind>, _>(
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

impl<Scalar, Kind, Allocator> ScannerLike<ShapeMetric<Scalar, Kind>>
    for ShapeScanner<Scalar, Kind, Allocator>
where
    Scalar: nk::Shapes + Copy + Sync + 'static,
    Scalar::Distance: ScoreLike + PartialOrd,
    Kind: ShapeDistanceLike,
    Allocator: AllocatorLike + Clone,
{
    fn scan(
        &mut self,
        queries: CloudBatch<'_, Scalar>,
        table: GraphTableView<'_>,
        scores: &mut Matrix<Scalar::Distance, impl AllocatorLike + Clone>,
    ) -> Result<(), Error> {
        cloud_column(table)?;
        let node_count = table.len();
        let query_count =
            <CloudBatch<'_, Scalar> as QueryBatchLike<ShapeMetric<Scalar, Kind>>>::count(&queries);
        if query_count != scores.cols() || node_count != scores.rows() {
            return Err(Error::Inconsistent);
        }

        let mut clouds: Vec<&[[Scalar; 3]]> = Vec::new();
        clouds
            .try_reserve_exact(query_count.max(NODES_PER_TILE.min(node_count)))
            .map_err(|_| Error::Alloc)?;
        for query_index in 0..query_count {
            let query =
                <CloudBatch<'_, Scalar> as QueryBatchLike<ShapeMetric<Scalar, Kind>>>::query(
                    &queries,
                    query_index,
                )
                .ok_or(Error::Inconsistent)?;
            clouds.push(points(query)?);
        }
        let packed_queries =
            nk::ShapesPackedMatrix::new_parallel_in(&clouds, self.alloc.clone(), &mut self.pool)
                .map_err(|_| Error::Alloc)?;

        for begin in (0..node_count).step_by(NODES_PER_TILE) {
            let end = node_count.min(begin + NODES_PER_TILE);
            clouds.clear();
            for node in begin..end {
                clouds.push(node_cloud(table, node)?);
            }
            let packed_nodes = nk::ShapesPackedMatrix::new_parallel_in(
                &clouds,
                self.alloc.clone(),
                &mut self.pool,
            )
            .map_err(|_| Error::Alloc)?;
            let rows = &mut scores.as_mut_slice()[begin * query_count..end * query_count];
            Kind::distances_parallel(&packed_nodes, &packed_queries, rows, &mut self.pool)?;
        }
        Ok(())
    }
}

// endregion: Scanner

// region: Searcher

/// The traversal half of every shape metric: one query at a fixed node width,
/// packed once per walk and scored against each hop's freshly packed neighbors.
pub struct ShapeWalker<Scalar: nk::Shapes, Kind, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, Scalar::Distance, Allocator>,
    /// The query being walked, packed by [`WalkerLike::search`] and cleared after,
    /// so scoring outside a walk is refused rather than scored against a stale pack.
    packed_query: Option<nk::ShapesPackedMatrix<Scalar, Allocator>>,
    alloc: Allocator,
    kind: core::marker::PhantomData<fn() -> Kind>,
}

impl<Scalar, Kind, Index, Allocator> ShapeWalker<Scalar, Kind, Index, Allocator>
where
    Scalar: nk::Shapes,
    Scalar::Distance: Copy + Default,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
{
    /// A walker with breadth `expansion_factor`, its scratch in `alloc`, uniform
    /// with [`DotWalker::new`].
    ///
    /// [`DotWalker::new`]: crate::search::dense::DotWalker::new
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        Ok(ShapeWalker {
            context: TraversalContext::new_in(expansion_factor, alloc.clone())?,
            packed_query: None,
            alloc,
            kind: core::marker::PhantomData,
        })
    }

    /// The traversal scratch, for draining results after a walk.
    pub fn context(&mut self) -> &mut TraversalContext<Index, Scalar::Distance, Allocator> {
        &mut self.context
    }
}

impl<Scalar, Kind, Index, Allocator> WalkerLike<ShapeMetric<Scalar, Kind>>
    for ShapeWalker<Scalar, Kind, Index, Allocator>
where
    Scalar: nk::Shapes + Copy + 'static,
    Scalar::Distance: ScoreLike + PartialOrd,
    Kind: ShapeDistanceLike,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        _query: &[Scalar],
        table: GraphTableView<'_>,
        nodes: &[Index],
        out: &mut [Scalar::Distance],
    ) -> Result<(), Error> {
        if nodes.len() != out.len() {
            return Err(Error::Inconsistent);
        }
        cloud_column(table)?;
        let packed_query = self.packed_query.as_ref().ok_or(Error::Inconsistent)?;
        let mut clouds: Vec<&[[Scalar; 3]]> = Vec::new();
        clouds
            .try_reserve_exact(nodes.len())
            .map_err(|_| Error::Alloc)?;
        for &node in nodes {
            let index: usize = node.try_into().map_err(|_| Error::TooLarge)?;
            clouds.push(node_cloud(table, index)?);
        }
        let packed_nodes = nk::ShapesPackedMatrix::new_in(&clouds, self.alloc.clone())
            .map_err(|_| Error::Alloc)?;
        Kind::distances(&packed_nodes, packed_query, out)
    }

    fn context(&mut self) -> &mut TraversalContext<Index, Scalar::Distance, Allocator> {
        &mut self.context
    }

    fn compare(a: Scalar::Distance, b: Scalar::Distance) -> Ordering {
        Ranking::Nearest.compare(a, b)
    }

    fn search(&mut self, query: &[Scalar], table: GraphTableView<'_>) -> Result<(), Error> {
        let packed = nk::ShapesPackedMatrix::new_in(&[points(query)?], self.alloc.clone())
            .map_err(|_| Error::Alloc)?;
        self.packed_query = Some(packed);
        let outcome = walk::<ShapeMetric<Scalar, Kind>, Self>(self, query, table);
        self.packed_query = None;
        outcome
    }
}

/// One [`ShapeWalker`] per pool thread, at whichever node width the first table names.
type ShapeWorkers<Scalar, Kind, Allocator> = Workers<
    ShapeWalker<Scalar, Kind, u32, Allocator>,
    ShapeWalker<Scalar, Kind, u64, Allocator>,
    Allocator,
>;

/// Batched shape walks, resolving the node width from the first table searched and
/// holding one [`ShapeWalker`] per pool thread.
pub struct ShapeSearcher<Scalar: nk::Shapes, Kind, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers: ShapeWorkers<Scalar, Kind, Allocator>,
    alloc: Allocator,
}

impl<Scalar, Kind, Allocator> ShapeSearcher<Scalar, Kind, Allocator>
where
    Scalar: nk::Shapes + Copy + Send + 'static,
    Scalar::Distance: ScoreLike + PartialOrd + Send,
    Kind: ShapeDistanceLike,
    Allocator: AllocatorLike + Clone + Send,
{
    /// A searcher walking with breadth `expansion_factor` across the workers `scope`
    /// names, uniform with [`DotSearcher::new`].
    ///
    /// [`DotSearcher::new`]: crate::search::dense::DotSearcher::new
    pub fn new(
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(ShapeSearcher {
            pool: open_pool(scope)?,
            expansion_factor,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, fanned across the pool. Uniform with
    /// [`ShapeScanner::scan`].
    pub fn search(
        &mut self,
        queries: CloudBatch<'_, Scalar>,
        table: &GraphTable<impl TableAllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let threads = self.pool.threads_count();
        let expansion_factor = self.expansion_factor;
        match table.neighbors_dtype() {
            OffsetDType::U32 => {
                if !matches!(self.workers, Workers::U32(_)) {
                    let alloc = &self.alloc;
                    self.workers = Workers::U32(build_workers(threads, alloc.clone(), || {
                        ShapeWalker::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<ShapeMetric<Scalar, Kind>, _>(
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
                        ShapeWalker::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<ShapeMetric<Scalar, Kind>, _>(
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

// region: Runtime dtype dispatch

/// A shape scanner chosen from a file's value dtype string, generic over the
/// distance kind: the dtype axis is this type's business, and which kind scores
/// stays the marker's.
pub enum ShapeScannerAny<Kind, Allocator: AllocatorLike = Global> {
    F32(ShapeScanner<f32, Kind, Allocator>),
    F16(ShapeScanner<nk::f16, Kind, Allocator>),
}

impl<Kind, Allocator> ShapeScannerAny<Kind, Allocator>
where
    Kind: ShapeDistanceLike,
    Allocator: AllocatorLike + Clone,
{
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "f32" => Self::F32(ShapeScanner::new(scope, alloc)?),
            "f16" => Self::F16(ShapeScanner::new(scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    /// Scan raw query bytes plus their scalar-counting prefix sum — the ragged
    /// input shape the router hands every cloud family.
    pub fn scan(
        &mut self,
        query_bytes: &[u8],
        query_offsets: &[u64],
        table: &GraphTable<impl TableAllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            Self::F32(scanner) => scanner.scan(
                CloudBatch::new(cast(query_bytes).ok_or(Error::WrongType)?, query_offsets)?,
                table,
                matches,
                distances,
                counts,
            ),
            Self::F16(scanner) => scanner.scan(
                CloudBatch::new(cast(query_bytes).ok_or(Error::WrongType)?, query_offsets)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// A shape searcher chosen from a file's value dtype string — the walk twin of
/// [`ShapeScannerAny`].
pub enum ShapeSearcherAny<Kind, Allocator: AllocatorLike = Global> {
    F32(ShapeSearcher<f32, Kind, Allocator>),
    F16(ShapeSearcher<nk::f16, Kind, Allocator>),
}

impl<Kind, Allocator> ShapeSearcherAny<Kind, Allocator>
where
    Kind: ShapeDistanceLike,
    Allocator: AllocatorLike + Clone + Send,
{
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f32" => Self::F32(ShapeSearcher::new(expansion_factor, scope, alloc)?),
            "f16" => Self::F16(ShapeSearcher::new(expansion_factor, scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn search(
        &mut self,
        query_bytes: &[u8],
        query_offsets: &[u64],
        table: &GraphTable<impl TableAllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            Self::F32(searcher) => searcher.search(
                CloudBatch::new(cast(query_bytes).ok_or(Error::WrongType)?, query_offsets)?,
                table,
                matches,
                distances,
                counts,
            ),
            Self::F16(searcher) => searcher.search(
                CloudBatch::new(cast(query_bytes).ok_or(Error::WrongType)?, query_offsets)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

// endregion: Runtime dtype dispatch

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::testing::outputs;
    use crate::{GraphLike, NeighborList};

    /// Four `f16` clouds of different sizes on a ring, written the way USearch
    /// writes them: a ragged `[0, 3]` column, `u64` offsets counting scalars, and
    /// 16-byte UUID keys holding the row number big-endian in their last 8 bytes.
    /// Node 2 is the query cloud rotated a quarter turn and shifted, so both
    /// distances score it zero; the others are other shapes of other sizes.
    struct Clouds {
        values: Vec<Vec<nk::f16>>,
        keys: Vec<[u8; 16]>,
    }

    fn query() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
        ]
    }

    fn clouds() -> Clouds {
        let rotated: Vec<[f32; 3]> = query()
            .iter()
            .map(|&[x, y, z]| [10.0 - y, 5.0 + x, z - 2.0])
            .collect();
        let shapes: [Vec<[f32; 3]>; 4] = [
            vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [1.0, 1.0, 1.0],
            ],
            rotated,
            vec![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        ];
        Clouds {
            values: shapes
                .iter()
                .map(|cloud| {
                    cloud
                        .iter()
                        .flatten()
                        .map(|&value| nk::f16::from_f32(value))
                        .collect()
                })
                .collect(),
            keys: (0..4u64)
                .map(|row| {
                    let mut key = [0u8; 16];
                    key[8..].copy_from_slice(&row.to_be_bytes());
                    key
                })
                .collect(),
        }
    }

    impl GraphLike for Clouds {
        fn len(&self) -> usize {
            4
        }
        fn key_dtype(&self) -> &str {
            "uuid"
        }
        fn value_dtype(&self) -> &str {
            "f16"
        }
        fn value_shape(&self) -> Option<Shape> {
            Some(Shape::Matrix(0, 3))
        }
        fn neighbors_degree(&self) -> Option<usize> {
            Some(2)
        }
        fn key(&self, i: usize) -> &[u8] {
            &self.keys[i]
        }
        fn value(&self, i: usize) -> &[u8] {
            let row = &self.values[i];
            // SAFETY: reinterpret the contiguous `f16` coordinates as their bytes.
            unsafe {
                core::slice::from_raw_parts(
                    row.as_ptr() as *const u8,
                    core::mem::size_of_val(row.as_slice()),
                )
            }
        }
        fn neighbors(&self, i: usize) -> NeighborList<'_> {
            const RING: [[u8; 8]; 4] = [
                [1, 0, 0, 0, 3, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0],
            ];
            NeighborList::new(&RING[i], OffsetDType::U32)
        }
    }

    /// The table a USearch monolith would be, after a save and a reopen.
    fn table() -> GraphTable {
        let built = GraphTable::build(&clouds(), None).unwrap();
        GraphTable::from_bytes(built.as_bytes()).unwrap()
    }

    fn query_bytes() -> Vec<u8> {
        query()
            .iter()
            .flatten()
            .flat_map(|&value| nk::f16::from_f32(value).0.to_le_bytes())
            .collect()
    }

    #[test]
    fn ragged_clouds_round_trip_as_usearch_writes_them() {
        let table = table();
        assert_eq!(table.value_shape(), Some(Shape::Matrix(0, 3)));
        assert_eq!(table.value(1).len(), 5 * 3 * 2);
        assert_eq!(&table.key(3)[8..], &3u64.to_be_bytes());
    }

    #[test]
    fn chamfer_scan_and_walk_find_the_rotated_cloud() {
        let table = table();
        let (bytes, offsets) = (query_bytes(), [0u64, 12]);

        let mut scanner =
            ShapeScannerAny::<ChamferDistance>::new("f16", ExecutionScope::CPUs(2), Global)
                .unwrap();
        let (mut matches, mut distances, mut counts) = outputs(1, 4);
        scanner
            .scan(
                &bytes,
                &offsets,
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(counts, [4]);
        assert_eq!(matches.row(0).unwrap()[0], 2);
        let row = distances.row(0).unwrap();
        assert!(row[0] < 1e-2, "{row:?}");
        assert!(row.windows(2).all(|pair| pair[0] <= pair[1]));

        let mut searcher =
            ShapeSearcherAny::<ChamferDistance>::new("f16", 4, ExecutionScope::Serial, Global)
                .unwrap();
        let (mut walked, mut walked_distances, mut walked_counts) = outputs(1, 4);
        searcher
            .search(
                &bytes,
                &offsets,
                &table,
                &mut walked,
                &mut walked_distances,
                &mut walked_counts,
            )
            .unwrap();
        assert_eq!(walked_counts, [4]);
        assert_eq!(walked.row(0).unwrap(), matches.row(0).unwrap());
    }

    #[test]
    fn hausdorff_ranks_the_rotated_cloud_first() {
        let table = table();
        let mut scanner =
            ShapeScannerAny::<HausdorffDistance>::new("f16", ExecutionScope::Serial, Global)
                .unwrap();
        let (mut matches, mut distances, mut counts) = outputs(1, 2);
        scanner
            .scan(
                &query_bytes(),
                &[0, 12],
                &table,
                &mut matches,
                &mut distances,
                &mut counts,
            )
            .unwrap();
        assert_eq!(matches.row(0).unwrap()[0], 2);
        let row = distances.row(0).unwrap();
        assert!(row[0] < 1e-2 && row[0] < row[1], "{row:?}");
    }

    #[test]
    fn a_partial_point_or_another_dtype_is_refused() {
        let table = table();
        let mut scanner =
            ShapeScannerAny::<ChamferDistance>::new("f16", ExecutionScope::Serial, Global).unwrap();
        let (mut matches, mut distances, mut counts) = outputs(1, 1);
        assert!(matches!(
            scanner.scan(
                &query_bytes(),
                &[0, 11],
                &table,
                &mut matches,
                &mut distances,
                &mut counts
            ),
            Err(Error::Inconsistent)
        ));
        assert!(
            ShapeScannerAny::<ChamferDistance>::new("bf16", ExecutionScope::Serial, Global)
                .is_err()
        );
    }
}

//! Mesh-superposition metric over NumKong. No StringZilla type appears here.
//!
//! A node's value is a point cloud: a fixed `points × 3` block of coordinates in
//! `f64`/`f32`/`f16`/`bf16`, interleaved `[x0,y0,z0, x1,y1,z1, ...]` — NumKong's
//! own `mesh.h` layout, adopted rather than converted. Its logical [`Shape`] is
//! `Matrix(points, 3)`; the byte layout is a regular `points*3`-wide row. A query
//! is another cloud of the __same cardinality__ and __same canonical ordering__
//! (point `i` of the query corresponds to point `i` of every node). The score is
//! the RMSD left after superposing the query onto the node, so the family ranks
//! nearest-first like Euclidean and Geospatial.
//!
//! Three superpositions share one generic body through the [`SuperposeLike`]
//! seam — the mesh analogue of geospatial's `KernelLike`:
//!
//! - [`RigidSuperpose`] — no alignment: RMSD of the clouds as given (`nk_rmsd`).
//! - [`RotationalSuperpose`] — Kabsch: optimal rotation + translation, no scale.
//! - [`SimilaritySuperpose`] — Umeyama: rotation + translation + uniform scale.
//!
//! Each superposition is a __whole-cloud reduction__: unlike geospatial there is
//! no per-element broadcast, so the scan aligns one query cloud against each node
//! cloud into one tile cell, fanning queries across the owned pool.

use super::batch::{batch_scan, batch_search, build_workers, open_pool, ShareRead, Workers};
use super::{
    walk, ExecutionScope, Matrix, MetricLike, QueryBatchLike, Ranking, ScannerLike, SearcherLike,
    TraversalContext,
};
use crate::search::dense::matrix_view;
use crate::{cast, Error, IndexLike, OffsetDType, Shape, Table, TableView};
use allocator_api2::alloc::{Allocator as AllocatorLike, Global};
use core::cmp::Ordering;
use forkunion as fu;
use numkong as nk;
use std::sync::OnceLock;

// region: Superposition seam

/// The seam between the shared mesh machinery and one NumKong superposition: the
/// RMSD left after aligning one `points × 3` query cloud onto one node cloud.
/// Implemented once, generically, for every coordinate the [`nk::MeshAlignment`]
/// trait covers, so [`MeshScanner`] and friends span all four dtypes the way
/// `DotMetric<Value>` spans the dense scalars.
///
/// The three calls have the identical `(a, b) -> rmsd` shape in `mesh.h`; only
/// which invariants they factor out (nothing / rotation / rotation+scale)
/// differs, so the seam is one method and the marker picks the kernel.
pub trait SuperposeLike<Coordinate>: 'static {
    /// The RMSD after superposing `query` onto `node`, both `points × 3`
    /// interleaved and exactly `points` triples. `None` is a length mismatch or
    /// a degenerate cloud, which the callers turn into [`Error::Inconsistent`].
    fn rmsd(query: &[Coordinate], node: &[Coordinate], points: usize) -> Option<f64>;
}

/// Kabsch rotation + translation, no scale — same rigid shape, any orientation.
pub struct RotationalSuperpose;
/// Raw RMSD, no alignment — same points in the same frame.
pub struct RigidSuperpose;
/// Umeyama rotation + translation + uniform scale — same shape up to a size.
pub struct SimilaritySuperpose;

/// Reshape a flat interleaved `points × 3` slice into `&[[Coordinate; 3]]`, or
/// `None` if the length is not exactly `points` triples. `slice::as_chunks`
/// replaces the hand-rolled `from_raw_parts` the draft reached for — no `unsafe`.
fn as_triples<Coordinate>(flat: &[Coordinate], points: usize) -> Option<&[[Coordinate; 3]]> {
    let (triples, remainder) = flat.as_chunks::<3>();
    if !remainder.is_empty() || triples.len() != points {
        return None;
    }
    Some(triples)
}

impl<Coordinate> SuperposeLike<Coordinate> for RotationalSuperpose
where
    Coordinate: nk::MeshAlignment + 'static,
    <Coordinate as nk::MeshAlignment>::Metric: Into<f64>,
{
    fn rmsd(query: &[Coordinate], node: &[Coordinate], points: usize) -> Option<f64> {
        let query = as_triples(query, points)?;
        let node = as_triples(node, points)?;
        <Coordinate as nk::MeshAlignment>::kabsch(query, node).map(|aligned| aligned.rmsd.into())
    }
}

impl<Coordinate> SuperposeLike<Coordinate> for RigidSuperpose
where
    Coordinate: nk::MeshAlignment + 'static,
    <Coordinate as nk::MeshAlignment>::Metric: Into<f64>,
{
    fn rmsd(query: &[Coordinate], node: &[Coordinate], points: usize) -> Option<f64> {
        let query = as_triples(query, points)?;
        let node = as_triples(node, points)?;
        <Coordinate as nk::MeshAlignment>::rmsd(query, node).map(|aligned| aligned.rmsd.into())
    }
}

impl<Coordinate> SuperposeLike<Coordinate> for SimilaritySuperpose
where
    Coordinate: nk::MeshAlignment + 'static,
    <Coordinate as nk::MeshAlignment>::Metric: Into<f64>,
{
    fn rmsd(query: &[Coordinate], node: &[Coordinate], points: usize) -> Option<f64> {
        let query = as_triples(query, points)?;
        let node = as_triples(node, points)?;
        <Coordinate as nk::MeshAlignment>::umeyama(query, node).map(|aligned| aligned.rmsd.into())
    }
}

// endregion: Superposition seam

// region: Metric

/// A mesh metric: a fixed-cardinality point cloud per node, an RMSD score, and a
/// query batch of like clouds. Each query is a `points*3`-wide row, so the dense
/// family's [`QueryBatchLike`] impl over `nk::MatrixView` serves unchanged.
pub struct MeshMetric<Coordinate, Superpose>(
    core::marker::PhantomData<fn() -> (Coordinate, Superpose)>,
);

impl<Coordinate, Superpose> MetricLike for MeshMetric<Coordinate, Superpose>
where
    Coordinate: nk::StorageElement + Copy + Default + 'static,
    Superpose: SuperposeLike<Coordinate>,
{
    type Query<'a> = &'a [Coordinate];
    type Queries<'a> = nk::MatrixView<'a, Coordinate>;
    /// The score is an RMSD, always `f64` — the superposition accumulates wider
    /// than its inputs, as `mesh.h` does, and every dtype's RMSD widens here.
    type Score = f64;
}

/// The cardinality of a mesh value column: `points`, requiring exactly three
/// coordinates per point. The single shape check every entry point repeats.
fn cloud_points(table: TableView<'_>) -> Result<usize, Error> {
    match table.value_shape() {
        // `Matrix(0, 3)` is the crate's ragged sentinel, not a zero-point cloud,
        // and a superposition needs at least three points — the floor every
        // `MeshAlignment` kernel enforces through its SVD. Checking both here
        // turns what would be an opaque `Inconsistent` on every node/query pair
        // into one `WrongType` at the shape boundary, where the mistake is.
        Some(Shape::Matrix(points, 3)) if points >= 3 => Ok(points),
        _ => Err(Error::WrongType),
    }
}

// endregion: Metric

// region: Scanner

/// The batch half of every mesh metric: owns a thread pool and the reusable
/// score tile. There is no packed form and no per-element broadcast — a
/// superposition consumes each node's interleaved cloud directly — so unlike
/// geospatial the scanner carries no per-worker scratch.
pub struct MeshScanner<Coordinate, Superpose, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    tile: Matrix<f64, Allocator>,
    alloc: Allocator,
    coordinate: core::marker::PhantomData<fn() -> Coordinate>,
    superpose: core::marker::PhantomData<fn() -> Superpose>,
}

impl<Coordinate, Superpose, Allocator> MeshScanner<Coordinate, Superpose, Allocator>
where
    Allocator: AllocatorLike + Clone,
{
    /// A scanner over the workers `scope` names, uniform with [`DotScanner::new`].
    /// NumKong's mesh kernels are CPU-only, so a GPU scope is refused.
    ///
    /// [`DotScanner::new`]: crate::search::dense::DotScanner::new
    pub fn new(scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(MeshScanner {
            pool: open_pool(scope)?,
            tile: Matrix::empty_in(alloc.clone()),
            alloc,
            coordinate: core::marker::PhantomData,
            superpose: core::marker::PhantomData,
        })
    }
}

impl<Coordinate, Superpose, Allocator> MeshScanner<Coordinate, Superpose, Allocator>
where
    Coordinate: nk::StorageElement + Copy + Default + Send + Sync + 'static,
    Superpose: SuperposeLike<Coordinate>,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    /// Batch brute-force top-k, nearest-first — an RMSD is smaller-is-better.
    /// `matches`/`distances` are `queries × k`; their shape carries both `k` and
    /// the query count, uniform with every dense scanner.
    pub fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Coordinate>,
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let table = table.view();
        let mut tile = std::mem::replace(&mut self.tile, Matrix::empty_in(self.alloc.clone()));
        let outcome = batch_scan::<MeshMetric<Coordinate, Superpose>, _>(
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

impl<Coordinate, Superpose, Allocator> ScannerLike<MeshMetric<Coordinate, Superpose>>
    for MeshScanner<Coordinate, Superpose, Allocator>
where
    Coordinate: nk::StorageElement + Copy + Default + Send + Sync + 'static,
    Superpose: SuperposeLike<Coordinate>,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    fn scan(
        &mut self,
        queries: nk::MatrixView<'_, Coordinate>,
        table: TableView<'_>,
        scores: &mut Matrix<f64, impl AllocatorLike + Clone>,
    ) -> Result<(), Error> {
        let points = cloud_points(table)?;
        let node_count = table.len();
        let query_count = queries.shape().first().copied().unwrap_or(0);
        if queries.shape().get(1) != Some(&(points * 3))
            || query_count != scores.cols()
            || node_count != scores.rows()
        {
            return Err(Error::Inconsistent);
        }

        // The mapped value column is contiguous `points*3` per node; read it once
        // as a flat slice and index each node's cloud by offset.
        let node_clouds: &[Coordinate] = cast(table.value_buffer()).ok_or(Error::WrongType)?;
        if node_clouds.len() != node_count * points * 3 {
            return Err(Error::Inconsistent);
        }

        // Fan NODES across the pool: the tile is node-major, so each task owns
        // its node's contiguous row by exclusive borrow — the borrow checker
        // proves the disjointness, and a superposition is per-pair anyway, so
        // no batch length is at stake. No per-worker scratch either: the task
        // reads shared slices and writes its own row.
        let mut rows: Vec<&mut [f64]> = Vec::new();
        rows.try_reserve_exact(node_count)
            .map_err(|_| Error::Alloc)?;
        rows.extend(scores.as_mut_slice().chunks_mut(query_count.max(1)));
        let queries_shared = ShareRead(&queries);
        let failure = OnceLock::new();

        fu::ParallelSliceMut::new(&mut rows).for_each_dynamic(&mut self.pool, |row, prong| {
            let queries = queries_shared.read();
            let outcome = scan_one_node::<Coordinate, Superpose>(
                queries,
                node_clouds,
                prong.task_index,
                points,
                row,
            );
            if let Err(error) = outcome {
                let _ = failure.set(error);
            }
        });

        match failure.into_inner() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// One node's superposition against every query — the body of the scan's
/// fan-out task, handed the node's contiguous tile row by exclusive borrow.
fn scan_one_node<Coordinate, Superpose>(
    queries: &nk::MatrixView<'_, Coordinate>,
    node_clouds: &[Coordinate],
    node: usize,
    points: usize,
    row: &mut [f64],
) -> Result<(), Error>
where
    Coordinate: nk::StorageElement + Copy + Default + 'static,
    Superpose: SuperposeLike<Coordinate>,
{
    let stride = points * 3;
    let cloud = &node_clouds[node * stride..node * stride + stride];
    for (query_index, cell) in row.iter_mut().enumerate() {
        let query = <nk::MatrixView<'_, Coordinate> as QueryBatchLike<
            MeshMetric<Coordinate, Superpose>,
        >>::query(queries, query_index)
        .ok_or(Error::Inconsistent)?;
        *cell = Superpose::rmsd(query, cloud, points).ok_or(Error::Inconsistent)?;
    }
    Ok(())
}

// endregion: Scanner

// region: Searcher

/// The traversal half of any mesh metric: owns the reusable scratch and scores
/// one expansion's staged nodes by superposing the query cloud onto each.
pub struct MeshSearcher<Coordinate, Superpose, Index = u32, Allocator: AllocatorLike = Global> {
    context: TraversalContext<Index, f64, Allocator>,
    marker: core::marker::PhantomData<fn() -> (Coordinate, Superpose)>,
}

impl<Coordinate, Superpose, Index, Allocator> MeshSearcher<Coordinate, Superpose, Index, Allocator>
where
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
{
    /// A searcher walking with breadth `expansion_factor`, uniform with
    /// [`DotSearcher::new`].
    ///
    /// [`DotSearcher::new`]: crate::search::dense::DotSearcher::new
    pub fn new(expansion_factor: usize, alloc: Allocator) -> Result<Self, Error> {
        nk::capabilities::configure_thread();
        Ok(MeshSearcher {
            context: TraversalContext::new_in(expansion_factor, alloc)?,
            marker: core::marker::PhantomData,
        })
    }

    /// The traversal scratch, for draining results after a walk.
    pub fn context(&mut self) -> &mut TraversalContext<Index, f64, Allocator> {
        &mut self.context
    }
}

impl<Coordinate, Superpose, Index, Allocator> SearcherLike<MeshMetric<Coordinate, Superpose>>
    for MeshSearcher<Coordinate, Superpose, Index, Allocator>
where
    Coordinate: nk::StorageElement + Copy + Default + 'static,
    Superpose: SuperposeLike<Coordinate>,
    Index: IndexLike,
    Allocator: AllocatorLike + Clone,
{
    type Index = Index;
    type Alloc = Allocator;

    fn score_nodes(
        &mut self,
        query: &[Coordinate],
        table: TableView<'_>,
        nodes: &[Index],
        out: &mut [f64],
    ) -> Result<(), Error> {
        if nodes.len() != out.len() {
            return Err(Error::Inconsistent);
        }
        let points = cloud_points(table)?;
        if query.len() != points * 3 {
            return Err(Error::Inconsistent);
        }
        for (slot, &node) in out.iter_mut().zip(nodes) {
            let index: usize = node.try_into().map_err(|_| Error::TooLarge)?;
            let cloud: &[Coordinate] = cast(table.value(index)).ok_or(Error::WrongType)?;
            *slot = Superpose::rmsd(query, cloud, points).ok_or(Error::Inconsistent)?;
        }
        Ok(())
    }

    fn context(&mut self) -> &mut TraversalContext<Index, f64, Allocator> {
        &mut self.context
    }

    fn compare(a: f64, b: f64) -> Ordering {
        Ranking::Nearest.compare(a, b)
    }

    fn search(&mut self, query: &[Coordinate], table: TableView<'_>) -> Result<(), Error> {
        walk::<MeshMetric<Coordinate, Superpose>, Self>(self, query, table)
    }
}

/// A mesh searcher whose neighbor width is settled by the tables it searches —
/// the family's width resolver, one leaf per pool thread, exactly as `DotSearch`
/// keeps them.
pub struct MeshSearch<Coordinate, Superpose, Allocator: AllocatorLike = Global> {
    pool: fu::ThreadPool,
    expansion_factor: usize,
    workers: Workers<
        MeshSearcher<Coordinate, Superpose, u32, Allocator>,
        MeshSearcher<Coordinate, Superpose, u64, Allocator>,
        Allocator,
    >,
    alloc: Allocator,
}

impl<Coordinate, Superpose, Allocator> MeshSearch<Coordinate, Superpose, Allocator>
where
    Coordinate: nk::StorageElement + Copy + Default + Send + Sync + 'static,
    Superpose: SuperposeLike<Coordinate>,
    Allocator: AllocatorLike + Clone + Send,
{
    /// A searcher walking with breadth `expansion_factor` across the workers
    /// `scope` names, uniform with [`DotSearch::new`].
    ///
    /// [`DotSearch::new`]: crate::search::dense::DotSearch::new
    pub fn new(
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(MeshSearch {
            pool: open_pool(scope)?,
            expansion_factor,
            workers: Workers::Unresolved,
            alloc,
        })
    }

    /// Batch top-k by walk, fanned across the pool. Uniform with
    /// [`MeshScanner::scan`].
    pub fn search(
        &mut self,
        queries: nk::MatrixView<'_, Coordinate>,
        table: &Table<impl AllocatorLike>,
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
                        MeshSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U32(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<MeshMetric<Coordinate, Superpose>, _>(
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
                        MeshSearcher::new(expansion_factor, alloc.clone())
                    })?);
                }
                let Workers::U64(workers) = &mut self.workers else {
                    return Err(Error::Inconsistent);
                };
                batch_search::<MeshMetric<Coordinate, Superpose>, _>(
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

// region: Metric aliases

/// Kabsch superposition — rotation + translation, nearest-first RMSD.
pub type RotationalMeshMetric<Coordinate> = MeshMetric<Coordinate, RotationalSuperpose>;
pub type RotationalMeshScanner<Coordinate, Allocator = Global> =
    MeshScanner<Coordinate, RotationalSuperpose, Allocator>;
pub type RotationalMeshSearch<Coordinate, Allocator = Global> =
    MeshSearch<Coordinate, RotationalSuperpose, Allocator>;

/// Raw RMSD — no alignment, nearest-first.
pub type RigidMeshMetric<Coordinate> = MeshMetric<Coordinate, RigidSuperpose>;
pub type RigidMeshScanner<Coordinate, Allocator = Global> =
    MeshScanner<Coordinate, RigidSuperpose, Allocator>;
pub type RigidMeshSearch<Coordinate, Allocator = Global> =
    MeshSearch<Coordinate, RigidSuperpose, Allocator>;

/// Umeyama superposition — rotation + translation + scale, nearest-first RMSD.
pub type SimilarityMeshMetric<Coordinate> = MeshMetric<Coordinate, SimilaritySuperpose>;
pub type SimilarityMeshScanner<Coordinate, Allocator = Global> =
    MeshScanner<Coordinate, SimilaritySuperpose, Allocator>;
pub type SimilarityMeshSearch<Coordinate, Allocator = Global> =
    MeshSearch<Coordinate, SimilaritySuperpose, Allocator>;

// endregion: Metric aliases

// region: Runtime dtype dispatch

/// A mesh scanner chosen from a file's value dtype string — clouds come in the
/// four coordinate dtypes NumKong's `mesh.h` ships. Generic over the
/// superposition, so one enum serves all three: the dtype axis is this type's
/// business, and which invariants to factor out stays the marker's.
pub enum MeshScannerAny<Superpose, Allocator: AllocatorLike = Global> {
    F64(MeshScanner<f64, Superpose, Allocator>),
    F32(MeshScanner<f32, Superpose, Allocator>),
    F16(MeshScanner<nk::f16, Superpose, Allocator>),
    Bf16(MeshScanner<nk::bf16, Superpose, Allocator>),
}

/// Raw-RMSD scanner over every coordinate dtype.
pub type RigidMeshScannerAny<Allocator = Global> = MeshScannerAny<RigidSuperpose, Allocator>;
/// Kabsch scanner over every coordinate dtype.
pub type RotationalMeshScannerAny<Allocator = Global> =
    MeshScannerAny<RotationalSuperpose, Allocator>;
/// Umeyama scanner over every coordinate dtype.
pub type SimilarityMeshScannerAny<Allocator = Global> =
    MeshScannerAny<SimilaritySuperpose, Allocator>;

impl<Superpose, Allocator> MeshScannerAny<Superpose, Allocator>
where
    Superpose:
        SuperposeLike<f64> + SuperposeLike<f32> + SuperposeLike<nk::f16> + SuperposeLike<nk::bf16>,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    pub fn new(dtype: &str, scope: ExecutionScope, alloc: Allocator) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => Self::F64(MeshScanner::new(scope, alloc)?),
            "f32" => Self::F32(MeshScanner::new(scope, alloc)?),
            "f16" => Self::F16(MeshScanner::new(scope, alloc)?),
            "bf16" => Self::Bf16(MeshScanner::new(scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            Self::F64(_) => "f64",
            Self::F32(_) => "f32",
            Self::F16(_) => "f16",
            Self::Bf16(_) => "bf16",
        }
    }

    pub fn scan(
        &mut self,
        queries_bytes: &[u8],
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let points = cloud_points(table.view())?;
        match self {
            Self::F64(s) => s.scan(
                view(queries_bytes, points)?,
                table,
                matches,
                distances,
                counts,
            ),
            Self::F32(s) => s.scan(
                view(queries_bytes, points)?,
                table,
                matches,
                distances,
                counts,
            ),
            Self::F16(s) => s.scan(
                view(queries_bytes, points)?,
                table,
                matches,
                distances,
                counts,
            ),
            Self::Bf16(s) => s.scan(
                view(queries_bytes, points)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// A mesh searcher chosen from a file's value dtype string — the walk twin of
/// [`MeshScannerAny`], generic over the superposition for the same reason.
pub enum MeshSearchAny<Superpose, Allocator: AllocatorLike = Global> {
    F64(MeshSearch<f64, Superpose, Allocator>),
    F32(MeshSearch<f32, Superpose, Allocator>),
    F16(MeshSearch<nk::f16, Superpose, Allocator>),
    Bf16(MeshSearch<nk::bf16, Superpose, Allocator>),
}

/// Raw-RMSD searcher over every coordinate dtype.
pub type RigidMeshSearchAny<Allocator = Global> = MeshSearchAny<RigidSuperpose, Allocator>;
/// Kabsch searcher over every coordinate dtype.
pub type RotationalMeshSearchAny<Allocator = Global> =
    MeshSearchAny<RotationalSuperpose, Allocator>;
/// Umeyama searcher over every coordinate dtype.
pub type SimilarityMeshSearchAny<Allocator = Global> =
    MeshSearchAny<SimilaritySuperpose, Allocator>;

impl<Superpose, Allocator> MeshSearchAny<Superpose, Allocator>
where
    Superpose:
        SuperposeLike<f64> + SuperposeLike<f32> + SuperposeLike<nk::f16> + SuperposeLike<nk::bf16>,
    Allocator: AllocatorLike + Clone + Send + Sync,
{
    pub fn new(
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match dtype {
            "f64" => Self::F64(MeshSearch::new(expansion_factor, scope, alloc)?),
            "f32" => Self::F32(MeshSearch::new(expansion_factor, scope, alloc)?),
            "f16" => Self::F16(MeshSearch::new(expansion_factor, scope, alloc)?),
            "bf16" => Self::Bf16(MeshSearch::new(expansion_factor, scope, alloc)?),
            _ => return Err(Error::UnknownDType),
        })
    }

    pub fn dtype(&self) -> &'static str {
        match self {
            Self::F64(_) => "f64",
            Self::F32(_) => "f32",
            Self::F16(_) => "f16",
            Self::Bf16(_) => "bf16",
        }
    }

    pub fn search(
        &mut self,
        queries_bytes: &[u8],
        table: &Table<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        let points = cloud_points(table.view())?;
        match self {
            Self::F64(s) => s.search(
                view(queries_bytes, points)?,
                table,
                matches,
                distances,
                counts,
            ),
            Self::F32(s) => s.search(
                view(queries_bytes, points)?,
                table,
                matches,
                distances,
                counts,
            ),
            Self::F16(s) => s.search(
                view(queries_bytes, points)?,
                table,
                matches,
                distances,
                counts,
            ),
            Self::Bf16(s) => s.search(
                view(queries_bytes, points)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

/// Reinterpret raw query bytes as a `queries × (points*3)` matrix view over the
/// coordinate dtype — the mesh counterpart of dense's `view_from_bytes`, with the
/// row width being one whole cloud rather than a single vector. Reuses dense's
/// `matrix_view`, so sub-byte `dimensions_per_value` packing is handled the same.
fn view<Coordinate: nk::StorageElement>(
    bytes: &[u8],
    points: usize,
) -> Result<nk::MatrixView<'_, Coordinate>, Error> {
    matrix_view(cast(bytes).ok_or(Error::WrongType)?, points * 3)
}

// endregion: Runtime dtype dispatch

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphLike, NeighborList, Shape};

    /// Three point clouds of `points = 3`, plus a query. Node 1 is the query
    /// exactly (RMSD 0), node 0 is it shifted +5 on x (RMSD 5), node 2 shifted
    /// +10 (RMSD 10) — so a raw (rigid) RMSD ranks `[1, 0, 2]`.
    struct Clouds {
        values: [[f32; 9]; 3],
    }

    fn clouds() -> Clouds {
        let base = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let shift = |dx: f32| {
            let mut row = base;
            for point in 0..3 {
                row[point * 3] += dx;
            }
            row
        };
        Clouds {
            values: [shift(5.0), base, shift(10.0)],
        }
    }

    fn query() -> [f32; 9] {
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    }

    impl GraphLike for Clouds {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "f32"
        }
        fn value_shape(&self) -> Option<Shape> {
            Some(Shape::Matrix(3, 3))
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
            let row = &self.values[i];
            // SAFETY: reinterpret the contiguous `[f32; 9]` cloud as its bytes.
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
            NeighborList::new(&ADJACENCY[i], OffsetDType::U32)
        }
    }

    fn outputs(queries: usize, k: usize) -> (Matrix<u64>, Matrix<f64>, Vec<u64>) {
        (
            Matrix::<u64>::new(queries, k).unwrap(),
            Matrix::<f64>::new(queries, k).unwrap(),
            vec![0u64; queries],
        )
    }

    #[test]
    fn rigid_scan_ranks_identical_cloud_first() {
        let table = crate::Table::build(&clouds(), None).unwrap();
        let mut scanner = RigidMeshScanner::<f32>::new(ExecutionScope::Serial, Global).unwrap();
        let queries = nk::Matrix::<f32>::try_from_scalars(&query(), &[1, 9]).unwrap();
        let (mut matches, mut distances, mut counts) = outputs(1, 3);
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
        assert_eq!(matches.row(0).unwrap(), &[1, 0, 2]);
        let row = distances.row(0).unwrap();
        assert!(row[0].abs() < 1e-4);
        assert!(row[0] < row[1] && row[1] < row[2]);
    }

    #[test]
    fn rotational_walk_agrees_with_scan() {
        let table = crate::Table::build(&clouds(), None).unwrap();
        let mut scanner =
            RotationalMeshScanner::<f32>::new(ExecutionScope::Serial, Global).unwrap();
        let mut searcher =
            RotationalMeshSearch::<f32>::new(3, ExecutionScope::Serial, Global).unwrap();
        let queries = nk::Matrix::<f32>::try_from_scalars(&query(), &[1, 9]).unwrap();

        let (mut sm, mut sd, mut sc) = outputs(1, 3);
        scanner
            .scan(queries.view(), &table, &mut sm, &mut sd, &mut sc)
            .unwrap();
        let (mut wm, mut wd, mut wc) = outputs(1, 3);
        searcher
            .search(queries.view(), &table, &mut wm, &mut wd, &mut wc)
            .unwrap();
        // Kabsch removes the translation, so every cloud aligns to ~0 RMSD; the
        // walk and the scan must still agree on the top result and counts.
        assert_eq!(sc, wc);
        assert_eq!(sm.row(0).unwrap()[0], wm.row(0).unwrap()[0]);
    }

    #[test]
    fn dtype_dispatch_round_trips_every_arm() {
        for dtype in ["f64", "f32", "f16", "bf16"] {
            let any = RotationalMeshScannerAny::new(dtype, ExecutionScope::Serial, Global).unwrap();
            assert_eq!(any.dtype(), dtype);
        }
        assert!(RotationalMeshScannerAny::new("i8", ExecutionScope::Serial, Global).is_err());
    }
}

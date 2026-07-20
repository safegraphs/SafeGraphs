//! Dense-vector metrics. A query batch is packed once in its native dtype `T`
//! (via [`prepare`]); [`ScoringDot`] then scans a table by streaming its value
//! column zero-copy from the memory map and scoring it in one NumKong
//! cross-product, or walks the neighbor graph vector-by-vector with `nk::Dot`.
//! Scores land in `T`'s native accumulator (`f64` for `f32`, `f32` for
//! `bf16`/`e4m3`, `i32` for `i8`) — the huge value side is never packed or
//! widened. Every NumKong bound lives here, not on the [`ScoringLike`] trait.

use super::{rank_tile, DTypeName, Match, ScoringLike, TraversalContext};
use crate::{cast, Error, GraphLike, Table};
use allocator_api2::alloc::{Allocator, Global};
use core::cmp::Ordering;
use nk::DotsPackedOps;
use numkong as nk;

/// Build a zero-copy `queries × dimension` view over caller-owned native rows.
pub fn query_view<T: DTypeName>(
    rows: &[T],
    dimension: usize,
) -> Result<nk::MatrixView<'_, T>, Error> {
    if dimension == 0 || rows.is_empty() || rows.len() % dimension != 0 {
        return Err(Error::Inconsistent);
    }
    let queries = rows.len() / dimension;
    let size = core::mem::size_of::<T>() as isize;
    // SAFETY: `rows` is `queries * dimension` contiguous `T`, so the shape and
    // unit strides describe exactly its extent, and the view borrows it for `'_`.
    Ok(unsafe {
        nk::MatrixView::<T>::from_raw_parts(
            rows.as_ptr(),
            &[queries, dimension],
            &[dimension as isize * size, size],
        )
    })
}

/// Eagerly pack a native query batch, once, for reuse across tables and shards.
/// The packed batch is small (`queries × dimension`) and lives on NumKong's own
/// heap, since its packer is tied to NumKong's allocator.
pub fn prepare<T: DTypeName>(
    queries: &nk::MatrixView<'_, T>,
) -> Result<nk::PackedMatrix<T>, Error> {
    nk::PackedMatrix::try_pack(queries).map_err(|_| Error::Inconsistent)
}

/// A zero-copy `nodes × dimension` view of a table's value column, for the scan.
fn value_view<T: DTypeName>(
    table: &Table<impl Allocator>,
    nodes: usize,
    dimension: usize,
) -> Result<nk::MatrixView<'_, T>, Error> {
    let scalars: &[T] = cast(table.value_buffer()).ok_or(Error::WrongType)?;
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

/// Inner-product similarity (larger is more similar), together with the reusable
/// traversal scratch it scans and walks with. Generic over the value scalar `T`,
/// the allocator `A`, and the node-index width `I`; the score is `T`'s native dot
/// accumulator (`<T as nk::Dot>::Output`).
pub struct ScoringDot<T, A = Global, I = u32>
where
    T: nk::Dot,
    A: Allocator,
{
    context: TraversalContext<A, I, <T as nk::Dot>::Output>,
}

impl<T: nk::Dot> ScoringDot<T, Global, u32> {
    /// A scorer sized for `nodes` nodes on the global heap, with `u32` indices.
    pub fn new(nodes: usize) -> Result<Self, Error> {
        ScoringDot::new_in(nodes, Global)
    }
}

impl<T: nk::Dot, A: Allocator + Clone, I> ScoringDot<T, A, I> {
    /// A scorer sized for `nodes` nodes, allocating its scratch in `alloc`.
    pub fn new_in(nodes: usize, alloc: A) -> Result<Self, Error> {
        Ok(ScoringDot {
            context: TraversalContext::new_in(nodes, alloc)?,
        })
    }
}

impl<T, A, I> ScoringLike for ScoringDot<T, A, I>
where
    T: DTypeName + nk::Dot + nk::Dots<Accumulator = <T as nk::Dot>::Output>,
    A: Allocator + Clone,
    I: Copy + TryFrom<usize> + TryInto<usize>,
    <T as nk::Dot>::Output: nk::StorageElement + PartialOrd,
{
    type Item<'a>
        = &'a [T]
    where
        Self: 'a;
    type Packed = nk::PackedMatrix<T>;
    type Score = <T as nk::Dot>::Output;
    type Alloc = A;
    type Index = I;

    fn view<'a>(&self, bytes: &'a [u8]) -> Option<&'a [T]> {
        cast(bytes)
    }

    fn score(&self, query: &[T], value: &[T]) -> Option<Self::Score> {
        <T as nk::Dot>::dot(query, value)
    }

    fn compare(a: Self::Score, b: Self::Score) -> Ordering {
        // Higher dot product ranks first; NaN sinks to the bottom.
        a.partial_cmp(&b).unwrap_or(Ordering::Less)
    }

    fn context(&mut self) -> &mut TraversalContext<A, I, Self::Score> {
        &mut self.context
    }

    fn scan(
        &mut self,
        packed: &nk::PackedMatrix<T>,
        table: &Table<impl Allocator>,
        k: usize,
        out: &mut [Match<I, Self::Score>],
    ) -> Result<(), Error> {
        if T::DTYPE != table.value_dtype() {
            return Err(Error::WrongType);
        }
        let nodes = table.len();
        let dimension = table.value_shape().ok_or(Error::Inconsistent)?;
        let queries = packed.dims().0;
        if k == 0 || out.len() != queries * k {
            return Err(Error::Inconsistent);
        }
        let values = value_view::<T>(table, nodes, dimension)?;

        // Fill the reusable `nodes × queries` tile in place: the value side
        // streams from the memory map, only the packed query side was prepared.
        let cells = nodes.checked_mul(queries).ok_or(Error::TooLarge)?;
        {
            let scratch = self.context();
            scratch.scores.clear();
            scratch
                .scores
                .try_reserve(cells)
                .map_err(|_| Error::Alloc)?;
            scratch.scores.resize(cells, Self::Score::default());
        }
        {
            let element_bytes = core::mem::size_of::<Self::Score>() as isize;
            let tile_ptr = self.context().scores.as_mut_ptr();
            let mut tile = unsafe {
                nk::MatrixSpan::<Self::Score>::from_raw_parts(
                    tile_ptr,
                    &[nodes, queries],
                    &[queries as isize * element_bytes, element_bytes],
                )
            };
            values
                .try_dots_packed_into(packed, &mut tile)
                .map_err(|_| Error::Inconsistent)?;
        }

        rank_tile(
            &self.context().scores,
            nodes,
            queries,
            k,
            Self::compare,
            out,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::ScoringLike;
    use crate::{GraphLike, NeighborList, OffsetDType};
    use allocator_api2::alloc::Global;

    /// Three 2-D vectors `[1,2] [3,4] [5,6]` as `i8`, fully connected so a walk
    /// from any root reaches every node.
    struct Ring;
    impl GraphLike for Ring {
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
            const VALUES: [[u8; 2]; 3] = [[1, 2], [3, 4], [5, 6]];
            &VALUES[i]
        }
        fn neighbors(&self, i: usize) -> NeighborList<'_> {
            // node i links to the other two, as little-endian `u32` indices.
            const ADJACENCY: [[u8; 8]; 3] = [
                [1, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ];
            NeighborList::new(&ADJACENCY[i], OffsetDType::U32)
        }
    }

    #[test]
    fn scan_i8_scores_in_i32() {
        let table = Table::build(&Ring, None).unwrap();
        let rows = [3i8, 4];
        let packed = prepare(&query_view(&rows, 2).unwrap()).unwrap();
        let mut scorer: ScoringDot<i8> = ScoringDot::new(table.len()).unwrap();
        let mut out = [Match {
            node: 0u32,
            score: 0i32,
        }; 2];

        scorer.scan(&packed, &table, 2, &mut out).unwrap();

        // Dots against [3,4]: node0=11, node1=25, node2=39 → best is node2.
        assert_eq!(out[0], Match { node: 2, score: 39 });
        assert_eq!(out[1], Match { node: 1, score: 25 });
    }

    #[test]
    fn search_i8_walks_to_best() {
        // The same top-k, reached by an A* walk from node 0 instead of a full scan.
        let table = Table::build(&Ring, None).unwrap();
        let query = [3i8, 4];
        let mut scorer: ScoringDot<i8> = ScoringDot::new(table.len()).unwrap();
        let mut out = [Match {
            node: 0u32,
            score: 0i32,
        }; 2];

        scorer
            .search(&query, &table, &[0u32], 3, 2, &mut out)
            .unwrap();

        assert_eq!(out[0], Match { node: 2, score: 39 });
        assert_eq!(out[1], Match { node: 1, score: 25 });
    }

    #[test]
    fn scan_f32_scores_in_f64() {
        // f32 values need 8-byte rows, so this fixture carries its own bytes.
        struct RingF32;
        impl GraphLike for RingF32 {
            fn len(&self) -> usize {
                3
            }
            fn key_dtype(&self) -> &str {
                "u64"
            }
            fn value_dtype(&self) -> &str {
                "f32"
            }
            fn value_shape(&self) -> Option<usize> {
                Some(2)
            }
            fn neighbors_degree(&self) -> Option<usize> {
                Some(0)
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
                // Little-endian bytes of [1,2] [3,4] [5,6] as `f32` pairs.
                const VALUES: [[u8; 8]; 3] = [
                    [0, 0, 0x80, 0x3F, 0, 0, 0, 0x40],
                    [0, 0, 0x40, 0x40, 0, 0, 0x80, 0x40],
                    [0, 0, 0xA0, 0x40, 0, 0, 0xC0, 0x40],
                ];
                &VALUES[i]
            }
            fn neighbors(&self, _i: usize) -> NeighborList<'_> {
                NeighborList::new(&[], OffsetDType::U32)
            }
        }

        let table = Table::build(&RingF32, None).unwrap();
        let rows = [3.0f32, 4.0];
        let packed = prepare(&query_view(&rows, 2).unwrap()).unwrap();
        let mut scorer: ScoringDot<f32, Global, u64> =
            ScoringDot::new_in(table.len(), Global).unwrap();
        let mut out = [Match {
            node: 0u64,
            score: 0.0f64,
        }; 2];

        scorer.scan(&packed, &table, 2, &mut out).unwrap();

        assert_eq!(
            out[0],
            Match {
                node: 2,
                score: 39.0
            }
        );
        assert_eq!(
            out[1],
            Match {
                node: 1,
                score: 25.0
            }
        );
    }

    /// A degree-0 fixture over an arbitrary dtype, its value bytes supplied raw.
    struct Rows {
        dtype: &'static str,
        rows: [[u8; 2]; 3],
    }
    impl GraphLike for Rows {
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
            Some(2)
        }
        fn neighbors_degree(&self) -> Option<usize> {
            Some(0)
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
            &self.rows[i]
        }
        fn neighbors(&self, _i: usize) -> NeighborList<'_> {
            NeighborList::new(&[], OffsetDType::U32)
        }
    }

    #[test]
    fn scan_u8_scores_in_u32() {
        // Unsigned kernel: u8 values and queries, dot products accumulated in u32.
        let table = Table::build(
            &Rows {
                dtype: "u8",
                rows: [[1, 2], [3, 4], [5, 6]],
            },
            None,
        )
        .unwrap();
        let rows = [3u8, 4];
        let packed = prepare(&query_view(&rows, 2).unwrap()).unwrap();
        let mut scorer: ScoringDot<u8> = ScoringDot::new(table.len()).unwrap();
        let mut out = [Match {
            node: 0u32,
            score: 0u32,
        }; 2];

        scorer.scan(&packed, &table, 2, &mut out).unwrap();

        assert_eq!(out[0], Match { node: 2, score: 39 });
        assert_eq!(out[1], Match { node: 1, score: 25 });
    }

    #[test]
    fn scan_e3m2_scores_in_f32() {
        // Exotic 6-bit float: 1..6 are exact in e3m2, so the dots are exact in f32.
        let byte =
            |value: f32| -> u8 { unsafe { core::mem::transmute(nk::e3m2::from_f32(value)) } };
        let table = Table::build(
            &Rows {
                dtype: "e3m2",
                rows: [
                    [byte(1.0), byte(2.0)],
                    [byte(3.0), byte(4.0)],
                    [byte(5.0), byte(6.0)],
                ],
            },
            None,
        )
        .unwrap();
        let rows = [nk::e3m2::from_f32(3.0), nk::e3m2::from_f32(4.0)];
        let packed = prepare(&query_view(&rows, 2).unwrap()).unwrap();
        let mut scorer: ScoringDot<nk::e3m2> = ScoringDot::new(table.len()).unwrap();
        let mut out = [Match {
            node: 0u32,
            score: 0.0f32,
        }; 2];

        scorer.scan(&packed, &table, 2, &mut out).unwrap();

        assert_eq!(
            out[0],
            Match {
                node: 2,
                score: 39.0
            }
        );
        assert_eq!(
            out[1],
            Match {
                node: 1,
                score: 25.0
            }
        );
    }
}

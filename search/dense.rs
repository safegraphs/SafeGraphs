//! Dense-vector metrics. A query batch is packed once in its native dtype `T`
//! (via [`prepare`]); [`ScoringDot::scan`] then streams a table's value column
//! straight from the memory map as a zero-copy `nk::MatrixView<T>` and scores it
//! in one cross-product, while [`ScoringDot::search`] walks the neighbor graph
//! scoring one node at a time with the vector-vector `nk::Dot`. Scores land in
//! `T`'s native accumulator (`f64` for `f32`, `f32` for `bf16`/`e4m3`, `i32` for
//! `i8`) — the huge value side is never packed or widened.

use super::{DTypeName, ScoringLike, TraversalContext};
use crate::{cast, Error};
use allocator_api2::alloc::{Allocator, Global};
use core::cmp::Ordering;
use numkong as nk;
use nk::DotsPackedOps;

/// Build a zero-copy `queries × dimension` view over caller-owned native rows.
pub fn query_view<T: DTypeName>(rows: &[T], dimension: usize) -> Result<nk::MatrixView<'_, T>, Error> {
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
pub fn prepare<T: DTypeName>(queries: &nk::MatrixView<'_, T>) -> Result<nk::PackedMatrix<T>, Error> {
    nk::PackedMatrix::try_pack(queries).map_err(|_| Error::Inconsistent)
}

/// Inner-product similarity (larger is more similar), together with the reusable
/// traversal scratch it scans and walks with. Generic over the allocator `A`,
/// the node-index width `I`, and the score `S` (the native dot accumulator).
pub struct ScoringDot<A: Allocator = Global, I = u32, S = f64> {
    context: TraversalContext<A, I, S>,
}

impl ScoringDot<Global, u32, f64> {
    /// A scorer sized for `nodes` nodes on the global heap, with `u32` node
    /// indices and `f64` scores (the accumulator for `f32` values).
    pub fn new(nodes: usize) -> Result<Self, Error> {
        ScoringDot::new_in(nodes, Global)
    }
}

impl<A: Allocator + Clone, I, S> ScoringDot<A, I, S> {
    /// A scorer sized for `nodes` nodes, allocating its scratch in `alloc`.
    pub fn new_in(nodes: usize, alloc: A) -> Result<Self, Error> {
        Ok(ScoringDot {
            context: TraversalContext::new_in(nodes, alloc)?,
        })
    }
}

impl<T, A, I, S> ScoringLike<T> for ScoringDot<A, I, S>
where
    T: DTypeName + nk::Dot<Output = S> + nk::Dots<Accumulator = S>,
    A: Allocator + Clone,
    I: Copy + TryFrom<usize> + TryInto<usize>,
    S: nk::StorageElement + PartialOrd,
{
    type Score = S;
    type Alloc = A;
    type Index = I;

    fn cross(
        &mut self,
        queries: &nk::PackedMatrix<T>,
        values: &nk::MatrixView<'_, T>,
        matches: &mut nk::MatrixSpan<'_, S>,
    ) -> Result<(), Error> {
        // `matches[node][query] = dot(value_node, query)`; the value side streams
        // from the memory map, only the packed query side was materialized.
        values
            .try_dots_packed_into(queries, matches)
            .map_err(|_| Error::Inconsistent)
    }

    fn score(&self, query: &[T], value: &[T]) -> Option<S> {
        <T as nk::Dot>::dot(query, value)
    }

    fn compare(a: S, b: S) -> Ordering {
        // Higher dot product ranks first; NaN sinks to the bottom.
        a.partial_cmp(&b).unwrap_or(Ordering::Less)
    }

    fn context(&mut self) -> &mut TraversalContext<A, I, S> {
        &mut self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{Match, ScoringLike};
    use crate::{GraphLike, NeighborList, OffsetDType, Table};
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
        let mut scorer: ScoringDot<Global, u32, i32> = ScoringDot::new_in(table.len(), Global).unwrap();
        let mut out = [Match { node: 0u32, score: 0i32 }; 2];

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
        let mut scorer: ScoringDot<Global, u32, i32> = ScoringDot::new_in(table.len(), Global).unwrap();
        let mut out = [Match { node: 0u32, score: 0i32 }; 2];

        scorer.search(&query, &table, &[0u32], 3, 2, &mut out).unwrap();

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
        let mut scorer: ScoringDot<Global, u64, f64> = ScoringDot::new_in(table.len(), Global).unwrap();
        let mut out = [Match { node: 0u64, score: 0.0f64 }; 2];

        scorer.scan(&packed, &table, 2, &mut out).unwrap();

        assert_eq!(out[0], Match { node: 2, score: 39.0 });
        assert_eq!(out[1], Match { node: 1, score: 25.0 });
    }
}

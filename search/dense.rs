//! Dense-vector metrics: each node's value is a fixed-width row, scored by
//! NumKong. Inner product today; Euclidean, Cosine, Hamming, and Jaccard next.

use super::ScoringLike;
use crate::{cast, Error, GraphLike, Table};
use allocator_api2::alloc::Allocator;
use core::cmp::Ordering;
use numkong::{DotsPackedOps, Matrix, MatrixSpan, PackedMatrix};

/// Inner-product similarity: larger is more similar. Backs BERT-style dense
/// embeddings compared by dot product.
pub struct Dot;

impl ScoringLike for Dot {
    type Query<'a> = &'a Matrix<f32>;
    // NumKong accumulates dot products in double precision.
    type Score = f64;

    fn query_count(queries: &Self::Query<'_>) -> usize {
        queries.shape()[0]
    }

    fn cross<A: Allocator>(
        &self,
        table: &Table<A>,
        queries: Self::Query<'_>,
        out: &mut MatrixSpan<'_, f64>,
    ) -> Result<(), Error> {
        // Reinterpret the value column as a dense `nodes × dim` matrix.
        let nodes = table.len();
        let dim = table.value_shape().ok_or(Error::Inconsistent)?;
        let scalars: &[f32] = cast(table.value_buffer()).ok_or(Error::WrongType)?;
        if scalars.len() != nodes * dim {
            return Err(Error::Inconsistent);
        }
        let values = Matrix::try_from_slice(scalars, &[nodes, dim]).map_err(|_| Error::Inconsistent)?;

        // Pack the values once, then multiply queries × valuesᵀ into `out`.
        let packed = PackedMatrix::try_pack(&values).map_err(|_| Error::Inconsistent)?;
        queries
            .try_dots_packed_into(&packed, out)
            .map_err(|_| Error::Inconsistent)
    }

    fn cmp(&self, a: f64, b: f64) -> Ordering {
        // Higher dot product ranks first; NaN sinks to the bottom.
        a.partial_cmp(&b).unwrap_or(Ordering::Less)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{Context, Match};
    use crate::{NeighborList, OffsetDType};
    use allocator_api2::alloc::Global;

    /// Three 2-D vectors: `[1,2]`, `[3,4]`, `[5,6]`.
    struct Toy;
    impl GraphLike for Toy {
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
            const VALUES: [[u8; 8]; 3] = [
                unsafe { core::mem::transmute::<[f32; 2], [u8; 8]>([1.0, 2.0]) },
                unsafe { core::mem::transmute::<[f32; 2], [u8; 8]>([3.0, 4.0]) },
                unsafe { core::mem::transmute::<[f32; 2], [u8; 8]>([5.0, 6.0]) },
            ];
            &VALUES[i]
        }
        fn neighbors(&self, _i: usize) -> NeighborList<'_> {
            NeighborList::new(&[], OffsetDType::U32)
        }
    }

    #[test]
    fn scan_ranks_by_dot() {
        let table = Table::build(&Toy, None).unwrap();
        let queries = Matrix::try_from_slice(&[3.0f32, 4.0], &[1, 2]).unwrap();
        let mut ctx: Context<Global, u64, f64> = Context::new_in(table.len(), Global).unwrap();
        let mut out = [Match { node: 0u64, score: 0.0f64 }; 2];

        table.scan(&queries, &Dot, 2, &mut ctx, &mut out).unwrap();

        // Dots against [3,4]: node0=11, node1=25, node2=39 → best is node2.
        assert_eq!(out[0], Match { node: 2, score: 39.0 });
        assert_eq!(out[1], Match { node: 1, score: 25.0 });
    }
}

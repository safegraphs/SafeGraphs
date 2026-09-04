//! Test-only helpers shared by every metric family.
//!
//! This is the one module that names NumKong outside a NumKong family, and it
//! does so only in a test build: comparing scores wants a tolerance, and
//! [`nk::is_close`] is the one the sibling crates already agree on, so inventing
//! a second formula here would be the divergence rather than the discipline.

use super::Matrix;
use numkong as nk;

/// Absolute floor, so a score expected to be exactly zero tolerates a denormal.
const ABSOLUTE_TOLERANCE: f64 = 1e-9;

/// Relative slack, a little wider than one `f32` ulp near one.
const RELATIVE_TOLERANCE: f64 = 1e-6;

/// Compare a row of scores against its expected values.
///
/// A packed kernel and a per-pair kernel compute the same quantity by different
/// routes, so they need not agree in the last bit, and an exact comparison turns
/// that into a spurious failure. Node IDs are integers and still compare exactly.
pub(crate) fn assert_scores_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "score row length");
    for (got, want) in actual.iter().zip(expected) {
        assert!(
            nk::is_close(*got, *want, ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE),
            "scores differ beyond tolerance: {actual:?} vs {expected:?}"
        );
    }
}

/// The `queries × k` outputs plus per-query counts every batch call writes into.
pub(crate) fn outputs(queries: usize, k: usize) -> (Matrix<u64>, Matrix<f64>, Vec<u64>) {
    (
        Matrix::<u64>::new(queries, k).unwrap(),
        Matrix::<f64>::new(queries, k).unwrap(),
        vec![0u64; queries],
    )
}

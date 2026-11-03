//! Query batches from JSON, in the shape [`dispatch`](super::dispatch) consumes.
//!
//! A query is a value, so a batch of them is a column: bytes in the table's own
//! dtype, plus a prefix sum when the rows are ragged.
//! This module is the one place that conversion lives, so a front end — the `sg`
//! binary, or any embedder accepting user input — does not reimplement it.
//!
//! The element type follows the metric, not the caller's preference: strings for
//! a sequence metric, nested arrays for point clouds and maxsim multi-vectors,
//! flat arrays otherwise.
//! Numbers always convert to the column's dtype, since the target type is known
//! from the file — there is no match-exactly case to fail on.

use crate::{Error, Shape};
use core::fmt;

// region: Errors

/// Why a JSON batch could not become a query column.
///
/// Separate from [`Error`], which names failures of a file and carries no
/// position: a malformed query is the caller's own input, and saying *which*
/// query and *what* was wrong is most of the value of reporting it at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryError {
    /// The document did not parse as JSON.
    Malformed,
    /// The document parsed, but the top level is not an array of queries.
    NotAnArray,
    /// Query `index` was not a number or an array of numbers.
    NotNumeric { index: usize },
    /// A sequence metric's queries must be JSON strings.
    NotAString { index: usize },
    /// Query `index` held `found` values where the column is `expected` wide.
    WrongWidth {
        index: usize,
        found: usize,
        expected: usize,
    },
    /// Query `index` is not a whole number of `depth`-wide tokens.
    RaggedRemainder { index: usize, depth: usize },
    /// A value was `NaN` or infinite, which no score can survive.
    NotFinite { index: usize },
    /// JSON cannot yet be encoded into this column's dtype.
    UnsupportedDType,
}

impl fmt::Display for QueryError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryError::Malformed => write!(out, "queries did not parse as JSON"),
            QueryError::NotAnArray => write!(out, "queries must be a JSON array"),
            QueryError::NotNumeric { index } => {
                write!(out, "query {index} must be a number or an array of numbers")
            }
            QueryError::NotAString { index } => {
                write!(out, "query {index} must be a JSON string for this metric")
            }
            QueryError::WrongWidth {
                index,
                found,
                expected,
            } => write!(
                out,
                "query {index} has {found} value(s), but the column is {expected} wide"
            ),
            QueryError::RaggedRemainder { index, depth } => write!(
                out,
                "query {index} is not a whole number of {depth}-wide tokens"
            ),
            QueryError::NotFinite { index } => {
                write!(out, "query {index} holds a value that is not finite")
            }
            QueryError::UnsupportedDType => {
                write!(out, "JSON queries cannot encode this column's dtype yet")
            }
        }
    }
}

impl std::error::Error for QueryError {}

impl From<QueryError> for Error {
    /// Collapse to the crate's own error where a caller wants one type.
    /// The detail is worth keeping, so prefer reporting [`QueryError`] itself.
    fn from(_: QueryError) -> Self {
        Error::Inconsistent
    }
}

// endregion: Errors

// region: Batches

/// A query batch in the router's input shape.
///
/// A fixed-width metric reads `bytes` through the table's shape; a ragged one
/// (sequence, maxsim, shapes) reads `offsets` beside it — byte offsets for a
/// sequence tape, token counts for a multi-vector, scalar counts for a cloud.
#[derive(Debug)]
pub struct Queries {
    bytes: Vec<u8>,
    offsets: Option<Vec<u64>>,
    count: usize,
}

impl Queries {
    /// The encoded batch, in the column's dtype.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The prefix sum a ragged metric needs, or `None` for a fixed-width one.
    pub fn offsets(&self) -> Option<&[u64]> {
        self.offsets.as_deref()
    }

    /// How many queries the batch holds.
    pub fn count(&self) -> usize {
        self.count
    }
}

// endregion: Batches

// region: Encoding

/// True when the metric's queries are strings rather than numbers.
fn is_textual(metric: &str) -> bool {
    matches!(metric, "levenshtein" | "levenshtein-utf8")
}

/// Encode one scalar into `dtype`'s storage width, appending to `out`.
fn push_scalar(value: f64, dtype: &str, index: usize, out: &mut Vec<u8>) -> Result<(), QueryError> {
    // JSON has no `NaN` or `Infinity` literal, so a non-finite value can only
    // have arrived by overflowing a finite one like `1e400`. It would poison
    // every score it touched, so it is refused here, where the caller can still
    // be told which query held it.
    if !value.is_finite() {
        return Err(QueryError::NotFinite { index });
    }
    match dtype {
        "f64" => out.extend_from_slice(&value.to_le_bytes()),
        "f32" => out.extend_from_slice(&(value as f32).to_le_bytes()),
        "f16" => out.extend_from_slice(&numkong::f16::from_f32(value as f32).0.to_le_bytes()),
        "bf16" => out.extend_from_slice(&numkong::bf16::from_f32(value as f32).0.to_le_bytes()),
        "i8" => out.push(value.clamp(-128.0, 127.0) as i8 as u8),
        "u8" => out.push(value.clamp(0.0, 255.0) as u8),
        _ => return Err(QueryError::UnsupportedDType),
    }
    Ok(())
}

/// Pack a row of 0/1 values into bits, least-significant bit first — the order a
/// `u1x8` column stores them in.
fn push_bits(row: &[f64], out: &mut Vec<u8>) {
    for chunk in row.chunks(8) {
        let mut byte = 0u8;
        for (bit, &value) in chunk.iter().enumerate() {
            if value != 0.0 {
                byte |= 1 << bit;
            }
        }
        out.push(byte);
    }
}

/// Every number under `value`, flattened depth-first.
/// A dense query is one level deep and a point cloud two, and both end as the
/// same flat row — the column's shape, not the nesting, decides what is legal.
fn flatten(
    value: &jsontape::Json<std::alloc::System>,
    index: usize,
    out: &mut Vec<f64>,
) -> Result<(), QueryError> {
    if let Some(number) = value.as_f64() {
        out.push(number);
        return Ok(());
    }
    // A number that has no `f64` reading is one whose magnitude the type cannot
    // hold — `1e400` and friends. It is numeric, so saying "not a number" would
    // misname it, and it is out of range, so no score could survive it either.
    if value.as_number_str().is_some() {
        return Err(QueryError::NotFinite { index });
    }
    let items = value.as_array().ok_or(QueryError::NotNumeric { index })?;
    for item in items {
        flatten(item, index, out)?;
    }
    Ok(())
}

/// Parse a JSON batch against a column's `metric`, `dtype`, and [`Shape`].
pub fn parse(text: &[u8], metric: &str, dtype: &str, shape: Shape) -> Result<Queries, QueryError> {
    let document =
        jsontape::parse_in(text, std::alloc::System).map_err(|_| QueryError::Malformed)?;
    let items = document.as_array().ok_or(QueryError::NotAnArray)?;

    // A sequence metric's queries are strings: their bytes concatenated with a
    // byte-offset prefix sum — exactly how the format stores an irregular column.
    if is_textual(metric) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u64];
        for (index, item) in items.iter().enumerate() {
            let text = item.as_str().ok_or(QueryError::NotAString { index })?;
            bytes.extend_from_slice(text.as_bytes());
            offsets.push(bytes.len() as u64);
        }
        return Ok(Queries {
            bytes,
            offsets: Some(offsets),
            count: items.len(),
        });
    }

    // Every other metric is numeric. MaxSim's and the shapes' rows are ragged, so
    // their prefix sums count tokens and scalars, the units each column's own
    // offsets count; every other shape must match the column's width exactly.
    let ragged = matches!(metric, "maxsim" | "chamfer" | "hausdorff");
    let mut bytes = Vec::new();
    let mut offsets = vec![0u64];
    let mut elements = 0u64;
    let expected = shape.regular_width();

    for (index, item) in items.iter().enumerate() {
        let mut row = Vec::new();
        flatten(item, index, &mut row)?;
        if let Some(width) = expected {
            if row.len() != width {
                return Err(QueryError::WrongWidth {
                    index,
                    found: row.len(),
                    expected: width,
                });
            }
        }
        if ragged {
            let depth = shape.inner().unwrap_or(0);
            if depth == 0 || !row.len().is_multiple_of(depth) {
                return Err(QueryError::RaggedRemainder { index, depth });
            }
            elements += match metric {
                "maxsim" => row.len() / depth,
                _ => row.len(),
            } as u64;
            offsets.push(elements);
        }
        if dtype == "u1x8" {
            push_bits(&row, &mut bytes);
        } else {
            for value in row {
                push_scalar(value, dtype, index, &mut bytes)?;
            }
        }
    }

    Ok(Queries {
        bytes,
        offsets: ragged.then_some(offsets),
        count: items.len(),
    })
}

// endregion: Encoding

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_rows_must_match_the_column_width() {
        let parsed = parse(b"[[1.0,2.0]]", "dot", "f32", Shape::Vector(2)).unwrap();
        assert_eq!(parsed.count(), 1);
        assert_eq!(parsed.bytes().len(), 8);
        assert!(parsed.offsets().is_none());

        let wrong = parse(b"[[1.0,2.0,3.0]]", "dot", "f32", Shape::Vector(2));
        assert_eq!(
            wrong.unwrap_err(),
            QueryError::WrongWidth {
                index: 0,
                found: 3,
                expected: 2
            }
        );
    }

    /// A fixed-size cloud nests, and the flat total is what the column checks —
    /// the nesting is a convenience for the writer, not a second constraint.
    #[test]
    fn fixed_clouds_may_nest() {
        let flat = parse(
            b"[[0,0,0,1,0,0,0,1,0]]",
            "chamfer",
            "f32",
            Shape::Matrix(3, 3),
        );
        let nested = parse(
            b"[[[0,0,0],[1,0,0],[0,1,0]]]",
            "chamfer",
            "f32",
            Shape::Matrix(3, 3),
        );
        assert_eq!(flat.unwrap().bytes(), nested.unwrap().bytes());
    }

    /// A sequence metric takes strings and reports byte offsets, so the batch is
    /// an irregular column in every respect.
    #[test]
    fn sequence_queries_carry_byte_offsets() {
        let parsed = parse(
            br#"["kitten","sitting"]"#,
            "levenshtein",
            "utf8",
            Shape::Scalar,
        )
        .unwrap();
        assert_eq!(parsed.count(), 2);
        assert_eq!(parsed.offsets(), Some(&[0u64, 6, 13][..]));
    }

    /// MaxSim's prefix sum counts tokens, not bytes, and a partial token is an
    /// error rather than a silently truncated row.
    #[test]
    fn maxsim_queries_count_tokens() {
        let parsed = parse(b"[[1,0,0,1],[1,0]]", "maxsim", "f32", Shape::Matrix(0, 2)).unwrap();
        assert_eq!(parsed.offsets(), Some(&[0u64, 2, 3][..]));

        let ragged = parse(b"[[1,0,0]]", "maxsim", "f32", Shape::Matrix(0, 2));
        assert_eq!(
            ragged.unwrap_err(),
            QueryError::RaggedRemainder { index: 0, depth: 2 }
        );
    }

    /// A cloud's prefix sum counts scalars, three per point, as USearch's own
    /// offsets do, and clouds in one batch may differ in size.
    #[test]
    fn cloud_queries_count_scalars() {
        let parsed = parse(
            b"[[[0,0,0],[1,0,0]],[[0,1,0]]]",
            "chamfer",
            "f16",
            Shape::Matrix(0, 3),
        )
        .unwrap();
        assert_eq!(parsed.offsets(), Some(&[0u64, 6, 9][..]));
        assert_eq!(parsed.bytes().len(), 9 * 2);
    }

    /// `1e400` names a magnitude no `f64` holds. JsonTape keeps it as an exact
    /// big number rather than saturating, so it arrives here as a number with no
    /// `f64` reading — still an error, and still worth naming as an out-of-range
    /// value rather than as a type mismatch.
    #[test]
    fn non_finite_values_are_refused() {
        let parsed = parse(b"[[1e400,0.2]]", "haversine", "f64", Shape::Vector(2));
        assert_eq!(parsed.unwrap_err(), QueryError::NotFinite { index: 0 });
    }

    #[test]
    fn malformed_input_is_named() {
        assert_eq!(
            parse(b"[[1.0,", "dot", "f32", Shape::Vector(2)).unwrap_err(),
            QueryError::Malformed
        );
        assert_eq!(
            parse(b"{}", "dot", "f32", Shape::Vector(2)).unwrap_err(),
            QueryError::NotAnArray
        );
    }
}

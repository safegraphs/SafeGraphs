//! Route a query to the right metric family at runtime — the one place the
//! __metric-name axis__ lives.
//!
//! Two axes, kept apart. A family owns its __dtype__ axis: `dense` resolves
//! `f64`…`u1x8` inside [`DotScannerAny`], `mesh` its four coordinate widths,
//! `maxsim` its three packable scalars. This module owns the axis above — which
//! family and which metric — so a file that declares `"haversine"` reaches the
//! geospatial scanner and one that declares `"smith-waterman"` reaches the
//! sequence engine. Neither axis leaks into the other: adding a dtype touches a
//! family, adding a metric touches only this file.
//!
//! `sequence` has no dtype axis at all — its four metrics differ by engine, not
//! by scalar — so its arms hold the concrete `SequenceScanner<Engine>` rather
//! than a family `*Any`. That asymmetry is the reason the router exists rather
//! than a uniform per-family enum.
//!
//! Queries cross as __bytes plus optional offsets__, the same model the format's
//! own columns use: a fixed-width metric reads the bytes through the table's
//! shape, a ragged one (sequence, maxsim) reads the offsets too. No query
//! wrapper type and no StringZilla type appears in this module's signatures.

use super::binary::{HammingScannerAny, HammingSearcherAny, JaccardScannerAny, JaccardSearcherAny};
use super::dense::{
    AngularScannerAny, AngularSearcherAny, DotScannerAny, DotSearcherAny, EuclideanScannerAny,
    EuclideanSearcherAny,
};
use super::geospatial::{
    HaversineScannerAny, HaversineSearcherAny, VincentyScannerAny, VincentySearcherAny,
};
use super::maxsim::{MaxSimScannerAny, MaxSimSearcherAny};
use super::mesh::{
    RigidMeshScannerAny, RigidMeshSearcherAny, RotationalMeshScannerAny, RotationalMeshSearcherAny,
    SimilarityMeshScannerAny, SimilarityMeshSearcherAny,
};
use super::sequence::{AlignmentCosts, LevenshteinCosts, SequenceScanner, SequenceSearcher};
use super::{ExecutionScope, Matrix, Ranking};
use crate::{Error, GraphTable};
use allocator_api2::alloc::{Allocator as AllocatorLike, Global};
use stringtape::BytesTapeView;
use stringzilla::szs;

/// Every metric name a file may declare, in the order [`ScannerAny`] matches
/// them. A CLI prints this for `--help`; a producer writing `__metadata__.metric`
/// picks from it. The strings are format contract — NumKong's kernel names keep
/// their underscores, and the sequence metrics the hyphens of the algorithms
/// they name. The last seven have no kernel here and are refused by name.
pub const METRICS: &[&str] = &[
    "dot",
    "euclidean",
    "sqeuclidean",
    "angular",
    "hamming",
    "jaccard",
    "haversine",
    "vincenty",
    "rmsd",
    "kabsch",
    "umeyama",
    "maxsim",
    "levenshtein",
    "levenshtein-utf8",
    "needleman-wunsch",
    "smith-waterman",
    "vdot",
    "bilinear",
    "mahalanobis",
    "kld",
    "jsd",
    "sparse_dot",
    "sparse_intersect",
];

/// True for a registered metric this reader has no kernel for: a file naming
/// one opens, and a scan or walk under it fails with [`Error::UnsupportedMetric`].
fn is_unscored(metric: &str) -> bool {
    matches!(
        metric,
        "vdot" | "bilinear" | "mahalanobis" | "kld" | "jsd" | "sparse_dot" | "sparse_intersect"
    )
}

/// True when the metric reads its queries as ragged rows — sequence tapes and
/// maxsim multi-vectors — so a caller knows whether offsets are required.
pub fn is_ragged(metric: &str) -> bool {
    matches!(
        metric,
        "maxsim" | "levenshtein" | "levenshtein-utf8" | "needleman-wunsch" | "smith-waterman"
    )
}

/// Which end of the score axis wins for `metric` — what a caller merging results
/// across shards needs, since the router's own arms know it but the erased
/// `distances` matrix does not. `None` for an unknown metric.
pub fn ranking(metric: &str) -> Option<Ranking> {
    Some(match metric {
        // Similarities: a larger product, overlap, or alignment score is a better match.
        "dot" | "vdot" | "bilinear" | "sparse_dot" | "sparse_intersect" | "needleman-wunsch"
        | "smith-waterman" => Ranking::Farthest,
        // Distances: everything else scores smaller-is-better, including MaxSim,
        // whose late interaction accumulates an angular distance.
        "euclidean" | "sqeuclidean" | "angular" | "mahalanobis" | "kld" | "jsd" | "hamming"
        | "jaccard" | "haversine" | "vincenty" | "rmsd" | "kabsch" | "umeyama" | "maxsim"
        | "levenshtein" | "levenshtein-utf8" => Ranking::Nearest,
        _ => return None,
    })
}

/// Build the borrowed tape a sequence engine consumes from raw bytes and their
/// prefix sums. Private, so no StringZilla type crosses this module's API.
fn sequence_tape<'a>(bytes: &'a [u8], offsets: &'a [u64]) -> Result<szs::AnyBytesTape<'a>, Error> {
    if offsets.first() != Some(&0) || offsets.last().map(|&end| end as usize) != Some(bytes.len()) {
        return Err(Error::Inconsistent);
    }
    // SAFETY: the pair is a prefix sum over the buffer beside it — checked above —
    // and both outlive the returned view.
    Ok(szs::AnyBytesTape::View64(unsafe {
        BytesTapeView::<u64>::from_raw_parts(bytes, offsets)
    }))
}

/// The offsets a ragged metric needs, or [`Error::MissingField`] when the caller
/// supplied none.
fn require_offsets(offsets: Option<&[u64]>) -> Result<&[u64], Error> {
    offsets.ok_or(Error::MissingField("query offsets"))
}

/// Token offsets as `usize`, the form [`MaxSimScannerAny`] takes. One entry per
/// query plus a terminator, so the copy is per-query, never per-byte.
fn token_offsets(offsets: &[u64]) -> Result<Vec<usize>, Error> {
    offsets
        .iter()
        .map(|&value| usize::try_from(value).map_err(|_| Error::TooLarge))
        .collect()
}

// region: Scanner

/// A brute-force scanner chosen from a metric name and a value dtype. Each arm
/// is a whole family dispatcher — the dtype is resolved one level down — except
/// the sequence arms, which are concrete because that family has no dtype axis.
pub enum ScannerAny<Allocator: AllocatorLike = Global> {
    Dot(DotScannerAny<Allocator>),
    Euclidean(EuclideanScannerAny<Allocator>),
    Angular(AngularScannerAny<Allocator>),
    Hamming(HammingScannerAny<Allocator>),
    Jaccard(JaccardScannerAny<Allocator>),
    Haversine(HaversineScannerAny<Allocator>),
    Vincenty(VincentyScannerAny<Allocator>),
    Rmsd(RigidMeshScannerAny<Allocator>),
    Kabsch(RotationalMeshScannerAny<Allocator>),
    Umeyama(SimilarityMeshScannerAny<Allocator>),
    MaxSim(MaxSimScannerAny<Allocator>),
    Levenshtein(SequenceScanner<szs::LevenshteinDistances, Allocator>),
    LevenshteinUtf8(SequenceScanner<szs::LevenshteinDistancesUtf8, Allocator>),
    NeedlemanWunsch(SequenceScanner<szs::NeedlemanWunschScores, Allocator>),
    SmithWaterman(SequenceScanner<szs::SmithWatermanScores, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send + Sync> ScannerAny<Allocator> {
    /// A scanner for `metric` over a value column of `dtype`. The metric is
    /// passed rather than read from the table so a caller may __override__ it —
    /// re-scoring one dataset under another metric is the point of a brute-force
    /// scan. An override the family cannot serve fails here, not silently later.
    ///
    /// Sequence costs default to unit edits and the identity substitution; a
    /// later revision reads them from `__metadata__`. `sqeuclidean` ranks as
    /// `euclidean` does, so it is served by that arm and reports the root.
    pub fn new(
        metric: &str,
        dtype: &str,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match metric {
            "dot" => ScannerAny::Dot(DotScannerAny::new(dtype, scope, alloc)?),
            "euclidean" | "sqeuclidean" => {
                ScannerAny::Euclidean(EuclideanScannerAny::new(dtype, scope, alloc)?)
            }
            "angular" => ScannerAny::Angular(AngularScannerAny::new(dtype, scope, alloc)?),
            "hamming" => ScannerAny::Hamming(HammingScannerAny::new(dtype, scope, alloc)?),
            "jaccard" => ScannerAny::Jaccard(JaccardScannerAny::new(dtype, scope, alloc)?),
            "haversine" => ScannerAny::Haversine(HaversineScannerAny::new(dtype, scope, alloc)?),
            "vincenty" => ScannerAny::Vincenty(VincentyScannerAny::new(dtype, scope, alloc)?),
            "rmsd" => ScannerAny::Rmsd(RigidMeshScannerAny::new(dtype, scope, alloc)?),
            "kabsch" => ScannerAny::Kabsch(RotationalMeshScannerAny::new(dtype, scope, alloc)?),
            "umeyama" => ScannerAny::Umeyama(SimilarityMeshScannerAny::new(dtype, scope, alloc)?),
            "maxsim" => ScannerAny::MaxSim(MaxSimScannerAny::new(dtype, scope, alloc)?),
            "levenshtein" => ScannerAny::Levenshtein(SequenceScanner::new(
                LevenshteinCosts::default(),
                scope,
                alloc,
            )?),
            "levenshtein-utf8" => ScannerAny::LevenshteinUtf8(SequenceScanner::new(
                LevenshteinCosts::default(),
                scope,
                alloc,
            )?),
            "needleman-wunsch" => ScannerAny::NeedlemanWunsch(SequenceScanner::new(
                AlignmentCosts::default(),
                scope,
                alloc,
            )?),
            "smith-waterman" => ScannerAny::SmithWaterman(SequenceScanner::new(
                AlignmentCosts::default(),
                scope,
                alloc,
            )?),
            name if is_unscored(name) => return Err(Error::UnsupportedMetric),
            _ => return Err(Error::UnknownMetric),
        })
    }

    /// The metric name this scanner serves — the string that selected it, or
    /// `euclidean` for `sqeuclidean`.
    pub fn metric(&self) -> &'static str {
        match self {
            ScannerAny::Dot(_) => "dot",
            ScannerAny::Euclidean(_) => "euclidean",
            ScannerAny::Angular(_) => "angular",
            ScannerAny::Hamming(_) => "hamming",
            ScannerAny::Jaccard(_) => "jaccard",
            ScannerAny::Haversine(_) => "haversine",
            ScannerAny::Vincenty(_) => "vincenty",
            ScannerAny::Rmsd(_) => "rmsd",
            ScannerAny::Kabsch(_) => "kabsch",
            ScannerAny::Umeyama(_) => "umeyama",
            ScannerAny::MaxSim(_) => "maxsim",
            ScannerAny::Levenshtein(_) => "levenshtein",
            ScannerAny::LevenshteinUtf8(_) => "levenshtein-utf8",
            ScannerAny::NeedlemanWunsch(_) => "needleman-wunsch",
            ScannerAny::SmithWaterman(_) => "smith-waterman",
        }
    }

    /// Batch brute-force top-k. `query_bytes` holds the batch in the table's
    /// dtype; `query_offsets` is the prefix sum a ragged metric needs and every
    /// fixed-width metric ignores. `matches`/`distances` are `queries × k`.
    pub fn scan(
        &mut self,
        query_bytes: &[u8],
        query_offsets: Option<&[u64]>,
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            ScannerAny::Dot(any) => any.scan(query_bytes, table, matches, distances, counts),
            ScannerAny::Euclidean(any) => any.scan(query_bytes, table, matches, distances, counts),
            ScannerAny::Angular(any) => any.scan(query_bytes, table, matches, distances, counts),
            ScannerAny::Hamming(any) => any.scan(query_bytes, table, matches, distances, counts),
            ScannerAny::Jaccard(any) => any.scan(query_bytes, table, matches, distances, counts),
            ScannerAny::Haversine(any) => any.scan(query_bytes, table, matches, distances, counts),
            ScannerAny::Vincenty(any) => any.scan(query_bytes, table, matches, distances, counts),
            ScannerAny::Rmsd(any) => any.scan(query_bytes, table, matches, distances, counts),
            ScannerAny::Kabsch(any) => any.scan(query_bytes, table, matches, distances, counts),
            ScannerAny::Umeyama(any) => any.scan(query_bytes, table, matches, distances, counts),
            ScannerAny::MaxSim(any) => {
                let offsets = token_offsets(require_offsets(query_offsets)?)?;
                any.scan(query_bytes, &offsets, table, matches, distances, counts)
            }
            ScannerAny::Levenshtein(scanner) => scanner.scan(
                sequence_tape(query_bytes, require_offsets(query_offsets)?)?,
                table,
                matches,
                distances,
                counts,
            ),
            ScannerAny::LevenshteinUtf8(scanner) => scanner.scan(
                sequence_tape(query_bytes, require_offsets(query_offsets)?)?,
                table,
                matches,
                distances,
                counts,
            ),
            ScannerAny::NeedlemanWunsch(scanner) => scanner.scan(
                sequence_tape(query_bytes, require_offsets(query_offsets)?)?,
                table,
                matches,
                distances,
                counts,
            ),
            ScannerAny::SmithWaterman(scanner) => scanner.scan(
                sequence_tape(query_bytes, require_offsets(query_offsets)?)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

// endregion: Scanner

// region: Searcher

/// A graph-walking searcher chosen from a metric name and a value dtype — the
/// walk twin of [`ScannerAny`].
///
/// Unlike a scan, a walk is only meaningful under the metric the graph's edges
/// were built for, so a caller overriding the file's metric here is asking for
/// wrong answers. This type does not stop it — it cannot know — but the CLI
/// refuses the override, and that is the reason why.
pub enum SearcherAny<Allocator: AllocatorLike = Global> {
    Dot(DotSearcherAny<Allocator>),
    Euclidean(EuclideanSearcherAny<Allocator>),
    Angular(AngularSearcherAny<Allocator>),
    Hamming(HammingSearcherAny<Allocator>),
    Jaccard(JaccardSearcherAny<Allocator>),
    Haversine(HaversineSearcherAny<Allocator>),
    Vincenty(VincentySearcherAny<Allocator>),
    Rmsd(RigidMeshSearcherAny<Allocator>),
    Kabsch(RotationalMeshSearcherAny<Allocator>),
    Umeyama(SimilarityMeshSearcherAny<Allocator>),
    MaxSim(MaxSimSearcherAny<Allocator>),
    Levenshtein(SequenceSearcher<szs::LevenshteinDistances, Allocator>),
    LevenshteinUtf8(SequenceSearcher<szs::LevenshteinDistancesUtf8, Allocator>),
    NeedlemanWunsch(SequenceSearcher<szs::NeedlemanWunschScores, Allocator>),
    SmithWaterman(SequenceSearcher<szs::SmithWatermanScores, Allocator>),
}

impl<Allocator: AllocatorLike + Clone + Send + Sync> SearcherAny<Allocator> {
    /// A searcher for `metric` over a value column of `dtype`, walking with
    /// breadth `expansion_factor`.
    pub fn new(
        metric: &str,
        dtype: &str,
        expansion_factor: usize,
        scope: ExecutionScope,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        Ok(match metric {
            "dot" => SearcherAny::Dot(DotSearcherAny::new(dtype, expansion_factor, scope, alloc)?),
            "euclidean" | "sqeuclidean" => SearcherAny::Euclidean(EuclideanSearcherAny::new(
                dtype,
                expansion_factor,
                scope,
                alloc,
            )?),
            "angular" => SearcherAny::Angular(AngularSearcherAny::new(
                dtype,
                expansion_factor,
                scope,
                alloc,
            )?),
            "hamming" => SearcherAny::Hamming(HammingSearcherAny::new(
                dtype,
                expansion_factor,
                scope,
                alloc,
            )?),
            "jaccard" => SearcherAny::Jaccard(JaccardSearcherAny::new(
                dtype,
                expansion_factor,
                scope,
                alloc,
            )?),
            "haversine" => SearcherAny::Haversine(HaversineSearcherAny::new(
                dtype,
                expansion_factor,
                scope,
                alloc,
            )?),
            "vincenty" => SearcherAny::Vincenty(VincentySearcherAny::new(
                dtype,
                expansion_factor,
                scope,
                alloc,
            )?),
            "rmsd" => SearcherAny::Rmsd(RigidMeshSearcherAny::new(
                dtype,
                expansion_factor,
                scope,
                alloc,
            )?),
            "kabsch" => SearcherAny::Kabsch(RotationalMeshSearcherAny::new(
                dtype,
                expansion_factor,
                scope,
                alloc,
            )?),
            "umeyama" => SearcherAny::Umeyama(SimilarityMeshSearcherAny::new(
                dtype,
                expansion_factor,
                scope,
                alloc,
            )?),
            "maxsim" => SearcherAny::MaxSim(MaxSimSearcherAny::new(
                dtype,
                expansion_factor,
                scope,
                alloc,
            )?),
            "levenshtein" => SearcherAny::Levenshtein(SequenceSearcher::new(
                LevenshteinCosts::default(),
                expansion_factor,
                scope,
                alloc,
            )?),
            "levenshtein-utf8" => SearcherAny::LevenshteinUtf8(SequenceSearcher::new(
                LevenshteinCosts::default(),
                expansion_factor,
                scope,
                alloc,
            )?),
            "needleman-wunsch" => SearcherAny::NeedlemanWunsch(SequenceSearcher::new(
                AlignmentCosts::default(),
                expansion_factor,
                scope,
                alloc,
            )?),
            "smith-waterman" => SearcherAny::SmithWaterman(SequenceSearcher::new(
                AlignmentCosts::default(),
                expansion_factor,
                scope,
                alloc,
            )?),
            name if is_unscored(name) => return Err(Error::UnsupportedMetric),
            _ => return Err(Error::UnknownMetric),
        })
    }

    /// Batch top-k by graph walk. Inputs mirror [`ScannerAny::scan`].
    pub fn search(
        &mut self,
        query_bytes: &[u8],
        query_offsets: Option<&[u64]>,
        table: &GraphTable<impl AllocatorLike>,
        matches: &mut Matrix<u64>,
        distances: &mut Matrix<f64>,
        counts: &mut [u64],
    ) -> Result<(), Error> {
        match self {
            SearcherAny::Dot(any) => any.search(query_bytes, table, matches, distances, counts),
            SearcherAny::Euclidean(any) => {
                any.search(query_bytes, table, matches, distances, counts)
            }
            SearcherAny::Angular(any) => any.search(query_bytes, table, matches, distances, counts),
            SearcherAny::Hamming(any) => any.search(query_bytes, table, matches, distances, counts),
            SearcherAny::Jaccard(any) => any.search(query_bytes, table, matches, distances, counts),
            SearcherAny::Haversine(any) => {
                any.search(query_bytes, table, matches, distances, counts)
            }
            SearcherAny::Vincenty(any) => {
                any.search(query_bytes, table, matches, distances, counts)
            }
            SearcherAny::Rmsd(any) => any.search(query_bytes, table, matches, distances, counts),
            SearcherAny::Kabsch(any) => any.search(query_bytes, table, matches, distances, counts),
            SearcherAny::Umeyama(any) => any.search(query_bytes, table, matches, distances, counts),
            SearcherAny::MaxSim(any) => {
                let offsets = token_offsets(require_offsets(query_offsets)?)?;
                any.search(query_bytes, &offsets, table, matches, distances, counts)
            }
            SearcherAny::Levenshtein(search) => search.search(
                sequence_tape(query_bytes, require_offsets(query_offsets)?)?,
                table,
                matches,
                distances,
                counts,
            ),
            SearcherAny::LevenshteinUtf8(search) => search.search(
                sequence_tape(query_bytes, require_offsets(query_offsets)?)?,
                table,
                matches,
                distances,
                counts,
            ),
            SearcherAny::NeedlemanWunsch(search) => search.search(
                sequence_tape(query_bytes, require_offsets(query_offsets)?)?,
                table,
                matches,
                distances,
                counts,
            ),
            SearcherAny::SmithWaterman(search) => search.search(
                sequence_tape(query_bytes, require_offsets(query_offsets)?)?,
                table,
                matches,
                distances,
                counts,
            ),
        }
    }
}

// endregion: Searcher

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name in [`METRICS`] ranks and constructs, or is refused as
    /// unscored — the router's whole contract in one pass. Dtypes are the widest
    /// each family accepts, so a family rejecting the dtype cannot mask a routing
    /// bug.
    #[test]
    fn every_metric_name_routes() {
        for metric in METRICS {
            assert!(ranking(metric).is_some(), "{metric} has no ranking");
            let dtype = match *metric {
                "hamming" | "jaccard" => "u1x8",
                "maxsim" => "f32",
                _ => "f32",
            };
            let scanner = ScannerAny::new(metric, dtype, ExecutionScope::Serial, Global);
            if is_unscored(metric) {
                assert!(matches!(scanner, Err(Error::UnsupportedMetric)));
                continue;
            }
            assert!(scanner.is_ok(), "{metric} failed to route");
            let served = match *metric {
                "sqeuclidean" => "euclidean",
                name => name,
            };
            assert_eq!(scanner.unwrap().metric(), served);
        }
    }

    /// The binary family's palette is wider than the packed cross-product
    /// kernels: those are `u1`-only, so the extra dtypes are served by the
    /// per-pair path. Both reach the caller through one metric name, because
    /// packed-versus-pair is a dtype decision the family owns, not a metric the
    /// router names.
    #[test]
    fn binary_metrics_route_their_whole_palette() {
        for (metric, dtype) in [
            ("hamming", "u1x8"),
            ("hamming", "u8"),
            ("jaccard", "u1x8"),
            ("jaccard", "u16"),
            ("jaccard", "u32"),
        ] {
            let scanner = ScannerAny::new(metric, dtype, ExecutionScope::Serial, Global);
            assert!(scanner.is_ok(), "{metric} over {dtype} failed to route");
            let searcher = SearcherAny::new(metric, dtype, 8, ExecutionScope::Serial, Global);
            assert!(
                searcher.is_ok(),
                "{metric} over {dtype} failed to route for a walk"
            );
        }
    }

    #[test]
    fn unknown_metric_is_refused() {
        assert!(matches!(
            ScannerAny::new("cosine", "f32", ExecutionScope::Serial, Global),
            Err(Error::UnknownMetric)
        ));
    }

    #[test]
    fn ragged_metrics_are_flagged() {
        assert!(is_ragged("levenshtein") && is_ragged("maxsim"));
        assert!(!is_ragged("dot") && !is_ragged("kabsch"));
    }
}

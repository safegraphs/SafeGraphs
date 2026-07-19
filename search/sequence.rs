//! Sequence metrics: each node's value (or key) is a variable-length byte
//! string — DNA, RNA, protein, or text — scored by StringZilla.
//!
//! Metrics (Levenshtein, Needleman-Wunsch, Smith-Waterman) land in Phase 2. The
//! engine trait and the `dense` module are the template.

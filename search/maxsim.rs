//! MaxSim (ColBERT late interaction): each node's value is a multi-vector, scored
//! against a multi-vector query by NumKong's MaxSim kernels.
//!
//! The metric lands in Phase 2. The engine trait and the `dense` module are the
//! template.

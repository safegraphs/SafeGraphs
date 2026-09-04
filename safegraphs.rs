//! SafeGraphs — the SafeTensors of graphs.
//!
//! A memory-mappable interchange format for flat search graphs: an 8-byte magic,
//! a little-endian `u64` header length, a JSON header that is the complete
//! schema, and a data segment of raw, Arrow-layout buffers.
//!
//! The crate is two pillars. The format layer reads, writes, and memory-maps the
//! `.safegraphs` container — zero-copy, never-panic, allocator-aware. [`search`]
//! scores queries against a [`GraphTable`], behind the `search` feature: a batched
//! brute-force scan and a greedy best-first walk, with one file per metric family.
//!
//! The default build only reads and unpacks. The `search` feature pulls NumKong,
//! StringZilla, and ForkUnion for scoring and parallelism.
//!
//! ```no_run
//! use safegraphs::{GraphLike, GraphTable};
//!
//! let table = GraphTable::open("index.safegraphs")?;
//! println!("{} nodes, {}", table.len(), table.value_dtype());
//! for node in 0..table.len() {
//!     let _payload = table.value(node);
//!     let _edges = table.neighbors(node);
//! }
//! # Ok::<(), safegraphs::Error>(())
//! ```

mod format;
pub use format::*;

pub mod search;

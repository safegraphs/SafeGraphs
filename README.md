# SafeGraphs

![SafeGraphs banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/SafeGraphs.jpg?raw=true)

SafeGraphs is the SafeTensors of graphs — a simple format to exchange graph-shaped search indexes between CPUs, GPUs, and machines.
It's designed as an interoperability layer between NSW, Cagra, and similar algorithms, often used in vector databases for approximate nearest neighbor search, or ANNS.
It aims to be simple, efficient, and easy to implement across different programming languages and platforms.
Exchanging a graph requires only its keys, values, and neighbor lists, with no knowledge of how it was built or how it will be searched.

## Design Philosophy

__Minimalism__.
No complex dependencies or external libraries should be required to read or write SafeGraphs files.
The format should be simple and easy to implement in any programming language.

__Memory Mapping__.
SafeGraphs should be memory-mappable, allowing for simple access to larger-than-memory graphs without loading them into memory.
With indexes often being constructed on 100+ core 8-GPU machines, they should be introspectable and usable on consumer-grade hardware.
This immediately cuts out compression schemes like [Snappy](https://github.com/google/snappy) and [Zstandard](https://github.com/facebook/zstd).

__Extensibility__.
Modern ML leans on mixed-precision arithmetic — `f16`, `bf16`, `i8`, and custom floating-point formats like `e4m3` and `e5m2` — over payloads that range from dense vectors to variable-length and sparse arrays.
SafeGraphs should absorb new element types without breaking file readability, while applications choose their own scoring and traversal methods.

__Easy Distribution__.
Two file layouts are supported: a single-file ["Monolith"](#monolith), or a `.safegraphs.json` manifest over a ["Distributed"](#distributed) array of `.safegraphs` shards.
The first is the simplest variant for indexes under 100 GB.
The second is optimal for distributing billion-scale indexes.

__Easy to Traverse & Evaluate__.
Optimal search indexes are asymptotically hard to build.
But it doesn't mean analyzing or evaluating them should be equally hard.
This repository includes an optional [search extension](search/README.md) for scoring values and traversing graphs.
Its algorithms and supported operations are independent of the file format.

__Least Machinery__.
A format cannot know a consumer's page size, its storage device, or whether it will scan or walk, and the layer above knows all three.
So SafeGraphs stores rows as the producer declared them and adds no layout machinery of its own: no per-entry padding, no block framing, no reordering.
A producer that wants every row to land inside one page picks a `shape` whose row divides it; one that would rather spend the loads leaves it alone.
Either way a row can straddle a page or a device sector, and how to handle that split read is the consumer's call, made where the access pattern is actually known.

## Principles

__Do__:

- Store one flat graph per file, with no layers and no hierarchy.
- Treat the JSON header as the whole schema, naming every column, its dtype, its shape, and its buffers.
- Locate every buffer by a `[begin, end]` byte pair relative to the data segment.
- Start every buffer at a multiple of `alignment`, with zeros filling the gap before it, so alignment applies to buffers and never to rows.
- Pack rows back to back inside a buffer, each exactly as wide as its column's `shape` and dtype make it, so no row width is ever stored.
- Pad a short fixed-width `neighbors` row with sentinels, any index at or past the node count, which a reader skips.
- Require `key`, `value`, and `neighbors`, and let graph readers ignore every other column.
- Write every integer little-endian.
- Keep every file memory-mappable, so a reader takes typed slices straight out of the data segment.
- Keep every shard of a Distributed index a complete Monolith whose edges stay inside it.
- Let the manifest route and size shards, and leave the schema and optional metadata to each shard's own header.

__Don't__:

- Don't compress buffers, since compression defeats memory mapping.
- Don't pad, frame, or reorder rows inside a buffer; a producer that wants page-aligned rows picks a `shape` whose row width divides the page.
- Don't store nulls, since every node is present in every column and an empty edge list is zero-length.
- Don't store keys or another file's indices in `neighbors`, which holds node indices into its own file.
- Don't restate a derivable fact like the node count in the header.
- Don't find buffers by their order, since only their pairs locate them.

## File Structure

Similar to Hugging Face's `safetensors`, a SafeGraphs file is a small JSON header followed by a single data segment of raw, concatenated buffers.
The JSON header is the complete schema: it names each column, its element type, and the byte range of its buffers within the data segment.
The buffers use Apache Arrow's physical memory layout, so they can be handed to Arrow zero-copy through the C Data Interface.
The file itself is not an Arrow IPC container, and it needs no Arrow or FlatBuffers library to read.
Reading is therefore dependency-free: parse the JSON with any standard library, then memory-map typed slices straight out of the data segment.

Each file stores a single flat graph.
There are no layers and no hierarchy, so a node's edges are simply its outgoing neighbors.

### Monolith

A single `.safegraphs` file holds everything: the schema, the keys, the payloads, and the graph.

```text
offset 0        ┌────────────────────────────────┐
                │ magic "SAFEGRPH"               │  8 bytes
offset 8        ├────────────────────────────────┤
                │ header size H                  │  u64, little-endian
offset 16       ├────────────────────────────────┤
                │ JSON header                    │  H bytes of UTF-8, padded with spaces
offset 16 + H   ├────────────────────────────────┤  data segment, aligned
                │ key offsets, if ragged         │
                │ key data                       │  each buffer starts at a multiple
                │ value offsets, if ragged       │  of the alignment, zeros fill the gap
                │ value data                     │  before it, and its rows follow one
                │ neighbors offsets, if ragged   │  another with no gap
                │ neighbors data                 │
                │ extra columns, such as norms   │
                └────────────────────────────────┘
```

An 8-byte magic and a `u64` header size keep everything 8-byte aligned from the first byte and let tools sniff the file type.
All integers are little-endian.
The header size counts the space padding, so the data segment begins at byte offset 16 plus the header size.
Each buffer is a `[begin, end]` pair of byte positions relative to the start of the data segment, named `data` for the rows and `offsets` for a ragged column's row starts, so the header stays position-independent even if the segment is padded ahead.
A pair of positions can only mean positions, where a start beside a length left the unit of each to the reader; the byte count is `end - begin`.
Buffers are found through those pairs alone, never by order, and they never overlap.

Every buffer, and the data segment itself, starts at a file offset that is a multiple of the `alignment` declared in `__metadata__`, a power of two defaulting to 4096 bytes, and a reader treats any declared value below 64 as 64.
That alignment reaches the start of a buffer and stops there: rows follow one another inside a buffer with no gap, so only the first row of a column is aligned by it.
Header bytes are padded with spaces, and data-segment padding is zeros, so a gap can sit between one buffer's end and the next's start.
The 4 KB default matches the x86 page, the common SSD logical sector, the `O_DIRECT` block, and GPUDirect Storage's DMA quantum, so an entire column can stream straight into GPU memory.
That default is small on purpose, for the reason RocksDB reads 4 KB SST blocks where a columnar store like ClickHouse reads 64 KB to 1 MB: a traversal touches scattered entries and wants the smallest unit that still amortizes a device read.
A producer targeting Apple Silicon can raise `alignment` to 16384, or to 2 MB for huge pages, and readers simply honor the declared value.

A flat graph has three graph columns — `key`, `value`, and `neighbors` — and every one of them is required.
Any other top-level key except `__metadata__` is a column in the same grammar with the same node count, which a producer writes for its own use and a graph reader ignores.
For example, an application can carry cached vector norms alongside the values as an extra column, with a name and meaning it defines.
[Metric metadata](#metrics) and extra columns are optional; neither is needed to read, write, or exchange the graph.

```json
{
  "__metadata__": {
    "metric": "dot",                # optional scoring metadata; interpreted by the application
    "expansion": 128,               # optional construction metadata; interpreted by the application
    "alignment": 4096,              # buffer alignment in bytes; a power of two, default 4096, floor 64
    "created_by": "usearch 3.x",    # free-form provenance, and anything else a producer wants to record
    "built_at": "2026-07-19"
  },
  "key": {
    "dtype": "uuid",          # node identity: u32, u64, uuid, or utf8
    "data": [0, 320000000]    # 20M keys of 16 bytes; no shape means one element per row
  },
  "value": {
    "dtype": "f16",           # embedding vector, or utf8 for a string payload
    "shape": 1536,            # fixed width, or [rows, cols]; offsets instead for ragged rows
    "data": [320000000, 61760000000]
  },
  "neighbors": {
    "dtype": "u32",           # node indices into this same file, never keys: u32 or u64
    "shape": 32,              # fixed degree; offsets instead for a variable one
    "data": [61760000000, 64320000000]
  }
}
```

A `utf8` key or value carries a `shape` when its rows are a fixed width, and the `offsets` buffer of the irregular layout below when they are not.
No column is nullable.
Every node is present in every column, so there is no validity bitmap and no null count.
The `key` column sets the node count, and every other column must agree with it.
No header field restates that count, since a second source of truth for a derivable fact can only disagree.
A node with no outgoing edges is a zero-length `neighbors` list, which is empty rather than null, and any optional payload must be encoded in-band, never as a null.
Keys are global to a dataset and need not be unique, so several nodes may carry the same key, each with its own value.

Within a `neighbors` list, any index at or past the node count is a __sentinel__ and names no node: a reader skips it.
This is what lets a fixed-degree `neighbors` column stand in for a variable-degree one — a producer pads a short adjacency row with a sentinel rather than dropping to the irregular layout.
A sentinel may sit anywhere in a row, though a producer normally pads the tail.
Any out-of-range value serves, so a reader needs no agreement to skip one, and a producer writes the index type's maximum by convention.
A reader that walks the graph honors this; one that only unpacks buffers hands the raw indices through untouched, since it relabels the buffer rather than reading it.

### Regular vs Irregular Columns

The presence of an `offsets` buffer selects the layout.

- __Regular__ has no `offsets`, so it is an Arrow primitive array when `shape` is omitted and a `FixedSizeList<T, shape>` otherwise.
  Its `data` buffer holds the concatenated rows and nothing else.
  `shape` is the fixed width, a bare number or a `[rows, cols]` pair whose product is the width, and defaults to 1 when omitted.
  A row occupies `shape` elements, so the node count is the `data` byte count divided by the byte length of one row, which is `shape × bits` rounded up to whole bytes and equals `shape` times the element size for every whole-byte dtype.
  Use this for scalar keys, vectors, and fixed-degree graphs.
- __Irregular__ has an `offsets` buffer and an `offsets_dtype` of `u32` or `u64`, so it is an Arrow `List<T>` or `LargeList<T>`, and carries no fixed width.
  A `[0, cols]` shape marks a ragged matrix whose inner axis is `cols`, so a row of `n` elements holds `n / cols` vectors.
  The `offsets` buffer holds `nodes + 1` non-decreasing prefix sums in elements, and `data` holds the concatenated rows.
  Node `i` is the elements of `data` between prefix sums `i` and `i + 1`, the first sum being zero and the last the total element count.

Arrow's offsets are signed but never negative, so an Arrow `List` or `LargeList` offsets buffer is copied under the `u32` or `u64` label unchanged, and a SafeGraphs column goes back to Arrow whenever its last prefix sum fits the signed width.

`dtype` names one element, and every storage width has a neutral spelling `b128`, `b64`, `b32`, `b16`, `b8` naming that many opaque bits.
Every other spelling is an alias that adds a reading to the same width: the floats `f64`, `f32`, `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`; the signed integers `i64`, `i32`, `i16`, `i8`, `i4x2`; the unsigned `u64`, `u32`, `u16`, `u8`, `u4x2`, `u1x8`; `uuid` for a 16-byte identity; and `utf8` for bytes a reader may assume are valid UTF-8.
A column of opaque bytes that are not text uses `b8`.
The `x2` and `x8` suffixes name packed sub-byte elements, two nibbles or eight bits to the byte, so `shape` counts logical elements and a row of `shape` such elements occupies `shape × bits / 8` bytes rounded up.

A concrete variable-degree `neighbors` column, with 3 nodes of degree 2, 3, and 1:

```json
"neighbors": {
  "dtype": "u32",
  "offsets_dtype": "u32",   # prefix sums [0, 2, 5, 6] — 4 × u32 = 16 bytes
  "offsets": [0, 16],
  "data": [4096, 4120]      # next alignment boundary; 6 × u32 node indices = 24 bytes
}
```

### String Columns

A string key or a string value uses `dtype: "utf8"`.
Without a `shape` it is the irregular layout above with byte elements, an Arrow `Utf8` or `LargeUtf8`: the `offsets` buffer holds `nodes + 1` prefix sums, `data` holds the concatenated UTF-8 bytes, and string `i` is the byte slice between prefix sums `i` and `i + 1`.
With a `shape` it is a fixed-width column of that many bytes per string, an Arrow `FixedSizeBinary`, which loses the string reading on the way into Arrow since Arrow has no fixed-width UTF-8 type.
There is no separate ASCII type.
All text is UTF-8, so any language or emoji round-trips without a codepage, and bytes that are not valid UTF-8 belong in a `u8` column instead, Arrow's `Binary` or `FixedSizeBinary`.

```json
"key": {
  "dtype": "utf8",
  "offsets_dtype": "u64",   # prefix sums [0, 2, 7, 13] — 4 × u64 = 32 bytes
  "offsets": [0, 32],
  "data": [4096, 4109]      # next alignment boundary; "α" 2 + "béé" 5 + "日本" 6 = 13 UTF-8 bytes
}
```

### Metrics

`__metadata__.metric` is __optional__ metadata describing how an application scores values.
The producer and consumer agree on its meaning; the format prescribes no metric registry, kernel library, or naming convention.
A file reader can preserve it without interpreting it, and an absent or unfamiliar metric does not prevent graph exchange.
Construction parameters and traversal choices likewise belong to the application.

The optional search extension has its own supported metric names and rejects unsupported ones when asked to score values.
That is a search capability limit, not a file validity rule.

### Distributed

For larger datasets, the nodes are partitioned across many `.safegraphs` shards described by a JSON manifest.
Each shard is a complete Monolith as above, self-describing and independently readable.
A shard's `neighbors` reference only nodes inside that same shard, so edges never cross files.
Node indices are local to each shard, so the same index names a different node in every file, while keys are global, and the manifest, not the file format, records how they are partitioned.

```text
dataset.safegraphs.json                  manifest: path, keys, bytes, nodes, sha256
 ├─→ dataset-00001-of-00002.safegraphs   a full Monolith, edges stay inside it
 └─→ dataset-00002-of-00002.safegraphs   a full Monolith, edges stay inside it
```

`safetensors` has no cross-file convention of its own, since it is single-file.
Hugging Face's sharded layout adds one on top, an index binding each tensor to its shard, which splits by column where this manifest splits by row.
It carries routing in `path` and `keys`, sizing in `bytes` and `nodes`, and integrity in `sha256`.
Each shard's header carries its own column schema and any optional metric or provenance metadata, so a shard lifted out of the set still describes itself completely.

```json
{
  "__metadata__": {              # free-form, and dataset-wide rather than per-shard
    "dataset": "wiki-1m"
  },
  "files": [
    {
      "path": "dataset-00001-of-00002.safegraphs",  # relative to the manifest
      "bytes": 21474836480,
      "nodes": 20000000,         # optional; lets a coordinator size a load without opening the shard
      "keys": "^[0-7]",          # optional regex over canonical key renderings
      "sha256": "9a1c…e4"        # optional; hex digest of the whole shard file
    },
    {
      "path": "dataset-00002-of-00002.safegraphs",
      "bytes": 21474836480,
      "nodes": 20000000,
      "keys": "^[89a-f]",
      "sha256": "1b7d…02"
    }
  ]
}
```

- `keys` matches the canonical rendering of a key — decimal for integers, lowercase dashed UUID for `uuid`, the string itself for `utf8` — in the RE2 syntax every mainstream engine shares.
  A coordinator routes a key lookup to the shard whose regex matches, while a vector query still fans out to every shard, since edges never cross files.
  `null` or an absent field means the shard is not key-partitioned, and the regexes of a partitioned dataset are disjoint.
- `sha256` digests the whole shard file, checked in a separate pass rather than through the memory map.

## Interoperability

SafeGraphs interoperates through a standard, not through per-tool adapters.
Because its data segment is Apache Arrow's physical memory layout, any Arrow-aware runtime can borrow a column zero-copy through the C Data Interface, in any language, without a SafeGraphs library.

That same flat layout is the shape the most common on-GPU graph indexes already use, so moving a graph in or out is a relabeling, not a conversion.

__Cagra__ from cuVS is a fixed-degree flat graph, a dense `[nodes, degree]` matrix of `u32` neighbor indices over a `[nodes, dim]` dataset.
It drops in with zero reshaping: `value` is `FixedSizeList<dtype, dim>`, `neighbors` is `FixedSizeList<u32, degree>`, and the neighbor indices are already SafeGraphs node indices.
Any dead slot Cagra leaves in a short row is an out-of-range index, which the sentinel convention already skips, so the matrix copies over verbatim.

__FAISS__ maps through `IndexNSG`, its flat navigable graph, rather than the hierarchical `IndexHNSW`.
Its adjacency is fixed-max-degree neighbor IDs that are node indices, so it drops in either way: keep the fixed matrix and let its padding become sentinels, or export a variable-length `neighbors` column that omits the padding entirely.
Vectors come from the paired storage index, and the external IDs of an `IndexIDMap` become the `key` column.
FAISS `IndexShards`, independent sub-indexes with their own row spaces merged at query time, is a direct match for the Distributed manifest with overlapping local offsets.

## Case Studies

### BERT-like Dense Document Embeddings

Points are stored as fixed-size `value` arrays representing high-dimensional vectors, and the graph links spatial neighbors.
Applications can rank them by dot product, cosine similarity, or Euclidean distance.

> See `numkong::Dot`, `numkong::Angular`, `numkong::Euclidean`, `numkong::Dots`, `numkong::Angulars`, and `numkong::Euclideans`.

### DNA, RNA, and Protein Strings

Biological sequences are stored as `utf8` values, one strand per node, over a graph whose edges link similar sequences.
Applications choose the edit distance or alignment score appropriate to their sequences.

> See `stringzilla::LevenshteinEngine`, `stringzilla::LevenshteinSymbol::Bytes`, and `stringzilla::LevenshteinSymbol::Runes`.

### Haversine & Vincenty Search

Points are stored as 2D `value` vectors of `f32` or `f64` coordinates, and the graph links geospatial neighbors.
Applications can compare locations using haversine or Vincenty distances.

> See `numkong::HaversinesPackedMatrix` and `numkong::VincentysPackedMatrix`.

### Binary and Weighted Sparse Feature Sets

Sparse, high-dimensional features are stored per node as a variable-length `value` column of index and weight pairs, rather than a dense vector.
Applications can compare the non-zero dimensions directly using sparse dot products or set overlap.
This fits bag-of-words retrieval, learned-sparse models like SPLADE, and one-hot categorical signals from recommender and ranking systems.

> See `numkong::Hamming`, `numkong::Jaccard`, `numkong::SparseDot`, and `numkong::SparseIntersect`.

### Molecular Conformers and 3D Shapes

Each entry is a point cloud of any size, stored as a `[0, 3]` `value` column of interleaved coordinates whose `offsets` count scalars, three per point.
Applications choose how to align and compare the clouds, with or without point correspondence.

> See `numkong::ShapesPackedMatrix::chamfers` and `numkong::ShapesPackedMatrix::hausdorffs`.

### ColBERT Late-Interaction Multi-Vector Entries

Each entry is a bag of per-token embeddings, a multi-vector, stored as a `[0, dim]` `value` column, one small matrix per node with as many rows as tokens.
Applications can score late interaction by matching every query token to its most similar entry token and summing those best matches.

> See `numkong::MaxSim` and `numkong::MaxSimPackedMatrix`.

## Where It Fits

SafeGraphs is an interchange format, so it sits alongside these tools rather than against them: it moves an already-built index between them, and reuses the best of them — Arrow's physical layout — rather than reinventing it.

__[Arrow](https://arrow.apache.org)__, __[Parquet](https://github.com/apache/parquet-format/)__, __[Iceberg](https://iceberg.apache.org)__.
These are columnar analytics formats built for scanning and filtering wide tables.
SafeGraphs borrows Arrow's physical layout for its buffers, but it is not a table: it stores one flat graph, tuned for random neighbor traversal rather than large sequential scans.
Parquet optimizes for compressed columnar scans and Iceberg for evolving, transactional tables — different goals than a single zero-copy, memory-mapped graph.

__[FlatBuffers](https://github.com/google/flatbuffers)__, __[Protocol Buffers](https://github.com/protocolbuffers/protobuf)__.
These are schema-driven serialization frameworks: you compile a schema into generated code and link a runtime to read a message.
They trade a build step for compact, evolvable messages; SafeGraphs makes the opposite trade — its schema is the JSON header itself, read with any JSON parser.

__[Lance](https://github.com/lancedb/lance)__.
Lance is a versioned columnar storage format with optional secondary indexes built on top, aimed at replacing Parquet for ML workloads and powering LanceDB.
SafeGraphs is the opposite slice of the problem: it has no runtime, no transaction log, no index-builder — it just freezes an already-built graph structure into a file that another library can memory-map.
Put differently, Lance _contains_ a graph index among its machinery; SafeGraphs _is_ the graph.

__[Lucene](https://lucene.apache.org)__.
Lucene has an on-disk format: a segmented codec whose KNN-vectors component stores an HNSW graph much as SafeGraphs stores a flat one.
But it is an internal codec, versioned per release and read through Lucene itself; SafeGraphs aims instead for a stable, language-neutral file any engine can map directly.

__[Pinecone](https://www.pinecone.io)__, __[Milvus](https://milvus.io)__, __[Elasticsearch](https://www.elastic.co/elasticsearch)__.
These are search systems, not file formats: each owns the whole path from ingestion to query behind a service or a cluster, and its storage is an internal detail, Elasticsearch's being Lucene's above.
SafeGraphs sits a layer below — the frozen index such a system could import or export, not a competitor to the system itself.

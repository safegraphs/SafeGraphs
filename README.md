# SafeGraphs

SafeGraphs is the SafeTensors of graphs - a simple format to exchange network-shaped search indexes between CPUs and GPUs and across the network.
It's designed as an interoperability layer between NSW, Cagra, and similar algorithms, often used in vector databases for approximate nearest neighbor search (ANNS).
It aims to be simple, efficient, and easy to implement across different programming languages and platforms.

## Design Philosophy

__Minimalism__.
No complex dependencies or external libraries should be required to read or write SafeGraphs files.
The format should be simple and easy to implement in any programming language.
This immediately cuts out formats like Google's [Protocol Buffers](https://github.com/protocolbuffers/protobuf), Apache [Parquet](https://github.com/apache/parquet-format/), etc.

__Memory Mapping__.
SafeGraphs should be memory-mappable, allowing for simple access to larger-than-memory graphs without loading them into memory.
With indexes often being constructed on 100+ core 8-GPU machines, they should be introspectable and usable on consumer-grade hardware.
This immediately cuts out compression schemes like [Snappy](https://github.com/google/snappy), [Zstandard](https://github.com/facebook/zstd), and suggests intra-entry padding to align data blocks to SSD page sizes.

__Extensibility__.
Modern ML and AI heavily relies on mixed-precision arithmetic, starting with types like `f16`, `bf16`, and `i8`, and ending with `e4m3`, `e5m2` and other custom floating point formats.
Most applications deal with dense high-dimensional vectors, which are often stored as arrays of these types, but some of those arrays may have a different length or be represented as sparse arrays.
Similarly, file-level metadata may be needed to interpret the links, such as the distance metric used for nearest neighbor search.
SafeGraphs should be easy to extend to support new types, metrics, and traversal mechanisms, while preserving file readability.

__Easy Distribution__.
Two file fromats are supported: a unified single-file bundle or separate `.json` manifest with an array of `.arrow` shards.
The first is the simplest variant for indexes under 100 GB.
The second is optimal for distributing billion-scale indexes.

__Easy to Evaluate__.
Optimal search indexes are asymptotically hard to build.
But it doesn't mean analyzing or evaluating them should be equally hard.


## File Structure

Similar to Hugging Face's `safetensors` format, SafeGraphs files consist of a simple JSON header followed by a series of named data blocks.
The JSON header contains human-readable metadata about the graph, such as the number of nodes and edges, the data types used, and the offsets of each data block within the file.
Each data block contains a serialized graph component in a CSR-like format, compatible with Apache Arrow arrays.

### Single File Format

For small to medium graphs, a single `.safegraphs` file contains everything.
An example may look like this:

```
┌─────────────────────────────────────┐
│ Magic number                        │
├─────────────────────────────────────┤
│ Header Size (8 bytes, little-endian)│
├─────────────────────────────────────┤
│ Header JSON (UTF-8)                 │
│ - file metadata                     │
│ - data block schemas and offsets    │
├─────────────────────────────────────┤
│ Data Block: graph 0 neighbor IDs    │
├─────────────────────────────────────┤
│ Data Block: graph 0 node vectors    │
├─────────────────────────────────────┤
│ Data Block: graph 1 ...             │
└─────────────────────────────────────┘
```

The header section here is responsible for describing the contents and layout of the file, while the data blocks contain the actual graph data.
It can easily store multiple graphs (for different layers of NSW) or arbitrary node/edge attributes packaged into a columnar form.
A minimalistic example header for a 2-layer NSW index might look like this:

```json
{
  "__metadata__": {
    "metric": "angular" | null, # optional
    "nodes": 0 | null, # optional
    "edges": 0 | null,  # optional
    "expansion": 0 | null, # optional, e.g. ef_construction
  },
  "graph_0": { ... }, # like below
  "graph_1": {
    "value": {
      "dtype": "f32", # or `u8`, bf16`, `f16`, etc.
      "shape": 0 | null, # for variable-length arrays
      "offsets": [0, 0], # integral begin and end offsets
    },
    "to": {
      "dtype": "u32", # or `u40`, `u64`, etc.
      "shape": 0 | null, # for variable-length arrays
      "offsets": [0, 0], # integral begin and end offsets
    },
    "from": { ... }, # like `to` if present
    "ids": { ... }, # commonly used to store node UUIDs or string names
    "something-else": { ... }, # just an example of extensibility
    "and-another-thing": { ... }, # etc.
  },
  "graph_2": { ... } # like above
}
```

In case of a simple Vector Search index:

- `value` array will contain equi-sized vectors, typically 768, 1024, or 1536 dimensions of `f16` half-precision floats.
- `to` array will contain a variable-length adjacency list, or fixed-length lists of 16-64 neighbor IDs per node.
- we don't need `from` incoming edges, as the greedy search only traverses outgoing edges.
- we don't need `ids` will typically contain `u64` or `uuid` node identifiers.

We also don't need additional columns for nodes or edges, but they can be added if needed.

### Sharded Format

For large graphs, split into multiple files with an index:

```
your_dataset.safegraphs.json                  # Manifest
your_dataset-00001-of-00005.safegraphs.arrow  # Shard 1
your_dataset-00002-of-00005.safegraphs.arrow  # Shard 2
...
```

Each of the individual shards is a complete `.safegraphs` file as described above, containing one or more graphs.
The neighbor IDs mentioned in their `to` and `from` columns may be referencing nodes in other shards.
The reader must first read the manifest file to determine which shards to load and how to interpret the node ID space.

## Case Studies

### BERT-like Dense Embeddings

### ColBERT Late-Interaction Multi-Vector Entries

## How Does it Compare to ...?

__Lance__.
Lance is a versioned columnar storage format with optional secondary indexes built on top, aimed at replacing Parquet for ML workloads and powering LanceDB.
SafeGraphs is the opposite slice of the problem: it has no runtime, no transaction log, no index-builder — it just freezes an already-built graph structure into a file that another library can memory-map.
Put differently, Lance _contains_ an NSW among its indexes; SafeGraphs _is_ the NSW.
Lance is the right choice when you want a mutable store with version history and compute pushdown; SafeGraphs is the right choice when library A has built an index and library B needs to load it without pulling in a database.
# IndexGraphs

IndexGraphs is a file format specification for storing and sharing graph-based index structures.
It's designed as an interoperability layer between NSW and HNSW implementations, often used in vector databases for nearest neighbor search.
Similar to Hugging Face's `safetensors` format for tensors, IndexGraphs aims to be simple, efficient, and easy to implement across different programming languages and platforms.

## Design Objectives

The design of IndexGraphs is driven by the following objectives:

__Minimalism__.
No complex dependencies or external libraries should be required to read or write IndexGraphs files.
The format should be simple and easy to implement in any programming language.
This immediately cuts out formats like Protocol Buffers, Parquet, etc.

__Memory Mapping__.
IndexGraphs files should be designed to be memory-mappable, allowing for efficient access to large graphs without loading the entire file into memory.
This greatly simplifies the implementation of readers and writers, also increasing compatibility with many new IO APIs, like Nvidia's GPUDirect Storage (GDS).
This immediately cuts out compression schemes like Snappy, Zstandard, and suggests intra-entry padding to align data blocks to SSD page sizes.

__Extensibility__.
Modern ML and AI heavily relies on mixed-precision arithmetic, starting with types like `float16`, `bfloat16`, and `int8`, and ending with `e4m3`, `e5m2` and other custom floating point formats.
Most applications deal with dense high-dimensional vectors, which are often stored as arrays of these types, but some of those arrays may have a different length or be represented as sparse arrays.
Similarly, file-level metadata may be needed to interpret the links, such as the distance metric used for nearest neighbor search.
IndexGraphs should be easy to extend to support new types, metrics, and traversal mechanisms, while preserving file readability.

## File Structure

Similar to Hugging Face's `safetensors` format, IndexGraphs files consist of a simple header followed by a series of named data blocks.
The header contains metadata about the graph, such as the number of nodes and edges, the data types used, and the offsets of each data block within the file.
Each data block contains a serialized graph component in a CSR-like format, compatible with Apache Arrow arrays.

### Single File Format

For small to medium graphs, a single `.indexgraphs` file contains everything.
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
It can easily store multiple graphs (for different layers of HNSW) or arbitrary node/edge attributes packaged into a columnar form.
A minimalistic example header for a 2-layer HNSW index might look like this:

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
your_dataset.indexgraphs.index.json       # Manifest
your_dataset-00001-of-00005.indexgraphs   # Shard 1
your_dataset-00002-of-00005.indexgraphs   # Shard 2
...
```

Each of the individual shards is a complete `.indexgraphs` file as described above, containing one or more graphs.
The neighbor IDs mentioned in their `to` and `from` columns may be referencing nodes in other shards.
The reader must first read the manifest file to determine which shards to load and how to interpret the node ID space.


### Index JSON Format

When sharding is used, the `.indexgraphs.index.json` file maps graphs to their physical locations:

```json
{
  "format_version": "0.1.0",
  "metadata": {
    "description": "HNSW index for embeddings",
    "created": "2025-10-15T10:30:00Z",
    "total_nodes": 10000000,
    "total_edges": 150000000
  },
  "shards": [
    {
      "file": "my_graph-00001-of-00005.indexgraphs",
      "graphs": ["layer_0_part_0", "layer_0_part_1"],
      "node_range": [0, 2000000],
      "size_bytes": 5368709120
    },
    {
      "file": "my_graph-00002-of-00005.indexgraphs",
      "graphs": ["layer_0_part_2", "layer_1"],
      "node_range": [2000000, 4000000],
      "size_bytes": 4294967296
    }
  ]
}
```

### Individual .indexgraphs File Format

Each `.indexgraphs` file follows this structure:

__1. Header Size__ (8 bytes)
- Unsigned 64-bit integer (little-endian)
- Specifies the byte length of the JSON header

__2. Header JSON__
- UTF-8 encoded JSON
- Contains all metadata and data block offsets

__3. Data Blocks__
- Raw binary data
- Offsets specified in header
- Aligned to 8-byte boundaries for efficient memory mapping

### Header JSON Schema

```json
{
  "format_version": "0.1.0",
  "metadata": {
    "distance_metric": "cosine",
    "description": "Optional description"
  },
  "graphs": {
    "layer_0": {
      "nodes": {
        "count": 1000,
        "dense": true,
        "columns": {
          "id": {
            "dtype": "uuid",
            "offset": 8192,
            "size": 16000
          },
          "entry": {
            "dtype": "float32",
            "shape": [128],
            "offset": 24192,
            "size": 512000
          },
          "timestamp": {
            "dtype": "uint64",
            "offset": 536192,
            "size": 8000
          }
        }
      },
      "edges": {
        "format": "adjacency_list",
        "from": {
          "dtype": "uint32",
          "offset": 544192,
          "size": 4000,
          "comment": "Optional, only if nodes.dense=false"
        },
        "to_offsets": {
          "dtype": "uint64",
          "offset": 548192,
          "size": 8008,
          "comment": "Length = node_count + 1"
        },
        "to_values": {
          "dtype": "uint32",
          "offset": 556200,
          "size": 60000,
          "comment": "Flattened adjacency lists"
        },
        "columns": {
          "distance": {
            "dtype": "float32",
            "offset": 616200,
            "size": 60000,
            "comment": "Per-edge attributes, same length as to_values"
          }
        }
      }
    },
    "layer_1": {
      "nodes": { "count": 100, "dense": true },
      "edges": { "format": "adjacency_list" }
    }
  }
}
```

### Supported Data Types

__Standard numeric types:__
- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`
- `float16`, `float32`, `float64`, `bfloat16`

__ML/AI types:__
- `e4m3`, `e5m2` (FP8 variants)
- Custom types: `{"custom": "uint40", "bits": 40}` (stored in next power-of-2)

__Structured types:__
- `uuid` (16 bytes)
- `string` (requires offsets buffer like Arrow)
- Fixed arrays: `{"array": "float32", "shape": [128]}`
- Variable arrays: `{"array": "uint32", "variable": true}` (requires offsets)

### Alignment and Padding

All data blocks must be aligned to 8-byte boundaries to ensure efficient memory mapping across platforms. Padding bytes (zeros) are inserted between blocks as needed.

## Examples

### Example 1: Simple Social Network

A small directed social graph with 4 users and their followers:

```
User 0 (alice) → follows [1, 2]
User 1 (bob)   → follows [2]
User 2 (carol) → follows [3]
User 3 (dave)  → follows []
```

__File structure (`social.indexgraphs`):__

```json
{
  "format_version": "0.1.0",
  "metadata": {
    "description": "Simple social network"
  },
  "graphs": {
    "follows": {
      "nodes": {
        "count": 4,
        "dense": true,
        "columns": {
          "username": {
            "dtype": "string",
            "offsets_offset": 8192,
            "offsets_size": 40,
            "values_offset": 8232,
            "values_size": 20
          }
        }
      },
      "edges": {
        "format": "adjacency_list",
        "to_offsets": {
          "dtype": "uint32",
          "offset": 8256,
          "size": 20
        },
        "to_values": {
          "dtype": "uint32",
          "offset": 8276,
          "size": 12
        }
      }
    }
  }
}
```

__Binary layout:__
```
| Offset | Size | Content                                         |
| ------ | ---- | ----------------------------------------------- |
| 0      | 8    | Header size: 658 (little-endian uint64)         |
| 8      | 658  | JSON header (above)                             |
| 666    | 1526 | Padding (to align to 8-byte boundary)           |
| 8192   | 40   | username offsets: [0, 5, 8, 13, 17] (uint64[5]) |
| 8232   | 20   | username values: "alicebobcaroldave" (UTF-8)    |
| 8256   | 20   | to_offsets: [0, 2, 3, 4, 4] (uint32[5])         |
| 8276   | 12   | to_values: [1, 2, 2, 3] (uint32[3])             |
```

### Example 2: HNSW Index (2-layer)

An HNSW vector index with 1000 vectors in layer 0 and 100 in layer 1:

__File structure (`embeddings.indexgraphs`):__

```json
{
  "format_version": "0.1.0",
  "metadata": {
    "distance_metric": "cosine",
    "embedding_dim": 128,
    "ef_construction": 200
  },
  "graphs": {
    "layer_0": {
      "nodes": {
        "count": 1000,
        "dense": true,
        "columns": {
          "id": {
            "dtype": "uint64",
            "offset": 8192,
            "size": 8000
          },
          "vector": {
            "dtype": "float32",
            "shape": [128],
            "offset": 16192,
            "size": 512000
          }
        }
      },
      "edges": {
        "format": "adjacency_list",
        "to_offsets": {
          "dtype": "uint64",
          "offset": 528192,
          "size": 8008
        },
        "to_values": {
          "dtype": "uint32",
          "offset": 536200,
          "size": 60000,
          "comment": "~15 neighbors per node avg"
        },
        "columns": {
          "distance": {
            "dtype": "float32",
            "offset": 596200,
            "size": 60000
          }
        }
      }
    },
    "layer_1": {
      "nodes": {
        "count": 100,
        "dense": true,
        "columns": {
          "ref_layer_0": {
            "dtype": "uint32",
            "offset": 656200,
            "size": 400,
            "comment": "Maps to layer_0 node indices"
          }
        }
      },
      "edges": {
        "format": "adjacency_list",
        "to_offsets": {
          "dtype": "uint64",
          "offset": 656600,
          "size": 808
        },
        "to_values": {
          "dtype": "uint32",
          "offset": 657408,
          "size": 2000,
          "comment": "~5 neighbors per node avg"
        },
        "columns": {
          "distance": {
            "dtype": "float32",
            "offset": 659408,
            "size": 2000
          }
        }
      }
    }
  }
}
```

### Example 3: Sparse Node IDs (Non-Dense Graph)

A graph where only certain nodes from a larger ID space are present:

```json
{
  "format_version": "0.1.0",
  "graphs": {
    "subgraph": {
      "nodes": {
        "count": 100,
        "dense": false,
        "comment": "Non-dense requires explicit 'from' in edges"
      },
      "edges": {
        "format": "adjacency_list",
        "from": {
          "dtype": "uint64",
          "offset": 8192,
          "size": 800,
          "comment": "Explicit node IDs: [1000, 1001, 1005, 1007, ...]"
        },
        "to_offsets": {
          "dtype": "uint64",
          "offset": 8992,
          "size": 808
        },
        "to_values": {
          "dtype": "uint64",
          "offset": 9800,
          "size": 2400,
          "comment": "References must match IDs in 'from' column"
        }
      }
    }
  }
}
```

### Example 4: Sharded Large Index

A 10M node HNSW index split across 5 files:

__Index file (`large_index.indexgraphs.index.json`):__

```json
{
  "format_version": "0.1.0",
  "metadata": {
    "description": "10M vector HNSW index",
    "distance_metric": "euclidean",
    "embedding_dim": 768,
    "total_nodes": 10000000,
    "total_edges": 150000000
  },
  "shards": [
    {
      "file": "large_index-00001-of-00005.indexgraphs",
      "graphs": ["layer_0_shard_0"],
      "node_range": [0, 2000000],
      "size_bytes": 6442450944
    },
    {
      "file": "large_index-00002-of-00005.indexgraphs",
      "graphs": ["layer_0_shard_1"],
      "node_range": [2000000, 4000000],
      "size_bytes": 6442450944
    },
    {
      "file": "large_index-00003-of-00005.indexgraphs",
      "graphs": ["layer_0_shard_2"],
      "node_range": [4000000, 6000000],
      "size_bytes": 6442450944
    },
    {
      "file": "large_index-00004-of-00005.indexgraphs",
      "graphs": ["layer_0_shard_3"],
      "node_range": [6000000, 8000000],
      "size_bytes": 6442450944
    },
    {
      "file": "large_index-00005-of-00005.indexgraphs",
      "graphs": ["layer_0_shard_4", "layer_1", "layer_2"],
      "node_range": [8000000, 10000000],
      "size_bytes": 6884974592,
      "comment": "Last shard also contains higher layers"
    }
  ]
}
```

## Comparison with Existing Formats


### Key Differentiators

__IndexGraphs is unique in combining:__

1. __SafeTensors-inspired simplicity__ - JSON header + raw binary, no complex libraries
2. __Graph-native design__ - Adjacency lists, layers, sparse/dense nodes built-in
3. __ML/AI focus__ - Mixed-precision types (bfloat16, e4m3, etc.), vector embeddings
4. __Interoperability__ - Explicitly designed for cross-library data exchange (USearch ↔ CuVS ↔ NetworkX)
5. __Sharding support__ - Multi-file indices with manifests (like large language models)
6. __Memory-mappable + zero dependencies__ - Critical combination for GPU/embedded use

__Closest alternatives:__
- __HDF5__ is conceptually similar but requires libhdf5 (complex, breaks minimalism goal)
- __Arrow IPC__ is columnar and memory-mappable but lacks graph structure
- __Vector library formats__ are memory-mappable but proprietary and single-library

__IndexGraphs fills the gap:__ A format that's as simple as SafeTensors, as structured as HDF5, and as graph-focused as GraphML, while being memory-mappable and interoperable.

## Implementation Notes

### Reading Files

__Minimal reader implementation:__

1. Read first 8 bytes to get header size
2. Read header JSON
3. Parse JSON to get data block offsets
4. Memory-map the file (or seek to offsets)
5. Cast pointers to appropriate types based on dtype

__Example pseudo-code:__
```python
def read_safegraph(path):
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size).decode('utf-8'))

    # Memory map the data section
    mmap = np.memmap(path, dtype=np.uint8, mode='r')

    graphs = {}
    for name, graph_spec in header['graphs'].items():
        # Extract node columns
        node_data = {}
        for col_name, col_spec in graph_spec['nodes']['columns'].items():
            offset = col_spec['offset']
            dtype = numpy_dtype(col_spec['dtype'])
            size = col_spec['size'] // dtype.itemsize
            node_data[col_name] = mmap[offset:offset+col_spec['size']].view(dtype)

        # Extract edge lists
        to_offsets = extract_column(mmap, graph_spec['edges']['to_offsets'])
        to_values = extract_column(mmap, graph_spec['edges']['to_values'])

        graphs[name] = Graph(node_data, to_offsets, to_values)

    return graphs
```

### Writing Files

__Key considerations:__

1. __Calculate all offsets first__ before writing data
   - Header starts at offset 8
   - Data blocks start after header + padding
   - Each block aligned to 8-byte boundaries

2. __Write in order:__
   - Header size (8 bytes)
   - Header JSON
   - Padding (if needed)
   - All data blocks in order of offsets specified in header

3. __Validate consistency:__
   - All offset + size calculations match actual data
   - to_offsets length = node_count + 1
   - to_values length matches last element of to_offsets
   - Per-edge columns have same length as to_values

### Memory Mapping Considerations

__Platform compatibility:__
- Use 8-byte alignment (works on x86, ARM, GPU)
- Little-endian byte order (most common)
- No compression (incompatible with mmap)

__Performance tips:__
- Large sequential reads benefit from mmap
- Random access patterns may prefer explicit I/O
- GPU: Use GPUDirect Storage when available
- Consider huge pages (Linux: `mmap(..., MAP_HUGETLB)`) for large indices

### Validation and Safety

__Required checks when reading:__

1. Format version compatibility
2. Offsets don't exceed file size
3. Offsets + sizes don't overflow
4. All referenced columns exist
5. Data types are supported
6. Alignment is correct (offset % 8 == 0)
7. For non-dense graphs: all `to_values` references exist in `from`
8. `to_offsets` is monotonically increasing
9. `to_offsets[0] == 0`
10. `to_offsets[-1] == len(to_values)`

### Extending the Format

__Adding new data types:__

Custom types should specify bit width and interpretation:
```json
{
  "dtype": {
    "custom": "posit16",
    "bits": 16,
    "spec_url": "https://posithub.org/docs/posit16"
  }
}
```

__Adding new graph formats:__

While adjacency lists are standard, other formats could be supported:
```json
{
  "edges": {
    "format": "edge_list",
    "from": {"dtype": "uint32", "offset": 8192, "size": 40000},
    "to": {"dtype": "uint32", "offset": 48192, "size": 40000}
  }
}
```

__Adding graph-level metadata:__

Use the `metadata` object for application-specific information:
```json
{
  "metadata": {
    "distance_metric": "cosine",
    "normalization": "l2",
    "quantization": "product_quantization_8x8",
    "training_dataset": "laion2B-en"
  }
}
```



### Alternatives Considered

IndexGraphs aims to satisfy three design objectives simultaneously: __minimalism__ around dependencies, __memory-mappable__ access, and __extensibility__ beyond traditional numerics.
Existing formats fail to meet all three:

| Format Category        | Examples                                      | Minimalism | Memory-Mappable | Extensibility | Why Not Suitable?                                             |
| ---------------------- | --------------------------------------------- | ---------- | --------------- | ------------- | ------------------------------------------------------------- |
| __Vector Indices__     | USearch, CuVS, FAISS, HNSWLib, Annoy, DiskANN | ✅ Simple   | ✅ Yes           | ❌ Proprietary | Single-library formats, poor interoperability, hard to extend |
| __Graph Formats__      | GraphML, GEXF, DOT, Edge List                 | ✅ Simple   | ❌ Text/XML      | ❌ None/Basic  | Human-readable but not memory-mappable, no vector support     |
| __Scientific Storage__ | HDF5, Arrow IPC, Parquet                      | ❌ Complex  | 🟡 Partial       | ✅ Excellent   | HDF5 needs `libhdf5`, Arrow/Parquet lack graph structure      |
| __Property Graphs__    | Neo4j CSV, GraphSON, RDF, JSON-LD             | 🟡 Medium   | ❌ Text          | 🟡 Schema-rich | Semantic/ontology-focused, not performance-optimized          |
| __Binary Serializers__ | Protocol Buffers, FlatBuffers, MessagePack    | ❌ Complex  | 🟡 Partial       | 🟡 Good        | Require code generation or complex parsers                    |

### Real World Challenges

__Directionality & Weights__.
Graphs come in all shapes and sizes, from simple undirected unweighted graphs to complex directed weighted multi-graphs.
Vector Search indexes typically come in a directed unweighted form, traversed in one direction only.
Distances betweens stored vectors may be considered as virtual weights, but are typically recomputed on-the-fly during construction.

__Labels & Attributes__.
Nodes and edges may have associated attributes, such as user names, timestamps, or distances.


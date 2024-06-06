# SafeGraphs

SafeGraphs is a file format specification for storing and sharing graph-like data, such as social networks, transportation networks, biological networks, and increasingly graph-based index structures for Machine Learning (ML) and Artificial Intelligence (AI) applications, such as NSW and HNSW graphs, often used in vector databases for nearest neighbor search.
It should empower interoperability between different indexing libraries, like Unum's USearch on CPUs and Nvidia's CuVS on GPUs, as well as more traditional symbolic graph libraries, like NetworkX and CuGraph.

## Design Objectives

The design of SafeGraphs is driven by the following objectives:

__Minimalism__.
No complex dependencies or external libraries should be required to read or write SafeGraphs files.
The format should be simple and easy to implement in any programming language.
This immediately cuts out formats like Protocol Buffers, Parquet, etc.

__Memory Mapping__.
SafeGraphs files should be designed to be memory-mappable, allowing for efficient access to large graphs without loading the entire file into memory.
This greatly simplifies the implementation of readers and writers, also increasing compatibility with many new IO APIs, like Nvidia's GPUDirect Storage (GDS).
This immediately cuts out compression schemes like Snappy, Zstandard, etc.

__Extensibility__.
Modern ML and AI heavily relies on mixed-precision arithmetic, starting with types like `float16`, `bfloat16`, and `int8`, and ending with `e4m3`, `e5m2` and other custom floating point formats.
Most applications deal with dense high-dimensional vectors, which are often stored as arrays of these types, but some of those arrays may have a different length or be represented as sparse arrays.
Similarly, file-level metadata may be needed to interpret the links, such as the distance metric used for nearest neighbor search.
SafeGraphs should be easy to extend to support new types, metrics, and traversal mechanisms, while preserving file readability.

## File Structure

Similar to Hugging Face's `safetensors` format, SafeGraphs files consist of a simple header followed by a series of named data blocks.
The header contains metadata about the graph, such as the number of nodes and edges, the data types used, and the offsets of each data block within the file.
Each data block contains a serialized graph component in a CSR-like format, compatible with Apache Arrow arrays.

Here's an ASCII diagram of the file structure:

```
```


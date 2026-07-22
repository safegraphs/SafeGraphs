//! The `.safegraphs` on-disk format: reader, columns, and materializer.
//!
//! A `.safegraphs` file is an 8-byte magic, a little-endian `u64` header length,
//! a JSON header that is the complete schema, and a data segment of raw,
//! Arrow-layout buffers.
//!
//! Reading is zero-copy. Buffers are reinterpreted as typed slices straight out
//! of the memory map; nothing is copied. Allocation happens only for result sets
//! and search state, through fallible APIs, so the reader never panics on memory
//! pressure.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use allocator_api2::alloc::{Allocator, Global};
use allocator_api2::vec::Vec as AVec;

use jsontape::{parse_in, Json, JsonError};

/// The 8-byte magic at offset 0 of every `.safegraphs` file.
pub const MAGIC: [u8; 8] = *b"SAFEGRPH";

/// A byte range in the data segment: a start and a length, mirroring the
/// header's `<name>_start` and `<name>_length`.
#[derive(Clone, Copy)]
pub struct ByteRange {
    pub start: usize,
    pub len: usize,
}

/// Storage width of a payload element, in bits. Storage is decoupled from
/// compute, so a `StorageDType` only says how wide an element is, never how to
/// interpret it. A 2-byte `Bin16` is equally an `f16`, a `bf16`, or a `u16`; the
/// numeric meaning lives in `metadata` and is the compute layer's concern. Widths
/// below a byte — `Bin4` for a `4`-bit nibble, `Bin1` for a packed bit — pack
/// several elements into a byte, so a row's byte length is [`row_bytes`], never a
/// per-element size times the width.
///
/// [`row_bytes`]: StorageDType::row_bytes
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StorageDType {
    Bin1,
    Bin4,
    Bin8,
    Bin16,
    Bin32,
    Bin64,
    Bin128,
}

impl StorageDType {
    /// Element width in bits, the one figure that stays exact for a sub-byte type.
    pub const fn bits(self) -> usize {
        match self {
            StorageDType::Bin1 => 1,
            StorageDType::Bin4 => 4,
            StorageDType::Bin8 => 8,
            StorageDType::Bin16 => 16,
            StorageDType::Bin32 => 32,
            StorageDType::Bin64 => 64,
            StorageDType::Bin128 => 128,
        }
    }

    /// Element size in bytes, rounded up. Meaningful only for whole-byte widths,
    /// where a column indexes elements one at a time — offsets, keys, neighbor
    /// ids. A sub-byte value column never reaches here; it measures rows through
    /// [`row_bytes`] instead.
    ///
    /// [`row_bytes`]: StorageDType::row_bytes
    pub const fn size(self) -> usize {
        self.bits().div_ceil(8)
    }

    /// Byte length of one regular row of `width` logical elements, packing sub-byte
    /// elements tight. For a whole-byte type this is `width * size()`; for `Bin4`
    /// it is `width` nibbles rounded up to bytes, for `Bin1` `width` bits.
    pub const fn row_bytes(self, width: usize) -> usize {
        (width * self.bits()).div_ceil(8)
    }
}

/// Width of a node index or prefix-sum entry. Unsigned, since offsets and node
/// ids are non-negative and no column is nullable.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OffsetDType {
    U32,
    U64,
}

impl OffsetDType {
    /// Element size in bytes.
    pub const fn size(self) -> usize {
        match self {
            OffsetDType::U32 => 4,
            OffsetDType::U64 => 8,
        }
    }
}

/// The typed counterpart to [`OffsetDType`]: the concrete scalar an offset column
/// stores, `u32` or `u64`, the same two widths that back a node index and a
/// prefix-sum entry alike. A file declares the width at runtime as an
/// [`OffsetDType`]; a searcher resolves the matching `OffsetScalar` from it and
/// traverses in that type, so the frontier and results stay as compact as the
/// graph — a `u32` node paired with an `f32` score is eight bytes. `Into<u64>`
/// widens a result only when it leaves the walk; the trait exists to seal the two
/// legal scalars and to name which [`OffsetDType`] each one is.
pub trait OffsetScalar: Copy + TryFrom<usize> + TryInto<usize> + Into<u64> + 'static {
    const OFFSET_DTYPE: OffsetDType;
}

impl OffsetScalar for u32 {
    const OFFSET_DTYPE: OffsetDType = OffsetDType::U32;
}

impl OffsetScalar for u64 {
    const OFFSET_DTYPE: OffsetDType = OffsetDType::U64;
}

/// The score-side twin of [`OffsetScalar::to_u64`]: a native metric accumulator
/// widened to the one score type a runtime caller can name. Where a walk leaves
/// the graph it widens its node through [`OffsetScalar`] and its score through
/// this, so the erased boundary is `(u64, f64)` and nothing wider. Every native
/// accumulator — `i32`, `u32`, `f32`, `f64` — fits exactly in `f64`'s mantissa, so
/// this loses nothing today; a metric scoring past 2^53 is the only thing it could.
pub trait Widen: Copy {
    fn widen(self) -> f64;
}

impl Widen for i32 {
    fn widen(self) -> f64 {
        self as f64
    }
}

impl Widen for u32 {
    fn widen(self) -> f64 {
        self as f64
    }
}

impl Widen for f32 {
    fn widen(self) -> f64 {
        self as f64
    }
}

impl Widen for f64 {
    fn widen(self) -> f64 {
        self
    }
}

/// The prefix-sum buffer of an irregular column.
#[derive(Clone, Copy)]
pub struct IrregularOffsets {
    pub dtype: OffsetDType,
    pub range: ByteRange,
}

/// How a column's rows sit in its values buffer. Regular or irregular, never
/// both.
#[derive(Clone, Copy)]
pub enum ColumnLayout {
    /// Regular rows of `width` elements. `width` is 1 for a scalar column.
    Regular { width: usize },
    /// Irregular rows addressed by a prefix-sum offsets buffer.
    Irregular { offsets: IrregularOffsets },
}

/// One column: an element type, a layout, and its values buffer. `E` is the
/// element type: `StorageDType` for payload columns whose bytes the reader does
/// not interpret, `OffsetDType` for the neighbor graph whose values are node
/// indices. The dtype string lives in the header document, reachable through
/// [`Table::dtype_name`].
#[derive(Clone, Copy)]
pub struct Column<E> {
    pub dtype: E,
    pub layout: ColumnLayout,
    pub values: ByteRange,
}

/// The bytes behind a file: a memory map, or an owned in-memory buffer. Both are
/// just memory from an allocator — the OS mmap allocator, or `A`.
enum Backing<A: Allocator = Global> {
    Mapped(Mmap),
    Owned(AVec<u8, A>),
}

impl<A: Allocator> Backing<A> {
    fn as_bytes(&self) -> &[u8] {
        match self {
            Backing::Mapped(mmap) => &mmap[..],
            Backing::Owned(v) => &v[..],
        }
    }
}

/// A SafeGraphs file, parsed down to its column descriptors.
///
/// The backing owns every byte, mapped or in-memory. The columns are small
/// copyable descriptors that point into it by offset, so buffers are handed out
/// as zero-copy slices. Only the free-form `metadata` document allocates,
/// through `A`.
pub struct Table<A: Allocator = Global> {
    /// The bytes behind every buffer, kept alive for the whole handle so borrowed
    /// slices stay valid.
    backing: Backing<A>,

    /// Byte offset of the data segment: the first byte past the padded header.
    /// Every column's `ByteRange` is relative to this.
    data_start: usize,

    /// The parsed header document, owned in `A`: the free-form `__metadata__`
    /// plus each column's descriptor and dtype string.
    header: Json<A>,

    /// Node identity: a scalar key, or an irregular column of string keys.
    key: Column<StorageDType>,

    /// Node payload: an embedding vector or a string, its bytes left opaque.
    value: Column<StorageDType>,

    /// Outgoing edges of each node, as node indices into this same file.
    neighbors: Column<OffsetDType>,
}

/// Every fallible entry point returns this. No code path panics on bad input.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The first 8 bytes are not the magic.
    BadMagic,
    /// The file is shorter than its header claims, or a buffer runs past its end.
    Truncated,
    /// The header JSON did not parse.
    Json(JsonError),
    /// A required field, such as a column or one of its buffers, is absent.
    MissingField(&'static str),
    /// A dtype string is not one the width table recognizes.
    UnknownDType,
    /// A JSON value was not the shape the schema expects.
    WrongType,
    /// A `u64` offset or length exceeds `usize` on this host.
    TooLarge,
    /// The columns disagree on node count, or a buffer size does not match its
    /// layout.
    Inconsistent,
    /// A fallible allocation failed.
    Alloc,
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<JsonError> for Error {
    fn from(e: JsonError) -> Self {
        Error::Json(e)
    }
}

/// An element type parsed from a header dtype string. `StorageDType` collapses the
/// semantic string to a storage width; `OffsetDType` keeps the two index widths.
pub trait ElementType: Copy {
    fn from_dtype(text: &[u8]) -> Result<Self, Error>;
    fn size(self) -> usize;
    /// Byte length of a regular row of `width` elements. Whole-byte by default —
    /// only [`StorageDType`] overrides it to pack sub-byte elements tight.
    fn row_bytes(self, width: usize) -> usize {
        width * self.size()
    }
}

impl ElementType for StorageDType {
    fn from_dtype(text: &[u8]) -> Result<Self, Error> {
        Ok(match text {
            b"u1x8" => StorageDType::Bin1,
            b"i4x2" | b"u4x2" => StorageDType::Bin4,
            b"u8" | b"i8" | b"e4m3" | b"e5m2" | b"e2m3" | b"e3m2" | b"utf8" => StorageDType::Bin8,
            b"u16" | b"i16" | b"f16" | b"bf16" => StorageDType::Bin16,
            b"u32" | b"i32" | b"f32" => StorageDType::Bin32,
            b"u64" | b"i64" | b"f64" => StorageDType::Bin64,
            b"bin128" | b"uuid" => StorageDType::Bin128,
            _ => return Err(Error::UnknownDType),
        })
    }

    fn size(self) -> usize {
        StorageDType::size(self)
    }

    fn row_bytes(self, width: usize) -> usize {
        StorageDType::row_bytes(self, width)
    }
}

impl ElementType for OffsetDType {
    fn from_dtype(text: &[u8]) -> Result<Self, Error> {
        Ok(match text {
            b"u32" => OffsetDType::U32,
            b"u64" => OffsetDType::U64,
            _ => return Err(Error::UnknownDType),
        })
    }

    fn size(self) -> usize {
        OffsetDType::size(self)
    }
}

fn to_usize(v: u64) -> Result<usize, Error> {
    usize::try_from(v).map_err(|_| Error::TooLarge)
}

fn field_usize<A: Allocator>(obj: &Json<A>, key: &'static str) -> Result<usize, Error> {
    let v = obj.get(key).ok_or(Error::MissingField(key))?;
    to_usize(v.as_u64().ok_or(Error::WrongType)?)
}

impl<E: ElementType> Column<E> {
    fn from_json<A: Allocator>(obj: &Json<A>) -> Result<Self, Error> {
        let dtype = E::from_dtype(
            obj.get("dtype")
                .ok_or(Error::MissingField("dtype"))?
                .as_str()
                .ok_or(Error::WrongType)?
                .as_bytes(),
        )?;

        let values = ByteRange {
            start: field_usize(obj, "values_start")?,
            len: field_usize(obj, "values_length")?,
        };

        let layout = match obj.get("offsets_dtype") {
            Some(offset_dtype) => {
                let offset_dtype = offset_dtype.as_str().ok_or(Error::WrongType)?;
                ColumnLayout::Irregular {
                    offsets: IrregularOffsets {
                        dtype: OffsetDType::from_dtype(offset_dtype.as_bytes())?,
                        range: ByteRange {
                            start: field_usize(obj, "offsets_start")?,
                            len: field_usize(obj, "offsets_length")?,
                        },
                    },
                }
            }
            None => {
                let width = match obj.get("shape") {
                    Some(v) => to_usize(v.as_u64().ok_or(Error::WrongType)?)?,
                    None => 1,
                };
                ColumnLayout::Regular { width }
            }
        };

        Ok(Column {
            dtype,
            layout,
            values,
        })
    }
}

fn column_field<E: ElementType, A: Allocator>(
    doc: &Json<A>,
    name: &'static str,
) -> Result<Column<E>, Error> {
    let obj = doc.get(name).ok_or(Error::MissingField(name))?;
    Column::<E>::from_json(obj)
}

impl Table<Global> {
    /// Memory-map a file read-only; pages fault in lazily. Metadata allocates in
    /// the global heap.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::open_in(path, Global)
    }

    /// Own the whole blob in memory, copying `bytes` in. Metadata allocates in the
    /// global heap.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut buffer: AVec<u8, Global> = AVec::new_in(Global);
        buffer
            .try_reserve_exact(bytes.len())
            .map_err(|_| Error::Alloc)?;
        buffer.extend_from_slice(bytes);
        Self::from_owned(buffer, Global)
    }

    /// Materialize a graph into the columnar form, in the global heap. The
    /// metadata object may come from any allocator.
    pub fn build<G: GraphLike>(graph: &G, metadata: Option<&Json<Global>>) -> Result<Self, Error> {
        Self::build_in(graph, metadata, Global)
    }
}

impl<A: Allocator + Clone> Table<A> {
    /// Memory-map a file read-only, allocating the metadata document in `alloc`.
    pub fn open_in<P: AsRef<Path>>(path: P, alloc: A) -> Result<Self, Error> {
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only; the file must not be mutated
        // externally for the lifetime of this handle, per mmap's contract.
        let map = unsafe { Mmap::map(&file)? };
        Self::from_backing(Backing::Mapped(map), alloc)
    }

    /// Take ownership of an in-memory blob, allocating metadata in `alloc`.
    pub fn from_owned(buffer: AVec<u8, A>, alloc: A) -> Result<Self, Error> {
        Self::from_backing(Backing::Owned(buffer), alloc)
    }

    /// Materialize a graph into the columnar form, allocating in `alloc`. The
    /// metadata object may come from any allocator `M`; it is serialized into the
    /// header.
    pub fn build_in<G: GraphLike, M: Allocator>(
        graph: &G,
        metadata: Option<&Json<M>>,
        alloc: A,
    ) -> Result<Self, Error> {
        let blob = materialize(graph, metadata, alloc.clone())?;
        Self::from_backing(Backing::Owned(blob), alloc)
    }

    /// Shared parse over any backing: frame it, parse the header, extract the
    /// columns and metadata, then validate.
    fn from_backing(backing: Backing<A>, alloc: A) -> Result<Self, Error> {
        let (data_start, header, key, value, neighbors) = {
            let bytes = backing.as_bytes();
            if bytes.get(..8) != Some(&MAGIC[..]) {
                return Err(Error::BadMagic);
            }
            let len_bytes = bytes.get(8..16).ok_or(Error::Truncated)?;
            let mut length_bytes = [0u8; 8];
            length_bytes.copy_from_slice(len_bytes);
            let header_len = to_usize(u64::from_le_bytes(length_bytes))?;
            let data_start = 16usize.checked_add(header_len).ok_or(Error::Truncated)?;
            if bytes.len() < data_start {
                return Err(Error::Truncated);
            }

            // Parse the header into an owned document.
            let header = &bytes[16..data_start];
            let doc = parse_in(header, alloc)?;
            let key = column_field::<StorageDType, A>(&doc, "key")?;
            let value = column_field::<StorageDType, A>(&doc, "value")?;
            let neighbors = column_field::<OffsetDType, A>(&doc, "neighbors")?;
            (data_start, doc, key, value, neighbors)
        };

        let file = Table {
            backing,
            data_start,
            header,
            key,
            value,
            neighbors,
        };
        file.validate()?;
        Ok(file)
    }
}

impl<A: Allocator> Table<A> {
    /// The bytes behind the file, mapped or in-memory.
    fn bytes(&self) -> &[u8] {
        self.backing.as_bytes()
    }

    /// The complete `.safegraphs` byte image — magic, header, and data segment —
    /// ready to write to a file or hand to [`from_bytes`](Table::from_bytes). For
    /// a built index this is the serialization; for a mapped one it is the mapping
    /// itself, so the borrow is always zero-copy.
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes()
    }

    /// The free-form `__metadata__` object, if present.
    pub fn metadata(&self) -> Option<&Json<A>> {
        self.header.get("__metadata__")
    }

    pub fn metadata_get(&self, key: &str) -> Option<&Json<A>> {
        self.metadata()?.get(key)
    }

    /// Declared buffer alignment, defaulting to 4096 and floored at 64.
    pub fn alignment(&self) -> usize {
        self.metadata_get("alignment")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(4096)
            .max(64)
    }

    pub fn expansion(&self) -> Option<u64> {
        self.metadata_get("expansion").and_then(|v| v.as_u64())
    }

    pub fn metric(&self) -> Option<&str> {
        self.metadata_get("metric").and_then(|v| v.as_str())
    }

    /// The original dtype string of a column, "key" / "value" / "neighbors", for
    /// the compute layer to reinterpret its opaque values buffer.
    pub fn dtype_name(&self, column: &str) -> Option<&str> {
        self.header.get(column)?.get("dtype")?.as_str()
    }

    pub fn key_column(&self) -> &Column<StorageDType> {
        &self.key
    }

    pub fn value_column(&self) -> &Column<StorageDType> {
        &self.value
    }

    pub fn neighbors_column(&self) -> &Column<OffsetDType> {
        &self.neighbors
    }

    /// The mean out-degree: total neighbor entries divided by node count. For a
    /// fixed-degree graph this is exactly the degree; for a variable-degree one it
    /// is the average. A walk seeds about this many random landmarks, the way a
    /// graph index seeds about `connectivity` — no magic constant, since the
    /// figure is read from the graph itself.
    pub fn mean_degree(&self) -> usize {
        let nodes = self.len();
        if nodes == 0 {
            return 0;
        }
        let entries = self.neighbors.values.len / self.neighbors.dtype.size();
        entries / nodes
    }
}

impl<E: ElementType> Column<E> {
    /// Number of nodes the column describes, derived from its own buffers.
    pub fn len(&self) -> usize {
        match self.layout {
            ColumnLayout::Regular { width } => {
                let per = self.dtype.row_bytes(width);
                self.values.len.checked_div(per).unwrap_or(0)
            }
            ColumnLayout::Irregular { offsets } => {
                let element_size = offsets.dtype.size();
                offsets
                    .range
                    .len
                    .checked_div(element_size)
                    .map_or(0, |count| count.saturating_sub(1))
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Reinterpret a byte slice as `&[T]`, refusing a wrong length or misalignment.
pub fn cast<T: Copy>(bytes: &[u8]) -> Option<&[T]> {
    let size = core::mem::size_of::<T>();
    if size == 0 || !bytes.len().is_multiple_of(size) {
        return None;
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<T>()) {
        return None;
    }
    // SAFETY: the pointer is aligned for `T` and the length is a whole number of
    // `T`; `T: Copy` and the reader's numeric use make the bytes a valid value.
    Some(unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / size) })
}

/// A node's outgoing edges: node indices in this file, as `u32` or `u64`. An
/// index at or past the file's node count names no node and is a sentinel by
/// convention, so a reader skips it; a producer may therefore pad a fixed-degree
/// row with such a value rather than emit a variable-length list. This type holds
/// only the raw bytes, so the skip is applied by whoever knows the node count,
/// namely the traversal.
pub struct NeighborList<'a> {
    bytes: &'a [u8],
    dtype: OffsetDType,
}

impl<'a> NeighborList<'a> {
    /// Wrap raw neighbor bytes tagged with their index width. Used by the reader
    /// and, across the crate, by search tests building fixtures.
    pub(crate) fn new(bytes: &'a [u8], dtype: OffsetDType) -> Self {
        NeighborList { bytes, dtype }
    }

    pub fn len(&self) -> usize {
        self.bytes.len() / self.dtype.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// This node's neighbors as a native `&[I]`, zero-copy, or `None` when `I`'s
    /// width does not match the stored one. A searcher resolves `I` from the same
    /// table, so the width agrees and the walk reads its neighbors without a copy
    /// or a widen; the check is what makes a mismatched reader fail cleanly rather
    /// than reinterpret bytes.
    pub fn as_slice<I: OffsetScalar>(&self) -> Option<&'a [I]> {
        if self.dtype != I::OFFSET_DTYPE {
            return None;
        }
        cast(self.bytes)
    }
}

impl<A: Allocator> Table<A> {
    /// The whole value buffer, contiguous, for a batch scan. Only meaningful for
    /// a regular value column; pair with [`GraphLike::value_shape`].
    pub fn value_buffer(&self) -> &[u8] {
        self.buffer(self.value.values)
    }

    /// The value column's prefix-sum buffer and its width, or `None` when the
    /// column is regular and has no offsets at all.
    ///
    /// Exposed because a variable-length column is exactly an Arrow-style tape —
    /// prefix sums plus concatenated bytes — and a sequence metric can hand that
    /// pair to its backend with no copy, provided it can reach both halves.
    pub fn value_offsets(&self) -> Option<(&[u8], OffsetDType)> {
        match self.value.layout {
            ColumnLayout::Regular { .. } => None,
            ColumnLayout::Irregular { offsets } => {
                Some((self.buffer(offsets.range), offsets.dtype))
            }
        }
    }

    /// The node whose key equals `key` byte-for-byte, by linear scan.
    pub fn find_key(&self, key: &[u8]) -> Option<usize> {
        (0..self.len()).find(|&node| self.node_bytes(&self.key, node) == Some(key))
    }

    /// Slice a validated buffer out of the data segment. Relies on the open-time
    /// invariant that every stored `ByteRange` lies within the map, and degrades
    /// to an empty slice rather than panicking if that were ever violated.
    fn buffer(&self, range: ByteRange) -> &[u8] {
        let start = self.data_start.wrapping_add(range.start);
        let end = start.wrapping_add(range.len);
        self.bytes().get(start..end).unwrap_or(&[])
    }

    /// Slice `len` bytes at `offset` within a column's values buffer, checking the
    /// sub-range stays inside it.
    fn subslice(&self, values: ByteRange, offset: usize, len: usize) -> Option<&[u8]> {
        if offset.checked_add(len)? > values.len {
            return None;
        }
        Some(self.buffer(ByteRange {
            start: values.start.checked_add(offset)?,
            len,
        }))
    }

    /// The `[begin, end)` element indices of node `node` in an irregular column,
    /// read from its prefix-sum buffer and checked monotonic.
    fn node_span(&self, offsets: IrregularOffsets, node: usize) -> Option<(usize, usize)> {
        let element_size = offsets.dtype.size();
        let offset_bytes = self.buffer(offsets.range);
        let read = |i: usize| -> Option<usize> {
            let slot = offset_bytes.get(
                i.checked_mul(element_size)?
                    ..i.checked_mul(element_size)?.checked_add(element_size)?,
            )?;
            match offsets.dtype {
                OffsetDType::U32 => Some(u32::from_le_bytes(slot.try_into().ok()?) as usize),
                OffsetDType::U64 => to_usize(u64::from_le_bytes(slot.try_into().ok()?)).ok(),
            }
        };
        let begin = read(node)?;
        let end = read(node.checked_add(1)?)?;
        if begin > end {
            None
        } else {
            Some((begin, end))
        }
    }

    /// Raw bytes of node `node` in any column, regular or irregular.
    fn node_bytes<E: ElementType>(&self, column: &Column<E>, node: usize) -> Option<&[u8]> {
        match column.layout {
            ColumnLayout::Regular { width } => {
                let per = column.dtype.row_bytes(width);
                self.subslice(column.values, node.checked_mul(per)?, per)
            }
            ColumnLayout::Irregular { offsets } => {
                let element_size = column.dtype.size();
                let (begin, end) = self.node_span(offsets, node)?;
                let offset = begin.checked_mul(element_size)?;
                let len = end.checked_sub(begin)?.checked_mul(element_size)?;
                self.subslice(column.values, offset, len)
            }
        }
    }

    /// Check every buffer lies within the map and the columns agree on shape.
    fn validate(&self) -> Result<(), Error> {
        let len = self.key.len();
        self.check_layout(&self.key, len)?;
        self.check_layout(&self.value, len)?;
        self.check_layout(&self.neighbors, len)?;
        if let Some(declared) = self.metadata_get("nodes").and_then(|v| v.as_u64()) {
            if usize::try_from(declared).ok() != Some(len) {
                return Err(Error::Inconsistent);
            }
        }
        Ok(())
    }

    /// Confirm one column's buffers fit the map and match `len` exactly.
    fn check_layout<E: ElementType>(&self, column: &Column<E>, len: usize) -> Result<(), Error> {
        match column.layout {
            ColumnLayout::Regular { width } => {
                let per = column.dtype.row_bytes(width);
                if per.checked_mul(len) != Some(column.values.len) {
                    return Err(Error::Inconsistent);
                }
            }
            ColumnLayout::Irregular { offsets } => {
                self.check_buffer(offsets.range)?;
                let expected = len
                    .checked_add(1)
                    .and_then(|count| count.checked_mul(offsets.dtype.size()))
                    .ok_or(Error::TooLarge)?;
                if offsets.range.len != expected {
                    return Err(Error::Inconsistent);
                }
            }
        }
        self.check_buffer(column.values)
    }

    /// Confirm a buffer's end falls within the mapped file.
    fn check_buffer(&self, range: ByteRange) -> Result<(), Error> {
        let end = self
            .data_start
            .checked_add(range.start)
            .and_then(|start| start.checked_add(range.len))
            .ok_or(Error::TooLarge)?;
        if end <= self.bytes().len() {
            Ok(())
        } else {
            Err(Error::Truncated)
        }
    }
}

impl<A: Allocator> GraphLike for Table<A> {
    fn len(&self) -> usize {
        self.key.len()
    }

    fn key_dtype(&self) -> &str {
        self.dtype_name("key").unwrap_or("")
    }

    fn value_dtype(&self) -> &str {
        self.dtype_name("value").unwrap_or("")
    }

    fn value_shape(&self) -> Option<usize> {
        match self.value.layout {
            ColumnLayout::Regular { width } => Some(width),
            ColumnLayout::Irregular { .. } => None,
        }
    }

    fn neighbors_degree(&self) -> Option<usize> {
        match self.neighbors.layout {
            ColumnLayout::Regular { width } => Some(width),
            ColumnLayout::Irregular { .. } => None,
        }
    }

    fn key(&self, node: usize) -> &[u8] {
        self.node_bytes(&self.key, node).unwrap_or(&[])
    }

    fn value(&self, node: usize) -> &[u8] {
        self.node_bytes(&self.value, node).unwrap_or(&[])
    }

    fn neighbors(&self, node: usize) -> NeighborList<'_> {
        NeighborList::new(
            self.node_bytes(&self.neighbors, node).unwrap_or(&[]),
            self.neighbors.dtype,
        )
    }
}

/// One node's three payloads, yielded by [`GraphLike::iter`].
pub struct Node<'a> {
    pub key: &'a [u8],
    pub value: &'a [u8],
    pub neighbors: NeighborList<'a>,
}

/// A graph read node by node. Both a producer's graph and a parsed [`Table`]
/// implement it, so a `Table` can be built from any `GraphLike`, including
/// another `Table`.
pub trait GraphLike {
    /// Number of nodes.
    fn len(&self) -> usize;

    /// True when there are no nodes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Header dtype strings for the payload columns, e.g. "u64", "f16".
    fn key_dtype(&self) -> &str;
    fn value_dtype(&self) -> &str;

    /// The value column's fixed shape, or None for variable-length values.
    fn value_shape(&self) -> Option<usize>;
    /// The neighbors column's fixed out-degree, or None if variable.
    fn neighbors_degree(&self) -> Option<usize>;

    /// Per-node payloads. Keys and values pass through verbatim; neighbors are
    /// node indices, u32 or u64, read through [`NeighborList`].
    fn key(&self, node: usize) -> &[u8];
    fn value(&self, node: usize) -> &[u8];
    fn neighbors(&self, node: usize) -> NeighborList<'_>;

    /// Iterate nodes, zipping the three columns.
    fn iter(&self) -> GraphIter<'_, Self>
    where
        Self: Sized,
    {
        GraphIter {
            graph: self,
            node: 0,
        }
    }
}

/// Iterator over a [`GraphLike`]'s nodes.
pub struct GraphIter<'a, G: GraphLike> {
    graph: &'a G,
    node: usize,
}

impl<'a, G: GraphLike> Iterator for GraphIter<'a, G> {
    type Item = Node<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.node >= self.graph.len() {
            return None;
        }
        let i = self.node;
        self.node += 1;
        Some(Node {
            key: self.graph.key(i),
            value: self.graph.value(i),
            neighbors: self.graph.neighbors(i),
        })
    }
}

fn push_bytes<A: Allocator>(v: &mut AVec<u8, A>, bytes: &[u8]) -> Result<(), Error> {
    v.try_reserve(bytes.len()).map_err(|_| Error::Alloc)?;
    v.extend_from_slice(bytes);
    Ok(())
}

fn push_offset<A: Allocator>(
    v: &mut AVec<u8, A>,
    dtype: OffsetDType,
    value: u64,
) -> Result<(), Error> {
    match dtype {
        OffsetDType::U32 => {
            let narrow = u32::try_from(value).map_err(|_| Error::TooLarge)?;
            push_bytes(v, &narrow.to_le_bytes())
        }
        OffsetDType::U64 => push_bytes(v, &value.to_le_bytes()),
    }
}

fn align_up(x: usize, alignment: usize) -> Result<usize, Error> {
    let rem = x % alignment;
    if rem == 0 {
        Ok(x)
    } else {
        x.checked_add(alignment - rem).ok_or(Error::TooLarge)
    }
}

fn reserve_span(cursor: &mut usize, len: usize, alignment: usize) -> Result<usize, Error> {
    let start = align_up(*cursor, alignment)?;
    *cursor = start.checked_add(len).ok_or(Error::TooLarge)?;
    Ok(start)
}

/// Accumulates one column's on-disk buffers: concatenated values, and u64
/// prefix-sum offsets when the column is irregular.
struct ColumnBuilder<A: Allocator> {
    dtype: StorageDType,
    width: Option<usize>,
    values: AVec<u8, A>,
    offsets: AVec<u8, A>,
    running: u64,
}

impl<A: Allocator + Clone> ColumnBuilder<A> {
    fn new(dtype: StorageDType, width: Option<usize>, alloc: &A) -> Result<Self, Error> {
        let mut offsets = AVec::new_in(alloc.clone());
        if width.is_none() {
            push_bytes(&mut offsets, &0u64.to_le_bytes())?;
        }
        Ok(ColumnBuilder {
            dtype,
            width,
            values: AVec::new_in(alloc.clone()),
            offsets,
            running: 0,
        })
    }

    fn push_row(&mut self, bytes: &[u8]) -> Result<(), Error> {
        match self.width {
            // Regular: every row is exactly `row_bytes(width)`, packing sub-byte
            // elements tight, so the length is checked against that, never against
            // a per-element size the packing would make fractional.
            Some(width) => {
                if bytes.len() != self.dtype.row_bytes(width) {
                    return Err(Error::Inconsistent);
                }
                push_bytes(&mut self.values, bytes)?;
            }
            // Irregular: whole-byte elements counted for the prefix sum. A sub-byte
            // dtype never arrives here — only sequences are irregular, and they are
            // `utf8`, a byte.
            None => {
                let element_size = self.dtype.size();
                if element_size == 0 || !bytes.len().is_multiple_of(element_size) {
                    return Err(Error::Inconsistent);
                }
                push_bytes(&mut self.values, bytes)?;
                self.running += (bytes.len() / element_size) as u64;
                push_bytes(&mut self.offsets, &self.running.to_le_bytes())?;
            }
        }
        Ok(())
    }

    fn is_irregular(&self) -> bool {
        self.width.is_none()
    }
}

fn reserve_offsets<A: Allocator + Clone>(
    column: &ColumnBuilder<A>,
    cursor: &mut usize,
    alignment: usize,
) -> Result<Option<usize>, Error> {
    if column.is_irregular() {
        Ok(Some(reserve_span(cursor, column.offsets.len(), alignment)?))
    } else {
        Ok(None)
    }
}

fn write_column<A: Allocator>(
    header: &mut String,
    name: &str,
    dtype: &str,
    column: &ColumnBuilder<A>,
    off_start: Option<usize>,
    val_start: usize,
) {
    header.push_str(&format!(",\"{name}\":{{\"dtype\":\"{dtype}\""));
    match off_start {
        Some(off) => header.push_str(&format!(
            ",\"offsets_dtype\":\"u64\",\"offsets_start\":{},\"offsets_length\":{}",
            off,
            column.offsets.len()
        )),
        None => header.push_str(&format!(",\"shape\":{}", column.width.unwrap_or(1))),
    }
    header.push_str(&format!(
        ",\"values_start\":{},\"values_length\":{}}}",
        val_start,
        column.values.len()
    ));
}

fn write_buffer<A: Allocator>(
    file: &mut AVec<u8, A>,
    data_start: usize,
    start: usize,
    bytes: &[u8],
) {
    let target = data_start + start;
    while file.len() < target {
        file.push(0);
    }
    file.extend_from_slice(bytes);
}

/// Serialize a graph into a complete `.safegraphs` blob in `alloc`.
fn materialize<G, M, A>(
    graph: &G,
    metadata: Option<&Json<M>>,
    alloc: A,
) -> Result<AVec<u8, A>, Error>
where
    G: GraphLike,
    M: Allocator,
    A: Allocator + Clone,
{
    let node_count = graph.len();
    let alignment = metadata
        .and_then(|object| object.get("alignment"))
        .and_then(|v| v.as_u64())
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(4096)
        .max(64);

    let key_dtype = graph.key_dtype();
    let key_storage = StorageDType::from_dtype(key_dtype.as_bytes())?;
    let key_width = if key_dtype == "utf8" { None } else { Some(1) };
    let mut key = ColumnBuilder::new(key_storage, key_width, &alloc)?;
    for i in 0..node_count {
        key.push_row(graph.key(i))?;
    }

    let value_dtype = graph.value_dtype();
    let value_storage = StorageDType::from_dtype(value_dtype.as_bytes())?;
    let mut value = ColumnBuilder::new(value_storage, graph.value_shape(), &alloc)?;
    for i in 0..node_count {
        value.push_row(graph.value(i))?;
    }

    // Widen the neighbor index type to whatever the node count needs.
    let neighbor_dtype = if node_count > u32::MAX as usize {
        OffsetDType::U64
    } else {
        OffsetDType::U32
    };
    let (neighbor_dtype_name, neighbor_storage) = match neighbor_dtype {
        OffsetDType::U32 => ("u32", StorageDType::Bin32),
        OffsetDType::U64 => ("u64", StorageDType::Bin64),
    };
    let mut neighbors = ColumnBuilder::new(neighbor_storage, graph.neighbors_degree(), &alloc)?;
    let mut scratch: AVec<u8, A> = AVec::new_in(alloc.clone());
    for i in 0..node_count {
        let list = graph.neighbors(i);
        if list.dtype == neighbor_dtype {
            // Same width: the bytes are already node indices in the target
            // encoding, so copy the row verbatim with no per-edge widening. This
            // is the path a re-materialized `Table` always takes.
            neighbors.push_row(list.bytes)?;
        } else {
            // The one place widths genuinely cross, when a producer's graph and
            // the chosen target width differ. Read each edge native, write it in
            // the target width.
            scratch.clear();
            match list.dtype {
                OffsetDType::U32 => {
                    for &edge in cast::<u32>(list.bytes).ok_or(Error::WrongType)? {
                        push_offset(&mut scratch, neighbor_dtype, edge as u64)?;
                    }
                }
                OffsetDType::U64 => {
                    for &edge in cast::<u64>(list.bytes).ok_or(Error::WrongType)? {
                        push_offset(&mut scratch, neighbor_dtype, edge)?;
                    }
                }
            }
            neighbors.push_row(&scratch)?;
        }
    }

    let mut cursor = 0usize;
    let key_offsets_start = reserve_offsets(&key, &mut cursor, alignment)?;
    let key_values_start = reserve_span(&mut cursor, key.values.len(), alignment)?;
    let value_offsets_start = reserve_offsets(&value, &mut cursor, alignment)?;
    let value_values_start = reserve_span(&mut cursor, value.values.len(), alignment)?;
    let neighbor_offsets_start = reserve_offsets(&neighbors, &mut cursor, alignment)?;
    let neighbor_values_start = reserve_span(&mut cursor, neighbors.values.len(), alignment)?;
    let data_len = cursor;

    let mut header = String::new();
    header.push('{');
    header.push_str("\"__metadata__\":");
    match metadata {
        Some(object) => object.write_json(&mut header).map_err(|_| Error::Alloc)?,
        None => header.push_str("{}"),
    }
    write_column(
        &mut header,
        "key",
        key_dtype,
        &key,
        key_offsets_start,
        key_values_start,
    );
    write_column(
        &mut header,
        "value",
        value_dtype,
        &value,
        value_offsets_start,
        value_values_start,
    );
    write_column(
        &mut header,
        "neighbors",
        neighbor_dtype_name,
        &neighbors,
        neighbor_offsets_start,
        neighbor_values_start,
    );
    header.push('}');

    let unpadded = 16usize.checked_add(header.len()).ok_or(Error::TooLarge)?;
    let header_pad = (alignment - (unpadded % alignment)) % alignment;
    let header_len = header.len() + header_pad;
    let data_start = 16 + header_len;
    let total = data_start.checked_add(data_len).ok_or(Error::TooLarge)?;

    let mut file: AVec<u8, A> = AVec::new_in(alloc);
    file.try_reserve_exact(total).map_err(|_| Error::Alloc)?;
    file.extend_from_slice(&MAGIC);
    file.extend_from_slice(&(header_len as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    for _ in 0..header_pad {
        file.push(b' ');
    }
    if let Some(off) = key_offsets_start {
        write_buffer(&mut file, data_start, off, &key.offsets);
    }
    write_buffer(&mut file, data_start, key_values_start, &key.values);
    if let Some(off) = value_offsets_start {
        write_buffer(&mut file, data_start, off, &value.offsets);
    }
    write_buffer(&mut file, data_start, value_values_start, &value.values);
    if let Some(off) = neighbor_offsets_start {
        write_buffer(&mut file, data_start, off, &neighbors.offsets);
    }
    write_buffer(
        &mut file,
        data_start,
        neighbor_values_start,
        &neighbors.values,
    );

    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Toy {
        nodes: usize,
        dim: usize,
        degree: usize,
        keys: Vec<u8>,
        values: Vec<u8>,
        adjacency: Vec<u8>, // u32 node indices, little-endian
    }

    impl GraphLike for Toy {
        fn len(&self) -> usize {
            self.nodes
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "f32"
        }
        fn value_shape(&self) -> Option<usize> {
            Some(self.dim)
        }
        fn neighbors_degree(&self) -> Option<usize> {
            Some(self.degree)
        }
        fn key(&self, i: usize) -> &[u8] {
            &self.keys[i * 8..i * 8 + 8]
        }
        fn value(&self, i: usize) -> &[u8] {
            let w = self.dim * 4;
            &self.values[i * w..i * w + w]
        }
        fn neighbors(&self, i: usize) -> NeighborList<'_> {
            let w = self.degree * 4;
            NeighborList {
                bytes: &self.adjacency[i * w..i * w + w],
                dtype: OffsetDType::U32,
            }
        }
    }

    fn toy() -> Toy {
        let mut keys = Vec::new();
        for k in [10u64, 20, 30] {
            keys.extend_from_slice(&k.to_le_bytes());
        }
        let mut values = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            values.extend_from_slice(&v.to_le_bytes());
        }
        let mut adjacency = Vec::new();
        for node in [1u32, 2, 2, 0, 0, 1] {
            adjacency.extend_from_slice(&node.to_le_bytes());
        }
        Toy {
            nodes: 3,
            dim: 2,
            degree: 2,
            keys,
            values,
            adjacency,
        }
    }

    #[test]
    fn round_trip_regular() {
        let table = Table::build(&toy(), None).unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(table.value_shape(), Some(2));
        assert_eq!(table.key(1), &20u64.to_le_bytes()[..]);

        let mut expected = Vec::new();
        expected.extend_from_slice(&5.0f32.to_le_bytes());
        expected.extend_from_slice(&6.0f32.to_le_bytes());
        assert_eq!(table.value(2), &expected[..]);

        let edges = table.neighbors(0);
        assert_eq!(edges.as_slice::<u32>().unwrap(), &[1u32, 2]);

        assert_eq!(table.dtype_name("value"), Some("f32"));
        assert_eq!(table.dtype_name("key"), Some("u64"));
    }

    #[test]
    fn round_trip_through_bytes() {
        let table = Table::build(&toy(), None).unwrap();
        let bytes = table.bytes().to_vec();
        let reopened = Table::from_bytes(&bytes).unwrap();
        assert_eq!(reopened.len(), 3);
        assert_eq!(reopened.key(2), &30u64.to_le_bytes()[..]);
    }

    #[test]
    fn table_is_graphlike() {
        // A Table implements GraphLike, so it can be re-materialized into another.
        let table = Table::build(&toy(), None).unwrap();
        let rebuilt = Table::build(&table, None).unwrap();
        assert_eq!(rebuilt.len(), 3);
        assert_eq!(rebuilt.key(0), &10u64.to_le_bytes()[..]);
        let edges = rebuilt.neighbors(1);
        assert_eq!(edges.as_slice::<u32>().unwrap(), &[2u32, 0]);
    }

    #[test]
    fn graph_iter_zips_columns() {
        let g = toy();
        let shapes: Vec<_> = g
            .iter()
            .map(|node| (node.key.len(), node.value.len(), node.neighbors.len()))
            .collect();
        assert_eq!(shapes, vec![(8, 8, 2), (8, 8, 2), (8, 8, 2)]);
    }
}

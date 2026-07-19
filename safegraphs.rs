//! SafeGraphs — the SafeTensors of graphs.
//!
//! A single-file reader for the `.safegraphs` format: an 8-byte magic, a
//! little-endian `u64` header length, a JSON header that is the complete schema,
//! and a data segment of raw, Arrow-layout buffers.
//!
//! Reading is zero-copy. Buffers are reinterpreted as typed slices straight out
//! of the memory map; nothing is copied. Allocation happens only for result sets
//! and search state, through fallible APIs, so the reader never panics on memory
//! pressure.
//!
//! The default build only reads and unpacks. The `vectors` feature pulls NumKong
//! for scans and A*; the `strings` feature pulls StringZilla for integrity and
//! biological metrics; both pull ForkUnion for parallelism.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use allocator_api2::alloc::{Allocator, Global};
use allocator_api2::vec::Vec as AVec;

use jsontape::{parse_in, JsonError, JsonView, Span};

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
/// compute, so a `StorageDType` only says how many bytes an element occupies, never how
/// to interpret it. A 2-byte `Bin16` is equally an `f16`, a `bf16`, or a `u16`;
/// the numeric meaning lives in `metadata` and is the compute layer's concern.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StorageDType {
    Bin8,
    Bin16,
    Bin32,
    Bin64,
    Bin128,
}

impl StorageDType {
    /// Element size in bytes.
    pub const fn size(self) -> usize {
        match self {
            StorageDType::Bin8 => 1,
            StorageDType::Bin16 => 2,
            StorageDType::Bin32 => 4,
            StorageDType::Bin64 => 8,
            StorageDType::Bin128 => 16,
        }
    }
}

/// Width of a row index or prefix-sum entry. Unsigned, since offsets and row
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
/// element type: `StorageDType` for payload columns whose bytes the reader does not
/// interpret, `OffsetDType` for the neighbor graph whose values are row indices.
#[derive(Clone, Copy)]
pub struct Column<E> {
    pub dtype: E,
    /// The original dtype string from the header, a span kept so the compute
    /// layer can reinterpret the opaque values buffer.
    pub name: Span,
    pub layout: ColumnLayout,
    pub values: ByteRange,
}

/// The `__metadata__` document, kept verbatim as a free-form JSON tape.
pub struct Metadata<A: Allocator = Global>(pub JsonView<A>);

/// The bytes behind a file: a memory map, or an owned in-memory buffer. Both are
/// just memory from an allocator — the OS mmap allocator, or `A`.
enum Backing<A: Allocator = Global> {
    Mapped(Mmap),
    Owned(AVec<u8, A>),
}

impl<A: Allocator> Backing<A> {
    fn as_bytes(&self) -> &[u8] {
        match self {
            Backing::Mapped(m) => &m[..],
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
pub struct SafeGraphsFile<A: Allocator = Global> {
    /// The bytes behind every buffer, kept alive for the whole handle so borrowed
    /// slices stay valid.
    backing: Backing<A>,

    /// Byte offset of the data segment: the first byte past the padded header.
    /// Every column's `ByteRange` is relative to this.
    data_start: usize,

    /// The `__metadata__` document, free-form and owned in `A`.
    metadata: Metadata<A>,

    /// Node identity: a scalar key, or an irregular column of string keys.
    key: Column<StorageDType>,

    /// Node payload: an embedding vector or a string, its bytes left opaque.
    value: Column<StorageDType>,

    /// Outgoing edges of each node, as row indices into this same file.
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
    fn from_str(s: &[u8]) -> Result<Self, Error>;
    fn size(self) -> usize;
}

impl ElementType for StorageDType {
    fn from_str(s: &[u8]) -> Result<Self, Error> {
        Ok(match s {
            b"u8" | b"i8" | b"e4m3" | b"e5m2" | b"utf8" => StorageDType::Bin8,
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
}

impl ElementType for OffsetDType {
    fn from_str(s: &[u8]) -> Result<Self, Error> {
        Ok(match s {
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

fn field_usize<A: Allocator>(
    header: &[u8],
    obj: &JsonView<A>,
    key: &'static str,
) -> Result<usize, Error> {
    let v = obj.get(header, key).ok_or(Error::MissingField(key))?;
    to_usize(v.as_u64().ok_or(Error::WrongType)?)
}

impl<E: ElementType> Column<E> {
    fn from_json<A: Allocator>(header: &[u8], obj: &JsonView<A>) -> Result<Self, Error> {
        let name = match obj.get(header, "dtype") {
            Some(JsonView::String(s)) => *s,
            Some(_) => return Err(Error::WrongType),
            None => return Err(Error::MissingField("dtype")),
        };
        let dtype = E::from_str(name.resolve(header).ok_or(Error::WrongType)?.as_bytes())?;

        let values = ByteRange {
            start: field_usize(header, obj, "values_start")?,
            len: field_usize(header, obj, "values_length")?,
        };

        let layout = match obj.get(header, "offsets_dtype") {
            Some(od) => {
                let od = od.as_str(header).ok_or(Error::WrongType)?;
                ColumnLayout::Irregular {
                    offsets: IrregularOffsets {
                        dtype: OffsetDType::from_str(od.as_bytes())?,
                        range: ByteRange {
                            start: field_usize(header, obj, "offsets_start")?,
                            len: field_usize(header, obj, "offsets_length")?,
                        },
                    },
                }
            }
            None => {
                let width = match obj.get(header, "shape") {
                    Some(v) => to_usize(v.as_u64().ok_or(Error::WrongType)?)?,
                    None => 1,
                };
                ColumnLayout::Regular { width }
            }
        };

        Ok(Column {
            dtype,
            name,
            layout,
            values,
        })
    }
}

fn column_field<E: ElementType, A: Allocator>(
    header: &[u8],
    doc: &JsonView<A>,
    name: &'static str,
) -> Result<Column<E>, Error> {
    let obj = doc.get(header, name).ok_or(Error::MissingField(name))?;
    Column::<E>::from_json(header, obj)
}

/// Move the `__metadata__` subtree out of the header document, or an empty
/// object if absent.
fn take_metadata<A: Allocator>(header: &[u8], doc: JsonView<A>, alloc: A) -> Metadata<A> {
    if let JsonView::Object(mut items) = doc {
        if let Some(i) = items
            .iter()
            .position(|(k, _)| k.resolve(header) == Some("__metadata__"))
        {
            let (_, v) = items.swap_remove(i);
            return Metadata(v);
        }
    }
    Metadata(JsonView::Object(AVec::new_in(alloc)))
}

impl SafeGraphsFile<Global> {
    /// Open and map a file read-only, allocating metadata in the global heap.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::open_in(path, Global)
    }

    /// Parse a file already in memory, copying `bytes` into an owned buffer.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, Error> {
        let mut buffer: AVec<u8, Global> = AVec::new_in(Global);
        buffer
            .try_reserve_exact(bytes.len())
            .map_err(|_| Error::Alloc)?;
        buffer.extend_from_slice(bytes);
        Self::from_bytes_in(buffer, Global)
    }
}

impl<A: Allocator + Clone> SafeGraphsFile<A> {
    /// Open and map a file read-only, allocating the metadata document in `alloc`.
    pub fn open_in<P: AsRef<Path>>(path: P, alloc: A) -> Result<Self, Error> {
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only; the file must not be mutated
        // externally for the lifetime of this handle, per mmap's contract.
        let map = unsafe { Mmap::map(&file)? };
        Self::from_backing(Backing::Mapped(map), alloc)
    }

    /// Parse a file from an owned in-memory buffer, allocating metadata in `alloc`.
    pub fn from_bytes_in(buffer: AVec<u8, A>, alloc: A) -> Result<Self, Error> {
        Self::from_backing(Backing::Owned(buffer), alloc)
    }

    /// Shared parse over any backing: frame it, parse the header, extract the
    /// columns and metadata, then validate.
    fn from_backing(backing: Backing<A>, alloc: A) -> Result<Self, Error> {
        let (data_start, metadata, key, value, neighbors) = {
            let bytes = backing.as_bytes();
            if bytes.get(..8) != Some(&MAGIC[..]) {
                return Err(Error::BadMagic);
            }
            let len_bytes = bytes.get(8..16).ok_or(Error::Truncated)?;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(len_bytes);
            let header_len = to_usize(u64::from_le_bytes(buf))?;
            let data_start = 16usize.checked_add(header_len).ok_or(Error::Truncated)?;
            if bytes.len() < data_start {
                return Err(Error::Truncated);
            }

            // Parse the header. String spans in the document are relative to it.
            let header = &bytes[16..data_start];
            let doc = parse_in(header, alloc.clone())?;
            let key = column_field::<StorageDType, A>(header, &doc, "key")?;
            let value = column_field::<StorageDType, A>(header, &doc, "value")?;
            let neighbors = column_field::<OffsetDType, A>(header, &doc, "neighbors")?;
            let metadata = take_metadata(header, doc, alloc);
            (data_start, metadata, key, value, neighbors)
        };

        let file = SafeGraphsFile {
            backing,
            data_start,
            metadata,
            key,
            value,
            neighbors,
        };
        file.validate()?;
        Ok(file)
    }
}

impl<A: Allocator> SafeGraphsFile<A> {
    /// The bytes behind the file, mapped or in-memory.
    fn bytes(&self) -> &[u8] {
        self.backing.as_bytes()
    }

    /// The header bytes, against which header spans resolve.
    fn header(&self) -> &[u8] {
        &self.bytes()[16..self.data_start]
    }

    pub fn metadata(&self) -> &Metadata<A> {
        &self.metadata
    }

    pub fn metadata_get(&self, key: &str) -> Option<&JsonView<A>> {
        self.metadata.0.get(self.header(), key)
    }

    /// Declared buffer alignment, defaulting to 4096 and floored at 64.
    pub fn alignment(&self) -> usize {
        self.metadata_get("alignment")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(4096)
            .max(64)
    }

    pub fn nodes(&self) -> Option<u64> {
        self.metadata_get("nodes").and_then(|v| v.as_u64())
    }

    pub fn expansion(&self) -> Option<u64> {
        self.metadata_get("expansion").and_then(|v| v.as_u64())
    }

    pub fn metric(&self) -> Option<&str> {
        self.metadata_get("metric")
            .and_then(|v| v.as_str(self.header()))
    }

    /// The original dtype string of a column, for the compute layer to
    /// reinterpret its opaque values buffer.
    pub fn dtype_name(&self, name: Span) -> Option<&str> {
        name.resolve(self.header())
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
}

impl<E: ElementType> Column<E> {
    /// Number of rows the column describes, derived from its own buffers.
    pub fn rows(&self) -> usize {
        match self.layout {
            ColumnLayout::Regular { width } => {
                let per = width * self.dtype.size();
                if per == 0 {
                    0
                } else {
                    self.values.len / per
                }
            }
            ColumnLayout::Irregular { offsets } => {
                let s = offsets.dtype.size();
                if s == 0 {
                    0
                } else {
                    (offsets.range.len / s).saturating_sub(1)
                }
            }
        }
    }
}

/// Reinterpret a byte slice as `&[T]`, refusing a wrong length or misalignment.
pub fn cast<T: Copy>(bytes: &[u8]) -> Option<&[T]> {
    let size = core::mem::size_of::<T>();
    if size == 0 || bytes.len() % size != 0 {
        return None;
    }
    if bytes.as_ptr() as usize % core::mem::align_of::<T>() != 0 {
        return None;
    }
    // SAFETY: the pointer is aligned for `T` and the length is a whole number of
    // `T`; `T: Copy` and the reader's numeric use make the bytes a valid value.
    Some(unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / size) })
}

/// A node's outgoing edges: row indices in this file, as `u32` or `u64`.
pub struct NeighborList<'a> {
    bytes: &'a [u8],
    dtype: OffsetDType,
}

impl<'a> NeighborList<'a> {
    pub fn len(&self) -> usize {
        self.bytes.len() / self.dtype.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `i`th neighbor's row index, widened to `u64`.
    pub fn get(&self, i: usize) -> Option<u64> {
        let s = self.dtype.size();
        let slot = self.bytes.get(i * s..i * s + s)?;
        Some(match self.dtype {
            OffsetDType::U32 => u32::from_le_bytes(slot.try_into().ok()?) as u64,
            OffsetDType::U64 => u64::from_le_bytes(slot.try_into().ok()?),
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.len()).filter_map(move |i| self.get(i))
    }
}

impl<A: Allocator> SafeGraphsFile<A> {
    /// Node count. Validated equal across all three columns at open.
    pub fn rows(&self) -> usize {
        self.key.rows()
    }

    /// Raw bytes of a row's key.
    pub fn key_bytes(&self, row: usize) -> Option<&[u8]> {
        self.row_bytes(&self.key, row)
    }

    /// Raw bytes of a row's value, for the compute layer to reinterpret using
    /// its dtype name.
    pub fn value_bytes(&self, row: usize) -> Option<&[u8]> {
        self.row_bytes(&self.value, row)
    }

    /// The whole value buffer, contiguous, for a batch scan. Only meaningful for
    /// a regular value column; pair with [`Self::value_dim`].
    pub fn value_buffer(&self) -> &[u8] {
        self.buffer(self.value.values)
    }

    /// Fixed vector dimensionality of the value column, or `None` if irregular.
    pub fn value_dim(&self) -> Option<usize> {
        match self.value.layout {
            ColumnLayout::Regular { width } => Some(width),
            ColumnLayout::Irregular { .. } => None,
        }
    }

    /// A row's outgoing neighbors.
    pub fn neighbors(&self, row: usize) -> Option<NeighborList<'_>> {
        let bytes = self.row_bytes(&self.neighbors, row)?;
        Some(NeighborList {
            bytes,
            dtype: self.neighbors.dtype,
        })
    }

    /// The row whose key equals `key` byte-for-byte, by linear scan.
    pub fn find_key(&self, key: &[u8]) -> Option<usize> {
        (0..self.rows()).find(|&r| self.row_bytes(&self.key, r) == Some(key))
    }

    /// Slice a validated buffer out of the data segment. Relies on the open-time
    /// invariant that every stored `ByteRange` lies within the map, and degrades
    /// to an empty slice rather than panicking if that were ever violated.
    fn buffer(&self, r: ByteRange) -> &[u8] {
        let start = self.data_start.wrapping_add(r.start);
        let end = start.wrapping_add(r.len);
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

    /// The `[begin, end)` element indices of row `row` in an irregular column,
    /// read from its prefix-sum buffer and checked monotonic.
    fn row_span(&self, offsets: IrregularOffsets, row: usize) -> Option<(usize, usize)> {
        let s = offsets.dtype.size();
        let obytes = self.buffer(offsets.range);
        let read = |i: usize| -> Option<usize> {
            let slot = obytes.get(i.checked_mul(s)?..i.checked_mul(s)?.checked_add(s)?)?;
            match offsets.dtype {
                OffsetDType::U32 => Some(u32::from_le_bytes(slot.try_into().ok()?) as usize),
                OffsetDType::U64 => to_usize(u64::from_le_bytes(slot.try_into().ok()?)).ok(),
            }
        };
        let begin = read(row)?;
        let end = read(row.checked_add(1)?)?;
        if begin > end {
            None
        } else {
            Some((begin, end))
        }
    }

    /// Raw bytes of row `row` in any column, regular or irregular.
    fn row_bytes<E: ElementType>(&self, c: &Column<E>, row: usize) -> Option<&[u8]> {
        match c.layout {
            ColumnLayout::Regular { width } => {
                let per = width.checked_mul(c.dtype.size())?;
                self.subslice(c.values, row.checked_mul(per)?, per)
            }
            ColumnLayout::Irregular { offsets } => {
                let esz = c.dtype.size();
                let (begin, end) = self.row_span(offsets, row)?;
                let offset = begin.checked_mul(esz)?;
                let len = end.checked_sub(begin)?.checked_mul(esz)?;
                self.subslice(c.values, offset, len)
            }
        }
    }

    /// Check every buffer lies within the map and the columns agree on shape.
    fn validate(&self) -> Result<(), Error> {
        let rows = self.key.rows();
        self.check_layout(&self.key, rows)?;
        self.check_layout(&self.value, rows)?;
        self.check_layout(&self.neighbors, rows)?;
        if let Some(declared) = self.nodes() {
            if usize::try_from(declared).ok() != Some(rows) {
                return Err(Error::Inconsistent);
            }
        }
        Ok(())
    }

    /// Confirm one column's buffers fit the map and match `rows` exactly.
    fn check_layout<E: ElementType>(&self, c: &Column<E>, rows: usize) -> Result<(), Error> {
        match c.layout {
            ColumnLayout::Regular { width } => {
                let per = width.checked_mul(c.dtype.size()).ok_or(Error::TooLarge)?;
                if per.checked_mul(rows) != Some(c.values.len) {
                    return Err(Error::Inconsistent);
                }
            }
            ColumnLayout::Irregular { offsets } => {
                self.check_buffer(offsets.range)?;
                let expected = rows
                    .checked_add(1)
                    .and_then(|r| r.checked_mul(offsets.dtype.size()))
                    .ok_or(Error::TooLarge)?;
                if offsets.range.len != expected {
                    return Err(Error::Inconsistent);
                }
            }
        }
        self.check_buffer(c.values)
    }

    /// Confirm a buffer's end falls within the mapped file.
    fn check_buffer(&self, r: ByteRange) -> Result<(), Error> {
        let end = self
            .data_start
            .checked_add(r.start)
            .and_then(|s| s.checked_add(r.len))
            .ok_or(Error::TooLarge)?;
        if end <= self.bytes().len() {
            Ok(())
        } else {
            Err(Error::Truncated)
        }
    }
}

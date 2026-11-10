//! The `.safegraphs` on-disk format: reader, columns, and materializer.
//!
//! A `.safegraphs` file is an 8-byte magic, a little-endian `u64` header length,
//! a JSON header that is the complete schema, and a data segment of raw,
//! Arrow-layout buffers.
//!
//! The header names three columns. `key` is the caller's identity, global across
//! shards and free to repeat; `value` is the payload; `neighbors` holds the
//! outgoing edges as node indices into this same file. Each column locates its
//! buffers with `[begin, end]` byte pairs relative to the data segment, `data`
//! for the rows and `offsets` for a ragged column's row starts.
//!
//! Two words divide the vocabulary. A __row__ is one column's slot, the unit a
//! buffer addresses, so [`StorageDType::row_bytes`] measures one and
//! [`Column::rows`] counts them. A __node__ is the tuple of rows sharing a
//! position across every column, so [`GraphLike::len`] counts nodes and
//! [`GraphLike`] yields them. They are equal because the `key` column sets the
//! node count and every other column must agree.
//!
//! Reading is zero-copy. Buffers are reinterpreted as typed slices straight out
//! of the memory map; nothing is copied. Allocation happens only for result sets
//! and search state, through fallible APIs, so the reader never panics on memory
//! pressure.
//!
//! ```no_run
//! use safegraphs::{GraphLike, GraphTable};
//!
//! let table = GraphTable::open("index.safegraphs")?;
//! let node = table.find_key(b"wiki-1m").unwrap_or(0);
//! let _payload: &[u8] = table.value(node);
//! let edges = table.neighbors(node);
//! if let Some(neighbors) = edges.as_slice::<u32>() {
//!     for neighbor in neighbors {
//!         // An index at or past the node count is a sentinel and names no node.
//!         if (*neighbor as usize) < table.len() {
//!             let _their_payload = table.value(*neighbor as usize);
//!         }
//!     }
//! }
//! # Ok::<(), safegraphs::Error>(())
//! ```

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use allocator_api2::alloc::{Allocator as AllocatorLike, Global};
use allocator_api2::vec::Vec as AVec;

use jsontape::{parse_in, Json, JsonError};

/// The 8-byte magic at offset 0 of every `.safegraphs` file.
pub const MAGIC: [u8; 8] = *b"SAFEGRPH";

/// A byte range in the data segment, parsed from a header's `[begin, end]` pair
/// and held as the start and byte count a slice wants.
#[derive(Clone, Copy)]
pub struct ByteRange {
    pub start: usize,
    pub len: usize,
}

/// Storage width of a payload element, in bits. Storage is decoupled from
/// compute, so a `StorageDType` only says how wide an element is, never how to
/// interpret it. Every width has a neutral `bN` spelling naming N opaque bits,
/// and the other spellings are aliases that add a reading to the same width: a
/// 2-byte `Bin16` is equally a `b16`, an `f16`, a `bf16`, or a `u16`; the
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
    /// IDs. A sub-byte value column never reaches here; it measures rows through
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

    /// [`row_bytes`](StorageDType::row_bytes) for a width that has not yet been
    /// validated — `None` when `width * bits()` would overflow. Header parsing
    /// goes through this; the per-node hot path keeps the infallible form, since
    /// by then `check_layout` has proven the width against the real file size.
    pub const fn checked_row_bytes(self, width: usize) -> Option<usize> {
        match width.checked_mul(self.bits()) {
            Some(bits) => Some(bits.div_ceil(8)),
            None => None,
        }
    }
}

/// Width of a node index or prefix-sum entry. Unsigned, since offsets and node
/// IDs are non-negative and no column is nullable.
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
/// [`OffsetDType`]; a searcher resolves the matching `IndexLike` from it and
/// traverses in that type, so the frontier and results stay as compact as the
/// graph — a `u32` node paired with an `f32` score is eight bytes. `Into<u64>`
/// widens a result only when it leaves the walk; the trait exists to seal the two
/// legal scalars and to name which [`OffsetDType`] each one is.
pub trait IndexLike: Copy + TryFrom<usize> + TryInto<usize> + Into<u64> + 'static {
    const OFFSET_DTYPE: OffsetDType;
}

impl IndexLike for u32 {
    const OFFSET_DTYPE: OffsetDType = OffsetDType::U32;
}

impl IndexLike for u64 {
    const OFFSET_DTYPE: OffsetDType = OffsetDType::U64;
}

/// The score-side twin of [`IndexLike`]: a native metric accumulator widened to
/// the one score type a runtime caller can name. Where a walk leaves the graph it
/// widens its node through [`IndexLike`] and its score through this, so the
/// erased boundary is `(u64, f64)` and nothing wider. Every native accumulator —
/// `i32`, `u32`, `f32`, `f64` — fits exactly in `f64`'s mantissa, so this loses
/// nothing today; a metric scoring past 2^53 is the only thing it could.
pub trait ScoreLike: Copy {
    fn widen(self) -> f64;
}

impl ScoreLike for i32 {
    fn widen(self) -> f64 {
        self as f64
    }
}

impl ScoreLike for u32 {
    fn widen(self) -> f64 {
        self as f64
    }
}

impl ScoreLike for f32 {
    fn widen(self) -> f64 {
        self as f64
    }
}

impl ScoreLike for f64 {
    fn widen(self) -> f64 {
        self
    }
}

/// An edit distance counts operations, so it is bounded by the longer input's
/// length and cannot approach the 2^53 this trait's contract worries about — a
/// 9-petabyte string would have to fit in the value column first.
impl ScoreLike for usize {
    fn widen(self) -> f64 {
        self as f64
    }
}

/// The offsets buffer of an irregular column, holding its prefix sums.
#[derive(Clone, Copy)]
pub struct IrregularOffsets {
    pub dtype: OffsetDType,
    pub range: ByteRange,
}

/// How a column's rows sit in its data buffer. Regular or irregular, never
/// both. This is the __byte layout__; the __logical shape__ a metric reads is
/// [`Shape`], carried separately so a regular `points × 3` cloud and a ragged
/// `[*, dim]` multi-vector each state their axes without changing addressing.
#[derive(Clone, Copy)]
pub enum ColumnLayout {
    /// Regular rows of `width` elements. `width` is 1 for a scalar column.
    Regular { width: usize },
    /// Irregular rows addressed by a prefix-sum offsets buffer.
    Irregular { offsets: IrregularOffsets },
}

/// The logical shape of one value row, threaded next to the dtype so every
/// metric reads its shape the way it reads its scalar type. An axis of `0` marks
/// the dimension that __resizes per node__ (ragged); every other axis is fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// A single element per row. Byte layout is `Regular { width: 1 }`.
    Scalar,
    /// `cols` elements per row — the dense vector or the geospatial pair.
    /// Byte layout is `Regular { width: cols }`.
    Vector(usize),
    /// `rows × cols` per node. `rows == 0` is the ragged case: the row count
    /// varies per node (maxsim `[*, dim]`), so the byte layout is `Irregular`
    /// and the reader recovers the count from the row's byte extent ÷ `cols`.
    /// A fixed `rows > 0` (a cloud `[points, 3]`) is `Regular { width: rows*cols }`.
    Matrix(usize, usize),
}

impl Shape {
    /// The flat element count of a fixed row, or `None` when an axis is ragged.
    /// This is what a regular byte layout needs; a ragged shape has no fixed
    /// row and returns `None`, which the builder reads as "irregular".
    pub const fn regular_width(self) -> Option<usize> {
        match self {
            Shape::Scalar => Some(1),
            Shape::Vector(cols) => Some(cols),
            Shape::Matrix(0, _) => None,
            // A corrupt header can name a `Matrix` no real file could back, so
            // the product is checked: `None` here becomes `Error::Inconsistent`
            // at parse time rather than a wrapped width that silently
            // mis-describes the column — or, in a debug build, a panic while
            // merely opening an untrusted file.
            Shape::Matrix(rows, cols) => rows.checked_mul(cols),
        }
    }

    /// The flat width for the vector-shaped families (dense, binary, geospatial),
    /// or `None` for anything with a second axis. These metrics never see a
    /// `Matrix` column, so a `Matrix` here is a mismatch they reject.
    pub const fn width(self) -> Option<usize> {
        match self {
            Shape::Scalar => Some(1),
            Shape::Vector(cols) => Some(cols),
            Shape::Matrix(..) => None,
        }
    }

    /// The inner (fixed) axis of a `Matrix` — `3` for a point cloud, `dim` for a
    /// maxsim multi-vector. `None` for the vector-shaped families.
    pub const fn inner(self) -> Option<usize> {
        match self {
            Shape::Matrix(_, cols) => Some(cols),
            _ => None,
        }
    }
}

/// One column: an element type, a layout, and its data buffer. `E` is the
/// element type: `StorageDType` for payload columns whose bytes the reader does
/// not interpret, `OffsetDType` for the neighbor graph whose values are node
/// indices. The dtype string lives in the header document, reachable through
/// [`GraphTable::dtype_name`].
#[derive(Clone, Copy)]
pub struct Column<Element> {
    /// Storage width of one element, resolved from the header's dtype string.
    pub dtype: Element,
    /// How the rows sit in `data`: a fixed width, or addressed by offsets.
    pub layout: ColumnLayout,
    /// The logical shape a metric reads, orthogonal to the byte `layout`.
    pub shape: Shape,
    /// The rows themselves, as a range of the data segment.
    pub data: ByteRange,
}

/// The bytes behind a file: a memory map, or an owned in-memory buffer. Both are
/// just memory from an allocator — the OS mmap allocator, or `A`.
enum Backing<Allocator: AllocatorLike = Global> {
    Mapped(Mmap),
    Owned(AVec<u8, Allocator>),
}

impl<Allocator: AllocatorLike> Backing<Allocator> {
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
pub struct GraphTable<Allocator: AllocatorLike = Global> {
    /// The bytes behind every buffer, kept alive for the whole handle so borrowed
    /// slices stay valid.
    backing: Backing<Allocator>,

    /// Byte offset of the data segment: the first byte past the padded header.
    /// Every column's `ByteRange` is relative to this.
    segment_start: usize,

    /// The parsed header document, owned in `A`: the free-form `__metadata__`
    /// plus each column's descriptor and dtype string.
    header: Json<Allocator>,

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
    /// A metric name is not one the router recognizes.
    UnknownMetric,
    /// A registered metric name this reader has no kernel for.
    UnsupportedMetric,
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

/// A short, caller-facing reason.
/// Every front end needs this — without it each one hand-rolls the same match —
/// and it is what lets `?` compose with `Box<dyn Error>` and `anyhow` alike.
/// Context that would need a position or a path stays with the caller, which is
/// the one that knows it.
impl core::fmt::Display for Error {
    fn fmt(&self, out: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Io(inner) => write!(out, "{inner}"),
            Error::BadMagic => out.write_str("not a .safegraphs file"),
            Error::Truncated => out.write_str("file ends inside a buffer it declares"),
            Error::Json(inner) => write!(out, "header is not valid JSON: {inner:?}"),
            Error::MissingField(field) => write!(out, "header is missing {field}"),
            Error::UnknownMetric => out.write_str("no metric family answers to that name"),
            Error::UnsupportedMetric => {
                out.write_str("the metric is registered, but this reader cannot score it")
            }
            Error::UnknownDType => out.write_str("no storage width answers to that dtype"),
            Error::WrongType => out.write_str("the column's dtype or shape does not suit this use"),
            Error::TooLarge => out.write_str("a declared size does not fit this platform"),
            Error::Inconsistent => out.write_str("the header and its buffers disagree"),
            Error::Alloc => out.write_str("allocation failed"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(inner) => Some(inner),
            _ => None,
        }
    }
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
pub trait ElementLike: Copy {
    fn from_dtype(text: &[u8]) -> Result<Self, Error>;
    fn size(self) -> usize;
    /// Byte length of a regular row of `width` elements. Whole-byte by default —
    /// only [`StorageDType`] overrides it to pack sub-byte elements tight.
    ///
    /// Call this only once the width has been proven against a real buffer;
    /// header parsing goes through [`checked_row_bytes`](ElementLike::checked_row_bytes).
    fn row_bytes(self, width: usize) -> usize {
        width * self.size()
    }

    /// [`row_bytes`](ElementLike::row_bytes) for a width straight out of a
    /// header, before anything has vetted it — `None` when the product would
    /// overflow, so a corrupt file yields an error instead of a wrapped length.
    fn checked_row_bytes(self, width: usize) -> Option<usize> {
        width.checked_mul(self.size())
    }
}

impl ElementLike for StorageDType {
    fn from_dtype(text: &[u8]) -> Result<Self, Error> {
        Ok(match text {
            b"u1x8" => StorageDType::Bin1,
            b"i4x2" | b"u4x2" => StorageDType::Bin4,
            b"b8" | b"u8" | b"i8" | b"e4m3" | b"e5m2" | b"e2m3" | b"e3m2" | b"utf8" => {
                StorageDType::Bin8
            }
            b"b16" | b"u16" | b"i16" | b"f16" | b"bf16" => StorageDType::Bin16,
            b"b32" | b"u32" | b"i32" | b"f32" => StorageDType::Bin32,
            b"b64" | b"u64" | b"i64" | b"f64" => StorageDType::Bin64,
            b"b128" | b"uuid" => StorageDType::Bin128,
            _ => return Err(Error::UnknownDType),
        })
    }

    fn size(self) -> usize {
        StorageDType::size(self)
    }

    fn row_bytes(self, width: usize) -> usize {
        StorageDType::row_bytes(self, width)
    }

    fn checked_row_bytes(self, width: usize) -> Option<usize> {
        StorageDType::checked_row_bytes(self, width)
    }
}

impl ElementLike for OffsetDType {
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

/// A buffer's `[begin, end]` pair, as a range of `end - begin` bytes at `begin`.
/// A pair of byte positions can only mean positions, where a `_start` beside a
/// `_length` left the unit of each to the reader.
fn span_field<Allocator: AllocatorLike>(
    obj: &Json<Allocator>,
    key: &'static str,
) -> Result<ByteRange, Error> {
    let pair = obj.get(key).ok_or(Error::MissingField(key))?;
    let pair = pair.as_array().ok_or(Error::WrongType)?;
    if pair.len() != 2 {
        return Err(Error::WrongType);
    }
    let begin = to_usize(pair[0].as_u64().ok_or(Error::WrongType)?)?;
    let end = to_usize(pair[1].as_u64().ok_or(Error::WrongType)?)?;
    Ok(ByteRange {
        start: begin,
        len: end.checked_sub(begin).ok_or(Error::Inconsistent)?,
    })
}

/// The buffer alignment `__metadata__` declares: a power of two, 4096 when
/// absent, and never below 64. Validated before it is floored, so a small
/// non-power of two is an error rather than a silent 64.
fn declared_alignment<Allocator: AllocatorLike>(
    metadata: Option<&Json<Allocator>>,
) -> Result<usize, Error> {
    let Some(value) = metadata.and_then(|object| object.get("alignment")) else {
        return Ok(4096);
    };
    let value = to_usize(value.as_u64().ok_or(Error::WrongType)?)?;
    if !value.is_power_of_two() {
        return Err(Error::Inconsistent);
    }
    Ok(value.max(64))
}

/// Parse a column's `"shape"`: absent is [`Shape::Scalar`], a bare number `w` is
/// [`Shape::Vector`]`(w)`, a two-element array `[rows, cols]` is
/// [`Shape::Matrix`] (a leading `0` marking the ragged axis). A one-element
/// array `[w]` reads as `Vector(w)`, so a producer may always emit an array.
fn shape_from_json<Allocator: AllocatorLike>(obj: &Json<Allocator>) -> Result<Shape, Error> {
    let Some(value) = obj.get("shape") else {
        return Ok(Shape::Scalar);
    };
    if let Some(scalar) = value.as_u64() {
        return Ok(Shape::Vector(to_usize(scalar)?));
    }
    let axes = value.as_array().ok_or(Error::WrongType)?;
    match axes.len() {
        1 => Ok(Shape::Vector(to_usize(
            axes[0].as_u64().ok_or(Error::WrongType)?,
        )?)),
        2 => Ok(Shape::Matrix(
            to_usize(axes[0].as_u64().ok_or(Error::WrongType)?)?,
            to_usize(axes[1].as_u64().ok_or(Error::WrongType)?)?,
        )),
        _ => Err(Error::Inconsistent),
    }
}

impl<Element: ElementLike> Column<Element> {
    fn from_json<Allocator: AllocatorLike>(obj: &Json<Allocator>) -> Result<Self, Error> {
        let dtype = Element::from_dtype(
            obj.get("dtype")
                .ok_or(Error::MissingField("dtype"))?
                .as_str()
                .ok_or(Error::WrongType)?
                .as_bytes(),
        )?;

        let data = span_field(obj, "data")?;

        // The logical shape: absent -> Scalar; a bare number -> Vector; a
        // two-element array `[rows, cols]` -> Matrix (rows 0 marks the ragged
        // axis). Threaded next to the dtype for every metric to read.
        let shape = shape_from_json(obj)?;

        let layout = match obj.get("offsets") {
            Some(_) => {
                // Offset-addressed rows may be shapeless (`Scalar` — a tape of
                // variable-length strings) or ragged (`Matrix(0, cols)` — rows of
                // fixed-width tuples, a varying number of them). What they may
                // never carry is a __fixed__ shape: `Vector(n)` or
                // `Matrix(rows > 0, _)` claims a constant row width this layout
                // contradicts, and a metric trusting the shape alone would
                // mis-stride the buffer — `geospatial` reads `value_buffer()` at
                // two-scalar strides on exactly that strength.
                if !matches!(shape, Shape::Scalar | Shape::Matrix(0, _)) {
                    return Err(Error::Inconsistent);
                }
                let offset_dtype = obj
                    .get("offsets_dtype")
                    .ok_or(Error::MissingField("offsets_dtype"))?
                    .as_str()
                    .ok_or(Error::WrongType)?;
                ColumnLayout::Irregular {
                    offsets: IrregularOffsets {
                        dtype: OffsetDType::from_dtype(offset_dtype.as_bytes())?,
                        range: span_field(obj, "offsets")?,
                    },
                }
            }
            // A regular byte layout needs a fixed row width; a ragged shape over
            // a column that declared no offsets is a malformed header.
            None => ColumnLayout::Regular {
                width: shape.regular_width().ok_or(Error::Inconsistent)?,
            },
        };

        Ok(Column {
            dtype,
            layout,
            shape,
            data,
        })
    }
}

fn column_field<Element: ElementLike, Allocator: AllocatorLike>(
    doc: &Json<Allocator>,
    name: &'static str,
) -> Result<Column<Element>, Error> {
    let obj = doc.get(name).ok_or(Error::MissingField(name))?;
    Column::<Element>::from_json(obj)
}

/// One past the last byte any buffer of `column` touches, relative to the data segment.
fn column_end<Element>(column: &Column<Element>) -> Result<usize, Error> {
    let data_end = column
        .data
        .start
        .checked_add(column.data.len)
        .ok_or(Error::TooLarge)?;
    match column.layout {
        ColumnLayout::Regular { .. } => Ok(data_end),
        ColumnLayout::Irregular { offsets } => {
            let offsets_end = offsets
                .range
                .start
                .checked_add(offsets.range.len)
                .ok_or(Error::TooLarge)?;
            Ok(data_end.max(offsets_end))
        }
    }
}

/// The parsed header of a SafeGraphs file, read from its first bytes alone: enough to name the
/// graph and its extent without touching the data segment, as a probe of a raw drive slot needs.
pub struct GraphHeader<Allocator: AllocatorLike = Global> {
    /// Byte offset of the data segment: the first byte past the padded header.
    pub segment_start: usize,

    /// The parsed header document, owned in `A`.
    pub header: Json<Allocator>,

    pub key: Column<StorageDType>,
    pub value: Column<StorageDType>,
    pub neighbors: Column<OffsetDType>,

    /// Bytes of the data segment up to the furthest buffer of any column, producer columns included.
    pub data_bytes: usize,
}

impl GraphHeader {
    /// Where the data segment starts, from the magic and header length in the first 16 bytes, so a
    /// reader holding only a first page knows how much more to read before [`GraphHeader::read_in`].
    pub fn segment_start_of(prefix: &[u8]) -> Result<usize, Error> {
        if prefix.get(..8) != Some(&MAGIC[..]) {
            return Err(Error::BadMagic);
        }
        let mut length_bytes = [0u8; 8];
        length_bytes.copy_from_slice(prefix.get(8..16).ok_or(Error::Truncated)?);
        let header_len = to_usize(u64::from_le_bytes(length_bytes))?;
        16usize.checked_add(header_len).ok_or(Error::Truncated)
    }
}

impl<Allocator: AllocatorLike + Clone> GraphHeader<Allocator> {
    /// Parses the header from `prefix`, which must reach the segment start; the data segment is
    /// never read, so nothing here proves the buffers exist.
    pub fn read_in(prefix: &[u8], alloc: Allocator) -> Result<Self, Error> {
        let segment_start = GraphHeader::segment_start_of(prefix)?;
        let header = parse_in(
            prefix.get(16..segment_start).ok_or(Error::Truncated)?,
            alloc,
        )?;
        let key = column_field::<StorageDType, Allocator>(&header, "key")?;
        let value = column_field::<StorageDType, Allocator>(&header, "value")?;
        let neighbors = column_field::<OffsetDType, Allocator>(&header, "neighbors")?;

        // Producer columns, such as cached norms, are any other object beside `__metadata__`;
        // this reader keeps none of them, but each still bounds the file.
        let mut data_bytes = column_end(&key)?
            .max(column_end(&value)?)
            .max(column_end(&neighbors)?);
        for (name, object) in header.as_object().ok_or(Error::WrongType)? {
            let name: &str = name;
            if object.as_object().is_none()
                || matches!(name, "__metadata__" | "key" | "value" | "neighbors")
            {
                continue;
            }
            data_bytes = data_bytes.max(column_end(&Column::<StorageDType>::from_json(object)?)?);
        }
        segment_start
            .checked_add(data_bytes)
            .ok_or(Error::TooLarge)?;

        Ok(GraphHeader {
            segment_start,
            header,
            key,
            value,
            neighbors,
            data_bytes,
        })
    }

    /// The whole file's length: the segment start plus the furthest column buffer.
    pub fn file_bytes(&self) -> usize {
        self.segment_start + self.data_bytes
    }
}

impl GraphTable<Global> {
    /// Memory-map a file read-only; pages fault in lazily. Metadata allocates in
    /// the global heap.
    pub fn open<PathRef: AsRef<Path>>(path: PathRef) -> Result<Self, Error> {
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
    pub fn build<Graph: GraphLike>(
        graph: &Graph,
        metadata: Option<&Json<Global>>,
    ) -> Result<Self, Error> {
        Self::build_in(graph, metadata, Global)
    }
}

impl<Allocator: AllocatorLike + Clone> GraphTable<Allocator> {
    /// Memory-map a file read-only, allocating the metadata document in `alloc`.
    pub fn open_in<PathRef: AsRef<Path>>(path: PathRef, alloc: Allocator) -> Result<Self, Error> {
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only; the file must not be mutated
        // externally for the lifetime of this handle, per mmap's contract.
        let map = unsafe { Mmap::map(&file)? };
        Self::from_backing(Backing::Mapped(map), alloc)
    }

    /// Take ownership of an in-memory blob, allocating metadata in `alloc`.
    pub fn from_owned(buffer: AVec<u8, Allocator>, alloc: Allocator) -> Result<Self, Error> {
        Self::from_backing(Backing::Owned(buffer), alloc)
    }

    /// Materialize a graph into the columnar form, allocating in `alloc`. The
    /// metadata object may come from any allocator `M`; it is serialized into the
    /// header.
    pub fn build_in<Graph: GraphLike, MetadataAllocator: AllocatorLike>(
        graph: &Graph,
        metadata: Option<&Json<MetadataAllocator>>,
        alloc: Allocator,
    ) -> Result<Self, Error> {
        let blob = materialize(graph, metadata, alloc.clone())?;
        Self::from_backing(Backing::Owned(blob), alloc)
    }

    /// Shared parse over any backing: read the header, then validate every
    /// column against the whole file.
    fn from_backing(backing: Backing<Allocator>, alloc: Allocator) -> Result<Self, Error> {
        let GraphHeader {
            segment_start,
            header,
            key,
            value,
            neighbors,
            ..
        } = GraphHeader::read_in(backing.as_bytes(), alloc)?;

        let file = GraphTable {
            backing,
            segment_start,
            header,
            key,
            value,
            neighbors,
        };
        file.validate()?;
        Ok(file)
    }
}

/// The query-time read surface of a [`GraphTable`], detached from its owner: the
/// mapped bytes plus the three `Copy` column descriptors and the header's dtype
/// strings, pre-extracted. `Copy + Send + Sync` by construction — a parallel
/// fold's tasks capture it directly, and the allocator behind the owning table
/// never appears — so the search layer speaks this type and stays free of both
/// the allocator generic and any unsafe shareability assertion.
#[derive(Clone, Copy)]
pub struct GraphTableView<'t> {
    /// The complete file image; every [`ByteRange`] is relative to `segment_start`.
    bytes: &'t [u8],
    /// Byte offset of the data segment, the first byte past the padded header.
    segment_start: usize,
    /// Node identity, one row per node.
    key: Column<StorageDType>,
    /// Node payload, one row per node.
    value: Column<StorageDType>,
    /// Outgoing edges, as node indices into this same file.
    neighbors: Column<OffsetDType>,
    /// The key column's dtype string, which the `Copy` descriptor cannot recover —
    /// `f16`, `bf16`, and `u16` all collapse to one storage width.
    key_dtype: &'t str,
    /// The value column's dtype string, for the same reason.
    value_dtype: &'t str,
}

impl<'t> GraphTableView<'t> {
    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.key.rows()
    }

    /// True when there are no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The key column's dtype string, as the header spells it.
    pub fn key_dtype(&self) -> &'t str {
        self.key_dtype
    }

    /// The value column's dtype string, as the header spells it.
    pub fn value_dtype(&self) -> &'t str {
        self.value_dtype
    }

    /// The key column's logical [`Shape`], `None` for a shapeless ragged column.
    pub fn key_shape(&self) -> Option<Shape> {
        declared_shape(&self.key)
    }

    /// The value column's logical [`Shape`], `None` for a shapeless ragged column.
    pub fn value_shape(&self) -> Option<Shape> {
        declared_shape(&self.value)
    }

    /// The value column's flat element width, for the rank-1 families.
    pub fn value_width(&self) -> Option<usize> {
        self.value_shape().and_then(Shape::width)
    }

    /// The neighbors column's fixed out-degree, or `None` if variable.
    pub fn neighbors_degree(&self) -> Option<usize> {
        match self.neighbors.layout {
            ColumnLayout::Regular { width } => Some(width),
            ColumnLayout::Irregular { .. } => None,
        }
    }

    /// The neighbor index width, for a width resolver settling its lane.
    pub fn neighbors_dtype(&self) -> OffsetDType {
        self.neighbors.dtype
    }

    /// The mean out-degree, the walk's landmark budget.
    pub fn mean_degree(&self) -> usize {
        let nodes = self.len();
        if nodes == 0 {
            return 0;
        }
        let entries = self.neighbors.data.len / self.neighbors.dtype.size();
        entries / nodes
    }

    /// Raw bytes of node `node`'s key.
    pub fn key(&self, node: usize) -> &'t [u8] {
        self.node_bytes(&self.key, node).unwrap_or(&[])
    }

    /// Raw bytes of node `node`'s value.
    pub fn value(&self, node: usize) -> &'t [u8] {
        self.node_bytes(&self.value, node).unwrap_or(&[])
    }

    /// Node `node`'s outgoing edges.
    pub fn neighbors(&self, node: usize) -> NeighborList<'t> {
        NeighborList::new(
            self.node_bytes(&self.neighbors, node).unwrap_or(&[]),
            self.neighbors.dtype,
        )
    }

    /// The whole value buffer, contiguous, for a batch scan over a regular
    /// column.
    pub fn value_buffer(&self) -> &'t [u8] {
        self.buffer(self.value.data)
    }

    /// The value column's offsets buffer and its width, or `None` when the
    /// column is regular and has no offsets at all.
    pub fn value_offsets(&self) -> Option<(&'t [u8], OffsetDType)> {
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
    fn buffer(&self, range: ByteRange) -> &'t [u8] {
        let start = self.segment_start.wrapping_add(range.start);
        let end = start.wrapping_add(range.len);
        self.bytes.get(start..end).unwrap_or(&[])
    }

    /// Slice `len` bytes at `offset` within a column's data buffer, checking the
    /// sub-range stays inside it.
    fn subslice(&self, data: ByteRange, offset: usize, len: usize) -> Option<&'t [u8]> {
        if offset.checked_add(len)? > data.len {
            return None;
        }
        Some(self.buffer(ByteRange {
            start: data.start.checked_add(offset)?,
            len,
        }))
    }

    /// The `[begin, end)` element indices of node `node` in an irregular column,
    /// read from its offsets buffer and checked monotonic.
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
    fn node_bytes<Element: ElementLike>(
        &self,
        column: &Column<Element>,
        node: usize,
    ) -> Option<&'t [u8]> {
        match column.layout {
            ColumnLayout::Regular { width } => {
                // Safe to use the infallible form: a `GraphTableView` only exists
                // after `check_layout` proved this width against the buffer.
                let per = column.dtype.row_bytes(width);
                self.subslice(column.data, node.checked_mul(per)?, per)
            }
            ColumnLayout::Irregular { offsets } => {
                let element_size = column.dtype.size();
                let (begin, end) = self.node_span(offsets, node)?;
                let offset = begin.checked_mul(element_size)?;
                let len = end.checked_sub(begin)?.checked_mul(element_size)?;
                self.subslice(column.data, offset, len)
            }
        }
    }
}

/// A view is a [`GraphLike`] too, so anything a producer's graph can do — a
/// rebuild through [`GraphTable::build`] included — a borrowed view can.
impl GraphLike for GraphTableView<'_> {
    fn len(&self) -> usize {
        GraphTableView::len(self)
    }
    fn key_dtype(&self) -> &str {
        self.key_dtype
    }
    fn value_dtype(&self) -> &str {
        self.value_dtype
    }
    fn key_shape(&self) -> Option<Shape> {
        GraphTableView::key_shape(self)
    }
    fn value_shape(&self) -> Option<Shape> {
        GraphTableView::value_shape(self)
    }
    fn neighbors_degree(&self) -> Option<usize> {
        GraphTableView::neighbors_degree(self)
    }
    fn key(&self, node: usize) -> &[u8] {
        GraphTableView::key(self, node)
    }
    fn value(&self, node: usize) -> &[u8] {
        GraphTableView::value(self, node)
    }
    fn neighbors(&self, node: usize) -> NeighborList<'_> {
        GraphTableView::neighbors(self, node)
    }
}

impl<Allocator: AllocatorLike> GraphTable<Allocator> {
    /// The bytes behind the file, mapped or in-memory.
    fn bytes(&self) -> &[u8] {
        self.backing.as_bytes()
    }

    /// The complete `.safegraphs` byte image — magic, header, and data segment —
    /// ready to write to a file or hand to [`from_bytes`](GraphTable::from_bytes). For
    /// a built index this is the serialization; for a mapped one it is the mapping
    /// itself, so the borrow is always zero-copy.
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes()
    }

    /// The free-form `__metadata__` object, if present.
    pub fn metadata(&self) -> Option<&Json<Allocator>> {
        self.header.get("__metadata__")
    }

    pub fn metadata_get(&self, key: &str) -> Option<&Json<Allocator>> {
        self.metadata()?.get(key)
    }

    /// Declared buffer alignment: a power of two, 4096 when absent, 64 at least.
    /// Opening the file proved it, so this cannot fail here.
    pub fn alignment(&self) -> usize {
        declared_alignment(self.metadata()).unwrap_or(4096)
    }

    /// The breadth the graph was __built__ with, `ef_construction` by its usual
    /// name. A walk's own breadth is a caller's argument and appears in no header.
    pub fn expansion(&self) -> Option<u64> {
        self.metadata_get("expansion").and_then(|v| v.as_u64())
    }

    pub fn metric(&self) -> Option<&str> {
        self.metadata_get("metric").and_then(|v| v.as_str())
    }

    /// The original dtype string of a column, "key" / "value" / "neighbors", for
    /// the compute layer to reinterpret its opaque data buffer.
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

    /// Borrow the query-time read surface as a [`GraphTableView`] — `Copy`, `Send`,
    /// and `Sync` regardless of this table's allocator, so parallel code takes
    /// the view and the owner stays behind.
    pub fn view(&self) -> GraphTableView<'_> {
        GraphTableView {
            key_dtype: self.dtype_name("key").unwrap_or(""),
            value_dtype: self.dtype_name("value").unwrap_or(""),
            ..self.columns_view()
        }
    }

    /// The data-only view the table's own accessors delegate through — no
    /// header lookups, so per-node delegation costs nothing. It carries no dtype
    /// strings, so `key_dtype` and `value_dtype` go to [`dtype_name`] instead.
    ///
    /// [`dtype_name`]: GraphTable::dtype_name
    fn columns_view(&self) -> GraphTableView<'_> {
        GraphTableView {
            bytes: self.bytes(),
            segment_start: self.segment_start,
            key: self.key,
            value: self.value,
            neighbors: self.neighbors,
            key_dtype: "",
            value_dtype: "",
        }
    }

    /// The mean out-degree: total neighbor entries divided by node count. For a
    /// fixed-degree graph this is exactly the degree; for a variable-degree one it
    /// is the average. A walk seeds about this many random landmarks, the way a
    /// graph index seeds about `connectivity` — no magic constant, since the
    /// figure is read from the graph itself.
    pub fn mean_degree(&self) -> usize {
        self.columns_view().mean_degree()
    }
}

impl<Element: ElementLike> Column<Element> {
    /// Number of rows the column holds, derived from its own buffers. Equal to
    /// the file's node count, which every column must agree on, but named for
    /// the column's own axis.
    pub fn rows(&self) -> usize {
        match self.layout {
            ColumnLayout::Regular { width } => {
                // Derived from the header before `check_layout` has vetted the
                // width, so the row length is computed checked: an overflowing
                // width describes no rows at all, which the layout check then
                // rejects.
                match self.dtype.checked_row_bytes(width) {
                    Some(per) => self.data.len.checked_div(per).unwrap_or(0),
                    None => 0,
                }
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
        self.rows() == 0
    }
}

/// A column's shape as a producer would declare it: `None` when the rows are
/// ragged and carry no fixed axis at all, so `None` means the same thing on both
/// sides of [`GraphLike`] and a rebuild picks the layout the file already has.
fn declared_shape<Element: ElementLike>(column: &Column<Element>) -> Option<Shape> {
    match (column.layout, column.shape) {
        (ColumnLayout::Irregular { .. }, Shape::Scalar) => None,
        _ => Some(column.shape),
    }
}

/// Reinterpret a byte slice as `&[T]`, refusing a wrong length or misalignment.
pub fn cast<Value: Copy>(bytes: &[u8]) -> Option<&[Value]> {
    let size = core::mem::size_of::<Value>();
    if size == 0 || !bytes.len().is_multiple_of(size) {
        return None;
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<Value>()) {
        return None;
    }
    // SAFETY: the pointer is aligned for `T` and the length is a whole number of
    // `T`; `T: Copy` and the reader's numeric use make the bytes a valid value.
    Some(unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const Value, bytes.len() / size) })
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
    /// Wrap raw neighbor bytes tagged with their index width. Public because
    /// [`GraphLike::neighbors`] returns one: a producer outside this crate has to
    /// build the list it yields, so a private constructor would leave the trait
    /// unimplementable.
    pub fn new(bytes: &'a [u8], dtype: OffsetDType) -> Self {
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
    pub fn as_slice<Index: IndexLike>(&self) -> Option<&'a [Index]> {
        if self.dtype != Index::OFFSET_DTYPE {
            return None;
        }
        cast(self.bytes)
    }
}

impl<Allocator: AllocatorLike> GraphTable<Allocator> {
    /// The whole value buffer, contiguous, for a batch scan. Only meaningful for
    /// a regular value column; pair with [`GraphLike::value_shape`].
    pub fn value_buffer(&self) -> &[u8] {
        self.columns_view().value_buffer()
    }

    /// The value column's offsets buffer and its width, or `None` when the
    /// column is regular and has no offsets at all.
    ///
    /// Exposed because a variable-length column is exactly an Arrow-style tape —
    /// prefix sums plus concatenated bytes — and a sequence metric can hand that
    /// pair to its backend with no copy, provided it can reach both halves.
    pub fn value_offsets(&self) -> Option<(&[u8], OffsetDType)> {
        self.columns_view().value_offsets()
    }

    /// The node whose key equals `key` byte-for-byte, by linear scan.
    pub fn find_key(&self, key: &[u8]) -> Option<usize> {
        self.columns_view().find_key(key)
    }

    /// Check every buffer lies within the map on its declared alignment, and
    /// that the columns agree on node count.
    fn validate(&self) -> Result<(), Error> {
        let alignment = declared_alignment(self.metadata())?;
        let len = self.key.rows();
        self.check_layout(&self.key, len, alignment)?;
        self.check_layout(&self.value, len, alignment)?;
        self.check_layout(&self.neighbors, len, alignment)
    }

    /// Confirm one column's buffers fit the map and match `len` exactly.
    fn check_layout<Element: ElementLike>(
        &self,
        column: &Column<Element>,
        len: usize,
        alignment: usize,
    ) -> Result<(), Error> {
        match column.layout {
            ColumnLayout::Regular { width } => {
                // The width is still untrusted here — this is the gate that
                // proves it against the real buffer — so the row length is
                // computed checked rather than wrapped.
                let per = column
                    .dtype
                    .checked_row_bytes(width)
                    .ok_or(Error::TooLarge)?;
                if per.checked_mul(len) != Some(column.data.len) {
                    return Err(Error::Inconsistent);
                }
            }
            ColumnLayout::Irregular { offsets } => {
                self.check_buffer(offsets.range, alignment)?;
                let expected = len
                    .checked_add(1)
                    .and_then(|count| count.checked_mul(offsets.dtype.size()))
                    .ok_or(Error::TooLarge)?;
                if offsets.range.len != expected {
                    return Err(Error::Inconsistent);
                }
            }
        }
        self.check_buffer(column.data, alignment)
    }

    /// Confirm a buffer begins on the declared alignment and ends within the
    /// mapped file.
    fn check_buffer(&self, range: ByteRange, alignment: usize) -> Result<(), Error> {
        if !range.start.is_multiple_of(alignment) {
            return Err(Error::Inconsistent);
        }
        let end = self
            .segment_start
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

impl<Allocator: AllocatorLike> GraphLike for GraphTable<Allocator> {
    fn len(&self) -> usize {
        self.key.rows()
    }

    fn key_dtype(&self) -> &str {
        self.dtype_name("key").unwrap_or("")
    }

    fn value_dtype(&self) -> &str {
        self.dtype_name("value").unwrap_or("")
    }

    fn key_shape(&self) -> Option<Shape> {
        declared_shape(&self.key)
    }

    fn value_shape(&self) -> Option<Shape> {
        declared_shape(&self.value)
    }

    fn neighbors_degree(&self) -> Option<usize> {
        self.columns_view().neighbors_degree()
    }

    fn key(&self, node: usize) -> &[u8] {
        self.columns_view().key(node)
    }

    fn value(&self, node: usize) -> &[u8] {
        self.columns_view().value(node)
    }

    fn neighbors(&self, node: usize) -> NeighborList<'_> {
        self.columns_view().neighbors(node)
    }
}

/// One node's three payloads, yielded by [`GraphLike::iter`].
pub struct Node<'a> {
    pub key: &'a [u8],
    pub value: &'a [u8],
    pub neighbors: NeighborList<'a>,
}

/// A graph read node by node. Both a producer's graph and a parsed [`GraphTable`]
/// implement it, so a `GraphTable` can be built from any `GraphLike`, including
/// another `GraphTable`.
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

    /// The key column's logical [`Shape`], or `None` for ragged keys of no fixed
    /// width. Defaults to a scalar key, and to ragged for `utf8`, which is what a
    /// producer of variable-length string keys wants without saying so.
    fn key_shape(&self) -> Option<Shape> {
        match self.key_dtype() {
            "utf8" => None,
            _ => Some(Shape::Scalar),
        }
    }

    /// The value column's logical [`Shape`], or None if the producer declares
    /// none. Rank-2 families (shapes, maxsim) read the second axis from it; the
    /// vector-shaped families call [`value_width`](GraphLike::value_width).
    fn value_shape(&self) -> Option<Shape>;

    /// The value column's flat element width for the vector-shaped families
    /// (dense, binary, geospatial), or None for a scalar-less / rank-2 column.
    /// A provided method over [`value_shape`](GraphLike::value_shape) so those
    /// families keep reading a single `usize`.
    fn value_width(&self) -> Option<usize> {
        self.value_shape().and_then(Shape::width)
    }

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
pub struct GraphIter<'a, Graph: GraphLike> {
    graph: &'a Graph,
    node: usize,
}

impl<'a, Graph: GraphLike> Iterator for GraphIter<'a, Graph> {
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

fn push_bytes<Allocator: AllocatorLike>(
    v: &mut AVec<u8, Allocator>,
    bytes: &[u8],
) -> Result<(), Error> {
    v.try_reserve(bytes.len()).map_err(|_| Error::Alloc)?;
    v.extend_from_slice(bytes);
    Ok(())
}

fn push_offset<Allocator: AllocatorLike>(
    v: &mut AVec<u8, Allocator>,
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

/// Accumulates one column's on-disk buffers: the concatenated rows, and `u64`
/// prefix-sum offsets when the column is irregular.
struct ColumnBuilder<Allocator: AllocatorLike> {
    /// Storage width of one element.
    dtype: StorageDType,
    /// Elements per row, or `None` when the rows are ragged.
    width: Option<usize>,
    /// The rows themselves, concatenated.
    data: AVec<u8, Allocator>,
    /// `nodes + 1` prefix sums, ragged columns only.
    offsets: AVec<u8, Allocator>,
    /// Elements written so far, the next row's start.
    running: u64,
}

impl<Allocator: AllocatorLike + Clone> ColumnBuilder<Allocator> {
    fn new(dtype: StorageDType, width: Option<usize>, alloc: &Allocator) -> Result<Self, Error> {
        let mut offsets = AVec::new_in(alloc.clone());
        if width.is_none() {
            push_bytes(&mut offsets, &0u64.to_le_bytes())?;
        }
        Ok(ColumnBuilder {
            dtype,
            width,
            data: AVec::new_in(alloc.clone()),
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
                let row = self.dtype.checked_row_bytes(width).ok_or(Error::TooLarge)?;
                if bytes.len() != row {
                    return Err(Error::Inconsistent);
                }
                push_bytes(&mut self.data, bytes)?;
            }
            // Irregular: whole-byte elements counted for the prefix sum. A sub-byte
            // dtype never arrives here — only sequences are irregular, and they are
            // `utf8`, a byte.
            None => {
                let element_size = self.dtype.size();
                if element_size == 0 || !bytes.len().is_multiple_of(element_size) {
                    return Err(Error::Inconsistent);
                }
                push_bytes(&mut self.data, bytes)?;
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

fn reserve_offsets<Allocator: AllocatorLike + Clone>(
    column: &ColumnBuilder<Allocator>,
    cursor: &mut usize,
    alignment: usize,
) -> Result<Option<usize>, Error> {
    if column.is_irregular() {
        Ok(Some(reserve_span(cursor, column.offsets.len(), alignment)?))
    } else {
        Ok(None)
    }
}

fn write_column<Allocator: AllocatorLike>(
    header: &mut String,
    name: &str,
    dtype: &str,
    column: &ColumnBuilder<Allocator>,
    shape: Shape,
    off_start: Option<usize>,
    data_start: usize,
) {
    header.push_str(&format!(",\"{name}\":{{\"dtype\":\"{dtype}\""));
    match off_start {
        // An irregular column carries its prefix sums — and, when its logical
        // shape is a ragged `Matrix`, that shape too, or a reader would parse
        // the column back as `Scalar` and lose the fixed inner axis. A plain
        // irregular sequence stays shapeless: absent means `Scalar`, so old
        // files and new writers agree byte for byte.
        Some(off) => {
            header.push_str(&format!(
                ",\"offsets_dtype\":\"u64\",\"offsets\":[{},{}]",
                off,
                off + column.offsets.len()
            ));
            if let Shape::Matrix(rows, cols) = shape {
                header.push_str(&format!(",\"shape\":[{rows},{cols}]"));
            }
        }
        // A `Matrix` row serializes both axes so the reader recovers the logical
        // shape, not just the flat byte width; everything else is a scalar width.
        None => match shape {
            Shape::Matrix(rows, cols) => header.push_str(&format!(",\"shape\":[{rows},{cols}]")),
            // An absent `shape` reads back as `Scalar`, so writing nothing is
            // what makes a scalar column round-trip as itself instead of
            // returning as `Vector(1)`.
            Shape::Scalar => {}
            Shape::Vector(cols) => header.push_str(&format!(",\"shape\":{cols}")),
        },
    }
    header.push_str(&format!(
        ",\"data\":[{},{}]}}",
        data_start,
        data_start + column.data.len()
    ));
}

fn write_buffer<Allocator: AllocatorLike>(
    file: &mut AVec<u8, Allocator>,
    segment_start: usize,
    start: usize,
    bytes: &[u8],
) {
    let target = segment_start + start;
    while file.len() < target {
        file.push(0);
    }
    file.extend_from_slice(bytes);
}

/// Serialize a graph into a complete `.safegraphs` blob in `alloc`.
fn materialize<Graph, MetadataAllocator, Allocator>(
    graph: &Graph,
    metadata: Option<&Json<MetadataAllocator>>,
    alloc: Allocator,
) -> Result<AVec<u8, Allocator>, Error>
where
    Graph: GraphLike,
    MetadataAllocator: AllocatorLike,
    Allocator: AllocatorLike + Clone,
{
    let node_count = graph.len();
    let alignment = declared_alignment(metadata)?;

    let key_dtype = graph.key_dtype();
    let key_storage = StorageDType::from_dtype(key_dtype.as_bytes())?;
    let key_shape = graph.key_shape();
    let mut key = ColumnBuilder::new(
        key_storage,
        key_shape.and_then(Shape::regular_width),
        &alloc,
    )?;
    for i in 0..node_count {
        key.push_row(graph.key(i))?;
    }

    let value_dtype = graph.value_dtype();
    let value_storage = StorageDType::from_dtype(value_dtype.as_bytes())?;
    let mut value = ColumnBuilder::new(
        value_storage,
        graph.value_shape().and_then(Shape::regular_width),
        &alloc,
    )?;
    for i in 0..node_count {
        value.push_row(graph.value(i))?;
    }

    // ScoreLike the neighbor index type to whatever the node count needs.
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
    let mut scratch: AVec<u8, Allocator> = AVec::new_in(alloc.clone());
    for i in 0..node_count {
        let list = graph.neighbors(i);
        if list.dtype == neighbor_dtype {
            // Same width: the bytes are already node indices in the target
            // encoding, so copy the row verbatim with no per-edge widening. This
            // is the path a re-materialized `GraphTable` always takes.
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
    let key_data_start = reserve_span(&mut cursor, key.data.len(), alignment)?;
    let value_offsets_start = reserve_offsets(&value, &mut cursor, alignment)?;
    let value_data_start = reserve_span(&mut cursor, value.data.len(), alignment)?;
    let neighbor_offsets_start = reserve_offsets(&neighbors, &mut cursor, alignment)?;
    let neighbor_data_start = reserve_span(&mut cursor, neighbors.data.len(), alignment)?;
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
        key_shape.unwrap_or(Shape::Scalar),
        key_offsets_start,
        key_data_start,
    );
    write_column(
        &mut header,
        "value",
        value_dtype,
        &value,
        graph.value_shape().unwrap_or(Shape::Scalar),
        value_offsets_start,
        value_data_start,
    );
    write_column(
        &mut header,
        "neighbors",
        neighbor_dtype_name,
        &neighbors,
        Shape::Vector(neighbors.width.unwrap_or(1)),
        neighbor_offsets_start,
        neighbor_data_start,
    );
    header.push('}');

    let unpadded = 16usize.checked_add(header.len()).ok_or(Error::TooLarge)?;
    let header_pad = (alignment - (unpadded % alignment)) % alignment;
    let header_len = header.len() + header_pad;
    let segment_start = 16 + header_len;
    let total = segment_start.checked_add(data_len).ok_or(Error::TooLarge)?;

    let mut file: AVec<u8, Allocator> = AVec::new_in(alloc);
    file.try_reserve_exact(total).map_err(|_| Error::Alloc)?;
    file.extend_from_slice(&MAGIC);
    file.extend_from_slice(&(header_len as u64).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    for _ in 0..header_pad {
        file.push(b' ');
    }
    if let Some(off) = key_offsets_start {
        write_buffer(&mut file, segment_start, off, &key.offsets);
    }
    write_buffer(&mut file, segment_start, key_data_start, &key.data);
    if let Some(off) = value_offsets_start {
        write_buffer(&mut file, segment_start, off, &value.offsets);
    }
    write_buffer(&mut file, segment_start, value_data_start, &value.data);
    if let Some(off) = neighbor_offsets_start {
        write_buffer(&mut file, segment_start, off, &neighbors.offsets);
    }
    write_buffer(
        &mut file,
        segment_start,
        neighbor_data_start,
        &neighbors.data,
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
        fn value_shape(&self) -> Option<Shape> {
            Some(Shape::Vector(self.dim))
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
        let table = GraphTable::build(&toy(), None).unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(table.value_shape(), Some(Shape::Vector(2)));
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
        let table = GraphTable::build(&toy(), None).unwrap();
        let bytes = table.bytes().to_vec();
        let reopened = GraphTable::from_bytes(&bytes).unwrap();
        assert_eq!(reopened.len(), 3);
        assert_eq!(reopened.key(2), &30u64.to_le_bytes()[..]);
    }

    #[test]
    fn header_reads_from_its_prefix_alone() {
        let bytes = GraphTable::build(&toy(), None).unwrap().bytes().to_vec();
        let segment_start = GraphHeader::segment_start_of(&bytes[..16]).unwrap();
        assert!(matches!(
            GraphHeader::read_in(&bytes[..segment_start - 1], Global),
            Err(Error::Truncated)
        ));
        let header = GraphHeader::read_in(&bytes[..segment_start], Global).unwrap();
        assert_eq!(header.file_bytes(), bytes.len());

        // A producer column past every reader column still bounds the file, as the C++ reader counts it.
        let json = core::str::from_utf8(&bytes[16..segment_start])
            .unwrap()
            .trim_end_matches(' ');
        let widened = format!(
            "{},\"norms\":{{\"dtype\":\"f32\",\"data\":[65536,65548]}}}}",
            json.strip_suffix('}').unwrap()
        );
        let mut prefix = MAGIC.to_vec();
        prefix.extend_from_slice(&(widened.len() as u64).to_le_bytes());
        prefix.extend_from_slice(widened.as_bytes());
        let header = GraphHeader::read_in(&prefix, Global).unwrap();
        assert_eq!(header.file_bytes(), prefix.len() + 65548);
    }

    #[test]
    fn table_is_graphlike() {
        // A GraphTable implements GraphLike, so it can be re-materialized into another.
        let table = GraphTable::build(&toy(), None).unwrap();
        let rebuilt = GraphTable::build(&table, None).unwrap();
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

    /// A ragged value column, the layout every sequence metric reads. Rebuilding
    /// one is what a shape of `None` on both sides of the trait makes possible.
    struct RaggedValues;

    impl GraphLike for RaggedValues {
        fn len(&self) -> usize {
            3
        }
        fn key_dtype(&self) -> &str {
            "u64"
        }
        fn value_dtype(&self) -> &str {
            "utf8"
        }
        fn value_shape(&self) -> Option<Shape> {
            None
        }
        fn neighbors_degree(&self) -> Option<usize> {
            Some(1)
        }
        fn key(&self, i: usize) -> &[u8] {
            const KEYS: [[u8; 8]; 3] = [
                (10u64).to_le_bytes(),
                (20u64).to_le_bytes(),
                (30u64).to_le_bytes(),
            ];
            &KEYS[i]
        }
        fn value(&self, i: usize) -> &[u8] {
            const WORDS: [&[u8]; 3] = [b"kitten", b"sitting", b"mitten"];
            WORDS[i]
        }
        fn neighbors(&self, i: usize) -> NeighborList<'_> {
            const ROWS: [[u8; 4]; 3] = [[1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]];
            NeighborList::new(&ROWS[i], OffsetDType::U32)
        }
    }

    #[test]
    fn a_ragged_value_column_rebuilds() {
        let table = GraphTable::build(&RaggedValues, None).unwrap();
        assert_eq!(table.value_shape(), None);
        assert_eq!(table.value(1), b"sitting");

        let rebuilt = GraphTable::build(&table, None).unwrap();
        assert_eq!(rebuilt.value_shape(), None);
        assert_eq!(rebuilt.value(0), b"kitten");
        assert_eq!(rebuilt.value(2), b"mitten");
        assert_eq!(rebuilt.as_bytes(), table.as_bytes());
    }

    /// Fixed-width `utf8` keys, whose layout a rebuild takes from the declared
    /// shape rather than from the dtype string.
    struct FixedKeys;

    impl GraphLike for FixedKeys {
        fn len(&self) -> usize {
            2
        }
        fn key_dtype(&self) -> &str {
            "utf8"
        }
        fn key_shape(&self) -> Option<Shape> {
            Some(Shape::Vector(8))
        }
        fn value_dtype(&self) -> &str {
            "f32"
        }
        fn value_shape(&self) -> Option<Shape> {
            Some(Shape::Vector(1))
        }
        fn neighbors_degree(&self) -> Option<usize> {
            Some(1)
        }
        fn key(&self, i: usize) -> &[u8] {
            const KEYS: [&[u8; 8]; 2] = [b"wiki-1m\0", b"deep-1b\0"];
            KEYS[i]
        }
        fn value(&self, _i: usize) -> &[u8] {
            &[0, 0, 128, 63]
        }
        fn neighbors(&self, i: usize) -> NeighborList<'_> {
            const ROWS: [[u8; 4]; 2] = [[1, 0, 0, 0], [0, 0, 0, 0]];
            NeighborList::new(&ROWS[i], OffsetDType::U32)
        }
    }

    #[test]
    fn a_fixed_width_key_column_stays_fixed_through_a_rebuild() {
        let table = GraphTable::build(&FixedKeys, None).unwrap();
        assert_eq!(table.key_shape(), Some(Shape::Vector(8)));
        assert_eq!(table.key(0), b"wiki-1m\0");

        let rebuilt = GraphTable::build(&table, None).unwrap();
        assert_eq!(rebuilt.key_shape(), Some(Shape::Vector(8)));
        assert_eq!(rebuilt.key(1), b"deep-1b\0");
        assert_eq!(rebuilt.as_bytes(), table.as_bytes());
    }

    /// The header names each buffer once, as a pair of byte positions.
    #[test]
    fn buffers_are_written_as_position_pairs() {
        let table = GraphTable::build(&RaggedValues, None).unwrap();
        let bytes = table.as_bytes();
        // The header is the `header_len` bytes after the magic and the length.
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header = core::str::from_utf8(&bytes[16..16 + header_len]).unwrap();
        assert!(header.contains("\"data\":["));
        assert!(header.contains("\"offsets\":["));
    }

    /// A monolith exactly as USearch writes one: string-valued metadata, UUID keys
    /// holding the row number big-endian in their last 8 bytes, and a ragged `f16`
    /// point-cloud column whose `u64` offsets count scalars, three per point.
    #[test]
    fn a_usearch_monolith_opens() {
        let header = br#"{"__metadata__":{"alignment":4096,"metric":"chamfer","created_by":"usearch","connectivity":"32","points":"0"},"key":{"dtype":"uuid","data":[0,32]},"value":{"dtype":"f16","shape":[0,3],"offsets_dtype":"u64","offsets":[4096,4120],"data":[8192,8210]},"neighbors":{"dtype":"u32","offsets_dtype":"u64","offsets":[12288,12312],"data":[16384,16392]}}"#;
        // USearch pads the header with spaces so the data segment starts aligned.
        let segment = 4096;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&(segment as u64 - 16).to_le_bytes());
        file.extend_from_slice(header);
        file.resize(segment, b' ');
        file.resize(segment + 16392, 0);
        let mut write = |at: usize, bytes: &[u8]| {
            file[segment + at..segment + at + bytes.len()].copy_from_slice(bytes)
        };
        write(24, &1u64.to_be_bytes());
        for (slot, offset) in [0u64, 3, 9].iter().enumerate() {
            write(4096 + slot * 8, &offset.to_le_bytes());
        }
        for (slot, offset) in [0u64, 1, 2].iter().enumerate() {
            write(12288 + slot * 8, &offset.to_le_bytes());
        }
        write(16384, &[1, 0, 0, 0, 0, 0, 0, 0]);

        let table = GraphTable::from_bytes(&file).unwrap();
        assert_eq!(table.metric(), Some("chamfer"));
        assert_eq!(table.len(), 2);
        assert_eq!(table.value_shape(), Some(Shape::Matrix(0, 3)));
        assert_eq!(table.value(0).len(), 3 * 2);
        assert_eq!(table.value(1).len(), 6 * 2);
        assert_eq!(table.key(1)[8..], 1u64.to_be_bytes());
        assert_eq!(table.neighbors(0).as_slice::<u32>(), Some(&[1u32][..]));
    }

    /// An alignment that is not a power of two is refused, whether it is above
    /// the floor or below it, rather than coerced to 64.
    #[test]
    fn a_bad_alignment_is_refused() {
        for alignment in ["96", "3"] {
            let metadata = format!("{{\"alignment\":{alignment}}}");
            let document = parse_in(metadata.as_bytes(), Global).unwrap();
            assert!(matches!(
                GraphTable::build(&toy(), Some(&document)),
                Err(Error::Inconsistent)
            ));
        }
    }
}

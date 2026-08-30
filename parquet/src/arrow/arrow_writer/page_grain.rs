// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Page-grain control over parquet column chunk encoding.
//!
//! The parquet writer already lets a caller drive row groups into a file
//! ([`SerializedFileWriter::next_row_group`]) and column chunks into a row group
//! ([`ArrowColumnWriter`] producing an [`ArrowColumnChunk`]). This module opens
//! the remaining grain: the *page*. It lets a caller encode the same rows
//! several ways, look at what each one actually cost, and commit the one it
//! prefers — without giving up any of the invariants that make the output a
//! valid parquet file.
//!
//! # The shape
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use arrow_array::{ArrayRef, StringArray};
//! # use arrow_schema::{DataType, Field};
//! # use parquet::arrow::arrow_writer::compute_leaves;
//! # use parquet::arrow::arrow_writer::page_grain::{Candidate, ColumnChunkBuilder};
//! # use parquet::basic::Encoding;
//! # use parquet::errors::Result;
//! # use parquet::file::properties::WriterProperties;
//! # use parquet::schema::types::ColumnDescPtr;
//! # fn run(descr: ColumnDescPtr, props: Arc<WriterProperties>, field: &Field, array: &ArrayRef) -> Result<()> {
//! let mut chunk = ColumnChunkBuilder::new(descr, props)?;
//!
//! for leaf in compute_leaves(field, array)? {
//!     let mut cursor = chunk.cursor(&leaf);
//!     while !cursor.is_empty() {
//!         // Encode the next page as indices into the chunk's dictionary, and
//!         // the very same rows again as PLAIN. The first candidate encodes
//!         // until its own page budget is reached, and that is where the page
//!         // ends; the alternatives are handed exactly the rows it consumed.
//!         let pages = chunk.encode_page_alternatives(
//!             &mut cursor,
//!             Candidate::Dictionary,
//!             &[Candidate::Encoding(Encoding::PLAIN)],
//!         )?;
//!
//!         // Nothing has entered the chunk yet: every page here is fully
//!         // encoded and measured, and rejecting one is just dropping it.
//!         // A dictionary-indices page is small because its bytes went into
//!         // the dictionary instead, so charge it what it added there.
//!         let best = pages
//!             .into_iter()
//!             .min_by_key(|p| p.compressed_len() + p.dictionary_growth())
//!             .expect("at least one page");
//!         chunk.append(best)?;
//!     }
//! }
//!
//! let chunk = chunk.close()?; // an ordinary `ArrowColumnChunk`
//! # Ok(())
//! # }
//! ```
//!
//! # What the API guarantees
//!
//! * **Decide before commit.** [`ColumnChunkBuilder::encode_page`] and
//!   [`ColumnChunkBuilder::encode_page_alternatives`] return sealed
//!   [`EncodedPage`]s. They are fully encoded and compressed, so
//!   [`EncodedPage::compressed_len`] is a measurement rather than an estimate,
//!   and dropping one costs nothing but the work already done.
//! * **The split point is an output.** The caller never names a row offset.
//!   [`LeafCursor`] advances only by whatever the encoding candidate's own budget
//!   ([`WriterProperties::data_page_size_limit`],
//!   [`WriterProperties::data_page_row_count_limit`], and the byte-budget walk
//!   in `count_values_within_byte_budget`) decided, and only ever at a record
//!   boundary.
//! * **Validity by construction.** The builder owns the dictionary, the
//!   dictionary page's position, the `encodings` set, the page encoding
//!   statistics, the column index, the offset index, boundary order, statistics
//!   truncation and the chunk metadata. A caller cannot supply any of them, and
//!   cannot append a dictionary-indices page after a page that is not one.
//! * **The dictionary is an object.** [`ColumnChunkBuilder::dictionary`] reads
//!   its live state at any time; a caller lands on another encoding simply by
//!   no longer offering [`Candidate::Dictionary`]. Dictionary-then-fallback is
//!   expressible; fallback-then-dictionary is not representable.
//!
//! # Pages and record batches
//!
//! A [`LeafCursor`] covers exactly one [`ArrowLeafColumn`], and encoding seals
//! whatever the candidate has buffered when that cursor runs out, so a page
//! never spans two leaves. A
//! caller feeding 8192 row record batches therefore gets 8192 row pages even
//! where [`WriterProperties::data_page_row_count_limit`] and the byte budget
//! would have allowed a larger page, and pays for the extra pages in page
//! headers, column and offset index entries, and compression window. Feed the
//! builder the largest leaves the caller can afford when page size matters.
//!
//! # Chunk-level accumulators
//!
//! Bloom filters and geospatial statistics are fed by *values*, once per value,
//! and belong to the chunk rather than to any page — so racing K candidate
//! encodings of the same rows must not insert each value K times. The builder
//! owns them and lends them only to the candidate that decides the page
//! boundary, for exactly the rows it consumes. This happens at *pace* time
//! rather than at append time because the values are in the chunk no matter
//! which encoding wins.
//!
//! [`SerializedFileWriter::next_row_group`]: crate::file::writer::SerializedFileWriter::next_row_group
//! [`WriterProperties::data_page_size_limit`]: crate::file::properties::WriterProperties::data_page_size_limit
//! [`WriterProperties::data_page_row_count_limit`]: crate::file::properties::WriterProperties::data_page_row_count_limit

use std::sync::Arc;

use super::byte_array::ByteArrayEncoder;
use super::levels::{ArrayLevels, LevelData};
use super::{
    ArrowColumnChunk, ArrowColumnWriterImpl, ArrowLeafColumn, ArrowPageWriter, SharedColumnChunk,
    write_levels,
};
use crate::basic::{Encoding, Type};
use crate::column::chunker::CdcChunk;
use crate::column::page::{CompressedPage, PageWriteSpec, PageWriter};
use crate::column::page_store::{InMemoryPageStoreFactory, PageStoreArgs, PageStoreFactory};
use crate::column::writer::encoder::{
    ColumnValueEncoder, DictionaryPage, DynDictionary, ValueAccumulators,
};
use crate::column::writer::{
    ColumnWriter, GenericColumnWriter, PageBoundaryAction, PreparedDataPage,
};
use crate::data_type::{
    BoolType, ByteArray, DoubleType, FixedLenByteArray, FixedLenByteArrayType, FloatType,
    Int32Type, Int64Type, Int96, Int96Type,
};
use crate::errors::{ParquetError, Result};
use crate::file::properties::WriterPropertiesPtr;
use crate::schema::types::ColumnDescPtr;

/// One way of encoding a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Candidate {
    /// Encode indices into the column chunk's dictionary.
    ///
    /// A page may be encoded this way only while the chunk still has a
    /// dictionary: it is dropped once a page that is not dictionary indices has
    /// been appended, or if dictionary encoding was disabled in the properties.
    /// One page has one dictionary candidate, because the chunk has one
    /// dictionary.
    Dictionary,
    /// Encode with this encoding, whatever the properties would have chosen.
    Encoding(Encoding),
}

/// A position within an [`ArrowLeafColumn`], advanced only by encoding.
///
/// There is deliberately no way to construct a cursor at, or move one to, a
/// caller-chosen row offset. The only thing that moves it is encoding a page,
/// by exactly as many rows as that page holds.
#[derive(Debug)]
pub struct LeafCursor {
    levels: ArrayLevels,
    /// Levels consumed so far.
    level_offset: usize,
    /// Entries of `non_null_indices` consumed so far.
    value_offset: usize,
    total_levels: usize,
    max_def_level: i16,
    /// Levels to hand the pacer per pacing step; see `ColumnChunkBuilder::pace`.
    window: usize,
}

impl LeafCursor {
    /// Whether every level in the leaf has been encoded.
    pub fn is_empty(&self) -> bool {
        self.level_offset >= self.total_levels
    }

    /// Levels (values plus nulls plus repeats) still to encode.
    fn levels_remaining(&self) -> usize {
        self.total_levels - self.level_offset
    }

    /// Cut the next pacing window, starting `levels_taken` levels past the
    /// cursor: at most `window` levels, extended forwards to the next record
    /// boundary so the window never ends inside a record. The window end is the
    /// one place the page budget can trip that `write_batch_inner`'s own
    /// record-aligned chunking does not protect, so it must be a record
    /// boundary itself.
    fn next_window(&self, levels_taken: usize, values_taken: usize) -> CdcChunk {
        let level_offset = self.level_offset + levels_taken;
        let remaining = self.total_levels - level_offset;
        let mut num_levels = self.window.min(remaining);

        // Extend to the next `rep == 0`. A record that starts inside the window
        // is carried whole; `write_batch_inner` will not split it either.
        if let LevelData::Materialized(rep) = self.levels.rep_level_data() {
            while num_levels < remaining && rep[level_offset + num_levels] != 0 {
                num_levels += 1;
            }
        }

        let def = self.levels.def_level_data().as_ref();
        let num_values = def
            .slice(level_offset, num_levels)
            .value_count(num_levels, self.max_def_level);

        CdcChunk {
            level_offset,
            num_levels,
            value_offset: self.value_offset + values_taken,
            num_values,
        }
    }

    /// Cut exactly the rows the first candidate consumed, so the alternatives encode
    /// the same rows.
    fn span(&self, num_levels: usize, num_values: usize) -> CdcChunk {
        CdcChunk {
            level_offset: self.level_offset,
            num_levels,
            value_offset: self.value_offset,
            num_values,
        }
    }

    fn advance(&mut self, num_levels: usize, num_values: usize) {
        self.level_offset += num_levels;
        self.value_offset += num_values;
    }
}

/// A live view of a column chunk's dictionary.
///
/// Cheap to obtain and never cached: read it whenever a decision depends on it.
#[derive(Debug, Clone, Copy)]
pub struct DictionaryView {
    entries: usize,
    encoded_bytes: usize,
}

impl DictionaryView {
    /// Distinct values interned so far.
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// Encoded size of the dictionary page's entries, in bytes.
    ///
    /// This is what the chunk will pay for the dictionary page at close, and it
    /// grows as pages intern new values.
    pub fn encoded_bytes(&self) -> usize {
        self.encoded_bytes
    }
}

/// The concrete `PreparedDataPage` behind an [`EncodedPage`].
///
/// Erasing the value type here is what lets one `EncodedPage` type describe a
/// page for any physical type; the builder checks the variant matches its own
/// column before committing.
enum PreparedPage {
    Bool(PreparedDataPage<bool>),
    Int32(PreparedDataPage<i32>),
    Int64(PreparedDataPage<i64>),
    Int96(PreparedDataPage<Int96>),
    Float(PreparedDataPage<f32>),
    Double(PreparedDataPage<f64>),
    ByteArray(PreparedDataPage<ByteArray>),
    FixedLenByteArray(PreparedDataPage<FixedLenByteArray>),
}

/// A fully encoded, compressed data page that has not entered any column chunk.
///
/// Everything here is measured rather than estimated. Dropping an `EncodedPage`
/// is the whole of "reject this candidate": no chunk state has been touched, so
/// there is nothing to undo.
pub struct EncodedPage {
    prepared: PreparedPage,
    encoding: Encoding,
    num_values: u32,
    num_rows: u32,
    compressed_len: usize,
    uncompressed_len: usize,
    dictionary_growth: usize,
}

impl std::fmt::Debug for EncodedPage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedPage")
            .field("encoding", &self.encoding)
            .field("num_values", &self.num_values)
            .field("num_rows", &self.num_rows)
            .field("compressed_len", &self.compressed_len)
            .field("uncompressed_len", &self.uncompressed_len)
            .field("dictionary_growth", &self.dictionary_growth)
            .finish_non_exhaustive()
    }
}

impl EncodedPage {
    /// The encoding this page's values were written with.
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Number of levels (values plus nulls) in the page.
    pub fn num_values(&self) -> u32 {
        self.num_values
    }

    /// Number of records in the page.
    pub fn num_rows(&self) -> u32 {
        self.num_rows
    }

    /// Compressed size of the page body in bytes, excluding the page header.
    ///
    /// This is the number to compare candidates on: the page has actually been
    /// compressed with the column's codec.
    pub fn compressed_len(&self) -> usize {
        self.compressed_len
    }

    /// Uncompressed size of the page body in bytes, excluding the page header.
    pub fn uncompressed_len(&self) -> usize {
        self.uncompressed_len
    }

    /// Bytes this page's encoding added to the column chunk's dictionary.
    ///
    /// Zero for every page that does not encode dictionary indices. A
    /// dictionary-indices page is small precisely because the values it interned
    /// are written once, later, in the chunk's dictionary page; those bytes are
    /// invisible in [`Self::compressed_len`], and this is how many of them this
    /// page created. Comparing `compressed_len() + dictionary_growth()` across
    /// the candidates for one page compares like with like.
    pub fn dictionary_growth(&self) -> usize {
        self.dictionary_growth
    }

    /// Whether this page encodes indices into the chunk's dictionary.
    fn is_dictionary_indices(&self) -> bool {
        matches!(
            self.encoding,
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
        )
    }
}

/// A page writer that accepts nothing.
///
/// Candidate writers only ever *assemble* pages; a page they assembled is
/// either committed to the builder's real writer or dropped, so their own page
/// writer must never be reached. Reaching it is a bug in this module rather
/// than a caller error, hence the explicit error rather than a buffer.
#[derive(Debug)]
struct NullPageWriter;

impl PageWriter for NullPageWriter {
    fn write_page(&mut self, _page: CompressedPage) -> Result<PageWriteSpec> {
        Err(general_err!(
            "internal error: a page-grain candidate writer tried to write a page"
        ))
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Builds one [`ArrowColumnChunk`] a page at a time.
///
/// The builder owns everything that makes the chunk valid: the dictionary and
/// the position of its page, the `encodings` set and page encoding statistics,
/// the column index, the offset index, boundary order, statistics truncation
/// and the chunk metadata. All of it is derived from the pages that were
/// actually appended.
pub struct ColumnChunkBuilder {
    descr: ColumnDescPtr,
    props: WriterPropertiesPtr,
    /// The real column writer: it commits appended pages and, at close, builds
    /// the chunk's metadata and indexes exactly as the normal writer does.
    writer: ArrowColumnWriterImpl,
    chunk: SharedColumnChunk,
    /// The chunk's dictionary, or `None` once it has been abandoned or was
    /// never enabled. Lent to a `Candidate::Dictionary` encoder for the
    /// duration of one page.
    dictionary: Option<Box<dyn DynDictionary>>,
    /// Set once a dictionary-indices page has been appended, so `close` knows a
    /// dictionary page is owed.
    dictionary_page_owed: bool,
    /// Set once a page that is *not* dictionary indices has been appended.
    /// After this, an indices page is rejected: parquet allows
    /// dictionary-then-fallback within a chunk, never the reverse.
    dictionary_closed: bool,
    /// Value-fed chunk accumulators; see the module docs and
    /// [`ValueAccumulators`].
    accumulators: ValueAccumulators,
    pages_appended: usize,
    pace_window: usize,
}

impl std::fmt::Debug for ColumnChunkBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnChunkBuilder")
            .field("column", &self.descr.path().string())
            .field("pages_appended", &self.pages_appended)
            .field("dictionary", &self.dictionary())
            .finish_non_exhaustive()
    }
}

impl ColumnChunkBuilder {
    /// Start a column chunk for `descr` under `props`.
    ///
    /// The physical type selects the encoder, matching what
    /// [`ArrowColumnWriter`] would use for the same leaf: `BYTE_ARRAY` columns
    /// go through the arrow byte-array encoder, everything else through the
    /// generic column writer.
    pub fn new(descr: ColumnDescPtr, props: WriterPropertiesPtr) -> Result<Self> {
        Self::new_with_page_store(descr, props, &InMemoryPageStoreFactory, 0)
    }

    /// As [`Self::new`], but with a caller-supplied [`PageStoreFactory`] and
    /// column index, matching what [`ArrowWriter`](super::ArrowWriter) does for
    /// spilling page stores.
    pub fn new_with_page_store(
        descr: ColumnDescPtr,
        props: WriterPropertiesPtr,
        page_store_factory: &dyn PageStoreFactory,
        column_index: usize,
    ) -> Result<Self> {
        let store = page_store_factory.create(&PageStoreArgs::new(column_index, &descr))?;
        let page_writer = Box::new(ArrowPageWriter::new(store));
        let chunk = page_writer.buffer.clone();

        // Build the writer the normal path would build, then lift the two
        // pieces of chunk-level state out of its encoder into the builder. That
        // is how the builder gets a dictionary and accumulators configured
        // exactly as `WriterProperties` says, without reimplementing either.
        //
        // The real writer must own no dictionary of its own: the chunk's
        // dictionary is an explicit object here, and a writer that thinks it has
        // one would buffer data pages and try to emit a dictionary page at
        // close. The accumulators are handed back to this encoder just before
        // `close`, so the chunk's bloom filter and geospatial statistics land in
        // the metadata by the normal path.
        let mut writer = build_writer(&descr, &props, page_writer)?;
        let accumulators = take_accumulators(&mut writer);
        let dictionary = take_dictionary(&mut writer);

        let batch = props.write_batch_size().max(1);
        let rows = props.data_page_row_count_limit();
        // A page never holds more than `data_page_row_count_limit` rows, so one
        // window of that many levels is enough for the common flat case. Round
        // up to a whole number of `write_batch_size` mini-batches so window
        // boundaries fall where mini-batch boundaries would have fallen anyway,
        // which is what keeps page boundaries identical to the normal path.
        let pace_window = match rows.checked_next_multiple_of(batch) {
            Some(w) if w >= batch && w <= 1 << 24 => w,
            _ => batch,
        };

        Ok(Self {
            descr,
            props,
            writer,
            chunk,
            dictionary,
            dictionary_page_owed: false,
            dictionary_closed: false,
            accumulators,
            pages_appended: 0,
            pace_window,
        })
    }

    /// The chunk's dictionary state, or `None` if the chunk has no dictionary
    /// (either it was never enabled, or a non-dictionary page has been appended
    /// and it has been dropped).
    pub fn dictionary(&self) -> Option<DictionaryView> {
        let dictionary = self.dictionary.as_ref()?;
        Some(DictionaryView {
            entries: dictionary.num_entries(),
            encoded_bytes: dictionary.dict_encoded_size(),
        })
    }

    /// Start a cursor over `leaf`.
    ///
    /// A cursor is independent of the builder, so pages can be appended while
    /// one is open; it just has no way to move except through
    /// [`Self::encode_page`].
    pub fn cursor(&self, leaf: &ArrowLeafColumn) -> LeafCursor {
        let levels = leaf.0.clone();
        let total_levels = levels
            .def_level_data()
            .as_ref()
            .len()
            .max(levels.rep_level_data().as_ref().len())
            .max(
                if self.descr.max_def_level() == 0 && self.descr.max_rep_level() == 0 {
                    levels.non_null_indices().len()
                } else {
                    0
                },
            );
        LeafCursor {
            total_levels,
            levels,
            level_offset: 0,
            value_offset: 0,
            max_def_level: self.descr.max_def_level(),
            window: self.pace_window,
        }
    }

    /// Encode the next page with `candidate`, without committing it.
    ///
    /// The page ends where `candidate`'s own page budget
    /// ([`WriterProperties::data_page_size_limit`],
    /// [`WriterProperties::data_page_row_count_limit`]) says it ends, or where
    /// the cursor runs out, whichever comes first; `cursor` advances by exactly
    /// that many rows. The returned page is fully encoded and compressed, and
    /// enters the chunk only when it is passed to [`Self::append`].
    ///
    /// [`WriterProperties::data_page_size_limit`]: crate::file::properties::WriterProperties::data_page_size_limit
    /// [`WriterProperties::data_page_row_count_limit`]: crate::file::properties::WriterProperties::data_page_row_count_limit
    pub fn encode_page(
        &mut self,
        cursor: &mut LeafCursor,
        candidate: Candidate,
    ) -> Result<EncodedPage> {
        let mut pages = self.encode_page_alternatives(cursor, candidate, &[])?;
        Ok(pages.pop().expect("one candidate encodes one page"))
    }

    /// Encode the next page with `candidate`, and the same rows again with each
    /// of `alternatives`, without committing any of them.
    ///
    /// `candidate` alone decides where the page ends, exactly as in
    /// [`Self::encode_page`]. Every alternative is then handed precisely the
    /// rows it consumed, so the returned pages all describe the same rows and
    /// their [`compressed_len`](EncodedPage::compressed_len)s can be compared
    /// directly. The result holds `candidate`'s page first, then one page per
    /// alternative in order, and the cursor advances once.
    ///
    /// Only `candidate` may be [`Candidate::Dictionary`]: an alternative
    /// encoding is never allowed to intern values into the chunk's dictionary,
    /// since it may be the page that is thrown away.
    pub fn encode_page_alternatives(
        &mut self,
        cursor: &mut LeafCursor,
        candidate: Candidate,
        alternatives: &[Candidate],
    ) -> Result<Vec<EncodedPage>> {
        if cursor.is_empty() {
            return Err(general_err!(
                "cannot encode a page from an exhausted cursor"
            ));
        }
        if alternatives.contains(&Candidate::Dictionary) {
            return Err(general_err!(
                "Candidate::Dictionary cannot be an alternative encoding: a column chunk \
                 has one dictionary, and only the candidate that decides the page \
                 boundary may add to it"
            ));
        }
        // `dictionary_closed` alone is not implied by `dictionary.is_none()`:
        // after dictionary-then-fallback the dictionary object is still held so
        // its page can be written at close, but no further indices page may
        // reference it. Rejecting here, before any encoding, matters because
        // encoding advances the cursor: a page rejected only at `append` would
        // leave its rows consumed with no page to show for them.
        if candidate == Candidate::Dictionary
            && (self.dictionary.is_none() || self.dictionary_closed)
        {
            return Err(general_err!(
                "this column chunk has no usable dictionary: it was disabled in the \
                 writer properties, or a non-dictionary page has been appended and the \
                 chunk can never return to dictionary encoding"
            ));
        }

        let dictionary_before = self.dictionary_encoded_size();
        let (span, mut pages) = self.pace(cursor, candidate)?;
        // Whatever this page interned is a real cost of encoding it this way,
        // and it is paid later in the chunk's dictionary page.
        pages[0].dictionary_growth = self
            .dictionary_encoded_size()
            .saturating_sub(dictionary_before);

        for alternative in alternatives {
            let mut writer = self.candidate_writer(*alternative, false)?;
            let sliced = cursor.levels.slice_for_chunk(&span);
            let (levels, _) = write_levels(&mut writer, &sliced)?;
            debug_assert_eq!(levels, span.num_levels);
            pages.push(self.seal(&mut writer, *alternative)?);
        }

        cursor.advance(span.num_levels, span.num_values);
        Ok(pages)
    }

    /// The dictionary's encoded size, or zero if the chunk has no dictionary.
    fn dictionary_encoded_size(&self) -> usize {
        self.dictionary
            .as_ref()
            .map_or(0, |d| d.dict_encoded_size())
    }

    /// Run the boundary-deciding candidate forwards until its page budget trips,
    /// returning the rows it settled on and its sealed page.
    fn pace(
        &mut self,
        cursor: &LeafCursor,
        candidate: Candidate,
    ) -> Result<(CdcChunk, Vec<EncodedPage>)> {
        let mut writer = self.candidate_writer(candidate, true)?;

        // This is the one candidate that sees each value of the page exactly
        // once, so it is the one that carries the chunk's value-fed
        // accumulators. They go in before any value is written and come back out
        // before the page is sealed, so they survive the page being rejected —
        // which is correct: the values are in the chunk whichever encoding wins.
        install_accumulators(&mut writer, std::mem::take(&mut self.accumulators));

        let mut levels_taken = 0usize;
        let mut values_taken = 0usize;
        let result = (|| -> Result<()> {
            loop {
                let window = cursor.next_window(levels_taken, values_taken);
                if window.num_levels == 0 {
                    break;
                }

                let sliced = cursor.levels.slice_for_chunk(&window);
                let (levels, values) = write_levels(&mut writer, &sliced)?;
                levels_taken += levels;
                values_taken += values;

                // Ask the writer, rather than infer from how much it consumed:
                // the budget can trip exactly at the end of a window, and
                // "consumed everything I was offered" would then look like
                // "still hungry".
                if page_boundary_reached(&writer) {
                    break;
                }
                if levels_taken >= cursor.levels_remaining() {
                    break;
                }
                // Not full yet; keep going with the writer's buffered state
                // intact, exactly as the default path continues across the
                // record batches of one row group.
            }
            Ok(())
        })();

        // Return the accumulators before anything else can go wrong with them.
        self.accumulators = take_accumulators(&mut writer);
        result?;

        if levels_taken == 0 {
            return Err(general_err!(
                "the candidate encoded no rows; this column cannot make progress"
            ));
        }

        let page = self.seal(&mut writer, candidate)?;
        Ok((cursor.span(levels_taken, values_taken), vec![page]))
    }

    /// Build a throwaway writer for one candidate encoding.
    fn candidate_writer(
        &mut self,
        candidate: Candidate,
        pacing: bool,
    ) -> Result<ArrowColumnWriterImpl> {
        let dictionary = match candidate {
            Candidate::Dictionary => Some(
                self.dictionary
                    .take()
                    .ok_or_else(|| general_err!("this column chunk has no dictionary"))?,
            ),
            Candidate::Encoding(_) => None,
        };
        let mut writer = build_writer(&self.descr, &self.props, Box::new(NullPageWriter))?;
        // A candidate encodes rows another candidate may end up winning, so it
        // must own no chunk-level state. Its own value-fed accumulators are
        // dropped here; its own dictionary is overwritten below, either by the
        // chunk's (`install_dictionary`) or by nothing (`pin_encoding` drops
        // it), so there is nothing left to strip.
        drop(take_accumulators(&mut writer));
        if let Some(dictionary) = dictionary {
            install_dictionary(&mut writer, dictionary)?;
        }
        if let Candidate::Encoding(encoding) = candidate {
            pin_encoding(&mut writer, encoding, &self.descr)?;
        }
        set_page_boundary_action(
            &mut writer,
            if pacing {
                PageBoundaryAction::Stop
            } else {
                PageBoundaryAction::Continue
            },
        );
        Ok(writer)
    }

    /// Assemble the candidate's buffered values into a page and reclaim the
    /// dictionary if it was lent.
    fn seal(
        &mut self,
        writer: &mut ArrowColumnWriterImpl,
        candidate: Candidate,
    ) -> Result<EncodedPage> {
        let prepared = assemble(writer)?;
        if candidate == Candidate::Dictionary {
            // Sealing flushed the indices, so the dictionary comes back with an
            // empty pending buffer. Entries a *losing* dictionary candidate
            // interned deliberately stay: every candidate for a span sees the
            // same values, so such an entry is a value that genuinely occurs in
            // the span, and a dictionary that is a superset of what its indices
            // reference is valid.
            self.dictionary = take_dictionary(writer);
        }
        Ok(prepared)
    }

    /// Append a sealed page to the chunk.
    ///
    /// Rejects a dictionary-indices page once a non-dictionary page has been
    /// appended, and rejects a page whose physical type is not this column's.
    /// Everything else — the `encodings` set, page encoding statistics, the
    /// column index and its boundary order, the offset index, chunk statistics
    /// and their truncation — is derived here from the page, and cannot be
    /// supplied by the caller.
    pub fn append(&mut self, page: EncodedPage) -> Result<()> {
        if page.is_dictionary_indices() {
            if self.dictionary_closed {
                return Err(general_err!(
                    "cannot append a dictionary-indices page after a non-dictionary page: \
                     a column chunk may fall back from the dictionary but never back to it"
                ));
            }
            self.dictionary_page_owed = true;
        } else {
            // Landing on another encoding: the dictionary can never be used
            // again, so drop it now rather than let it grow.
            self.dictionary_closed = true;
            if !self.dictionary_page_owed {
                self.dictionary = None;
            }
        }

        commit(&mut self.writer, page)?;
        self.pages_appended += 1;
        Ok(())
    }

    /// Finish the chunk.
    ///
    /// Writes the dictionary page (if any indices page was appended) and closes
    /// the underlying column writer, producing an ordinary [`ArrowColumnChunk`]
    /// that is appended to a row group exactly as one from
    /// [`ArrowColumnWriter::close`] would be.
    pub fn close(mut self) -> Result<ArrowColumnChunk> {
        if self.pages_appended == 0 {
            return Err(general_err!(
                "cannot close a column chunk with no appended pages"
            ));
        }
        if self.dictionary_page_owed {
            let dictionary = self.dictionary.take().ok_or_else(|| {
                general_err!("internal error: dictionary page owed but no dictionary held")
            })?;
            let page = dictionary.into_dictionary_page()?;
            write_dictionary_page(&mut self.writer, page)?;
        }

        // Hand the value-fed accumulators back so the ordinary close path folds
        // the bloom filter and geospatial statistics into the chunk metadata.
        install_accumulators(&mut self.writer, self.accumulators);

        let close = match self.writer {
            ArrowColumnWriterImpl::ByteArray(c) => c.close()?,
            ArrowColumnWriterImpl::Column(c) => c.close()?,
        };
        let chunk = Arc::try_unwrap(self.chunk)
            .map_err(|_| general_err!("internal error: column chunk buffer still shared"))?;
        let data = chunk
            .into_inner()
            .map_err(|_| general_err!("internal error: column chunk buffer poisoned"))?;
        Ok(ArrowColumnChunk { data, close })
    }
}

/// Run `$body` with `$w` bound to the concrete [`GenericColumnWriter`] behind an
/// [`ArrowColumnWriterImpl`].
///
/// The page-grain API is written once against `GenericColumnWriter`'s
/// `pub(crate)` seams; this macro is the only place the nine concrete writer
/// types are enumerated.
macro_rules! dispatch_writer {
    ($writer:expr, $w:ident, $body:expr) => {
        match $writer {
            ArrowColumnWriterImpl::ByteArray($w) => $body,
            ArrowColumnWriterImpl::Column(c) => match c {
                ColumnWriter::BoolColumnWriter($w) => $body,
                ColumnWriter::Int32ColumnWriter($w) => $body,
                ColumnWriter::Int64ColumnWriter($w) => $body,
                ColumnWriter::Int96ColumnWriter($w) => $body,
                ColumnWriter::FloatColumnWriter($w) => $body,
                ColumnWriter::DoubleColumnWriter($w) => $body,
                ColumnWriter::ByteArrayColumnWriter($w) => $body,
                ColumnWriter::FixedLenByteArrayColumnWriter($w) => $body,
            },
        }
    };
}

fn set_page_boundary_action(writer: &mut ArrowColumnWriterImpl, action: PageBoundaryAction) {
    dispatch_writer!(writer, w, w.set_page_boundary_action(action))
}

fn pin_encoding(
    writer: &mut ArrowColumnWriterImpl,
    encoding: Encoding,
    descr: &ColumnDescPtr,
) -> Result<()> {
    dispatch_writer!(writer, w, w.encoder_mut().pin_encoding(encoding, descr))
}

fn install_accumulators(writer: &mut ArrowColumnWriterImpl, accumulators: ValueAccumulators) {
    dispatch_writer!(
        writer,
        w,
        w.encoder_mut().install_value_accumulators(accumulators)
    )
}

fn take_accumulators(writer: &mut ArrowColumnWriterImpl) -> ValueAccumulators {
    dispatch_writer!(writer, w, w.encoder_mut().take_value_accumulators())
}

fn page_boundary_reached(writer: &ArrowColumnWriterImpl) -> bool {
    dispatch_writer!(writer, w, w.page_boundary_reached())
}

fn take_dictionary(writer: &mut ArrowColumnWriterImpl) -> Option<Box<dyn DynDictionary>> {
    dispatch_writer!(writer, w, w.encoder_mut().take_dictionary())
}

fn install_dictionary(
    writer: &mut ArrowColumnWriterImpl,
    dictionary: Box<dyn DynDictionary>,
) -> Result<()> {
    dispatch_writer!(writer, w, w.encoder_mut().install_dictionary(dictionary))
}

fn write_dictionary_page(writer: &mut ArrowColumnWriterImpl, page: DictionaryPage) -> Result<()> {
    dispatch_writer!(writer, w, w.write_dictionary_page_data(page))
}

/// Build a column writer for `descr`, choosing the encoder the arrow writer
/// would choose for the same leaf.
///
/// This is the only place the module builds a writer. The builder lifts the
/// chunk-level state out of the one it keeps ([`ColumnChunkBuilder::new`]) and
/// throws it away for the throwaway ones ([`ColumnChunkBuilder::candidate_writer`]).
fn build_writer(
    descr: &ColumnDescPtr,
    props: &WriterPropertiesPtr,
    page_writer: Box<dyn PageWriter + 'static>,
) -> Result<ArrowColumnWriterImpl> {
    macro_rules! typed {
        ($t:ty, $variant:ident) => {{
            let encoder = <crate::column::writer::encoder::ColumnValueEncoderImpl<$t> as ColumnValueEncoder>::try_new(descr, props)?;
            ArrowColumnWriterImpl::Column(ColumnWriter::$variant(
                GenericColumnWriter::new_with_encoder(
                    descr.clone(),
                    props.clone(),
                    page_writer,
                    encoder,
                ),
            ))
        }};
    }

    Ok(match descr.physical_type() {
        // `BYTE_ARRAY` leaves go through the arrow byte-array encoder, matching
        // `ArrowColumnWriterFactory`. A `FIXED_LEN_BYTE_ARRAY` leaf goes to the
        // generic writer; the factory routes an *arrow dictionary* of
        // fixed-size-binary to the byte-array encoder instead, which the generic
        // writer handles by materializing the dictionary. Same values, same
        // output, one fewer specialisation.
        Type::BYTE_ARRAY => {
            let encoder = <ByteArrayEncoder as ColumnValueEncoder>::try_new(descr, props)?;
            ArrowColumnWriterImpl::ByteArray(GenericColumnWriter::new_with_encoder(
                descr.clone(),
                props.clone(),
                page_writer,
                encoder,
            ))
        }
        Type::BOOLEAN => typed!(BoolType, BoolColumnWriter),
        Type::INT32 => typed!(Int32Type, Int32ColumnWriter),
        Type::INT64 => typed!(Int64Type, Int64ColumnWriter),
        Type::INT96 => typed!(Int96Type, Int96ColumnWriter),
        Type::FLOAT => typed!(FloatType, FloatColumnWriter),
        Type::DOUBLE => typed!(DoubleType, DoubleColumnWriter),
        Type::FIXED_LEN_BYTE_ARRAY => {
            typed!(FixedLenByteArrayType, FixedLenByteArrayColumnWriter)
        }
    })
}

/// Assemble the writer's buffered values into a sealed [`EncodedPage`].
fn assemble(writer: &mut ArrowColumnWriterImpl) -> Result<EncodedPage> {
    macro_rules! seal {
        ($w:expr, $variant:ident) => {{
            let prepared = $w.assemble_data_page()?;
            EncodedPage::new(PreparedPage::$variant(prepared))
        }};
    }
    Ok(match writer {
        ArrowColumnWriterImpl::ByteArray(w) => seal!(w, ByteArray),
        ArrowColumnWriterImpl::Column(c) => match c {
            ColumnWriter::BoolColumnWriter(w) => seal!(w, Bool),
            ColumnWriter::Int32ColumnWriter(w) => seal!(w, Int32),
            ColumnWriter::Int64ColumnWriter(w) => seal!(w, Int64),
            ColumnWriter::Int96ColumnWriter(w) => seal!(w, Int96),
            ColumnWriter::FloatColumnWriter(w) => seal!(w, Float),
            ColumnWriter::DoubleColumnWriter(w) => seal!(w, Double),
            ColumnWriter::ByteArrayColumnWriter(w) => seal!(w, ByteArray),
            ColumnWriter::FixedLenByteArrayColumnWriter(w) => seal!(w, FixedLenByteArray),
        },
    })
}

/// Commit a sealed page to the chunk's real writer.
fn commit(writer: &mut ArrowColumnWriterImpl, page: EncodedPage) -> Result<()> {
    let mismatch = || general_err!("this page was encoded for a different column's physical type");
    macro_rules! put {
        ($w:expr, $variant:ident) => {
            match page.prepared {
                PreparedPage::$variant(p) => $w.commit_data_page(p),
                _ => Err(mismatch()),
            }
        };
    }
    match writer {
        ArrowColumnWriterImpl::ByteArray(w) => put!(w, ByteArray),
        ArrowColumnWriterImpl::Column(c) => match c {
            ColumnWriter::BoolColumnWriter(w) => put!(w, Bool),
            ColumnWriter::Int32ColumnWriter(w) => put!(w, Int32),
            ColumnWriter::Int64ColumnWriter(w) => put!(w, Int64),
            ColumnWriter::Int96ColumnWriter(w) => put!(w, Int96),
            ColumnWriter::FloatColumnWriter(w) => put!(w, Float),
            ColumnWriter::DoubleColumnWriter(w) => put!(w, Double),
            ColumnWriter::ByteArrayColumnWriter(w) => put!(w, ByteArray),
            ColumnWriter::FixedLenByteArrayColumnWriter(w) => put!(w, FixedLenByteArray),
        },
    }
}

impl EncodedPage {
    /// Lift a `PreparedDataPage` into the type-erased, inspectable form.
    fn new(prepared: PreparedPage) -> Self {
        macro_rules! describe {
            ($p:expr) => {{
                (
                    $p.encoding(),
                    $p.metrics_num_values(),
                    $p.metrics_num_rows(),
                    $p.compressed_len(),
                    $p.uncompressed_len(),
                )
            }};
        }
        let (encoding, num_values, num_rows, compressed_len, uncompressed_len) = match &prepared {
            PreparedPage::Bool(p) => describe!(p),
            PreparedPage::Int32(p) => describe!(p),
            PreparedPage::Int64(p) => describe!(p),
            PreparedPage::Int96(p) => describe!(p),
            PreparedPage::Float(p) => describe!(p),
            PreparedPage::Double(p) => describe!(p),
            PreparedPage::ByteArray(p) => describe!(p),
            PreparedPage::FixedLenByteArray(p) => describe!(p),
        };
        Self {
            prepared,
            encoding,
            num_values,
            num_rows,
            compressed_len,
            uncompressed_len,
            dictionary_growth: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::ArrowSchemaConverter;
    use crate::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use crate::arrow::arrow_writer::{ArrowWriter, compute_leaves};
    use crate::basic::Encoding;
    use crate::file::properties::{EnabledStatistics, WriterProperties};
    use crate::file::writer::SerializedFileWriter;
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use std::sync::Arc;

    fn string_batch(values: &[&str]) -> (Arc<Schema>, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let array: ArrayRef = Arc::new(StringArray::from(values.to_vec()));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        (schema, batch)
    }

    fn int_batch(values: Vec<i64>) -> (Arc<Schema>, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int64, false)]));
        let array: ArrayRef = Arc::new(Int64Array::from(values));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        (schema, batch)
    }

    /// Write `batch` through the page-grain API, choosing a page from
    /// `candidates` with `pick`.
    fn write_page_grain(
        schema: &Arc<Schema>,
        batches: &[RecordBatch],
        props: Arc<WriterProperties>,
        mut plan: impl FnMut(&ColumnChunkBuilder) -> Vec<Candidate>,
        mut pick: impl FnMut(&[EncodedPage]) -> usize,
    ) -> Bytes {
        let parquet_schema = ArrowSchemaConverter::new()
            .with_coerce_types(props.coerce_types())
            .convert(schema)
            .unwrap();
        // `ArrowWriter` stamps the encoded arrow schema into the file metadata;
        // do the same so a byte comparison is about the pages, not the footer.
        let props = {
            let mut p = (*props).clone();
            crate::arrow::schema::add_encoded_arrow_schema_to_metadata(schema, &mut p);
            Arc::new(p)
        };
        let mut out = Vec::new();
        let mut file_writer =
            SerializedFileWriter::new(&mut out, parquet_schema.root_schema_ptr(), props.clone())
                .unwrap();

        let descr = parquet_schema.columns()[0].clone();
        let field = schema.field(0);

        let mut row_group = file_writer.next_row_group().unwrap();
        let mut chunk = ColumnChunkBuilder::new(descr, props).unwrap();
        for batch in batches {
            for leaf in compute_leaves(field, batch.column(0)).unwrap() {
                let mut cursor = chunk.cursor(&leaf);
                while !cursor.is_empty() {
                    let candidates = plan(&chunk);
                    let (first, alternatives) = candidates.split_first().unwrap();
                    let pages = chunk
                        .encode_page_alternatives(&mut cursor, *first, alternatives)
                        .unwrap();
                    let winner = pick(&pages);
                    chunk
                        .append(pages.into_iter().nth(winner).unwrap())
                        .unwrap();
                }
            }
        }
        chunk
            .close()
            .unwrap()
            .append_to_row_group(&mut row_group)
            .unwrap();
        row_group.close().unwrap();
        file_writer.close().unwrap();
        Bytes::from(out)
    }

    fn write_arrow(
        schema: &Arc<Schema>,
        batches: &[RecordBatch],
        props: Arc<WriterProperties>,
    ) -> Bytes {
        let mut out = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut out, schema.clone(), Some((*props).clone())).unwrap();
        for batch in batches {
            writer.write(batch).unwrap();
        }
        writer.close().unwrap();
        Bytes::from(out)
    }

    fn read_back(bytes: Bytes) -> Vec<RecordBatch> {
        ParquetRecordBatchReaderBuilder::try_new(bytes)
            .unwrap()
            .build()
            .unwrap()
            .map(|b| b.unwrap())
            .collect()
    }

    /// A single-candidate harness run must be byte-identical to `ArrowWriter`
    /// at the same properties. This is the acceptance test for the claim that
    /// the page-grain API reuses the existing page assembly rather than
    /// reimplementing it: same encoder, same budget, same split points, same
    /// bytes.
    #[test]
    fn single_candidate_is_byte_identical_to_arrow_writer() {
        let values: Vec<String> = (0..40_000).map(|i| format!("v{}", i % 512)).collect();
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let (schema, batch) = string_batch(&refs);
        let props = Arc::new(
            WriterProperties::builder()
                .set_statistics_enabled(EnabledStatistics::Page)
                .build(),
        );

        let ours = write_page_grain(
            &schema,
            std::slice::from_ref(&batch),
            props.clone(),
            |_| vec![Candidate::Dictionary],
            |_| 0,
        );
        let theirs = write_arrow(&schema, std::slice::from_ref(&batch), props);
        assert_eq!(ours, theirs, "page-grain output diverged from ArrowWriter");
    }

    #[test]
    fn single_candidate_int64_is_byte_identical_to_arrow_writer() {
        let (schema, batch) = int_batch((0..60_000).map(|i| (i % 1000) as i64).collect());
        let props = Arc::new(
            WriterProperties::builder()
                .set_statistics_enabled(EnabledStatistics::Page)
                .build(),
        );
        let ours = write_page_grain(
            &schema,
            std::slice::from_ref(&batch),
            props.clone(),
            |_| vec![Candidate::Dictionary],
            |_| 0,
        );
        let theirs = write_arrow(&schema, std::slice::from_ref(&batch), props);
        assert_eq!(ours, theirs);
    }

    /// Nullable data: definition levels flow through the same seams, and the
    /// output must still be byte-identical to `ArrowWriter`.
    #[test]
    fn single_candidate_nullable_is_byte_identical_to_arrow_writer() {
        let values: Vec<Option<String>> = (0..40_000)
            .map(|i| (i % 3 != 0).then(|| format!("v{}", i % 512)))
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let array: ArrayRef = Arc::new(StringArray::from(values));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        let props = Arc::new(
            WriterProperties::builder()
                .set_statistics_enabled(EnabledStatistics::Page)
                .set_data_page_row_count_limit(3_000)
                .build(),
        );

        let ours = write_page_grain(
            &schema,
            std::slice::from_ref(&batch),
            props.clone(),
            |_| vec![Candidate::Dictionary],
            |_| 0,
        );
        let theirs = write_arrow(&schema, std::slice::from_ref(&batch), props);
        assert_eq!(ours, theirs, "page-grain output diverged from ArrowWriter");
    }

    /// Racing candidates and picking the smallest must still round-trip
    /// exactly, and must actually pick more than one encoding across the file.
    #[test]
    fn raced_pages_round_trip() {
        let values: Vec<String> = (0..30_000)
            .map(|i| {
                if i < 15_000 {
                    format!("low{}", i % 8)
                } else {
                    format!("high-cardinality-value-{i}")
                }
            })
            .collect();
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let (schema, batch) = string_batch(&refs);
        // Small pages so the file has enough of them for the harness's
        // decisions to be visible.
        let props = Arc::new(
            WriterProperties::builder()
                .set_data_page_row_count_limit(2_000)
                .build(),
        );

        let mut encodings = Vec::new();
        let bytes = write_page_grain(
            &schema,
            std::slice::from_ref(&batch),
            props,
            |chunk| {
                // Dictionary-watching: stop offering the dictionary once it
                // has interned enough distinct values to have stopped paying
                // for itself.
                match chunk.dictionary() {
                    Some(d) if d.entries() < 4_000 => {
                        vec![Candidate::Dictionary, Candidate::Encoding(Encoding::PLAIN)]
                    }
                    _ => vec![
                        Candidate::Encoding(Encoding::PLAIN),
                        Candidate::Encoding(Encoding::DELTA_BYTE_ARRAY),
                    ],
                }
            },
            |pages| {
                let best = pages
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, p)| p.compressed_len())
                    .map(|(i, _)| i)
                    .unwrap();
                encodings.push(pages[best].encoding());
                best
            },
        );

        assert!(
            encodings
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                > 1,
            "expected the race to land on more than one encoding, got {encodings:?}"
        );

        let read = read_back(bytes);
        let total: usize = read.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 30_000);
        let mut seen = Vec::new();
        for b in &read {
            let col = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            seen.extend(col.iter().map(|v| v.unwrap().to_string()));
        }
        assert_eq!(seen, values);
    }

    /// Falling back from the dictionary is expressible; falling back *to* it is
    /// not representable, and the builder says so rather than producing an
    /// invalid chunk.
    #[test]
    fn dictionary_after_fallback_is_rejected() {
        let values: Vec<String> = (0..60_000).map(|i| format!("v{}", i % 64)).collect();
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let (schema, batch) = string_batch(&refs);
        let props = Arc::new(WriterProperties::builder().build());

        let parquet_schema = ArrowSchemaConverter::new().convert(&schema).unwrap();
        let descr = parquet_schema.columns()[0].clone();
        let mut chunk = ColumnChunkBuilder::new(descr, props).unwrap();

        let leaf = compute_leaves(schema.field(0), batch.column(0))
            .unwrap()
            .remove(0);
        let mut cursor = chunk.cursor(&leaf);

        // Land on PLAIN first.
        let page = chunk
            .encode_page(&mut cursor, Candidate::Encoding(Encoding::PLAIN))
            .unwrap();
        chunk.append(page).unwrap();

        // The dictionary is gone, so even asking for it fails.
        assert!(chunk.dictionary().is_none());
        let err = chunk
            .encode_page(&mut cursor, Candidate::Dictionary)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dictionary"), "{err}");
    }

    /// The other half of dictionary-then-fallback: once indices pages exist the
    /// dictionary object is retained (its page is owed at close), but asking to
    /// encode with it again must fail at *encode* time, before the cursor moves.
    #[test]
    fn dictionary_after_owed_fallback_is_rejected_before_encoding() {
        let values: Vec<String> = (0..6_000).map(|i| format!("v{}", i % 64)).collect();
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let (schema, batch) = string_batch(&refs);
        let props = Arc::new(
            WriterProperties::builder()
                .set_data_page_row_count_limit(2_000)
                .build(),
        );

        let parquet_schema = ArrowSchemaConverter::new().convert(&schema).unwrap();
        let descr = parquet_schema.columns()[0].clone();
        let mut chunk = ColumnChunkBuilder::new(descr, props).unwrap();
        let leaf = compute_leaves(schema.field(0), batch.column(0))
            .unwrap()
            .remove(0);
        let mut cursor = chunk.cursor(&leaf);

        // An indices page first (the dictionary page is now owed), then a
        // fallback page.
        let page = chunk
            .encode_page(&mut cursor, Candidate::Dictionary)
            .unwrap();
        assert!(page.is_dictionary_indices());
        chunk.append(page).unwrap();
        let page = chunk
            .encode_page(&mut cursor, Candidate::Encoding(Encoding::PLAIN))
            .unwrap();
        chunk.append(page).unwrap();

        // Asking for the dictionary again fails without consuming the cursor.
        let before = cursor.levels_remaining();
        let err = chunk
            .encode_page(&mut cursor, Candidate::Dictionary)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dictionary"), "{err}");
        assert_eq!(
            cursor.levels_remaining(),
            before,
            "the failed encode must not move the cursor"
        );

        // The chunk is still usable and still owes (and writes) its dictionary page.
        while !cursor.is_empty() {
            let page = chunk
                .encode_page(&mut cursor, Candidate::Encoding(Encoding::PLAIN))
                .unwrap();
            chunk.append(page).unwrap();
        }
        chunk.close().unwrap();
    }

    /// The two argument-shape rejections: an exhausted cursor, and
    /// `Candidate::Dictionary` offered as an alternative.
    #[test]
    fn invalid_encode_arguments_are_rejected() {
        let (schema, batch) = int_batch((0..100i64).collect());
        let props = Arc::new(WriterProperties::builder().build());
        let parquet_schema = ArrowSchemaConverter::new().convert(&schema).unwrap();
        let descr = parquet_schema.columns()[0].clone();
        let mut chunk = ColumnChunkBuilder::new(descr, props).unwrap();
        let leaf = compute_leaves(schema.field(0), batch.column(0))
            .unwrap()
            .remove(0);
        let mut cursor = chunk.cursor(&leaf);

        let err = chunk
            .encode_page_alternatives(&mut cursor, Candidate::Dictionary, &[Candidate::Dictionary])
            .unwrap_err()
            .to_string();
        assert!(err.contains("alternative"), "{err}");

        let page = chunk
            .encode_page(&mut cursor, Candidate::Dictionary)
            .unwrap();
        assert!(cursor.is_empty());
        chunk.append(page).unwrap();
        let err = chunk
            .encode_page(&mut cursor, Candidate::Dictionary)
            .unwrap_err()
            .to_string();
        assert!(err.contains("exhausted"), "{err}");
    }

    /// `dictionary_growth` is what makes candidates comparable: the indices
    /// page is cheap only because its bytes went into the dictionary, and the
    /// alternative that never touches the dictionary is charged nothing.
    #[test]
    fn dictionary_growth_is_charged_to_the_indices_page() {
        let values: Vec<String> = (0..20_000).map(|i| format!("value-number-{i}")).collect();
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let (schema, batch) = string_batch(&refs);
        let props = Arc::new(
            WriterProperties::builder()
                .set_data_page_row_count_limit(2_000)
                .build(),
        );

        let parquet_schema = ArrowSchemaConverter::new().convert(&schema).unwrap();
        let descr = parquet_schema.columns()[0].clone();
        let mut chunk = ColumnChunkBuilder::new(descr, props).unwrap();
        let leaf = compute_leaves(schema.field(0), batch.column(0))
            .unwrap()
            .remove(0);
        let mut cursor = chunk.cursor(&leaf);

        let pages = chunk
            .encode_page_alternatives(
                &mut cursor,
                Candidate::Dictionary,
                &[Candidate::Encoding(Encoding::PLAIN)],
            )
            .unwrap();

        assert_eq!(pages.len(), 2);
        let dictionary_page = &pages[0];
        let plain_page = &pages[1];
        assert!(dictionary_page.is_dictionary_indices());
        assert_eq!(plain_page.dictionary_growth(), 0);
        // Every value in this page is distinct, so the dictionary grew by
        // roughly the whole page's worth of bytes, and the raw indices page is
        // far smaller than the PLAIN page it is being compared against.
        assert!(
            dictionary_page.dictionary_growth() > dictionary_page.compressed_len(),
            "{dictionary_page:?}"
        );
        assert!(
            dictionary_page.compressed_len() + dictionary_page.dictionary_growth()
                > plain_page.compressed_len(),
            "charging the dictionary must make the indices page the loser here"
        );

        // A page whose boundary was decided without the dictionary is charged
        // nothing, even when the chunk still has one.
        let page = chunk
            .encode_page(&mut cursor, Candidate::Encoding(Encoding::PLAIN))
            .unwrap();
        assert_eq!(page.dictionary_growth(), 0);
    }

    /// The dictionary page is written first even though it is produced last.
    #[test]
    fn dictionary_page_is_written_first() {
        let values: Vec<String> = (0..8_000).map(|i| format!("v{}", i % 32)).collect();
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let (schema, batch) = string_batch(&refs);
        let props = Arc::new(WriterProperties::builder().build());

        let bytes = write_page_grain(
            &schema,
            std::slice::from_ref(&batch),
            props,
            |_| vec![Candidate::Dictionary],
            |_| 0,
        );

        let metadata = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .unwrap()
            .metadata()
            .clone();
        let column = metadata.row_group(0).column(0);
        let dict_offset = column.dictionary_page_offset().unwrap();
        assert!(
            dict_offset < column.data_page_offset(),
            "dictionary page must precede the data pages"
        );
        assert_eq!(
            read_back(bytes).iter().map(|b| b.num_rows()).sum::<usize>(),
            8_000
        );
    }

    /// The crux: chunk-level accumulators are fed by values, and racing K
    /// candidates must still feed each value exactly once.
    ///
    /// This test pins the failure mode a per-candidate design would have. The
    /// pacer here is the dictionary candidate and it *always loses* — every
    /// appended page is the pinned PLAIN one. If the bloom filter travelled
    /// with a candidate and only the winner's were kept, the chunk's filter
    /// would be empty. It must instead contain every value, and be identical to
    /// the one `ArrowWriter` builds for the same data.
    #[test]
    fn bloom_filter_survives_the_boundary_page_always_losing() {
        let values: Vec<String> = (0..40_000).map(|i| format!("value-{i}")).collect();
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let (schema, batch) = string_batch(&refs);
        let props = Arc::new(
            WriterProperties::builder()
                .set_bloom_filter_enabled(true)
                .set_data_page_row_count_limit(2_000)
                .build(),
        );

        let mut appended = Vec::new();
        let bytes = write_page_grain(
            &schema,
            std::slice::from_ref(&batch),
            props.clone(),
            |chunk| {
                // The page boundary is decided by the dictionary while there
                // is one, and by DELTA_BYTE_ARRAY afterwards; either way it is
                // index 0, and never the page that gets appended.
                if chunk.dictionary().is_some() {
                    vec![Candidate::Dictionary, Candidate::Encoding(Encoding::PLAIN)]
                } else {
                    vec![
                        Candidate::Encoding(Encoding::DELTA_BYTE_ARRAY),
                        Candidate::Encoding(Encoding::PLAIN),
                    ]
                }
            },
            |pages| {
                // Always take the alternative, never the page whose candidate
                // decided the boundary.
                assert_eq!(pages.len(), 2);
                appended.push(pages[1].encoding());
                1
            },
        );
        assert!(
            appended.len() > 10,
            "expected many pages, got {}",
            appended.len()
        );
        assert!(appended.iter().all(|e| *e == Encoding::PLAIN));

        let ours = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
        let filter = ours
            .get_row_group_column_bloom_filter(0, 0)
            .unwrap()
            .expect("chunk should have a bloom filter");
        for value in &values {
            assert!(
                filter.check(&ByteArray::from(value.as_str())),
                "bloom filter is missing {value}: a value fed by a rejected page was lost"
            );
        }

        // And it is the same filter the ordinary writer builds.
        let theirs = write_arrow(&schema, std::slice::from_ref(&batch), props);
        let reference = ParquetRecordBatchReaderBuilder::try_new(theirs)
            .unwrap()
            .get_row_group_column_bloom_filter(0, 0)
            .unwrap()
            .unwrap();
        for value in values.iter().take(2_000) {
            let v = ByteArray::from(value.as_str());
            assert_eq!(filter.check(&v), reference.check(&v));
        }
    }

    /// A repeated column: pages must break only at record boundaries, and the
    /// round trip must be exact.
    #[test]
    fn repeated_column_pages_never_split_a_record() {
        use arrow_array::{Int32Array, ListArray};
        use arrow_buffer::OffsetBuffer;

        // 20_000 records of 5 values each.
        let n = 20_000;
        let values = Int32Array::from((0..(n * 5) as i32).collect::<Vec<_>>());
        let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(5usize, n));
        let field = Arc::new(Field::new("item", DataType::Int32, false));
        let list = ListArray::new(field.clone(), offsets, Arc::new(values), None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "l",
            DataType::List(field),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list) as ArrayRef]).unwrap();

        let props = Arc::new(
            WriterProperties::builder()
                .set_data_page_row_count_limit(1_500)
                .build(),
        );

        let mut page_rows = Vec::new();
        let bytes = write_page_grain(
            &schema,
            std::slice::from_ref(&batch),
            props.clone(),
            |_| vec![Candidate::Dictionary],
            |pages| {
                // Every page holds whole records: 5 levels per record.
                assert!(pages[0].num_values().is_multiple_of(5));
                page_rows.push(pages[0].num_rows());
                0
            },
        );
        assert!(page_rows.len() > 5);
        assert_eq!(page_rows.iter().map(|r| *r as usize).sum::<usize>(), n);

        let read = read_back(bytes);
        assert_eq!(read.iter().map(|b| b.num_rows()).sum::<usize>(), n);

        // And identical to what the ordinary writer produces.
        let theirs = write_arrow(&schema, std::slice::from_ref(&batch), props);
        let ours2 = write_page_grain(
            &schema,
            std::slice::from_ref(&batch),
            Arc::new(
                WriterProperties::builder()
                    .set_data_page_row_count_limit(1_500)
                    .build(),
            ),
            |_| vec![Candidate::Dictionary],
            |_| 0,
        );
        assert_eq!(ours2, theirs);
    }

    /// A repeated column whose records do not divide the pacing window evenly:
    /// every page boundary must still land on a record boundary, including the
    /// boundaries picked after the pacer has walked more than one window into
    /// the leaf.
    #[test]
    fn variable_length_records_never_split_across_pages() {
        use arrow_array::{Int32Array, ListArray};
        use arrow_buffer::OffsetBuffer;

        // One record of 4 values, then records of 3: record starts fall at
        // 0, 4, 7, 10, ... so windows aligned to round numbers end mid-record.
        let n = 2_000usize;
        let lengths: Vec<usize> = (0..n).map(|i| if i == 0 { 4 } else { 3 }).collect();
        let total: usize = lengths.iter().sum();
        let values = Int32Array::from((0..total as i32).collect::<Vec<_>>());
        let offsets = OffsetBuffer::from_lengths(lengths.iter().copied());
        let field = Arc::new(Field::new("item", DataType::Int32, false));
        let list = ListArray::new(field.clone(), offsets, Arc::new(values), None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "l",
            DataType::List(field),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list) as ArrayRef]).unwrap();

        // A row budget that trips exactly at the end of the pacer's second
        // window, which ends mid-record when window offsets are miscounted.
        let props = Arc::new(
            WriterProperties::builder()
                .set_write_batch_size(100)
                .set_data_page_row_count_limit(135)
                .build(),
        );

        let mut level_boundaries = Vec::new();
        let mut cumulative = 0u64;
        let bytes = write_page_grain(
            &schema,
            std::slice::from_ref(&batch),
            props,
            |_| vec![Candidate::Dictionary],
            |pages| {
                cumulative += u64::from(pages[0].num_values());
                level_boundaries.push(cumulative);
                0
            },
        );

        // Record starts sit at level 0 and 4 + 3k, so a page boundary is legal
        // only at 4 + 3k (or the very end of the leaf).
        for boundary in &level_boundaries {
            assert!(
                *boundary >= 4 && (*boundary - 4).is_multiple_of(3),
                "page boundary at level {boundary} splits a record; boundaries: {level_boundaries:?}"
            );
        }
        assert_eq!(*level_boundaries.last().unwrap() as usize, total);

        let read = read_back(bytes);
        assert_eq!(read.iter().map(|b| b.num_rows()).sum::<usize>(), n);
    }

    /// Index and chunk metadata are derived by the builder, so they must match
    /// what `ArrowWriter` produces for the same shape.
    #[test]
    fn index_and_metadata_match_arrow_writer() {
        let (schema, batch) = int_batch((0..50_000i64).collect());
        let props = Arc::new(
            WriterProperties::builder()
                .set_statistics_enabled(EnabledStatistics::Page)
                .build(),
        );
        let ours = write_page_grain(
            &schema,
            std::slice::from_ref(&batch),
            props.clone(),
            |_| vec![Candidate::Dictionary],
            |_| 0,
        );
        let theirs = write_arrow(&schema, std::slice::from_ref(&batch), props);

        let load = |bytes: Bytes| {
            ParquetRecordBatchReaderBuilder::try_new(bytes)
                .unwrap()
                .metadata()
                .clone()
        };
        let (a, b) = (load(ours), load(theirs));
        let (ca, cb) = (a.row_group(0).column(0), b.row_group(0).column(0));
        assert_eq!(
            ca.encodings().collect::<Vec<_>>(),
            cb.encodings().collect::<Vec<_>>()
        );
        assert_eq!(ca.page_encoding_stats(), cb.page_encoding_stats());
        assert_eq!(ca.num_values(), cb.num_values());
        assert_eq!(ca.statistics(), cb.statistics());
        assert_eq!(ca.compressed_size(), cb.compressed_size());
        assert_eq!(ca.uncompressed_size(), cb.uncompressed_size());
    }
}

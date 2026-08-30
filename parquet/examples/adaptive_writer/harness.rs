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

//! An adaptive parquet writer that chooses each column's encoding by measuring
//! it, rather than by being told.
//!
//! [`AdaptiveWriter`] is a drop-in shape over [`ArrowWriter`]: `write` a batch,
//! `flush` to end a row group, `close`. What it does differently is route every
//! leaf column, every row group, down one of two paths:
//!
//! * A leaf that is **still deciding** is written with
//!   [`ColumnChunkBuilder`](parquet::arrow::arrow_writer::page_grain::ColumnChunkBuilder).
//!   Each page is encoded several ways, the results are measured, and the
//!   cheapest is committed. This costs extra encoding, and it is the only way
//!   to change encoding *within* a column chunk.
//! * A leaf that has **settled** is written with an ordinary
//!   [`ArrowColumnWriter`], configured with the encoding it settled on. This is
//!   the library's normal write path at its normal speed, and it is where most
//!   leaves spend most of the file.
//!
//! Both paths produce an [`ArrowColumnChunk`], so the two kinds of chunk are
//! appended to one row group in leaf order, and both allocate their pages from
//! the file writer's own page store.
//!
//! # The two halves of this file
//!
//! **Policy** is [`LeafPolicy`] and the functions around it: which encodings to
//! try, what a page costs, when a leaf has seen enough to settle, when to look
//! again, and when to give up on a dictionary. It is the half to rewrite for
//! different data, and nothing in it is privileged: it only ever reads
//! measurements the library reports.
//!
//! **Plumbing** is [`AdaptiveWriter`]: opening a row group, routing leaves,
//! feeding batches, closing chunks and appending them in order. It is the half
//! to copy.

use std::io::Write;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use parquet::arrow::arrow_writer::page_grain::{Candidate, ColumnChunkBuilder, EncodedPage};
use parquet::arrow::arrow_writer::{
    ArrowColumnChunk, ArrowColumnWriter, ArrowLeafColumn, ArrowRowGroupWriterFactory, ArrowWriter,
    compute_leaves,
};
use parquet::basic::{Encoding, Type as PhysicalType};
use parquet::errors::Result;
use parquet::file::properties::{
    DictionaryFallback, WriterProperties, WriterPropertiesBuilder, WriterPropertiesPtr,
};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::{ColumnDescPtr, ColumnPath};

// ===========================================================================
// Policy
//
// Everything from here to the plumbing section is a decision about data, and a
// different writer would decide differently. The constants are the ones tuned
// against the bakeoff datasets.
// ===========================================================================

/// Dictionary page size limit for the columns this writer decides for,
/// deliberately below the 1 MiB default so a chunk can actually reach it.
pub const DICT_PAGE_SIZE_LIMIT: usize = 64 * 1024;
/// `worth_ratio` for [`DictionaryFallback::WhenProfitable`] on the settled path.
pub const DICT_WORTH_RATIO: f64 = 0.25;
/// Absolute cap on a retained dictionary page.
pub const DICT_MAX_PAGE_SIZE: usize = 8 * 1024 * 1024;
/// Two candidates within this fraction of each other are a tie, and the tie is
/// broken towards the page that is cheaper to read.
pub const NEAR_TIE: f64 = 0.02;
/// A settled leaf tries everything again this often, in row groups, so that a
/// file whose data changes shape is not stuck with an early answer.
pub const REOPEN_EVERY: usize = 4;
/// Pages that must agree before a leaf stops trying alternatives.
const PAGES_TO_SETTLE: usize = 2;
/// A dictionary is watched only after this many values have gone through it;
/// before that its entry count says more about the data's start than its shape.
const DICT_WATCH_AFTER: u64 = 50_000;

/// Whether this leaf is worth deciding for at all.
///
/// Repeated leaves and the types with no useful alternative encoding
/// (`BOOLEAN`, `FIXED_LEN_BYTE_ARRAY`, `INT96`) are written exactly as the
/// file's properties say, on the ordinary path.
pub fn is_decidable(descr: &ColumnDescPtr) -> bool {
    if descr.max_rep_level() > 0 {
        return false;
    }
    matches!(
        descr.physical_type(),
        PhysicalType::INT32 | PhysicalType::INT64 | PhysicalType::BYTE_ARRAY | PhysicalType::DOUBLE
    )
}

/// What a leaf is currently doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// No decision yet: keep encoding pages more than one way.
    Undecided,
    /// Settled on the chunk's dictionary.
    Dictionary,
    /// Settled on a fixed encoding.
    Fixed(Encoding),
}

/// Relative cost of decoding, lowest first. Only ever breaks a near-tie, so
/// that two encodings of the same size resolve in the reader's favour.
fn decode_cost(encoding: Encoding) -> u8 {
    match encoding {
        Encoding::PLAIN => 0,
        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => 1,
        Encoding::DELTA_BINARY_PACKED | Encoding::DELTA_LENGTH_BYTE_ARRAY => 2,
        Encoding::DELTA_BYTE_ARRAY => 3,
        _ => 4,
    }
}

/// Whether an encoding indexes into the column chunk's dictionary.
fn is_dictionary_indices(encoding: Encoding) -> bool {
    matches!(
        encoding,
        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
    )
}

/// What a page really costs: its own compressed bytes, plus whatever it added
/// to the chunk's dictionary, which is paid later in the dictionary page.
///
/// Comparing `compressed_len` alone would pick the dictionary essentially
/// always, because an indices page is small precisely to the extent that it
/// pushed bytes somewhere this comparison cannot see.
fn page_cost(page: &EncodedPage) -> usize {
    page.compressed_len() + page.dictionary_growth()
}

/// Index of the winning page: the cheapest, unless something within
/// [`NEAR_TIE`] of it is cheaper to decode.
fn pick_page(pages: &[EncodedPage]) -> usize {
    let cheapest = pages.iter().map(page_cost).min().unwrap_or(0);
    let budget = (cheapest as f64 * (1.0 + NEAR_TIE)) as usize;
    pages
        .iter()
        .enumerate()
        .filter(|(_, p)| page_cost(p) <= budget)
        .min_by_key(|(_, p)| (decode_cost(p.encoding()), page_cost(p)))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// What one committed page measured, kept to inform the next one.
struct PageReport {
    encoding: Encoding,
    compressed: usize,
    uncompressed: usize,
    is_dictionary: bool,
}

/// One leaf column's policy, and what it has learned across row groups.
pub struct LeafPolicy {
    /// The choice carried forward from the previous row group.
    learned: Choice,
    /// The encodings to try against the dictionary.
    alternatives: Vec<Encoding>,
    /// Keep choosing per page from the previous page's numbers even after
    /// settling. Used for data whose answer drifts (floats) rather than
    /// switching outright.
    adapt_per_page: bool,
    /// This leaf is written as the properties say, and decides nothing.
    passthrough: bool,
    /// Dictionary page size limit for this leaf, from the properties.
    dictionary_limit: usize,
    /// Route every leaf through the page grain, whatever it has settled on.
    /// The point of this writer is that it does not, but measuring the
    /// difference needs a way to ask for it.
    pub always_page_grain: bool,
    row_groups: usize,
    /// Reporting only: pages encoded, and pages encoded more than one way.
    pub total_pages: usize,
    pub multi_pages: usize,
}

impl LeafPolicy {
    pub fn new(descr: &ColumnDescPtr, props: &WriterProperties) -> Self {
        let decidable = is_decidable(descr);
        let alternatives = match descr.physical_type() {
            PhysicalType::BYTE_ARRAY => vec![Encoding::PLAIN, Encoding::DELTA_BYTE_ARRAY],
            PhysicalType::INT32 | PhysicalType::INT64 => {
                vec![Encoding::PLAIN, Encoding::DELTA_BINARY_PACKED]
            }
            _ => vec![Encoding::PLAIN],
        };
        Self {
            dictionary_limit: props.column_dictionary_page_size_limit(descr.path()),
            learned: Choice::Undecided,
            alternatives,
            // A float column has no second encoding worth switching to, so it
            // settles quickly; but how well PLAIN does against the dictionary
            // drifts with the data, so it keeps re-deciding per page.
            adapt_per_page: decidable && descr.physical_type() == PhysicalType::DOUBLE,
            passthrough: !decidable,
            always_page_grain: false,
            row_groups: 0,
            total_pages: 0,
            multi_pages: 0,
        }
    }

    /// Whether this leaf takes the page-grain path for the row group about to
    /// be written: this is the routing rule, and the whole hybrid.
    ///
    /// A leaf needs the page grain while it is actively deciding: it has never
    /// settled, it is trying everything again this row group, or it re-decides
    /// per page and so is never finished. Every other leaf, and every
    /// passthrough leaf, takes the ordinary write path.
    pub fn needs_page_grain(&self) -> bool {
        if self.passthrough {
            return false;
        }
        self.always_page_grain
            || self.adapt_per_page
            || self.reopens()
            || self.learned == Choice::Undecided
    }

    /// Whether this leaf tries everything again for the coming row group.
    fn reopens(&self) -> bool {
        self.row_groups.is_multiple_of(REOPEN_EVERY)
    }

    /// The candidates for the next page: the first decides the page boundary,
    /// the rest encode the same rows.
    fn plan(
        &self,
        chunk: &ColumnChunkBuilder,
        choice: Choice,
        last: Option<&PageReport>,
    ) -> Vec<Candidate> {
        let dictionary = chunk.dictionary();
        let has_dictionary = dictionary.is_some();

        if self.passthrough {
            // One candidate, and no decision. Nothing leaves a dictionary on
            // this path by itself, so leave it once it has outgrown the
            // configured dictionary page size limit.
            let within_limit =
                dictionary.is_some_and(|d| d.encoded_bytes() <= self.dictionary_limit);
            return if within_limit {
                vec![Candidate::Dictionary]
            } else {
                vec![Candidate::Encoding(Encoding::PLAIN)]
            };
        }

        match choice {
            Choice::Undecided => {
                let mut candidates = Vec::with_capacity(self.alternatives.len() + 1);
                // The dictionary decides the boundary when there is one, so
                // that the page ends where its own budget says it ends.
                if has_dictionary {
                    candidates.push(Candidate::Dictionary);
                }
                candidates.extend(self.alternatives.iter().copied().map(Candidate::Encoding));
                candidates
            }
            Choice::Dictionary if has_dictionary => {
                if self.adapt_per_page {
                    vec![self.adapt(choice, last, has_dictionary)]
                } else {
                    vec![Candidate::Dictionary]
                }
            }
            // Settled on a dictionary that has since been abandoned: try the
            // alternatives and land on whichever wins now.
            Choice::Dictionary => self
                .alternatives
                .iter()
                .copied()
                .map(Candidate::Encoding)
                .collect(),
            Choice::Fixed(encoding) => {
                if self.adapt_per_page {
                    vec![self.adapt(choice, last, has_dictionary)]
                } else {
                    vec![Candidate::Encoding(encoding)]
                }
            }
        }
    }

    /// One candidate, chosen from the previous page's measurements. Used only
    /// after a leaf has settled, and only when it re-decides per page.
    fn adapt(&self, choice: Choice, last: Option<&PageReport>, has_dictionary: bool) -> Candidate {
        let Some(prev) = last else {
            // First page of a chunk: continue from what the leaf settled on.
            // Defaulting to the dictionary here would open every chunk with a
            // dictionary page it then abandons, which costs a dictionary page
            // per chunk for nothing.
            return match choice {
                Choice::Dictionary if has_dictionary => Candidate::Dictionary,
                Choice::Fixed(encoding) => Candidate::Encoding(encoding),
                _ if has_dictionary => Candidate::Dictionary,
                _ => Candidate::Encoding(self.alternatives[0]),
            };
        };

        let ratio = prev.compressed as f64 / prev.uncompressed.max(1) as f64;
        if prev.is_dictionary && ratio > 0.9 && has_dictionary {
            // A dictionary page that barely compressed has stopped helping.
            Candidate::Encoding(self.alternatives[0])
        } else if prev.is_dictionary && has_dictionary {
            Candidate::Dictionary
        } else {
            Candidate::Encoding(prev.encoding)
        }
    }

    /// Is the dictionary still deduplicating anything? Once enough values have
    /// gone through it, distinct entries above a quarter of them means it is
    /// not.
    ///
    /// `values` is this policy's own count of the values it sent through the
    /// dictionary, summed from the pages the dictionary candidate encoded.
    fn dictionary_is_paying(chunk: &ColumnChunkBuilder, values: u64) -> bool {
        match chunk.dictionary() {
            None => false,
            Some(d) => values < DICT_WATCH_AFTER || (d.entries() as u64) * 4 < values,
        }
    }

    /// The ordinary-path settings matching what this leaf settled on, applied
    /// when it is routed off the page grain.
    pub fn settled_properties(
        &self,
        builder: WriterPropertiesBuilder,
        descr: &ColumnDescPtr,
    ) -> WriterPropertiesBuilder {
        let path = descr.path().clone();
        match self.learned {
            _ if self.passthrough => builder,
            Choice::Dictionary => dictionary_properties(builder, path),
            Choice::Fixed(Encoding::PLAIN) => builder
                .set_column_dictionary_enabled(path.clone(), false)
                .set_column_encoding(path, Encoding::PLAIN),
            Choice::Fixed(encoding) => builder
                .set_column_dictionary_enabled(path.clone(), false)
                .set_column_encoding(path, encoding),
            // Never settled: write it as the file's properties say.
            Choice::Undecided => builder,
        }
    }
}

/// Dictionary settings for a column, on either path: keep the dictionary while
/// it pays for itself, and leave it when it stops.
pub fn dictionary_properties(
    builder: WriterPropertiesBuilder,
    path: ColumnPath,
) -> WriterPropertiesBuilder {
    builder
        .set_column_dictionary_enabled(path.clone(), true)
        .set_column_dictionary_page_size_limit(path.clone(), DICT_PAGE_SIZE_LIMIT)
        .set_column_dictionary_fallback(
            path,
            DictionaryFallback::WhenProfitable {
                worth_ratio: DICT_WORTH_RATIO,
                max_dictionary_page_size: DICT_MAX_PAGE_SIZE,
            },
        )
}

// ===========================================================================
// Plumbing
//
// From here down nothing decides anything: it opens row groups, routes leaves
// to the path the policy asked for, and appends the resulting chunks in order.
// ===========================================================================

/// One column chunk being built a page at a time, resumable across the record
/// batches of a row group.
struct PageGrainChunk {
    chunk: ColumnChunkBuilder,
    choice: Choice,
    last: Option<PageReport>,
    /// Pages that must still agree before this leaf settles.
    pages_to_settle: usize,
    /// Values sent through the chunk's dictionary so far.
    dictionary_values: u64,
}

impl PageGrainChunk {
    fn new(
        policy: &LeafPolicy,
        descr: ColumnDescPtr,
        props: WriterPropertiesPtr,
        factory: &ArrowRowGroupWriterFactory,
        leaf: usize,
    ) -> Result<Self> {
        // Carry the previous row group's answer in, unless this is a row group
        // where the leaf tries everything again. Pages come from the file
        // writer's own page store, so a page-grain chunk is bounded exactly as
        // an ordinary one is.
        let choice = if policy.reopens() {
            Choice::Undecided
        } else {
            policy.learned
        };
        Ok(Self {
            chunk: ColumnChunkBuilder::new_with_page_store(
                descr,
                props,
                factory.page_store_factory().as_ref(),
                leaf,
            )?,
            choice,
            last: None,
            pages_to_settle: PAGES_TO_SETTLE,
            dictionary_values: 0,
        })
    }

    /// Encode one leaf's values, a page at a time.
    fn write(&mut self, policy: &mut LeafPolicy, leaf: &ArrowLeafColumn) -> Result<()> {
        let mut cursor = self.chunk.cursor(leaf);
        while !cursor.is_empty() {
            // Leave the dictionary the moment it stops paying, and let the
            // alternatives decide where to land.
            if self.choice == Choice::Dictionary
                && !LeafPolicy::dictionary_is_paying(&self.chunk, self.dictionary_values)
            {
                self.choice = Choice::Undecided;
                self.pages_to_settle = 1;
            }

            let candidates = policy.plan(&self.chunk, self.choice, self.last.as_ref());
            let (first, alternatives) = candidates.split_first().expect("at least one candidate");
            let pages = self
                .chunk
                .encode_page_alternatives(&mut cursor, *first, alternatives)?;

            policy.total_pages += 1;
            if pages.len() > 1 {
                policy.multi_pages += 1;
            }
            if *first == Candidate::Dictionary {
                self.dictionary_values += u64::from(pages[0].num_values());
            }

            let winner = pick_page(&pages);
            let page = pages
                .into_iter()
                .nth(winner)
                .expect("pick_page returns an index into pages");

            self.last = Some(PageReport {
                encoding: page.encoding(),
                compressed: page.compressed_len(),
                uncompressed: page.uncompressed_len(),
                is_dictionary: is_dictionary_indices(page.encoding()),
            });

            // Settle once enough consecutive pages have agreed.
            if self.choice == Choice::Undecided && !policy.passthrough {
                self.pages_to_settle = self.pages_to_settle.saturating_sub(1);
                if self.pages_to_settle == 0 {
                    self.choice = if is_dictionary_indices(page.encoding()) {
                        Choice::Dictionary
                    } else {
                        Choice::Fixed(page.encoding())
                    };
                }
            }

            self.chunk.append(page)?;
        }
        Ok(())
    }

    /// Close the chunk and record what this row group learned.
    fn close(self, policy: &mut LeafPolicy) -> Result<ArrowColumnChunk> {
        if !policy.passthrough {
            policy.learned = self.choice;
        }
        self.chunk.close()
    }
}

/// The column writers of one open row group, one entry per leaf, on exactly one
/// of the two paths.
struct OpenRowGroup {
    page_grain: Vec<Option<PageGrainChunk>>,
    standard: Vec<Option<ArrowColumnWriter>>,
}

/// A parquet writer that measures each column's encoding instead of being told
/// it. See the module docs.
pub struct AdaptiveWriter<W: Write + Send> {
    file_writer: SerializedFileWriter<W>,
    factory: ArrowRowGroupWriterFactory,
    schema: SchemaRef,
    descrs: Vec<ColumnDescPtr>,
    /// One per leaf column, in leaf order, and the only state that survives a
    /// row group.
    pub policies: Vec<LeafPolicy>,
    props: WriterProperties,
    /// Properties for the page-grain path: the file's, plus a dictionary for
    /// every column, since a page-grain leaf can only leave a dictionary it was
    /// given.
    page_grain_props: WriterPropertiesPtr,
    row_group: usize,
    open: Option<OpenRowGroup>,
}

impl<W: Write + Send> AdaptiveWriter<W> {
    /// Start a file, as [`ArrowWriter::try_new`] would.
    pub fn try_new(sink: W, schema: SchemaRef, props: WriterProperties) -> Result<Self> {
        // `ArrowWriter` is used only to build the file writer and the row group
        // factory, and for the arrow schema it stamps into the file metadata.
        let (file_writer, factory) =
            ArrowWriter::try_new(sink, schema.clone(), Some(props.clone()))?
                .into_serialized_writer()?;

        let descrs: Vec<ColumnDescPtr> = file_writer.schema_descr().columns().to_vec();
        let policies = descrs
            .iter()
            .map(|descr| LeafPolicy::new(descr, &props))
            .collect();

        let mut builder = file_writer.properties().as_ref().clone().into_builder();
        for descr in &descrs {
            builder = dictionary_properties(builder, descr.path().clone());
        }

        Ok(Self {
            page_grain_props: Arc::new(builder.build()),
            file_writer,
            factory,
            schema,
            descrs,
            policies,
            props,
            row_group: 0,
            open: None,
        })
    }

    /// Write a batch into the open row group, starting one if needed.
    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.open.is_none() {
            self.open = Some(self.open_row_group()?);
        }
        let open = self.open.as_mut().expect("just opened");

        // `compute_leaves` yields a field's leaves in schema descriptor order,
        // so a running index over the fields is correct for nested fields too.
        let mut leaf = 0usize;
        for (field, column) in self.schema.fields().iter().zip(batch.columns()) {
            for values in compute_leaves(field.as_ref(), column)? {
                match &mut open.page_grain[leaf] {
                    Some(chunk) => chunk.write(&mut self.policies[leaf], &values)?,
                    None => open.standard[leaf]
                        .as_mut()
                        .expect("every leaf is on exactly one path")
                        .write(&values)?,
                }
                leaf += 1;
            }
        }
        Ok(())
    }

    /// Route every leaf and create the writers for a new row group.
    fn open_row_group(&mut self) -> Result<OpenRowGroup> {
        let routes: Vec<bool> = self.policies.iter().map(|p| p.needs_page_grain()).collect();

        // Ordinary writers for the settled leaves, carrying what each settled
        // on. Nothing is allocated for the leaves on the page-grain path.
        let mut builder = self.props.clone().into_builder();
        for (leaf, policy) in self.policies.iter().enumerate() {
            if !routes[leaf] {
                builder = policy.settled_properties(builder, &self.descrs[leaf]);
            }
        }
        let standard = self.factory.create_selected_column_writers(
            self.row_group,
            &Arc::new(builder.build()),
            |leaf| !routes[leaf],
        )?;

        let mut page_grain = Vec::with_capacity(self.descrs.len());
        for (leaf, policy) in self.policies.iter().enumerate() {
            page_grain.push(match routes[leaf] {
                false => None,
                true => Some(PageGrainChunk::new(
                    policy,
                    self.descrs[leaf].clone(),
                    self.page_grain_props.clone(),
                    &self.factory,
                    leaf,
                )?),
            });
        }

        Ok(OpenRowGroup {
            page_grain,
            standard,
        })
    }

    /// End the current row group. A no-op when none is open.
    pub fn flush(&mut self) -> Result<()> {
        let Some(open) = self.open.take() else {
            return Ok(());
        };
        let OpenRowGroup {
            page_grain,
            mut standard,
        } = open;

        // Close both kinds of chunk and append them in leaf order.
        let mut chunks = Vec::with_capacity(page_grain.len());
        for (leaf, chunk) in page_grain.into_iter().enumerate() {
            chunks.push(match chunk {
                Some(chunk) => chunk.close(&mut self.policies[leaf])?,
                None => standard[leaf]
                    .take()
                    .expect("every leaf is on exactly one path")
                    .close()?,
            });
        }

        let mut row_group = self.file_writer.next_row_group()?;
        for chunk in chunks {
            chunk.append_to_row_group(&mut row_group)?;
        }
        row_group.close()?;

        self.row_group += 1;
        for policy in &mut self.policies {
            policy.row_groups += 1;
        }
        Ok(())
    }

    /// Finish the file.
    pub fn close(mut self) -> Result<()> {
        self.flush()?;
        self.file_writer.close()?;
        Ok(())
    }
}

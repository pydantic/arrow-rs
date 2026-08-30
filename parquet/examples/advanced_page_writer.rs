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

//! A minimal adaptive parquet writer built on the page-grain API.
//!
//! This is the harness side of
//! [`parquet::arrow::arrow_writer::page_grain`]: a writer that chooses a
//! column's encoding from measurements rather than from configuration. It is
//! deliberately small, and shows the three things that are not obvious the
//! first time you write one:
//!
//! * **Race then settle.** Encode a span as dictionary, delta and plain,
//!   compare real compressed bytes, and commit the winner. Once enough pages
//!   agree, stop paying for the race and run a single encoder.
//! * **Charge the dictionary for its page.** A dictionary-indices page looks
//!   tiny precisely because its bytes went into the dictionary page instead.
//!   Comparing raw `compressed_len` is systematically biased towards the
//!   dictionary; see [`page_cost`].
//! * **Watch the dictionary and leave it.** Read the live dictionary while
//!   writing and abandon it mid-chunk when it stops deduplicating, letting the
//!   next race pick the landing encoding.
//!
//! The dataset shifts character halfway through the file, so a writer that
//! decides once at the top gets the second half wrong.
//!
//! For the full measured comparison against a stock `ArrowWriter`, across five
//! datasets and two compression codecs, see `examples/bakeoff.rs` and
//! `parquet/BAKEOFF.md`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --features arrow --example advanced_page_writer
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::page_grain::{
    Candidate, ColumnChunkBuilder, EncodedPage, LeafCursor,
};
use parquet::arrow::arrow_writer::{ArrowWriter, compute_leaves};
use parquet::arrow::{ArrowSchemaConverter, add_encoded_arrow_schema_to_metadata};
use parquet::basic::Encoding;
use parquet::errors::Result;
use parquet::file::properties::{WriterProperties, WriterPropertiesPtr};
use parquet::file::writer::SerializedFileWriter;

const ROWS: usize = 2_000_000;
const BATCH: usize = 65_536;
const ROWS_PER_ROW_GROUP: usize = 250_000;

// ---------------------------------------------------------------------------
// Deterministic data generation
// ---------------------------------------------------------------------------

/// A tiny deterministic PRNG, so every run produces the same file.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// A string column whose character changes halfway: dictionary-friendly first
/// half, high-cardinality second half.
fn shifting_strings() -> (Arc<Schema>, Vec<RecordBatch>) {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Utf8, false)]));
    let vocabulary: Vec<String> = (0..32).map(|i| format!("region-{i:02}")).collect();
    let mut rng = Rng::new(4);
    let mut row = 0usize;
    let batches = (0..ROWS)
        .step_by(BATCH)
        .map(|start| {
            let len = BATCH.min(ROWS - start);
            let values: Vec<String> = (0..len)
                .map(|_| {
                    let v = if row < ROWS / 2 {
                        vocabulary[rng.below(vocabulary.len() as u64) as usize].clone()
                    } else {
                        format!("urn:evt:{:016x}", rng.next_u64())
                    };
                    row += 1;
                    v
                })
                .collect();
            let array: ArrayRef = Arc::new(StringArray::from(values));
            RecordBatch::try_new(schema.clone(), vec![array]).unwrap()
        })
        .collect();
    (schema, batches)
}

// ---------------------------------------------------------------------------
// The policy
// ---------------------------------------------------------------------------

/// How this column is currently being written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Settled {
    /// Still racing; no choice has been made for this chunk yet.
    Racing,
    /// Settled on the chunk's dictionary.
    Dictionary,
    /// Settled on a fixed encoding.
    Pinned(Encoding),
}

/// Ranks encodings by how cheap they are to decode, lowest first. Used to break
/// near-ties in favour of the reader: a page that is 1% larger but plain is
/// usually the better trade.
fn decode_cost(encoding: Encoding) -> u8 {
    match encoding {
        Encoding::PLAIN => 0,
        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => 1,
        Encoding::DELTA_BINARY_PACKED | Encoding::DELTA_LENGTH_BYTE_ARRAY => 2,
        Encoding::DELTA_BYTE_ARRAY => 3,
        _ => 4,
    }
}

/// The true cost of a page.
///
/// [`EncodedPage::compressed_len`] is the data page only. A dictionary-indices
/// page is cheap precisely because its bytes went into the dictionary page
/// instead, and that page is written once at close where no per-page comparison
/// can see it. Comparing raw `compressed_len` is therefore systematically
/// biased towards the dictionary: an indices page over 20 000 distinct values
/// is tiny and the 500 KiB of dictionary entries it just created is invisible.
///
/// `dictionary_growth` is how many bytes the chunk's dictionary gained while
/// encoding this span, read from [`ColumnChunkBuilder::dictionary`] either side
/// of the call. Charging it to the indices candidate is what makes the
/// comparison honest, and it is the difference between this writer abandoning a
/// dictionary on high-cardinality data and riding it off a cliff.
fn page_cost(page: &EncodedPage, dictionary_growth: usize) -> usize {
    if page.is_dictionary_indices() {
        page.compressed_len() + dictionary_growth
    } else {
        page.compressed_len()
    }
}

/// Pick a winner: cheapest page, but prefer a cheaper-to-decode page when it is
/// within `tolerance` of the cheapest.
fn pick_winner(pages: &[EncodedPage], dictionary_growth: usize, tolerance: f64) -> usize {
    let cost = |p: &EncodedPage| page_cost(p, dictionary_growth);
    let smallest = pages.iter().map(cost).min().unwrap_or(0);
    let budget = (smallest as f64 * (1.0 + tolerance)) as usize;
    pages
        .iter()
        .enumerate()
        .filter(|(_, p)| cost(p) <= budget)
        .min_by_key(|(_, p)| (decode_cost(p.encoding()), cost(p)))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Per-column state that outlives a single row group. This is where
/// "cross-row-group learning" lives.
struct ColumnPolicy {
    /// Choice carried forward from the previous row group.
    learned: Settled,
    /// Race candidates for this column's physical type.
    challengers: Vec<Encoding>,
    /// How many row groups have been written.
    row_groups: usize,
    /// Tally of landed encodings, for the report.
    landed: HashMap<Encoding, usize>,
    /// How many pages were raced (i.e. encoded more than once).
    raced_pages: usize,
    total_pages: usize,
}

/// Re-open the race every this many row groups, so the writer can notice the
/// data changing under it.
const REOPEN_EVERY: usize = 4;

/// Pages that must agree before the race is closed.
const RACES_BEFORE_SETTLING: usize = 2;

impl ColumnPolicy {
    fn new(challengers: Vec<Encoding>) -> Self {
        Self {
            learned: Settled::Racing,
            challengers,
            row_groups: 0,
            landed: HashMap::new(),
            raced_pages: 0,
            total_pages: 0,
        }
    }

    /// Decide what to offer for the next page.
    fn plan(&self, chunk: &ColumnChunkBuilder, state: Settled) -> Vec<Candidate> {
        let has_dictionary = chunk.dictionary().is_some();
        let challengers = || self.challengers.iter().copied().map(Candidate::Pinned);

        match state {
            Settled::Racing => {
                // The dictionary paces when it is available: it is the encoding
                // most likely to win on this kind of data, and pacing with it
                // keeps its budget accounting honest.
                has_dictionary
                    .then_some(Candidate::Dictionary)
                    .into_iter()
                    .chain(challengers())
                    .collect()
            }
            Settled::Dictionary if has_dictionary => vec![Candidate::Dictionary],
            // The dictionary was abandoned since we settled on it: race the
            // fallbacks now, and land on whichever wins at this moment.
            Settled::Dictionary => challengers().collect(),
            Settled::Pinned(encoding) => vec![Candidate::Pinned(encoding)],
        }
    }

    /// Should the dictionary be abandoned now?
    ///
    /// The rule: once enough values have gone through it, if distinct entries
    /// are more than a quarter of the values written, it has stopped
    /// deduplicating anything and is just an extra indirection plus a page.
    fn dictionary_is_still_paying(chunk: &ColumnChunkBuilder) -> bool {
        match chunk.dictionary() {
            None => false,
            Some(d) => d.values_written() < 50_000 || (d.entries() as u64) * 4 < d.values_written(),
        }
    }
}

// ---------------------------------------------------------------------------
// The write loop
// ---------------------------------------------------------------------------

/// Write one column chunk under `policy`, returning the chunk.
fn write_chunk(
    policy: &mut ColumnPolicy,
    descr: parquet::schema::types::ColumnDescPtr,
    props: WriterPropertiesPtr,
    leaves: &[parquet::arrow::arrow_writer::ArrowLeafColumn],
) -> Result<parquet::arrow::arrow_writer::ArrowColumnChunk> {
    let mut chunk = ColumnChunkBuilder::new(descr, props)?;

    // Cross-row-group learning: start from what the previous row group settled
    // on, unless it is time to re-open the race.
    let reopen = policy.row_groups.is_multiple_of(REOPEN_EVERY);
    let mut state = if reopen {
        Settled::Racing
    } else {
        policy.learned
    };
    let mut races_left = RACES_BEFORE_SETTLING;

    for leaf in leaves {
        let mut cursor: LeafCursor = chunk.cursor(leaf);
        while !cursor.is_empty() {
            // Dictionary watching: drop out of the dictionary the moment it
            // stops paying, and let the next race pick the landing encoding.
            if matches!(state, Settled::Dictionary)
                && !ColumnPolicy::dictionary_is_still_paying(&chunk)
            {
                state = Settled::Racing;
                races_left = 1;
            }

            let candidates = policy.plan(&chunk, state);

            // Read the dictionary either side of the encode so the indices
            // candidate can be charged for the dictionary bytes it created.
            let dictionary_before = chunk.dictionary().map(|d| d.encoded_bytes()).unwrap_or(0);
            let pages = chunk.encode_page(&mut cursor, &candidates)?;
            let dictionary_growth = chunk
                .dictionary()
                .map(|d| d.encoded_bytes())
                .unwrap_or(dictionary_before)
                .saturating_sub(dictionary_before);

            policy.total_pages += 1;
            if pages.len() > 1 {
                policy.raced_pages += 1;
            }

            let winner = pick_winner(&pages, dictionary_growth, 0.02);
            let page = pages.into_iter().nth(winner).unwrap();
            *policy.landed.entry(page.encoding()).or_default() += 1;

            // Settle: after enough raced pages agree, stop paying for the race.
            if matches!(state, Settled::Racing) {
                races_left = races_left.saturating_sub(1);
                if races_left == 0 {
                    state = if page.is_dictionary_indices() {
                        Settled::Dictionary
                    } else {
                        Settled::Pinned(page.encoding())
                    };
                }
            }

            chunk.append(page)?;
        }
    }

    policy.row_groups += 1;
    policy.learned = state;
    chunk.close()
}

/// Write a whole file through the page-grain API.
fn write_adaptive(
    schema: &Arc<Schema>,
    batches: &[RecordBatch],
    props: Arc<WriterProperties>,
    path: &str,
) -> Result<(u64, ColumnPolicy)> {
    let parquet_schema = ArrowSchemaConverter::new()
        .with_coerce_types(props.coerce_types())
        .convert(schema)?;

    let mut props = (*props).clone();
    add_encoded_arrow_schema_to_metadata(schema, &mut props);
    let props = Arc::new(props);

    let mut policy = ColumnPolicy::new(vec![Encoding::PLAIN, Encoding::DELTA_BYTE_ARRAY]);

    let file = File::create(path)?;
    let mut writer =
        SerializedFileWriter::new(file, parquet_schema.root_schema_ptr(), props.clone())?;

    let field = schema.field(0);
    let descr = parquet_schema.columns()[0].clone();
    let mut pending: Vec<RecordBatch> = Vec::new();
    let mut pending_rows = 0usize;

    let flush = |writer: &mut SerializedFileWriter<File>,
                 policy: &mut ColumnPolicy,
                 batches: &mut Vec<RecordBatch>|
     -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }
        let mut row_group = writer.next_row_group()?;
        let mut leaves = Vec::new();
        for batch in batches.iter() {
            leaves.extend(compute_leaves(field, batch.column(0))?);
        }
        write_chunk(policy, descr.clone(), props.clone(), &leaves)?
            .append_to_row_group(&mut row_group)?;
        row_group.close()?;
        batches.clear();
        Ok(())
    };

    for batch in batches {
        pending_rows += batch.num_rows();
        pending.push(batch.clone());
        if pending_rows >= ROWS_PER_ROW_GROUP {
            flush(&mut writer, &mut policy, &mut pending)?;
            pending_rows = 0;
        }
    }
    flush(&mut writer, &mut policy, &mut pending)?;
    writer.close()?;
    Ok((std::fs::metadata(path)?.len(), policy))
}

/// The baseline: plain `ArrowWriter` at the same properties.
fn write_baseline(
    schema: &Arc<Schema>,
    batches: &[RecordBatch],
    props: Arc<WriterProperties>,
    path: &str,
) -> Result<u64> {
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some((*props).clone()))?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok(std::fs::metadata(path)?.len())
}

/// Read the file back and require exact row equality against the source.
fn verify(path: &str, batches: &[RecordBatch]) -> Result<usize> {
    let concat = |arrays: &[ArrayRef]| -> Result<ArrayRef> {
        arrow_select::concat::concat(&arrays.iter().map(|a| a.as_ref()).collect::<Vec<_>>())
            .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))
    };

    let expected = concat(
        &batches
            .iter()
            .map(|b| b.column(0).clone())
            .collect::<Vec<_>>(),
    )?;

    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?
        .with_batch_size(8192)
        .build()?;
    let mut actual: Vec<ArrayRef> = Vec::new();
    for batch in reader {
        actual.push(batch?.column(0).clone());
    }
    let actual = concat(&actual)?;

    assert_eq!(&expected, &actual, "row values differ in {path}");
    Ok(actual.len())
}

fn human(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:.1} {}", UNITS[u])
}

fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("advanced_page_writer");
    std::fs::create_dir_all(&dir)?;
    let adaptive_path = dir.join("adaptive.parquet");
    let baseline_path = dir.join("baseline.parquet");
    let adaptive_path = adaptive_path.to_str().unwrap();
    let baseline_path = baseline_path.to_str().unwrap();

    // The same properties for both writers, including the row group size: a
    // dictionary page is written per column chunk, so comparing bytes across
    // different row group counts compares the wrong thing.
    let props = Arc::new(
        WriterProperties::builder()
            .set_data_page_row_count_limit(20_000)
            .set_max_row_group_row_count(Some(ROWS_PER_ROW_GROUP))
            .build(),
    );

    let (schema, batches) = shifting_strings();

    println!(
        "adaptive parquet writing over the page-grain API\n\
         strings that change character mid-file, {ROWS} rows, \
         {ROWS_PER_ROW_GROUP} rows per row group\n"
    );

    let (adaptive_bytes, policy) = write_adaptive(&schema, &batches, props.clone(), adaptive_path)?;
    let baseline_bytes = write_baseline(&schema, &batches, props, baseline_path)?;

    let rows = verify(adaptive_path, &batches)?;
    assert_eq!(rows, verify(baseline_path, &batches)?);

    let delta = adaptive_bytes as f64 / baseline_bytes as f64 - 1.0;
    println!("  adaptive : {:>10}", human(adaptive_bytes));
    println!(
        "  baseline : {:>10}   (adaptive is {:+.1}% bytes)",
        human(baseline_bytes),
        delta * 100.0,
    );

    let mut landed: Vec<_> = policy.landed.iter().collect();
    landed.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
    let landed: Vec<String> = landed
        .iter()
        .map(|(encoding, count)| format!("{encoding} x{count}"))
        .collect();
    println!(
        "  pages {} raced {} landed on {}",
        policy.total_pages,
        policy.raced_pages,
        landed.join(", ")
    );
    println!("  verified {rows} rows read back exactly");

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

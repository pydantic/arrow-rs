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

//! An adaptive parquet writer built on the page-grain API.
//!
//! This is the harness side of
//! [`parquet::arrow::arrow_writer::page_grain`]: a writer that chooses each
//! column's encoding from measurements rather than from configuration. It
//! demonstrates four policies that are only expressible once the page grain is
//! open:
//!
//! * **Race then settle.** Encode the first spans of a chunk as dictionary,
//!   delta and plain, compare real compressed bytes, and settle on a winner
//!   with a near-tie preference for the cheaper thing to decode. Afterwards run
//!   a single encoder.
//! * **Dictionary watching.** Read the live dictionary while writing and
//!   abandon it mid-chunk when its entries-to-values ratio stops paying,
//!   choosing the landing encoding at that moment from a race.
//! * **Cross-row-group learning.** Carry the settled choice into later row
//!   groups, and re-open the race every N row groups so the writer can notice
//!   the data changing under it.
//! * **Adapt per page.** For one column, keep revisiting the choice after it
//!   has settled, from the numbers on the previous sealed page, rather than
//!   pinning the winner for the rest of the chunk.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --features arrow --example advanced_page_writer
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
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

// ---------------------------------------------------------------------------
// Deterministic data generation
// ---------------------------------------------------------------------------

/// A tiny deterministic PRNG, so every run of this example produces the same
/// files and the numbers it prints are comparable across runs.
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

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

const ROWS: usize = 2_000_000;
const BATCH: usize = 65_536;

struct Dataset {
    name: &'static str,
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

/// (1) Low-cardinality strings: the dictionary should win everywhere.
fn low_cardinality_strings() -> Dataset {
    let schema = Arc::new(Schema::new(vec![Field::new("tag", DataType::Utf8, false)]));
    let vocabulary: Vec<String> = (0..48).map(|i| format!("service-{i:02}-eu-west")).collect();
    let mut rng = Rng::new(1);
    let batches = (0..ROWS)
        .step_by(BATCH)
        .map(|start| {
            let len = BATCH.min(ROWS - start);
            let values: Vec<&str> = (0..len)
                .map(|_| vocabulary[rng.below(vocabulary.len() as u64) as usize].as_str())
                .collect();
            let array: ArrayRef = Arc::new(StringArray::from(values));
            RecordBatch::try_new(schema.clone(), vec![array]).unwrap()
        })
        .collect();
    Dataset {
        name: "low-cardinality strings",
        schema,
        batches,
    }
}

/// (2) High-cardinality int64 timestamps: monotonic-ish, overflows any
/// dictionary, and is exactly what delta encoding is for.
fn timestamps() -> Dataset {
    let schema = Arc::new(Schema::new(vec![Field::new("ts", DataType::Int64, false)]));
    let mut rng = Rng::new(2);
    let mut now: i64 = 1_700_000_000_000;
    let batches = (0..ROWS)
        .step_by(BATCH)
        .map(|start| {
            let len = BATCH.min(ROWS - start);
            let values: Vec<i64> = (0..len)
                .map(|_| {
                    now += rng.below(4_000) as i64;
                    now
                })
                .collect();
            let array: ArrayRef = Arc::new(Int64Array::from(values));
            RecordBatch::try_new(schema.clone(), vec![array]).unwrap()
        })
        .collect();
    Dataset {
        name: "high-cardinality int64 timestamps",
        schema,
        batches,
    }
}

/// (3) f64 values: nothing compresses these well; the point is that the writer
/// should notice and stop paying to find out.
fn floats() -> Dataset {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        false,
    )]));
    let mut rng = Rng::new(3);
    let batches = (0..ROWS)
        .step_by(BATCH)
        .map(|start| {
            let len = BATCH.min(ROWS - start);
            let values: Vec<f64> = (0..len).map(|_| rng.unit() * 1_000.0).collect();
            let array: ArrayRef = Arc::new(Float64Array::from(values));
            RecordBatch::try_new(schema.clone(), vec![array]).unwrap()
        })
        .collect();
    Dataset {
        name: "f64 values",
        schema,
        batches,
    }
}

/// (4) A string column whose character changes halfway: dictionary-friendly
/// first half, high-cardinality second half. A writer that decides once at the
/// top of the file gets the second half wrong.
fn shifting_strings() -> Dataset {
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
    Dataset {
        name: "strings that change character mid-file",
        schema,
        batches,
    }
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
/// `EncodedPage::compressed_len` is the data page only. A dictionary-indices
/// page is cheap precisely because its bytes went into the dictionary page
/// instead, and that page is written once at close where no per-page comparison
/// can see it. Comparing raw `compressed_len` is therefore systematically
/// biased towards the dictionary: an indices page over 20 000 distinct values
/// is tiny and the 500 KiB of dictionary entries it just created is invisible.
///
/// `dictionary_growth` is how many bytes the chunk's dictionary gained while
/// encoding this span, read from `ColumnChunkBuilder::dictionary` either side of
/// the call. Charging it to the indices candidate is what makes the comparison
/// honest, and it is the difference between this writer abandoning a dictionary
/// on high-cardinality data and riding it off a cliff.
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
    name: String,
    /// Choice carried forward from the previous row group.
    learned: Settled,
    /// Re-open the race every this many row groups.
    reopen_every: usize,
    /// Race candidates for this column's physical type.
    challengers: Vec<Encoding>,
    /// After this column has settled, re-decide each page from the previous
    /// sealed page's numbers instead of pinning the settled choice outright.
    /// This refines a settled column; it does not replace the race.
    adapt_per_page: bool,
    /// How many row groups have been written.
    row_groups: usize,
    /// Tally of landed encodings, for the report.
    landed: HashMap<Encoding, usize>,
    /// How many pages were raced (i.e. encoded more than once).
    raced_pages: usize,
    total_pages: usize,
}

impl ColumnPolicy {
    fn new(name: &str, challengers: Vec<Encoding>, adapt_per_page: bool) -> Self {
        Self {
            name: name.to_string(),
            learned: Settled::Racing,
            reopen_every: 4,
            challengers,
            adapt_per_page,
            row_groups: 0,
            landed: HashMap::new(),
            raced_pages: 0,
            total_pages: 0,
        }
    }

    /// Decide what to offer for the next page.
    fn plan(
        &self,
        chunk: &ColumnChunkBuilder,
        state: Settled,
        last: Option<&PageReport>,
    ) -> Vec<Candidate> {
        let has_dictionary = chunk.dictionary().is_some();

        match state {
            Settled::Racing => {
                let mut candidates = Vec::new();
                // The dictionary paces when it is available: it is the encoding
                // most likely to win on this kind of data, and pacing with it
                // keeps its budget accounting honest.
                if has_dictionary {
                    candidates.push(Candidate::Dictionary);
                }
                for encoding in &self.challengers {
                    candidates.push(Candidate::Pinned(*encoding));
                }
                candidates
            }
            Settled::Dictionary if has_dictionary => {
                if self.adapt_per_page {
                    vec![self.adapt(state, last, has_dictionary)]
                } else {
                    vec![Candidate::Dictionary]
                }
            }
            // The dictionary was abandoned since we settled on it: race the
            // fallbacks now, and land on whichever wins at this moment.
            Settled::Dictionary => self
                .challengers
                .iter()
                .map(|e| Candidate::Pinned(*e))
                .collect(),
            Settled::Pinned(encoding) => {
                if self.adapt_per_page {
                    vec![self.adapt(state, last, has_dictionary)]
                } else {
                    vec![Candidate::Pinned(encoding)]
                }
            }
        }
    }

    /// One candidate, chosen from the previous sealed page's measured numbers.
    ///
    /// Used only after a column has settled. A page that barely compressed is
    /// evidence the current encoding has stopped helping, and the dictionary
    /// can only be left, never rejoined.
    fn adapt(&self, state: Settled, last: Option<&PageReport>, has_dictionary: bool) -> Candidate {
        match last {
            // Cold start: the first page of a chunk has no previous page to
            // learn from, so it continues from what this column settled on. It
            // must not default to the dictionary, or every settled chunk would
            // open with a dictionary page it then abandons.
            None => match state {
                Settled::Dictionary if has_dictionary => Candidate::Dictionary,
                Settled::Pinned(encoding) => Candidate::Pinned(encoding),
                _ if has_dictionary => Candidate::Dictionary,
                _ => Candidate::Pinned(self.challengers[0]),
            },
            Some(prev) => {
                let ratio = prev.compressed as f64 / prev.uncompressed.max(1) as f64;
                if prev.encoding == Encoding::RLE_DICTIONARY && ratio > 0.9 && has_dictionary {
                    Candidate::Pinned(self.challengers[0])
                } else if prev.is_dictionary && has_dictionary {
                    Candidate::Dictionary
                } else {
                    Candidate::Pinned(prev.encoding)
                }
            }
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

/// What one sealed page measured, kept to feed the next decision.
struct PageReport {
    encoding: Encoding,
    compressed: usize,
    uncompressed: usize,
    is_dictionary: bool,
}

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
    let reopen = policy.row_groups.is_multiple_of(policy.reopen_every);
    let mut state = if reopen {
        Settled::Racing
    } else {
        policy.learned
    };
    let mut last: Option<PageReport> = None;
    // Pages to race before settling, when racing.
    let mut races_left = 2usize;

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

            let candidates = policy.plan(&chunk, state, last.as_ref());

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

            last = Some(PageReport {
                encoding: page.encoding(),
                compressed: page.compressed_len(),
                uncompressed: page.uncompressed_len(),
                is_dictionary: page.is_dictionary_indices(),
            });
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
    dataset: &Dataset,
    props: Arc<WriterProperties>,
    path: &str,
    rows_per_row_group: usize,
) -> Result<(u64, Duration, Vec<ColumnPolicy>)> {
    let parquet_schema = ArrowSchemaConverter::new()
        .with_coerce_types(props.coerce_types())
        .convert(&dataset.schema)?;

    let mut props = (*props).clone();
    add_encoded_arrow_schema_to_metadata(&dataset.schema, &mut props);
    let props = Arc::new(props);

    let mut policies: Vec<ColumnPolicy> = dataset
        .schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let challengers = match f.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary => {
                    vec![Encoding::PLAIN, Encoding::DELTA_BYTE_ARRAY]
                }
                DataType::Int64 | DataType::Int32 => {
                    vec![Encoding::PLAIN, Encoding::DELTA_BINARY_PACKED]
                }
                _ => vec![Encoding::PLAIN],
            };
            // One column per file here; the float column is the one that adapts
            // per page instead of racing.
            let adapt = matches!(f.data_type(), DataType::Float64);
            ColumnPolicy::new(f.name(), challengers, adapt && i == 0)
        })
        .collect();

    let start = Instant::now();
    let file = File::create(path)?;
    let mut writer =
        SerializedFileWriter::new(file, parquet_schema.root_schema_ptr(), props.clone())?;

    let mut pending: Vec<RecordBatch> = Vec::new();
    let mut pending_rows = 0usize;

    let flush = |writer: &mut SerializedFileWriter<File>,
                 policies: &mut Vec<ColumnPolicy>,
                 batches: &mut Vec<RecordBatch>|
     -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }
        let mut row_group = writer.next_row_group()?;
        for (col, policy) in policies.iter_mut().enumerate() {
            let field = dataset.schema.field(col);
            let descr = parquet_schema.columns()[col].clone();
            let mut leaves = Vec::new();
            for batch in batches.iter() {
                leaves.extend(compute_leaves(field, batch.column(col))?);
            }
            let chunk = write_chunk(policy, descr, props.clone(), &leaves)?;
            chunk.append_to_row_group(&mut row_group)?;
        }
        row_group.close()?;
        batches.clear();
        Ok(())
    };

    for batch in &dataset.batches {
        pending_rows += batch.num_rows();
        pending.push(batch.clone());
        if pending_rows >= rows_per_row_group {
            flush(&mut writer, &mut policies, &mut pending)?;
            pending_rows = 0;
        }
    }
    flush(&mut writer, &mut policies, &mut pending)?;
    writer.close()?;
    let elapsed = start.elapsed();
    let bytes = std::fs::metadata(path)?.len();
    Ok((bytes, elapsed, policies))
}

/// The baseline: plain `ArrowWriter` at the same properties.
fn write_baseline(
    dataset: &Dataset,
    props: Arc<WriterProperties>,
    path: &str,
) -> Result<(u64, Duration)> {
    let start = Instant::now();
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, dataset.schema.clone(), Some((*props).clone()))?;
    for batch in &dataset.batches {
        writer.write(batch)?;
    }
    writer.close()?;
    let elapsed = start.elapsed();
    Ok((std::fs::metadata(path)?.len(), elapsed))
}

/// Read both files back and require exact row equality against the source.
fn verify(path: &str, dataset: &Dataset) -> Result<usize> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(8192)
        .build()?;

    let expected: Vec<ArrayRef> = dataset
        .batches
        .iter()
        .map(|b| b.column(0).clone())
        .collect();
    let expected =
        arrow_select::concat::concat(&expected.iter().map(|a| a.as_ref()).collect::<Vec<_>>())
            .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))?;

    let mut actual: Vec<ArrayRef> = Vec::new();
    for batch in reader {
        actual.push(batch?.column(0).clone());
    }
    let actual =
        arrow_select::concat::concat(&actual.iter().map(|a| a.as_ref()).collect::<Vec<_>>())
            .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))?;

    assert_eq!(expected.len(), actual.len(), "row count mismatch in {path}");
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

    let rows_per_row_group = 250_000;
    // The same properties for both writers, including the row group size: a
    // dictionary page is written per column chunk, so comparing bytes across
    // different row group counts compares the wrong thing.
    let props = Arc::new(
        WriterProperties::builder()
            .set_data_page_row_count_limit(20_000)
            .set_max_row_group_row_count(Some(rows_per_row_group))
            .build(),
    );

    let datasets = [
        low_cardinality_strings(),
        timestamps(),
        floats(),
        shifting_strings(),
    ];

    println!(
        "adaptive parquet writing over the page-grain API\n\
         {ROWS} rows per dataset, {rows_per_row_group} rows per row group\n"
    );

    for dataset in &datasets {
        let adaptive_path = dir.join("adaptive.parquet");
        let baseline_path = dir.join("baseline.parquet");
        let adaptive_path = adaptive_path.to_str().unwrap();
        let baseline_path = baseline_path.to_str().unwrap();

        let (adaptive_bytes, adaptive_time, policies) =
            write_adaptive(dataset, props.clone(), adaptive_path, rows_per_row_group)?;
        let (baseline_bytes, baseline_time) =
            write_baseline(dataset, props.clone(), baseline_path)?;

        let adaptive_rows = verify(adaptive_path, dataset)?;
        let baseline_rows = verify(baseline_path, dataset)?;
        assert_eq!(adaptive_rows, baseline_rows);

        let delta = adaptive_bytes as f64 / baseline_bytes as f64 - 1.0;
        println!("== {} ==", dataset.name);
        println!(
            "  adaptive : {:>10}  {:>7.2}s",
            human(adaptive_bytes),
            adaptive_time.as_secs_f64()
        );
        println!(
            "  baseline : {:>10}  {:>7.2}s   (adaptive is {:+.1}% bytes, {:.2}x time)",
            human(baseline_bytes),
            baseline_time.as_secs_f64(),
            delta * 100.0,
            adaptive_time.as_secs_f64() / baseline_time.as_secs_f64().max(1e-9),
        );
        for policy in &policies {
            let mut landed: Vec<_> = policy.landed.iter().collect();
            landed.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
            let landed: Vec<String> = landed
                .iter()
                .map(|(encoding, count)| format!("{encoding} x{count}"))
                .collect();
            println!(
                "  column {:<8} pages {:<5} raced {:<5} landed on {}",
                policy.name,
                policy.total_pages,
                policy.raced_pages,
                landed.join(", ")
            );
        }
        println!("  verified {adaptive_rows} rows read back exactly\n");
    }

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

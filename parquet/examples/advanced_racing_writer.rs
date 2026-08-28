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

//! Picks a parquet encoding per column by *racing* candidate encodings against
//! each other, rather than by guessing from the schema.
//!
//! For each row group, every leaf column is encoded several times in parallel,
//! once per candidate encoding, and only the smallest resulting column chunk is
//! appended to the file. The rest are dropped. Because a parquet column chunk
//! is self describing, chunks encoded with different properties can be mixed
//! freely within one file, and any reader will read them back.
//!
//! Racing every row group would cost K times a single writer's work forever, so
//! a leaf *settles* on its winner once a row group produces a decisive result,
//! and reuses that winner without racing. Settled leaves are periodically
//! reopened so that a column whose data changes character part way through the
//! file can change its mind.
//!
//! This is built entirely on public API:
//!
//! * `ArrowWriter::into_serialized_writer` hands back the lower level
//!   `SerializedFileWriter` and an `ArrowRowGroupWriterFactory`.
//! * `ArrowRowGroupWriterFactory::create_column_writers_with_properties`
//!   builds one set of column writers per candidate, each with its own
//!   `WriterProperties`, while keeping the file writer's page store factory
//!   and encryption setup.
//! * `compute_leaves` flattens a column into leaves that can be fed to every
//!   candidate's writer.
//! * `DictionaryFallback::WhenProfitable` lets the dictionary candidate keep
//!   its dictionary past the dictionary page size limit while it is still
//!   paying for itself, instead of falling back on the byte limit alone.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example advanced_racing_writer
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use arrow_select::concat::concat_batches;

use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::{ArrowColumnChunk, ArrowColumnWriter, compute_leaves};
use parquet::basic::{Compression, Encoding, Type as PhysicalType};
use parquet::errors::Result;
use parquet::file::properties::{DictionaryFallback, WriterProperties};
use parquet::schema::types::ColumnPath;

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/// Rows per dataset.
const TOTAL_ROWS: usize = 2_000_000;
/// Rows per row group: 2M rows over 20 row groups.
const ROW_GROUP_ROWS: usize = 100_000;
/// Rows per record batch fed to the writers.
const BATCH_ROWS: usize = 25_000;

/// Dictionary page size limit used by the dictionary candidate.
///
/// Deliberately below the 1 MiB default: with 100k row row groups a column has
/// to be extremely high cardinality to build a 1 MiB dictionary within a single
/// chunk, so the default limit would never be reached and the dictionary
/// fallback policy would never be exercised at all.
const DICT_PAGE_SIZE_LIMIT: usize = 64 * 1024;

/// `worth_ratio` for [`DictionaryFallback::WhenProfitable`]: keep the
/// dictionary past the page size limit while it stays under a quarter of the
/// PLAIN encoded size of the values it has absorbed.
const DICT_WORTH_RATIO: f64 = 0.25;

/// Absolute cap on a retained dictionary page, bounding reader memory.
const DICT_MAX_PAGE_SIZE: usize = 8 * 1024 * 1024;

/// A row group settles a leaf when the gap between the best and worst candidate
/// is at least this fraction of the worst. A narrow spread means the row group
/// did not really discriminate, so the leaf keeps racing.
const SETTLE_GAP: f64 = 0.10;

/// Two candidates within this fraction of each other are treated as tied, and
/// the tie is broken on decode cost rather than on bytes.
const NEAR_TIE: f64 = 0.02;

/// Re-race a settled leaf every this many row groups.
const REOPEN_EVERY: usize = 8;

// ---------------------------------------------------------------------------
// Candidate encodings
// ---------------------------------------------------------------------------

/// One candidate encoding strategy for a leaf column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Candidate {
    /// Dictionary encoding, retaining the dictionary while it is profitable.
    Dictionary,
    /// Dictionary encoding with the stock byte-limit fallback. Only used to
    /// isolate the effect of [`DictionaryFallback::WhenProfitable`].
    DictionaryOnPageSizeLimit,
    /// No dictionary, PLAIN.
    Plain,
    /// No dictionary, DELTA_BINARY_PACKED or DELTA_BYTE_ARRAY by physical type.
    Delta,
}

impl Candidate {
    fn label(self) -> &'static str {
        match self {
            Candidate::Dictionary => "dictionary",
            Candidate::DictionaryOnPageSizeLimit => "dictionary(page-limit)",
            Candidate::Plain => "plain",
            Candidate::Delta => "delta",
        }
    }

    /// Relative cost of decoding this encoding, lowest first. Used only to
    /// break near-ties, where the byte difference does not justify handing
    /// readers a more expensive encoding.
    fn decode_cost(self) -> u8 {
        match self {
            Candidate::Dictionary | Candidate::DictionaryOnPageSizeLimit => 0,
            Candidate::Plain => 1,
            Candidate::Delta => 2,
        }
    }

    /// The delta encoding for `physical`, if one applies.
    fn delta_encoding(physical: PhysicalType) -> Option<Encoding> {
        match physical {
            PhysicalType::INT32 | PhysicalType::INT64 => Some(Encoding::DELTA_BINARY_PACKED),
            PhysicalType::BYTE_ARRAY | PhysicalType::FIXED_LEN_BYTE_ARRAY => {
                Some(Encoding::DELTA_BYTE_ARRAY)
            }
            _ => None,
        }
    }

    /// Candidates worth racing for `physical`, pruned to those the type
    /// actually supports.
    fn for_physical_type(physical: PhysicalType) -> Vec<Candidate> {
        // BOOLEAN has no dictionary support and no delta encoding worth
        // racing, so there is nothing to choose between.
        if physical == PhysicalType::BOOLEAN {
            return vec![Candidate::Plain];
        }
        let mut out = vec![Candidate::Dictionary, Candidate::Plain];
        if Self::delta_encoding(physical).is_some() {
            out.push(Candidate::Delta);
        }
        out
    }

    /// Applies this candidate to `col` in `builder`.
    fn apply(
        self,
        builder: parquet::file::properties::WriterPropertiesBuilder,
        col: ColumnPath,
        physical: PhysicalType,
    ) -> parquet::file::properties::WriterPropertiesBuilder {
        match self {
            Candidate::Dictionary => builder
                .set_column_dictionary_enabled(col.clone(), true)
                .set_column_dictionary_page_size_limit(col.clone(), DICT_PAGE_SIZE_LIMIT)
                .set_column_dictionary_fallback(
                    col,
                    DictionaryFallback::WhenProfitable {
                        worth_ratio: DICT_WORTH_RATIO,
                        max_dictionary_page_size: DICT_MAX_PAGE_SIZE,
                    },
                ),
            Candidate::DictionaryOnPageSizeLimit => builder
                .set_column_dictionary_enabled(col.clone(), true)
                .set_column_dictionary_page_size_limit(col.clone(), DICT_PAGE_SIZE_LIMIT)
                .set_column_dictionary_fallback(col, DictionaryFallback::OnPageSizeLimit),
            Candidate::Plain => builder
                .set_column_dictionary_enabled(col.clone(), false)
                .set_column_encoding(col, Encoding::PLAIN),
            Candidate::Delta => {
                let encoding = Self::delta_encoding(physical)
                    .expect("Delta candidate built for a type without a delta encoding");
                builder
                    .set_column_dictionary_enabled(col.clone(), false)
                    .set_column_encoding(col, encoding)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-leaf racing state
// ---------------------------------------------------------------------------

/// What the racer knows about one leaf column.
struct LeafState {
    path: ColumnPath,
    physical: PhysicalType,
    candidates: Vec<Candidate>,
    /// The candidate this leaf has settled on, and the row group it settled at.
    settled: Option<(Candidate, usize)>,
    /// How many row groups each outcome has won, keyed by the candidate label
    /// and the encodings the winning chunk actually ended up using.
    wins: HashMap<String, usize>,
}

impl LeafState {
    fn new(path: ColumnPath, physical: PhysicalType) -> Self {
        Self {
            path,
            physical,
            candidates: Candidate::for_physical_type(physical),
            settled: None,
            wins: HashMap::new(),
        }
    }

    /// Whether this leaf is racing in `row_group`, reopening a settled leaf
    /// every [`REOPEN_EVERY`] row groups.
    fn is_racing(&self, row_group: usize) -> bool {
        if self.candidates.len() < 2 {
            return false;
        }
        match self.settled {
            None => true,
            Some((_, settled_at)) => row_group.saturating_sub(settled_at) >= REOPEN_EVERY,
        }
    }

    /// The candidate this leaf uses in candidate set `k` of `row_group`.
    ///
    /// A leaf that is not racing uses its settled candidate in every set. A
    /// racing leaf with fewer candidates than the widest leaf repeats its last
    /// candidate in the surplus sets; those sets produce identical chunks and
    /// are deduplicated when the winner is chosen.
    fn candidate_for_set(&self, row_group: usize, k: usize) -> Candidate {
        if !self.is_racing(row_group)
            && let Some((settled, _)) = self.settled
        {
            return settled;
        }
        self.candidates[k.min(self.candidates.len() - 1)]
    }
}

/// Chooses the winning candidate set for a leaf from the sizes each produced.
///
/// Returns the index into `sizes`. The smallest chunk wins, except that any
/// candidate within [`NEAR_TIE`] of the smallest is preferred if it is cheaper
/// to decode: a fraction of a percent is not worth making every future reader
/// pay for delta decoding.
fn pick_winner(sizes: &[(Candidate, u64)]) -> usize {
    let best = sizes
        .iter()
        .enumerate()
        .min_by_key(|(_, (_, bytes))| *bytes)
        .expect("at least one candidate")
        .0;
    let best_bytes = sizes[best].1 as f64;

    sizes
        .iter()
        .enumerate()
        .filter(|(_, (_, bytes))| *bytes as f64 <= best_bytes * (1.0 + NEAR_TIE))
        .min_by_key(|(idx, (candidate, bytes))| (candidate.decode_cost(), *bytes, *idx))
        .expect("the best candidate is always within the tie window")
        .0
}

// ---------------------------------------------------------------------------
// The racer
// ---------------------------------------------------------------------------

/// What one racing write produced.
struct RaceOutcome {
    bytes: u64,
    elapsed: Duration,
    /// Per column, the candidates that won row groups and how often.
    chosen: Vec<(String, Vec<(String, usize)>)>,
}

/// Writes `batches` to `path`, racing candidate encodings per leaf per row
/// group and keeping only the smallest column chunk.
///
/// `dictionary_candidate` selects which dictionary policy the dictionary
/// candidate uses, so the caller can measure the effect of
/// [`DictionaryFallback::WhenProfitable`] against the stock byte limit.
fn race_write(
    path: &str,
    schema: &SchemaRef,
    batches: &[RecordBatch],
    dictionary_candidate: Candidate,
) -> Result<RaceOutcome> {
    let start = Instant::now();

    // File level properties. These govern everything the candidates do not:
    // compression, statistics, page and row group limits, bloom filters.
    let base_props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let file = File::create(path)?;
    let arrow_writer = ArrowWriter::try_new(file, schema.clone(), Some(base_props.clone()))?;
    let (mut file_writer, factory) = arrow_writer.into_serialized_writer()?;

    let mut leaves: Vec<LeafState> = file_writer
        .schema_descr()
        .columns()
        .iter()
        .map(|c| {
            let mut state = LeafState::new(c.path().clone(), c.physical_type());
            // Swap in the requested dictionary policy.
            for candidate in &mut state.candidates {
                if *candidate == Candidate::Dictionary {
                    *candidate = dictionary_candidate;
                }
            }
            state
        })
        .collect();

    let num_sets = leaves.iter().map(|l| l.candidates.len()).max().unwrap_or(1);

    for (row_group, group) in batches.chunks(ROW_GROUP_ROWS / BATCH_ROWS).enumerate() {
        // Which candidate each leaf uses in each candidate set.
        let plan: Vec<Vec<Candidate>> = (0..num_sets)
            .map(|k| {
                leaves
                    .iter()
                    .map(|leaf| leaf.candidate_for_set(row_group, k))
                    .collect()
            })
            .collect();

        // One full set of column writers per candidate set. This is the
        // headline cost of the approach: K times the column writers, K times
        // the encoding work, for every racing row group.
        let mut writer_sets: Vec<Vec<ArrowColumnWriter>> = Vec::with_capacity(num_sets);
        for set in &plan {
            let mut builder = base_props.clone().into_builder();
            for (leaf, candidate) in leaves.iter().zip(set) {
                builder = candidate.apply(builder, leaf.path.clone(), leaf.physical);
            }
            writer_sets.push(
                factory
                    .create_column_writers_with_properties(row_group, &Arc::new(builder.build()))?,
            );
        }

        // Feed every leaf of every batch to every candidate set. The leaves are
        // computed once and borrowed by each set's writer.
        for batch in group {
            for (field, column) in schema.fields().iter().zip(batch.columns()) {
                let computed = compute_leaves(field.as_ref(), column)?;
                for (leaf_offset, leaf) in computed.iter().enumerate() {
                    let idx = leaf_index(schema, field.name(), leaf_offset, &leaves);
                    for writers in writer_sets.iter_mut() {
                        writers[idx].write(leaf)?;
                    }
                }
            }
        }

        // Close every candidate's writers and keep the winners.
        // `Option` so a winning chunk can be moved out while the rest of the
        // set stays in place to be dropped.
        let mut closed: Vec<Vec<Option<ArrowColumnChunk>>> = Vec::with_capacity(num_sets);
        for writers in writer_sets {
            let chunks = writers
                .into_iter()
                .map(|w| w.close().map(Some))
                .collect::<Result<Vec<_>>>()?;
            closed.push(chunks);
        }

        let mut winners: Vec<Option<ArrowColumnChunk>> = (0..leaves.len()).map(|_| None).collect();
        for (idx, leaf) in leaves.iter_mut().enumerate() {
            // Deduplicate the sets that ran the same candidate for this leaf.
            let mut seen: Vec<Candidate> = Vec::new();
            let mut distinct: Vec<(usize, Candidate, u64)> = Vec::new();
            for (k, set) in plan.iter().enumerate() {
                let candidate = set[idx];
                if seen.contains(&candidate) {
                    continue;
                }
                seen.push(candidate);
                let bytes = closed[k][idx]
                    .as_ref()
                    .expect("chunk is still present")
                    .close()
                    .metadata
                    .compressed_size() as u64;
                distinct.push((k, candidate, bytes));
            }

            let sizes: Vec<(Candidate, u64)> = distinct.iter().map(|(_, c, b)| (*c, *b)).collect();
            let choice = pick_winner(&sizes);
            let (winning_set, winning_candidate, best_bytes) = distinct[choice];

            // Report what the chunk actually contains, not just which
            // candidate produced it: a dictionary candidate that overflowed
            // and fell back is materially different from one that kept its
            // dictionary, and the label alone would hide that.
            let winning_chunk = closed[winning_set][idx]
                .as_ref()
                .expect("winning chunk is still present");
            let mut encodings: Vec<String> = winning_chunk
                .close()
                .metadata
                .encodings()
                .map(|e| e.to_string())
                .collect();
            encodings.sort();
            encodings.dedup();
            let key = format!("{} [{}]", winning_candidate.label(), encodings.join("+"));
            *leaf.wins.entry(key).or_default() += 1;

            // Settle the leaf when this row group was decisive.
            if leaf.is_racing(row_group) && distinct.len() > 1 {
                let worst = distinct
                    .iter()
                    .map(|(_, _, b)| *b)
                    .max()
                    .unwrap_or(best_bytes);
                let gap = if worst == 0 {
                    0.0
                } else {
                    (worst - best_bytes) as f64 / worst as f64
                };
                if gap >= SETTLE_GAP {
                    leaf.settled = Some((winning_candidate, row_group));
                }
            }

            // Keep the winning chunk; every other chunk for this leaf is
            // dropped, taking its buffered pages with it.
            winners[idx] = closed[winning_set][idx].take();
            for chunks in closed.iter_mut() {
                chunks[idx] = None;
            }
        }
        drop(closed);

        let mut rg = file_writer.next_row_group()?;
        for winner in winners {
            winner
                .expect("every leaf selects a winning chunk")
                .append_to_row_group(&mut rg)?;
        }
        rg.close()?;
    }

    let metadata = file_writer.close()?;
    let bytes = std::fs::metadata(path)?.len();
    let _ = metadata;

    let chosen = leaves
        .iter()
        .map(|leaf| {
            let mut wins: Vec<(String, usize)> = leaf
                .wins
                .iter()
                .map(|(label, count)| (label.clone(), *count))
                .collect();
            wins.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            (leaf.path.string(), wins)
        })
        .collect();

    Ok(RaceOutcome {
        bytes,
        elapsed: start.elapsed(),
        chosen,
    })
}

/// Index of the `leaf_offset`th leaf of `field_name` among all leaves.
///
/// The datasets here are flat, so each field is exactly one leaf; this keeps
/// the mapping explicit rather than implicit in iteration order.
fn leaf_index(
    schema: &SchemaRef,
    field_name: &str,
    leaf_offset: usize,
    leaves: &[LeafState],
) -> usize {
    let field_pos = schema.index_of(field_name).expect("field is in the schema");
    debug_assert_eq!(schema.fields().len(), leaves.len(), "flat schema expected");
    field_pos + leaf_offset
}

// ---------------------------------------------------------------------------
// Baseline
// ---------------------------------------------------------------------------

/// Writes `batches` with a stock [`ArrowWriter`] and default properties.
fn baseline_write(
    path: &str,
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<(u64, Duration)> {
    let start = Instant::now();
    // Same row group sizing as the racer, so the comparison isolates the
    // encoding choice rather than measuring row group count.
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
        .build();
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok((std::fs::metadata(path)?.len(), start.elapsed()))
}

/// Writes `batches` with one fixed candidate applied to every column, to
/// isolate a single encoding's cost.
fn fixed_candidate_write(
    path: &str,
    schema: &SchemaRef,
    batches: &[RecordBatch],
    candidate: Candidate,
    compression: Compression,
) -> Result<u64> {
    let mut builder = WriterProperties::builder()
        .set_compression(compression)
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS));
    let descr = parquet::arrow::ArrowSchemaConverter::new().convert(schema)?;
    for col in descr.columns() {
        builder = candidate.apply(builder, col.path().clone(), col.physical_type());
    }
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(builder.build()))?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok(std::fs::metadata(path)?.len())
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Reads `path` back and asserts exact row equality against `batches`.
fn verify(path: &str, schema: &SchemaRef, batches: &[RecordBatch]) -> Result<()> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(BATCH_ROWS)
        .build()?;
    let read: Vec<RecordBatch> = reader.collect::<std::result::Result<_, _>>()?;

    let expected = concat_batches(schema, batches).expect("concat source batches");
    let actual = concat_batches(schema, &read).expect("concat read batches");
    assert_eq!(
        expected.num_rows(),
        actual.num_rows(),
        "{path}: row count mismatch"
    );
    assert_eq!(expected, actual, "{path}: rows differ after roundtrip");
    Ok(())
}

// ---------------------------------------------------------------------------
// Deterministic data generation
// ---------------------------------------------------------------------------

/// SplitMix64: a tiny, deterministic, seeded PRNG, so the example needs no
/// random number crate and reports identical numbers on every run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// A double in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One named dataset: a schema and its batches.
struct Dataset {
    name: &'static str,
    description: &'static str,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

/// Builds batches of `TOTAL_ROWS` rows from a per-batch array builder.
fn build_dataset(
    name: &'static str,
    description: &'static str,
    field: Field,
    mut make: impl FnMut(usize, usize) -> ArrayRef,
) -> Dataset {
    let schema = Arc::new(Schema::new(vec![field]));
    let mut batches = Vec::with_capacity(TOTAL_ROWS / BATCH_ROWS);
    let mut offset = 0;
    while offset < TOTAL_ROWS {
        let len = BATCH_ROWS.min(TOTAL_ROWS - offset);
        let array = make(offset, len);
        batches.push(RecordBatch::try_new(schema.clone(), vec![array]).expect("valid batch"));
        offset += len;
    }
    Dataset {
        name,
        description,
        schema,
        batches,
    }
}

/// (1) Low cardinality strings: 1000 distinct category labels.
fn dataset_low_cardinality_strings() -> Dataset {
    let pool: Vec<String> = (0..1000).map(|i| format!("category-{i:04}")).collect();
    let mut rng = Rng::new(0x5EED_0001);
    build_dataset(
        "low-cardinality strings",
        "1000 distinct category labels, uniformly sampled",
        Field::new("category", DataType::Utf8, false),
        move |_, len| {
            let values: Vec<&str> = (0..len)
                .map(|_| pool[rng.next_usize(pool.len())].as_str())
                .collect();
            Arc::new(StringArray::from(values)) as ArrayRef
        },
    )
}

/// Events sharing each distinct millisecond in dataset (2).
const EVENTS_PER_MS: usize = 12;

/// (2) High cardinality int64 millisecond timestamps that overflow the
/// dictionary: ascending, with [`EVENTS_PER_MS`] events sharing each distinct
/// millisecond. A row group sees 100000/12 = 8333 distinct values, so the
/// dictionary passes the 64 KiB page size limit, but each value arrives as a
/// consecutive run, so by the time the limit is reached the dictionary has
/// already absorbed many times its own size. That is exactly the case
/// [`DictionaryFallback::WhenProfitable`] exists for: the run structure is what
/// makes the dictionary profitable, and a column whose repeats were scattered
/// uniformly instead would saturate the dictionary before the repetition
/// accumulated, and would correctly not be judged profitable.
fn dataset_high_cardinality_timestamps() -> Dataset {
    let base = 1_700_000_000_000i64;
    build_dataset(
        "high-cardinality timestamps",
        "ascending int64 epoch millis, 12 events per distinct millisecond",
        Field::new("event_time_ms", DataType::Int64, false),
        move |offset, len| {
            let values: Vec<i64> = (0..len)
                .map(|i| base + ((offset + i) / EVENTS_PER_MS) as i64)
                .collect();
            Arc::new(Int64Array::from(values)) as ArrayRef
        },
    )
}

/// (3) f64 measurements: effectively unique, no exploitable structure.
fn dataset_floats() -> Dataset {
    let mut rng = Rng::new(0x5EED_0003);
    build_dataset(
        "f64 measurements",
        "pseudo-random doubles, effectively all distinct",
        Field::new("value", DataType::Float64, false),
        move |_, len| {
            let values: Vec<f64> = (0..len).map(|_| rng.next_f64() * 1000.0).collect();
            Arc::new(Float64Array::from(values)) as ArrayRef
        },
    )
}

/// (4) A string column that changes character half way through: 200 distinct
/// labels for the first million rows, then unique identifiers. A leaf that
/// settles on the dictionary early must reopen to notice.
fn dataset_shifting_strings() -> Dataset {
    let pool: Vec<String> = (0..200).map(|i| format!("tenant-{i:03}")).collect();
    let mut rng = Rng::new(0x5EED_0004);
    build_dataset(
        "shifting strings",
        "200 distinct labels for the first half, unique ids for the second",
        Field::new("key", DataType::Utf8, false),
        move |offset, len| {
            let values: Vec<String> = (0..len)
                .map(|i| {
                    let row = offset + i;
                    if row < TOTAL_ROWS / 2 {
                        pool[rng.next_usize(pool.len())].clone()
                    } else {
                        format!("req-{row:012}-{:08x}", rng.next_u64() as u32)
                    }
                })
                .collect();
            Arc::new(StringArray::from(values)) as ArrayRef
        },
    )
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("parquet_racing_writer");
    std::fs::create_dir_all(&dir)?;

    let datasets = vec![
        dataset_low_cardinality_strings(),
        dataset_high_cardinality_timestamps(),
        dataset_floats(),
        dataset_shifting_strings(),
    ];

    println!(
        "racing parquet writer: {TOTAL_ROWS} rows per dataset, {ROW_GROUP_ROWS} rows per row group ({} row groups)",
        TOTAL_ROWS / ROW_GROUP_ROWS
    );
    println!(
        "compression: SNAPPY, dictionary page size limit for dictionary candidates: {DICT_PAGE_SIZE_LIMIT} bytes\n"
    );

    for dataset in &datasets {
        let raced_path = dir.join(format!("{}.raced.parquet", dataset.name.replace(' ', "_")));
        let baseline_path = dir.join(format!(
            "{}.baseline.parquet",
            dataset.name.replace(' ', "_")
        ));
        let raced_path = raced_path.to_str().expect("utf8 path");
        let baseline_path = baseline_path.to_str().expect("utf8 path");

        let raced = race_write(
            raced_path,
            &dataset.schema,
            &dataset.batches,
            Candidate::Dictionary,
        )?;
        let (baseline_bytes, baseline_time) =
            baseline_write(baseline_path, &dataset.schema, &dataset.batches)?;

        verify(raced_path, &dataset.schema, &dataset.batches)?;
        verify(baseline_path, &dataset.schema, &dataset.batches)?;

        println!("=== {} ===", dataset.name);
        println!("    {}", dataset.description);
        println!(
            "    baseline (ArrowWriter defaults): {:>9.2} MiB  in {:>7.2?}",
            mib(baseline_bytes),
            baseline_time
        );
        let saved = 100.0 * (baseline_bytes as f64 - raced.bytes as f64) / baseline_bytes as f64;
        println!(
            "    raced:                           {:>9.2} MiB  in {:>7.2?}   ({saved:+.1}% smaller than baseline)",
            mib(raced.bytes),
            raced.elapsed
        );
        for (column, wins) in &raced.chosen {
            let summary: Vec<String> = wins
                .iter()
                .map(|(label, count)| format!("{label} x{count}"))
                .collect();
            println!("    column {column:<16} chose  {}", summary.join(", "));
        }

        // The dictionary fallback knob, isolated: the same dictionary
        // candidate with `WhenProfitable` and with the stock byte limit.
        if dataset.name == "high-cardinality timestamps" {
            println!("    DictionaryFallback knob, dictionary candidate in isolation:");
            for (codec, name) in [
                (Compression::UNCOMPRESSED, "uncompressed"),
                (Compression::SNAPPY, "snappy"),
            ] {
                let on = fixed_candidate_write(
                    &format!("{raced_path}.dict_on"),
                    &dataset.schema,
                    &dataset.batches,
                    Candidate::Dictionary,
                    codec,
                )?;
                let off = fixed_candidate_write(
                    &format!("{raced_path}.dict_off"),
                    &dataset.schema,
                    &dataset.batches,
                    Candidate::DictionaryOnPageSizeLimit,
                    codec,
                )?;
                println!(
                    "      {name:<13} WhenProfitable {:>8.2} MiB   OnPageSizeLimit {:>8.2} MiB   ({:+.1}% smaller)",
                    mib(on),
                    mib(off),
                    100.0 * (off as f64 - on as f64) / off as f64
                );
            }
        }
        println!();
    }

    println!("all files verified: every dataset read back with exact row equality");
    Ok(())
}

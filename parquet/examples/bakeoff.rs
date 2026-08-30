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

//! A head-to-head bakeoff of four parquet writing strategies over identical
//! inputs, identical row group boundaries and identical writer properties.
//!
//! 1. **Baseline** — a stock [`ArrowWriter`] at the shared properties.
//! 2. **Option A** — `examples/adaptive_writer/harness.rs` with
//!    `always_page_grain` set: every decidable leaf stays on the page-grain API
//!    (`parquet::arrow::arrow_writer::page_grain`) for the whole file, encoding
//!    each page several ways, charging each page the dictionary bytes it
//!    creates, settling on a winner, watching the dictionary, and carrying the
//!    settled choice across row groups.
//! 3. **Option B** — a K-full-column-writer racer over
//!    `ArrowRowGroupWriterFactory::create_column_writers_with_properties` and
//!    `DictionaryFallback::WhenProfitable`: encode every leaf K times per row
//!    group with K whole sets of column writers, keep the smallest chunk, drop
//!    the rest. Adapted from `advanced_racing_writer`.
//! 4. **Option C** — the same harness as it ships: each leaf takes the
//!    page-grain path while it is still deciding, and an ordinary column writer
//!    once it has settled, so only the leaves actually making a decision leave
//!    the normal write path. A and C are the same code, differing in the
//!    routing rule alone.
//!
//! The point of this example is that the four writers see *exactly* the same
//! bytes in, the same rows per row group, the same compression and the same
//! page size limits, so the reported sizes and times differ only because of the
//! encoding decisions each one makes.
//!
//! Run the synthetic suite with:
//!
//! ```text
//! cargo run --release --features "arrow snap zstd" --example bakeoff
//! ```
//!
//! Or rewrite real parquet files through the same four writers, reporting to
//! stdout only:
//!
//! ```text
//! cargo run --release --features "arrow snap zstd" --example bakeoff -- a.parquet b.parquet
//! ```

use std::fmt::Write as _;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use parquet::arrow::ArrowSchemaConverter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::{
    ArrowColumnChunk, ArrowColumnWriter, ArrowWriter, compute_leaves,
};
use parquet::basic::{Compression, Encoding, PageType, Type as PhysicalType};
use parquet::errors::{ParquetError, Result};
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::properties::{WriterProperties, WriterPropertiesBuilder};
use parquet::schema::types::{ColumnDescPtr, ColumnPath};

// ---------------------------------------------------------------------------
// Shared configuration
//
// Every knob here is applied identically to all four writers. Nothing below
// may set a property that is not also set for the other two.
// ---------------------------------------------------------------------------

#[path = "adaptive_writer/harness.rs"]
mod harness;

/// Rows per synthetic dataset.
const ROWS: usize = 2_000_000;
/// Rows per row group: 2M rows over 20 row groups.
const ROW_GROUP_ROWS: usize = 100_000;
/// Rows per record batch fed to the writers.
const BATCH_ROWS: usize = 25_000;
/// Page size limit, in rows, shared by all four writers.
const DATA_PAGE_ROW_LIMIT: usize = 20_000;
/// Timed runs per (dataset, compression, writer). The median is reported.
const RUNS: usize = 3;

/// Option B settles a leaf when best and worst differ by at least this much.
const SETTLE_GAP: f64 = 0.10;
/// Option B re-races a settled leaf every this many row groups.
const B_REOPEN_EVERY: usize = 8;

/// The dictionary policy, the tie window and the re-open cadence are the
/// shipped harness's, so options A, B and C are tuned identically.
use harness::{
    AdaptiveWriter, DICT_PAGE_SIZE_LIMIT, DICT_WORTH_RATIO, NEAR_TIE,
    REOPEN_EVERY as A_REOPEN_EVERY, dictionary_properties, is_decidable,
};

/// The shared property set. `compression` is the only thing that varies.
fn shared_properties(compression: Compression) -> WriterPropertiesBuilder {
    WriterProperties::builder()
        .set_compression(compression)
        .set_data_page_row_count_limit(DATA_PAGE_ROW_LIMIT)
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
}

/// The compressions to measure: uncompressed always, plus ZSTD when the feature
/// is on and SNAPPY otherwise.
fn compressions() -> Vec<(&'static str, Compression)> {
    #[cfg(feature = "zstd")]
    let second = (
        "zstd",
        Compression::ZSTD(parquet::basic::ZstdLevel::default()),
    );
    #[cfg(not(feature = "zstd"))]
    let second = ("snappy", Compression::SNAPPY);

    vec![("none", Compression::UNCOMPRESSED), second]
}

// ---------------------------------------------------------------------------
// Deterministic data generation
// ---------------------------------------------------------------------------

/// splitmix64, so every run produces byte-identical inputs.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next_u64(&mut self) -> u64 {
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

/// Builds one of the synthetic datasets.
type DatasetBuilder = fn() -> Dataset;

struct Dataset {
    name: &'static str,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

/// Builds a dataset batch by batch, so nothing larger than one batch of columns
/// is materialised at a time.
fn build<F>(name: &'static str, schema: SchemaRef, mut make: F) -> Dataset
where
    F: FnMut(usize, usize) -> Vec<ArrayRef>,
{
    let batches = (0..ROWS)
        .step_by(BATCH_ROWS)
        .map(|start| {
            let len = BATCH_ROWS.min(ROWS - start);
            RecordBatch::try_new(schema.clone(), make(start, len)).unwrap()
        })
        .collect();
    Dataset {
        name,
        schema,
        batches,
    }
}

/// (1) Low-cardinality strings: the dictionary should win everywhere.
fn low_cardinality_strings() -> Dataset {
    let schema = Arc::new(Schema::new(vec![Field::new("tag", DataType::Utf8, false)]));
    let vocabulary: Vec<String> = (0..48).map(|i| format!("service-{i:02}-eu-west")).collect();
    let mut rng = Rng::new(1);
    build("low-cardinality strings", schema, move |_, len| {
        let values: Vec<&str> = (0..len)
            .map(|_| vocabulary[rng.below(vocabulary.len() as u64) as usize].as_str())
            .collect();
        vec![Arc::new(StringArray::from(values)) as ArrayRef]
    })
}

/// (2) High-cardinality ascending int64 timestamps: overflows any dictionary,
/// and is exactly what delta encoding is for.
fn timestamps() -> Dataset {
    let schema = Arc::new(Schema::new(vec![Field::new("ts", DataType::Int64, false)]));
    let mut rng = Rng::new(2);
    let mut now: i64 = 1_700_000_000_000;
    build(
        "high-cardinality int64 timestamps",
        schema,
        move |_, len| {
            let values: Vec<i64> = (0..len)
                .map(|_| {
                    now += rng.below(4_000) as i64;
                    now
                })
                .collect();
            vec![Arc::new(Int64Array::from(values)) as ArrayRef]
        },
    )
}

/// (3) f64 measurements: nothing compresses these well, and the point is that a
/// writer should notice quickly and stop paying to find out.
fn floats() -> Dataset {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        false,
    )]));
    let mut rng = Rng::new(3);
    build("f64 measurements", schema, move |_, len| {
        let values: Vec<f64> = (0..len).map(|_| rng.unit() * 1_000.0).collect();
        vec![Arc::new(Float64Array::from(values)) as ArrayRef]
    })
}

/// (4) Strings that change character halfway: dictionary-friendly first half,
/// unique-per-row second half. A writer that decides once at the top of the
/// file gets the second half wrong.
fn shifting_strings() -> Dataset {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Utf8, false)]));
    let vocabulary: Vec<String> = (0..32).map(|i| format!("region-{i:02}")).collect();
    let mut rng = Rng::new(4);
    build("shifting strings", schema, move |start, len| {
        let values: Vec<String> = (0..len)
            .map(|i| {
                if start + i < ROWS / 2 {
                    vocabulary[rng.below(vocabulary.len() as u64) as usize].clone()
                } else {
                    format!("urn:evt:{:016x}", rng.next_u64())
                }
            })
            .collect();
        vec![Arc::new(StringArray::from(values)) as ArrayRef]
    })
}

/// (5) A records-like schema mixing all of the above shapes in one file, which
/// is where a per-column decision has to be made per column rather than per
/// file.
fn records() -> Dataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("service", DataType::Utf8, false),
        Field::new("event_time_ms", DataType::Int64, false),
        Field::new("latency_ms", DataType::Float64, false),
        Field::new("trace_id", DataType::Utf8, false),
        Field::new("status_code", DataType::Int32, false),
    ]));
    let services: Vec<String> = (0..24).map(|i| format!("svc-{i:02}")).collect();
    let statuses = [200i32, 200, 200, 201, 204, 301, 400, 404, 500];
    let mut rng = Rng::new(5);
    let mut now: i64 = 1_700_000_000_000;
    build("records-like mixed schema", schema, move |start, len| {
        let service: Vec<&str> = (0..len)
            .map(|_| services[rng.below(services.len() as u64) as usize].as_str())
            .collect();
        let event_time: Vec<i64> = (0..len)
            .map(|_| {
                now += rng.below(500) as i64;
                now
            })
            .collect();
        let latency: Vec<f64> = (0..len).map(|_| rng.unit() * 250.0).collect();
        let trace: Vec<String> = (0..len)
            .map(|i| {
                if start + i < ROWS / 2 {
                    format!("trace-pool-{:03}", rng.below(200))
                } else {
                    format!("{:032x}", rng.next_u64())
                }
            })
            .collect();
        let status: Vec<i32> = (0..len)
            .map(|_| statuses[rng.below(statuses.len() as u64) as usize])
            .collect();
        vec![
            Arc::new(StringArray::from(service)) as ArrayRef,
            Arc::new(Int64Array::from(event_time)) as ArrayRef,
            Arc::new(Float64Array::from(latency)) as ArrayRef,
            Arc::new(StringArray::from(trace)) as ArrayRef,
            Arc::new(Int32Array::from(status)) as ArrayRef,
        ]
    })
}

// ---------------------------------------------------------------------------
// Row group boundaries
//
// All four writers must cut row groups in exactly the same places, or the
// byte comparison is measuring row group count rather than encoding choice.
// The baseline gets there via `max_row_group_row_count`; A and B drive the
// boundaries themselves and use this exact split.
// ---------------------------------------------------------------------------

/// Splits `batches` into groups with exactly the row counts in `sizes`.
///
/// This is what makes the four writers comparable: they all cut row groups in
/// the same places. For a supplied file the sizes are the input's own row group
/// row counts, which are often uneven, so a single "rows per row group" number
/// cannot reproduce them.
fn split_into_row_groups(batches: &[RecordBatch], sizes: &[usize]) -> Vec<Vec<RecordBatch>> {
    let mut groups: Vec<Vec<RecordBatch>> = Vec::new();
    let mut current: Vec<RecordBatch> = Vec::new();
    let mut filled = 0usize;
    let mut target = sizes.first().copied().unwrap_or(usize::MAX).max(1);
    let mut next_size = 1usize;

    for batch in batches {
        let mut offset = 0usize;
        while offset < batch.num_rows() {
            let take = (target - filled).min(batch.num_rows() - offset);
            current.push(batch.slice(offset, take));
            offset += take;
            filled += take;
            if filled == target {
                groups.push(std::mem::take(&mut current));
                filled = 0;
                // Any rows beyond the sizes given continue at the last size.
                target = sizes
                    .get(next_size)
                    .copied()
                    .unwrap_or(*sizes.last().unwrap_or(&target))
                    .max(1);
                next_size += 1;
            }
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

// ---------------------------------------------------------------------------
// Leaf classification, shared by both harnesses
//
// A leaf either has candidates worth racing, or it is passed through on a
// single candidate. Both harnesses use the same predicate, so a column that one
// harness passes through is passed through by the other too, and the two
// columns of the table stay comparable.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Options A and C: the adaptive writer
//
// Both arms are `examples/adaptive_writer/harness.rs`, the harness this branch
// ships, driven over the same batches and the same row group boundaries as
// every other arm. They differ in one field:
//
// * **Option C** is the harness as it ships: a leaf takes the page-grain path
//   while it is still deciding, and an ordinary column writer once it has
//   settled.
// * **Option A** sets `always_page_grain`, so every decidable leaf stays on the
//   page-grain path for the whole file. That is the pure page-grain arm, kept
//   to price what the routing rule saves and costs.
// ---------------------------------------------------------------------------

/// Writes a whole file with the adaptive writer.
fn write_adaptive(
    schema: &SchemaRef,
    groups: &[Vec<RecordBatch>],
    props: &WriterProperties,
    path: &Path,
    always_page_grain: bool,
) -> Result<Duration> {
    let start = Instant::now();
    let mut writer = AdaptiveWriter::try_new(File::create(path)?, schema.clone(), props.clone())?;
    for policy in &mut writer.policies {
        policy.always_page_grain = always_page_grain;
    }

    for group in groups {
        for batch in group {
            writer.write(batch)?;
        }
        // Ends the row group; a no-op if the group contributed no rows.
        writer.flush()?;
    }
    writer.close()?;
    Ok(start.elapsed())
}

fn write_option_a(
    schema: &SchemaRef,
    groups: &[Vec<RecordBatch>],
    props: &WriterProperties,
    path: &Path,
) -> Result<Duration> {
    write_adaptive(schema, groups, props, path, true)
}

fn write_option_c(
    schema: &SchemaRef,
    groups: &[Vec<RecordBatch>],
    props: &WriterProperties,
    path: &Path,
) -> Result<Duration> {
    write_adaptive(schema, groups, props, path, false)
}

/// The leaves both harnesses pass through on a single candidate rather than
/// racing, for reporting. Both use `is_decidable`, so the list is shared.
fn passthrough_leaves(schema: &SchemaRef, props: &WriterProperties) -> Result<Vec<String>> {
    let parquet_schema = ArrowSchemaConverter::new()
        .with_coerce_types(props.coerce_types())
        .convert(schema)?;
    Ok(parquet_schema
        .columns()
        .iter()
        .filter(|d| !is_decidable(d))
        .map(|d| d.path().string())
        .collect())
}

// ---------------------------------------------------------------------------
// Option B: the K-full-column-writer racer
// ---------------------------------------------------------------------------

/// One candidate encoding strategy for a leaf column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CandidateB {
    /// Dictionary encoding, retaining the dictionary while it is profitable.
    Dictionary,
    /// No dictionary, PLAIN.
    Plain,
    /// No dictionary, DELTA_BINARY_PACKED or DELTA_BYTE_ARRAY by physical type.
    Delta,
    /// The file's own properties, untouched: used for leaves with no useful
    /// race, so that a passthrough column is written exactly as the baseline
    /// would write it.
    Passthrough,
}

impl CandidateB {
    /// Relative cost of decoding, lowest first. Only breaks near-ties.
    fn decode_cost(self) -> u8 {
        match self {
            CandidateB::Dictionary | CandidateB::Passthrough => 0,
            CandidateB::Plain => 1,
            CandidateB::Delta => 2,
        }
    }

    fn delta_encoding(physical: PhysicalType) -> Option<Encoding> {
        match physical {
            PhysicalType::INT32 | PhysicalType::INT64 => Some(Encoding::DELTA_BINARY_PACKED),
            PhysicalType::BYTE_ARRAY => Some(Encoding::DELTA_BYTE_ARRAY),
            _ => None,
        }
    }

    /// Candidates worth racing for `descr`.
    fn for_leaf(descr: &ColumnDescPtr) -> Vec<CandidateB> {
        if !is_decidable(descr) {
            return vec![CandidateB::Passthrough];
        }
        let mut out = vec![CandidateB::Dictionary, CandidateB::Plain];
        if Self::delta_encoding(descr.physical_type()).is_some() {
            out.push(CandidateB::Delta);
        }
        out
    }

    /// Applies this candidate to `col` in `builder`, using the #10775
    /// `DictionaryFallback::WhenProfitable` policy for the dictionary.
    fn apply(
        self,
        builder: WriterPropertiesBuilder,
        col: ColumnPath,
        physical: PhysicalType,
    ) -> WriterPropertiesBuilder {
        self.apply_with(builder, col, physical, true)
    }

    /// Applies this candidate to `col` in `builder`.
    ///
    /// With `fallback` false the dictionary candidate is configured with stock
    /// upstream behaviour alone: dictionary on, and the ordinary byte-size
    /// `dictionary_page_size_limit` check that every released writer already
    /// has. That is the tier-0-minimal configuration, which assumes the #10775
    /// `DictionaryFallback` port is not available.
    fn apply_with(
        self,
        builder: WriterPropertiesBuilder,
        col: ColumnPath,
        physical: PhysicalType,
        fallback: bool,
    ) -> WriterPropertiesBuilder {
        match self {
            CandidateB::Passthrough => builder,
            CandidateB::Dictionary if !fallback => builder
                .set_column_dictionary_enabled(col.clone(), true)
                .set_column_dictionary_page_size_limit(col, DICT_PAGE_SIZE_LIMIT),
            CandidateB::Dictionary => dictionary_properties(builder, col),
            CandidateB::Plain => builder
                .set_column_dictionary_enabled(col.clone(), false)
                .set_column_encoding(col, Encoding::PLAIN),
            CandidateB::Delta => {
                let encoding = Self::delta_encoding(physical)
                    .expect("Delta candidate built for a type without a delta encoding");
                builder
                    .set_column_dictionary_enabled(col.clone(), false)
                    .set_column_encoding(col, encoding)
            }
        }
    }
}

/// What Option B knows about one leaf column.
struct LeafStateB {
    path: ColumnPath,
    physical: PhysicalType,
    candidates: Vec<CandidateB>,
    /// The candidate this leaf has settled on, and the row group it settled at.
    settled: Option<(CandidateB, usize)>,
}

impl LeafStateB {
    fn new(descr: &ColumnDescPtr) -> Self {
        Self {
            path: descr.path().clone(),
            physical: descr.physical_type(),
            candidates: CandidateB::for_leaf(descr),
            settled: None,
        }
    }

    fn is_racing(&self, row_group: usize) -> bool {
        if self.candidates.len() < 2 {
            return false;
        }
        match self.settled {
            None => true,
            Some((_, at)) => row_group.saturating_sub(at) >= B_REOPEN_EVERY,
        }
    }

    /// The candidate this leaf uses in candidate set `k` of `row_group`. A leaf
    /// with fewer candidates than the widest leaf repeats its last candidate in
    /// the surplus sets; those are deduplicated when the winner is chosen.
    fn candidate_for_set(&self, row_group: usize, k: usize) -> CandidateB {
        if !self.is_racing(row_group)
            && let Some((settled, _)) = self.settled
        {
            return settled;
        }
        self.candidates[k.min(self.candidates.len() - 1)]
    }
}

/// Chooses the winning candidate set for a leaf from the sizes each produced.
fn pick_chunk(sizes: &[(CandidateB, u64)]) -> usize {
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

/// Writes a whole file, racing candidate encodings per leaf per row group and
/// keeping only the smallest column chunk.
fn write_option_b(
    schema: &SchemaRef,
    groups: &[Vec<RecordBatch>],
    props: &WriterProperties,
    path: &Path,
) -> Result<Duration> {
    let start = Instant::now();

    let file = File::create(path)?;
    let arrow_writer = ArrowWriter::try_new(file, schema.clone(), Some(props.clone()))?;
    let (mut file_writer, factory) = arrow_writer.into_serialized_writer()?;

    let mut leaves: Vec<LeafStateB> = file_writer
        .schema_descr()
        .columns()
        .iter()
        .map(LeafStateB::new)
        .collect();

    let num_sets = leaves.iter().map(|l| l.candidates.len()).max().unwrap_or(1);

    for (row_group, group) in groups.iter().enumerate() {
        // Which candidate each leaf uses in each candidate set.
        let plan: Vec<Vec<CandidateB>> = (0..num_sets)
            .map(|k| {
                leaves
                    .iter()
                    .map(|leaf| leaf.candidate_for_set(row_group, k))
                    .collect()
            })
            .collect();

        // One full set of column writers per candidate set. This is the
        // headline cost of the approach: K times the column writers and K times
        // the encoding work, for every racing row group.
        let mut writer_sets: Vec<Vec<ArrowColumnWriter>> = Vec::with_capacity(num_sets);
        for set in &plan {
            let mut builder = props.clone().into_builder();
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
            // `compute_leaves` yields a field's leaves in schema descriptor
            // order, so a running index over the fields is correct for nested
            // fields as well as flat ones.
            let mut leaf_idx = 0usize;
            for (field, column) in schema.fields().iter().zip(batch.columns()) {
                for leaf in compute_leaves(field.as_ref(), column)? {
                    for writers in writer_sets.iter_mut() {
                        writers[leaf_idx].write(&leaf)?;
                    }
                    leaf_idx += 1;
                }
            }
        }

        // Close every candidate's writers and keep the winners. `Option` so a
        // winning chunk can be moved out while the rest of the set stays in
        // place to be dropped.
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
            let mut seen: Vec<CandidateB> = Vec::new();
            let mut distinct: Vec<(usize, CandidateB, u64)> = Vec::new();
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

            let sizes: Vec<(CandidateB, u64)> = distinct.iter().map(|(_, c, b)| (*c, *b)).collect();
            let choice = pick_chunk(&sizes);
            let (winning_set, winning_candidate, best_bytes) = distinct[choice];

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

    file_writer.close()?;
    Ok(start.elapsed())
}

// ---------------------------------------------------------------------------
// Tier 0: probe-then-commit at chunk grain
//
// The question this arm answers is how much of Option C's result survives if
// the library gains *only* the two small factory accessors this branch adds
// (`create_selected_column_writers` and `page_store_factory`), the #10775
// dictionary-fallback port, and nothing else. In particular it never touches
// `parquet::arrow::arrow_writer::page_grain`, so it needs no
// `GenericColumnWriter` seam and no new page-level API at all.
//
// Per leaf, per row group:
//
// * A **settled** leaf gets one ordinary `ArrowColumnWriter` configured with
//   the candidate it settled on, and writes the whole row group through it.
//   That is the library's normal write path at its normal speed, exactly as in
//   Option C.
// * A **deciding** leaf is probed first: K throwaway `ArrowColumnWriter`s are
//   created for that one leaf through `create_selected_column_writers`, fed the
//   *first* `TIER0_PROBE_ROWS` rows of the row group, and closed. Their
//   `ColumnCloseResult` compressed sizes pick a winner by the same near-tie and
//   decode-rank rule Option B uses; the probe chunks are then dropped. The leaf
//   then writes its whole row group, probe rows included, through one real
//   writer configured with that winner.
//
// The accepted cost is that a deciding leaf encodes its probe span K+1 times:
// K throwaway passes and one real one. That is the tier's only overhead knob,
// and it is bounded by the probe span rather than by the row group.
//
// The candidate set, the near-tie rule, the settle gap and the re-open cadence
// are Option B's, so tier 0 and Option B differ in exactly one thing: whether
// the race sees the whole row group or only its first span.
// ---------------------------------------------------------------------------

/// Rows of each row group the tier-0 probe encodes K ways before committing.
///
/// One data page's worth: large enough that a dictionary candidate has built a
/// real dictionary and the compressor has something to work with, small enough
/// that the doubled encoding is a fraction of the row group rather than all of
/// it. Capped at the row group's own row count for a group shorter than this.
const TIER0_PROBE_ROWS: usize = DATA_PAGE_ROW_LIMIT;

/// The leading `rows` rows of a row group, cut at record batch boundaries with
/// a slice of the batch that straddles the cut.
fn probe_span(group: &[RecordBatch], rows: usize) -> Vec<RecordBatch> {
    let mut out = Vec::new();
    let mut taken = 0usize;
    for batch in group {
        if taken >= rows {
            break;
        }
        let take = (rows - taken).min(batch.num_rows());
        out.push(batch.slice(0, take));
        taken += take;
    }
    out
}

/// Which dictionary machinery the tier-0 arm is allowed to use.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier0Dictionary {
    /// The #10775 port is available: a dictionary candidate is configured with
    /// `DictionaryFallback::WhenProfitable`, and the library leaves a
    /// dictionary that stops paying without the harness being told.
    Fallback,
    /// The #10775 port is *not* available, simulating its rejection upstream. A
    /// dictionary candidate gets stock behaviour only: dictionary on, plus the
    /// byte-size `dictionary_page_size_limit` check every released writer
    /// already has. The harness then owns the dictionary heuristic itself, at
    /// chunk grain, from the closed chunk's public metadata.
    ChunkGrain,
}

/// What a finished column chunk's public metadata says about its dictionary.
///
/// Read from `ColumnChunkMetaData` alone: the dictionary page offset and the
/// per-page encoding stats, both of which every released writer already
/// records. This is the whole of the tier-0-minimal dictionary heuristic's
/// evidence.
fn dictionary_paid_off(metadata: &ColumnChunkMetaData) -> bool {
    if metadata.dictionary_page_offset().is_none() {
        // No dictionary page was written, so there is nothing to regret.
        return true;
    }
    let Some(stats) = metadata.page_encoding_stats() else {
        return true;
    };

    let mut dictionary_pages = 0i32;
    let mut other_pages = 0i32;
    for stat in stats {
        if stat.page_type == PageType::DICTIONARY_PAGE {
            continue;
        }
        match stat.encoding {
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => dictionary_pages += stat.count,
            _ => other_pages += stat.count,
        }
    }

    // A dictionary page was written and paid for. It only earned its keep if
    // every data page in the chunk actually indexed into it. A chunk that
    // overflowed mid-way carries both kinds of page and paid for a dictionary
    // covering only part of itself; a chunk with a dictionary page and no
    // dictionary data page paid for one nothing referenced at all.
    other_pages == 0 && dictionary_pages > 0
}

/// Writes a whole file, probing the first span of each row group for the leaves
/// still deciding and committing the row group to the probe's winner.
fn write_tier0_with(
    schema: &SchemaRef,
    groups: &[Vec<RecordBatch>],
    props: &WriterProperties,
    path: &Path,
    mode: Tier0Dictionary,
) -> Result<Duration> {
    let fallback = mode == Tier0Dictionary::Fallback;
    let start = Instant::now();

    let file = File::create(path)?;
    let arrow_writer = ArrowWriter::try_new(file, schema.clone(), Some(props.clone()))?;
    let (mut file_writer, factory) = arrow_writer.into_serialized_writer()?;

    let mut leaves: Vec<LeafStateB> = file_writer
        .schema_descr()
        .columns()
        .iter()
        .map(LeafStateB::new)
        .collect();

    // Tier-0-minimal only: the row group at which each leaf's dictionary was
    // last found not to have paid off. While a leaf is banned its dictionary
    // candidate is withheld, and the ban lapses on the ordinary re-open
    // cadence so a column whose data turns dictionary-friendly again can win
    // the dictionary back.
    let mut dictionary_banned_at: Vec<Option<usize>> = vec![None; leaves.len()];

    for (row_group, group) in groups.iter().enumerate() {
        if group.iter().all(|b| b.num_rows() == 0) {
            continue;
        }

        // The candidates each leaf may use this row group. Identical to the
        // leaf's own list except under tier-0-minimal, where a leaf whose
        // dictionary just failed to pay off has it withheld.
        let active: Vec<Vec<CandidateB>> = leaves
            .iter()
            .enumerate()
            .map(|(idx, leaf)| {
                let banned = dictionary_banned_at[idx]
                    .is_some_and(|at| row_group.saturating_sub(at) < B_REOPEN_EVERY);
                if !banned {
                    return leaf.candidates.clone();
                }
                let kept: Vec<CandidateB> = leaf
                    .candidates
                    .iter()
                    .copied()
                    .filter(|c| *c != CandidateB::Dictionary)
                    .collect();
                if kept.is_empty() {
                    leaf.candidates.clone()
                } else {
                    kept
                }
            })
            .collect();

        // A leaf settled on a dictionary it may no longer use has to decide
        // again, among what is left.
        for (idx, leaf) in leaves.iter_mut().enumerate() {
            if let Some((CandidateB::Dictionary, _)) = leaf.settled
                && !active[idx].contains(&CandidateB::Dictionary)
            {
                leaf.settled = None;
            }
        }

        // The candidate each leaf commits this row group to. A leaf that is not
        // deciding keeps what it settled on; a passthrough leaf keeps its only
        // candidate.
        let mut chosen: Vec<CandidateB> = leaves
            .iter()
            .enumerate()
            .map(|(idx, leaf)| {
                leaf.settled
                    .map(|(candidate, _)| candidate)
                    .unwrap_or(active[idx][0])
            })
            .collect();

        let deciding: Vec<bool> = leaves.iter().map(|l| l.is_racing(row_group)).collect();

        if deciding.iter().any(|d| *d) {
            let probe = probe_span(group, TIER0_PROBE_ROWS);
            let num_sets = active
                .iter()
                .zip(&deciding)
                .filter(|(_, d)| **d)
                .map(|(c, _)| c.len())
                .max()
                .unwrap_or(1);

            // Which candidate each deciding leaf uses in each probe set. A leaf
            // with fewer candidates than the widest repeats its last one; the
            // duplicates are deduplicated when the winner is chosen.
            let plan: Vec<Vec<CandidateB>> = (0..num_sets)
                .map(|k| {
                    active
                        .iter()
                        .map(|candidates| candidates[k.min(candidates.len() - 1)])
                        .collect()
                })
                .collect();

            // K throwaway writers, created only for the deciding leaves. A
            // settled leaf allocates nothing here: that is what
            // `create_selected_column_writers` buys.
            let mut probe_sets: Vec<Vec<Option<ArrowColumnWriter>>> = Vec::with_capacity(num_sets);
            for set in &plan {
                let mut builder = props.clone().into_builder();
                for (idx, leaf) in leaves.iter().enumerate() {
                    if deciding[idx] {
                        builder = set[idx].apply_with(
                            builder,
                            leaf.path.clone(),
                            leaf.physical,
                            fallback,
                        );
                    }
                }
                probe_sets.push(factory.create_selected_column_writers(
                    row_group,
                    &Arc::new(builder.build()),
                    |leaf| deciding[leaf],
                )?);
            }

            for batch in &probe {
                let mut leaf_idx = 0usize;
                for (field, column) in schema.fields().iter().zip(batch.columns()) {
                    for leaf in compute_leaves(field.as_ref(), column)? {
                        if deciding[leaf_idx] {
                            for writers in probe_sets.iter_mut() {
                                writers[leaf_idx]
                                    .as_mut()
                                    .expect("a deciding leaf has a probe writer")
                                    .write(&leaf)?;
                            }
                        }
                        leaf_idx += 1;
                    }
                }
            }

            // Close the probes and read what each candidate cost over the span.
            // The chunks themselves are dropped: only their sizes are kept.
            let mut probe_bytes: Vec<Vec<u64>> = Vec::with_capacity(num_sets);
            for writers in probe_sets {
                let mut sizes = Vec::with_capacity(writers.len());
                for writer in writers {
                    sizes.push(match writer {
                        None => 0,
                        Some(writer) => writer.close()?.close().metadata.compressed_size() as u64,
                    });
                }
                probe_bytes.push(sizes);
            }

            for (idx, leaf) in leaves.iter_mut().enumerate() {
                if !deciding[idx] {
                    continue;
                }
                // Deduplicate the sets that ran the same candidate for this leaf.
                let mut distinct: Vec<(CandidateB, u64)> = Vec::new();
                for (k, set) in plan.iter().enumerate() {
                    if distinct.iter().any(|(c, _)| *c == set[idx]) {
                        continue;
                    }
                    distinct.push((set[idx], probe_bytes[k][idx]));
                }

                let winner = pick_chunk(&distinct);
                let (winning_candidate, best_bytes) = distinct[winner];
                chosen[idx] = winning_candidate;

                // Settle on the same rule Option B uses, read off the probe
                // span rather than off the finished row group.
                if distinct.len() > 1 {
                    let worst = distinct.iter().map(|(_, b)| *b).max().unwrap_or(best_bytes);
                    let gap = if worst == 0 {
                        0.0
                    } else {
                        (worst - best_bytes) as f64 / worst as f64
                    };
                    if gap >= SETTLE_GAP {
                        leaf.settled = Some((winning_candidate, row_group));
                    }
                }
            }
        }

        // One ordinary set of writers at the committed candidates, and the whole
        // row group written through it: the probe rows are encoded a second
        // time here, for real.
        let mut builder = props.clone().into_builder();
        for (leaf, candidate) in leaves.iter().zip(&chosen) {
            builder = candidate.apply_with(builder, leaf.path.clone(), leaf.physical, fallback);
        }
        let mut writers =
            factory.create_column_writers_with_properties(row_group, &Arc::new(builder.build()))?;

        for batch in group {
            let mut leaf_idx = 0usize;
            for (field, column) in schema.fields().iter().zip(batch.columns()) {
                for leaf in compute_leaves(field.as_ref(), column)? {
                    writers[leaf_idx].write(&leaf)?;
                    leaf_idx += 1;
                }
            }
        }

        let mut rg = file_writer.next_row_group()?;
        for (idx, writer) in writers.into_iter().enumerate() {
            let chunk = writer.close()?;

            // The tier-0-minimal dictionary heuristic, and the only thing that
            // separates the two tier-0 variants once the properties are set:
            // look at what the finished chunk says about its dictionary, and
            // withhold the dictionary candidate from the next row groups if it
            // did not pay off. Nothing here is a new API; it is the metadata
            // the writer already returns.
            if mode == Tier0Dictionary::ChunkGrain && chosen[idx] == CandidateB::Dictionary {
                if dictionary_paid_off(&chunk.close().metadata) {
                    dictionary_banned_at[idx] = None;
                } else {
                    dictionary_banned_at[idx] = Some(row_group + 1);
                    leaves[idx].settled = None;
                }
            }

            chunk.append_to_row_group(&mut rg)?;
        }
        rg.close()?;
    }

    file_writer.close()?;
    Ok(start.elapsed())
}

fn write_tier0(
    schema: &SchemaRef,
    groups: &[Vec<RecordBatch>],
    props: &WriterProperties,
    path: &Path,
) -> Result<Duration> {
    write_tier0_with(schema, groups, props, path, Tier0Dictionary::Fallback)
}

fn write_tier0_minimal(
    schema: &SchemaRef,
    groups: &[Vec<RecordBatch>],
    props: &WriterProperties,
    path: &Path,
) -> Result<Duration> {
    write_tier0_with(schema, groups, props, path, Tier0Dictionary::ChunkGrain)
}

// ---------------------------------------------------------------------------
// Baseline
// ---------------------------------------------------------------------------

/// A stock [`ArrowWriter`] at the shared properties.
///
/// The row group boundaries are driven explicitly with `flush`, rather than
/// left to `max_row_group_row_count`, so the baseline cuts row groups in
/// exactly the same places as the other three arms even when the sizes are
/// uneven. The properties still carry a row count limit large enough never to
/// pre-empt these boundaries.
fn write_baseline(
    schema: &SchemaRef,
    groups: &[Vec<RecordBatch>],
    props: &WriterProperties,
    path: &Path,
) -> Result<Duration> {
    let start = Instant::now();
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props.clone()))?;
    for group in groups {
        for batch in group {
            writer.write(batch)?;
        }
        // Ends the row group. A no-op if the group contributed no rows.
        writer.flush()?;
    }
    writer.close()?;
    Ok(start.elapsed())
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Writer {
    Baseline,
    OptionA,
    OptionB,
    OptionC,
    Tier0,
    Tier0Minimal,
}

impl Writer {
    fn label(self) -> &'static str {
        match self {
            Writer::Baseline => "baseline (branch default)",
            Writer::OptionA => "option A (page-grain)",
            Writer::OptionB => "option B (K writers)",
            Writer::OptionC => "option C (hybrid)",
            Writer::Tier0 => "tier 0 (probe)",
            Writer::Tier0Minimal => "tier 0 minimal (no #10775)",
        }
    }

    fn run(
        self,
        schema: &SchemaRef,
        groups: &[Vec<RecordBatch>],
        props: &WriterProperties,
        path: &Path,
    ) -> Result<Duration> {
        match self {
            Writer::Baseline => write_baseline(schema, groups, props, path),
            Writer::OptionA => write_option_a(schema, groups, props, path),
            Writer::OptionB => write_option_b(schema, groups, props, path),
            Writer::OptionC => write_option_c(schema, groups, props, path),
            Writer::Tier0 => write_tier0(schema, groups, props, path),
            Writer::Tier0Minimal => write_tier0_minimal(schema, groups, props, path),
        }
    }
}

/// One measured (dataset, compression, writer) cell.
struct Measurement {
    writer: Writer,
    bytes: u64,
    median: Duration,
    /// Per leaf column, the encodings the finished file actually uses.
    encodings: Vec<(String, String)>,
    row_groups: usize,
}

/// Reads the finished file's footer and reports the encodings each leaf column
/// actually ended up with. Doing this from the file rather than from harness
/// bookkeeping keeps the four writers comparable.
fn final_encodings(path: &Path) -> Result<(Vec<(String, String)>, usize)> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let metadata = builder.metadata().clone();
    let descr = metadata.file_metadata().schema_descr_ptr();

    let mut per_column: Vec<Vec<String>> = vec![Vec::new(); descr.num_columns()];
    for rg in metadata.row_groups() {
        for (idx, col) in rg.columns().iter().enumerate() {
            for encoding in col.encodings() {
                let text = encoding.to_string();
                if !per_column[idx].contains(&text) {
                    per_column[idx].push(text);
                }
            }
        }
    }

    let out = descr
        .columns()
        .iter()
        .zip(per_column)
        .map(|(c, mut encodings)| {
            encodings.sort();
            (c.path().string(), encodings.join("+"))
        })
        .collect();
    Ok((out, metadata.num_row_groups()))
}

/// Reads `path` back and requires exact equality against the source batches.
///
/// The comparison streams: it walks the read batches against a cursor over the
/// source batches and compares equal-length slices, rather than concatenating
/// either side. At ClickBench width (about 100 columns, long string values) a
/// concatenating check would hold two extra full copies of the data.
fn verify(path: &Path, source: &[RecordBatch]) -> Result<usize> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(8192)
        .build()?;

    let mut src_idx = 0usize;
    let mut src_off = 0usize;
    let mut rows = 0usize;

    for batch in reader {
        let batch = batch?;
        let mut taken = 0usize;
        while taken < batch.num_rows() {
            // Advance past any exhausted (or empty) source batch.
            while src_idx < source.len() && src_off == source[src_idx].num_rows() {
                src_idx += 1;
                src_off = 0;
            }
            assert!(
                src_idx < source.len(),
                "{} read back more rows than the source has",
                path.display()
            );
            let src = &source[src_idx];
            let n = (src.num_rows() - src_off).min(batch.num_rows() - taken);
            for col in 0..src.num_columns() {
                // Arrow array equality is logical, so comparing slices at
                // different offsets is exact.
                let expected = src.column(col).slice(src_off, n);
                let actual = batch.column(col).slice(taken, n);
                assert_eq!(
                    &expected,
                    &actual,
                    "row values differ for column {col} at row {} in {}",
                    rows + taken,
                    path.display()
                );
            }
            src_off += n;
            taken += n;
        }
        rows += batch.num_rows();
    }

    // Every source row must have been consumed.
    while src_idx < source.len() && src_off == source[src_idx].num_rows() {
        src_idx += 1;
        src_off = 0;
    }
    assert!(
        src_idx == source.len(),
        "{} read back fewer rows than the source has",
        path.display()
    );
    Ok(rows)
}

/// Runs one writer `RUNS` times, checks the output is deterministic, and
/// returns the median time with the final file's measured facts.
fn measure(
    writer: Writer,
    schema: &SchemaRef,
    flat: &[RecordBatch],
    groups: &[Vec<RecordBatch>],
    props: &WriterProperties,
    path: &Path,
) -> Result<Measurement> {
    let mut times = Vec::with_capacity(RUNS);
    let mut bytes = 0u64;
    for run in 0..RUNS {
        let elapsed = writer.run(schema, groups, props, path)?;
        let size = std::fs::metadata(path)?.len();
        if run == 0 {
            bytes = size;
        } else {
            assert_eq!(
                bytes,
                size,
                "{} is not deterministic across runs",
                writer.label()
            );
        }
        times.push(elapsed);
    }
    times.sort();

    verify(path, flat)?;
    let (encodings, row_groups) = final_encodings(path)?;

    Ok(Measurement {
        writer,
        bytes,
        median: times[times.len() / 2],
        encodings,
        row_groups,
    })
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn human(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:.2} {}", UNITS[u])
}

/// Bytes produced by a stock `ArrowWriter` at these same properties on
/// **vanilla upstream main** (merge base `2567a32`), measured separately with
/// the identical data, identical properties and the same preserved row group
/// boundaries as everything else here.
///
/// This is the primary reference for every percentage in the report. The
/// baseline arm measured in this tree is *not* the same thing: this branch
/// carries the #10777 dictionary-fallback re-encode port, which changes what a
/// default writer emits on any column whose dictionary overflows. Comparing an
/// option only against the branch default would silently fold #10777's effect
/// into each option's result.
///
/// Keyed by synthetic dataset name and compression label, or by input file name
/// for the public corpora, where only ZSTD was measured.
fn vanilla_main_bytes(dataset: &str, compression: &str) -> Option<u64> {
    let key = (dataset, compression);
    let synthetic = match key {
        // Byte-identical to the branch default: no dictionary overflows here,
        // so #10777 never fires.
        ("low-cardinality strings", "none") => 1_536_969,
        ("low-cardinality strings", "zstd") => 1_520_065,
        ("high-cardinality int64 timestamps", "none") => 20_052_805,
        ("high-cardinality int64 timestamps", "zstd") => 8_154_384,
        ("f64 measurements", "none") => 20_056_464,
        ("f64 measurements", "zstd") => 18_937_240,
        // Dictionary-overflow columns: the branch default differs.
        ("shifting strings", "none") => 29_367_978,
        ("shifting strings", "zstd") => 10_028_648,
        ("records-like mixed schema", "none") => 79_708_639,
        ("records-like mixed schema", "zstd") => 39_049_927,
        _ => 0,
    };
    if synthetic != 0 {
        return Some(synthetic);
    }
    // Public corpora, ZSTD only.
    match dataset {
        "lineitem.parquet" => Some(183_251_967),
        "orders.parquet" => Some(45_834_964),
        "hits_0.parquet" => Some(85_354_208),
        "hits_1.parquet" => Some(121_026_589),
        "hits_2.parquet" => Some(166_652_721),
        _ => None,
    }
}

/// One row of the printed and written table.
struct Row {
    dataset: String,
    compression: String,
    writer: String,
    bytes: u64,
    seconds: f64,
    /// Against vanilla upstream main, the primary reference.
    vs_vanilla_bytes: Option<f64>,
    /// Against this branch's own default writer, which includes #10777.
    vs_branch_bytes: f64,
    vs_baseline_time: f64,
    /// The vanilla reference itself, so each row stands on its own.
    vanilla: Option<u64>,
}

fn render_table(rows: &[Row]) -> String {
    let headers = [
        "dataset",
        "compression",
        "writer",
        "bytes",
        "size",
        "median s",
        "vanilla base",
        "vs vanilla",
        "vs branch base",
        "time vs branch",
    ];
    let mut cells: Vec<Vec<String>> = vec![headers.iter().map(|h| h.to_string()).collect()];
    for row in rows {
        cells.push(vec![
            row.dataset.clone(),
            row.compression.clone(),
            row.writer.clone(),
            row.bytes.to_string(),
            human(row.bytes),
            if row.seconds.is_nan() {
                "-".to_string()
            } else {
                format!("{:.3}", row.seconds)
            },
            row.vanilla.map(|v| v.to_string()).unwrap_or("n/a".into()),
            row.vs_vanilla_bytes
                .map(|v| format!("{:+.2}%", v * 100.0))
                .unwrap_or("n/a".into()),
            if row.vs_branch_bytes.is_nan() {
                "n/a".to_string()
            } else {
                format!("{:+.2}%", row.vs_branch_bytes * 100.0)
            },
            if row.vs_baseline_time.is_nan() {
                "-".to_string()
            } else {
                format!("{:.2}x", row.vs_baseline_time)
            },
        ]);
    }

    let widths: Vec<usize> = (0..headers.len())
        .map(|c| {
            cells
                .iter()
                .map(|r| r[c].chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();

    let mut out = String::new();
    for (i, row) in cells.iter().enumerate() {
        let line: Vec<String> = row
            .iter()
            .zip(&widths)
            .map(|(cell, w)| format!("{cell:<w$}"))
            .collect();
        out.push_str(line.join("  ").trim_end());
        out.push('\n');
        if i == 0 {
            let rule: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
            out.push_str(&rule.join("  "));
            out.push('\n');
        }
    }
    out
}

/// Prints the per-column encoding detail for one dataset and compression.
fn print_encodings(measurements: &[Measurement], passthrough: &[String]) {
    for m in measurements {
        let detail: Vec<String> = m
            .encodings
            .iter()
            .map(|(name, encodings)| format!("{name}={encodings}"))
            .collect();
        println!(
            "    {:<22} {} row groups, {}",
            m.writer.label(),
            m.row_groups,
            detail.join("  ")
        );
    }
    if !passthrough.is_empty() {
        println!(
            "    passed through (no race in either harness): {}",
            passthrough.join(", ")
        );
    }
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// Runs the four writers over one dataset at one compression.
fn run_case(
    name: &str,
    schema: &SchemaRef,
    batches: &[RecordBatch],
    row_group_sizes: &[usize],
    compression: (&str, Compression),
    dir: &Path,
    rows: &mut Vec<Row>,
) -> Result<Vec<String>> {
    let (compression_label, compression_codec) = compression;
    // Large enough that the baseline's own limit never fires before the
    // explicit flush that ends each group.
    let cap = row_group_sizes
        .iter()
        .copied()
        .max()
        .unwrap_or(ROW_GROUP_ROWS);
    let props = shared_properties(compression_codec)
        .set_max_row_group_row_count(Some(cap))
        .build();
    let groups = split_into_row_groups(batches, row_group_sizes);

    let mut measurements = Vec::new();
    for writer in [
        Writer::Baseline,
        Writer::OptionA,
        Writer::OptionB,
        Writer::OptionC,
        Writer::Tier0,
        Writer::Tier0Minimal,
    ] {
        let path = dir.join("bakeoff.parquet");
        measurements.push(measure(writer, schema, batches, &groups, &props, &path)?);
        let _ = std::fs::remove_file(&path);
    }

    let baseline = &measurements[0];
    let vanilla = vanilla_main_bytes(name, compression_label);

    // The vanilla reference leads each group, so the table reads against it.
    rows.push(Row {
        dataset: name.to_string(),
        compression: compression_label.to_string(),
        writer: "vanilla main (reference)".to_string(),
        bytes: vanilla.unwrap_or(0),
        seconds: f64::NAN,
        vs_vanilla_bytes: vanilla.map(|_| 0.0),
        vs_branch_bytes: vanilla
            .map(|v| v as f64 / baseline.bytes as f64 - 1.0)
            .unwrap_or(f64::NAN),
        vs_baseline_time: f64::NAN,
        vanilla,
    });

    for m in &measurements {
        rows.push(Row {
            dataset: name.to_string(),
            compression: compression_label.to_string(),
            writer: m.writer.label().to_string(),
            bytes: m.bytes,
            seconds: m.median.as_secs_f64(),
            vs_vanilla_bytes: vanilla.map(|v| m.bytes as f64 / v as f64 - 1.0),
            vs_branch_bytes: m.bytes as f64 / baseline.bytes as f64 - 1.0,
            vs_baseline_time: m.median.as_secs_f64() / baseline.median.as_secs_f64().max(1e-9),
            vanilla,
        });
    }

    let passthrough = passthrough_leaves(schema, &props)?;
    println!("  {name} / {compression_label}");
    print_encodings(&measurements, &passthrough);
    println!(
        "    verified {} rows read back exactly from all six files",
        batches.iter().map(|b| b.num_rows()).sum::<usize>()
    );
    Ok(encoding_summary(&measurements, &passthrough))
}

/// A compact account of where the arms' final encodings differ.
///
/// At ClickBench width, printing every column's encodings for every arm is
/// several thousand characters of noise. What is actually informative is how
/// many columns each arm moved off the baseline's choice, and which columns the
/// arms disagree with each other about.
fn encoding_summary(measurements: &[Measurement], passthrough: &[String]) -> Vec<String> {
    /// Column names where two arms' final encodings differ.
    fn differing<'a>(left: &'a Measurement, right: &Measurement) -> Vec<&'a str> {
        left.encodings
            .iter()
            .zip(&right.encodings)
            .filter(|((_, l), (_, r))| l != r)
            .map(|((name, _), _)| name.as_str())
            .collect()
    }

    /// Renders at most `cap` names, then says how many were left out.
    fn listed(names: &[&str], cap: usize) -> String {
        if names.is_empty() {
            return "none".to_string();
        }
        let shown: Vec<&str> = names.iter().take(cap).copied().collect();
        if names.len() > cap {
            format!("{} and {} more", shown.join(", "), names.len() - cap)
        } else {
            shown.join(", ")
        }
    }

    let total = measurements[0].encodings.len();
    let mut notes = Vec::new();

    let baseline = &measurements[0];
    let counts: Vec<String> = measurements[1..]
        .iter()
        .map(|m| format!("{} on {}", m.writer.label(), differing(m, baseline).len()))
        .collect();
    notes.push(format!(
        "Of {total} leaf columns, the arms change the baseline's encoding on: {}.",
        counts.join("; ")
    ));

    if let (Some(a), Some(b), Some(c)) = (
        measurements.iter().find(|m| m.writer == Writer::OptionA),
        measurements.iter().find(|m| m.writer == Writer::OptionB),
        measurements.iter().find(|m| m.writer == Writer::OptionC),
    ) {
        let ab = differing(a, b);
        notes.push(format!(
            "A and B disagree on {} columns: {}.",
            ab.len(),
            listed(&ab, 12)
        ));
        let ca = differing(c, a);
        notes.push(format!(
            "C differs from A on {} columns: {}.",
            ca.len(),
            listed(&ca, 12)
        ));
        let cb = differing(c, b);
        notes.push(format!("C differs from B on {} columns.", cb.len()));
    }

    notes.push(if passthrough.is_empty() {
        "No column shape was passed through: every leaf was raceable.".to_string()
    } else {
        format!(
            "Passed through unraced in both harnesses ({}): {}.",
            passthrough.len(),
            passthrough.join(", ")
        )
    });

    notes
}

/// The synthetic suite. Writes `parquet/BAKEOFF.md` alongside its stdout table.
fn run_synthetic(dir: &Path) -> Result<()> {
    // Each dataset carries its own name, so the list is just the builders.
    let builders: [DatasetBuilder; 5] = [
        low_cardinality_strings,
        timestamps,
        floats,
        shifting_strings,
        records,
    ];

    println!(
        "bakeoff: {ROWS} rows per dataset, {ROW_GROUP_ROWS} rows per row group \
         ({} row groups), {BATCH_ROWS} rows per batch, median of {RUNS} runs\n",
        ROWS / ROW_GROUP_ROWS
    );

    let mut rows: Vec<Row> = Vec::new();
    for builder in builders {
        // One dataset at a time: 2M rows across five datasets does not need to
        // be resident at once.
        let dataset = builder();
        for (label, compression) in compressions() {
            let _ = run_case(
                dataset.name,
                &dataset.schema,
                &dataset.batches,
                &[ROW_GROUP_ROWS],
                (label, compression),
                dir,
                &mut rows,
            )?;
        }
        println!();
    }

    let table = render_table(&rows);
    println!("{table}");
    write_report(&table)?;
    println!("wrote parquet/BAKEOFF.md");
    Ok(())
}

/// Rewrites externally supplied parquet files through the four writers.
///
/// By default the results go to stdout only: nothing is written into
/// `BAKEOFF.md`, and no input file is copied into the tree. That is the mode
/// for private or unpublished inputs.
///
/// With `public = true` the table is additionally recorded in `BAKEOFF.md`
/// under a "Public datasets" section, which is appropriate only for inputs
/// anyone can reproduce, such as a published benchmark corpus or data from a
/// generator anyone can run. `label` names the corpus in that section.
fn run_files(paths: &[String], dir: &Path, public: Option<&str>) -> Result<()> {
    let compression = compressions().pop().expect("at least one compression");

    println!(
        "bakeoff over supplied files: compression {}, median of {RUNS} runs\n\
         (results are {})\n",
        compression.0,
        match public {
            Some(label) => format!("recorded in BAKEOFF.md under \"{label}\""),
            None => "printed here only and are not written to any file".to_string(),
        }
    );

    let mut public_rows: Vec<Row> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    for path in paths {
        let input = Path::new(path);
        let file = File::open(input)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let metadata = builder.metadata().clone();
        let input_bytes = std::fs::metadata(input)?.len();

        // Preserve the input's own row group boundaries, which are often
        // uneven, rather than collapsing them to a single size.
        let row_group_sizes: Vec<usize> = metadata
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows() as usize)
            .filter(|n| *n > 0)
            .collect();
        let row_group_sizes = if row_group_sizes.is_empty() {
            vec![ROW_GROUP_ROWS]
        } else {
            row_group_sizes
        };
        let sizing = {
            let min = row_group_sizes.iter().min().copied().unwrap_or(0);
            let max = row_group_sizes.iter().max().copied().unwrap_or(0);
            if min == max {
                format!("{min} rows per row group")
            } else {
                format!("{min} to {max} rows per row group")
            }
        };

        let schema = builder.schema().clone();
        let reader = builder.with_batch_size(8192).build()?;
        let batches: Vec<RecordBatch> = reader.collect::<std::result::Result<Vec<_>, _>>()?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        println!(
            "== {} ==\n  input {} ({}), {} rows, {} row groups, reproducing its boundaries ({sizing})",
            input.display(),
            human(input_bytes),
            input_bytes,
            total_rows,
            metadata.num_row_groups(),
        );

        let mut rows: Vec<Row> = Vec::new();
        // Only the file name identifies the dataset in the report: an absolute
        // path would leak where the corpus happened to be staged.
        let name = input
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| input.display().to_string());
        let summary = run_case(
            &name,
            &schema,
            &batches,
            &row_group_sizes,
            compression,
            dir,
            &mut rows,
        )?;
        println!("{}", render_table(&rows));
        if public.is_some() {
            notes.push(format!(
                "* **`{name}`**: {total_rows} rows in {} row groups ({sizing}), \
                 source file {}; the input's own row group boundaries are reproduced \
                 by all four arms.",
                metadata.num_row_groups(),
                human(input_bytes)
            ));
            notes.extend(summary.into_iter().map(|line| format!("  * {line}")));
            public_rows.extend(rows);
        }
    }

    if let Some(label) = public {
        record_public_results(label, &public_rows, &notes)?;
        println!("recorded the {label} results in BAKEOFF.md");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------

/// Heading that opens the public-dataset section of the report.
///
/// The synthetic run rewrites `BAKEOFF.md` wholesale, so it carries any
/// existing public-dataset section through unchanged rather than dropping the
/// results of a separate corpus run.
const PUBLIC_MARKER: &str = "## Public datasets";

fn report_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("BAKEOFF.md")
}

/// The existing public-dataset section, if the report already has one.
fn existing_public_section() -> String {
    let text = std::fs::read_to_string(report_path()).unwrap_or_default();
    match text.find(PUBLIC_MARKER) {
        Some(at) => text[at..].to_string(),
        None => String::new(),
    }
}

/// Records one corpus's results in the report.
///
/// Keeps everything the synthetic run wrote above the public section, and keeps
/// every *other* corpus already recorded in it: only the subsection with this
/// label is replaced, so corpora can be measured in separate runs.
fn record_public_results(label: &str, rows: &[Row], notes: &[String]) -> Result<()> {
    let path = report_path();
    let existing = std::fs::read_to_string(&path).unwrap_or_default();

    let (head, section) = match existing.find(PUBLIC_MARKER) {
        Some(at) => (existing[..at].to_string(), existing[at..].to_string()),
        None => {
            let mut head = existing;
            if !head.is_empty() && !head.ends_with("\n\n") {
                head.push('\n');
            }
            (head, String::new())
        }
    };

    // Split the existing public section into its preamble and its per-corpus
    // subsections, keyed by heading.
    let heading = format!("### {label}");
    let mut preamble = String::new();
    let mut others: Vec<String> = Vec::new();
    if !section.is_empty() {
        let mut parts = section.split("\n### ");
        preamble = parts.next().unwrap_or_default().to_string();
        for part in parts {
            let sub = format!("### {part}");
            // Drop any previous run of this same corpus; keep the rest.
            if !sub.starts_with(&heading) {
                others.push(sub);
            }
        }
    }
    if preamble.is_empty() {
        preamble = format!(
            "{PUBLIC_MARKER}\n\nResults over corpora anyone can obtain, measured the same way as the\n\
             synthetic suite: ZSTD, the input's own row group sizing preserved, exact\n\
             read-back verification, median of {RUNS} release runs. Only the file name of\n\
             each input is recorded.\n"
        );
    }

    let mut fresh = String::new();
    let _ = writeln!(fresh, "{heading}\n");
    for note in notes {
        let _ = writeln!(fresh, "{note}");
    }
    fresh.push_str("\n```\n");
    fresh.push_str(&render_table(rows));
    fresh.push_str("```\n");

    let mut out = head;
    out.push_str(preamble.trim_end());
    out.push_str("\n\n");
    for sub in others {
        out.push_str(sub.trim_end());
        out.push_str("\n\n");
    }
    out.push_str(fresh.trim_end());
    out.push('\n');

    std::fs::write(&path, out)?;
    Ok(())
}

fn write_report(table: &str) -> Result<()> {
    // Anchored to the crate directory so the report lands in the right place
    // whatever the working directory of the run.
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let harness = count_lines(&crate_dir.join("examples/adaptive_writer/harness.rs"));
    let demo = count_lines(&crate_dir.join("examples/adaptive_writer/main.rs"));
    let bakeoff = count_lines(&crate_dir.join("examples/bakeoff.rs"));
    // Counted from the harness's own section markers, so the split cannot
    // drift away from the file it describes.
    let (policy_lines, plumbing_lines) =
        harness_split(&crate_dir.join("examples/adaptive_writer/harness.rs"));

    let body = format!(
        r#"# Parquet encoding policy bakeoff

Generated by `cargo run --release --features "arrow snap zstd" --example bakeoff`.
Every number below comes from that run; re-running the example regenerates this
file.

## Method

All six writers see identical inputs and identical writer properties. The only
difference between them is how each decides a column's encoding.

* Datasets are deterministic ({ROWS} rows each, splitmix64 seeded per dataset),
  cut into {BATCH_ROWS} row record batches.
* Row group boundaries are identical for all six writers: {ROW_GROUP_ROWS}
  rows, {row_groups} row groups per file. The baseline reaches them through
  `max_row_group_row_count`; every other arm cuts the batches itself at the
  same offsets.
* Shared properties: `data_page_row_count_limit = {DATA_PAGE_ROW_LIMIT}`, and
  the compression named in each row.
* Every file is read back and compared column by column against the source
  arrays before its numbers are reported.
* Time is the median of {RUNS} runs in a release build. Output size is asserted
  identical across those runs, so each writer is deterministic.

The six writers:

1. **Baseline** — a stock `ArrowWriter` at the shared properties.
2. **Option A (page-grain)** — `examples/adaptive_writer/harness.rs` with
   `always_page_grain` set, so every decidable leaf stays on the page-grain
   path for the whole file. It encodes each page several ways through
   `ColumnChunkBuilder::encode_page_alternatives`, charges each page the
   dictionary bytes it created (`EncodedPage::dictionary_growth`), settles
   after two agreeing pages, watches the live dictionary and abandons it when
   entries exceed a quarter of the values sent through it, and carries the
   settled choice into later row groups, looking again every
   {A_REOPEN_EVERY}.
3. **Option B (K writers)** — builds K complete sets of column writers per row
   group through `create_column_writers_with_properties`, one per candidate,
   feeds every leaf to all of them, keeps the smallest finished chunk and drops
   the rest. The dictionary candidate uses
   `DictionaryFallback::WhenProfitable {{ worth_ratio: {DICT_WORTH_RATIO} }}`.
   A leaf settles when a row group's best and worst differ by at least
   {settle_pct}%, and re-races every {B_REOPEN_EVERY} row groups.
4. **Option C (hybrid)** — the same harness as it ships, with the routing rule
   on: each leaf, each row group, takes whichever of the two paths it needs.
   See below. A and C are the same code driven over the same batches, so the
   difference between them is the routing rule and nothing else.
5. **Tier 0 (probe)** — chunk-grain probe-then-commit, built on nothing beyond
   the two factory accessors and the #10775 dictionary-fallback port. A
   deciding leaf's first {TIER0_PROBE_ROWS} rows of the row group are encoded
   K ways through K throwaway single-leaf writers created with
   `create_selected_column_writers`; their `ColumnCloseResult` compressed sizes
   pick a winner by Option B's rule; the probe chunks are dropped and the leaf
   writes its whole row group through one ordinary writer at that winner. A
   settled leaf is never probed. It shares Option B's candidate set, near-tie
   window, {settle_pct}% settle gap and {B_REOPEN_EVERY} row group re-open
   cadence, so tier 0 and Option B differ only in whether the race sees the
   whole row group or just its first span. It never touches `page_grain`.
6. **Tier 0 minimal (no #10775)** — the same probe arm with the
   `DictionaryFallback` port assumed rejected upstream. Its dictionary
   candidate gets stock behaviour only: dictionary on plus the byte-size
   `dictionary_page_size_limit` check every released writer already has. The
   harness then owns the dictionary heuristic itself, at chunk grain and from
   already-public metadata: after each row group it reads the closed chunk's
   `ColumnChunkMetaData` (dictionary page offset and per-page encoding stats),
   and if the chunk paid for a dictionary that some of its data pages did not
   index into, it withholds the dictionary candidate from that column for the
   next {B_REOPEN_EVERY} row groups and makes it decide again. This arm needs
   *no* library change at all beyond the two factory accessors.

### Option C's composition rule

Per leaf, per row group:

* A leaf that is **actively deciding** goes to the **page-grain builder** and
  races candidates a page at a time with the dictionary-growth charge, exactly
  as Option A does. "Actively deciding" means it has never settled, its race
  re-opens this row group, or it adapts per page and so is never done deciding.
* Every other leaf — every **settled** leaf and every leaf with **nothing to
  race** — goes to an ordinary `ArrowColumnWriter` from
  `create_selected_column_writers`, configured with the candidate that leaf
  settled on. That is Option B's path, and it gets the library's normal write
  path with no page-grain involvement at all. No writer is created for the
  leaves on the page-grain path.
* Wherever a dictionary runs on that standard path it runs under
  `DictionaryFallback::WhenProfitable`, so a settled dictionary leaf can still
  leave its dictionary without the page grain watching it.

Both paths produce an `ArrowColumnChunk`, and the two kinds are appended to one
`SerializedRowGroupWriter` in leaf order. The intent is A's decisions at less
than A's per-leaf library exposure and much less than B's CPU, since only the
leaves actually deciding pay anything beyond the ordinary write path.

### Seams the hybrid exposed

Three. Two are now fixed in the library, and the third is deliberate.

1. **Subset writer creation (fixed).** `create_column_writers_with_properties`
   was all-or-nothing, so the hybrid allocated a writer per leaf and dropped
   the ones it wrote itself: with a spilling page store that is a temp file per
   unused leaf per row group. `create_selected_column_writers` takes a
   predicate over leaf index and returns `Vec<Option<ArrowColumnWriter>>`,
   creating nothing for the leaves the caller declined.
2. **Sharing the file writer's page store (fixed).**
   `ArrowRowGroupWriterFactory::with_page_store_factory` had no getter, so the
   page-grain leaves buffered in memory while the standard leaves spilled, and
   the memory bound did not cover the whole row group. `page_store_factory()`
   returns it, and the harness hands it to
   `ColumnChunkBuilder::new_with_page_store`.
3. **The two paths abandon a dictionary by different mechanisms (kept).** The
   standard path uses `DictionaryFallback`; the page grain has no automatic
   fallback, because deciding that per page with the numbers in hand is the
   whole point of it. So a leaf changes mechanism when it changes path, and
   keeping the two consistent is the harness's job. This is the direct cause of
   the byte gap between C and A on the two uncompressed mixed datasets below.

## Results

Every percentage in this report is measured against **vanilla upstream main**
(merge base `2567a32`), not against the baseline arm measured in this tree.

* `vanilla base` / `vs vanilla` — a stock `ArrowWriter` at these same
  properties on unmodified upstream main. This is the reference a maintainer
  cares about, because it is what the options would actually be replacing.
* `baseline (branch default)` / `vs branch base` — a stock `ArrowWriter` in
  *this* tree, which includes the #10777 dictionary-fallback re-encode port. It
  is kept as a secondary reference so the options can also be read against the
  writer they were developed alongside.

The two baselines differ only on columns whose dictionary overflows, and the gap
between them is #10777's isolated effect on default output. That gap is
attributed in the next section; it is not any option's doing, and folding it
into an option's result would overstate or understate that option.

```
{table}```

## What separates the two baselines

The branch default and vanilla main emit identical bytes on every dataset with
no dictionary overflow: low-cardinality strings, timestamps and f64 measurements
match exactly at both compressions. Every cell where they differ is a cell with
at least one dictionary-overflow column, and #10777 moves the bytes in both
directions depending on *when* the overflow happens.

**Overflow after a page has already been sealed makes the branch default
larger.** This is the common case and the larger effect: shifting strings is
+15.02% uncompressed and +11.50% under ZSTD, records-like is +3.70% and +1.51%,
TPC-H `lineitem` is +3.16%. When the dictionary overflows, #10777 re-encodes the
buffered in-progress page through the fallback encoder rather than emitting it
as one more dictionary-encoded page. Vanilla main kept those already-buffered
values as `RLE_DICTIONARY` indices; the branch rewrites them as `PLAIN`, which
is bigger. Every column that ends up delta encoded is by definition a column
whose dictionary overflowed, so this is exactly the population where the two
baselines part company.

**Overflow before the first page is sealed makes the branch default smaller.**
TPC-H `orders` is the one public file where the branch default is *smaller* than
vanilla, by 0.81%. Its `o_comment` column overflows before any dictionary page
has been referenced, in 10 of its 16 row groups. Under #10777 no dictionary page
is written at all in those row groups and the whole chunk uses the fallback
encoding; vanilla main still writes a nearly 1 MiB dictionary page that almost
nothing references. Dropping an unreferenced dictionary page is a clear win, and
it is why the sign flips here.

The three ClickBench files sit between the two effects at a consistent +0.55% to
+0.57%: at 105 columns both cases occur across the schema and largely cancel.

**Implication for #10777 upstream.** The two directions are separable. The win
comes from not writing a dictionary page that ends up unreferenced; the loss
comes from re-encoding values that were already validly written as dictionary
indices. A narrower fix that re-encodes the buffered page *only* when the
dictionary would otherwise be left unreferenced would keep the `orders` win and
avoid the `lineitem` and shifting-strings regressions. That is a suggestion from
these measurements, not a tested change: nothing in this repository implements
or evaluates the narrowed variant.

## Complexity

Library cost, from each option's design document.

| | Option A/C (page-grain) | Option B (merged bespoke APIs) |
| --- | ---: | ---: |
| Library production lines added | 1 687 | 374 |
| Library test lines added | 529 | 623 |
| Library lines removed or rewritten | 92 | 109 |
| Files touched in the library | 5 | 6 |

The page-grain figure is measured from the merge of the Option B ports, so the
two columns do not overlap. 1 011 of its production lines are one new
self-contained module (`page_grain`), and the largest change to an existing
file is a mechanical split of `write_data_page` into `assemble_data_page` and
`commit_data_page`. Option B adds no new module: its 374 lines are spread over
the properties type, the column writer's dictionary fallback, and the encoders.

Public items: 22 in `page_grain` plus 2 on `ArrowRowGroupWriterFactory` for
A/C; Option B's surface is one method on the factory and one properties enum.
`PAGE_API_DESIGN.md` lists both and the reasoning behind each cut.

Harness cost, counted from the files in this repository:

| Harness | Lines |
| --- | ---: |
| `examples/adaptive_writer/harness.rs` (policy {policy}, plumbing {plumbing}) | {harness} |
| `examples/adaptive_writer/main.rs` (dataset, comparison, verification) | {demo} |
| `examples/bakeoff.rs` (four writers, five datasets and reporting) | {bakeoff} |

`harness.rs` is the harness measured as options A and C above: the bakeoff
includes that same file rather than a copy, so these numbers describe the code
that ships. Its policy half is the part a different writer would rewrite: which
encodings to try, what a page costs, when to settle, when to look again, when
to leave a dictionary, and which path a leaf takes. Its plumbing half opens row
groups, routes leaves and appends the chunks both paths produce.

{interpretation}

{tier0}
"#,
        row_groups = ROWS / ROW_GROUP_ROWS,
        policy = policy_lines,
        plumbing = plumbing_lines,
        settle_pct = (SETTLE_GAP * 100.0) as u32,
        interpretation = INTERPRETATION,
        tier0 = TIER0,
    );

    let public = existing_public_section();
    let body = if public.is_empty() {
        body
    } else {
        format!("{body}\n{public}")
    };
    std::fs::write(crate_dir.join("BAKEOFF.md"), body)?;
    Ok(())
}

/// Lines in the harness's policy half and plumbing half, delimited by the
/// `// Policy` and `// Plumbing` banners in that file.
fn harness_split(path: &Path) -> (usize, usize) {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let find = |needle: &str| lines.iter().position(|l| l.trim_start() == needle);
    match (find("// Policy"), find("// Plumbing")) {
        // Each banner opens two lines above its `// ===` rule.
        (Some(policy), Some(plumbing)) => (
            plumbing.saturating_sub(policy),
            lines.len().saturating_sub(plumbing),
        ),
        _ => (0, 0),
    }
}

fn count_lines(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .map(|s| s.lines().count())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------

/// The Tier 0 section: what the probe arm is, what it measured, and what that
/// says about the minimum upstream ask. Numbers quoted here are read off the
/// tables above, which the same run regenerates.
const TIER0: &str = r#"## Tier 0

Tier 0 asks the narrowest version of the question this whole branch exists to
answer: **how much of the result survives if upstream merges almost none of
it?** Three variants are measured, differing only in which library changes they
are allowed to assume.

| variant | library changes assumed | measured where |
| --- | --- | --- |
| `tier 0 (probe)` | the two factory accessors, #10775, #10777 | this tree |
| `tier 0 minimal (no #10775)` | the two accessors and #10777; the harness owns the dictionary rule | this tree |
| `tier 0 floor (accessors only)` | the two accessors and nothing else | a worktree at merge base `2567a32` |

All three are the same writer: per leaf, per row group, a leaf that has settled
writes its whole row group through one ordinary `ArrowColumnWriter`, and a leaf
that is still deciding first has its first 20000 rows encoded K ways through K
throwaway single-leaf writers from `create_selected_column_writers`. The winner
is chosen from those probes' `ColumnCloseResult` compressed sizes by Option B's
near-tie and decode-rank rule, the probe chunks are dropped, and the leaf then
writes the whole row group, probe rows included, at the winning candidate. A
deciding leaf therefore encodes its probe span K+1 times, and that doubled span
is the tier's only overhead knob. None of the three touches `page_grain`.

The floor variant is a separate checkout of the merge base carrying exactly one
cherry-picked change, `+67 -13` in `parquet/src/arrow/arrow_writer/mod.rs`:
`page_store_factory()` and `create_selected_column_writers()`. Its stock
`ArrowWriter` arm reproduced the published `vanilla main` byte count exactly on
all five public files and all ten synthetic cells, which is what makes its
numbers comparable with the rest of this report.

### Bytes against vanilla main

Public files, ZSTD, the production-relevant cells:

| file | option C | tier 0 | tier 0 minimal | tier 0 floor |
| --- | ---: | ---: | ---: | ---: |
| `orders` | -24.69% | -26.62% | -26.62% | -26.62% |
| `lineitem` | -23.14% | -23.41% | -23.41% | -23.41% |
| `hits_0` | -6.31% | -7.93% | -7.69% | -7.71% |
| `hits_1` | -4.51% | -5.18% | -4.97% | -5.00% |
| `hits_2` | -6.69% | -6.59% | -6.49% | -6.51% |

Synthetic, both compressions:

| dataset | comp | option C | tier 0 | tier 0 minimal | tier 0 floor |
| --- | --- | ---: | ---: | ---: | ---: |
| low-cardinality strings | none | +0.07% | +0.00% | +0.00% | +0.00% |
| low-cardinality strings | zstd | +0.08% | +0.00% | +0.00% | +0.00% |
| timestamps | none | -84.77% | -84.78% | -84.78% | -84.78% |
| timestamps | zstd | -62.55% | -62.56% | -62.56% | -62.56% |
| f64 measurements | none | -20.17% | -20.18% | -20.18% | -20.18% |
| f64 measurements | zstd | -20.50% | -20.52% | -20.52% | -20.52% |
| shifting strings | none | -33.73% | -18.10% | -18.10% | -37.63% |
| shifting strings | zstd | -7.00% | -7.77% | -7.77% | -8.80% |
| records-like | none | -47.54% | -37.77% | -37.77% | -49.99% |
| records-like | zstd | -25.67% | -25.83% | -25.83% | -25.90% |

### Interpretation

**On the production-relevant cells the probe arm is not a compromise, it is the
better answer.** On the five public files under ZSTD, tier 0 beats Option C on
four and trails it by 0.10 points on the fifth, and it does so while running at
0.79x to 0.85x the branch default's wall clock against C's 0.72x to 1.03x and
Option B's 1.78x to 2.66x. It reaches Option B's bytes to within 0.4 points
everywhere and matches them exactly on `lineitem` and the synthetic cells,
because it makes B's decision, a whole-chunk choice, from a sample rather than
from the whole chunk. What the sample costs is visible only on `hits_0` and
`hits_1`, where the probe is 20000 rows of a 450000-row chunk and tier 0 gives
up 0.4 and 0.7 points against B, and what it saves is the other two thirds of
B's CPU.

**Where it visibly loses is uncompressed data that changes shape mid-file.** On
shifting strings and the records-like schema uncompressed, tier 0 lands on
Option B's -18.10% and -37.77% against C's -33.73% and -47.54%, a gap of 10 to
15 points. This is the one thing a chunk-grain writer structurally cannot do:
the data changes character in the middle of a row group, and a decision taken
once per chunk from its first 20000 rows cannot react until the next row group.
Option A's page grain re-decides every page and abandons a live dictionary
inside the chunk, which is exactly the capability being priced. The gap closes
under ZSTD, where the general compressor removes most of the redundancy that
finer-grained switching targets, and tier 0 then wins these cells outright.
Files with very few row groups are the same weakness in another form: `hits_0`
has 2 row groups, so a leaf gets one settle and one commit, and it is the file
where tier 0 gives up most against B.

**The two contested heuristic PRs turn out not to be load-bearing for this
writer, and #10777 is actively harmful to it.** #10775's `DictionaryFallback`
is worth nothing at all on the synthetic suite (tier 0 minimal is byte-identical
to tier 0 in all ten cells) and 0.10 to 0.24 points on the three ClickBench
files: a harness that reads the closed chunk's `ColumnChunkMetaData` and
withholds the dictionary candidate from a column that paid for a dictionary its
data pages did not fully use recovers nearly all of it, from already-public API.
The floor variant, which assumes neither PR, is within 0.22 points of full tier
0 on every public file, and on the two uncompressed shape-shifting cells it is
**better than every other arm in this report**, at -37.63% and -49.99% against
Option A's -37.34% and -50.16%. The cause is #10777: when a dictionary
overflows mid-chunk it re-encodes the already-buffered dictionary-indexed values
as `PLAIN`, and for exactly this data those indices were the right encoding for
the part of the chunk they covered. This is the reviewer objection to #10777
(that a dictionary can remain effective after overflow, so unconditional
re-encode is a trade rather than a fix) reproduced as a measurement: on a writer
that chooses candidates by measuring them, #10777 costs 19.5 points uncompressed
on shifting strings and 12.2 on records-like.

**What this does to the upstream ask.** Options A and C need about 1690 lines of
new library production code, a new 22-item public `page_grain` module, and a
split of `write_data_page` into assemble and commit halves inside
`GenericColumnWriter`, which is a seam on the hot path of every parquet writer
in the ecosystem. The floor variant needs `+67 -13` lines in one file, adds two
methods to one existing type, changes no write path, and has no hot-path seam at
all: `create_column_writers` becomes the new call with every column selected, so
there is one implementation rather than two. For that, on the cells a
maintainer is most likely to care about, it delivers -23.41% to -26.62% on
TPC-H and -5.00% to -7.71% on ClickBench, against Option C's -23.14% to -24.69%
and -4.51% to -6.69%. The page grain buys one thing the accessors cannot: the
10-to-15-point advantage on uncompressed data whose character changes inside a
row group. Whether that case is worth 1690 lines and a `GenericColumnWriter`
seam is the decision these numbers are for, and nothing else in this report
turns on it.

The floor variant is reproduced with:

```text
git worktree add ../arrow-rs-floor 2567a32
git -C ../arrow-rs-floor cherry-pick -n <the accessors commit>   # keep only the two methods
cargo run --release --features "arrow snap zstd" --example bakeoff_floor -- <files>
```
"#;

/// The written interpretation, kept beside the harness that produced the
/// numbers it describes.
const INTERPRETATION: &str = r"## Interpretation

Byte percentages below are measured against **vanilla upstream main**, are
deterministic, and reproduce exactly on a re-run. Times are medians from one
machine and move a little between runs, and are ratios against this branch's
default writer since that is the one actually executed here; read them as
ratios, not as absolutes.

### Where the baseline is already right

**Low-cardinality strings.** Nothing overflows a dictionary here, so both
baselines are byte-identical. Option B reproduces them byte for byte, C is 0.07%
larger and A 0.29% larger, and all four land on the same encodings. B
ties exactly because its winning candidate is the dictionary, and with a 48
entry dictionary neither the 64 KiB candidate limit nor the 1 MiB default is
approached, so the candidate properties and the baseline properties are
operationally identical for this column. The baseline is also the fastest.
Racing costs time and buys nothing on data whose right answer is the default.

### Where every option wins, by the same amount

**High-cardinality int64 timestamps.** Both baselines are identical here too.
A, B and C all land on `DELTA_BINARY_PACKED` and cut the file by 84.8%
uncompressed and 62.5% with ZSTD, agreeing with each other to within 0.1%. The
baseline footer shows
`PLAIN+RLE+RLE_DICTIONARY`: it builds a dictionary over effectively unique
values and then spills to plain. All three options are also *faster* than the
baseline here, for the same reason: none of them pays to build and then abandon
a two million entry dictionary. This is the largest effect in the table, and no
particular API is required to get it, only the willingness to measure.

**f64 measurements.** Both baselines are identical here as well. Once Option A
races floats (the dictionary against `PLAIN`) rather than only adapting per
page, A and C match B: -20.17% uncompressed and -20.50% with ZSTD, within 0.02%
of B's bytes, all three landing
on `PLAIN+RLE`. Before that change A trailed at -16.3% / -16.5%, and the cause
was visible in the footer as a residual `RLE_DICTIONARY`: the adapt-per-page
rule's cold-start branch chose the dictionary for the first page of every chunk,
because a fresh chunk has a dictionary and no previous page to learn from. One
dictionary-encoded page per chunk is enough to force a dictionary page into that
chunk. Honouring the settled choice at a chunk boundary removed it. The residual
was a harness policy defect, not a property of the page-grain API.

### Where A wins

**Uncompressed data whose character changes mid-file.** On shifting strings A is
37.34% below vanilla main against B's 18.10%, and on the records-like schema
50.16% against B's 37.77%. (Against the branch default these read 45.52% and
28.80%, and 51.94% and 39.99%: both of those cells are dictionary-overflow
cells, so the branch default is the inflated reference and flatters every arm.)
B's unit of decision is a whole column chunk, and a settled
leaf re-races only every 8 row groups, so after the data changes at row group 10
it keeps writing the settled dictionary candidate until its next scheduled race.
A re-opens every 4 row groups *and* watches the live dictionary inside a chunk,
abandoning it as soon as distinct entries pass a quarter of the values written,
so it reacts within the row group in which the data changes. That is the
concrete capability the page grain buys.

### Where B wins

**Compressed versions of the two mixed datasets.** Under ZSTD, B is 7.77% and
25.83% below vanilla main on shifting strings and the records-like schema,
against A's 4.71% and 24.92%. A's advantage on these two datasets is largest
uncompressed and shrinks or reverses once ZSTD runs, which is consistent with
the general compressor already removing much of the redundancy that
finer-grained encoding switching targets. Every option remains well ahead of the
baseline in all of these cells.

### Where C lands

C is the intended shape in most cells and the fastest arm overall: it is at or
below A's time everywhere except a noise-level float cell, and far below B's
everywhere, because only the leaves actually deciding leave the ordinary write
path. On bytes it matches A exactly on floats and timestamps, beats A on
low-cardinality strings, and beats B on both uncompressed mixed datasets.

C fails to match its stronger parent in three cells, all for the same reason:

* shifting strings, uncompressed: C is -33.73% against A's -37.34%.
* records-like, uncompressed: C is -47.54% against A's -50.16%.
* shifting strings, ZSTD: C is -7.00% against B's -7.77%.

In the two uncompressed cells C should have matched A and did not. Between
scheduled re-races a settled leaf sits on the standard path, where its only way
to react to changing data is `DictionaryFallback::WhenProfitable`, a chunk-level
dictionary decision. A, which stays on the page grain for every chunk, re-decides
every page and can also switch to a delta encoding mid-chunk. That is the third
seam above, priced: routing a settled leaf back to the standard path costs roughly
2.5 to 3 percentage points on data that changes character between races. The
gap would close if C looked again more often, at CPU it currently does not
spend, or if the harness applied a chunk-level dictionary rule on the standard
path that matched its per-page one more closely.

### Cost

A's median write time is below the baseline on most cells and its worst is about
1.4x. C is below the baseline on more cells still and never much above it. B is
below the baseline on four of the ten cells and reaches roughly 3x on the two
string datasets, where it races the widest candidate set. The structural reason
is in the designs rather than in tuning: B allocates and drives K complete sets
of column writers for every leaf of every racing row group, including leaves
that are not racing, while A's extra work is one additional encode per candidate
per raced page and falls to a single encode once the column settles. C pays that
only for leaves that are still deciding. B's cost is also charged per row group,
so a file with many small row groups pays it more often.


### Public corpora, and what 105 columns change

TPC-H and ClickBench are in the Public datasets section below. Three findings
there do not appear anywhere in the synthetic suite.

**The hybrid makes exactly A's decisions and is still smaller than A.** On all
five public files C's final encodings differ from A's on *zero* columns, yet C
is smaller than A on every one of them, by 0.70 to 1.86 percentage points of
vanilla main. Same encoding vocabulary, different bytes: for a column that has
settled, C writes through the ordinary column writer while A stays on the page
grain. The cause is page cadence rather than encoding: a page-grain page can
never span two `ArrowLeafColumn`s, because a `LeafCursor` covers one leaf and
encoding seals whatever the candidate has buffered when that cursor runs out.
With 8192 row record batches every one of A's pages is one batch, so A writes
`ceil(rows / 8192)` pages per chunk (55 and 68 on `hits_0`) where the ordinary
writer cuts at the 20000 row page budget (23 and 28). That is 12 922 pages for A
against C's 8 212 and the baseline's 4 492, and it accounts for all but 112
bytes of the 1 070 057 byte gap on `hits_0`: 826 987 bytes (77%) of extra
compressed payload from smaller compression windows, with the uncompressed
payload differing by only 39 462; 96 980 bytes (9%) of extra page headers; and
145 978 bytes (14%) of extra column and offset index, which carry per page
min/max. Re-reading the same input into whole row group record batches shrinks
the gap to 53 192 bytes on `hits_0` and reverses it on TPC-H `orders`, where A
then finishes 6 421 bytes *below* C, which places the cause in the cadence
rather than in the path. Routing settled leaves off the page grain is one way to
recover the ordinary cadence; feeding the page grain larger leaves is the other,
at the memory cost of holding them.

**Width costs Option B, not Option A or C.** ClickBench `hits` is about 105
mixed-character columns: URLs and titles, user agents, tiny enums, timestamps
and 64 bit IDs. B is about 1.9x to 2.7x the branch default's wall clock on these
files against 0.93x to 1.18x for A and 0.87x to 0.94x for C, because B builds
and drives
K complete sets of column writers for all 105 leaves of every racing row group
whether or not a given leaf is still racing. A and C pay only for the leaves
that are actually deciding, and C is the fastest arm on all three files. B still
wins bytes on two of the three, by 1.35 to 1.99 points.

**Row group shape moves the answer more than the data does.** `hits_2` has very
uneven input row groups (13 006 to 335 872 rows). There B changes the branch
default's encoding on 68 of 105 columns while A and C change 23, and A and B disagree on
58 columns, against 15 on the other two files. B decides per whole column chunk,
so a 13 006 row chunk gives it very different evidence than a 335 872 row one.
`hits_2` is also the one file where C beats B outright (-6.69% against
-6.53%).

No column shape in either corpus was passed through: all 9, 16 and 105 leaves
were raceable, so nothing in these results is masked by an unraced column.

### Anomalies

* Option A is 0.29% *larger* than both baselines on low-cardinality strings
  (4 479 bytes over 20 row groups, about 224 bytes per row group) while landing
  on the same encodings. Since the encodings match, this is page-boundary drift:
  the candidate that decides A's page boundaries seals pages at slightly
  different offsets than the baseline's own page budget does. A candidate that
  loses is never committed and cannot contribute bytes.
* Option A's synthetic bytes moved by at most 0.016% (four cells, largest 1 498
  bytes) when both adaptive arms became one shared harness. The harness counts
  the values it has sent through a dictionary from each page's `num_values`,
  which counts levels, where the previous page-grain-only harness read a
  library-side counter of non-null values. The two differ only on nullable
  columns, and only in when the dictionary-watching rule fires. Every Option B
  and Option C cell, and every cell of every public file, is byte-identical
  across that change.
* The options differ from each other by around 0.1% on datasets where they all
  choose the same encoding for every row group, which is the same
  page-boundary effect.
* The baseline's ZSTD file for timestamps is 7.78 MiB against 19.12 MiB
  uncompressed, while every option's file is 2.91 MiB either way. Delta encoded
  output is already close to incompressible, so ZSTD narrows the gap from 84.8%
  to 62.5% without changing the ordering.
* On TPC-H `orders`, A and C pick `DELTA_BINARY_PACKED` for `o_orderdate` and
  `DELTA_BYTE_ARRAY` for `o_clerk` where B keeps the dictionary, and B is still
  smaller overall. Per-page switching can lose to a chunk-level dictionary once
  a general compressor runs.
";

// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("parquet_bakeoff");
    std::fs::create_dir_all(&dir)?;

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--public <label> <files...>` records the results in BAKEOFF.md; without
    // it, supplied files are reported to stdout only.
    let public = if args.first().map(String::as_str) == Some("--public") {
        args.remove(0);
        if args.is_empty() {
            return Err(ParquetError::General(
                "--public needs a corpus label followed by the files".to_string(),
            ));
        }
        Some(args.remove(0))
    } else {
        None
    };

    let result = if args.is_empty() {
        if public.is_some() {
            return Err(ParquetError::General(
                "--public needs at least one parquet file".to_string(),
            ));
        }
        run_synthetic(&dir)
    } else {
        run_files(&args, &dir, public.as_deref())
    };

    let _ = std::fs::remove_dir_all(&dir);
    result
}

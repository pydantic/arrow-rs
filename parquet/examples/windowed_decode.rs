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

//! Compares three ways of driving [`ParquetPushDecoder`] over a file with one
//! large row group:
//!
//! 1. **row group** — today's behaviour: ask for the whole row group, then
//!    decode it.
//! 2. **naive windows** — a *fresh decoder per window*, carving the row group
//!    up with `with_offset` / `with_limit`. This needs no new API at all and
//!    gets low time-to-first-batch and low resident bytes — but every window
//!    rebuilds the column readers, re-decodes dictionaries, and refetches
//!    pages.
//! 3. **retained windows** — `with_row_window`, which keeps the row group's
//!    decode state alive across windows.
//!
//! The target for (3) is to match (1) on bytes and CPU while matching (2) on
//! time-to-first-batch and peak resident bytes.
//!
//! Wall-clock here is pure decode: "fetching" a range is a slice of an
//! in-memory buffer. To say anything about latency the example also reports a
//! *modelled* elapsed time, charging each request round a fixed latency plus
//! its bytes at a fixed bandwidth. That model is deliberately crude — it
//! assumes requests are issued serially, which is the worst case for (2) and
//! (3) and the best case for (1) — but it is enough to show where the
//! row-group-granular shape hurts.
//!
//! Run with:
//! ```text
//! cargo run --release -p parquet --features arrow,async --example windowed_decode
//! ```

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::{
    ArrowPredicateFn, ArrowReaderMetadata, ArrowReaderOptions, RowFilter,
};
use parquet::arrow::push_decoder::{ParquetPushDecoder, ParquetPushDecoderBuilder};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::properties::WriterProperties;
use std::sync::Arc;
use std::time::{Duration, Instant};

const NUM_ROWS: usize = 2_000_000;
const ROWS_PER_PAGE: usize = 20_000;
const BATCH_SIZE: usize = 8192;
/// Rows of lookahead for the windowed runs.
const WINDOW_ROWS: usize = 8192;

/// Per-request latency in the modelled elapsed time.
const IO_LATENCY: Duration = Duration::from_millis(20);
/// Bytes per second in the modelled elapsed time (~1 Gbit/s).
const IO_BANDWIDTH: f64 = 125_000_000.0;

fn build_file() -> (Bytes, ArrowReaderMetadata) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let props = WriterProperties::builder()
        // One row group for the whole file: the case windowing is for.
        .set_max_row_group_row_count(Some(NUM_ROWS))
        .set_data_page_row_count_limit(ROWS_PER_PAGE)
        .set_write_batch_size(ROWS_PER_PAGE)
        .build();

    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, schema.clone(), Some(props)).unwrap();
    let chunk = 100_000;
    for start in (0..NUM_ROWS).step_by(chunk) {
        let end = (start + chunk).min(NUM_ROWS);
        let id: Vec<i64> = (start as i64..end as i64).collect();
        let value: Vec<i64> = id.iter().map(|v| v % 1000).collect();
        // ~100 distinct names, so the dictionary is real but not the whole
        // chunk — refetching it is a visible but not absurd cost.
        let name: Vec<String> = id.iter().map(|v| format!("name-{:04}", v % 100)).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(id)),
                Arc::new(Int64Array::from(value)),
                Arc::new(StringArray::from(name)),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();
    let bytes = Bytes::from(buffer);

    let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
    let metadata = ArrowReaderMetadata::load(&bytes, options).unwrap();
    (bytes, metadata)
}

/// `value < 500`: keeps about half the rows, and every page has matches, so
/// the filter cannot be turned into page skipping.
fn row_filter(metadata: &ArrowReaderMetadata) -> RowFilter {
    let schema = metadata.metadata().file_metadata().schema_descr();
    let mask = ProjectionMask::columns(schema, ["value"]);
    RowFilter::new(vec![Box::new(ArrowPredicateFn::new(
        mask,
        |batch: RecordBatch| {
            let values = batch.column(0).as_primitive::<Int64Type>();
            Ok(values
                .iter()
                .map(|v| Some(v.unwrap() < 500))
                .collect::<BooleanArray>())
        },
    ))])
}

#[derive(Debug, Default)]
struct Stats {
    rounds: usize,
    bytes: u64,
    peak_resident: u64,
    rows: usize,
    batches: usize,
    /// Wall-clock decode time (no I/O).
    cpu: Duration,
    cpu_to_first_batch: Duration,
    /// Modelled I/O time before the first batch, and in total.
    io_to_first_batch: Duration,
    io: Duration,
    bytes_to_first_batch: u64,
}

impl Stats {
    fn charge(&mut self, bytes: u64) {
        let cost = IO_LATENCY + Duration::from_secs_f64(bytes as f64 / IO_BANDWIDTH);
        self.io += cost;
    }

    fn modelled_ttfb(&self) -> Duration {
        self.cpu_to_first_batch + self.io_to_first_batch
    }

    fn modelled_total(&self) -> Duration {
        self.cpu + self.io
    }

    fn print(&self, label: &str) {
        println!(
            "  {label:<20} rounds {:>5}  bytes {:>10}  peak {:>9}  cpu {:>7.1?}               | first batch after {:>9} bytes, modelled {:>8.1?}   modelled total {:>8.1?}",
            self.rounds,
            self.bytes,
            self.peak_resident,
            self.cpu,
            self.bytes_to_first_batch,
            self.modelled_ttfb(),
            self.modelled_total(),
        );
    }
}

/// Drive one decoder to exhaustion, accumulating into `stats`.
fn drive(decoder: &mut ParquetPushDecoder, file: &Bytes, stats: &mut Stats, start: Instant) {
    loop {
        match decoder.try_decode().unwrap() {
            DecodeResult::NeedsData(ranges) => {
                stats.rounds += 1;
                let data: Vec<Bytes> = ranges
                    .iter()
                    .map(|r| file.slice(r.start as usize..r.end as usize))
                    .collect();
                let bytes: u64 = data.iter().map(|d| d.len() as u64).sum();
                stats.bytes += bytes;
                stats.charge(bytes);
                decoder.push_ranges(ranges, data).unwrap();
            }
            DecodeResult::Data(batch) => {
                if stats.batches == 0 {
                    stats.cpu_to_first_batch = start.elapsed();
                    stats.io_to_first_batch = stats.io;
                    stats.bytes_to_first_batch = stats.bytes;
                }
                stats.batches += 1;
                stats.rows += batch.num_rows();
            }
            DecodeResult::Finished => break,
        }
        stats.peak_resident = stats.peak_resident.max(decoder.buffered_bytes());
    }
}

fn base_builder(metadata: &ArrowReaderMetadata, filtered: bool) -> ParquetPushDecoderBuilder {
    let builder =
        ParquetPushDecoderBuilder::new_with_metadata(metadata.clone()).with_batch_size(BATCH_SIZE);
    if filtered {
        builder.with_row_filter(row_filter(metadata))
    } else {
        builder
    }
}

/// (1) Today's behaviour: a whole row group at a time.
fn run_row_group(file: &Bytes, metadata: &ArrowReaderMetadata, filtered: bool) -> Stats {
    let mut stats = Stats::default();
    let start = Instant::now();
    let mut decoder = base_builder(metadata, filtered).build().unwrap();
    drive(&mut decoder, file, &mut stats, start);
    stats.cpu = start.elapsed();
    stats
}

/// (2) A fresh decoder per window, using only today's public API.
fn run_naive_windows(
    file: &Bytes,
    metadata: &ArrowReaderMetadata,
    filtered: bool,
    window: usize,
) -> Stats {
    let mut stats = Stats::default();
    let start = Instant::now();
    let total_rows: usize = metadata
        .metadata()
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows() as usize)
        .sum();

    let mut offset = 0;
    while offset < total_rows {
        let mut decoder = base_builder(metadata, filtered)
            .with_offset(offset)
            .with_limit(window)
            .build()
            .unwrap();
        drive(&mut decoder, file, &mut stats, start);
        offset += window;
    }
    stats.cpu = start.elapsed();
    stats
}

/// (3) One decoder, windowed demand, retained decode state.
fn run_retained_windows(
    file: &Bytes,
    metadata: &ArrowReaderMetadata,
    filtered: bool,
    window: usize,
) -> Stats {
    let mut stats = Stats::default();
    let start = Instant::now();
    let mut decoder = base_builder(metadata, filtered)
        .with_row_window(window)
        .build()
        .unwrap();
    drive(&mut decoder, file, &mut stats, start);
    stats.cpu = start.elapsed();
    stats
}

fn scenario(file: &Bytes, metadata: &ArrowReaderMetadata, filtered: bool) {
    println!(
        "{}:",
        if filtered {
            "with a RowFilter (value < 500, ~50% selective, no page skipping)"
        } else {
            "plain scan"
        }
    );
    // Warm up.
    run_row_group(file, metadata, filtered);
    run_retained_windows(file, metadata, filtered, WINDOW_ROWS);

    let a = run_row_group(file, metadata, filtered);
    let b = run_naive_windows(file, metadata, filtered, WINDOW_ROWS);
    assert_eq!(a.rows, b.rows, "naive windows produced different rows");
    println!("  ({} rows out)", a.rows);
    a.print("(a) row group");
    b.print("(b) naive window");
    // Window size is the knob trading request count against resident bytes:
    // the same retained state serves any of them.
    for window in [WINDOW_ROWS, 8 * WINDOW_ROWS, 32 * WINDOW_ROWS] {
        let c = run_retained_windows(file, metadata, filtered, window);
        assert_eq!(a.rows, c.rows, "retained windows produced different rows");
        c.print(&format!("(c) retained w={window}"));
    }
    println!();
}

fn main() {
    let (file, metadata) = build_file();
    println!(
        "file {} bytes, {} row group(s), {NUM_ROWS} rows, {ROWS_PER_PAGE} rows/page, \
         batch_size {BATCH_SIZE}, window {WINDOW_ROWS}",
        file.len(),
        metadata.metadata().num_row_groups(),
    );
    println!(
        "modelled I/O: {IO_LATENCY:?} per request + {} MB/s, requests issued *serially*.",
        IO_BANDWIDTH / 1_000_000.0
    );
    println!(
        "NOTE: the serial assumption makes `modelled total` a floor for (a) and a ceiling for\n         (b)/(c): expressing demand incrementally only pays off if the caller keeps several\n         requests in flight, which this model cannot express. Read `modelled total` as \"what\n         a caller that never overlaps would get\", and the window sweep as the knob for trading\n         request count against resident bytes.\n"
    );

    scenario(&file, &metadata, false);
    scenario(&file, &metadata, true);
}

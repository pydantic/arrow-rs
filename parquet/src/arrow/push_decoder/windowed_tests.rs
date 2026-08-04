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

//! Equivalence tests for windowed decoding.
//!
//! Every test here asserts the same property: a windowed read produces the
//! same batches, in the same order, as the non-windowed read of the same
//! scan. Windowing changes *when bytes are asked for*, never what comes out.

use crate::DecodeResult;
use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use crate::arrow::arrow_reader::{
    ArrowPredicateFn, ArrowReaderMetadata, ArrowReaderOptions, RowFilter, RowSelection,
    RowSelectionPolicy, RowSelector,
};
use crate::arrow::push_decoder::{ParquetPushDecoder, ParquetPushDecoderBuilder};
use crate::arrow::{ArrowWriter, ProjectionMask};
use crate::file::metadata::PageIndexPolicy;
use crate::file::properties::WriterProperties;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use std::ops::Range;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// test file
// ---------------------------------------------------------------------------

const ROWS_PER_ROW_GROUP: usize = 600;
const ROWS_PER_PAGE: usize = 25;
const NUM_ROWS: usize = 1800;

/// Three row groups of 600 rows, 25 rows per page, columns:
/// * `a`: 0, 1, 2, ...
/// * `b`: a % 10  (used for selective predicates)
/// * `c`: a string, wide enough that page granularity is visible
///
/// Dictionary encoding is off. With unique values a dictionary page would be
/// most of the column chunk, and since it is needed for *every* row it is
/// resident for the whole row group — which would swamp the byte-shape
/// assertions without saying anything about windowing. Dictionary handling is
/// covered separately by [`dictionary_page_is_fetched_once`].
fn test_file() -> (Bytes, ArrowReaderMetadata) {
    test_file_with_dictionary(false)
}

fn test_file_with_dictionary(dictionary: bool) -> (Bytes, ArrowReaderMetadata) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Utf8, false),
    ]));
    let a: Vec<i64> = (0..NUM_ROWS as i64).collect();
    let b: Vec<i64> = a.iter().map(|v| v % 10).collect();
    let c: Vec<String> = a.iter().map(|v| format!("row-{v:08}-padding")).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(a)),
            Arc::new(Int64Array::from(b)),
            Arc::new(StringArray::from(c)),
        ],
    )
    .unwrap();

    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROWS_PER_ROW_GROUP))
        .set_data_page_row_count_limit(ROWS_PER_PAGE)
        .set_write_batch_size(ROWS_PER_PAGE)
        .set_dictionary_enabled(dictionary)
        .build();
    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let bytes = Bytes::from(buffer);

    let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
    let metadata = ArrowReaderMetadata::load(&bytes, options).unwrap();
    assert!(metadata.metadata().offset_index().is_some());
    (bytes, metadata)
}

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------

/// What a decode run cost, alongside its output.
#[derive(Debug, Default, PartialEq, Eq)]
struct Cost {
    /// Number of `NeedsData` round trips.
    rounds: usize,
    /// Total bytes handed to the decoder.
    bytes: u64,
    /// Highest `buffered_bytes()` observed.
    peak_buffered: u64,
}

fn drive(mut decoder: ParquetPushDecoder, file: &Bytes) -> (Vec<RecordBatch>, Cost) {
    let mut batches = Vec::new();
    let mut cost = Cost::default();
    loop {
        match decoder.try_decode().unwrap() {
            DecodeResult::NeedsData(ranges) => {
                cost.rounds += 1;
                let data: Vec<Bytes> = ranges
                    .iter()
                    .map(|r| file.slice(r.start as usize..r.end as usize))
                    .collect();
                cost.bytes += data.iter().map(|d| d.len() as u64).sum::<u64>();
                decoder.push_ranges(ranges, data).unwrap();
            }
            DecodeResult::Data(batch) => {
                cost.peak_buffered = cost.peak_buffered.max(decoder.buffered_bytes());
                batches.push(batch);
            }
            DecodeResult::Finished => break,
        }
        cost.peak_buffered = cost.peak_buffered.max(decoder.buffered_bytes());
    }
    (batches, cost)
}

/// A scan, so the same configuration can be built windowed and not.
#[derive(Clone, Default)]
struct Scan {
    batch_size: Option<usize>,
    projection: Option<ProjectionMask>,
    selection: Option<RowSelection>,
    limit: Option<usize>,
    offset: Option<usize>,
    /// Built fresh for each decoder: predicates are stateful.
    predicates: Vec<PredicateSpec>,
    policy: Option<RowSelectionPolicy>,
    metrics: Option<ArrowReaderMetrics>,
    max_predicate_cache_size: Option<usize>,
}

#[derive(Clone)]
enum PredicateSpec {
    /// `column` compared against a threshold with the given kind.
    Filter {
        column: &'static str,
        kind: Cmp,
        value: i64,
    },
}

#[derive(Clone, Copy)]
enum Cmp {
    Lt,
    Ge,
    ModNotZero,
    All,
    None,
}

impl Scan {
    fn build(&self, metadata: &ArrowReaderMetadata, file: &Bytes) -> ParquetPushDecoder {
        let mut builder = ParquetPushDecoderBuilder::new_with_metadata(metadata.clone());
        if let Some(batch_size) = self.batch_size {
            builder = builder.with_batch_size(batch_size);
        }
        if let Some(projection) = &self.projection {
            builder = builder.with_projection(projection.clone());
        }
        if let Some(selection) = &self.selection {
            builder = builder.with_row_selection(selection.clone());
        }
        if let Some(limit) = self.limit {
            builder = builder.with_limit(limit);
        }
        if let Some(offset) = self.offset {
            builder = builder.with_offset(offset);
        }
        if let Some(policy) = self.policy {
            builder = builder.with_row_selection_policy(policy);
        }
        if let Some(metrics) = &self.metrics {
            builder = builder.with_metrics(metrics.clone());
        }
        if let Some(size) = self.max_predicate_cache_size {
            builder = builder.with_max_predicate_cache_size(size);
        }
        if !self.predicates.is_empty() {
            let schema = metadata.metadata().file_metadata().schema_descr();
            let predicates = self
                .predicates
                .iter()
                .map(|spec| spec.build(schema))
                .collect();
            builder = builder.with_row_filter(RowFilter::new(predicates));
        }
        let _ = file;
        builder.build().unwrap()
    }
}

impl PredicateSpec {
    fn build(
        &self,
        schema: &crate::schema::types::SchemaDescriptor,
    ) -> Box<dyn crate::arrow::arrow_reader::ArrowPredicate> {
        let PredicateSpec::Filter {
            column,
            kind,
            value,
        } = *self;
        let mask = ProjectionMask::columns(schema, [column]);
        Box::new(ArrowPredicateFn::new(mask, move |batch: RecordBatch| {
            let values = batch.column(0).as_primitive::<Int64Type>();
            let result: BooleanArray = values
                .iter()
                .map(|v| {
                    let v = v.unwrap();
                    Some(match kind {
                        Cmp::Lt => v < value,
                        Cmp::Ge => v >= value,
                        Cmp::ModNotZero => v % value != 0,
                        Cmp::All => true,
                        Cmp::None => false,
                    })
                })
                .collect();
            Ok(result)
        }))
    }
}

/// Assert a windowed read of `scan` matches the non-windowed read batch for
/// batch, and return the two costs (non-windowed, windowed).
#[track_caller]
fn assert_windowed_matches(scan: &Scan, window_rows: usize) -> (Cost, Cost) {
    let (file, metadata) = test_file();

    let (expected, base_cost) = drive(scan.build(&metadata, &file), &file);
    let windowed = scan
        .build(&metadata, &file)
        .into_builder()
        .unwrap()
        .with_row_window(window_rows)
        .build()
        .unwrap();
    let (actual, windowed_cost) = drive(windowed, &file);

    assert_eq!(
        actual.len(),
        expected.len(),
        "batch count differs: windowed {:?} vs plain {:?}",
        actual.iter().map(|b| b.num_rows()).collect::<Vec<_>>(),
        expected.iter().map(|b| b.num_rows()).collect::<Vec<_>>(),
    );
    for (i, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(actual, expected, "batch {i} differs");
    }
    (base_cost, windowed_cost)
}

// ---------------------------------------------------------------------------
// unfiltered
// ---------------------------------------------------------------------------

#[test]
fn plain_scan() {
    let scan = Scan {
        batch_size: Some(100),
        ..Default::default()
    };
    let (base, windowed) = assert_windowed_matches(&scan, 100);
    // Same bytes, but asked for in many more, smaller rounds.
    assert_eq!(base.bytes, windowed.bytes);
    assert!(
        windowed.rounds > base.rounds,
        "windowed should ask in more rounds: {windowed:?} vs {base:?}"
    );
    assert!(
        windowed.peak_buffered < base.peak_buffered,
        "windowed should hold fewer bytes: {windowed:?} vs {base:?}"
    );
}

#[test]
fn plain_scan_window_larger_than_row_group() {
    let scan = Scan {
        batch_size: Some(128),
        ..Default::default()
    };
    let (base, windowed) = assert_windowed_matches(&scan, 100_000);
    assert_eq!(base.bytes, windowed.bytes);
}

#[test]
fn plain_scan_window_smaller_than_batch_is_raised() {
    // A window below `batch_size` cannot be honoured (a batch is the smallest
    // decodable unit) and is raised to it rather than deadlocking.
    let scan = Scan {
        batch_size: Some(256),
        ..Default::default()
    };
    assert_windowed_matches(&scan, 1);
}

#[test]
fn with_projection() {
    let (_, metadata) = test_file();
    let schema = metadata.metadata().file_metadata().schema_descr();
    let scan = Scan {
        batch_size: Some(100),
        projection: Some(ProjectionMask::columns(schema, ["a", "c"])),
        ..Default::default()
    };
    let (base, windowed) = assert_windowed_matches(&scan, 200);
    assert_eq!(base.bytes, windowed.bytes);
}

#[test]
fn with_row_selection() {
    // Alternating runs, so pages are skipped.
    let selectors: Vec<RowSelector> = (0..NUM_ROWS / 60)
        .flat_map(|_| [RowSelector::select(20), RowSelector::skip(40)])
        .collect();
    let scan = Scan {
        batch_size: Some(64),
        selection: Some(RowSelection::from(selectors)),
        ..Default::default()
    };
    let (base, windowed) = assert_windowed_matches(&scan, 128);
    assert_eq!(base.bytes, windowed.bytes);
}

#[test]
fn with_limit_and_offset() {
    let scan = Scan {
        batch_size: Some(70),
        offset: Some(650),
        limit: Some(500),
        ..Default::default()
    };
    let (base, windowed) = assert_windowed_matches(&scan, 140);
    assert_eq!(base.bytes, windowed.bytes);
}

#[test]
fn empty_result_from_selection() {
    let scan = Scan {
        batch_size: Some(64),
        selection: Some(RowSelection::from(vec![RowSelector::skip(NUM_ROWS)])),
        ..Default::default()
    };
    let (_, windowed) = assert_windowed_matches(&scan, 128);
    assert_eq!(windowed.bytes, 0);
}

#[test]
fn dictionary_page_is_fetched_once() {
    // The point of retaining decode state across windows: a dictionary is
    // fetched (and decoded) once per column chunk, not once per window. The
    // naive alternative — a fresh decoder per window — refetches it every
    // time.
    let (file, metadata) = test_file_with_dictionary(true);
    let scan = Scan {
        batch_size: Some(50),
        ..Default::default()
    };
    let mut decoder = scan
        .build(&metadata, &file)
        .into_builder()
        .unwrap()
        .with_row_window(50)
        .build()
        .unwrap();

    let mut requests: Vec<Range<u64>> = Vec::new();
    loop {
        match decoder.try_decode().unwrap() {
            DecodeResult::NeedsData(ranges) => {
                let data = ranges
                    .iter()
                    .map(|r| file.slice(r.start as usize..r.end as usize))
                    .collect();
                requests.extend(ranges.iter().cloned());
                decoder.push_ranges(ranges, data).unwrap();
            }
            DecodeResult::Data(_) => {}
            DecodeResult::Finished => break,
        }
    }

    // Every dictionary page is the run of bytes before a chunk's first data
    // page; assert each was asked for exactly once.
    let meta = metadata.metadata();
    for rg in 0..meta.num_row_groups() {
        let offset_index = &meta.offset_index().unwrap()[rg];
        for (col, column) in meta.row_group(rg).columns().iter().enumerate() {
            let (chunk_start, _) = column.byte_range();
            let first_page = offset_index[col].page_locations()[0].offset as u64;
            assert!(first_page > chunk_start, "expected a dictionary page");
            let dictionary = chunk_start..first_page;
            let count = requests.iter().filter(|r| **r == dictionary).count();
            assert_eq!(count, 1, "dictionary {dictionary:?} fetched {count} times");
        }
    }
}

#[test]
fn falls_back_without_offset_index() {
    // Without page locations there is no way to express per-page demand, so
    // the row group is read whole — but the output must be unchanged.
    let (file, _) = test_file();
    let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Skip);
    let metadata = ArrowReaderMetadata::load(&file, options).unwrap();
    assert!(metadata.metadata().offset_index().is_none());

    let scan = Scan {
        batch_size: Some(100),
        ..Default::default()
    };
    let (expected, _) = drive(scan.build(&metadata, &file), &file);
    let windowed = scan
        .build(&metadata, &file)
        .into_builder()
        .unwrap()
        .with_row_window(100)
        .build()
        .unwrap();
    let (actual, _) = drive(windowed, &file);
    assert_eq!(actual, expected);
}

#[test]
fn try_next_reader_rejected_when_windowed() {
    let (file, metadata) = test_file();
    let scan = Scan::default();
    let mut decoder = scan
        .build(&metadata, &file)
        .into_builder()
        .unwrap()
        .with_row_window(1024)
        .build()
        .unwrap();
    let err = decoder.try_next_reader().unwrap_err().to_string();
    assert!(
        err.contains("not supported while decoding in windows"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// filtered
// ---------------------------------------------------------------------------

/// Windowed output is driven by explicit selectors, so compare against a
/// non-windowed read pinned to the same strategy.
fn filtered_scan(predicates: Vec<PredicateSpec>) -> Scan {
    Scan {
        batch_size: Some(100),
        predicates,
        policy: Some(RowSelectionPolicy::Selectors),
        ..Default::default()
    }
}

#[test]
fn single_predicate() {
    let scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "a",
        kind: Cmp::Lt,
        value: 900,
    }]);
    let (base, windowed) = assert_windowed_matches(&scan, 200);
    assert!(windowed.rounds > base.rounds);
}

#[test]
fn predicate_chain() {
    let scan = filtered_scan(vec![
        PredicateSpec::Filter {
            column: "a",
            kind: Cmp::Ge,
            value: 300,
        },
        PredicateSpec::Filter {
            column: "b",
            kind: Cmp::ModNotZero,
            value: 3,
        },
    ]);
    assert_windowed_matches(&scan, 200);
}

#[test]
fn three_predicates() {
    let scan = filtered_scan(vec![
        PredicateSpec::Filter {
            column: "a",
            kind: Cmp::Ge,
            value: 100,
        },
        PredicateSpec::Filter {
            column: "a",
            kind: Cmp::Lt,
            value: 1500,
        },
        PredicateSpec::Filter {
            column: "b",
            kind: Cmp::ModNotZero,
            value: 2,
        },
    ]);
    assert_windowed_matches(&scan, 300);
}

#[test]
fn predicate_with_row_selection() {
    let selectors: Vec<RowSelector> = (0..NUM_ROWS / 100)
        .flat_map(|_| [RowSelector::select(60), RowSelector::skip(40)])
        .collect();
    let mut scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "b",
        kind: Cmp::ModNotZero,
        value: 4,
    }]);
    scan.selection = Some(RowSelection::from(selectors));
    assert_windowed_matches(&scan, 200);
}

#[test]
fn predicate_with_limit() {
    let mut scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "a",
        kind: Cmp::Ge,
        value: 250,
    }]);
    scan.limit = Some(333);
    assert_windowed_matches(&scan, 200);
}

#[test]
fn predicate_with_limit_and_offset() {
    let mut scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "b",
        kind: Cmp::ModNotZero,
        value: 3,
    }]);
    scan.offset = Some(211);
    scan.limit = Some(400);
    assert_windowed_matches(&scan, 200);
}

#[test]
fn predicate_empty_early_then_matching() {
    // Nothing matches in the first row group and a bit beyond, so several
    // windows produce no rows at all before any batch can be emitted.
    let scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "a",
        kind: Cmp::Ge,
        value: 750,
    }]);
    let (_, windowed) = assert_windowed_matches(&scan, 150);
    assert!(windowed.rounds > 0);
}

#[test]
fn predicate_matches_nothing() {
    let scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "a",
        kind: Cmp::None,
        value: 0,
    }]);
    assert_windowed_matches(&scan, 200);
}

#[test]
fn predicate_matches_everything() {
    let scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "a",
        kind: Cmp::All,
        value: 0,
    }]);
    assert_windowed_matches(&scan, 200);
}

#[test]
fn predicate_with_projection_disjoint_from_filter() {
    let (_, metadata) = test_file();
    let schema = metadata.metadata().file_metadata().schema_descr();
    let mut scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "b",
        kind: Cmp::ModNotZero,
        value: 3,
    }]);
    scan.projection = Some(ProjectionMask::columns(schema, ["a", "c"]));
    assert_windowed_matches(&scan, 200);
}

#[test]
fn predicate_window_smaller_than_page() {
    // Window below the page size: every window touches a single page, which
    // is the most adversarial case for the per-window bookkeeping.
    let mut scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "b",
        kind: Cmp::ModNotZero,
        value: 5,
    }]);
    scan.batch_size = Some(10);
    assert_windowed_matches(&scan, 10);
}

#[test]
fn predicate_cache_is_still_used() {
    // `b` is both filtered on and projected, so the output phase should read
    // it from the predicate cache rather than decoding it a second time.
    let (file, metadata) = test_file();
    let schema = metadata.metadata().file_metadata().schema_descr();
    let metrics = ArrowReaderMetrics::enabled();
    let mut scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "b",
        kind: Cmp::ModNotZero,
        value: 3,
    }]);
    scan.projection = Some(ProjectionMask::columns(schema, ["a", "b"]));
    scan.metrics = Some(metrics.clone());

    let windowed = scan
        .build(&metadata, &file)
        .into_builder()
        .unwrap()
        .with_row_window(200)
        .build()
        .unwrap();
    let (batches, _) = drive(windowed, &file);
    assert!(!batches.is_empty());
    let from_cache = metrics.records_read_from_cache().unwrap();
    assert!(
        from_cache > 0,
        "windowed decoding lost predicate-cache reuse: from_cache={from_cache}, \
         from_inner={:?}",
        metrics.records_read_from_inner()
    );
}

#[test]
fn filtered_output_matches_default_policy_end_to_end() {
    // The windowed output phase pins the `Selectors` strategy, so batch
    // *boundaries* can differ from an `Auto`-policy read. The rows themselves
    // must not.
    let (file, metadata) = test_file();
    let mut scan = filtered_scan(vec![PredicateSpec::Filter {
        column: "b",
        kind: Cmp::ModNotZero,
        value: 3,
    }]);
    scan.policy = None;
    let (expected, _) = drive(scan.build(&metadata, &file), &file);

    let windowed = scan
        .build(&metadata, &file)
        .into_builder()
        .unwrap()
        .with_row_window(200)
        .build()
        .unwrap();
    let (actual, _) = drive(windowed, &file);

    let schema = expected[0].schema();
    let expected = arrow_select::concat::concat_batches(&schema, &expected).unwrap();
    let actual = arrow_select::concat::concat_batches(&schema, &actual).unwrap();
    assert_eq!(actual, expected);
}

// ---------------------------------------------------------------------------
// demand shape
// ---------------------------------------------------------------------------

#[test]
fn first_request_covers_only_the_first_window() {
    let (file, metadata) = test_file();
    let scan = Scan {
        batch_size: Some(50),
        ..Default::default()
    };

    let mut plain = scan.build(&metadata, &file);
    let DecodeResult::NeedsData(plain_ranges) = plain.try_decode().unwrap() else {
        panic!("expected NeedsData")
    };
    let plain_bytes: u64 = plain_ranges.iter().map(|r| r.end - r.start).sum();

    let mut windowed = scan
        .build(&metadata, &file)
        .into_builder()
        .unwrap()
        .with_row_window(50)
        .build()
        .unwrap();
    let DecodeResult::NeedsData(windowed_ranges) = windowed.try_decode().unwrap() else {
        panic!("expected NeedsData")
    };
    let windowed_bytes: u64 = windowed_ranges.iter().map(|r| r.end - r.start).sum();

    // A row group is 600 rows; the first window is 50, so the first request
    // should be far smaller.
    assert!(
        windowed_bytes * 4 < plain_bytes,
        "first windowed request {windowed_bytes} not much smaller than {plain_bytes}"
    );
}

#[test]
fn resident_bytes_stay_bounded_across_a_row_group() {
    let (file, metadata) = test_file();
    let scan = Scan {
        batch_size: Some(50),
        ..Default::default()
    };
    let mut decoder = scan
        .build(&metadata, &file)
        .into_builder()
        .unwrap()
        .with_row_window(50)
        .build()
        .unwrap();

    let mut peak = 0u64;
    let mut requests: Vec<Range<u64>> = Vec::new();
    loop {
        match decoder.try_decode().unwrap() {
            DecodeResult::NeedsData(ranges) => {
                let data = ranges
                    .iter()
                    .map(|r| file.slice(r.start as usize..r.end as usize))
                    .collect();
                requests.extend(ranges.iter().cloned());
                decoder.push_ranges(ranges, data).unwrap();
            }
            DecodeResult::Data(_) => {}
            DecodeResult::Finished => break,
        }
        peak = peak.max(decoder.buffered_bytes());
    }

    let row_group_bytes: u64 = metadata
        .metadata()
        .row_group(0)
        .columns()
        .iter()
        .map(|c| c.byte_range().1)
        .sum();
    assert!(
        peak * 3 < row_group_bytes,
        "peak resident {peak} not well under one row group {row_group_bytes}"
    );
    // Nothing should be requested twice: retained state means no refetching.
    let mut sorted = requests.clone();
    sorted.sort_by_key(|r| (r.start, r.end));
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        requests.len(),
        "a range was requested more than once"
    );
}

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

//! Writes one file with [`AdaptiveWriter`] and the same file with a stock
//! [`ArrowWriter`] at identical properties, and reports the difference.
//!
//! The harness itself is in `harness.rs` beside this file, split into the
//! policy half and the plumbing half. This file is only a dataset and a
//! comparison.
//!
//! The dataset changes character halfway through: dictionary-friendly for the
//! first million rows, unique per row for the second. It is the case that
//! cannot be expressed by configuring a writer up front, because the right
//! encoding for the file is two different encodings.
//!
//! ```text
//! cargo run --release --features arrow --example adaptive_writer
//! ```

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::Result;
use parquet::file::properties::WriterProperties;

#[path = "harness.rs"]
mod harness;

use harness::AdaptiveWriter;

const ROWS: usize = 2_000_000;
const ROWS_PER_BATCH: usize = 25_000;
const ROWS_PER_ROW_GROUP: usize = 250_000;

/// splitmix64, so the dataset is identical on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]))
}

/// One batch of the dataset: 32 repeated values before the halfway point, and a
/// distinct value per row after it.
fn batch(schema: &SchemaRef, rng: &mut Rng, start: usize) -> RecordBatch {
    let values: Vec<String> = (start..start + ROWS_PER_BATCH)
        .map(|row| {
            let r = rng.next();
            if row < ROWS / 2 {
                format!("category-{:02}", r % 32)
            } else {
                format!("event-{row:08}-{r:016x}")
            }
        })
        .collect();
    let array: ArrayRef = Arc::new(StringArray::from(values));
    RecordBatch::try_new(schema.clone(), vec![array]).expect("valid batch")
}

fn properties() -> WriterProperties {
    WriterProperties::builder()
        .set_data_page_row_count_limit(20_000)
        .build()
}

/// Writes the dataset with the adaptive writer, ending a row group every
/// `ROWS_PER_ROW_GROUP` rows.
fn write_adaptive(path: &Path) -> Result<()> {
    let schema = schema();
    let mut rng = Rng(0x5EED);
    let mut writer = AdaptiveWriter::try_new(File::create(path)?, schema.clone(), properties())?;

    for start in (0..ROWS).step_by(ROWS_PER_BATCH) {
        writer.write(&batch(&schema, &mut rng, start))?;
        if (start + ROWS_PER_BATCH).is_multiple_of(ROWS_PER_ROW_GROUP) {
            writer.flush()?;
        }
    }
    writer.close()
}

/// The same dataset, the same properties, the same row group boundaries.
fn write_baseline(path: &Path) -> Result<()> {
    let schema = schema();
    let mut rng = Rng(0x5EED);
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema.clone(), Some(properties()))?;

    for start in (0..ROWS).step_by(ROWS_PER_BATCH) {
        writer.write(&batch(&schema, &mut rng, start))?;
        if (start + ROWS_PER_BATCH).is_multiple_of(ROWS_PER_ROW_GROUP) {
            writer.flush()?;
        }
    }
    writer.close()?;
    Ok(())
}

/// Reads `path` back, checks it against the source rows, and reports the
/// encodings its column actually ended up with.
fn verify(path: &Path) -> Result<Vec<String>> {
    let schema = schema();
    let mut rng = Rng(0x5EED);
    let mut expected: Vec<RecordBatch> = Vec::new();
    for start in (0..ROWS).step_by(ROWS_PER_BATCH) {
        expected.push(batch(&schema, &mut rng, start));
    }

    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let metadata = builder.metadata().clone();
    let mut encodings: Vec<String> = Vec::new();
    for row_group in metadata.row_groups() {
        for encoding in row_group.column(0).encodings() {
            let text = encoding.to_string();
            if !encodings.contains(&text) {
                encodings.push(text);
            }
        }
    }
    encodings.sort();

    let mut rows = 0usize;
    for read in builder.with_batch_size(ROWS_PER_BATCH).build()? {
        let read = read?;
        let source = &expected[rows / ROWS_PER_BATCH];
        assert_eq!(
            source.column(0),
            read.column(0),
            "row values differ at row {rows} of {}",
            path.display()
        );
        rows += read.num_rows();
    }
    assert_eq!(rows, ROWS, "{} lost rows", path.display());
    Ok(encodings)
}

fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("parquet-adaptive-writer");
    std::fs::create_dir_all(&dir)?;
    let adaptive_path = dir.join("adaptive.parquet");
    let baseline_path = dir.join("baseline.parquet");

    write_adaptive(&adaptive_path)?;
    write_baseline(&baseline_path)?;

    let adaptive_encodings = verify(&adaptive_path)?;
    let baseline_encodings = verify(&baseline_path)?;

    let adaptive = std::fs::metadata(&adaptive_path)?.len();
    let baseline = std::fs::metadata(&baseline_path)?.len();
    let delta = 100.0 * (adaptive as f64 - baseline as f64) / baseline as f64;

    println!("{ROWS} rows, identical writer properties, identical row groups\n");
    println!(
        "  adaptive  {:>10} bytes   {}",
        adaptive,
        adaptive_encodings.join("+")
    );
    println!(
        "  baseline  {:>10} bytes   {}",
        baseline,
        baseline_encodings.join("+")
    );
    println!("\n  {delta:+.2}% against the stock writer");
    println!("\nfiles left in {}", dir.display());
    Ok(())
}

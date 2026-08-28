# Option B: merged writer APIs

This branch makes the "race K candidate encodings per column chunk" writing
style a first-class, supported use of the parquet writer, and adds the
dictionary-fallback control that style needs, with no other architectural
change.

It is the Option B entry in the parquet writer API bakeoff. The motivating
consumer builds K complete sets of `ArrowColumnWriter`s per row group, feeds
all of them the same leaves, and appends only the smallest chunk. That
consumer previously depended on the free function
`get_column_writers(schema_descr, &props, &arrow_schema)`, removed upstream in
3df22cd ("parquet: Remove deprecated functions (#10565)").

## What landed

Four commits:

| Commit | Content |
| --- | --- |
| `67f282c` | `ArrowRowGroupWriterFactory::create_column_writers_with_properties` |
| `d7e5758` | Port of apache/arrow-rs#10777, re-encode buffered values on dictionary fallback |
| `ce4a031` | Port of apache/arrow-rs#10775, the `DictionaryFallback` policy |
| `e9082e4` | `parquet/examples/advanced_racing_writer.rs` |

### Public items added

```rust
// parquet::arrow::arrow_writer
impl ArrowRowGroupWriterFactory {
    pub fn create_column_writers_with_properties(
        &self,
        row_group_index: usize,
        props: &WriterPropertiesPtr,
    ) -> Result<Vec<ArrowColumnWriter>>;
}

// parquet::file::properties
#[non_exhaustive]
pub enum DictionaryFallback {
    OnPageSizeLimit,
    WhenProfitable { worth_ratio: f64, max_dictionary_page_size: usize },
}
impl Default for DictionaryFallback { /* OnPageSizeLimit */ }

pub const DEFAULT_DICTIONARY_FALLBACK: DictionaryFallback;

impl WriterProperties {
    pub fn dictionary_fallback(&self) -> DictionaryFallback;
    pub fn column_dictionary_fallback(&self, col: &ColumnPath) -> DictionaryFallback;
}
impl WriterPropertiesBuilder {
    pub fn set_dictionary_fallback(self, value: DictionaryFallback) -> Self;
    pub fn set_column_dictionary_fallback(self, col: ColumnPath, value: DictionaryFallback) -> Self;
}

// parquet::column::writer::encoder (trait)
pub trait ColumnValueEncoder {
    fn estimated_plain_encoded_bytes(&self) -> Option<u64> { None }
    fn fall_back_from_dictionary(&mut self, retain_dictionary: bool) -> Result<()>;
}

// parquet::encodings::encoding
impl<T: DataType> DictEncoder<T> {
    pub fn plain_encoded_bytes(&self) -> u64;
    pub fn take_buffered_values(&mut self) -> Vec<T::T>;
}
```

`ColumnValueEncoder` is a public trait, so `fall_back_from_dictionary` is a
breaking addition for any external implementor. It is required rather than
defaulted because a silent default would reintroduce the #10777 bug for that
implementor. `estimated_plain_encoded_bytes` defaults to `None`, which
disables only the `WhenProfitable` refinement and preserves the absolute-limit
behavior.

### Provenance

Deliverable 2 is a faithful port of two public upstream proposals rather than
an independent design:

* **apache/arrow-rs#10775** is the `DictionaryFallback` API: the enum, both
  setters, both getters, `DEFAULT_DICTIONARY_FALLBACK`, the
  `estimated_plain_encoded_bytes` trait method, the `should_dict_fallback`
  rewrite, and the accompanying tests. Applied essentially verbatim.
* **apache/arrow-rs#10777** fixes dictionary fallback to re-encode the values
  buffered for the in-progress data page through the fallback encoder, instead
  of flushing them as one more dictionary-encoded page and always writing a
  dictionary page. This fork exhibited that bug (its `dict_fallback` was the
  unfixed version), and the racing example exercises fallback heavily, so the
  fix is ported too.
* **apache/arrow-rs#10780** (`DictionaryFallback::Adaptive`) is deliberately
  **not** implemented. See "Proposed next variant" below.

An earlier iteration of this branch carried a different, invented pair of knobs
(a standalone `set_column_dictionary_fallback_encoding`, and a
`dict_entries / values_written` ratio trigger). That approximation was discarded
once the real proposals became readable; nothing of it remains.

### Deviations from upstream, and why

1. **One test assertion changed.** #10775's
   `test_dictionary_fallback_explicit_default_policy_unchanged` asserted both
   `num_dict > 0` and `num_fallback > 0`. With #10777 also on this branch, that
   case now overflows the dictionary *before* the first data page is flushed,
   so the buffered values are re-encoded and the dictionary discarded: the
   chunk has no dictionary-encoded pages and no dictionary page at all. The
   assertion is updated to `num_dict == 0` plus `dictionary_page_offset()
   .is_none()`. The byte-identity assertion the test exists for is untouched
   and still passes. The two PRs are independent branches off the same upstream
   base, so upstream has not yet had to reconcile them.

2. **`create_column_writers_with_properties` takes `&WriterPropertiesPtr`, not
   `&WriterProperties`.** The column writers clone the `Arc` internally
   (`get_column_writer(desc.clone(), props.clone(), page_writer)`), so taking
   the pointer lets a caller building K candidate sets construct each `Arc`
   once instead of forcing one allocation and clone per leaf per candidate.
   This matches the shape of the removed free function and of the factory's own
   `props` field.

3. **`create_column_writers` is now a wrapper** that forwards its own
   `self.props`, rather than the two sharing a copied body. The deleted free
   function bypassed both the page-store factory and the file encryptor; the
   new method routes through `column_writer_factory(row_group_index)` exactly
   as `create_column_writers` did, so per-row-group properties do not silently
   cost a caller their page store or encryption setup.

## Complexity accounting

Added lines, splitting library production code from test code (`git diff
2567a32..HEAD`, added lines only; "test" is everything inside a `#[cfg(test)]
module` plus `parquet/tests/`):

| Area | Production | Test |
| --- | ---: | ---: |
| `file/properties.rs` | 126 | 49 |
| `column/writer/mod.rs` | 68 | 120 |
| `column/writer/encoder.rs` | 64 | 0 |
| `arrow/arrow_writer/byte_array.rs` | 56 | 0 |
| `encodings/encoding/dict_encoder.rs` | 35 | 0 |
| `arrow/arrow_writer/mod.rs` | 25 | 416 |
| `tests/arrow_writer/layout.rs` | 0 | 38 |
| **Library total** | **374** | **623** |
| `examples/advanced_racing_writer.rs` | 837 | n/a |
| `Cargo.toml` | 5 | n/a |

109 library lines were deleted or rewritten, almost all of them the
`should_dict_fallback` / `dict_fallback` / `flush_dict_page` bodies and the
layout-test expectations the #10777 fix invalidates.

The ratio is the headline result for the bakeoff: **374 production lines in the
library buy the capability; 837 lines in the example are what a consumer still
writes for itself.**

### What the example still has to implement itself

* **Property-set construction per candidate.** There is no "encode this chunk
  with these N alternatives" API. The example builds a whole
  `WriterProperties` per candidate set per row group, applying each leaf's
  chosen candidate to that leaf's `ColumnPath`. Because properties are
  file-shaped and candidates are leaf-shaped, a set is a cross product: K sets
  each carrying a full per-column map. Leaves with fewer candidates than the
  widest leaf repeat their last candidate into the surplus sets, and the
  duplicates are deduplicated when the winner is picked.
* **The probe costs K times a full set of column writers.** Not K encoders for
  the one leaf under test: `create_column_writers_with_properties` returns
  writers for *every* leaf, so racing one column still allocates and drives
  writers for all of them. Every candidate set also buffers its own complete
  set of compressed pages until the winner is chosen. For a wide table this is
  the dominant cost, and it is why the example settles leaves and only reopens
  them periodically. In the measurements below the raced write costs roughly
  2-3x the baseline's wall time on the string columns, where K = 3.
* **No mid-chunk decisions.** A candidate can only be judged once its chunk is
  closed and its compressed size is known. The writer exposes no hook to
  abandon a losing candidate part way through a row group, so every candidate
  encodes every row of every racing row group in full. All of the settle,
  reopen, near-tie and decode-cost policy in the example exists purely to
  amortize that, and none of it can react within a chunk.
* **Winner selection policy.** Smallest-wins, the near-tie window, the
  decode-cost preference (dictionary/plain over delta), the settle threshold
  and the reopen interval are all example policy. The library has no opinion.
* **Leaf-to-writer index mapping.** `compute_leaves` yields leaves per field;
  mapping those onto positions in the flat writer vector is the caller's job,
  and gets genuinely fiddly for nested schemas. The example's datasets are
  flat and it says so.

### What it gets for free

Everything the column writer already does keeps working, per candidate,
untouched:

* Mini-batching and the byte-budget sub-batching of writes.
* Data page and dictionary page size limits, and row count limits.
* Statistics (chunk and page level), and the column and offset indexes, which
  ride along on each candidate's `ColumnCloseResult` and are appended with the
  winner.
* Bloom filters.
* Compression, per column.
* The page store factory and, under the `encryption` feature, the file
  encryptor, which the removed free function silently bypassed.

A losing candidate's chunk is simply dropped, taking its buffered pages,
statistics and indexes with it; nothing has to be unwound.

## Proposed next variant, not implemented

apache/arrow-rs#10780 proposes a third variant, `DictionaryFallback::Adaptive`,
which would let the writer decide profitability without the caller supplying a
`worth_ratio`. It is still a draft upstream and is out of scope here, but it is
directly relevant to this bakeoff: it targets the same decision the example
currently makes by brute force, and a good enough `Adaptive` would remove the
dictionary-versus-plain leg of the race for most columns, leaving only the
delta candidates worth probing. Whether that is enough depends on how well a
purely local heuristic can do against an actual measurement, which is exactly
what the numbers below start to answer.

## Measurements

`cargo run --release --example advanced_racing_writer`, 2,000,000 rows per
dataset across 20 row groups of 100,000, SNAPPY, dictionary candidates using a
64 KiB dictionary page size limit. The baseline is a stock `ArrowWriter` with
default properties at the same row group size. Every file is read back and
compared for exact row equality.

| Dataset | Baseline | Raced | Smaller by | Baseline time | Raced time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low-cardinality strings | 2.49 MiB | 2.49 MiB | 0.0% | 58 ms | 192 ms |
| High-cardinality timestamps | 1.12 MiB | 0.03 MiB | 97.7% | 33 ms | 54 ms |
| f64 measurements | 19.13 MiB | 15.27 MiB | 20.2% | 95 ms | 34 ms |
| Shifting strings | 16.38 MiB | 12.67 MiB | 22.7% | 130 ms | 268 ms |

Encodings chosen, as recorded in the winning chunks:

| Dataset | Outcome |
| --- | --- |
| Low-cardinality strings | dictionary, all 20 row groups |
| High-cardinality timestamps | `DELTA_BINARY_PACKED`, all 20 row groups |
| f64 measurements | dictionary candidate, fell back to `PLAIN`, all 20 |
| Shifting strings | dictionary retained x10, dictionary fallen back to `PLAIN` x6, `DELTA_BYTE_ARRAY` x4 |

Reading these honestly:

* **Low-cardinality strings** is the null result. The default writer already
  picks the best encoding, and racing costs 3.3x the time to confirm it.
* **High-cardinality timestamps** is won by delta, not by anything to do with
  the dictionary. Ascending integers are the delta encoding's best case, and no
  dictionary policy competes with it.
* **f64 measurements** is where the race beats the default on *both* axes, and
  the reason is instructive: with the stock 1 MiB dictionary page size limit
  the baseline builds a ~100k entry dictionary per row group for a column of
  effectively unique doubles, never quite trips the limit, and pays for a
  dictionary that buys nothing. The raced dictionary candidate uses a 64 KiB
  limit, overflows almost immediately, falls back to `PLAIN`, and both wins on
  size and takes less time than the baseline.
* **Shifting strings** is the case the settle-and-reopen machinery exists for,
  and the outcome shows all three phases: the dictionary is retained while the
  column is 200 distinct labels, falls back once the column turns unique, and
  delta wins the row groups where prefix sharing in the generated ids pays.

### The `DictionaryFallback` knob, isolated

The race picks delta for the timestamp column, so it does not by itself measure
the knob. Writing that column with the dictionary candidate alone, varying only
`DictionaryFallback`:

| Codec | `WhenProfitable` | `OnPageSizeLimit` | Smaller by |
| --- | ---: | ---: | ---: |
| Uncompressed | 1.76 MiB | 4.22 MiB | 58.4% |
| SNAPPY | 1.12 MiB | 1.30 MiB | 13.7% |

Two caveats worth stating:

* **The general-purpose codec absorbs most of the win.** The dictionary's
  advantage is 58% before compression and 14% after, because SNAPPY recovers
  much of the same redundancy from the plain-encoded fallback. The
  profitability estimate is computed against *PLAIN* size and is therefore
  blind to the codec; it will systematically overvalue the dictionary on a
  compressed column.
* **`WhenProfitable` needs repetition to be clustered, not merely present.**
  This dataset delivers each distinct millisecond as a consecutive run of 12
  rows, so by the time the dictionary crosses the page size limit it has
  already absorbed many times its own size. A column with the same cardinality
  and the same repeat count, but scattered uniformly, saturates its dictionary
  before the repetition accumulates, is correctly judged unprofitable, and the
  knob then makes no difference at all. Both shapes were measured while
  building the example; only the clustered one shows a win.

## Validation

* `cargo fmt -p parquet`
* `cargo clippy -p parquet --all-targets --all-features -- -D warnings`, which
  is what `.github/workflows/arrow.yml` runs
* `cargo test -p parquet --lib` and `cargo test -p parquet --test arrow_writer`

93 lib tests fail in this environment both before and after these changes; all
93 are the tests that read from the `testing/` and `parquet-testing/`
submodules, which are not initialized here. The failing set is byte-identical
to the pre-change baseline.

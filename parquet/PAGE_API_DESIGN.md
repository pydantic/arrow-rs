# Opening the page grain

The parquet writer already lets a caller drive two of its three grains.
Row groups go into a file through `SerializedFileWriter::next_row_group`, and
column chunks go into a row group through `ArrowColumnWriter` producing an
`ArrowColumnChunk` that a caller splices in with `append_to_row_group`. The page
is the odd one out: between `ArrowColumnWriter::write` and a closed chunk, every
decision — where the page ends, what encoding it uses, whether the dictionary is
still worth keeping — is internal and unobservable.

This document describes `parquet::arrow::arrow_writer::page_grain`, which opens
that grain, and is honest about the two places where opening it got ugly.

## The shape that survived

```rust
let mut chunk = ColumnChunkBuilder::new(descr, props)?;

for leaf in compute_leaves(field, array)? {
    let mut cursor = chunk.cursor(&leaf);
    while !cursor.is_empty() {
        let pages = chunk.encode_page(
            &mut cursor,
            &[Candidate::Dictionary, Candidate::Pinned(Encoding::DELTA_BYTE_ARRAY)],
        )?;
        let winner = pick(&pages);                 // measure, compare, choose
        chunk.append(pages.into_iter().nth(winner).unwrap())?;
    }
}

let chunk: ArrowColumnChunk = chunk.close()?;      // existing type, existing splice
```

Five public types: `ColumnChunkBuilder`, `LeafCursor`, `Candidate`,
`EncodedPage`, `DictionaryView`.

### What changed from the original sketch, and why

The brief sketched a wider surface: a free-standing `DictionaryEncoder` object,
per-candidate `PageEncoder`s with `fill` / `write` / `seal`, and a `Span` value
passed between them.

```rust
let mut a = PageEncoder::dictionary(&dict)?;
let span = a.fill(&mut cursor)?;
let mut b = PageEncoder::pinned(Encoding::DELTA_BINARY_PACKED, &descr)?;
b.write(&span)?;
let (a, b) = (a.seal()?, b.seal()?);
chunk.append(if a.compressed_len() <= b.compressed_len() { a } else { b })?;
```

That shape was implemented far enough to see what it costs, and then collapsed
into one call. Three reasons, in order of how much they mattered.

**1. It is easy to hold wrong, and the ways to hold it wrong are silent.** The
sketch lets a caller call `fill` on two encoders (advancing the cursor twice and
racing two different spans), call `seal` before `fill`, pass a `Span` from one
cursor to an encoder built from another column's `descr`, or hold a
`PageEncoder` across pages. Every one of those produces either a panic deep in
the encoder or, worse, a valid-looking file with wrong statistics. Making the
builder cut the span itself removes all of them by construction: there is
exactly one pacer per page because `encode_page` designates it, and the span
never becomes a value the caller can misroute.

**2. The dictionary object and the accumulator lending both needed interior
mutability that the collapsed shape does not.** In the sketch, `&dict` is shared
across a `PageEncoder`'s lifetime while the caller also holds `&mut chunk`, so
`DictionaryEncoder` has to be an `Rc<RefCell<...>>` cell that lends its inner
`DictEncoder<T>` out and takes it back — and the same trick, separately, for the
bloom filter and geo accumulator, which the pacer must borrow from the chunk
while the chunk is also borrowed by `append`. Once `encode_page` owns the whole
page, both lends are ordinary `Option::take` / assignment inside one `&mut self`
method. That deleted a whole category of runtime borrow panics.

**3. The harness got shorter.** Race-then-settle in the sketch is roughly a
dozen lines per page plus the bookkeeping to keep the pacer and the pinned
candidates straight. In the collapsed shape it is: build a `Vec<Candidate>`,
call `encode_page`, `min_by_key` on `compressed_len`, `append`. The
`advanced_page_writer` example's *entire* policy engine — race-then-settle,
dictionary watching, cross-row-group learning and adapt-per-page — is under 200
lines of decision logic (see the accounting section).

What did **not** change is the four invariants, which are the actual contract:

| Invariant | How it is held |
| --- | --- |
| Decide before commit | `encode_page` returns sealed `EncodedPage`s. Nothing touches chunk state until `append`; dropping a page is the whole of rejecting it. |
| Split is an output of encoding | The pacer runs `GenericColumnWriter`'s own budget machinery and stops where `should_add_data_page` says. The caller has no way to name a row offset: `LeafCursor` has no seek. |
| Validity by construction | The builder derives dictionary-page position, `encodings`, `encoding_stats`, column index, offset index, boundary order, statistics truncation and chunk metadata from appended pages. An indices page after a non-dictionary page is rejected. |
| Dictionary is an explicit object | `ColumnChunkBuilder::dictionary()` reads entries / encoded bytes / values written at any time. A caller lands elsewhere by no longer offering `Candidate::Dictionary`. Dictionary-then-fallback is expressible; the reverse is not representable. |

The one thing the sketch had that the collapsed shape lost is the ability to run
candidates on different threads. It is recoverable — `encode_page` could take a
closure or return an iterator of unstarted candidates — but it buys nothing
today, because the parallel unit in this writer is the column chunk, and a
harness that wants more parallelism has whole chunks to spread.

## The hard problem: chunk-level accumulators

This is the question the whole design turns on, and it is where a wrongly drawn
builder boundary shows up.

Most of what a column chunk records is **page-local and additive**. Per-page
min/max, null counts, NaN counts, `variable_length_bytes`, level histograms, row
and value counts: each is computed for one page, and the chunk's version is the
sum or the extremum over the pages that were *actually written*. Racing K
candidates is trivially safe for all of these, because the K-1 losers are never
committed and so never contribute. `GenericColumnWriter::commit_data_page`
already computes every one of them this way, from a `PreparedDataPage` that
carries its own metrics.

Two things are not like that. A **bloom filter** and a **geospatial statistics
accumulator** are fed by *values*, one call per value, and are per chunk. They
cannot be reconstructed from page-local facts, and they live inside
`ColumnValueEncoderImpl::write_slice` — which means that in the naive design,
every candidate encoder that sees a span inserts that span's values, and racing
K candidates inserts every value K times.

### What actually goes wrong

It is worth being precise, because the obvious worry is not the real one.

*Duplicate inserts are not the bug.* `Sbbf::insert` is idempotent: inserting the
same value twice sets the same bits. K-fold duplication costs K times the hash
work and nothing else. A design that let every candidate feed its own copy of
the chunk's filter would produce a *correct* filter, slowly.

*The bug is the opposite: a value fed only by a page that was rejected.* If each
candidate carries its own accumulator and only the winner's is merged, then
every span whose pacer lost contributes nothing. The chunk's bloom filter then
claims a value is absent when it is present, which is the one direction a bloom
filter is not allowed to be wrong in, and readers will skip row groups that
contain matching rows. This is a silent data-loss-shaped bug, not a performance
bug, and it is entirely a consequence of where the accumulator lives.

The test `bloom_filter_survives_the_pacer_always_losing` pins exactly this: it
races two candidates and appends the *non-pacing* one on every single page, then
asserts every value probes positive. Under a per-candidate design that test
fails with an empty filter.

*Distinct count is worse still and is simply not offered.* It is neither
page-local nor unionable without a sketch data structure. The existing writer
only accepts a caller-supplied distinct count when nothing else has been
written; the page-grain builder never sets it, exactly as `ArrowWriter` never
sets it.

### The resolution

The builder owns the value-fed accumulators, and lends them to the pacing
candidate for exactly the span it consumes.

Concretely, in `ColumnChunkBuilder::pace`:

1. Take `ValueAccumulators` (bloom filter, its target FPP, geo accumulator) out
   of the builder and install them into the pacing candidate's encoder.
2. Run the pacer forward until its budget trips. Every value in the span is fed
   through `write_slice` exactly once, so each reaches the accumulators once.
3. Take the accumulators back out of the pacer's encoder and return them to the
   builder — **before the page is sealed, and regardless of whether the page is
   ever appended.**

Every other candidate is constructed by `ColumnValueEncoder::try_new_page_candidate`,
which strips the bloom filter, the geo accumulator and the dictionary. A pinned
candidate has no chunk-level state at all and cannot contribute to any.

At `close`, the accumulators are installed back into the *real* writer's encoder
just before `GenericColumnWriter::close`, so `flush_bloom_filter` and
`flush_geospatial_statistics` fold them into the chunk metadata by the ordinary
path. No metadata assembly is duplicated.

Three properties fall out:

* **Exactly once.** Exactly one candidate per span is the pacer, and the pacer is
  the only thing that advances the cursor. K candidates, one feed.
* **Survives rejection.** The commit point for value-fed state is *pace* time,
  not *append* time. This is the correct place, and the reason is a fact about
  parquet rather than about this API: the values in a span are in the chunk no
  matter which encoding wins. Which encoding a page uses is not information a
  bloom filter is allowed to depend on.
* **No duplicate work.** The K-1 losers do zero accumulator work.

### Where this is ugly, stated plainly

**A page encoder transiently owns chunk-level state.** For the duration of one
`pace` call, a throwaway candidate writer holds the column chunk's bloom filter.
That is a layering inversion: the object with the shortest lifetime holds the
object with the longest. It is done because the alternative is worse. Feeding
the accumulators from outside the encoder would mean re-deriving parquet values
from the arrow array — the `Int8 → Int32` conversions, the byte-array view
walks, the dictionary materialisation in `write_leaf` — a second time, in a
second implementation, for every span. That duplication would be both a
performance cost and a correctness hazard far larger than the one it removes.

The mitigations are that the lend is confined to one method with no `?` between
the take and the return (the inner loop is a closure whose result is checked
only *after* the accumulators are given back, so an encoding error cannot strand
them), and that `try_new_page_candidate` makes "candidate encoders own no chunk
state" a property of construction rather than of discipline.

**`ValueAccumulators` names two unrelated things.** Bloom filters and
geospatial statistics have nothing in common except *when* they need to be fed.
Grouping them is a statement about lifetime, not about meaning. It is the right
grouping for this API and a slightly odd one to read cold; the type's doc
comment says so.

**Geo statistics ride along untested end to end.** The geo accumulator travels
by exactly the same lend-and-return path as the bloom filter, in the same struct,
through the same two call sites. It is exercised by construction on every geo
column but the repository has no page-grain geo test, because building a WKB
fixture large enough to span pages is a meaningful chunk of work for a path that
is structurally identical to the one that *is* tested. This is a documented gap,
not a design difference.

## Keeping one implementation of page assembly

The requirement was to refactor internals into constructible pieces rather than
build a parallel page assembler. The refactor is:

```
add_data_page()                       assemble_data_page() -> PreparedDataPage
  = encode + compress + header   →      + 
  + index + metrics + write           commit_data_page(PreparedDataPage)

add_data_page() = commit_data_page(assemble_data_page())
```

`PreparedDataPage` carries the page's own `PageMetrics` and its untruncated
statistics, rather than leaving them on the writer to be read back. That single
change is what lets a page assembled by a throwaway candidate writer be
committed to a completely different writer. Candidates call `assemble`; the
builder's real writer calls `commit`. Page assembly exists once.

The same trick applies twice more:

* `write_dictionary_page_data(DictionaryPage)` takes a caller-supplied
  dictionary page, and the existing `write_dictionary_page` is now
  `write_dictionary_page_data(self.encoder.flush_dict_page()?)`.
* `new_with_encoder` takes a caller-supplied `ColumnValueEncoder`, and `new` is
  `new_with_encoder(..., E::try_new(...))`.

The evidence that this worked is the acceptance test: a single-candidate harness
write is **byte-identical** to `ArrowWriter` at the same properties, for the
generic path (`Int64`), the arrow byte-array path (`Utf8`), and a repeated
column (`List<Int32>`). Not metadata-equivalent — the same bytes.

Getting there found one real bug worth recording, because it is the kind of
thing a metadata-level test would have missed. The pacer originally inferred
"my budget tripped" from "I consumed less than I was offered". That is wrong
when the budget trips *exactly* at the end of the input, which for default
properties happens on essentially every page (the row limit is 20 000, the
mini-batch is 1 024, so pages end at 20 480 — exactly a window boundary). The
pacer read the signal as "still hungry", took another window, and produced pages
of 21 504 values instead of 20 480. Correct files, wrong page boundaries. The
fix is `GenericColumnWriter::page_boundary_reached`, which exposes the same
predicate the default path acts on, so the pacer *asks* rather than infers.

## Deferred flush, and not regressing the hot path

Two booleans were added to `GenericColumnWriter`:

* `defer_page_flush` — never flush a page or fall back from the dictionary on
  its own. Checked once per mini-batch, at the *end* of `write_mini_batch`,
  where a `should_add_data_page()` call already lived.
* `stop_at_page_boundary` — additionally *stop* the write loop at the boundary
  rather than merely declining to act on it. This is the pacer/non-pacer
  distinction: the pacer stops where its budget says, the candidates handed its
  span must encode all of it.

Both default to `false` and the default path takes one extra, perfectly
predictable, not-taken branch per mini-batch — that is, once per ~1 024 values,
not once per value. No per-value code changed.

`encoder.rs` has a comment warning that inserting code above the
`ColumnValueEncoder` trait perturbs downstream code placement enough to
regress the `string` and `string_and_binary_view` arrow-writer benchmarks by
5-9%. That note was taken seriously: the new `DynDictionary` trait,
`ValueAccumulators` and the `DictEncoder` impl are placed *above* the trait,
which is the position the comment warns about. The new trait methods
(`try_new_page_candidate`, `pin_encoding`, the four lend/return hooks) are all
appended at the *end* of the trait and the end of each impl, and none of them is
called on any default-path code path. To respect it, every
page-grain addition to that module — the `DynDictionary` trait and its
`DictEncoder` impl, `ValueAccumulators` — is placed at the *end* of the module,
below the hot encoder code, with a comment pointing back at the original note.
The six new trait methods are appended at the end of the trait and of each impl,
and none is reachable from the default write path. This is the right shape by
the module's own documented rule, but it has not been benchmarked here; it
should be measured before upstreaming rather than assumed.

Windowing is the other place performance was considered. The pacer cannot hand
the writer the whole remaining leaf, because `write_leaf` materialises type
conversions (`Int8 → Int32`, dictionary `take`) over whatever it is given, which
would make a chunk quadratic in its page count. It instead advances in windows
of `data_page_row_count_limit` levels rounded up to a whole number of
`write_batch_size` mini-batches. Rounding to a mini-batch multiple is not
cosmetic: it is what keeps window boundaries coincident with the mini-batch
boundaries the default path would have used, which is a precondition for the
byte-identity result.

## What it buys

`cargo run --release --features arrow --example advanced_page_writer`, 2 000 000
rows per dataset, 250 000 rows per row group, identical `WriterProperties` for
both writers. "Baseline" is plain `ArrowWriter`. Every file is read back and
checked for exact row equality.

| dataset | adaptive | baseline | bytes | time | landed on |
| --- | --- | --- | --- | --- | --- |
| low-cardinality strings | 1.5 MiB | 1.5 MiB | +0.1% | 1.16x | `RLE_DICTIONARY` x122 |
| high-cardinality int64 timestamps | 2.9 MiB | 17.3 MiB | **-83.2%** | 0.39x | `DELTA_BINARY_PACKED` x122 |
| f64 values | 15.6 MiB | 17.3 MiB | -10.2% | 0.43x | `PLAIN` x114, `RLE_DICTIONARY` x8 |
| strings changing character mid-file | 17.0 MiB | 27.6 MiB | **-38.4%** | 1.35x | `RLE_DICTIONARY` x64, `DELTA_BYTE_ARRAY` x58 |

Read this as four different things going right rather than as a benchmark:

* The low-cardinality column is the control. The default writer's choice is
  already correct, the adaptive writer races four pages, agrees, and settles —
  and the +0.1% is the residue of racing before settling. Costing 0.1% to
  confirm a decision is the price of the mechanism.
* The timestamp column is where a default is simply wrong for the data: the
  dictionary overflows, the writer falls back to `PLAIN`, and delta would have
  been 6x smaller. It is also *faster*, because racing four pages is much
  cheaper than plain-encoding 15 MiB it did not need to write.
* The f64 column never races at all — it is the adapt-per-page column, deciding
  from the previous sealed page's compression ratio — and still finds 10%.
* The last column is the one that cannot be expressed at all without the page
  grain: it needs two different encodings *within one file*, chosen at the point
  the data changes, and it finds the switch on its own.

## Where the boundary may still be drawn wrong

Three candidates, in decreasing order of confidence.

**The pacer is privileged, and its privilege is invisible.** `candidates[0]`
decides the span *and* carries the accumulators. Two consequences: a race
between a dictionary pacer and a plain pacer would cut different spans, so
"which candidate paces" is itself a policy decision the API currently makes by
positional convention; and the accumulator lending is a property of position in
a slice, which is not the most legible way to express "this one is special". A
`Candidate::pacing(...)` constructor, or a separate `pacer` argument, would say
it out loud. This was left alone because the positional rule is one sentence and
the alternative adds a type.

**`encode_page` conflates "how do I cut this page" with "what do I compare".**
For adapt-per-page with a single candidate these are the same question. For a
race they are not, and a harness that wants to cut by the dictionary's budget
but compare four encodings must accept that the dictionary is candidate zero.
Splitting into `cut_page` and `encode_span` would separate them, at the cost of
handing the caller a `Span` value again — which is the thing the collapse
removed. The current guess is that the conflation is worth it.

**The dictionary is dropped at `append`, not at `encode_page`.** Appending a
non-dictionary page sets `dictionary_closed` and drops the dictionary, so the
*next* `encode_page` that offers `Candidate::Dictionary` fails with an error
rather than being unrepresentable. Type-level unrepresentability would need the
builder to change type on fallback (`ColumnChunkBuilder` → `FallenBackBuilder`),
which is a real option and was not taken because it makes the common loop
generic over a state parameter. As it stands, invariant 4's "unrepresentable" is
enforced dynamically for this one transition, and by construction for
everything else. That is a gap between the claim and the code, and it is here
rather than hidden.

One more, smaller: `ColumnChunkBuilder::new` dispatches on **physical** type,
so a `Dictionary(_, FixedSizeBinary)` arrow column goes through the generic
writer (which materialises the dictionary) where `ArrowColumnWriterFactory`
would send it to the byte-array encoder. Same values, same output, one fewer
specialisation — but it means page-grain and `ArrowWriter` byte-identity is not
claimed for that one arrow type.

## Accounting

Measured with `git diff --stat` against the branch point (`2567a32`).

| | files | added | removed |
| --- | --- | --- | --- |
| **Library total** | 5 | 2 071 | 79 |
| — new `page_grain` module, excluding its tests | 1 | 1 020 | 0 |
| — `page_grain` tests | 1 | 470 | 0 |
| — refactor of existing files | 4 | 581 | 79 |
| **Example (harness)** | 1 | 691 | 0 |
| Cargo wiring | 1 | 5 | 0 |
| This document | 1 | 402 | 0 |

Refactor of existing files, broken down: `column/writer/mod.rs` +309/-57 (the
assemble/commit split, deferred flush, the new accessors),
`column/writer/encoder.rs` +164 (`DynDictionary`, `ValueAccumulators`, six trait
methods with defaults, the `DictEncoder` impls),
`arrow_writer/byte_array.rs` +74/-1 (the same hooks for the byte-array encoder),
`arrow_writer/mod.rs` +34/-21 (`write_levels` extracted so page-grain and
`ArrowColumnWriter` share one write path), `dict_encoder.rs` +9
(`clear_pending`).

The ratio worth reading is that ~1 000 of the ~2 100 library lines are one new
module and none of the existing hot code was rewritten — the largest single
change to an existing file is a mechanical split of one function into two.

Of the example's 691 lines, roughly 250 are deterministic data generation,
baseline comparison, round-trip verification and printing. The actual encoding
policy — race-then-settle, dictionary watching with honest cost accounting,
cross-row-group learning, adapt-per-page, and the winner-picking rule — is about
190 lines. That is the number to read as "what a harness costs".

### Public items added

All under `parquet::arrow::arrow_writer::page_grain`:

- `ColumnChunkBuilder`, with `new`, `new_with_page_store`, `descr`,
  `pages_appended`, `dictionary`, `cursor`, `encode_page`, `append`, `close`
- `LeafCursor`, with `is_empty`, `levels_remaining`, `values_remaining`
- `Candidate` (`Dictionary`, `Pinned(Encoding)`)
- `EncodedPage`, with `encoding`, `num_values`, `num_rows`, `null_count`,
  `compressed_len`, `uncompressed_len`, `min_bytes`, `max_bytes`,
  `variable_length_bytes`, `is_dictionary_indices`
- `DictionaryView`, with `entries`, `encoded_bytes`, `values_written`

Crate-internal additions: `PreparedDataPage`, `DynDictionary`,
`ValueAccumulators`, and on `GenericColumnWriter` the methods
`new_with_encoder`, `assemble_data_page`, `commit_data_page`,
`write_dictionary_page_data`, `write_batch_inner`, `set_defer_page_flush`,
`page_boundary_reached`, `encoder_mut`. On `ColumnValueEncoder`:
`try_new_page_candidate`, `pin_encoding`, `take_dictionary`,
`install_dictionary`, `take_value_accumulators`, `install_value_accumulators` —
all with defaults, so the public trait is not a breaking change.

### Harder than the sketch implied

**The dictionary is two dictionaries.** `encodings::encoding::DictEncoder<T>`
and a private `DictEncoder` inside `arrow_writer/byte_array.rs` are unrelated
types with different APIs, and the byte-array one is the one that matters for
the most interesting column type. "Cross-page state is an object" therefore
required a type-erasing trait (`DynDictionary`) with a downcast on reinstall,
not the `DictEncoder` the sketch names.

**"Same rows: pages are comparable" needed a second flag.** The sketch's `fill`
and `write` look symmetric. They are not: `fill` must stop at its budget and
`write` must ignore its own budget entirely and consume everything. Running both
in the same deferred mode produced candidates that silently encoded *fewer* rows
than the span, which showed up as a `debug_assert` and would otherwise have been
a wrong file. That is the `stop_at_page_boundary` flag.

**Byte-identity is a much sharper test than it sounds.** Two attempts produced
files with identical column-chunk sizes, identical offsets and identical
statistics that were still not byte-identical — once because the harness was not
writing `ArrowWriter`'s `ARROW:schema` footer key, and once because of the
off-by-one-window pacing bug above. Both would have passed a metadata-level
parity test. It is the acceptance test for a reason.

**The comparison the API most obviously invites is biased.** `EncodedPage::compressed_len`
is the data page, and the whole point of a dictionary-indices page is that its
bytes are somewhere else. Picking `min_by_key(compressed_len)` therefore chooses
the dictionary essentially always, because an indices page over 20 000 distinct
values is a few KiB and the half-megabyte of dictionary entries it just created
is invisible until close. The first run of the example did exactly this: on the
dataset that changes character halfway, it rode the dictionary through 122
consecutive pages, produced a file 6.9% *larger* than the baseline, and took
6.5x as long doing it.

The fix is in the harness, not the library — read
`ColumnChunkBuilder::dictionary().encoded_bytes()` either side of `encode_page`
and charge the growth to the indices candidate — and with it the same dataset
goes to 38.4% *smaller* than the baseline, switching from dictionary to
`DELTA_BYTE_ARRAY` at the point the data changes. But the fact that the obvious
call is the wrong call is a genuine sharp edge in this API, and arguably an
argument that `EncodedPage` should expose an `amortized_cost` that the builder
computes, rather than leaving every harness to rediscover this. It was left to
the harness because the right amortisation depends on how many more pages the
dictionary will serve, which the builder cannot know and the harness's policy
implicitly does.

**The page store never sees a candidate.** Candidates assemble pages and never
write them, so their page writer must be unreachable rather than merely unused;
it is a `NullPageWriter` that errors. Reaching it is an internal bug, and saying
so in code was clearer than handing candidates a real buffer they would fill and
throw away.

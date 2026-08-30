# Opening the page grain

The parquet writer lets a caller drive two of its three grains. Row groups go
into a file through `SerializedFileWriter::next_row_group`, and column chunks go
into a row group through an `ArrowColumnWriter` producing an `ArrowColumnChunk`
that `append_to_row_group` splices in. The page is the odd one out: between
`ArrowColumnWriter::write` and a closed chunk, every decision — where the page
ends, what encoding it uses, whether the dictionary is still worth keeping — is
internal and unobservable.

This proposes `parquet::arrow::arrow_writer::page_grain`, which opens that
grain, plus two accessors on `ArrowRowGroupWriterFactory`. Together they are
enough for a caller to write a parquet file whose encodings are *measured*
rather than configured, at a cost that is smaller than the default writer's on
every dataset we tried, and slower than it only where it is also smaller.

The measured result is in `BAKEOFF.md`: on ClickBench and TPC-H files, 4.5% to
24.7% smaller than upstream `main` at identical properties, faster than the
default writer, at equal or lower writer peak memory. The reference caller is
`examples/adaptive_writer/`: 666 lines of harness, of which the 318 in its
policy half are what a different writer would rewrite.

## The API

Everything below is under
`parquet::arrow::arrow_writer::page_grain`, except the last two items.

```rust
ColumnChunkBuilder::new(descr, props)                           -> Result<Self>
ColumnChunkBuilder::new_with_page_store(descr, props, factory, column_index)
    .dictionary()                                               -> Option<DictionaryView>
    .cursor(&ArrowLeafColumn)                                   -> LeafCursor
    .encode_page(&mut LeafCursor, Candidate)                    -> Result<EncodedPage>
    .encode_page_alternatives(&mut LeafCursor, Candidate, &[Candidate])
                                                                -> Result<Vec<EncodedPage>>
    .append(EncodedPage)                                        -> Result<()>
    .close()                                                    -> Result<ArrowColumnChunk>

enum Candidate { Dictionary, Encoding(Encoding) }

LeafCursor::is_empty()                                          -> bool

DictionaryView::entries()                                       -> usize
DictionaryView::encoded_bytes()                                 -> usize

EncodedPage::encoding()                                         -> Encoding
EncodedPage::num_values()                                       -> u32
EncodedPage::num_rows()                                         -> u32
EncodedPage::compressed_len()                                   -> usize
EncodedPage::uncompressed_len()                                 -> usize
EncodedPage::dictionary_growth()                                -> usize

// on ArrowRowGroupWriterFactory
ArrowRowGroupWriterFactory::page_store_factory()                -> &Arc<dyn PageStoreFactory>
ArrowRowGroupWriterFactory::create_selected_column_writers(row_group, props, select)
                                            -> Result<Vec<Option<ArrowColumnWriter>>>
```

That is 22 items in `page_grain` (5 types, 17 methods and variants) and 2
elsewhere. The whole of the happy path:

```rust
let mut chunk = ColumnChunkBuilder::new(descr, props)?;

for leaf in compute_leaves(field, array)? {
    let mut cursor = chunk.cursor(&leaf);
    while !cursor.is_empty() {
        let pages = chunk.encode_page_alternatives(
            &mut cursor,
            Candidate::Dictionary,
            &[Candidate::Encoding(Encoding::PLAIN)],
        )?;
        let best = pages
            .into_iter()
            .min_by_key(|p| p.compressed_len() + p.dictionary_growth())
            .expect("at least one page");
        chunk.append(best)?;
    }
}

let chunk: ArrowColumnChunk = chunk.close()?;
```

`encode_page` is the same loop with one candidate and no `Vec`, which is what a
caller that has stopped comparing should be writing.

### Reading the signature

Three things are deliberate.

**The candidate that decides the page boundary is an argument, not a position.**
An earlier revision took `&[Candidate]` and made `candidates[0]` special: it
paced the page *and* carried the chunk's value-fed accumulators. That is a
protocol to memorise rather than a signature, and it is invisible at the call
site. Naming it costs one argument and removes the rule.

**Alternatives cannot be `Candidate::Dictionary`.** A chunk has one dictionary,
and an encoding that may be thrown away must not intern into it. The API rejects
it rather than documenting it.

**The comparison the API invites is the correct one.** `compressed_len` alone is
a biased comparison: a dictionary-indices page is small precisely because its
bytes went into the dictionary page, which is written once at close where no
per-page comparison can see it. The first harness written against this API rode
the dictionary through 122 consecutive pages and produced a file 6.9% *larger*
than the baseline. `dictionary_growth` is the builder reporting what that page
actually added to the dictionary, so the honest comparison is one line. It
reports a measurement, not a verdict: how to amortise it over the pages the
dictionary will still serve is a policy question, and stays with the caller.

### What is deliberately absent

The library reports facts and enforces validity. It does not carry vocabulary
for any particular policy, and a reviewer should not be able to infer ours from
it. Removed while preparing this proposal, each because it encoded a decision
rather than a measurement:

* `DictionaryView::values_written` — a builder-accumulated count of values a
  caller had sent through the dictionary, whose only use is the ratio
  `entries / values_written`. That ratio is one heuristic for abandoning a
  dictionary, and a caller that wants it sums `num_values()` over the pages it
  chose to encode that way.
* `EncodedPage::is_dictionary_indices` — derivable from `encoding()`.
* `LeafCursor::levels_remaining` — no consumer.
* `EncodedPage::amortized_cost`, considered and not added, for the reason above.

There is also no `Span` value, no free-standing `DictionaryEncoder`, and no
per-candidate `PageEncoder` with `fill` / `write` / `seal`. That was the
original sketch; it was implemented far enough to price and then collapsed into
`encode_page`. It let a caller `fill` two encoders (advancing the cursor twice
and comparing two different spans), `seal` before `fill`, pass a `Span` built
from one column to an encoder built from another, or hold an encoder across
pages. Each of those produced either a panic deep in the encoder or a
valid-looking file with wrong statistics. Cutting the page inside the builder
removes all of them by construction, and it removed two `Rc<RefCell<..>>` cells
that the sketch needed to lend the dictionary and the accumulators out. The one
thing it lost is running candidates on separate threads, which buys nothing
today because the parallel unit in this writer is the column chunk.

## The invariants

These are the contract, and they are the reason this is an API rather than a
recipe for corrupting files.

| Invariant | How it is held |
| --- | --- |
| Decide before commit | Encoding returns sealed `EncodedPage`s. No chunk state is touched until `append`, so rejecting a page is dropping it. |
| The split point is an output | The page ends where the candidate's own budget ends it. `LeafCursor` has no seek and no constructor, and the only thing that moves it is encoding. Records are never split. |
| Validity by construction | The builder derives the dictionary page's position, the `encodings` set, page encoding statistics, the column index, the offset index, boundary order, statistics truncation and the chunk metadata from the pages actually appended. None can be supplied. |
| The dictionary is one object | `dictionary()` reads it; `Candidate::Dictionary` uses it. A caller lands elsewhere by no longer offering it. Dictionary-then-fallback is expressible; the reverse is rejected. |

The fourth is enforced dynamically for exactly one transition: appending a
non-dictionary page drops the dictionary, so a later `Candidate::Dictionary`
fails with an error rather than failing to compile. Type-level
unrepresentability would need the builder to change type on fallback
(`ColumnChunkBuilder` -> `FallenBackBuilder`), which makes the common loop
generic over a state parameter. That trade was declined; the gap is here rather
than hidden.

## What the caller owns

Everything that is a decision. Concretely, in the reference harness:

* which encodings are worth trying for a leaf, and whether a leaf is worth
  deciding for at all;
* what a page costs, including how to charge `dictionary_growth`;
* how near a tie has to be before decode cost breaks it;
* when enough pages agree to stop comparing (the harness settles after two);
* when to look again (every 4 row groups);
* when a dictionary has stopped paying (entries above a quarter of the values
  sent through it, after 50 000 values);
* and the routing rule that makes the whole thing affordable: a leaf uses the
  page grain while it is still deciding, and an ordinary `ArrowColumnWriter`
  once it has settled.

The last one is the entire hybrid, and it is a caller-side rule, not a library
feature. It works because both paths produce an `ArrowColumnChunk` and both can
be handed to one `SerializedRowGroupWriter`. It is what makes the approach cost
0.5x to 0.9x the default writer's time on real files instead of paying page-
grain overhead on 105 columns that decided everything in row group one.

### Harness accounting

`examples/adaptive_writer/harness.rs`, 666 lines, split by its own section
markers (the same count `BAKEOFF.md` reports, taken from those markers):

| | lines |
| --- | ---: |
| Policy — the half a user rewrites | 318 |
| Plumbing — the half a user copies | 279 |
| Licence, module docs, imports | 69 |
| `main.rs` — dataset, comparison, verification | 195 |

The plumbing is 279 lines because it does three things: open a row group by
routing every leaf and creating the two kinds of writer, feed each leaf's values
to whichever it is on, and close both kinds and append them in leaf order. A
caller that only ever wants the page grain (no routing) needs roughly the
`PageGrainChunk` half of it.

## Load-bearing complexity, stated plainly

Four places where this cost more than it looks, all of them in the library.

**The bloom filter and geospatial accumulators are lent, not copied.** Both are
fed by *values*, one call per value, and belong to the chunk. They live inside
`ColumnValueEncoderImpl::write_slice`, so in a naive design every candidate
encoder inserts every value and racing K candidates inserts each value K times.
Duplicate inserts are not the bug: `Sbbf::insert` is idempotent, so that design
produces a correct filter slowly. The bug is the opposite. If each candidate
carried its own accumulator and only the winner's were merged, every value on
every page whose candidate lost would be missing from the filter, and readers
would skip row groups that contain matching rows. That is silent data loss, and
it is a consequence of where the accumulator lives.

The resolution: the builder owns them, and lends them to the one candidate that
decides the page boundary, for exactly the values it consumes — before the page
is sealed, and regardless of whether that page is ever appended. Committing
value-fed state at encode time rather than append time is correct because of a
fact about parquet rather than about this API: the values are in the chunk
whichever encoding wins, and a bloom filter is not allowed to depend on which
one did. Every other candidate is built by the ordinary `try_new` and then has
its own accumulators taken away and dropped in `candidate_writer`, which is the
single site where a candidate is built.

The ugliness is real and worth naming: for the duration of one call, a throwaway
page encoder owns the column chunk's bloom filter, which is the shortest-lived
object holding the longest-lived one. The alternative is feeding the
accumulators from outside the encoder, which means re-deriving parquet values
from the arrow array (the `Int8 -> Int32` conversions, the byte-array view
walks, the dictionary materialisation in `write_leaf`) a second time, in a
second implementation, for every page. The mitigations are that the lend is
confined to one method with no `?` between the take and the return, and that
`ValueAccumulators` groups the two by lifetime, which its doc comment says out
loud because it is otherwise an odd pairing. `bloom_filter_survives_the_boundary
_page_always_losing` pins it: it encodes two candidates and appends the
alternative on every single page, then asserts every value probes positive.

**Truncated and untruncated statistics are different numbers, and both travel
with the page.** `PreparedDataPage` carries its own `PageMetrics`, its
*untruncated* `index_statistics` for the column index, its truncated header
statistics inside the compressed page, its `variable_length_bytes` and its
min/max. That is what lets a page assembled by a throwaway candidate writer be
committed to a completely different writer. Leaving any of it on the writer to
be read back afterwards would have silently mixed one candidate's statistics
into another candidate's page.

**Eight arms, in three places.** `PreparedDataPage<T>` is generic over the
physical type, so `EncodedPage` erases it into an eight-variant enum and the
builder checks the variant matches its column before committing. The nine
concrete writer types are enumerated once, in a `dispatch_writer!` macro; the
assemble and commit paths enumerate the eight page variants. This is
mechanical, not subtle, but it is real lines and it is where a new physical type
would have to be added.

**Deferred flush is one enum, and the hot path pays one branch.**
`GenericColumnWriter` gained a `PageBoundaryAction` with three variants: `Flush`
(the default, the only value the ordinary path ever holds), `Stop` (the
candidate that decides the boundary) and `Continue` (an alternative, which must
encode the whole page it was given and ignore its own budget). An earlier
revision spelled this as two booleans where the second was meaningless without
the first; the enum makes the fourth state unrepresentable. It is checked once
per mini-batch, at the end of `write_mini_batch` where a `should_add_data_page`
call already lived, so the default path takes one predictable not-taken branch
per ~1024 values and no per-value code changed.

`encoder.rs` carries a comment warning that inserting code *above* the
`ColumnValueEncoder` trait regresses the `string` and `string_and_binary_view`
arrow-writer benchmarks by 5-9% through code placement alone. Every addition to
that module is therefore placed at the end, below the hot encoder code, with a
comment pointing back at that note. The five new trait methods all have
defaults and none is reachable from the default write path. This is the right
shape by the module's own rule, but it has not been benchmarked here and should
be measured before merging.

## One implementation of page assembly

The requirement was to make the existing writer constructible in pieces, not to
build a second page assembler. The refactor is one function split in two:

```
add_data_page()                      assemble_data_page() -> PreparedDataPage
  = encode + compress + header   ->     +
  + index + metrics + write           commit_data_page(PreparedDataPage)
```

with `add_data_page() = commit_data_page(assemble_data_page())`. Candidates
assemble; the builder's real writer commits. The same trick appears twice more:
`write_dictionary_page_data(DictionaryPage)` takes a caller-supplied dictionary
page and the existing `write_dictionary_page` calls it, and `new_with_encoder`
takes a caller-supplied encoder and `new` calls it.

The evidence that this worked is the acceptance test: a single-candidate
page-grain write is **byte-identical** to `ArrowWriter` at the same properties,
for the generic path (`Int64`), the arrow byte-array path (`Utf8`) and a
repeated column (`List<Int32>`). Not metadata-equivalent: the same bytes.

Getting there found a bug worth recording, because a metadata-level test would
have missed it. The boundary-deciding candidate originally inferred "my budget
tripped" from "I consumed less than I was offered". That is wrong when the
budget trips exactly at the end of the input, which under default properties
happens on essentially every page (row limit 20 000, mini-batch 1 024, so pages
end at 20 480 — exactly a window boundary). It read that as "still hungry" and
produced 21 504 value pages. Correct files, wrong page boundaries. The fix is
`page_boundary_reached`, which exposes the same predicate the default path acts
on, so the writer is *asked* rather than inferred from.

Windowing is the other place performance was considered. A candidate cannot be
handed the whole remaining leaf, because `write_leaf` materialises type
conversions over whatever it is given, which would make a chunk quadratic in its
page count. It advances in windows of `data_page_row_count_limit` levels rounded
up to a whole number of `write_batch_size` mini-batches. The rounding is not
cosmetic: it keeps window boundaries coincident with the mini-batch boundaries
the default path would have used, which is a precondition for byte identity.

## The known limitation: page cadence

**A page never spans two `ArrowLeafColumn`s.** A `LeafCursor` covers one leaf,
and encoding seals whatever the candidate has buffered when that cursor runs
out. A caller feeding 8192-row record batches therefore gets 8192-row pages even
where `data_page_row_count_limit` and the byte budget would have allowed 20 000,
and pays for the extra pages three times over: compressed payload from smaller
compression windows, page headers, and column and offset index entries that
carry per-page min/max.

It is measured, on ClickBench `hits_0`: the pure page-grain arm writes 12 922
pages against the hybrid's 8 212 and the baseline's 4 492, and that accounts for
all but 112 bytes of the 1 070 057 byte gap between the two arms — 77% extra
compressed payload, 9% page headers, 14% index. Re-reading the same input into
whole-row-group batches shrinks the gap to 53 192 bytes and reverses it on TPC-H
`orders`, which places the cause in the cadence rather than in the path.

There are two mitigations that need no API change, and the reference harness
uses the first: route a leaf off the page grain once it has settled, which
restores the ordinary cadence for the majority of leaves and the majority of the
file; or feed the builder larger leaves, at the memory cost of holding them.

Removing the limitation outright is a genuine design change, and it was
re-examined for this proposal rather than assumed. It requires the builder to
keep the boundary-deciding candidate's writer alive across `encode_page` calls
and across leaves, returning "no page yet" when a cursor runs out before the
budget trips. Two consequences make it more than plumbing. The span an
alternative must encode becomes multi-leaf, so the builder has to retain the
level data of every leaf the page spans and re-slice across them, which changes
the memory story from "one page" to "one page's worth of input arrays, held by
the library". And the cursor stops being the caller's unit of progress, since
the caller can no longer tell whether a returned leaf has been committed. The
last point of the gap is not worth destabilising the API for.

## Seams on `ArrowRowGroupWriterFactory`

Two were fixed for this proposal, and both were a missing half of an existing
pair.

**`page_store_factory()` (fixed).** `with_page_store_factory` had a setter and
no getter, so a caller assembling part of a row group by another route could not
use the file writer's page store for it. Its chunks were unbounded on the heap
while the factory's spilled, and the memory bound the writer offers did not
cover the whole row group. One accessor.

**`create_selected_column_writers` (fixed).** The all-or-nothing
`create_column_writers_with_properties` forced a hybrid caller to allocate a
writer per leaf and drop the ones it wrote itself. With an in-memory page store
that is a wasted allocation; with a spilling one it is a temp file per unused
leaf, per row group, at ClickBench width. The new call takes a predicate over
leaf index and returns `Vec<Option<ArrowColumnWriter>>` — one entry per leaf, in
leaf order, `None` where the caller declined, nothing allocated for those. The
`Option` shape is what such a caller already builds by hand, because it has to
close and move out one writer at a time.
`create_column_writers_with_properties` is now that call with every column
selected.

**The two paths abandon a dictionary by different mechanisms (re-affirmed).** On
the ordinary path a dictionary is abandoned by `DictionaryFallback`, which is a
property; on the page grain the caller simply stops offering
`Candidate::Dictionary`. So a leaf changes mechanism when it changes path. This
looks like an asymmetry to fix, and it is not: the page grain exists precisely
so that this decision is the caller's, per page, with the numbers in front of
it. Adding an automatic fallback to the page-grain path would be adding a policy
to the mechanism, and a caller that wants `WhenProfitable` semantics there can
implement exactly that rule in nine lines against `dictionary()`. What the
harness must do is keep the two consistent, which is a harness job.

**Physical-type dispatch in `ColumnChunkBuilder::new` (re-affirmed).** The
builder chooses its encoder by physical type, so a
`Dictionary(_, FixedSizeBinary)` arrow column goes through the generic writer
(which materialises the dictionary) where `ArrowColumnWriterFactory` would send
it to the byte-array encoder. Same values, same output, one fewer
specialisation, but byte identity with `ArrowWriter` is not claimed for that one
arrow type.

## Accounting

Library cost of the page grain, measured from the merge of the Option B ports
(which are unrelated and independently proposed upstream as
apache/arrow-rs#10775 and #10777):

| | added | removed |
| --- | ---: | ---: |
| `arrow_writer/page_grain.rs` — production | 1 011 | 0 |
| `arrow_writer/page_grain.rs` — tests | 529 | 0 |
| `column/writer/mod.rs` — assemble/commit split, `PageBoundaryAction`, accessors | 308 | 58 |
| `column/writer/encoder.rs` — `DynDictionary`, `ValueAccumulators`, five defaulted trait methods | 133 | 0 |
| `arrow_writer/mod.rs` — `write_levels` extraction, the two factory seams, their test | 179 | 33 |
| `arrow_writer/byte_array.rs` — the same encoder hooks | 56 | 1 |
| **Total** | **2 216** | **92** |
| **Total, excluding tests** | **1 687** | **92** |

Two thirds of the production lines are one new self-contained module, and the
largest change to an existing file is a mechanical split of one function into
two. No hot-path code was rewritten.

Public items: 22 in `page_grain`, 2 on `ArrowRowGroupWriterFactory`.
Crate-internal additions: `PreparedDataPage`, `PageBoundaryAction`,
`DynDictionary`, `ValueAccumulators`; on `GenericColumnWriter` the methods
`new_with_encoder`, `assemble_data_page`, `commit_data_page`,
`write_dictionary_page_data`, `write_batch_inner`, `set_page_boundary_action`,
`page_boundary_reached`, `encoder_mut`; on `ColumnValueEncoder` the methods
`pin_encoding`, `take_dictionary`, `install_dictionary`,
`take_value_accumulators`, `install_value_accumulators`, all with defaults, so
this is not a breaking change to the public trait.

Revision history of the surface: the first working version exposed 30 public
items in `page_grain`; the same-numbers simplification pass took it to 23 by
dropping accessors with no consumer; this proposal's pass took it to 22 by
removing three items that encoded a policy or were derivable, adding
`dictionary_growth`, and splitting `encode_page` in two. Every byte count in
`BAKEOFF.md` is unchanged for the hybrid arm across all of it.

## Harder than it looks, for the record

**The dictionary is two dictionaries.** `encodings::encoding::DictEncoder<T>`
and a private `DictEncoder` inside `arrow_writer/byte_array.rs` are unrelated
types with different APIs, and the byte-array one is the one that matters for
the most interesting column type. "The dictionary is an object" therefore needed
a type-erasing trait with a downcast on reinstall.

**"Same rows, comparable pages" needed a second mode.** An alternative must
ignore its own page budget and encode everything it was given. Running it in the
same deferred mode as the boundary-deciding candidate produced candidates that
silently encoded *fewer* rows than the page, which surfaced as a `debug_assert`
and would otherwise have been a wrong file. That is `Continue` versus `Stop`.

**Byte identity is a sharper test than it sounds.** Two attempts produced files
with identical column-chunk sizes, identical offsets and identical statistics
that were still not byte-identical: once because the harness was not writing
`ArrowWriter`'s `ARROW:schema` footer key, and once because of the off-by-one
window bug above. Both would have passed a metadata-level parity test.

**The page store must be unreachable for a candidate, not merely unused.**
Candidates assemble pages and never write them, so their page writer is a
`NullPageWriter` that errors. Reaching it is an internal bug, and saying so in
code was clearer than handing candidates a real buffer to fill and throw away.

**Geo statistics ride along untested end to end.** The geo accumulator travels
by exactly the same lend-and-return path as the bloom filter, in the same
struct, through the same two call sites, and is exercised by construction on
every geo column. There is no page-grain geo test, because a WKB fixture large
enough to span pages is real work for a path that is structurally identical to
the one that is tested. A documented gap, not a design difference.

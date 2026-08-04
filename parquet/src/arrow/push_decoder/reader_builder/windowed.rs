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

//! Batch-granular decoding of a single row group.
//!
//! The default row-group decoder ([`super::RowGroupReaderBuilder`]) asks for
//! every projected byte of a row group before it will hand back a reader.
//! [`WindowedRowGroup`] asks for the bytes one *window* of rows at a time
//! while keeping all of the row group's decode state alive between windows.
//!
//! Two things make that possible:
//!
//! * [`PageStore`] — pages can be added to a column chunk *after* the reader
//!   that reads it was built, so a reader can outlive the bytes it started
//!   with.
//! * [`ParquetRecordBatchReader::into_array_reader`] — the stateful
//!   [`ArrayReader`] (column readers, decoded dictionaries) survives being
//!   driven by a succession of short [`ReadPlan`]s.
//!
//! # Two modes
//!
//! **Unfiltered.** The final selection is known up front, so one [`ReadPlan`]
//! and one [`ParquetRecordBatchReader`] are built for the whole row group and
//! kept. Batches come out of that single reader; the only thing windowing adds
//! is a check, before each pull, that the pages the next window needs are
//! resident. Batching is therefore *bit-identical* to a non-windowed read,
//! whatever row-selection strategy is chosen.
//!
//! **Filtered.** The final selection is not known until the predicates have
//! run, so filtering and output are interleaved per window: each window's rows
//! are pushed through the predicate chain, the survivors are appended to a
//! ready queue, and output batches are emitted from that queue. Predicate
//! columns for window *k+1* can therefore be in flight while window *k*'s
//! output decodes. One [`ArrayReader`] per predicate plus one for the output
//! is built and retained for the whole row group.
//!
//! [`ArrayReader`]: crate::arrow::array_reader::ArrayReader

use super::{RowBudget, override_selector_strategy_if_needed};
use crate::arrow::ProjectionMask;
use crate::arrow::array_reader::{ArrayReader, ArrayReaderBuilder, CacheOptions, RowGroupCache};
use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use crate::arrow::arrow_reader::{
    ParquetRecordBatchReader, ReadPlanBuilder, RowFilter, RowSelection, RowSelectionPolicy,
    RowSelector,
};
use crate::arrow::in_memory_row_group::{ColumnChunkData, InMemoryRowGroup};
use crate::arrow::push_decoder::page_store::PageStore;
use crate::arrow::push_decoder::scan_plan::{ScanPlan, plan_scan_ranges};
use crate::arrow::schema::ParquetField;
use crate::errors::ParquetError;
use crate::file::metadata::ParquetMetaData;
use crate::file::page_index::offset_index::OffsetIndexMetaData;
use crate::util::push_buffers::PushBuffers;
use arrow_array::{Array, RecordBatch};
use std::ops::Range;
use std::sync::{Arc, RwLock};

/// What one step of a [`WindowedRowGroup`] produced.
#[derive(Debug)]
pub(crate) enum WindowedResult {
    /// Bytes needed before the *next batch* can be decoded.
    NeedsData(Vec<Range<u64>>),
    /// The next output batch.
    Batch(RecordBatch),
    /// This row group is done.
    Finished { remaining_budget: RowBudget },
}

/// Everything a [`WindowedRowGroup`] needs that is owned by the enclosing
/// [`RowGroupReaderBuilder`](super::RowGroupReaderBuilder).
#[derive(Debug, Clone)]
pub(crate) struct WindowedConfig {
    pub(crate) batch_size: usize,
    /// Rows of *demand lookahead*. Always at least `batch_size`.
    pub(crate) window_rows: usize,
    pub(crate) projection: ProjectionMask,
    pub(crate) metadata: Arc<ParquetMetaData>,
    pub(crate) fields: Option<Arc<ParquetField>>,
    pub(crate) metrics: ArrowReaderMetrics,
    pub(crate) max_predicate_cache_size: usize,
    pub(crate) row_selection_policy: RowSelectionPolicy,
}

/// Decodes one row group a window of rows at a time.
pub(crate) struct WindowedRowGroup {
    config: WindowedConfig,
    row_group_idx: usize,
    row_count: usize,
    /// Pages resident for this row group. Shared with every live reader.
    store: Arc<PageStore>,
    /// Remaining offset/limit budget.
    budget: RowBudget,
    mode: Mode,
}

impl std::fmt::Debug for WindowedRowGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowedRowGroup")
            .field("row_group_idx", &self.row_group_idx)
            .field("row_count", &self.row_count)
            .field("budget", &self.budget)
            .field("filtered", &matches!(self.mode, Mode::Filtered(_)))
            .finish()
    }
}

enum Mode {
    Unfiltered(Unfiltered),
    Filtered(Box<Filtered>),
}

// ---------------------------------------------------------------------------
// Unfiltered
// ---------------------------------------------------------------------------

struct Unfiltered {
    /// Pages of this row group tagged with the span of *selected* rows each
    /// serves. Built once, up front.
    plan: ScanPlan,
    /// The read plan for the whole row group, taken when the reader is built.
    pending_plan: Option<ReadPlanBuilder>,
    /// The single long-lived reader. Built after the first window lands.
    reader: Option<ParquetRecordBatchReader>,
    /// Selected rows emitted so far.
    emitted: u64,
    done: bool,
}

// ---------------------------------------------------------------------------
// Filtered
// ---------------------------------------------------------------------------

/// Where the filtered state machine is.
enum Stage {
    /// Decide whether to open a new window or emit a batch.
    Idle,
    /// Evaluating predicate `idx` over `cand` (absolute selected-row ranges
    /// remaining in the current window).
    Predicate {
        cand: Vec<Range<usize>>,
        idx: usize,
    },
    /// Emitting one batch covering `out` (absolute selected-row ranges).
    Output {
        out: Vec<Range<usize>>,
    },
    Done,
}

struct Filtered {
    filter: RowFilter,
    cache: Arc<RwLock<RowGroupCache>>,
    cache_projection: ProjectionMask,
    /// The scan's row selection for this row group, as absolute row ranges.
    base: Vec<Range<usize>>,
    /// First row of the next window to filter.
    window_start: usize,
    /// No more windows will be opened.
    filter_done: bool,
    /// One retained [`ArrayReader`] per predicate, and the absolute row each
    /// has consumed up to.
    pred_readers: Vec<Option<Box<dyn ArrayReader>>>,
    pred_pos: Vec<usize>,
    /// Rows that survived every predicate and the offset/limit budget, in
    /// absolute row order, not yet output.
    ready: Vec<Range<usize>>,
    ready_rows: usize,
    /// Retained output [`ArrayReader`] and the absolute row it has consumed to.
    out_reader: Option<Box<dyn ArrayReader>>,
    out_pos: usize,
    stage: Stage,
}

impl WindowedRowGroup {
    /// Prepare to decode `row_group_idx` in windows.
    ///
    /// Returns `Ok(None)` when this row group cannot be decoded at page
    /// granularity — currently that means it has no offset index, since page
    /// locations are what make per-page demand expressible. The caller should
    /// fall back to the whole-row-group path.
    pub(crate) fn try_new(
        config: WindowedConfig,
        row_group_idx: usize,
        row_count: usize,
        selection: Option<RowSelection>,
        budget: RowBudget,
        filter: Option<RowFilter>,
    ) -> Result<Option<(Self, Option<RowFilter>)>, ParquetError> {
        if row_group_offset_index(&config.metadata, row_group_idx).is_none() {
            return Ok(None);
        }

        let store = Arc::new(PageStore::default());
        let has_predicates = filter
            .as_ref()
            .is_some_and(|f| !f.predicates.is_empty())
            // A filter whose predicates are all empty is equivalent to none.
            ;

        if !has_predicates {
            // The selection is final, so plan the whole row group up front.
            let plan_builder = ReadPlanBuilder::new(config.batch_size)
                .with_selection(selection)
                .with_row_selection_policy(config.row_selection_policy);
            let budgeted = budget.apply_to_plan(plan_builder, row_count);
            let remaining_budget = budgeted.remaining_budget;
            if budgeted.rows_after_budget == 0 {
                // Nothing to read; report as an immediately-finished group.
                return Ok(Some((
                    Self {
                        config,
                        row_group_idx,
                        row_count,
                        store,
                        budget: remaining_budget,
                        mode: Mode::Unfiltered(Unfiltered {
                            plan: ScanPlan {
                                ranges: vec![],
                                total_selected_rows: 0,
                            },
                            pending_plan: None,
                            reader: None,
                            emitted: 0,
                            done: true,
                        }),
                    },
                    filter,
                )));
            }

            let mut plan_builder = budgeted
                .plan_builder
                .with_row_selection_policy(config.row_selection_policy);
            plan_builder = override_selector_strategy_if_needed(
                plan_builder,
                &config.projection,
                row_group_offset_index(&config.metadata, row_group_idx),
            );

            let Some(plan) = plan_scan_ranges(
                &config.metadata,
                &[row_group_idx],
                &config.projection,
                plan_builder.selection(),
            ) else {
                return Ok(None);
            };

            return Ok(Some((
                Self {
                    config,
                    row_group_idx,
                    row_count,
                    store,
                    budget: remaining_budget,
                    mode: Mode::Unfiltered(Unfiltered {
                        plan,
                        pending_plan: Some(plan_builder),
                        reader: None,
                        emitted: 0,
                        done: false,
                    }),
                },
                filter,
            )));
        }

        let filter = filter.expect("has_predicates implies a filter");
        let cache_projection = compute_cache_projection(&config, row_group_idx, &filter);
        let cache = Arc::new(RwLock::new(RowGroupCache::new(
            config.batch_size,
            config.max_predicate_cache_size,
        )));
        let base = match &selection {
            Some(selection) => selection_to_ranges(selection, 0),
            None => std::iter::once(0..row_count).collect(),
        };
        let num_predicates = filter.predicates.len();

        Ok(Some((
            Self {
                config,
                row_group_idx,
                row_count,
                store,
                budget,
                mode: Mode::Filtered(Box::new(Filtered {
                    filter,
                    cache,
                    cache_projection,
                    base,
                    window_start: 0,
                    filter_done: false,
                    pred_readers: (0..num_predicates).map(|_| None).collect(),
                    pred_pos: vec![0; num_predicates],
                    ready: Vec::new(),
                    ready_rows: 0,
                    out_reader: None,
                    out_pos: 0,
                    stage: Stage::Idle,
                })),
            },
            None,
        )))
    }

    /// Bytes this row group is holding resident.
    pub(crate) fn buffered_bytes(&self) -> u64 {
        self.store.buffered_bytes()
    }

    /// Hand the [`RowFilter`] back to the enclosing builder when finished.
    pub(crate) fn take_filter(&mut self) -> Option<RowFilter> {
        match &mut self.mode {
            Mode::Unfiltered(_) => None,
            Mode::Filtered(filtered) => Some(std::mem::replace(
                &mut filtered.filter,
                RowFilter::new(vec![]),
            )),
        }
    }

    /// Produce the next batch, or say what bytes are missing for it.
    pub(crate) fn try_next(
        &mut self,
        buffers: &mut PushBuffers,
    ) -> Result<WindowedResult, ParquetError> {
        match &mut self.mode {
            Mode::Unfiltered(_) => self.try_next_unfiltered(buffers),
            Mode::Filtered(_) => self.try_next_filtered(buffers),
        }
    }

    // -----------------------------------------------------------------
    // shared helpers
    // -----------------------------------------------------------------

    fn offset_index(&self) -> Option<&[OffsetIndexMetaData]> {
        row_group_offset_index(&self.config.metadata, self.row_group_idx)
    }

    /// A row group whose column chunks read from [`Self::store`].
    ///
    /// Cheap to build; the readers it produces hold `Arc`s to the store, so
    /// they see pages pushed later.
    fn shared_row_group(&self, projection: &ProjectionMask) -> InMemoryRowGroup<'_> {
        let meta = self.config.metadata.row_group(self.row_group_idx);
        let column_chunks = meta
            .columns()
            .iter()
            .enumerate()
            .map(|(idx, column)| {
                if !projection.leaf_included(idx) {
                    return None;
                }
                Some(Arc::new(ColumnChunkData::Shared {
                    length: column.byte_range().1 as usize,
                    store: Arc::clone(&self.store),
                }))
            })
            .collect();
        InMemoryRowGroup {
            row_count: self.row_count,
            column_chunks,
            offset_index: self.offset_index(),
            row_group_idx: self.row_group_idx,
            metadata: &self.config.metadata,
        }
    }

    /// Page ranges required to read `selection`'s rows for `projection`.
    fn plan_ranges(
        &self,
        projection: &ProjectionMask,
        selection: &RowSelection,
        cache_mask: Option<&ProjectionMask>,
    ) -> Vec<Range<u64>> {
        let num_columns = self
            .config
            .metadata
            .row_group(self.row_group_idx)
            .columns()
            .len();
        let planning = InMemoryRowGroup {
            row_count: self.row_count,
            column_chunks: vec![None; num_columns],
            offset_index: self.offset_index(),
            row_group_idx: self.row_group_idx,
            metadata: &self.config.metadata,
        };
        planning
            .fetch_ranges(
                projection,
                Some(selection),
                self.config.batch_size,
                cache_mask,
            )
            .ranges
    }

    /// Move `ranges` from `buffers` into the page store.
    ///
    /// Returns the ranges that are still missing; when that is empty every
    /// range is resident and the corresponding bytes have been released from
    /// `buffers`.
    fn ingest(
        &self,
        buffers: &mut PushBuffers,
        ranges: Vec<Range<u64>>,
    ) -> Result<Vec<Range<u64>>, ParquetError> {
        let wanted: Vec<Range<u64>> = ranges
            .into_iter()
            .filter(|range| !self.store.contains(range))
            .collect();
        let missing: Vec<Range<u64>> = wanted
            .iter()
            .filter(|range| !buffers.has_range(range))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Ok(missing);
        }
        for range in &wanted {
            let len = usize::try_from(range.end - range.start)
                .map_err(|e| ParquetError::General(format!("range too large: {e}")))?;
            let bytes = crate::file::reader::ChunkReader::get_bytes(buffers, range.start, len)
                .map_err(|e| {
                    ParquetError::General(format!(
                        "Internal Error missing data for range {range:?} in buffers: {e}"
                    ))
                })?;
            self.store.insert(range.start, bytes);
        }
        buffers.clear_ranges(&wanted);
        Ok(vec![])
    }

    /// Drop data pages that no live reader can reach any more.
    ///
    /// `min_row` is the lowest absolute row any retained reader is still
    /// positioned at; every data page that ends at or before it is dead.
    /// Dictionary pages are never dropped: they are needed for every row of
    /// the chunk, and a reader built later in the row group (the output reader
    /// is built after the first predicate window) would have to refetch one.
    fn evict_before_row(&self, min_row: usize) {
        let Some(offset_index) = self.offset_index() else {
            return;
        };
        let dead = offset_index.iter().flat_map(|column| {
            let locations = column.page_locations();
            locations
                .iter()
                .enumerate()
                .take_while(move |(idx, _)| {
                    // A page is dead once the row *after* its last is behind
                    // the cursor, i.e. the next page starts at or before it.
                    locations
                        .get(idx + 1)
                        .is_some_and(|next| (next.first_row_index as usize) <= min_row)
                })
                .map(|(_, location)| location.offset as u64)
        });
        self.store.remove(dead);
    }

    // -----------------------------------------------------------------
    // unfiltered
    // -----------------------------------------------------------------

    fn try_next_unfiltered(
        &mut self,
        buffers: &mut PushBuffers,
    ) -> Result<WindowedResult, ParquetError> {
        let Mode::Unfiltered(state) = &self.mode else {
            unreachable!()
        };
        if state.done {
            return Ok(WindowedResult::Finished {
                remaining_budget: self.budget,
            });
        }

        // What the next window of selected rows needs.
        let emitted = state.emitted;
        let window_end = emitted + self.config.window_rows as u64;
        let ranges: Vec<Range<u64>> = state
            .plan
            .ranges
            .iter()
            .filter(|planned| planned.first_row < window_end && planned.last_row > emitted)
            .map(|planned| planned.range.clone())
            .collect();

        let missing = self.ingest(buffers, ranges)?;
        if !missing.is_empty() {
            return Ok(WindowedResult::NeedsData(missing));
        }

        // Build the (single) reader once the first window has landed.
        let Mode::Unfiltered(state) = &mut self.mode else {
            unreachable!()
        };
        if state.reader.is_none() {
            let plan_builder = state
                .pending_plan
                .take()
                .ok_or_else(|| ParquetError::General("windowed plan already taken".into()))?;
            let plan = plan_builder.build();
            let row_group = self.shared_row_group(&self.config.projection);
            let array_reader = ArrayReaderBuilder::new(&row_group, &self.config.metrics)
                .with_batch_size(self.config.batch_size)
                .with_parquet_metadata(&self.config.metadata)
                .build_array_reader(self.config.fields.as_deref(), &self.config.projection)?;
            let Mode::Unfiltered(state) = &mut self.mode else {
                unreachable!()
            };
            state.reader = Some(ParquetRecordBatchReader::new(array_reader, plan));
        }

        let Mode::Unfiltered(state) = &mut self.mode else {
            unreachable!()
        };
        let reader = state.reader.as_mut().expect("reader built above");
        match reader.next() {
            Some(Ok(batch)) => {
                state.emitted += batch.num_rows() as u64;
                let emitted = state.emitted;
                // Every page whose last selected row is behind the cursor is
                // dead; releasing them is what bounds resident bytes to the
                // window rather than the row group.
                let dead = state
                    .plan
                    .ranges
                    .iter()
                    .filter(|planned| planned.last_row <= emitted)
                    .map(|planned| planned.range.start)
                    .collect::<Vec<_>>();
                self.store.remove(dead);
                Ok(WindowedResult::Batch(batch))
            }
            Some(Err(e)) => Err(ParquetError::ArrowError(e.to_string())),
            None => {
                state.done = true;
                state.reader = None;
                self.store.clear();
                Ok(WindowedResult::Finished {
                    remaining_budget: self.budget,
                })
            }
        }
    }

    // -----------------------------------------------------------------
    // filtered
    // -----------------------------------------------------------------

    fn try_next_filtered(
        &mut self,
        buffers: &mut PushBuffers,
    ) -> Result<WindowedResult, ParquetError> {
        loop {
            let Mode::Filtered(state) = &mut self.mode else {
                unreachable!()
            };
            match std::mem::replace(&mut state.stage, Stage::Idle) {
                Stage::Done => {
                    state.stage = Stage::Done;
                    self.store.clear();
                    return Ok(WindowedResult::Finished {
                        remaining_budget: self.budget,
                    });
                }
                Stage::Idle => {
                    // Emit a batch as soon as a full one is available, or once
                    // no more rows can arrive.
                    if state.ready_rows >= self.config.batch_size || state.filter_done {
                        if state.ready_rows == 0 {
                            if state.filter_done {
                                state.stage = Stage::Done;
                                continue;
                            }
                        } else {
                            let out = take_rows(&mut state.ready, self.config.batch_size);
                            state.ready_rows -= total_rows(&out);
                            state.stage = Stage::Output { out };
                            continue;
                        }
                    }
                    // Otherwise filter another window.
                    if state.filter_done
                        || state.window_start >= self.row_count
                        || self.budget.is_exhausted()
                    {
                        state.filter_done = true;
                        continue;
                    }
                    let window_end =
                        (state.window_start + self.config.window_rows).min(self.row_count);
                    let cand = intersect(&state.base, state.window_start..window_end);
                    state.window_start = window_end;
                    if cand.is_empty() {
                        continue;
                    }
                    state.stage = Stage::Predicate { cand, idx: 0 };
                }
                Stage::Predicate { cand, idx } => {
                    if idx >= state.filter.predicates.len() {
                        // Survived every predicate: apply the running
                        // offset/limit budget and queue for output.
                        let selected = total_rows(&cand);
                        let plan_builder = ReadPlanBuilder::new(self.config.batch_size)
                            .with_selection(Some(ranges_to_selection(&cand, 0)));
                        let budgeted = self.budget.apply_to_plan(plan_builder, self.row_count);
                        self.budget = budgeted.remaining_budget;
                        debug_assert_eq!(budgeted.rows_before_budget, selected);
                        let kept = budgeted
                            .plan_builder
                            .selection()
                            .map(|selection| selection_to_ranges(selection, 0))
                            .unwrap_or_default();
                        let Mode::Filtered(state) = &mut self.mode else {
                            unreachable!()
                        };
                        state.ready_rows += total_rows(&kept);
                        state.ready.extend(kept);
                        if self.budget.is_exhausted() {
                            let Mode::Filtered(state) = &mut self.mode else {
                                unreachable!()
                            };
                            state.filter_done = true;
                        }
                        let Mode::Filtered(state) = &mut self.mode else {
                            unreachable!()
                        };
                        state.stage = Stage::Idle;
                        // Predicate readers have moved on; release what they
                        // and the output reader are both past.
                        let min_row = state
                            .pred_pos
                            .iter()
                            .copied()
                            .chain(std::iter::once(state.out_pos))
                            .min()
                            .unwrap_or(0);
                        self.evict_before_row(min_row);
                        continue;
                    }

                    match self.step_predicate(buffers, cand, idx)? {
                        Some(missing) => return Ok(WindowedResult::NeedsData(missing)),
                        None => continue,
                    }
                }
                Stage::Output { out } => match self.step_output(buffers, out)? {
                    Ok(batch) => return Ok(WindowedResult::Batch(batch)),
                    Err(missing) => return Ok(WindowedResult::NeedsData(missing)),
                },
            }
        }
    }

    /// Evaluate predicate `idx` over `cand`. On success the stage is advanced
    /// with the narrowed candidate set; otherwise the missing ranges are
    /// returned and the stage is left untouched so the call can be retried.
    fn step_predicate(
        &mut self,
        buffers: &mut PushBuffers,
        cand: Vec<Range<usize>>,
        idx: usize,
    ) -> Result<Option<Vec<Range<u64>>>, ParquetError> {
        let Mode::Filtered(state) = &mut self.mode else {
            unreachable!()
        };
        let projection = state.filter.predicates[idx].projection().clone();
        let cache_projection = state.cache_projection.clone();
        let absolute = ranges_to_selection(&cand, 0);
        let ranges = self.plan_ranges(&projection, &absolute, Some(&cache_projection));
        let missing = self.ingest(buffers, ranges)?;
        if !missing.is_empty() {
            let Mode::Filtered(state) = &mut self.mode else {
                unreachable!()
            };
            state.stage = Stage::Predicate { cand, idx };
            return Ok(Some(missing));
        }

        // Build (once) the retained reader for this predicate's columns.
        let Mode::Filtered(state) = &self.mode else {
            unreachable!()
        };
        if state.pred_readers[idx].is_none() {
            let row_group = self.shared_row_group(&projection);
            let Mode::Filtered(state) = &self.mode else {
                unreachable!()
            };
            let cache_options = crate::arrow::array_reader::CacheOptionsBuilder::new(
                &state.cache_projection,
                &state.cache,
            )
            .producer();
            let array_reader = ArrayReaderBuilder::new(&row_group, &self.config.metrics)
                .with_batch_size(self.config.batch_size)
                .with_cache_options(Some(&cache_options))
                .with_parquet_metadata(&self.config.metadata)
                .build_array_reader(self.config.fields.as_deref(), &projection)?;
            let Mode::Filtered(state) = &mut self.mode else {
                unreachable!()
            };
            state.pred_readers[idx] = Some(array_reader);
        }

        let Mode::Filtered(state) = &mut self.mode else {
            unreachable!()
        };
        let pos = state.pred_pos[idx];
        let relative = ranges_to_selection(&cand, pos).trim();
        let consumed = relative.row_count() + relative.skipped_row_count();
        let array_reader = state.pred_readers[idx]
            .take()
            .expect("predicate reader built above");
        let plan = ReadPlanBuilder::new(self.config.batch_size)
            .with_selection(Some(relative.clone()))
            .with_row_selection_policy(RowSelectionPolicy::Selectors)
            .build();

        let mut reader = ParquetRecordBatchReader::new(array_reader, plan);
        let mut filters = Vec::new();
        let predicate = state.filter.predicates[idx].as_mut();
        let mut input_rows_total = 0usize;
        loop {
            let batch = match reader.next() {
                Some(Ok(batch)) => batch,
                Some(Err(e)) => {
                    // Keep the reader so the row group can be torn down cleanly.
                    state.pred_readers[idx] = Some(reader.into_array_reader());
                    return Err(ParquetError::ArrowError(e.to_string()));
                }
                None => break,
            };
            let input_rows = batch.num_rows();
            input_rows_total += input_rows;
            let filter = predicate.evaluate(batch)?;
            if filter.len() != input_rows {
                state.pred_readers[idx] = Some(reader.into_array_reader());
                return Err(ParquetError::ArrowError(format!(
                    "ArrowPredicate predicate returned {} rows, expected {input_rows}",
                    filter.len()
                )));
            }
            let filter = match filter.null_count() {
                0 => filter,
                _ => arrow_select::filter::prep_null_mask_filter(&filter),
            };
            filters.push(filter);
        }
        state.pred_readers[idx] = Some(reader.into_array_reader());
        state.pred_pos[idx] = pos + consumed;

        debug_assert_eq!(input_rows_total, relative.row_count());
        let raw = RowSelection::from_filters(&filters);
        let narrowed = relative.and_then(&raw);
        let next_cand = selection_to_ranges(&narrowed, pos);
        state.stage = Stage::Predicate {
            cand: next_cand,
            idx: idx + 1,
        };
        Ok(None)
    }

    /// Emit exactly one batch covering `out`.
    fn step_output(
        &mut self,
        buffers: &mut PushBuffers,
        out: Vec<Range<usize>>,
    ) -> Result<Result<RecordBatch, Vec<Range<u64>>>, ParquetError> {
        let absolute = ranges_to_selection(&out, 0);
        let ranges = self.plan_ranges(&self.config.projection.clone(), &absolute, None);
        let missing = self.ingest(buffers, ranges)?;
        if !missing.is_empty() {
            let Mode::Filtered(state) = &mut self.mode else {
                unreachable!()
            };
            state.stage = Stage::Output { out };
            return Ok(Err(missing));
        }

        let Mode::Filtered(state) = &self.mode else {
            unreachable!()
        };
        if state.out_reader.is_none() {
            let row_group = self.shared_row_group(&self.config.projection);
            let Mode::Filtered(state) = &self.mode else {
                unreachable!()
            };
            let cache_options: CacheOptions<'_> =
                crate::arrow::array_reader::CacheOptionsBuilder::new(
                    &state.cache_projection,
                    &state.cache,
                )
                .consumer();
            let array_reader = ArrayReaderBuilder::new(&row_group, &self.config.metrics)
                .with_batch_size(self.config.batch_size)
                .with_cache_options(Some(&cache_options))
                .with_parquet_metadata(&self.config.metadata)
                .build_array_reader(self.config.fields.as_deref(), &self.config.projection)?;
            let Mode::Filtered(state) = &mut self.mode else {
                unreachable!()
            };
            state.out_reader = Some(array_reader);
        }

        let Mode::Filtered(state) = &mut self.mode else {
            unreachable!()
        };
        let pos = state.out_pos;
        let relative = ranges_to_selection(&out, pos).trim();
        let consumed = relative.row_count() + relative.skipped_row_count();
        let array_reader = state.out_reader.take().expect("output reader built above");
        let plan = ReadPlanBuilder::new(self.config.batch_size)
            .with_selection(Some(relative))
            .with_row_selection_policy(RowSelectionPolicy::Selectors)
            .build();
        // The plan covers at most `batch_size` selected rows, so it yields
        // exactly one batch — the same batch the non-windowed reader would
        // have produced at this point.
        let mut reader = ParquetRecordBatchReader::new(array_reader, plan);
        let batch = reader.next();
        let extra = reader.next();
        state.out_reader = Some(reader.into_array_reader());
        if extra.is_some() {
            return Err(ParquetError::General(
                "Internal Error: windowed output plan produced more than one batch".into(),
            ));
        }
        state.out_pos = pos + consumed;
        state.stage = Stage::Idle;

        match batch {
            Some(Ok(batch)) => Ok(Ok(batch)),
            Some(Err(e)) => Err(ParquetError::ArrowError(e.to_string())),
            None => Err(ParquetError::General(
                "windowed output plan produced no batch".into(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn row_group_offset_index(
    metadata: &ParquetMetaData,
    row_group_idx: usize,
) -> Option<&[OffsetIndexMetaData]> {
    metadata
        .offset_index()
        .filter(|index| !index.is_empty())
        .and_then(|index| index.get(row_group_idx))
        .map(|columns| columns.as_slice())
        .filter(|columns| columns.iter().all(|c| !c.page_locations().is_empty()))
}

fn compute_cache_projection(
    config: &WindowedConfig,
    row_group_idx: usize,
    filter: &RowFilter,
) -> ProjectionMask {
    let meta = config.metadata.row_group(row_group_idx);
    let none = || ProjectionMask::none(meta.columns().len());
    if config.max_predicate_cache_size == 0 {
        return none();
    }
    let Some(first) = filter.predicates.first() else {
        return none();
    };
    let mut cache_projection = first.projection().clone();
    for predicate in filter.predicates.iter() {
        cache_projection.union(predicate.projection());
    }
    cache_projection.intersect(&config.projection);
    cache_projection
        .without_nested_types(config.metadata.file_metadata().schema_descr())
        .unwrap_or_else(none)
}

/// Absolute selected-row ranges -> a [`RowSelection`] whose row 0 is `from`.
fn ranges_to_selection(ranges: &[Range<usize>], from: usize) -> RowSelection {
    let mut selectors = Vec::with_capacity(ranges.len() * 2);
    let mut pos = from;
    for range in ranges {
        debug_assert!(range.start >= pos, "ranges must be sorted and disjoint");
        if range.start > pos {
            selectors.push(RowSelector::skip(range.start - pos));
        }
        selectors.push(RowSelector::select(range.end - range.start));
        pos = range.end;
    }
    RowSelection::from(selectors)
}

/// Inverse of [`ranges_to_selection`].
fn selection_to_ranges(selection: &RowSelection, from: usize) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    let mut pos = from;
    for selector in selection.iter() {
        if !selector.skip && selector.row_count > 0 {
            match out.last_mut() {
                Some(last) if last.end == pos => last.end = pos + selector.row_count,
                _ => out.push(pos..pos + selector.row_count),
            }
        }
        pos += selector.row_count;
    }
    out
}

fn total_rows(ranges: &[Range<usize>]) -> usize {
    ranges.iter().map(|r| r.end - r.start).sum()
}

/// Clip `ranges` to `window`.
fn intersect(ranges: &[Range<usize>], window: Range<usize>) -> Vec<Range<usize>> {
    ranges
        .iter()
        .filter_map(|range| {
            let start = range.start.max(window.start);
            let end = range.end.min(window.end);
            (start < end).then_some(start..end)
        })
        .collect()
}

/// Remove and return the first `n` rows from `ranges`.
fn take_rows(ranges: &mut Vec<Range<usize>>, n: usize) -> Vec<Range<usize>> {
    let mut taken = Vec::new();
    let mut remaining = n;
    let mut consumed_ranges = 0;
    for range in ranges.iter_mut() {
        if remaining == 0 {
            break;
        }
        let len = range.end - range.start;
        if len <= remaining {
            taken.push(range.clone());
            remaining -= len;
            consumed_ranges += 1;
        } else {
            taken.push(range.start..range.start + remaining);
            range.start += remaining;
            remaining = 0;
        }
    }
    ranges.drain(..consumed_ranges);
    taken
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_ranges_and_selection() {
        let ranges = vec![3..7, 10..12];
        let selection = ranges_to_selection(&ranges, 0);
        assert_eq!(selection.row_count(), 6);
        assert_eq!(selection_to_ranges(&selection, 0), ranges);

        // Relative framing: row 0 of the selection is row 3 of the row group.
        let relative = ranges_to_selection(&ranges, 3);
        assert_eq!(selection_to_ranges(&relative, 3), ranges);
    }

    #[test]
    fn intersect_clips_to_window() {
        let ranges = vec![0..10, 20..30];
        assert_eq!(intersect(&ranges, 5..25), vec![5..10, 20..25]);
        assert_eq!(intersect(&ranges, 10..20), Vec::<Range<usize>>::new());
        assert_eq!(intersect(&ranges, 0..100), ranges);
    }

    #[test]
    fn take_rows_splits_ranges() {
        let mut ranges = vec![0..10, 20..30];
        assert_eq!(take_rows(&mut ranges, 4), vec![0..4]);
        assert_eq!(ranges, vec![4..10, 20..30]);
        assert_eq!(take_rows(&mut ranges, 8), vec![4..10, 20..22]);
        assert_eq!(ranges, vec![22..30]);
        assert_eq!(take_rows(&mut ranges, 100), vec![22..30]);
        assert!(ranges.is_empty());
    }
}

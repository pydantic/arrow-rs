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

//! [`PageStore`]: a *growable* page-level byte store shared with in-flight
//! [`ParquetRecordBatchReader`]s.
//!
//! [`ParquetRecordBatchReader`]: crate::arrow::arrow_reader::ParquetRecordBatchReader
//!
//! # Why this exists
//!
//! [`ColumnChunkData`] is immutable: a
//! [`ParquetRecordBatchReader`](crate::arrow::arrow_reader::ParquetRecordBatchReader)
//! can only be built once every byte it will ever read is already in hand.
//! That is what forces the push decoder to buffer a whole row group before
//! handing out a reader.
//!
//! A `PageStore` breaks that coupling. Pages are keyed by their file offset
//! and can be *added after* the reader that reads them was built, so one
//! reader — and therefore one set of column readers, with their decoded
//! dictionaries — can be kept alive for a whole row group while the bytes it
//! feeds on arrive a window at a time.
//!
//! [`ColumnChunkData`]: crate::arrow::in_memory_row_group::ColumnChunkData
//!
//! # Why keying by page offset is sound
//!
//! When an [`OffsetIndex`] is available,
//! [`SerializedPageReader`](crate::file::serialized_reader::SerializedPageReader)
//! runs in its `Pages` state, where it
//!
//! * reads exactly one page per call, via `get_bytes(page.offset,
//!   page.compressed_page_size)` — always an *exact* page start and length,
//! * visits pages in increasing offset order, and
//! * skips pages it does not need without any read at all.
//!
//! So a map from page offset to page bytes is exactly the access pattern, and
//! a page can be dropped as soon as the decode cursor is past it. This is the
//! same contract [`ColumnChunkData::Sparse`] already relies on; `PageStore`
//! only makes the set of resident pages change over time.
//!
//! [`OffsetIndex`]: crate::file::page_index::offset_index::OffsetIndexMetaData
//! [`ColumnChunkData::Sparse`]: crate::arrow::in_memory_row_group::ColumnChunkData::Sparse

use bytes::Bytes;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Mutex;

/// A growable, shared store of Parquet pages keyed by file offset.
///
/// Page offsets are unique within a file, so a single store serves every
/// column chunk of a row group. Cloning is by [`Arc`](std::sync::Arc) at the
/// call site; the store itself is interior-mutable so pages can be added and
/// dropped while readers hold references to it.
#[derive(Debug, Default)]
pub(crate) struct PageStore {
    /// file offset -> bytes of the page starting at that offset
    pages: Mutex<BTreeMap<u64, Bytes>>,
}

impl PageStore {
    /// Add the bytes for the page (or dictionary-page prefix) starting at
    /// `offset`. Re-inserting an existing offset is a no-op, so pushing the
    /// same range twice never doubles memory.
    pub(crate) fn insert(&self, offset: u64, data: Bytes) {
        self.pages.lock().unwrap().entry(offset).or_insert(data);
    }

    /// True if the page starting at `range.start` is resident and at least as
    /// long as `range`.
    pub(crate) fn contains(&self, range: &Range<u64>) -> bool {
        self.pages
            .lock()
            .unwrap()
            .get(&range.start)
            .is_some_and(|bytes| bytes.len() as u64 >= range.end - range.start)
    }

    /// Bytes for the page starting exactly at `start`, if resident.
    pub(crate) fn get(&self, start: u64) -> Option<Bytes> {
        self.pages.lock().unwrap().get(&start).cloned()
    }

    /// Drop the named pages.
    ///
    /// Eviction is by explicit offset rather than by a watermark because
    /// column chunks are laid out one after another in the file: a single
    /// "everything below X" cutoff cannot express "column A is done with its
    /// first five pages and so is column B", which is exactly the shape of a
    /// decode cursor moving through a row group.
    pub(crate) fn remove(&self, offsets: impl IntoIterator<Item = u64>) {
        let mut pages = self.pages.lock().unwrap();
        for offset in offsets {
            pages.remove(&offset);
        }
    }

    /// Drop everything.
    pub(crate) fn clear(&self) {
        self.pages.lock().unwrap().clear();
    }

    /// Total resident bytes.
    pub(crate) fn buffered_bytes(&self) -> u64 {
        self.pages
            .lock()
            .unwrap()
            .values()
            .map(|b| b.len() as u64)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_contains_get() {
        let store = PageStore::default();
        assert!(!store.contains(&(10..20)));
        store.insert(10, Bytes::from_static(&[0u8; 10]));
        assert!(store.contains(&(10..20)));
        // A longer request than the resident page is not satisfied
        assert!(!store.contains(&(10..21)));
        // Only exact page starts resolve
        assert!(!store.contains(&(11..20)));
        assert_eq!(store.get(10).unwrap().len(), 10);
        assert!(store.get(11).is_none());
        assert_eq!(store.buffered_bytes(), 10);
    }

    #[test]
    fn insert_is_idempotent() {
        let store = PageStore::default();
        store.insert(0, Bytes::from_static(&[0u8; 8]));
        store.insert(0, Bytes::from_static(&[0u8; 8]));
        assert_eq!(store.buffered_bytes(), 8);
    }

    #[test]
    fn remove_drops_only_named_pages() {
        let store = PageStore::default();
        store.insert(0, Bytes::from_static(&[0u8; 4]));
        store.insert(4, Bytes::from_static(&[0u8; 4]));
        store.insert(8, Bytes::from_static(&[0u8; 4]));
        store.remove([0, 8]);
        assert!(store.get(0).is_none());
        assert!(store.get(4).is_some());
        assert!(store.get(8).is_none());
        assert_eq!(store.buffered_bytes(), 4);
        store.clear();
        assert_eq!(store.buffered_bytes(), 0);
    }
}

use crate::errors::{ParquetError, Result};
use bytes::Bytes;

pub const SYMBOL_WIDTH_BYTES: usize = 8;
pub const NUM_TABLE_ELEMENTS: usize = 256;
pub const ESCAPE_CODE: u8 = 0xFF;

#[derive(Debug, Clone)]
pub struct SymbolTable {
    inner: Bytes,
}

impl SymbolTable {
    pub fn index(&self, i: u8) -> &[u8; 8] {
        let offset = i as usize * 8;

        // safety: we know there are exactly 256 * 8 bytes and i is typed to range from 0..256
        let slice = unsafe { self.inner.get_unchecked(offset..offset + 8) };

        // safety: we just sliced 8 bytes above
        unsafe { &*(slice.as_ptr() as *const [u8; 8]) }
    }
}

#[derive(Debug, Clone)]
pub struct LengthTable {
    inner: Bytes,
}

impl LengthTable {
    pub fn index(&self, i: u8) -> u8 {
        // safety: we know there are exactly 256 bytes and i is typed to range from 0..256
        unsafe { *self.inner.get_unchecked(i as usize) }
    }
}

#[derive(Debug, Clone)]
pub struct FSSTHeader {
    pub(crate) symbol_table: SymbolTable,
    pub(crate) length_table: LengthTable,
}

/// reads and builds up the symbol and length table
pub fn read_fsst_header(bytes: Bytes) -> Result<(FSSTHeader, Bytes)> {
    const SYMBOL_TABLE_BYTES: usize = NUM_TABLE_ELEMENTS * SYMBOL_WIDTH_BYTES;
    const HEADER_SIZE: usize = SYMBOL_TABLE_BYTES + NUM_TABLE_ELEMENTS;

    if bytes.len() < HEADER_SIZE {
        return Err(ParquetError::General(
            "insufficient bytes for FSST header".into(),
        ));
    }

    let symbol_table = SymbolTable {
        inner: bytes.slice(0..SYMBOL_TABLE_BYTES),
    };

    let length_table = LengthTable {
        inner: bytes.slice(SYMBOL_TABLE_BYTES..HEADER_SIZE),
    };

    let header = FSSTHeader {
        symbol_table,
        length_table,
    };

    let data = bytes.slice(HEADER_SIZE..);

    Ok((header, data))
}

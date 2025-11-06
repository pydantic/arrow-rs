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

use std::marker::PhantomData;

use bytes::Bytes;

use super::{Decoder, DeltaBitPackDecoder};
use crate::basic::{Encoding, Type};
use crate::data_type::private::ParquetValueType;
use crate::data_type::{DataType, Int32Type};
use crate::encodings::fsst::{self, FSSTHeader};
use crate::errors::{ParquetError, Result};

pub const ESCAPE_CODE: u8 = 255;

pub enum DecodeResult<'a> {
    /// Escape code encountered - the next byte should be copied literally
    Escape,
    /// A symbol was found: (symbol_bytes, actual_length)
    Symbol(&'a [u8; 8], u8),
}

pub struct FSSTDecoder<T> {
    header: Option<FSSTHeader>,

    compressed_lengths: Vec<i32>,
    length_cursor: usize,

    data: Option<Bytes>,
    data_cursor: usize,

    num_values: usize,
    _phantom: PhantomData<T>,
}

impl<T> FSSTDecoder<T> {
    pub fn new() -> Self {
        Self {
            header: None,
            compressed_lengths: Vec::new(),
            length_cursor: 0,
            data: None,
            data_cursor: 0,
            num_values: 0,
            _phantom: PhantomData,
        }
    }

    /// Decodes a complete string by reading exactly `compressed_len` bytes.
    /// Returns the decompressed string as Bytes.
    fn decode_string(&mut self, compressed_len: usize) -> Result<Bytes> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| ParquetError::General("not initialized".into()))?;

        let header = self
            .header
            .as_ref()
            .ok_or_else(|| ParquetError::General("not initialized".into()))?;

        let mut output = Vec::new();

        let end_offset = self.data_cursor + compressed_len;

        while self.data_cursor < end_offset {
            let code = data
                .get(self.data_cursor)
                .ok_or_else(|| ParquetError::General("unexpected end of data".into()))?;

            self.data_cursor += 1;

            if *code == ESCAPE_CODE {
                let &byte = data.get(self.data_cursor).ok_or_else(|| {
                    ParquetError::General("unexpected end of data after escape".into())
                })?;

                self.data_cursor += 1;
                output.push(byte);
            } else {
                let sym = header.symbol_table.index(*code);
                let len = header.length_table.index(*code);
                output.extend_from_slice(&sym[..len as usize]);
            }
        }

        Ok(Bytes::from(output))
    }
}

impl<T: DataType> Decoder<T> for FSSTDecoder<T> {
    fn set_data(&mut self, data: Bytes, num_values: usize) -> Result<()> {
        match T::get_physical_type() {
            Type::BYTE_ARRAY | Type::FIXED_LEN_BYTE_ARRAY => {
                let (header, data) = fsst::read_fsst_header(data)?;
                self.header = Some(header);

                // decode all compressed lengths
                let mut len_decoder = DeltaBitPackDecoder::<Int32Type>::new();
                len_decoder.set_data(data.clone(), num_values)?;

                let num_lengths = len_decoder.values_left();
                self.compressed_lengths.resize(num_lengths, 0);
                len_decoder.get(&mut self.compressed_lengths[..])?;

                // initialize the decoder
                self.data = Some(data.slice(len_decoder.get_offset()..));
                self.data_cursor = 0;
                self.length_cursor = 0;
                self.num_values = num_lengths;

                Ok(())
            }
            _ => Err(general_err!(
                "FSSTDecoder only supports ByteArrayType and FixedLenByteArrayType"
            )),
        }
    }

    fn get(&mut self, buffer: &mut [<T as DataType>::T]) -> Result<usize> {
        match T::get_physical_type() {
            Type::BYTE_ARRAY | Type::FIXED_LEN_BYTE_ARRAY => {
                let num_values = buffer.len().min(self.num_values);

                for item in buffer.iter_mut().take(num_values) {
                    let compressed_len = self.compressed_lengths[self.length_cursor] as usize;
                    let decompressed = self.decode_string(compressed_len)?;
                    item.set_from_bytes(decompressed);

                    self.length_cursor += 1;
                }

                self.num_values -= num_values;
                Ok(num_values)
            }
            _ => Err(general_err!(
                "FSSTDecoder only supports ByteArrayType and FixedLenByteArrayType"
            )),
        }
    }

    fn values_left(&self) -> usize {
        // todo: wire this up properly
        self.num_values
    }

    fn encoding(&self) -> Encoding {
        Encoding::FSST
    }

    fn skip(&mut self, num_values: usize) -> Result<usize> {
        let mut buffer = vec![T::T::default(); num_values];
        self.get(&mut buffer)
    }
}

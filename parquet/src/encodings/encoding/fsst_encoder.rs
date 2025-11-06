use std::marker::PhantomData;

use crate::{
    basic::{Encoding, Type},
    data_type::{DataType, SliceAsBytes},
    encodings::encoding::Encoder,
};

pub struct FSSTEncoder<T> {
    buffer: Vec<u8>,
    _phantom: PhantomData<T>,
}

impl<T: DataType> FSSTEncoder<T> {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

impl<T: DataType> Encoder<T> for FSSTEncoder<T> {
    fn put(&mut self, values: &[<T as DataType>::T]) -> crate::errors::Result<()> {
        self.buffer
            .extend(<T as DataType>::T::slice_as_bytes(values));

        ensure_phys_ty!(
            Type::BYTE_ARRAY | Type::FIXED_LEN_BYTE_ARRAY,
            "FSSTEncoder does not support other types"
        );

        Ok(())
    }

    fn encoding(&self) -> crate::basic::Encoding {
        Encoding::FSST
    }

    fn estimated_data_encoded_size(&self) -> usize {
        self.buffer.len()
    }

    fn estimated_memory_size(&self) -> usize {
        self.buffer.capacity()
    }

    fn flush_buffer(&mut self) -> crate::errors::Result<bytes::Bytes> {
        todo!()
    }
}

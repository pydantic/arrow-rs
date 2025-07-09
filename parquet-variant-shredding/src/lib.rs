use arrow_schema::ArrowError;
use parquet_variant::Variant;

#[derive(Debug)]
pub enum ParquetPhysicalType<'v> {
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    Float(f32),
    Double(f64),
    ByteArray(&'v [u8]),
    Binary(&'v [u8]),

    // complex
    Array(ShreddedArray),
    Object(ShreddedObject),
}

#[derive(Debug)]
pub struct ShreddedArray {}

#[derive(Debug)]
pub struct ShreddedObject {}

#[derive(Debug)]
pub struct ShreddedVariant<'m, 'v> {
    metadata: &'m [u8],
    value: Option<&'v [u8]>,
    typed_value: Option<ParquetPhysicalType<'v>>,
}

impl<'m, 'v> ShreddedVariant<'m, 'v> {
    pub fn missing_value(&self) -> bool {
        self.value.is_none() && self.typed_value.is_none()
    }

    pub fn partially_shredded(&self) -> bool {
        self.value.is_some() && self.typed_value.is_some()
    }

    pub fn shred(metadata: Vec<u8>, value: Vec<u8>, schema: ()) -> Result<Self, ArrowError> {
        todo!();
    }

    /// Reconstruct a shredded variant
    ///
    /// Constructs a Variant from value and typed_value
    pub fn construct_variant(self) -> Result<Variant<'m, 'v>, ArrowError> {
        match self.typed_value {
            Some(typed_value) => match typed_value {
                ParquetPhysicalType::Object(obj) => {
                    unimplemented!()
                }
                ParquetPhysicalType::Array(array) => {
                    unimplemented!()
                }
                ParquetPhysicalType::Boolean(bool) => Ok(Variant::from(bool)),
                ParquetPhysicalType::Int32(num) => Ok(Variant::from(num)),
                ParquetPhysicalType::Int64(num) => Ok(Variant::from(num)),
                ParquetPhysicalType::Float(float) => Ok(Variant::from(float)),
                ParquetPhysicalType::Double(double) => Ok(Variant::from(double)),
                ParquetPhysicalType::ByteArray(items) => Ok(Variant::from(items)),
                ParquetPhysicalType::Binary(items) => Ok(Variant::from(items)),
            },
            None => match self.value {
                Some(value) => Variant::try_new(self.metadata, value),
                None => Err(ArrowError::InvalidArgumentError(
                    "value is missing".to_string(),
                )),
            },
        }
    }
}

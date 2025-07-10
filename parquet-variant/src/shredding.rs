use arrow_array::{ArrayRef, BinaryArray, StructArray};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields};
use std::{collections::HashSet, sync::Arc};

use crate::{ValueBuffer, Variant, VariantBuilder, VariantMetadata};

/// Shred the variant

fn create_sample_variant() -> StructArray {
    let metadata: Vec<&[u8]> = vec![&[
        0b0001_0001,
        3, // dictionary size
        0, // "active"
        6, // "age"
        9, // "name"
        13,
        b'a',
        b'c',
        b't',
        b'i',
        b'v',
        b'e',
        b'a',
        b'g',
        b'e',
        b'n',
        b'a',
        b'm',
        b'e',
    ]];

    let metadata = Arc::new(BinaryArray::from_vec(metadata));

    let value: Vec<&[u8]> = vec![&[
        0x02, // header: basic_type=2, value_header=0x00
        3,    // num_elements = 3
        // Field IDs (1 byte each): active=0, age=1, name=2
        0, 1, 2,
        // Field offsets (1 byte each): 4 offsets total
        0, // offset to first value (boolean true)
        1, // offset to second value (int8)
        3, // offset to third value (short string)
        9, // end offset
        // Values:
        0x04, // boolean true: primitive_header=1, basic_type=0 -> (1 << 2) | 0 = 0x04
        0x0C,
        42, // int8: primitive_header=3, basic_type=0 -> (3 << 2) | 0 = 0x0C, then value 42
        0x15, b'h', b'e', b'l', b'l',
        b'o', // short string: length=5, basic_type=1 -> (5 << 2) | 1 = 0x15
    ]];
    let value = Arc::new(BinaryArray::from_vec(value));

    StructArray::from(vec![
        (
            Arc::new(Field::new("metadata", DataType::Binary, false)),
            metadata.clone() as ArrayRef,
        ),
        (
            Arc::new(Field::new("value", DataType::Binary, false)),
            value.clone() as ArrayRef,
        ),
    ])
}

/*
Ideal struct array is:

StructArray {
    metadata: &[u8],
    value: &[u8]
}

    =>

StructArray {
    metadata: &[u8],
    value: Option<&'v [u8]>,
    typed_value: Option<ParquetPhysicalType<'v>>,
}
*/

pub fn shred(
    mut variant_struct_array: StructArray,
    shred_schema: Fields,
) -> Result<StructArray, ArrowError> {
    // first validate that metadata and value fields exist
    let metadata_column = variant_struct_array
        .column_by_name("metadata")
        .ok_or_else(|| {
            ArrowError::InvalidArgumentError(
                "variant struct array must contain a metadata column".to_string(),
            )
        })?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("shouldn't be a binary array");

    let value_column = variant_struct_array
        .column_by_name("value")
        .ok_or_else(|| {
            ArrowError::InvalidArgumentError(
                "variant struct array must contain a value column".to_string(),
            )
        })?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("shouldn't be a binary array");

    // iterate through the metadata
    // parse out VariantMetadata
    // and check if any of the fields in the shred_schmea exists

    // if not -> skip
    // else -> actually shred out the value
    //
    // the spec says i don't need to touch the metadata
    // so i just need to update the value

    let shred_field_names: HashSet<&str> =
        HashSet::from_iter(shred_schema.into_iter().map(|f| f.name().as_str()));

    for (i, metadata) in metadata_column.iter().enumerate() {
        if let Some(metadata) = metadata {
            let variant_metadata = VariantMetadata::try_new(metadata)?;

            let dictionary_keys = HashSet::from_iter(variant_metadata.iter());

            if shred_field_names.is_disjoint(&dictionary_keys) {
                continue;
            }

            let value = value_column.value(i);

            let variant = Variant::try_new_with_metadata(variant_metadata, value)?;
            let variant_object = variant.as_object().ok_or(ArrowError::InvalidArgumentError(
                "can't cast to variant object".to_string(),
            ))?;
        }
    }

    todo!();
}

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
    List(Vec<ShreddedElement<'v>>),
    Object(Vec<ShreddedField<'v>>),
}

#[derive(Debug)]
pub struct ShreddedField<'v> {
    key: &'v str,
    value: ShreddedElement<'v>,
}

#[derive(Debug)]
pub struct ShreddedElement<'v> {
    value: Option<&'v [u8]>,
    typed_value: Option<&'v str>,
}

pub fn construct_variant<'v>(
    metadata: &[u8],
    value: Option<&'v [u8]>,
    typed_value: Option<ParquetPhysicalType<'v>>,
) -> Result<(Vec<u8>, Vec<u8>), ArrowError> {
    if let Some(typed_value) = typed_value {
        let mut value_buffer = ValueBuffer::default();

        match typed_value {
            ParquetPhysicalType::List(shredded_elements) => {
                if value.is_some() {
                    return Err(ArrowError::InvalidArgumentError(
                        "shredded array must not conflict with variant value".to_string(),
                    ));
                }

                let mut variant_builder = VariantBuilder::new();

                let mut list_builder = variant_builder.new_list();

                for ShreddedElement { value, typed_value } in shredded_elements {
                    let typed_value =
                        typed_value.map(|v| ParquetPhysicalType::ByteArray(v.as_bytes()));

                    let (m, v) = construct_variant(metadata, value, typed_value)?;

                    list_builder.append_value(Variant::try_new(&m, &v)?);
                }

                list_builder.finish();

                return Ok(variant_builder.finish());
            }
            ParquetPhysicalType::Object(shredded_fields) => {
                let mut shredded_object_keys = HashSet::with_capacity(shredded_fields.len());

                let mut variant_builder = VariantBuilder::new();
                let mut object_builder = variant_builder.new_object();

                for ShreddedField {
                    key,
                    value: ShreddedElement { value, typed_value },
                } in shredded_fields
                {
                    shredded_object_keys.insert(key);

                    let typed_value =
                        typed_value.map(|v| ParquetPhysicalType::ByteArray(v.as_bytes()));
                    let (m, v) = construct_variant(metadata, value, typed_value)?;

                    object_builder.insert(key, Variant::try_new(&m, &v)?);
                }

                if let Some(value) = value {
                    let partial_object = Variant::try_new(metadata, value)?;

                    let partial_object = match partial_object.as_object() {
                        None => {
                            return Err(ArrowError::InvalidArgumentError(
                                "partially shredded value must be an object".to_string(),
                            ))
                        }
                        Some(partial_object) => partial_object,
                    };

                    let partial_object_keys: HashSet<&str> =
                        HashSet::from_iter(partial_object.iter().map(|(k, _)| k));

                    if !shredded_object_keys.is_disjoint(&partial_object_keys) {
                        return Err(ArrowError::InvalidArgumentError(
                            "object keys must be disjoint".to_string(),
                        ));
                    }

                    for (key, value) in partial_object.iter() {
                        object_builder.insert(key, value);
                    }
                }

                object_builder.finish()?;

                return Ok(variant_builder.finish());
            }
            ParquetPhysicalType::Boolean(b) => {
                value_buffer.append_non_nested_value(b);
            }
            ParquetPhysicalType::Int32(n) => {
                value_buffer.append_non_nested_value(n);
            }
            ParquetPhysicalType::Int64(n) => {
                value_buffer.append_non_nested_value(n);
            }
            ParquetPhysicalType::Float(d) => {
                value_buffer.append_non_nested_value(d);
            }
            ParquetPhysicalType::Double(d) => {
                value_buffer.append_non_nested_value(d);
            }
            ParquetPhysicalType::ByteArray(items) => {
                value_buffer.append_non_nested_value(items);
            }
            ParquetPhysicalType::Binary(items) => {
                value_buffer.append_non_nested_value(items);
            }
        };

        return Ok((vec![], value_buffer.into_inner()));
    }

    if let Some(value) = value {
        return Ok((metadata.to_vec(), value.to_vec()));
    }

    Err(ArrowError::InvalidArgumentError(
        "Value is missing".to_string(),
    ))
}

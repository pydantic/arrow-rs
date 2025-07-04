use arrow_schema::ArrowError;
use parquet_variant::{json_to_variant, variant_to_json_string, Variant, VariantBuilder};

struct JsonToVariantTest<'a> {
    json: &'a str,
    expected: Variant<'a, 'a>,
}

impl<'a> JsonToVariantTest<'a> {
    fn run(self) -> Result<(), ArrowError> {
        let mut variant_builder = VariantBuilder::new();
        json_to_variant(self.json, &mut variant_builder)?;
        let (metadata, value) = variant_builder.finish();
        let variant = Variant::try_new(&metadata, &value)?;
        assert_eq!(variant, self.expected);
        Ok(())
    }
}

fn main() {
    // 256 elements (keys: 000-255) - each element is an object of 256 elements (240-495) - each
    // element a list of numbers from 0-127
    let keys: Vec<String> = (0..=255).map(|n| format!("{n:03}")).collect();
    let innermost_list: String = format!(
        "[{}]",
        (0..=127)
            .map(|n| format!("{n}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let inner_keys: Vec<String> = (240..=495).map(|n| format!("{n}")).collect();
    let inner_object = format!(
        "{{{}:{}}}",
        inner_keys
            .iter()
            .map(|k| format!("\"{k}\""))
            .collect::<Vec<String>>()
            .join(format!(":{innermost_list},").as_str()),
        innermost_list
    );
    let json = format!(
        "{{{}:{}}}",
        keys.iter()
            .map(|k| format!("\"{k}\""))
            .collect::<Vec<String>>()
            .join(format!(":{inner_object},").as_str()),
        inner_object
    );
    // Manually verify raw JSON value size
    let mut variant_builder = VariantBuilder::new();
    json_to_variant(&json, &mut variant_builder).unwrap();
    let (metadata, value) = variant_builder.finish();
    let v = parquet_variant::Variant::try_new(&metadata, &value).unwrap();
    let output_string = variant_to_json_string(&v).unwrap();
    assert_eq!(output_string, json);
    // Verify metadata size = 1 + 2 + 2 * 497 + 3 * 496
    assert_eq!(metadata.len(), 2485);
    // Verify value size.
    // Size of innermost_list: 1 + 1 + 258 + 256 = 516
    // Size of inner object: 1 + 4 + 256 + 257 * 3 + 256 * 516 = 133128
    // Size of json: 1 + 4 + 512 + 1028 + 256 * 133128 = 34082313
    assert_eq!(value.len(), 34082313);

    let mut variant_builder = VariantBuilder::new();
    let mut object_builder = variant_builder.new_object();
    keys.iter().for_each(|key| {
        let mut inner_object_builder = object_builder.new_object(key);
        inner_keys.iter().for_each(|inner_key| {
            let mut list_builder = inner_object_builder.new_list(inner_key);
            for i in 0..=127 {
                list_builder.append_value(Variant::Int8(i));
            }
            list_builder.finish();
        });
        inner_object_builder.finish().unwrap();
    });
    object_builder.finish().unwrap();
    let (metadata, value) = variant_builder.finish();
    let variant = Variant::try_new(&metadata, &value).unwrap();

    JsonToVariantTest {
        json: &json,
        expected: variant,
    }
    .run();
}

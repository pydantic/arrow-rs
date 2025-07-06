use parquet_variant::{json_to_variant, Variant, VariantBuilder};

fn generate_large_object() -> (Vec<u8>, Vec<u8>) {
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
    variant_builder.finish()
}

fn main() {
    let (m, v) = generate_large_object();
    std::hint::black_box(Variant::try_new(&m, &v).unwrap());
}

use parquet_variant::{Variant, VariantBuilder};

fn test_variant_object_with_count(count: i32) {
    let mut builder = VariantBuilder::new();
    let mut obj = builder.new_object();
    for val in 0..count {
        let key = format!("id_{}", val);
        obj.insert(&key, val);
    }

    obj.finish().unwrap();
    let (metadata, value) = builder.finish();
    let variant = Variant::try_new(&metadata, &value).unwrap();
}

fn main() {
    test_variant_object_with_count(2_i32.pow(24) + 1);
}

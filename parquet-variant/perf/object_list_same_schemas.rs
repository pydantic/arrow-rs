use std::{hint, ops::Range};

use parquet_variant::VariantBuilder;
use rand::{
    distr::{uniform::SampleUniform, Alphanumeric},
    rngs::ThreadRng,
    Rng,
};

fn random<T: SampleUniform + PartialEq + PartialOrd>(rng: &mut ThreadRng, range: Range<T>) -> T {
    rng.random_range::<T, _>(range)
}

// generates a string with a 50/50 chance whether it's a short or a long string
fn random_string(rng: &mut ThreadRng) -> String {
    let len = rng.random_range::<usize, _>(1..128);

    rng.sample_iter(&Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

// generates a string guaranteed to be longer than 64 bytes
fn random_long_string(rng: &mut ThreadRng) -> String {
    let len = rng.random_range::<usize, _>(65..200);

    rng.sample_iter(&Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

fn main() {
    let mut rng = rand::rng();

    let mut variant = VariantBuilder::new();

    let mut list_builder = variant.new_list();

    for _ in 0..10_000 {
        let mut object_builder = list_builder.new_object();
        object_builder.insert("name", random_string(&mut rng).as_str());
        object_builder.insert("age", random::<u32>(&mut rng, 18..100) as i32);
        object_builder.insert("likes_cilantro", rng.random_bool(0.5));
        object_builder.insert("comments", random_long_string(&mut rng).as_str());

        let mut list_builder = object_builder.new_list("dishes");
        list_builder.append_value(random_string(&mut rng).as_str());
        list_builder.append_value(random_string(&mut rng).as_str());
        list_builder.append_value(random_string(&mut rng).as_str());

        list_builder.finish();
        object_builder.finish();
    }

    list_builder.finish();
    hint::black_box(variant.finish());
}

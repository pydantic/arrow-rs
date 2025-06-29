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

fn main() {
    let mut rng = rand::rng();

    let mut variant = VariantBuilder::new();

    let mut list_builder = variant.new_list();

    for _ in 0..1_000 {
        let mut object_builder = list_builder.new_object();

        for _num_fields in 0..random::<u8>(&mut rng, 0..100) {
            if rng.random_bool(0.33) {
                object_builder.insert(
                    random_string(&mut rng).as_str(),
                    random_string(&mut rng).as_str(),
                );
                continue;
            }

            if rng.random_bool(0.33) {
                let mut inner_object_builder = object_builder.new_object("rand_object");

                for _num_fields in 0..random::<u8>(&mut rng, 0..25) {
                    inner_object_builder.insert(
                        random_string(&mut rng).as_str(),
                        random_string(&mut rng).as_str(),
                    );
                }
                inner_object_builder.finish();

                continue;
            }

            let mut inner_list_builder = object_builder.new_list("rand_list");

            for _num_elements in 0..random::<u8>(&mut rng, 0..25) {
                inner_list_builder.append_value(random_string(&mut rng).as_str());
            }

            inner_list_builder.finish();
        }
        object_builder.finish();
    }

    list_builder.finish();
    hint::black_box(variant.finish());
}

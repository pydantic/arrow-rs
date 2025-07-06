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

extern crate parquet_variant;

use criterion::*;

use parquet_variant::{json_to_variant, ObjectBuilder, Variant, VariantBuilder};
use rand::{distr::Alphanumeric, rngs::StdRng, Rng, SeedableRng};

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

// Generates a large object and performs full validation
fn bench_validate_large_object(c: &mut Criterion) {
    c.bench_function("bench_validate_large_object", |b| {
        let (m, v) = generate_large_object();

        b.iter(|| {
            std::hint::black_box(Variant::try_new(&m, &v).unwrap());
        })
    });
}

criterion_group!(benches, bench_validate_large_object);

criterion_main!(benches);

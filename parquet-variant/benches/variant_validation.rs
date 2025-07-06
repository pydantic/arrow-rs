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

use parquet_variant::{Variant, VariantBuilder};
use rand::{distr::Alphanumeric, rngs::StdRng, Rng, SeedableRng};

// generates a string with a 50/50 chance whether it's a short or a long string
fn random_string(rng: &mut StdRng) -> String {
    let len = rng.random_range::<usize, _>(1..128);

    rng.sample_iter(&Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

struct RandomStringGenerator {
    cursor: usize,
    table: Vec<String>,
}

impl RandomStringGenerator {
    pub fn new(rng: &mut StdRng, capacity: usize) -> Self {
        let table = (0..capacity)
            .map(|_| random_string(rng))
            .collect::<Vec<_>>();

        Self { cursor: 0, table }
    }

    pub fn next(&mut self) -> &str {
        let this = &self.table[self.cursor];

        self.cursor = (self.cursor + 1) % self.table.len();

        this
    }
}

// generates a large object variant
// contains 1000 fields
fn build_large_object(rng: &mut StdRng) -> (Vec<u8>, Vec<u8>) {
    let mut string_table = RandomStringGenerator::new(rng, 1001);

    let mut builder = VariantBuilder::new();
    {
        let mut obj = builder.new_object();
        for _ in 0..1_000 {
            let k = string_table.next();
            obj.insert(k, k);
        }
        obj.finish().unwrap();
    }

    builder.finish()
}

// Generates a large object and performs full validation
fn bench_validate_large_object(c: &mut Criterion) {
    c.bench_function("bench_validate_large_object", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        let (m, v) = build_large_object(&mut rng);

        b.iter(|| {
            std::hint::black_box(Variant::try_new(&m, &v).unwrap());
        })
    });
}

criterion_group!(benches, bench_validate_large_object);

criterion_main!(benches);

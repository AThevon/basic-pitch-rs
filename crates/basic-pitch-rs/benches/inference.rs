// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! The network alone: one 2 s window, and loading the model.
//!
//! ```text
//! cargo bench -p basic-pitch-rs --bench inference
//! cargo bench -p basic-pitch-rs --bench inference --features ort-download
//! ```

use std::hint::black_box;

use basic_pitch::constants::AUDIO_N_SAMPLES;
use basic_pitch::model::{Backend, TractBackend};
use basic_pitch::window::fill_window;
use criterion::{Criterion, criterion_group, criterion_main};

mod common;

fn window() -> Vec<f32> {
    let audio = common::audio("vocadito_14");
    let mut window = vec![0.0; AUDIO_N_SAMPLES];
    fill_window(&audio, 1, &mut window).expect("vocadito_14 has several windows");
    window
}

fn bench_window(c: &mut Criterion) {
    let window = window();
    let mut group = c.benchmark_group("window");

    let tract = TractBackend::new().expect("bundled model");
    group.bench_function("tract", |b| {
        b.iter(|| tract.predict_window(black_box(&window)).expect("inference"));
    });

    #[cfg(feature = "ort")]
    {
        let ort = basic_pitch::model::OrtBackend::new().expect("bundled model");
        group.bench_function("ort", |b| {
            b.iter(|| ort.predict_window(black_box(&window)).expect("inference"));
        });
    }
    group.finish();
}

fn bench_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load");
    group.sample_size(20);
    group.bench_function("tract", |b| {
        b.iter(|| TractBackend::new().expect("bundled model"));
    });
    #[cfg(feature = "ort")]
    group.bench_function("ort", |b| {
        b.iter(|| basic_pitch::model::OrtBackend::new().expect("bundled model"));
    });
    group.finish();
}

criterion_group!(benches, bench_window, bench_load);
criterion_main!(benches);

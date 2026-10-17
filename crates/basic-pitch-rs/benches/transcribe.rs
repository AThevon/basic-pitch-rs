// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! The whole pipeline on one minute of audio (37 windows), from samples to notes, with
//! a model loaded beforehand.
//!
//! ```text
//! cargo bench -p basic-pitch-rs --bench transcribe
//! cargo bench -p basic-pitch-rs --bench transcribe --features ort-download
//! ```

use std::hint::black_box;
use std::time::Duration;

use basic_pitch::BasicPitch;
use criterion::{Criterion, criterion_group, criterion_main};

mod common;

fn bench_transcribe(c: &mut Criterion) {
    let audio = common::one_minute();
    let mut group = c.benchmark_group("transcribe 60 s");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let tract = BasicPitch::new().expect("bundled model");
    group.bench_function("tract", |b| {
        b.iter(|| tract.transcribe(black_box(&audio)).expect("transcription"));
    });

    let cores = std::thread::available_parallelism().unwrap_or(std::num::NonZeroUsize::MIN);
    let tract_threads = BasicPitch::new()
        .expect("bundled model")
        .with_threads(cores);
    group.bench_function(format!("tract, {cores} threads"), |b| {
        b.iter(|| {
            tract_threads
                .transcribe(black_box(&audio))
                .expect("transcription")
        });
    });

    #[cfg(feature = "ort")]
    {
        let backend = basic_pitch::model::OrtBackend::new().expect("bundled model");
        let ort = BasicPitch::with_backend(backend);
        group.bench_function("ort", |b| {
            b.iter(|| ort.transcribe(black_box(&audio)).expect("transcription"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_transcribe);
criterion_main!(benches);

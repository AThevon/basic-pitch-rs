// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Note decoding alone, from the posteriorgrams Python computed for `vocadito_14`
//! (12.2 s of singing, 1049 frames).
//!
//! ```text
//! cargo bench -p basic-pitch-rs --bench postprocessing
//! ```

use std::hint::black_box;

use basic_pitch::NoteParams;
use basic_pitch::notes::model_output_to_notes;
use criterion::{Criterion, criterion_group, criterion_main};

mod common;

fn bench_postprocessing(c: &mut Criterion) {
    let posteriorgrams = common::posteriorgrams("vocadito_14");
    let mut group = c.benchmark_group("postprocessing");

    let defaults = NoteParams::default();
    group.bench_function("notes", |b| {
        b.iter(|| model_output_to_notes(black_box(&posteriorgrams), &defaults).expect("notes"));
    });

    let no_bends = NoteParams {
        include_pitch_bends: false,
        ..NoteParams::default()
    };
    group.bench_function("notes without pitch bends", |b| {
        b.iter(|| model_output_to_notes(black_box(&posteriorgrams), &no_bends).expect("notes"));
    });

    #[cfg(feature = "midi")]
    {
        let midi = basic_pitch::MidiParams::default();
        group.bench_function("notes and MIDI file", |b| {
            b.iter(|| {
                let notes =
                    model_output_to_notes(black_box(&posteriorgrams), &defaults).expect("notes");
                basic_pitch::midi::note_events_to_midi_bytes(&notes, &midi).expect("MIDI")
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_postprocessing);
criterion_main!(benches);

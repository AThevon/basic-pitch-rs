// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! The pipeline step by step: windows, raw model outputs, posteriorgrams, then notes
//! decoded twice with different settings, without running the model again.
//!
//! ```text
//! cargo run --release --example posteriorgrams
//! ```
//!
//! The input is synthesized in code: an A minor arpeggio played with a harmonic tone,
//! then a slow glide from E4 to G4.

use std::error::Error;
use std::f32::consts::TAU;

use basic_pitch::constants::{
    ANNOT_N_FRAMES, AUDIO_SAMPLE_RATE, CONTOURS_BINS_PER_SEMITONE, MIDI_OFFSET,
};
use basic_pitch::model::{self, TractBackend};
use basic_pitch::notes::{self, NoteParams};
use basic_pitch::window::{self, TrimMode};
use basic_pitch::{Posteriorgram, Posteriorgrams};

const SR: f32 = AUDIO_SAMPLE_RATE as f32;

fn main() -> Result<(), Box<dyn Error>> {
    let audio = synthesize();
    println!(
        "input: {:.2} s at {} Hz",
        audio.len() as f32 / SR,
        AUDIO_SAMPLE_RATE
    );

    // 1. The audio is cut into overlapping windows of 43 844 samples.
    let n_windows = window::window_count(audio.len());
    println!("{n_windows} model windows of {ANNOT_N_FRAMES} frames each");

    // 2. The backend runs the network on each window. `run_windows` returns the raw,
    //    overlapping outputs; `run_inference` also stitches and trims them.
    let backend = TractBackend::new()?;
    let raw = model::run_windows(&backend, &audio)?;
    let output: Posteriorgrams = model::run_inference(&backend, &audio, TrimMode::Main)?;
    println!(
        "raw output: {} frames, unwrapped: {} frames ({:.1} frames/s)\n",
        raw.n_frames(),
        output.n_frames(),
        output.n_frames() as f32 / (audio.len() as f32 / SR)
    );

    // 3. The posteriorgrams are plain row-major matrices: one row per frame.
    println!("note activations over time (rows: pitches, one column every 5 frames):");
    print_piano_roll(&output.note, 5);

    // The contour has 3 bins per semitone. It peaks one bin above the true pitch
    // (spotify/basic-pitch#87), hence the extra third of a semitone.
    println!("\nstrongest pitch, from the 1/3-semitone contour output:");
    for frame in (0..output.n_frames()).step_by(output.n_frames() / 8) {
        let row = output.contour.row(frame).expect("frame in range");
        let (bin, activation) = argmax(row);
        let time = notes::frame_to_time(frame);
        if activation < 0.3 {
            println!("  {time:5.2} s  no clear pitch");
            continue;
        }
        let midi = f32::from(MIDI_OFFSET) + bin as f32 / CONTOURS_BINS_PER_SEMITONE as f32;
        println!("  {time:5.2} s  MIDI {midi:5.1}  (activation {activation:.2})");
    }

    // 4. Notes are decoded from the posteriorgrams, as often as needed.
    for (label, params) in [
        ("default settings", NoteParams::default()),
        (
            "stricter onsets, no melodia trick, bends centered on zero",
            NoteParams {
                onset_threshold: 0.7,
                melodia_trick: false,
                bend_offset: -1,
                ..NoteParams::default()
            },
        ),
    ] {
        let mut decoded = notes::model_output_to_notes(&output, &params)?;
        decoded.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
        println!("\n{} notes with {label}:", decoded.len());
        for note in &decoded {
            let bends = note.pitch_bends.as_deref().unwrap_or_default();
            println!(
                "  {:5.2} s to {:5.2} s  MIDI {:3}  velocity {:3}  bends {:+} to {:+}",
                note.start_s,
                note.end_s,
                note.pitch,
                note.velocity(),
                bends.iter().min().unwrap_or(&0),
                bends.iter().max().unwrap_or(&0),
            );
        }
    }
    Ok(())
}

/// A harmonic tone with an exponential decay, like a plucked string.
fn tone(out: &mut [f32], start_s: f32, duration_s: f32, freq_hz: impl Fn(f32) -> f32) {
    let start = (start_s * SR) as usize;
    let len = ((duration_s * SR) as usize).min(out.len().saturating_sub(start));
    let mut phase = 0.0_f32;
    for (i, sample) in out[start..start + len].iter_mut().enumerate() {
        let t = i as f32 / SR;
        phase += TAU * freq_hz(t) / SR;
        let envelope = (-3.0 * t).exp() * (1.0 - (-200.0 * t).exp());
        let harmonics: f32 = (1..=5_u8)
            .map(|k| (f32::from(k) * phase).sin() / f32::from(k * k))
            .sum();
        *sample += 0.15 * envelope * harmonics;
    }
}

fn synthesize() -> Vec<f32> {
    let midi_hz = |midi: f32| 440.0 * 2.0_f32.powf((midi - 69.0) / 12.0);
    let mut audio = vec![0.0; (4.5 * SR) as usize];
    for (k, midi) in [57.0, 60.0, 64.0, 69.0].into_iter().enumerate() {
        tone(&mut audio, 0.4 * k as f32, 1.2, |_| midi_hz(midi));
    }
    // E4 to G4 over 1.5 s.
    tone(&mut audio, 2.4, 2.0, |t| {
        midi_hz(64.0 + 3.0 * (t / 1.5).min(1.0))
    });
    audio
}

fn argmax(row: &[f32]) -> (usize, f32) {
    row.iter().copied().enumerate().fold(
        (0, f32::MIN),
        |best, (i, x)| if x > best.1 { (i, x) } else { best },
    )
}

/// Prints the pitches that are active somewhere, one line each, from high to low.
fn print_piano_roll(note: &Posteriorgram, frames_per_column: usize) {
    let shades = [' ', '.', ':', '+', '#'];
    let active: Vec<usize> = (0..note.n_bins())
        .filter(|&bin| note.rows().any(|row| row[bin] > 0.3))
        .collect();
    for &bin in active.iter().rev() {
        let line: String = note
            .as_slice()
            .chunks_exact(note.n_bins() * frames_per_column)
            .map(|chunk| {
                let peak = chunk
                    .chunks_exact(note.n_bins())
                    .map(|row| row[bin])
                    .fold(0.0_f32, f32::max);
                shades[((peak * shades.len() as f32) as usize).min(shades.len() - 1)]
            })
            .collect();
        println!("  MIDI {:3} |{line}|", bin + usize::from(MIDI_OFFSET));
    }
}

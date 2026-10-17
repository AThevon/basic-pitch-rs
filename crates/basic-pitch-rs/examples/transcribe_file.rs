// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Transcribes a WAV file to notes and, optionally, to a MIDI file.
//!
//! ```text
//! cargo run --release --example transcribe_file -- recording.wav [notes.mid]
//! ```
//!
//! Without arguments, it transcribes `tests/fixtures/audio/pluck.wav`, a synthetic
//! guitar-like recording (48 kHz, stereo).
//!
//! The library takes mono samples at 22 050 Hz, so this example decodes the WAV file
//! with `hound`, averages the channels and resamples with a short windowed-sinc filter.
//! That is good enough for a demonstration, but the notes can differ slightly from Basic
//! Pitch's, which resamples with soxr. For other formats and exact parity, use the
//! `basic-pitch-rs` command-line tool.

use std::error::Error;
use std::f64::consts::PI;
use std::path::PathBuf;
use std::time::Instant;

use basic_pitch::constants::AUDIO_SAMPLE_RATE;
use basic_pitch::{BasicPitch, MidiParams};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let input = args.next().map_or_else(
        || {
            let fixture = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/fixtures/audio/pluck.wav"
            );
            PathBuf::from(fixture)
                .canonicalize()
                .unwrap_or_else(|_| fixture.into())
        },
        PathBuf::from,
    );
    let midi_output = args.next().map(PathBuf::from);

    let (audio, source_rate, channels) = read_wav(&input)?;
    println!(
        "{}: {:.1} s, {source_rate} Hz, {channels} channel(s)",
        input.display(),
        audio.len() as f64 / f64::from(source_rate)
    );
    let audio = resample(&audio, source_rate, AUDIO_SAMPLE_RATE);

    let start = Instant::now();
    let transcriber = BasicPitch::new()?;
    let loaded = start.elapsed();
    let transcription = transcriber.transcribe(&audio)?;
    println!(
        "model loaded in {loaded:.0?}, transcribed in {:.0?}\n",
        start.elapsed() - loaded
    );

    println!("  start      end  note  velocity  pitch bend (semitones)");
    for note in transcription.sorted_notes() {
        let bend = note
            .pitch_bends
            .as_deref()
            .map_or_else(String::new, |bends| {
                // Bends are in thirds of a semitone. Basic Pitch reports +1/3 on in-tune
                // notes (see NoteParams::bend_offset), so the range is the useful part.
                let min = bends.iter().min().copied().unwrap_or(0);
                let max = bends.iter().max().copied().unwrap_or(0);
                format!(
                    "{:+.2} to {:+.2}",
                    f64::from(min) / 3.0,
                    f64::from(max) / 3.0
                )
            });
        println!(
            "{:7.3}  {:7.3}  {:>4}  {:8}  {bend}",
            note.start_s,
            note.end_s,
            note_name(note.pitch),
            note.velocity(),
        );
    }
    println!("\n{} notes", transcription.notes.len());

    if let Some(path) = midi_output {
        std::fs::write(&path, transcription.to_midi(&MidiParams::default())?)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

/// Reads a WAV file and returns its samples averaged to mono, its sample rate and its
/// number of channels. Integer samples are scaled to [-1, 1) like libsndfile does.
fn read_wav(path: &PathBuf) -> Result<(Vec<f32>, u32, u16), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / f64::from(1_u32 << (spec.bits_per_sample - 1));
            reader
                .samples::<i32>()
                .map(|s| s.map(|s| (f64::from(s) * scale) as f32))
                .collect::<Result<_, _>>()?
        }
    };
    let channels = usize::from(spec.channels);
    let mono = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, spec.sample_rate, spec.channels))
}

/// Resamples with a Hann-windowed sinc filter (16 zero crossings), with its cutoff a
/// little below the lower Nyquist frequency.
fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    const ZERO_CROSSINGS: f64 = 16.0;
    let ratio = f64::from(to) / f64::from(from);
    let cutoff = 0.95 * ratio.min(1.0); // in cycles per input sample, times 2
    let half_width = ZERO_CROSSINGS / cutoff; // in input samples
    let out_len = (input.len() as f64 * ratio).ceil() as usize;
    (0..out_len)
        .map(|i| {
            let center = i as f64 / ratio;
            let first = (center - half_width).ceil().max(0.0) as usize;
            let last = ((center + half_width).floor() as usize).min(input.len() - 1);
            let mut acc = 0.0;
            for (j, &x) in input.iter().enumerate().take(last + 1).skip(first) {
                let t = j as f64 - center;
                let window = 0.5 + 0.5 * (PI * t / half_width).cos();
                acc += f64::from(x) * cutoff * sinc(cutoff * t) * window;
            }
            acc as f32
        })
        .collect()
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// The name of a MIDI note, such as `A4` for 69.
fn note_name(pitch: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!(
        "{}{}",
        NAMES[usize::from(pitch % 12)],
        i32::from(pitch / 12) - 1
    )
}

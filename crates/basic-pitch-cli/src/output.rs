// Copyright 2022 Spotify AB (original: basic_pitch/inference.py)
// Copyright 2026 Adrien Thevon (Rust port, modified)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Ported from spotify/basic-pitch v0.4.0. Changes: the CSV writer reproduces
// `save_note_events` in Rust; the JSON document and the NPZ layout are new.

//! Output formats: the JSON document, the CSV note list and the NPZ posteriorgrams.
//!
//! MIDI files come straight from [`basic_pitch::Transcription::to_midi`].

use std::io::{self, Write};

use basic_pitch::constants::CONTOURS_BINS_PER_SEMITONE;
use basic_pitch::{NoteEvent, Posteriorgram, Posteriorgrams};
use serde::{Deserialize, Serialize};

use crate::npz;

/// Version of the JSON document layout, in its `schema_version` field.
///
/// It changes only when a field is removed, renamed or changes meaning. New optional
/// fields can be added within a version: consumers should ignore unknown fields.
pub const JSON_SCHEMA_VERSION: u32 = 1;

/// URL of the JSON Schema that describes [`JsonDocument`], in its `$schema` field.
pub const JSON_SCHEMA_URL: &str = "https://raw.githubusercontent.com/AThevon/basic-pitch-rs/main/crates/basic-pitch-cli/schema/transcription.v1.schema.json";

/// The JSON Schema of the document, as shipped in `schema/`.
pub const JSON_SCHEMA: &str = include_str!("../schema/transcription.v1.schema.json");

/// The JSON output of `basic-pitch-rs --json`.
///
/// The layout is described by the JSON Schema in
/// `crates/basic-pitch-cli/schema/transcription.v1.schema.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonDocument {
    /// [`JSON_SCHEMA_URL`].
    #[serde(rename = "$schema")]
    pub schema: String,
    /// [`JSON_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// The program that wrote the document.
    pub generator: Generator,
    /// The model and how it was run.
    pub model: ModelInfo,
    /// The input file.
    pub input: InputInfo,
    /// The audio as analyzed by the model.
    pub analysis: AnalysisInfo,
    /// The parameters of the note decoding and of the MIDI export.
    pub params: Params,
    /// The notes, sorted by start time, then pitch.
    pub notes: Vec<JsonNote>,
}

/// The program that wrote a [`JsonDocument`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Generator {
    /// Always `basic-pitch-rs`.
    pub name: String,
    /// Version of basic-pitch-rs.
    pub version: String,
}

/// The model of a [`JsonDocument`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInfo {
    /// Always `basic-pitch-icassp-2022`, the model of spotify/basic-pitch.
    pub name: String,
    /// SHA-256 of the ONNX file, in lowercase hexadecimal.
    pub sha256: String,
    /// The inference engine: `tract` or `ort`.
    pub backend: String,
}

/// The input of a [`JsonDocument`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputInfo {
    /// The path as given on the command line.
    pub path: String,
    /// Sample rate of the file, in Hz.
    pub sample_rate: u32,
    /// Number of channels of the file.
    pub channels: usize,
    /// Number of frames (samples per channel) of the file.
    pub frames: usize,
    /// Duration of the file, in seconds.
    pub duration_s: f64,
    /// Short name of the audio codec, as reported by the decoder.
    pub codec: String,
}

/// The analysis of a [`JsonDocument`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisInfo {
    /// Always 22050: the model's sample rate.
    pub sample_rate: u32,
    /// Number of mono samples given to the model.
    pub samples: usize,
    /// The resampler used, or `null` when the file was already at 22 050 Hz.
    pub resampler: Option<String>,
    /// How the model output is trimmed: `main` or `v0.4.0`.
    pub trim_mode: String,
    /// Number of frames of the posteriorgrams.
    pub frames: usize,
}

/// The parameters of a [`JsonDocument`], with the names of the command-line flags.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "mirrors the command-line flags"
)]
pub struct Params {
    /// `--onset-threshold`.
    pub onset_threshold: f64,
    /// `--frame-threshold`.
    pub frame_threshold: f64,
    /// `--minimum-note-length`, in milliseconds.
    pub minimum_note_length_ms: f64,
    /// `--minimum-frequency`, in Hz, or `null`.
    pub minimum_frequency: Option<f64>,
    /// `--maximum-frequency`, in Hz, or `null`.
    pub maximum_frequency: Option<f64>,
    /// `false` with `--no-infer-onsets`.
    pub infer_onsets: bool,
    /// `false` with `--no-melodia`.
    pub melodia_trick: bool,
    /// `false` with `--no-pitch-bends`.
    pub include_pitch_bends: bool,
    /// `--bend-offset`, in thirds of a semitone.
    pub bend_offset: i32,
    /// `--multiple-pitch-bends` (MIDI export only).
    pub multiple_pitch_bends: bool,
    /// `--midi-tempo`, in beats per minute (MIDI export only).
    pub midi_tempo: f64,
}

/// A note of a [`JsonDocument`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonNote {
    /// Start time, in seconds.
    pub start_s: f64,
    /// End time, in seconds.
    pub end_s: f64,
    /// First frame of the note.
    pub start_frame: usize,
    /// Frame after the note (exclusive).
    pub end_frame: usize,
    /// MIDI note number, 21 (A0) to 108 (C8).
    pub pitch: u8,
    /// Scientific pitch notation with sharps, such as `C#4` (MIDI 61).
    pub note_name: String,
    /// Mean note activation over the note, in `[0, 1]`.
    pub amplitude: f32,
    /// MIDI velocity: `round(127 * amplitude)`.
    pub velocity: u8,
    /// One pitch bend per frame of the note, in thirds of a semitone relative to
    /// `pitch`, or `null` with `--no-pitch-bends`. With the default `--bend-offset 0`,
    /// in-tune notes read `+1`, as in Python (spotify/basic-pitch#87).
    #[serde(serialize_with = "inline")]
    pub pitch_bends: Option<Vec<i32>>,
    /// The same pitch bends in semitones (`pitch_bends / 3`).
    #[serde(serialize_with = "inline")]
    pub pitch_bends_semitones: Option<Vec<f64>>,
}

impl From<&NoteEvent> for JsonNote {
    fn from(note: &NoteEvent) -> Self {
        #[allow(clippy::cast_precision_loss, reason = "a third of a semitone count")]
        let semitones = |bends: &Vec<i32>| {
            bends
                .iter()
                .map(|&b| f64::from(b) / CONTOURS_BINS_PER_SEMITONE as f64)
                .collect()
        };
        Self {
            start_s: note.start_s,
            end_s: note.end_s,
            start_frame: note.start_frame,
            end_frame: note.end_frame,
            pitch: note.pitch,
            note_name: note_name(note.pitch),
            amplitude: note.amplitude,
            velocity: note.velocity(),
            pitch_bends: note.pitch_bends.clone(),
            pitch_bends_semitones: note.pitch_bends.as_ref().map(semitones),
        }
    }
}

/// Serializes a value as compact JSON, so that long arrays of numbers stay on one line
/// in the pretty-printed document.
fn inline<T: Serialize, S: serde::Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::Error as _;
    let compact = serde_json::to_string(value).map_err(S::Error::custom)?;
    serde_json::value::RawValue::from_string(compact)
        .map_err(S::Error::custom)?
        .serialize(serializer)
}

/// Writes `doc` as pretty-printed JSON, followed by a newline.
///
/// # Errors
///
/// Any error of `out`.
pub fn write_json(doc: &JsonDocument, mut out: impl Write) -> io::Result<()> {
    serde_json::to_writer_pretty(&mut out, doc)?;
    out.write_all(b"\n")?;
    out.flush()
}

/// Name of a MIDI note in scientific pitch notation, with sharps: 60 is `C4`, 69 is `A4`.
///
/// ```
/// assert_eq!(basic_pitch_cli::output::note_name(21), "A0");
/// assert_eq!(basic_pitch_cli::output::note_name(61), "C#4");
/// assert_eq!(basic_pitch_cli::output::note_name(108), "C8");
/// ```
#[must_use]
pub fn note_name(pitch: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = i32::from(pitch / 12) - 1;
    format!("{}{octave}", NAMES[usize::from(pitch % 12)])
}

/// Writes notes as CSV, exactly like Basic Pitch's `--save-note-events`.
///
/// One row per note, in the order given (Basic Pitch's order, not sorted):
/// `start_time_s,end_time_s,pitch_midi,velocity` followed by the pitch bends of the
/// note, in thirds of a semitone. Floats are printed like Python's `str`, and rows end
/// with `\r\n`, the default of Python's `csv` module (and RFC 4180).
///
/// # Errors
///
/// Any error of `out`.
pub fn write_csv(notes: &[NoteEvent], mut out: impl Write) -> io::Result<()> {
    out.write_all(b"start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\r\n")?;
    for note in notes {
        write!(
            out,
            "{},{},{},{}",
            python_float(note.start_s),
            python_float(note.end_s),
            note.pitch,
            note.velocity()
        )?;
        for bend in note.pitch_bends.iter().flatten() {
            write!(out, ",{bend}")?;
        }
        out.write_all(b"\r\n")?;
    }
    out.flush()
}

/// Formats a float like Python's `str` for the values Basic Pitch writes: the shortest
/// representation that round-trips, always with a decimal point.
fn python_float(x: f64) -> String {
    let s = x.to_string();
    if x.is_finite() && !s.contains('.') {
        s + ".0"
    } else {
        s
    }
}

/// Writes the posteriorgrams as an uncompressed NumPy `.npz` archive with three
/// `float32` arrays: `note` and `onset` (`frames x 88`) and `contour` (`frames x 264`).
///
/// ```python
/// import numpy as np
/// outputs = np.load("song_basic_pitch.npz")
/// outputs["note"].shape  # (n_frames, 88)
/// ```
///
/// # Errors
///
/// Any error of `out`.
pub fn write_npz(posteriorgrams: &Posteriorgrams, out: impl Write) -> io::Result<()> {
    npz::write_npz(
        &[
            matrix("note", &posteriorgrams.note),
            matrix("onset", &posteriorgrams.onset),
            matrix("contour", &posteriorgrams.contour),
        ],
        out,
    )
}

fn matrix<'a>(name: &'a str, p: &'a Posteriorgram) -> npz::Matrix<'a> {
    npz::Matrix {
        name,
        rows: p.n_frames(),
        cols: p.n_bins(),
        data: p.as_slice(),
    }
}

#[cfg(test)]
mod tests {
    use ndarray_npy::ReadNpyExt;

    use super::*;

    fn note(bends: Option<Vec<i32>>) -> NoteEvent {
        NoteEvent {
            start_frame: 0,
            end_frame: 3,
            start_s: 0.0,
            end_s: 0.034_829_931_972_789_115,
            pitch: 69,
            amplitude: 0.5,
            pitch_bends: bends,
        }
    }

    #[test]
    fn note_names_cover_the_piano() {
        assert_eq!(note_name(21), "A0");
        assert_eq!(note_name(60), "C4");
        assert_eq!(note_name(69), "A4");
        assert_eq!(note_name(70), "A#4");
        assert_eq!(note_name(108), "C8");
        assert_eq!(note_name(0), "C-1");
    }

    #[test]
    fn csv_matches_python() {
        let mut out = Vec::new();
        write_csv(
            &[note(Some(vec![1, 0, -2])), note(None), note(Some(vec![]))],
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(
            text,
            "start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\r\n\
             0.0,0.034829931972789115,69,64,1,0,-2\r\n\
             0.0,0.034829931972789115,69,64\r\n\
             0.0,0.034829931972789115,69,64\r\n"
        );
    }

    #[test]
    fn python_floats() {
        assert_eq!(python_float(0.0), "0.0");
        assert_eq!(python_float(2.0), "2.0");
        assert_eq!(python_float(0.1), "0.1");
        assert_eq!(python_float(1.0 / 3.0), "0.3333333333333333");
        assert_eq!(
            python_float(0.034_829_931_972_789_115),
            "0.034829931972789115"
        );
    }

    #[test]
    fn json_note_has_bends_in_both_units() {
        let n = JsonNote::from(&note(Some(vec![3, -1])));
        assert_eq!(n.note_name, "A4");
        assert_eq!(n.velocity, 64);
        assert_eq!(n.pitch_bends, Some(vec![3, -1]));
        assert_eq!(n.pitch_bends_semitones, Some(vec![1.0, -1.0 / 3.0]));
        let n = JsonNote::from(&note(None));
        assert_eq!(n.pitch_bends_semitones, None);
    }

    #[test]
    fn npz_holds_the_three_posteriorgrams() {
        let p = Posteriorgrams::new(
            Posteriorgram::from_vec((0..176).map(|i| i as f32).collect(), 88).unwrap(),
            Posteriorgram::zeros(2, 88),
            Posteriorgram::zeros(2, 264),
        )
        .unwrap();
        let mut zip = Vec::new();
        write_npz(&p, &mut zip).unwrap();
        let entries = npz::read_npz(&zip).unwrap();
        let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["note.npy", "onset.npy", "contour.npy"]);
        let note = ndarray::Array2::<f32>::read_npy(entries[0].1.as_slice()).unwrap();
        assert_eq!(note.dim(), (2, 88));
        assert_eq!(note[[1, 0]], 88.0);
        let contour = ndarray::Array2::<f32>::read_npy(entries[2].1.as_slice()).unwrap();
        assert_eq!(contour.dim(), (2, 264));
    }
}

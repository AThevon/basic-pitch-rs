// Copyright 2022 Spotify AB (original: basic_pitch/note_creation.py)
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
// Ported from spotify/basic-pitch v0.4.0. Changes: translated to Rust; inputs are never
// mutated (constrain_frequency works on copies); the melodia loop finds its global
// maximum with a segment tree instead of a full scan, with the same tie-breaking; empty
// input returns no notes instead of raising; an optional constant offset can be added to
// the pitch bends (`NoteParams::bend_offset`, 0 by default).

//! Post-processing: from posteriorgrams to note events.
//!
//! This is a faithful port of `basic_pitch/note_creation.py`. The numerical semantics of
//! numpy and scipy are reproduced wherever they can change a result: round-half-to-even,
//! `argrelmax` without edges or plateaus, first-index `argmax`, NaN propagation in
//! [`infer_onsets`], float64 intermediate arrays, numpy's pairwise float32 summation for
//! amplitudes, and the unsorted order of the returned notes. Fed with the same
//! posteriorgrams, [`model_output_to_notes`] returns exactly the notes of Python.
//!
//! The functions are public so that advanced users can run single steps, for instance
//! to re-decode notes with other thresholds without running the model again. Their
//! names and argument orders follow the Python functions, so that both codes can be read
//! side by side: note that [`output_to_notes_polyphonic`] takes `(frames, onsets)` while
//! [`infer_onsets`] and [`constrain_frequency`] take `(onsets, frames)`.
//! [`model_output_to_notes`] takes the whole [`Posteriorgrams`] and avoids the question.
//!
//! Two deliberate differences with Python, both on inputs Python cannot handle: a
//! negative `frame_threshold` is rejected (the melodia loop of Python never ends), and
//! the MIDI export refuses notes that cannot be written to a MIDI file (Python raises too,
//! but only when the file is written).
//!
//! ```
//! use basic_pitch::notes::{model_output_to_notes, NoteParams};
//! use basic_pitch::{Posteriorgram, Posteriorgrams};
//!
//! // A synthetic 40-frame posteriorgram with one A4 (MIDI 69, bin 48) from frame 5 on.
//! let n = 40;
//! let mut note = Posteriorgram::zeros(n, 88);
//! let mut onset = Posteriorgram::zeros(n, 88);
//! for t in 5..n {
//!     note.as_mut_slice()[t * 88 + 48] = 0.8;
//! }
//! onset.as_mut_slice()[5 * 88 + 48] = 0.9;
//! let output = Posteriorgrams::new(note, onset, Posteriorgram::zeros(n, 264))?;
//!
//! let notes = model_output_to_notes(&output, &NoteParams::default())?;
//! assert_eq!(notes.len(), 1);
//! assert_eq!((notes[0].pitch, notes[0].start_frame, notes[0].end_frame), (69, 5, 39));
//! # Ok::<(), basic_pitch::Error>(())
//! ```

use crate::constants::{
    ANNOT_N_FRAMES, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, CONTOURS_BINS_PER_SEMITONE,
    DEFAULT_ENERGY_TOLERANCE, DEFAULT_FRAME_THRESHOLD, DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    DEFAULT_ONSET_THRESHOLD, DEFAULT_PITCH_BEND_TOLERANCE, FFT_HOP, MAGIC_ALIGNMENT_OFFSET,
    MAX_FREQ_IDX, MIDI_OFFSET, N_FREQ_BINS_CONTOURS, N_FREQ_BINS_NOTES, hz_to_midi,
};
use crate::error::{Error, Result};
use crate::posteriorgram::{Posteriorgram, Posteriorgrams};

const N_BINS: usize = N_FREQ_BINS_NOTES;

/// Parameters of the note decoding, with the defaults of `basic_pitch.inference.predict`.
///
/// ```
/// use basic_pitch::notes::NoteParams;
///
/// let params = NoteParams {
///     onset_threshold: 0.6,
///     minimum_frequency: Some(80.0),
///     ..NoteParams::default()
/// };
/// assert_eq!(params.min_note_len_frames()?, 11);
/// # Ok::<(), basic_pitch::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct NoteParams {
    /// Minimum onset activation to start a note (Python `onset_thresh`, default 0.5).
    /// Compared with `>=`.
    pub onset_threshold: f64,
    /// Minimum note activation to keep a note going (Python `frame_thresh`, default 0.3).
    ///
    /// Must be zero or more: with a negative threshold, the melodia loop of Python never
    /// ends, so [`validate`](Self::validate) rejects it.
    pub frame_threshold: f64,
    /// Notes must last strictly longer than this, in milliseconds (default 127.70,
    /// which is 11 frames: notes are at least 12 frames long).
    pub minimum_note_length_ms: f64,
    /// Ignore notes below this frequency, in Hz (default: none).
    ///
    /// The bound becomes the key index `round(hz_to_midi(f) - 21)`, used as a numpy slice
    /// bound like in Python. Below about 26.7 Hz (half a semitone under A0) the index is
    /// negative and counts from the top: 20 Hz removes every key but the top 6 or so, as
    /// in Python. Use bounds within the piano range (27.5 Hz to 4186 Hz).
    pub minimum_frequency: Option<f64>,
    /// Ignore notes above this frequency, in Hz (default: none).
    ///
    /// Same conversion as [`minimum_frequency`](Self::minimum_frequency): below about
    /// 26.7 Hz, the bound only removes the top keys.
    pub maximum_frequency: Option<f64>,
    /// Add onsets where note activations jump, on top of the onset output (default `true`).
    pub infer_onsets: bool,
    /// Also decode notes that have no onset, from the remaining note energy
    /// (the "melodia trick", default `true`).
    pub melodia_trick: bool,
    /// Estimate per-frame pitch bends from the contour output (default `true`).
    pub include_pitch_bends: bool,
    /// Number of consecutive frames below `frame_threshold` that ends a note (default 11).
    pub energy_tolerance: usize,
    /// Constant added to every pitch bend, in thirds of a semitone (default 0).
    ///
    /// The model's contour peaks one bin (a third of a semitone) above the bin that
    /// `get_pitch_bends` expects, so Basic Pitch reports a bend of `+1` on notes that are
    /// perfectly in tune (spotify/basic-pitch#87). The default keeps this bias for parity
    /// with Python. Set it to `-1` to center in-tune notes on zero.
    pub bend_offset: i32,
}

impl Default for NoteParams {
    fn default() -> Self {
        Self {
            onset_threshold: DEFAULT_ONSET_THRESHOLD,
            frame_threshold: DEFAULT_FRAME_THRESHOLD,
            minimum_note_length_ms: DEFAULT_MINIMUM_NOTE_LENGTH_MS,
            minimum_frequency: None,
            maximum_frequency: None,
            infer_onsets: true,
            melodia_trick: true,
            include_pitch_bends: true,
            energy_tolerance: DEFAULT_ENERGY_TOLERANCE,
            bend_offset: 0,
        }
    }
}

impl NoteParams {
    /// The minimum note length in frames: `round(ms / 1000 * 22050 / 256)`, rounding
    /// half to even like `np.round`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if the length is negative or not finite.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "the value is checked to be finite and non-negative"
    )]
    pub fn min_note_len_frames(&self) -> Result<usize> {
        let ms = self.minimum_note_length_ms;
        if !ms.is_finite() || ms < 0.0 {
            return Err(Error::invalid(
                "minimum_note_length_ms",
                format!("must be finite and >= 0, got {ms}"),
            ));
        }
        let frames_per_second = f64::from(AUDIO_SAMPLE_RATE) / FFT_HOP as f64;
        Ok((ms / 1000.0 * frames_per_second).round_ties_even() as usize)
    }

    /// Checks that every parameter is usable.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] for a NaN threshold, a negative `frame_threshold`, a
    /// negative or non-finite note length, or a frequency bound that is not a positive
    /// finite number.
    pub fn validate(&self) -> Result<()> {
        for (name, v) in [
            ("onset_threshold", self.onset_threshold),
            ("frame_threshold", self.frame_threshold),
        ] {
            if v.is_nan() {
                return Err(Error::invalid(name, "must not be NaN"));
            }
        }
        // The melodia trick zeroes the energy it consumes and stops when the maximum is
        // no longer above the threshold: below zero, it would never stop.
        if self.frame_threshold < 0.0 {
            return Err(Error::invalid(
                "frame_threshold",
                format!("must be zero or more, got {}", self.frame_threshold),
            ));
        }
        self.min_note_len_frames()?;
        for (name, v) in [
            ("minimum_frequency", self.minimum_frequency),
            ("maximum_frequency", self.maximum_frequency),
        ] {
            if let Some(hz) = v {
                freq_to_bin(name, hz)?;
            }
        }
        Ok(())
    }
}

/// A note in frame units, as returned by [`output_to_notes_polyphonic`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameNote {
    /// First frame of the note.
    pub start_frame: usize,
    /// Frame after the note (exclusive end).
    pub end_frame: usize,
    /// MIDI pitch, 21 (A0) to 108 (C8).
    pub pitch: u8,
    /// Mean note activation over the note, in `[0, 1]`.
    pub amplitude: f32,
}

/// A transcribed note.
///
/// Times come from [`model_frames_to_time`], so `start_s` and `end_s` are exactly the
/// values Python Basic Pitch reports.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NoteEvent {
    /// First frame of the note.
    pub start_frame: usize,
    /// Frame after the note (exclusive end).
    pub end_frame: usize,
    /// Start time, in seconds.
    pub start_s: f64,
    /// End time, in seconds.
    pub end_s: f64,
    /// MIDI pitch, 21 (A0) to 108 (C8).
    pub pitch: u8,
    /// Mean note activation over the note, in `[0, 1]`.
    pub amplitude: f32,
    /// One pitch bend per frame (`end_frame - start_frame` values), in thirds of a
    /// semitone relative to `pitch`; `None` when pitch bends are disabled.
    pub pitch_bends: Option<Vec<i32>>,
}

impl NoteEvent {
    /// MIDI velocity, `round(127 * amplitude)` with ties to even, as written by Basic Pitch.
    ///
    /// The result is clamped to `0..=127`, and a NaN amplitude gives 0. Amplitudes from
    /// the model are always in `[0, 1]`; for other values, Python fails when it writes the
    /// MIDI file, and so does `midi::note_events_to_midi`.
    ///
    /// ```
    /// # use basic_pitch::notes::NoteEvent;
    /// let note = NoteEvent {
    ///     start_frame: 0, end_frame: 12, start_s: 0.0, end_s: 0.139,
    ///     pitch: 60, amplitude: 0.5, pitch_bends: None,
    /// };
    /// assert_eq!(note.velocity(), 64); // 63.5 rounds to the even 64
    /// ```
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the MIDI velocity range first"
    )]
    pub fn velocity(&self) -> u8 {
        // numpy computes 127 * amplitude in float32 before rounding.
        (127.0_f32 * self.amplitude)
            .round_ties_even()
            .clamp(0.0, 127.0) as u8
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f64 {
        self.end_s - self.start_s
    }
}

/// Start time in seconds of each of the first `n_frames` frames of an unwrapped output.
///
/// Reproduces `model_frames_to_time`, including its empirical 0.0018 s correction.
/// Frame `i` starts at `i * 256 / 22050 - floor(i / 172) * offset`, where the offset
/// compensates the difference between the nominal hop (256 samples) and the real one
/// (36164 / 142 = 254.68 samples).
///
/// ```
/// let t = basic_pitch::notes::model_frames_to_time(200);
/// assert_eq!(t[0], 0.0);
/// assert!((t[1] - 256.0 / 22050.0).abs() < 1e-15);
/// ```
#[must_use]
pub fn model_frames_to_time(n_frames: usize) -> Vec<f64> {
    (0..n_frames).map(frame_to_time).collect()
}

/// Start time in seconds of frame `frame`; see [`model_frames_to_time`].
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "frame indices are far below 2^52"
)]
pub fn frame_to_time(frame: usize) -> f64 {
    let hop_s = FFT_HOP as f64 / f64::from(AUDIO_SAMPLE_RATE);
    let window_offset = hop_s * (ANNOT_N_FRAMES as f64 - (AUDIO_N_SAMPLES as f64 / FFT_HOP as f64))
        + MAGIC_ALIGNMENT_OFFSET;
    let original = (frame * FFT_HOP) as f64 / f64::from(AUDIO_SAMPLE_RATE);
    let window_number = (frame / ANNOT_N_FRAMES) as f64;
    original - window_offset * window_number
}

/// Zeroes the note and onset activations outside `[min_freq, max_freq]`.
///
/// Returns `(onsets, frames)` copies, in the argument order of the Python function.
/// The bounds become bin indices with `round(hz_to_midi(f) - 21)`; out-of-range indices
/// follow numpy slicing rules, so they behave exactly like Python. Unlike Python, the
/// inputs are left untouched.
///
/// # Errors
///
/// [`Error::Shape`] if the matrices do not have 88 bins or differ in length, and
/// [`Error::InvalidParameter`] if a bound is not a positive finite frequency.
pub fn constrain_frequency(
    onsets: &Posteriorgram,
    frames: &Posteriorgram,
    max_freq: Option<f64>,
    min_freq: Option<f64>,
) -> Result<(Posteriorgram, Posteriorgram)> {
    check_note_pair(frames, onsets)?;
    let mut onsets = onsets.clone();
    let mut frames = frames.clone();
    let (lo, hi) = constrained_bins(max_freq, min_freq)?;
    for m in [&mut onsets, &mut frames] {
        for row in m.as_mut_slice().as_chunks_mut::<N_BINS>().0 {
            row[..lo].fill(0.0);
            row[hi..].fill(0.0);
        }
    }
    Ok((onsets, frames))
}

/// Bins kept by `constrain_frequency`: `[lo, hi)`, possibly empty.
fn constrained_bins(max_freq: Option<f64>, min_freq: Option<f64>) -> Result<(usize, usize)> {
    let mut hi = N_BINS;
    let mut lo = 0;
    if let Some(f) = max_freq {
        // onsets[:, max_freq_idx:] = 0
        hi = py_slice_index(freq_to_bin("maximum_frequency", f)?, N_BINS);
    }
    if let Some(f) = min_freq {
        // onsets[:, :min_freq_idx] = 0
        lo = py_slice_index(freq_to_bin("minimum_frequency", f)?, N_BINS);
    }
    Ok((lo, hi.max(lo)))
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "saturating float to int conversion, then numpy slice clamping"
)]
fn freq_to_bin(name: &'static str, hz: f64) -> Result<i64> {
    if !hz.is_finite() || hz <= 0.0 {
        return Err(Error::invalid(
            name,
            format!("must be a positive frequency in Hz, got {hz}"),
        ));
    }
    Ok((hz_to_midi(hz) - f64::from(MIDI_OFFSET)).round_ties_even() as i64)
}

/// Normalizes a Python slice bound against a length, like `slice.indices`.
fn py_slice_index(idx: i64, len: usize) -> usize {
    let len_i = i64::try_from(len).unwrap_or(i64::MAX);
    let i = if idx < 0 {
        (idx + len_i).max(0)
    } else {
        idx.min(len_i)
    };
    usize::try_from(i).unwrap_or(0)
}

/// Adds onsets where the note activation rises sharply (`get_infered_onsets`).
///
/// The result is `max(onsets, rescaled frame difference)` element-wise, as a row-major
/// `n_frames x 88` float64 matrix like numpy's. As in Python, a posteriorgram with no
/// rising activation (silence) makes the rescaling divide 0 by 0, and every value is NaN:
/// no note can then start from an onset.
///
/// # Errors
///
/// [`Error::Shape`] if the matrices do not have 88 bins or differ in length.
#[doc(alias = "get_infered_onsets")]
pub fn infer_onsets(onsets: &Posteriorgram, frames: &Posteriorgram) -> Result<Vec<f64>> {
    check_note_pair(frames, onsets)?;
    Ok(infer_onsets_unchecked(onsets, frames))
}

fn infer_onsets_unchecked(onsets: &Posteriorgram, frames: &Posteriorgram) -> Vec<f64> {
    const N_DIFF: usize = 2;
    let n = frames.n_frames();
    if n == 0 {
        return Vec::new();
    }
    let f = frames.as_slice();
    let at = |t: usize, b: usize| f64::from(f[t * N_BINS + b]);
    let mut diff = vec![0.0_f64; n * N_BINS];
    for t in N_DIFF..n {
        for b in 0..N_BINS {
            // np.min over [x_t - x_{t-1}, x_t - x_{t-2}], zeros before the start.
            let d1 = at(t, b) - at(t - 1, b);
            let d2 = at(t, b) - at(t - 2, b);
            let d = np_minimum(d1, d2);
            diff[t * N_BINS + b] = if d < 0.0 { 0.0 } else { d };
        }
    }
    // frame_diff[:n_diff, :] = 0 is the zero initialisation of the first rows.
    let max_onset = f64::from(np_max_f32(onsets.as_slice()));
    let max_diff = np_max_f64(&diff);
    onsets
        .as_slice()
        .iter()
        .zip(&diff)
        .map(|(&o, &d)| np_maximum(f64::from(o), max_onset * d / max_diff))
        .collect()
}

/// Decodes note events from the note and onset posteriorgrams
/// (`output_to_notes_polyphonic`).
///
/// Notes are returned in Python's order, which is not sorted: first the notes started
/// by an onset, by decreasing onset time (then decreasing pitch), then the melodia
/// notes in order of decreasing peak energy.
///
/// # Errors
///
/// [`Error::Shape`] if the matrices do not have 88 bins or differ in length, and
/// [`Error::InvalidParameter`] if `params` is invalid (see [`NoteParams::validate`]).
pub fn output_to_notes_polyphonic(
    frames: &Posteriorgram,
    onsets: &Posteriorgram,
    params: &NoteParams,
) -> Result<Vec<FrameNote>> {
    check_note_pair(frames, onsets)?;
    params.validate()?;
    let min_note_len = params.min_note_len_frames()?;
    let n = frames.n_frames();
    if n == 0 {
        return Ok(Vec::new());
    }
    let (onsets, frames) = constrain_frequency(
        onsets,
        frames,
        params.maximum_frequency,
        params.minimum_frequency,
    )?;
    let onsets: Vec<f64> = if params.infer_onsets {
        infer_onsets_unchecked(&onsets, &frames)
    } else {
        onsets.as_slice().iter().map(|&x| f64::from(x)).collect()
    };

    // peak_thresh_mat: onsets at the local maxima along time (scipy.signal.argrelmax,
    // order 1, mode "clip": never the first or last frame, never a plateau), 0 elsewhere.
    let threshold = params.onset_threshold;
    let mut onset_idx = Vec::new();
    for t in 0..n {
        for b in 0..N_BINS {
            let v = onsets[t * N_BINS + b];
            let is_peak = t >= 1
                && t + 1 < n
                && v > onsets[(t - 1) * N_BINS + b]
                && v > onsets[(t + 1) * N_BINS + b];
            if (if is_peak { v } else { 0.0 }) >= threshold {
                onset_idx.push((t, b));
            }
        }
    }

    let frames = frames.as_slice();
    let mut energy: Vec<f64> = frames.iter().map(|&x| f64::from(x)).collect();
    let frame_thresh = params.frame_threshold;
    let energy_tol = params.energy_tolerance;
    let mut notes = Vec::new();

    // Onsets are visited backwards in time, so a note ends where the next one starts.
    for &(start, b) in onset_idx.iter().rev() {
        if start >= n - 1 {
            continue;
        }
        let mut i = start + 1;
        let mut k = 0;
        while i < n - 1 && k < energy_tol {
            if energy[i * N_BINS + b] < frame_thresh {
                k += 1;
            } else {
                k = 0;
            }
            i += 1;
        }
        i -= k;
        if i - start <= min_note_len {
            continue;
        }
        for t in start..i {
            zero_with_neighbours(&mut energy, t, b);
        }
        notes.push(FrameNote {
            start_frame: start,
            end_frame: i,
            pitch: pitch_of_bin(b),
            amplitude: column_mean(frames, b, start, i),
        });
    }

    if params.melodia_trick {
        melodia(
            frames,
            &mut energy,
            n,
            frame_thresh,
            energy_tol,
            min_note_len,
            &mut notes,
        );
    }
    Ok(notes)
}

/// The melodia trick: repeatedly grows a note around the global energy maximum.
fn melodia(
    frames: &[f32],
    energy: &mut [f64],
    n: usize,
    frame_thresh: f64,
    energy_tol: usize,
    min_note_len: usize,
    notes: &mut Vec<FrameNote>,
) {
    let mut tree = MaxTree::new(energy, n);
    // np.max(remaining_energy) > frame_thresh; a NaN maximum ends the loop, as in numpy.
    while let Some((i_mid, b)) = tree
        .argmax()
        .filter(|&(t, b)| energy[t * N_BINS + b] > frame_thresh)
    {
        energy[i_mid * N_BINS + b] = 0.0;

        // Forward pass.
        let mut i = i_mid + 1;
        let mut k = 0;
        while i + 1 < n && k < energy_tol {
            if energy[i * N_BINS + b] < frame_thresh {
                k += 1;
            } else {
                k = 0;
            }
            zero_with_neighbours(energy, i, b);
            i += 1;
        }
        let forward_end = i; // exclusive bound of the rows touched
        let i_end = i - 1 - k;

        // Backward pass; frame 0 is never visited, as in Python.
        let mut i = i_mid.cast_signed() - 1;
        let mut k = 0_isize;
        while i > 0 && k < energy_tol.cast_signed() {
            let t = i.cast_unsigned();
            if energy[t * N_BINS + b] < frame_thresh {
                k += 1;
            } else {
                k = 0;
            }
            zero_with_neighbours(energy, t, b);
            i -= 1;
        }
        let backward_start = (i + 1).cast_unsigned(); // first row touched
        let i_start = (i + 1 + k).cast_unsigned();

        tree.refresh(
            energy,
            backward_start.min(i_mid)..forward_end.max(i_mid + 1),
        );

        if i_end.saturating_sub(i_start) <= min_note_len {
            continue;
        }
        notes.push(FrameNote {
            start_frame: i_start,
            end_frame: i_end,
            pitch: pitch_of_bin(b),
            amplitude: column_mean(frames, b, i_start, i_end),
        });
    }
}

fn zero_with_neighbours(energy: &mut [f64], t: usize, b: usize) {
    let row = &mut energy[t * N_BINS..(t + 1) * N_BINS];
    row[b] = 0.0;
    if b < MAX_FREQ_IDX {
        row[b + 1] = 0.0;
    }
    if b > 0 {
        row[b - 1] = 0.0;
    }
}

#[allow(clippy::cast_possible_truncation, reason = "bins are below 88")]
fn pitch_of_bin(b: usize) -> u8 {
    b as u8 + MIDI_OFFSET
}

/// Estimates per-frame pitch bends for each note from the contour posteriorgram
/// (`get_pitch_bends`).
///
/// For every frame of a note, the contour around the note's bin (plus or minus
/// `n_bins_tolerance` bins) is weighted by a Gaussian of standard deviation 5 bins, and
/// the bend is the offset of the maximum, in thirds of a semitone. The default tolerance
/// is [`DEFAULT_PITCH_BEND_TOLERANCE`] (25). Frames past the end of `contours` are
/// ignored, like Python slicing does. Tolerances above 264 bins give the same bends as
/// 264, since the window then covers every bin whatever the pitch.
///
/// # Errors
///
/// [`Error::Shape`] if `contours` does not have 264 bins.
pub fn get_pitch_bends(
    contours: &Posteriorgram,
    notes: &[FrameNote],
    n_bins_tolerance: usize,
) -> Result<Vec<Vec<i32>>> {
    contours.expect_bins("contour posteriorgram", N_FREQ_BINS_CONTOURS)?;
    // Beyond the number of bins, the window always spans the whole contour and the
    // Gaussian weights only depend on the distance to the note's bin.
    let n_bins_tolerance = n_bins_tolerance.min(N_FREQ_BINS_CONTOURS);
    let gaussian = gaussian_window(n_bins_tolerance);
    let tol = n_bins_tolerance.cast_signed();
    let n_contour = N_FREQ_BINS_CONTOURS.cast_signed();
    let window_length = gaussian.len().cast_signed();
    let data = contours.as_slice();
    let mut out = Vec::with_capacity(notes.len());
    for note in notes {
        // round(midi_pitch_to_contour_bin(pitch)) is exactly 3 * (pitch - 21).
        let freq_idx = (isize::from(note.pitch) - isize::from(MIDI_OFFSET))
            * CONTOURS_BINS_PER_SEMITONE.cast_signed();
        let freq_start = (freq_idx - tol).max(0);
        let freq_end = (freq_idx + tol + 1).min(n_contour);
        let g_start = (tol - freq_idx).max(0);
        let g_end = window_length - (freq_idx - (n_contour - tol - 1)).max(0);
        let width = (freq_end - freq_start).min(g_end - g_start);
        if width <= 0 {
            out.push(Vec::new());
            continue;
        }
        let (freq_start, g_start, width) = (
            freq_start.cast_unsigned(),
            g_start.cast_unsigned(),
            width.cast_unsigned(),
        );
        let pb_shift = tol - g_start.cast_signed();
        let end = note.end_frame.min(contours.n_frames());
        let mut bends = Vec::with_capacity(end.saturating_sub(note.start_frame));
        for t in note.start_frame..end {
            let row = &data[t * N_FREQ_BINS_CONTOURS + freq_start..][..width];
            let best = np_argmax(
                row.iter()
                    .zip(&gaussian[g_start..])
                    .map(|(&c, &g)| f64::from(c) * g),
            );
            #[allow(
                clippy::cast_possible_truncation,
                reason = "bend offsets are bounded by the window length"
            )]
            bends.push((best.cast_signed() - pb_shift) as i32);
        }
        out.push(bends);
    }
    Ok(out)
}

/// `scipy.signal.windows.gaussian(2 * tol + 1, std=5)`.
fn gaussian_window(tol: usize) -> Vec<f64> {
    // The default window, as computed by scipy 1.17 (bit-exact; libm `exp` may differ by
    // one ulp across platforms, and wasm has its own).
    const HALF_25: [f64; 26] = [
        3.726_653_172_078_671e-6,
        9.929_504_305_851_081e-6,
        2.541_934_651_619_924_7e-5,
        6.252_150_377_482_026e-5,
        0.000_147_748_360_232_033_64,
        0.000_335_462_627_902_511_85,
        0.000_731_802_418_880_472_8,
        0.001_533_810_679_324_463,
        0.003_088_715_408_236_768_7,
        0.005_976_022_895_005_943,
        0.011_108_996_538_242_306,
        0.019_841_094_744_370_288,
        0.034_047_454_734_599_344,
        0.056_134_762_834_133_725,
        0.088_921_617_459_386_34,
        0.135_335_283_236_612_7,
        0.197_898_699_083_614_65,
        0.278_037_300_453_194_14,
        0.375_311_098_851_399_57,
        0.486_752_255_959_971_7,
        0.606_530_659_712_633_4,
        0.726_149_037_073_690_9,
        0.835_270_211_411_272,
        0.923_116_346_386_635_8,
        0.980_198_673_306_755_3,
        1.0,
    ];
    if tol == DEFAULT_PITCH_BEND_TOLERANCE {
        return HALF_25
            .iter()
            .chain(HALF_25.iter().rev().skip(1))
            .copied()
            .collect();
    }
    let std = 5.0_f64;
    let sig2 = 2.0 * std * std;
    #[allow(clippy::cast_precision_loss, reason = "small window indices")]
    (0..=2 * tol)
        .map(|i| {
            let x = i as f64 - tol as f64;
            (-(x * x) / sig2).exp()
        })
        .collect()
}

/// Runs the whole post-processing on unwrapped model outputs (`model_output_to_notes`,
/// without the MIDI part, which lives in the `midi` module).
///
/// # Errors
///
/// [`Error::Shape`] if `output` has inconsistent shapes, and
/// [`Error::InvalidParameter`] if `params` is invalid.
pub fn model_output_to_notes(
    output: &Posteriorgrams,
    params: &NoteParams,
) -> Result<Vec<NoteEvent>> {
    output.validate()?;
    let notes = output_to_notes_polyphonic(&output.note, &output.onset, params)?;
    let bends = if params.include_pitch_bends {
        let mut bends = get_pitch_bends(&output.contour, &notes, DEFAULT_PITCH_BEND_TOLERANCE)?;
        if params.bend_offset != 0 {
            for b in bends.iter_mut().flatten() {
                *b = b.saturating_add(params.bend_offset);
            }
        }
        bends.into_iter().map(Some).collect()
    } else {
        vec![None; notes.len()]
    };
    Ok(notes
        .into_iter()
        .zip(bends)
        .map(|(n, pitch_bends)| NoteEvent {
            start_frame: n.start_frame,
            end_frame: n.end_frame,
            start_s: frame_to_time(n.start_frame),
            end_s: frame_to_time(n.end_frame),
            pitch: n.pitch,
            amplitude: n.amplitude,
            pitch_bends,
        })
        .collect())
}

fn check_note_pair(frames: &Posteriorgram, onsets: &Posteriorgram) -> Result<()> {
    frames.expect_bins("note posteriorgram", N_BINS)?;
    onsets.expect_bins("onset posteriorgram", N_BINS)?;
    if frames.n_frames() == onsets.n_frames() {
        Ok(())
    } else {
        Err(Error::shape(
            "onset frames",
            frames.n_frames(),
            onsets.n_frames(),
        ))
    }
}

/// `np.mean(frames[start:end, bin])` in float32, with numpy's pairwise summation.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "numpy semantics"
)]
fn column_mean(frames: &[f32], bin: usize, start: usize, end: usize) -> f32 {
    let column: Vec<f32> = (start..end).map(|t| frames[t * N_BINS + bin]).collect();
    // np.add.reduce starts from the identity 0, then numpy divides the float32 sum by
    // an intp count, which promotes to float64 before casting back.
    let sum = 0.0_f32 + pairwise_sum(&column);
    (f64::from(sum) / column.len() as f64) as f32
}

/// numpy's pairwise summation (`pairwise_sum_FLOAT` in `loops_utils.h.src`).
fn pairwise_sum(a: &[f32]) -> f32 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0_f32;
        for &x in a {
            res += x;
        }
        res
    } else if n <= BLOCK {
        let mut r = [0.0_f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[i..] {
            res += x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.minimum`: NaN if either side is NaN.
fn np_minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// `np.maximum`: NaN if either side is NaN.
fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// `np.max` of a non-empty array: NaN if any value is NaN.
fn np_max_f32(a: &[f32]) -> f32 {
    a.iter().copied().fold(f32::NEG_INFINITY, |m, x| {
        if m.is_nan() || x.is_nan() {
            f32::NAN
        } else {
            m.max(x)
        }
    })
}

fn np_max_f64(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::NEG_INFINITY, np_maximum)
}

/// `np.argmax`: index of the first maximum, or of the first NaN.
fn np_argmax(values: impl Iterator<Item = f64>) -> usize {
    let mut best = 0;
    let mut best_v = f64::NEG_INFINITY;
    for (i, v) in values.enumerate() {
        if v.is_nan() {
            return i;
        }
        if v > best_v || i == 0 {
            best = i;
            best_v = v;
        }
    }
    best
}

/// `true` if `a` comes before `b` in numpy's `argmax` order: larger value first, NaN
/// above everything; equal values keep the earlier one.
fn beats(a: f64, b: f64) -> bool {
    match (a.is_nan(), b.is_nan()) {
        (true, false) => true,
        (false, false) => a > b,
        _ => false,
    }
}

/// Segment tree over frames giving numpy's flat `argmax` of the energy matrix.
///
/// Each leaf holds the maximum of one frame and its first bin. Internal nodes keep the
/// better child, preferring the left (earlier) one on ties, so the root is the first
/// maximum in row-major order: exactly `np.unravel_index(np.argmax(energy))`.
struct MaxTree {
    size: usize,
    // (value, frame, bin); padding leaves hold (-inf, usize::MAX, 0).
    nodes: Vec<(f64, usize, usize)>,
}

impl MaxTree {
    fn new(energy: &[f64], n: usize) -> Self {
        let size = n.next_power_of_two();
        let mut nodes = vec![(f64::NEG_INFINITY, usize::MAX, 0); 2 * size];
        for t in 0..n {
            nodes[size + t] = Self::leaf(energy, t);
        }
        let mut tree = Self { size, nodes };
        for i in (1..size).rev() {
            tree.pull(i);
        }
        tree
    }

    fn leaf(energy: &[f64], t: usize) -> (f64, usize, usize) {
        let row = &energy[t * N_BINS..(t + 1) * N_BINS];
        let b = np_argmax(row.iter().copied());
        (row[b], t, b)
    }

    fn pull(&mut self, i: usize) {
        let (l, r) = (self.nodes[2 * i], self.nodes[2 * i + 1]);
        self.nodes[i] = if beats(r.0, l.0) { r } else { l };
    }

    fn refresh(&mut self, energy: &[f64], rows: std::ops::Range<usize>) {
        for t in rows {
            let mut i = self.size + t;
            self.nodes[i] = Self::leaf(energy, t);
            while i > 1 {
                i /= 2;
                self.pull(i);
            }
        }
    }

    fn argmax(&self) -> Option<(usize, usize)> {
        let (_, t, b) = self.nodes[1];
        (t != usize::MAX).then_some((t, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grams(n: usize) -> (Posteriorgram, Posteriorgram) {
        (Posteriorgram::zeros(n, 88), Posteriorgram::zeros(n, 88))
    }

    #[test]
    fn contour_bin_is_three_per_semitone() {
        for p in 21..=108_u8 {
            let hz = crate::constants::midi_to_hz(f64::from(p));
            let bin = (36.0 * (hz / 27.5).log2()).round_ties_even();
            assert_eq!(bin, f64::from(3 * (u32::from(p) - 21)), "pitch {p}");
        }
    }

    #[test]
    fn gaussian_table_matches_formula_shape() {
        let g = gaussian_window(25);
        assert_eq!(g.len(), 51);
        assert_eq!(g[25], 1.0);
        assert_eq!(g[0], g[50]);
        let generic = gaussian_window(24);
        assert_eq!(generic.len(), 49);
        assert!((generic[0] - (-(24.0_f64 * 24.0) / 50.0).exp()).abs() < 1e-18);
    }

    #[test]
    fn pairwise_sum_matches_reference_grouping() {
        // Values chosen so that the grouping changes the float32 result.
        let a: Vec<f32> = (0..300)
            .map(|i| 1.0 + (i as f32) * 1e-7 + if i % 3 == 0 { 3e-1 } else { 0.0 })
            .collect();
        let naive: f32 = a.iter().sum();
        let pw = pairwise_sum(&a);
        let exact: f64 = a.iter().map(|&x| f64::from(x)).sum();
        assert!((f64::from(pw) - exact).abs() <= (f64::from(naive) - exact).abs());
    }

    #[test]
    fn argmax_semantics() {
        assert_eq!(np_argmax([1.0, 3.0, 3.0, 2.0].into_iter()), 1);
        assert_eq!(np_argmax([1.0, f64::NAN, 9.0].into_iter()), 1);
        assert_eq!(np_argmax([f64::NEG_INFINITY; 3].into_iter()), 0);
    }

    #[test]
    fn slice_bounds_follow_python() {
        assert_eq!(py_slice_index(-3, 88), 85);
        assert_eq!(py_slice_index(-200, 88), 0);
        assert_eq!(py_slice_index(200, 88), 88);
        assert_eq!(
            constrained_bins(Some(1000.0), Some(80.0)).unwrap(),
            (18, 62)
        );
        // A very low minimum frequency gives a negative index: numpy zeroes `[:, :-k]`.
        let (lo, hi) = constrained_bins(None, Some(20.0)).unwrap();
        assert_eq!((lo, hi), (82, 88));
    }

    #[test]
    fn silence_gives_nan_inferred_onsets_and_no_notes() {
        let (frames, onsets) = grams(30);
        let inferred = infer_onsets(&onsets, &frames).unwrap();
        assert!(inferred.iter().all(|x| x.is_nan()));
        let notes = output_to_notes_polyphonic(&frames, &onsets, &NoteParams::default()).unwrap();
        assert!(notes.is_empty());
    }

    #[test]
    fn empty_and_tiny_inputs_do_not_panic() {
        for n in 0..4 {
            let (mut frames, onsets) = grams(n);
            frames.as_mut_slice().fill(0.9);
            let params = NoteParams {
                minimum_note_length_ms: 0.0,
                ..NoteParams::default()
            };
            assert!(output_to_notes_polyphonic(&frames, &onsets, &params).is_ok());
        }
    }

    #[test]
    fn nan_activations_do_not_panic() {
        let (mut frames, mut onsets) = grams(50);
        frames.as_mut_slice().fill(f32::NAN);
        onsets.as_mut_slice()[10 * 88 + 3] = f32::NAN;
        let out = Posteriorgrams::new(frames, onsets, Posteriorgram::zeros(50, 264)).unwrap();
        assert!(
            model_output_to_notes(&out, &NoteParams::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn rejects_bad_parameters_and_shapes() {
        let (frames, onsets) = grams(10);
        let bad = NoteParams {
            minimum_frequency: Some(-1.0),
            ..NoteParams::default()
        };
        assert!(output_to_notes_polyphonic(&frames, &onsets, &bad).is_err());
        let nan = NoteParams {
            frame_threshold: f64::NAN,
            ..NoteParams::default()
        };
        assert!(nan.validate().is_err());
        assert!(
            output_to_notes_polyphonic(
                &frames,
                &Posteriorgram::zeros(9, 88),
                &NoteParams::default()
            )
            .is_err()
        );
        assert!(get_pitch_bends(&Posteriorgram::zeros(10, 88), &[], 25).is_err());
    }

    #[test]
    fn negative_frame_threshold_is_rejected() {
        // Python never returns: melodia zeroes the energy it visits, and 0 > -0.1.
        let (mut frames, onsets) = grams(30);
        frames.as_mut_slice()[10 * 88 + 40] = 0.5;
        let params = NoteParams {
            frame_threshold: -0.1,
            ..NoteParams::default()
        };
        let err = params.validate().unwrap_err();
        assert!(matches!(
            err,
            Error::InvalidParameter {
                name: "frame_threshold",
                ..
            }
        ));
        assert!(output_to_notes_polyphonic(&frames, &onsets, &params).is_err());
        let zero = NoteParams {
            frame_threshold: 0.0,
            ..NoteParams::default()
        };
        assert!(output_to_notes_polyphonic(&frames, &onsets, &zero).is_ok());
        let negative_zero = NoteParams {
            frame_threshold: -0.0,
            ..NoteParams::default()
        };
        assert!(negative_zero.validate().is_ok());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn params_reject_unknown_fields() {
        let p: NoteParams = serde_json::from_str(r#"{"onset_threshold": 0.7}"#).unwrap();
        assert_eq!(p.onset_threshold, 0.7);
        assert_eq!(p.frame_threshold, NoteParams::default().frame_threshold);
        assert!(serde_json::from_str::<NoteParams>(r#"{"onsetThreshold": 0.7}"#).is_err());
        #[cfg(feature = "midi")]
        {
            use crate::midi::MidiParams;
            assert!(serde_json::from_str::<MidiParams>(r#"{"midi_tempo": 90.0}"#).is_ok());
            assert!(serde_json::from_str::<MidiParams>(r#"{"tempo": 90.0}"#).is_err());
        }
    }

    #[test]
    fn huge_bend_tolerances_do_not_overflow() {
        let n = 6;
        let mut contour = Posteriorgram::zeros(n, 264);
        for (t, row) in contour
            .as_mut_slice()
            .as_chunks_mut::<264>()
            .0
            .iter_mut()
            .enumerate()
        {
            for (b, v) in row.iter_mut().enumerate() {
                *v = ((t * 31 + b * 17) % 101) as f32 / 100.0;
            }
        }
        let notes = [
            FrameNote {
                start_frame: 0,
                end_frame: n,
                pitch: 21,
                amplitude: 0.5,
            },
            FrameNote {
                start_frame: 1,
                end_frame: n,
                pitch: 108,
                amplitude: 0.5,
            },
        ];
        // Reference: the Gaussian-weighted argmax over every bin, relative to the note.
        let expected: Vec<Vec<i32>> = notes
            .iter()
            .map(|note| {
                let center = 3 * (usize::from(note.pitch) - 21);
                (note.start_frame..note.end_frame)
                    .map(|t| {
                        let row = &contour.as_slice()[t * 264..(t + 1) * 264];
                        let best = np_argmax(row.iter().enumerate().map(|(b, &c)| {
                            let x = b as f64 - center as f64;
                            f64::from(c) * (-(x * x) / 50.0).exp()
                        }));
                        best as i32 - center as i32
                    })
                    .collect()
            })
            .collect();
        for tol in [264, 300, 10_000, usize::MAX / 2 + 1, usize::MAX] {
            assert_eq!(
                get_pitch_bends(&contour, &notes, tol).unwrap(),
                expected,
                "tolerance {tol}"
            );
        }
    }

    #[test]
    fn melodia_picks_notes_without_onsets() {
        let (mut frames, onsets) = grams(60);
        for t in 10..40 {
            frames.as_mut_slice()[t * 88 + 40] = 0.7;
        }
        let params = NoteParams {
            infer_onsets: false,
            ..NoteParams::default()
        };
        let notes = output_to_notes_polyphonic(&frames, &onsets, &params).unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(
            (notes[0].start_frame, notes[0].end_frame, notes[0].pitch),
            (10, 39, 61)
        );
        let none = NoteParams {
            melodia_trick: false,
            ..params
        };
        assert!(
            output_to_notes_polyphonic(&frames, &onsets, &none)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn bend_offset_shifts_every_bend() {
        let n = 30;
        let mut note = Posteriorgram::zeros(n, 88);
        for t in 2..n {
            note.as_mut_slice()[t * 88 + 39] = 0.9;
        }
        let mut contour = Posteriorgram::zeros(n, 264);
        for t in 0..n {
            contour.as_mut_slice()[t * 264 + 3 * 39 + 1] = 0.8;
        }
        let out = Posteriorgrams::new(note, Posteriorgram::zeros(n, 88), contour).unwrap();
        let parity = model_output_to_notes(&out, &NoteParams::default()).unwrap();
        assert!(
            parity[0]
                .pitch_bends
                .as_ref()
                .unwrap()
                .iter()
                .all(|&b| b == 1)
        );
        let centred = NoteParams {
            bend_offset: -1,
            ..NoteParams::default()
        };
        let fixed = model_output_to_notes(&out, &centred).unwrap();
        assert!(
            fixed[0]
                .pitch_bends
                .as_ref()
                .unwrap()
                .iter()
                .all(|&b| b == 0)
        );
    }
}

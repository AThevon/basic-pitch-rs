// Copyright 2022 Spotify AB (original: basic_pitch/constants.py, basic_pitch/inference.py)
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
// Ported from spotify/basic-pitch v0.4.0.

//! Constants of the Basic Pitch model and of its pre- and post-processing.
//!
//! The values and their derivations are those of `basic_pitch/constants.py` and
//! `basic_pitch/inference.py`. Most users never need them directly; they are public
//! for code that drives the low-level pipeline (custom windowing, streaming, plots).

/// Sample rate the model expects, in Hz. Audio must be mono at this rate.
pub const AUDIO_SAMPLE_RATE: u32 = 22_050;

/// Hop between two model frames inside one window, in samples.
pub const FFT_HOP: usize = 256;

/// Duration of one model window, in seconds.
pub const AUDIO_WINDOW_LENGTH: usize = 2;

/// Nominal number of frames per second: `22050 // 256`, an integer division.
///
/// The true frame rate of the unwrapped output is closer to 86.58 frames/s
/// (see [`model_frames_to_time`](crate::notes::model_frames_to_time)), but the
/// truncated value is what Basic Pitch uses, and parity depends on it.
pub const ANNOTATIONS_FPS: usize = AUDIO_SAMPLE_RATE as usize / FFT_HOP;

/// Number of frames the model outputs for one window (172).
pub const ANNOT_N_FRAMES: usize = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH;

/// Number of samples in one model window (43 844).
pub const AUDIO_N_SAMPLES: usize = AUDIO_SAMPLE_RATE as usize * AUDIO_WINDOW_LENGTH - FFT_HOP;

/// Number of frames shared by two consecutive windows.
pub const N_OVERLAPPING_FRAMES: usize = 30;

/// Overlap between two consecutive windows, in samples (7 680).
pub const OVERLAP_LEN: usize = N_OVERLAPPING_FRAMES * FFT_HOP;

/// Hop between two consecutive windows, in samples (36 164).
pub const HOP_SIZE: usize = AUDIO_N_SAMPLES - OVERLAP_LEN;

/// Zeros prepended to the audio before windowing (3 840 samples).
pub const PADDING_SAMPLES: usize = OVERLAP_LEN / 2;

/// Frames removed at each end of every window when unwrapping (15).
pub const N_OVERLAP_FRAMES_PER_SIDE: usize = N_OVERLAPPING_FRAMES / 2;

/// Frames kept per window after unwrapping (142).
pub const FRAMES_PER_WINDOW: usize = ANNOT_N_FRAMES - N_OVERLAPPING_FRAMES;

/// Number of semitones covered by the note and onset outputs (piano range A0 to C8).
pub const ANNOTATIONS_N_SEMITONES: usize = 88;

/// Frequency of the lowest note bin (A0), in Hz.
pub const ANNOTATIONS_BASE_FREQUENCY: f64 = 27.5;

/// Contour bins per semitone.
pub const CONTOURS_BINS_PER_SEMITONE: usize = 3;

/// Number of bins of the note and onset outputs (88).
pub const N_FREQ_BINS_NOTES: usize = ANNOTATIONS_N_SEMITONES;

/// Number of bins of the contour output (264, one third of a semitone each).
pub const N_FREQ_BINS_CONTOURS: usize = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE;

/// MIDI pitch of note bin 0 (A0).
pub const MIDI_OFFSET: u8 = 21;

/// Index of the highest note bin (87).
pub const MAX_FREQ_IDX: usize = N_FREQ_BINS_NOTES - 1;

/// Empirical offset, in seconds, used by Basic Pitch to align frame times across windows.
pub const MAGIC_ALIGNMENT_OFFSET: f64 = 0.0018;

/// Default onset threshold of `basic_pitch.inference.predict`.
pub const DEFAULT_ONSET_THRESHOLD: f64 = 0.5;

/// Default frame threshold of `basic_pitch.inference.predict`.
pub const DEFAULT_FRAME_THRESHOLD: f64 = 0.3;

/// Default minimum note length of `basic_pitch.inference.predict`, in milliseconds.
pub const DEFAULT_MINIMUM_NOTE_LENGTH_MS: f64 = 127.70;

/// Default MIDI tempo of `basic_pitch.inference.predict`, in beats per minute.
pub const DEFAULT_MIDI_TEMPO: f64 = 120.0;

/// Default number of consecutive low-energy frames that ends a note.
pub const DEFAULT_ENERGY_TOLERANCE: usize = 11;

/// Default pitch bend search range around a note, in contour bins.
pub const DEFAULT_PITCH_BEND_TOLERANCE: usize = 25;

/// Name of the model input tensor in `nmp.onnx`.
pub const MODEL_INPUT_NAME: &str = "serving_default_input_2:0";

/// Name of the note (frame activation) output tensor in `nmp.onnx`.
pub const MODEL_OUTPUT_NOTE: &str = "StatefulPartitionedCall:1";

/// Name of the onset output tensor in `nmp.onnx`.
pub const MODEL_OUTPUT_ONSET: &str = "StatefulPartitionedCall:2";

/// Name of the contour output tensor in `nmp.onnx`.
pub const MODEL_OUTPUT_CONTOUR: &str = "StatefulPartitionedCall:0";

/// Converts a frequency in Hz to a (fractional) MIDI pitch, like `librosa.hz_to_midi`.
#[must_use]
pub fn hz_to_midi(hz: f64) -> f64 {
    12.0 * (hz.log2() - 440.0_f64.log2()) + 69.0
}

/// Converts a (fractional) MIDI pitch to a frequency in Hz, like `librosa.midi_to_hz`.
#[must_use]
pub fn midi_to_hz(midi: f64) -> f64 {
    440.0 * 2.0_f64.powf((midi - 69.0) / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_values_match_basic_pitch() {
        assert_eq!(ANNOTATIONS_FPS, 86);
        assert_eq!(ANNOT_N_FRAMES, 172);
        assert_eq!(AUDIO_N_SAMPLES, 43_844);
        assert_eq!(OVERLAP_LEN, 7_680);
        assert_eq!(HOP_SIZE, 36_164);
        assert_eq!(PADDING_SAMPLES, 3_840);
        assert_eq!(FRAMES_PER_WINDOW, 142);
        assert_eq!(N_FREQ_BINS_CONTOURS, 264);
    }

    #[test]
    fn pitch_conversions_round_trip() {
        assert!((hz_to_midi(440.0) - 69.0).abs() < 1e-12);
        assert!((midi_to_hz(21.0) - ANNOTATIONS_BASE_FREQUENCY).abs() < 1e-12);
        assert!((hz_to_midi(midi_to_hz(60.25)) - 60.25).abs() < 1e-9);
    }
}

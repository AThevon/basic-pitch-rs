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
// Ported from spotify/basic-pitch v0.4.0. Changes: translated to Rust, windows are
// produced lazily, and the trim of `unwrap_output` is selectable between v0.4.0 and
// the fix merged on main after v0.4.0 (spotify/basic-pitch#179).

//! Cutting audio into model windows, and stitching the window outputs back together.
//!
//! The model sees 2-second windows of [`AUDIO_N_SAMPLES`] samples. Basic Pitch prepends
//! [`PADDING_SAMPLES`] zeros to the audio, slides a window by [`HOP_SIZE`] samples and
//! zero-pads the last one. Each window yields [`ANNOT_N_FRAMES`] frames; consecutive
//! windows overlap by [`N_OVERLAPPING_FRAMES`](crate::constants::N_OVERLAPPING_FRAMES), so [`unwrap_output`] drops half of the
//! overlap on each side of every window before concatenating them.
//!
//! ```
//! use basic_pitch::constants::AUDIO_N_SAMPLES;
//! use basic_pitch::window::{window_count, windows};
//!
//! let audio = vec![0.0_f32; 22_050 * 5]; // 5 s at 22 050 Hz
//! assert_eq!(window_count(audio.len()), 4);
//! for w in windows(&audio) {
//!     assert_eq!(w.len(), AUDIO_N_SAMPLES);
//! }
//! ```

use std::fmt;
use std::str::FromStr;

use crate::constants::{
    ANNOT_N_FRAMES, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FRAMES_PER_WINDOW, HOP_SIZE,
    N_OVERLAP_FRAMES_PER_SIDE, PADDING_SAMPLES,
};
use crate::error::{Error, Result};
use crate::posteriorgram::Posteriorgram;

/// How [`unwrap_output`] trims the concatenated frames to the audio length.
///
/// The two Basic Pitch revisions only differ here. Everything else in the pipeline is
/// identical, so this enum is the whole difference between the two parity targets.
///
/// Its text form, used by [`Display`](std::fmt::Display), [`FromStr`] and serde, is
/// `"main"` or `"v0.4.0"`:
///
/// ```
/// use basic_pitch::TrimMode;
///
/// assert_eq!("v0.4.0".parse::<TrimMode>()?, TrimMode::V0_4_0);
/// assert_eq!(TrimMode::Main.to_string(), "main");
/// # Ok::<(), basic_pitch::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum TrimMode {
    /// `int(n / 36164 * 142)` frames, as on spotify/basic-pitch main since PR #179
    /// (commit `e989e40`, not yet on PyPI).
    ///
    /// This follows the real frame rate of the unwrapped output (about 86.58 frames/s)
    /// and keeps the end of the file.
    #[default]
    #[cfg_attr(feature = "serde", serde(rename = "main"))]
    Main,
    /// `floor(n * 86 / 22050)` frames, as in the basic-pitch 0.4.0 release on PyPI.
    ///
    /// The nominal 86 frames/s is lower than the real rate, so this drops the last
    /// frames of the file: about 0.39 s of a 60 s recording.
    #[cfg_attr(feature = "serde", serde(rename = "v0.4.0"))]
    V0_4_0,
}

impl TrimMode {
    /// Every trim mode.
    pub const ALL: [Self; 2] = [Self::Main, Self::V0_4_0];

    /// The text form: `"main"` or `"v0.4.0"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::V0_4_0 => "v0.4.0",
        }
    }

    /// Number of frames kept for `n_samples` samples of audio at 22 050 Hz.
    ///
    /// ```
    /// use basic_pitch::window::TrimMode;
    ///
    /// assert_eq!(TrimMode::Main.frames_for(1_323_000), 5194); // 60 s
    /// assert_eq!(TrimMode::V0_4_0.frames_for(1_323_000), 5160);
    /// ```
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "mirrors Python's int() of a non-negative float"
    )]
    pub fn frames_for(self, n_samples: usize) -> usize {
        let n = n_samples as f64;
        match self {
            // int((audio_original_length / hop_size) * (AUDIO_WINDOW_LENGTH * ANNOTATIONS_FPS - n_overlapping_frames))
            Self::Main => ((n / HOP_SIZE as f64) * FRAMES_PER_WINDOW as f64) as usize,
            // int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
            Self::V0_4_0 => {
                let fps = crate::constants::ANNOTATIONS_FPS as f64 / f64::from(AUDIO_SAMPLE_RATE);
                (n * fps).floor() as usize
            }
        }
    }
}

impl fmt::Display for TrimMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TrimMode {
    type Err = Error;

    /// Parses `"main"` or `"v0.4.0"`.
    fn from_str(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.as_str() == s)
            .ok_or_else(|| {
                Error::invalid(
                    "trim_mode",
                    format!("expected \"main\" or \"v0.4.0\", got {s:?}"),
                )
            })
    }
}

/// Number of model windows needed for `n_samples` samples: `ceil((n + 3840) / 36164)`.
///
/// Empty audio still yields one (silent) window, exactly like Basic Pitch.
#[must_use]
pub fn window_count(n_samples: usize) -> usize {
    (n_samples + PADDING_SAMPLES).div_ceil(HOP_SIZE)
}

/// Writes window `index` of `audio` into `out`, which must hold [`AUDIO_N_SAMPLES`] samples.
///
/// This is the allocation-free building block of [`windows`]. Indices past
/// [`window_count`] produce silence.
///
/// # Errors
///
/// [`Error::WindowLength`] if `out` has the wrong length.
pub fn fill_window(audio: &[f32], index: usize, out: &mut [f32]) -> Result<()> {
    if out.len() != AUDIO_N_SAMPLES {
        return Err(Error::WindowLength {
            expected: AUDIO_N_SAMPLES,
            actual: out.len(),
        });
    }
    out.fill(0.0);
    // Window `index` covers [start, start + AUDIO_N_SAMPLES) of the padded signal, whose
    // first PADDING_SAMPLES samples are zeros.
    let start = index.saturating_mul(HOP_SIZE);
    let end = start.saturating_add(AUDIO_N_SAMPLES);
    let audio_start = start.saturating_sub(PADDING_SAMPLES);
    let audio_end = end.saturating_sub(PADDING_SAMPLES).min(audio.len());
    if audio_start < audio_end {
        let offset = (audio_start + PADDING_SAMPLES) - start;
        out[offset..offset + (audio_end - audio_start)]
            .copy_from_slice(&audio[audio_start..audio_end]);
    }
    Ok(())
}

/// Iterates over the model windows of `audio` (mono, 22 050 Hz).
///
/// Every item is a fresh `Vec` of [`AUDIO_N_SAMPLES`] samples. Use [`fill_window`] to
/// reuse a buffer instead.
#[must_use]
pub fn windows(audio: &[f32]) -> Windows<'_> {
    Windows {
        audio,
        next: 0,
        count: window_count(audio.len()),
    }
}

/// Iterator returned by [`windows`].
#[derive(Clone, Debug)]
pub struct Windows<'a> {
    audio: &'a [f32],
    next: usize,
    count: usize,
}

impl Iterator for Windows<'_> {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Vec<f32>> {
        if self.next >= self.count {
            return None;
        }
        let mut w = vec![0.0; AUDIO_N_SAMPLES];
        // The buffer has the right length by construction.
        fill_window(self.audio, self.next, &mut w).ok()?;
        self.next += 1;
        Some(w)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.count - self.next;
        (n, Some(n))
    }
}

impl ExactSizeIterator for Windows<'_> {}
impl std::iter::FusedIterator for Windows<'_> {}

/// Stitches stacked window outputs into one matrix covering `n_samples` of audio.
///
/// `raw` holds the outputs of consecutive windows stacked along the frame axis
/// (`n_windows * 172` frames, any number of bins). The first and last 15 frames of each
/// window are dropped, the rest is concatenated and trimmed according to `trim`.
///
/// # Errors
///
/// [`Error::Shape`] if the number of frames of `raw` is not a multiple of 172.
pub fn unwrap_output(
    raw: &Posteriorgram,
    n_samples: usize,
    trim: TrimMode,
) -> Result<Posteriorgram> {
    if !raw.n_frames().is_multiple_of(ANNOT_N_FRAMES) {
        return Err(Error::shape(
            "stacked window outputs",
            format!("a multiple of {ANNOT_N_FRAMES} frames"),
            raw.n_frames(),
        ));
    }
    let n_bins = raw.n_bins();
    if n_bins == 0 {
        return Ok(Posteriorgram::default());
    }
    let n_windows = raw.n_frames() / ANNOT_N_FRAMES;
    // Python slices past the end silently; the trim never exceeds the frames available
    // for the matching audio length, but a caller may pass fewer windows.
    let n_keep = trim
        .frames_for(n_samples)
        .min(n_windows * FRAMES_PER_WINDOW);
    let mut data = Vec::with_capacity(n_keep * n_bins);
    let src = raw.as_slice();
    let mut kept = 0;
    for w in 0..n_windows {
        if kept == n_keep {
            break;
        }
        let first = w * ANNOT_N_FRAMES + N_OVERLAP_FRAMES_PER_SIDE;
        let take = FRAMES_PER_WINDOW.min(n_keep - kept);
        data.extend_from_slice(&src[first * n_bins..(first + take) * n_bins]);
        kept += take;
    }
    Posteriorgram::from_vec(data, n_bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_count_matches_python_range() {
        // len(range(0, n + 3840, 36164))
        for n in [0, 1, 32_323, 32_324, 32_325, 22_050, 220_500, 1_323_000] {
            let expected = (0..n + PADDING_SAMPLES).step_by(HOP_SIZE).count();
            assert_eq!(window_count(n), expected, "n = {n}");
        }
    }

    #[test]
    fn windows_are_padded_slices() {
        let audio: Vec<f32> = (1..=80_000).map(|i| i as f32).collect();
        let ws: Vec<_> = windows(&audio).collect();
        assert_eq!(ws.len(), 3);
        assert!(ws[0][..PADDING_SAMPLES].iter().all(|&x| x == 0.0));
        assert_eq!(ws[0][PADDING_SAMPLES], 1.0);
        assert_eq!(ws[1][0], audio[HOP_SIZE - PADDING_SAMPLES]);
        let tail = 80_000 + PADDING_SAMPLES - 2 * HOP_SIZE;
        assert_eq!(ws[2][tail - 1], 80_000.0);
        assert!(ws[2][tail..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn fill_window_checks_length_and_handles_far_indices() {
        let mut short = vec![0.0; 3];
        assert!(fill_window(&[1.0], 0, &mut short).is_err());
        let mut w = vec![1.0; AUDIO_N_SAMPLES];
        fill_window(&[1.0], usize::MAX, &mut w).unwrap();
        assert!(w.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn trim_formulas_match_the_measured_table() {
        for (n, v040, main) in [
            (22_050, 86, 86),
            (44_100, 172, 173),
            (220_500, 860, 865),
            (1_323_000, 5160, 5194),
        ] {
            assert_eq!(TrimMode::V0_4_0.frames_for(n), v040);
            assert_eq!(TrimMode::Main.frames_for(n), main);
        }
    }

    #[test]
    fn trim_mode_text_round_trips() {
        for mode in TrimMode::ALL {
            assert_eq!(mode.to_string().parse::<TrimMode>().unwrap(), mode);
        }
        assert!("V0_4_0".parse::<TrimMode>().is_err());
        #[cfg(feature = "serde")]
        {
            assert_eq!(
                serde_json::to_string(&TrimMode::V0_4_0).unwrap(),
                "\"v0.4.0\""
            );
            assert_eq!(
                serde_json::from_str::<TrimMode>("\"main\"").unwrap(),
                TrimMode::Main
            );
        }
    }

    #[test]
    fn unwrap_drops_overlaps_and_trims() {
        let n_windows = 2;
        let data: Vec<f32> = (0..n_windows * ANNOT_N_FRAMES).map(|i| i as f32).collect();
        let raw = Posteriorgram::from_vec(data, 1).unwrap();
        let out = unwrap_output(&raw, 44_100, TrimMode::Main).unwrap();
        assert_eq!(out.n_frames(), 173);
        assert_eq!(out[(0, 0)], 15.0);
        assert_eq!(out[(141, 0)], 156.0);
        assert_eq!(out[(142, 0)], 187.0);
        assert!(unwrap_output(&Posteriorgram::zeros(171, 1), 10, TrimMode::Main).is_err());
        assert!(unwrap_output(&raw, 0, TrimMode::Main).unwrap().is_empty());
    }
}

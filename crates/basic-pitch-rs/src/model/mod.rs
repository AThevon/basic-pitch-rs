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
// Ported from spotify/basic-pitch v0.4.0. Changes: translated to Rust, and inference
// goes through a backend trait instead of the TensorFlow, CoreML, TFLite and ONNX
// wrappers of the Python package.

//! Running the Basic Pitch neural network.
//!
//! The network (`nmp.onnx`, from spotify/basic-pitch v0.4.0) takes one window of
//! [`AUDIO_N_SAMPLES`] samples and returns 172 frames of note, onset and contour
//! activations. The constant-Q transform and the harmonic stacking are part of the
//! ONNX graph, so a backend only has to execute the graph.
//!
//! Two backends are provided:
//!
//! - [`TractBackend`](struct.TractBackend.html) (feature `tract`, default): pure Rust, also runs on
//!   `wasm32-unknown-unknown`. Its outputs differ from ONNX Runtime by about 1e-6.
//! - [`OrtBackend`](struct.OrtBackend.html) (feature `ort`): ONNX Runtime, faster on native targets, and
//!   bit-identical to Python's `onnxruntime` on the CPU.
//!
//! Any other runtime can be plugged in by implementing [`Backend`].

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use crate::constants::{
    ANNOT_N_FRAMES, AUDIO_N_SAMPLES, FRAMES_PER_WINDOW, N_FREQ_BINS_CONTOURS, N_FREQ_BINS_NOTES,
    N_OVERLAP_FRAMES_PER_SIDE,
};
use crate::error::{Error, Result};
use crate::posteriorgram::{Posteriorgram, Posteriorgrams};
use crate::window::{TrimMode, fill_window, window_count};

#[cfg(feature = "ort")]
mod ort;
#[cfg(feature = "tract")]
mod tract;

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use self::ort::OrtBackend;
#[cfg(feature = "tract")]
#[cfg_attr(docsrs, doc(cfg(feature = "tract")))]
pub use self::tract::TractBackend;

/// The official Basic Pitch model, `nmp.onnx` from spotify/basic-pitch v0.4.0, unmodified.
///
/// Distributed under the Apache License 2.0 by Spotify AB; see the `NOTICE` file.
#[cfg(feature = "bundled-model")]
#[cfg_attr(docsrs, doc(cfg(feature = "bundled-model")))]
pub const MODEL_ONNX: &[u8] = include_bytes!("../../model/nmp.onnx");

/// SHA-256 of the official `nmp.onnx` (the bundled `MODEL_ONNX`), as a lowercase hex string.
pub const MODEL_SHA256: &str = "2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec";

/// An engine able to run the Basic Pitch network on one window of audio.
///
/// Implementations must be deterministic and process exactly one window per call
/// (batch size 1): batching windows changes the outputs of ONNX Runtime by about 1e-5,
/// which is enough to move a note across a threshold.
///
/// ```
/// use basic_pitch::model::Backend;
/// use basic_pitch::{Posteriorgram, Posteriorgrams, Result};
///
/// /// A backend that hears nothing, handy in tests.
/// struct Deaf;
///
/// impl Backend for Deaf {
///     fn predict_window(&self, _window: &[f32]) -> Result<Posteriorgrams> {
///         Posteriorgrams::new(
///             Posteriorgram::zeros(172, 88),
///             Posteriorgram::zeros(172, 88),
///             Posteriorgram::zeros(172, 264),
///         )
///     }
/// }
///
/// let raw = basic_pitch::model::run_windows(&Deaf, &[0.0; 50_000])?;
/// assert_eq!(raw.n_frames(), 2 * 172);
/// # Ok::<(), basic_pitch::Error>(())
/// ```
pub trait Backend: Send + Sync {
    /// Runs the network on one window of [`AUDIO_N_SAMPLES`] samples.
    ///
    /// Returns 172 frames of note (88 bins), onset (88 bins) and contour (264 bins)
    /// activations.
    ///
    /// # Errors
    ///
    /// [`Error::WindowLength`] if `window` has the wrong length, and
    /// [`Error::Backend`] if the engine fails.
    fn predict_window(&self, window: &[f32]) -> Result<Posteriorgrams>;
}

impl<B: Backend + ?Sized> Backend for Box<B> {
    fn predict_window(&self, window: &[f32]) -> Result<Posteriorgrams> {
        (**self).predict_window(window)
    }
}

impl<B: Backend + ?Sized> Backend for std::sync::Arc<B> {
    fn predict_window(&self, window: &[f32]) -> Result<Posteriorgrams> {
        (**self).predict_window(window)
    }
}

/// Runs `backend` on every window of `audio` and stacks the raw outputs.
///
/// The result has `window_count(audio.len()) * 172` frames, ready for
/// [`unwrap_output`](crate::window::unwrap_output). This is `run_inference` without the
/// unwrapping.
///
/// # Errors
///
/// Propagates the errors of the backend, and returns [`Error::Shape`] if it returns
/// matrices of the wrong shape.
pub fn run_windows<B: Backend + ?Sized>(backend: &B, audio: &[f32]) -> Result<Posteriorgrams> {
    run_windows_parallel(backend, audio, NonZeroUsize::MIN)
}

/// [`run_windows`] on up to `threads` threads.
///
/// The windows are independent, so the result is identical to [`run_windows`] for any
/// number of threads. Where threads cannot be spawned (`wasm32-unknown-unknown`), the
/// calling thread does all the work.
///
/// # Errors
///
/// Same as [`run_windows`]. When several windows fail, the error of the first one is
/// returned.
pub fn run_windows_parallel<B: Backend + ?Sized>(
    backend: &B,
    audio: &[f32],
    threads: NonZeroUsize,
) -> Result<Posteriorgrams> {
    let n_frames = window_count(audio.len()) * ANNOT_N_FRAMES;
    let layout = Layout {
        frames_per_window: ANNOT_N_FRAMES,
        first_frame: 0,
        n_frames,
    };
    run_layout(backend, audio, threads, layout)
}

/// Runs the model on `audio` and returns the unwrapped posteriorgrams
/// (`run_inference`): one frame every ~11.6 ms, aligned with the audio.
///
/// The result is identical to [`run_windows`] followed by
/// [`unwrap_output`](crate::window::unwrap_output), but the frames are written in
/// place, without keeping the stacked window outputs.
///
/// # Errors
///
/// Propagates the errors of the backend.
pub fn run_inference<B: Backend + ?Sized>(
    backend: &B,
    audio: &[f32],
    trim: TrimMode,
) -> Result<Posteriorgrams> {
    run_inference_parallel(backend, audio, trim, NonZeroUsize::MIN)
}

/// [`run_inference`] on up to `threads` threads, with an identical result.
///
/// # Errors
///
/// Same as [`run_windows_parallel`].
pub fn run_inference_parallel<B: Backend + ?Sized>(
    backend: &B,
    audio: &[f32],
    trim: TrimMode,
    threads: NonZeroUsize,
) -> Result<Posteriorgrams> {
    let n_windows = window_count(audio.len());
    // Same bound as unwrap_output: never more frames than the windows hold.
    let n_frames = trim
        .frames_for(audio.len())
        .min(n_windows * FRAMES_PER_WINDOW);
    let layout = Layout {
        frames_per_window: FRAMES_PER_WINDOW,
        first_frame: N_OVERLAP_FRAMES_PER_SIDE,
        n_frames,
    };
    run_layout(backend, audio, threads, layout)
}

/// Which frames of each window output are kept, and how many frames the result has.
#[derive(Clone, Copy)]
struct Layout {
    /// Frames kept per window: window `w` fills the output from frame
    /// `w * frames_per_window`.
    frames_per_window: usize,
    /// First frame of a window output that is kept.
    first_frame: usize,
    /// Frames of the result; the last windows may contribute fewer frames, or none.
    n_frames: usize,
}

/// A contiguous range of windows, with the part of the output it writes.
struct Job<'a> {
    windows: Range<usize>,
    note: &'a mut [f32],
    onset: &'a mut [f32],
    contour: &'a mut [f32],
}

impl Job<'_> {
    fn run<B: Backend + ?Sized>(self, backend: &B, audio: &[f32], layout: Layout) -> Result<()> {
        let mut window = vec![0.0_f32; AUDIO_N_SAMPLES];
        let first_out = self.windows.start * layout.frames_per_window;
        for w in self.windows {
            fill_window(audio, w, &mut window)?;
            let out = backend.predict_window(&window)?;
            out.validate()?;
            if out.n_frames() != ANNOT_N_FRAMES {
                return Err(Error::shape(
                    "window output frames",
                    ANNOT_N_FRAMES,
                    out.n_frames(),
                ));
            }
            let start = w * layout.frames_per_window;
            let take = layout
                .frames_per_window
                .min(layout.n_frames.saturating_sub(start));
            if take == 0 {
                // A last window that only holds trimmed frames.
                continue;
            }
            let rows = layout.first_frame..layout.first_frame + take;
            let at = start - first_out;
            for (dst, src, bins) in [
                (&mut *self.note, &out.note, N_FREQ_BINS_NOTES),
                (&mut *self.onset, &out.onset, N_FREQ_BINS_NOTES),
                (&mut *self.contour, &out.contour, N_FREQ_BINS_CONTOURS),
            ] {
                dst[at * bins..(at + take) * bins]
                    .copy_from_slice(&src.as_slice()[rows.start * bins..rows.end * bins]);
            }
        }
        Ok(())
    }
}

fn run_layout<B: Backend + ?Sized>(
    backend: &B,
    audio: &[f32],
    threads: NonZeroUsize,
    layout: Layout,
) -> Result<Posteriorgrams> {
    let n_windows = window_count(audio.len());
    let mut note = vec![0.0_f32; layout.n_frames * N_FREQ_BINS_NOTES];
    let mut onset = vec![0.0_f32; layout.n_frames * N_FREQ_BINS_NOTES];
    let mut contour = vec![0.0_f32; layout.n_frames * N_FREQ_BINS_CONTOURS];

    // One job per thread, each writing its own frames of the output.
    let threads = threads.get().min(n_windows).max(1);
    let per_job = n_windows.div_ceil(threads).max(1);
    let mut jobs = Vec::with_capacity(threads);
    let (mut rest_note, mut rest_onset, mut rest_contour) =
        (&mut note[..], &mut onset[..], &mut contour[..]);
    for start in (0..n_windows).step_by(per_job) {
        let end = (start + per_job).min(n_windows);
        let out_frames = |w: usize| (w * layout.frames_per_window).min(layout.n_frames);
        let frames = out_frames(end) - out_frames(start);
        let (job_note, tail) = rest_note.split_at_mut(frames * N_FREQ_BINS_NOTES);
        rest_note = tail;
        let (job_onset, tail) = rest_onset.split_at_mut(frames * N_FREQ_BINS_NOTES);
        rest_onset = tail;
        let (job_contour, tail) = rest_contour.split_at_mut(frames * N_FREQ_BINS_CONTOURS);
        rest_contour = tail;
        jobs.push(Mutex::new(Some(Job {
            windows: start..end,
            note: job_note,
            onset: job_onset,
            contour: job_contour,
        })));
    }

    // Workers take jobs in order; the calling thread works too, and does everything
    // when no thread can be spawned. The error of the earliest failing job wins.
    // Jobs after a failed one are skipped, never jobs before it, so the reported error
    // does not depend on the scheduling.
    let next = AtomicUsize::new(0);
    let first_failed = AtomicUsize::new(usize::MAX);
    let errors = Mutex::new(Vec::new());
    let work = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= jobs.len() || i > first_failed.load(Ordering::Relaxed) {
                break;
            }
            let job = jobs[i]
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            if let Some(Err(e)) = job.map(|job| job.run(backend, audio, layout)) {
                first_failed.fetch_min(i, Ordering::Relaxed);
                errors
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push((i, e));
            }
        }
    };
    std::thread::scope(|scope| {
        let workers: Vec<_> = (1..threads)
            .map_while(|n| {
                std::thread::Builder::new()
                    .name(format!("basic-pitch-{n}"))
                    .spawn_scoped(scope, work)
                    .ok()
            })
            .collect();
        work();
        for worker in workers {
            if let Err(panic) = worker.join() {
                std::panic::resume_unwind(panic);
            }
        }
    });
    let errors = errors.into_inner().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, e)) = errors.into_iter().min_by_key(|&(i, _)| i) {
        return Err(e);
    }

    Posteriorgrams::new(
        Posteriorgram::from_vec(note, N_FREQ_BINS_NOTES)?,
        Posteriorgram::from_vec(onset, N_FREQ_BINS_NOTES)?,
        Posteriorgram::from_vec(contour, N_FREQ_BINS_CONTOURS)?,
    )
}

/// Checks the window length shared by all backends.
#[cfg_attr(not(any(feature = "tract", feature = "ort")), allow(dead_code))]
pub(crate) fn check_window(window: &[f32]) -> Result<()> {
    if window.len() == AUDIO_N_SAMPLES {
        Ok(())
    } else {
        Err(Error::WindowLength {
            expected: AUDIO_N_SAMPLES,
            actual: window.len(),
        })
    }
}

/// Builds window outputs from flat backend buffers.
#[cfg_attr(not(any(feature = "tract", feature = "ort")), allow(dead_code))]
pub(crate) fn window_output(
    note: Vec<f32>,
    onset: Vec<f32>,
    contour: Vec<f32>,
) -> Result<Posteriorgrams> {
    let p = Posteriorgrams::new(
        Posteriorgram::from_vec(note, N_FREQ_BINS_NOTES)?,
        Posteriorgram::from_vec(onset, N_FREQ_BINS_NOTES)?,
        Posteriorgram::from_vec(contour, N_FREQ_BINS_CONTOURS)?,
    )?;
    if p.n_frames() == ANNOT_N_FRAMES {
        Ok(p)
    } else {
        Err(Error::shape(
            "window output frames",
            ANNOT_N_FRAMES,
            p.n_frames(),
        ))
    }
}

#[cfg(all(test, feature = "bundled-model"))]
mod tests {
    use sha2::{Digest, Sha256};

    #[test]
    fn bundled_model_matches_its_checksum() {
        let digest = Sha256::digest(super::MODEL_ONNX);
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, super::MODEL_SHA256);
    }
}

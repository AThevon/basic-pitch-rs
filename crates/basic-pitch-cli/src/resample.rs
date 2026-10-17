// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Resampling to 22 050 Hz.
//!
//! Basic Pitch resamples with libsoxr in its `HQ` quality (`librosa.load`'s default
//! `res_type="soxr_hq"`). Two resamplers are available:
//!
//! - [`Resampler::Rubato`]: pure Rust ([rubato](https://docs.rs/rubato)), always
//!   available. Its FFT resampler is configured so that its anti-aliasing filter is
//!   close to soxr's: from 44.1 or 48 kHz, on speech and music, the difference is around
//!   1e-6. Upsampling from 8 to 16 kHz is further from soxr (up to about 5e-3 near the
//!   Nyquist frequency of the file), which can change a few notes or pitch bends.
//! - [`Resampler::Soxr`]: libsoxr itself, with the same recipe as librosa
//!   (`SOXR_HQ`, which is `SOXR_20_BITQ`, in `float32`), for sample-level parity. It needs
//!   the `soxr` feature, which compiles and statically links libsoxr
//!   (LGPL-2.1-or-later).
//!
//! Both produce exactly `ceil(n * 22050 / sr)` samples, like `librosa.resample(fix=True)`.

use std::fmt;

use basic_pitch::constants::AUDIO_SAMPLE_RATE;

use crate::audio::AudioError;

/// A resampling algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Resampler {
    /// rubato's FFT resampler (pure Rust, always available).
    Rubato,
    /// libsoxr in `HQ` quality, like librosa (needs the `soxr` feature).
    Soxr,
}

impl Resampler {
    /// The default: soxr when it is compiled in, rubato otherwise.
    pub const DEFAULT: Self = if cfg!(feature = "soxr") {
        Self::Soxr
    } else {
        Self::Rubato
    };

    /// Whether this resampler is compiled in.
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Self::Rubato => true,
            Self::Soxr => cfg!(feature = "soxr"),
        }
    }

    /// The name used on the command line and in the JSON output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Rubato => "rubato",
            Self::Soxr => "soxr",
        }
    }
}

impl Default for Resampler {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl fmt::Display for Resampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Number of samples librosa produces when resampling `n` samples from `rate` Hz to
/// 22 050 Hz: `int(np.ceil(n * (22050 / rate)))`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "mirrors Python's int() of a non-negative float"
)]
pub fn output_len(n: usize, rate: u32) -> usize {
    let ratio = f64::from(AUDIO_SAMPLE_RATE) / f64::from(rate);
    (n as f64 * ratio).ceil() as usize
}

/// Resamples mono `samples` from `rate` Hz to 22 050 Hz.
///
/// Returns a copy when `rate` is already 22 050 Hz.
///
/// # Errors
///
/// [`AudioError::SoxrUnavailable`] when soxr is requested but not compiled in, and
/// [`AudioError::Resample`] if the resampler fails (for instance on a zero rate).
pub fn resample(samples: &[f32], rate: u32, resampler: Resampler) -> Result<Vec<f32>, AudioError> {
    if rate == 0 {
        return Err(AudioError::Resample("the sample rate is 0 Hz".into()));
    }
    if rate == AUDIO_SAMPLE_RATE {
        return Ok(samples.to_vec());
    }
    let n_out = output_len(samples.len(), rate);
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = match resampler {
        Resampler::Rubato => rubato_fft::resample(samples, rate)?,
        #[cfg(feature = "soxr")]
        Resampler::Soxr => soxr::resample(samples, rate, n_out)?,
        #[cfg(not(feature = "soxr"))]
        Resampler::Soxr => return Err(AudioError::SoxrUnavailable),
    };
    // librosa.util.fix_length: zero-pad or cut to the expected length.
    out.resize(n_out, 0.0);
    Ok(out)
}

mod rubato_fft {
    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
    use rubato::{Fft, FixedSync, Indexing, Resampler as _, WindowFunction};

    use super::AUDIO_SAMPLE_RATE;
    use crate::audio::AudioError;

    /// Target FFT block size, in frames, on the lower-rate side.
    ///
    /// The cutoff of rubato's anti-aliasing filter follows the block size. Around 316
    /// frames it is closest to soxr's `HQ` filter: on the vocadito recordings, the
    /// maximum difference with librosa drops from 9e-6 (1024 frames) to 6e-7.
    const TARGET_BLOCK: usize = 316;

    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    /// Number of minimal blocks per FFT block.
    ///
    /// Both FFT sizes must be even: rubato reports a delay of half the output block,
    /// rounded down, so an odd block would shift the output by half a sample.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "small positive block counts"
    )]
    pub(super) fn blocks(rate_in: usize, rate_out: usize) -> usize {
        let g = gcd(rate_in, rate_out);
        let (min_in, min_out) = (rate_in / g, rate_out / g);
        let smallest = min_in.min(min_out);
        let mut blocks = ((TARGET_BLOCK as f64 / smallest as f64).round() as usize).max(1);
        if (min_in % 2 == 1 || min_out % 2 == 1) && blocks % 2 == 1 {
            blocks += 1;
        }
        blocks
    }

    fn make(rate: u32) -> Result<Fft<f64>, AudioError> {
        let (rate_in, rate_out) = (rate as usize, AUDIO_SAMPLE_RATE as usize);
        let chunk = blocks(rate_in, rate_out) * (rate_in / gcd(rate_in, rate_out));
        Fft::<f64>::new_custom(
            rate_in,
            rate_out,
            chunk,
            1,
            1,
            WindowFunction::BlackmanHarris2,
            FixedSync::Input,
        )
        .map_err(|e| AudioError::Resample(e.to_string()))
    }

    /// Resamples chunk by chunk, with the same calls as rubato's `process_all` (so the
    /// same output, bit for bit), but without converting the whole signal to `f64`
    /// first: the memory used is the output plus a few blocks.
    pub(super) fn resample(samples: &[f32], rate: u32) -> Result<Vec<f32>, AudioError> {
        let err = |e: &dyn std::fmt::Display| AudioError::Resample(e.to_string());
        let mut resampler = make(rate)?;
        let chunk = resampler.input_frames_next();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "the length computed by rubato's process_all"
        )]
        let expected = (resampler.resample_ratio() * samples.len() as f64).ceil() as usize;
        let delay = resampler.output_delay();

        let mut input = vec![vec![0.0_f64; chunk]];
        let mut output = vec![vec![0.0_f64; resampler.output_frames_max()]];
        let mut result = Vec::with_capacity(expected);
        let mut produced = 0;
        // Appends what the resampler wrote, minus its startup delay.
        let mut keep = |output: &[f64], n: usize, produced: &mut usize| {
            for &x in &output[..n] {
                if *produced >= delay && result.len() < expected {
                    #[allow(clippy::cast_possible_truncation, reason = "f64 to f32 audio samples")]
                    result.push(x as f32);
                }
                *produced += 1;
            }
        };
        let mut run = |input: &[Vec<f64>],
                       output: &mut [Vec<f64>],
                       partial_len: Option<usize>|
         -> Result<usize, AudioError> {
            let frames = input[0].len();
            let adapter_in = SequentialSliceOfVecs::new(input, 1, frames).map_err(|e| err(&e))?;
            let frames_out = output[0].len();
            let mut adapter_out =
                SequentialSliceOfVecs::new_mut(output, 1, frames_out).map_err(|e| err(&e))?;
            let indexing = Indexing {
                partial_len,
                ..Indexing::default()
            };
            let (_, n_out) = resampler
                .process_into_buffer(&adapter_in, &mut adapter_out, Some(&indexing))
                .map_err(|e| err(&e))?;
            Ok(n_out)
        };

        let mut pos = 0;
        while samples.len() - pos > chunk {
            for (dst, &src) in input[0].iter_mut().zip(&samples[pos..pos + chunk]) {
                *dst = f64::from(src);
            }
            let n = run(&input, &mut output, None)?;
            keep(&output[0], n, &mut produced);
            pos += chunk;
        }
        let left = samples.len() - pos;
        if left > 0 {
            input[0].fill(0.0);
            for (dst, &src) in input[0].iter_mut().zip(&samples[pos..]) {
                *dst = f64::from(src);
            }
            let n = run(&input, &mut output, Some(left))?;
            keep(&output[0], n, &mut produced);
        }
        // Flush the filter with silence until the delayed output is complete.
        while produced < expected + delay {
            let n = run(&input, &mut output, Some(0))?;
            keep(&output[0], n, &mut produced);
        }
        Ok(result)
    }

    /// The previous implementation, with rubato's `process_all` on a full `f64` copy.
    #[cfg(test)]
    pub(super) fn resample_all(samples: &[f32], rate: u32) -> Result<Vec<f32>, AudioError> {
        let err = |e: &dyn std::fmt::Display| AudioError::Resample(e.to_string());
        let mut resampler = make(rate)?;
        let input = vec![samples.iter().copied().map(f64::from).collect::<Vec<_>>()];
        let adapter = SequentialSliceOfVecs::new(&input, 1, samples.len()).map_err(|e| err(&e))?;
        let output = resampler
            .process_all(&adapter, samples.len(), None)
            .map_err(|e| err(&e))?;
        #[allow(clippy::cast_possible_truncation, reason = "f64 to f32 audio samples")]
        Ok(output.take_data().into_iter().map(|x| x as f32).collect())
    }
}

#[cfg(feature = "soxr")]
#[allow(unsafe_code, reason = "FFI calls into libsoxr")]
mod soxr {
    use std::ffi::CStr;
    use std::ptr;

    use soxr_sys::{
        SOXR_20_BITQ, soxr_datatype_t_SOXR_FLOAT32_I, soxr_io_spec, soxr_oneshot, soxr_quality_spec,
    };

    use super::AUDIO_SAMPLE_RATE;
    use crate::audio::AudioError;

    /// Resamples with `soxr_oneshot`, in the quality librosa uses (`soxr_hq`).
    pub(super) fn resample(
        samples: &[f32],
        rate: u32,
        n_out: usize,
    ) -> Result<Vec<f32>, AudioError> {
        let mut out = vec![0.0_f32; n_out];
        let mut done = 0_usize;
        // SAFETY: the specs are plain values built by libsoxr; the input and output
        // pointers are valid for `samples.len()` and `n_out` f32 values (one channel),
        // which is what the lengths passed to soxr_oneshot say.
        let error = unsafe {
            let quality = soxr_quality_spec(SOXR_20_BITQ.into(), 0);
            let io = soxr_io_spec(
                soxr_datatype_t_SOXR_FLOAT32_I,
                soxr_datatype_t_SOXR_FLOAT32_I,
            );
            soxr_oneshot(
                f64::from(rate),
                f64::from(AUDIO_SAMPLE_RATE),
                1,
                samples.as_ptr().cast(),
                samples.len(),
                ptr::null_mut(),
                out.as_mut_ptr().cast(),
                n_out,
                &raw mut done,
                &raw const io,
                &raw const quality,
                ptr::null(),
            )
        };
        if !error.is_null() {
            // SAFETY: a non-null soxr_error_t is a static NUL-terminated message.
            let message = unsafe { CStr::from_ptr(error) }.to_string_lossy();
            return Err(AudioError::Resample(format!("libsoxr: {message}")));
        }
        out.truncate(done);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, hz: f64, seconds: f64) -> Vec<f32> {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "test signal"
        )]
        let n = (f64::from(rate) * seconds) as usize;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            reason = "test signal"
        )]
        (0..n)
            .map(|i| (0.5 * (std::f64::consts::TAU * hz * i as f64 / f64::from(rate)).sin()) as f32)
            .collect()
    }

    /// Delay of `y` relative to a 0.5-amplitude sine of `hz` at 22 050 Hz, in samples,
    /// by least squares on the interior.
    #[allow(clippy::cast_precision_loss, reason = "test signal")]
    fn delay(y: &[f32], hz: f64) -> f64 {
        let w = std::f64::consts::TAU * hz / 22_050.0;
        let (mut ss, mut sc, mut cc, mut ys, mut yc) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (i, &v) in y.iter().enumerate().take(y.len() - 3000).skip(3000) {
            let (s, c) = (w * i as f64).sin_cos();
            ss += s * s;
            sc += s * c;
            cc += c * c;
            ys += f64::from(v) * s;
            yc += f64::from(v) * c;
        }
        let det = ss * cc - sc * sc;
        let a = (ys * cc - yc * sc) / det;
        let b = (yc * ss - ys * sc) / det;
        (-b).atan2(a) / w
    }

    #[test]
    fn output_length_matches_librosa() {
        assert_eq!(output_len(330_750, 44_100), 165_375);
        assert_eq!(output_len(480_000, 48_000), 220_500);
        assert_eq!(output_len(1, 48_000), 1);
        assert_eq!(output_len(0, 48_000), 0);
        assert_eq!(output_len(100, 8_000), 276); // 275.625 rounded up
    }

    #[test]
    fn same_rate_is_a_copy() {
        let x = sine(22_050, 440.0, 0.1);
        assert_eq!(resample(&x, 22_050, Resampler::Rubato).unwrap(), x);
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert!(resample(&[], 44_100, Resampler::Rubato).unwrap().is_empty());
    }

    #[test]
    fn zero_rate_is_an_error() {
        assert!(resample(&[0.0], 0, Resampler::Rubato).is_err());
    }

    #[test]
    fn fft_blocks_are_even_when_needed() {
        for rate in [
            8_000, 11_025, 16_000, 22_051, 24_000, 32_000, 44_056, 44_100, 48_000, 96_000, 192_000,
        ] {
            let g = rubato_fft::blocks(rate, 22_050);
            let gcd = (1..=rate.min(22_050))
                .rev()
                .find(|d| rate % d == 0 && 22_050 % d == 0)
                .unwrap();
            assert_eq!((g * rate / gcd) % 2, 0, "{rate}: odd input block");
            assert_eq!((g * 22_050 / gcd) % 2, 0, "{rate}: odd output block");
        }
    }

    /// The chunked rubato path gives exactly what rubato's `process_all` gives.
    #[test]
    fn chunked_rubato_matches_process_all() {
        for rate in [
            8_000, 11_025, 16_000, 32_000, 44_100, 48_000, 96_000, 22_051,
        ] {
            for seconds in [0.0001, 0.01, 0.5, 2.3] {
                let x = sine(rate, 987.6, seconds);
                if x.is_empty() {
                    continue;
                }
                let streamed = rubato_fft::resample(&x, rate).unwrap();
                let all = rubato_fft::resample_all(&x, rate).unwrap();
                assert_eq!(streamed.len(), all.len(), "{rate} Hz, {seconds} s");
                assert!(
                    streamed
                        .iter()
                        .zip(&all)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{rate} Hz, {seconds} s"
                );
            }
        }
    }

    /// rubato must not shift the signal, whatever the input rate.
    #[test]
    fn rubato_is_aligned_at_common_rates() {
        for rate in [
            8_000, 16_000, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000, 22_051,
        ] {
            let x = sine(rate, 1234.5, 1.0);
            let y = resample(&x, rate, Resampler::Rubato).unwrap();
            assert_eq!(y.len(), output_len(x.len(), rate));
            let d = delay(&y, 1234.5);
            assert!(d.abs() < 1e-3, "{rate} Hz: delay of {d} samples");
        }
    }

    /// A plucked-string-like test signal: decaying harmonics, up to the Nyquist frequency.
    #[cfg(feature = "soxr")]
    fn pluck(rate: u32) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss, reason = "test signal")]
        (0..rate as usize * 2)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let mut x = 0.0;
                let mut k = 1.0;
                while 196.0 * k < f64::from(rate) / 2.0 {
                    x += (std::f64::consts::TAU * 196.0 * k * t).sin() * (-3.0 * k * t).exp() / k;
                    k += 1.0;
                }
                #[allow(clippy::cast_possible_truncation, reason = "test signal")]
                let y = (0.3 * x) as f32;
                y
            })
            .collect()
    }

    /// Measured distance between rubato and soxr, as documented in `--help` and the
    /// module docs: close when downsampling, further when upsampling.
    #[cfg(feature = "soxr")]
    #[test]
    fn rubato_stays_close_to_soxr() {
        for (rate, bound) in [
            (8_000, 1e-3),
            (11_025, 1e-3),
            (16_000, 1e-3),
            (44_100, 5e-5),
            (48_000, 5e-5),
            (96_000, 5e-5),
        ] {
            let x = pluck(rate);
            let a = resample(&x, rate, Resampler::Rubato).unwrap();
            let b = resample(&x, rate, Resampler::Soxr).unwrap();
            let max = a
                .iter()
                .zip(&b)
                .skip(500)
                .take(a.len() - 1000)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max);
            println!("{rate} Hz: max |rubato - soxr| = {max:.2e}");
            assert!(f64::from(max) < bound, "{rate} Hz: {max:e} >= {bound:e}");
        }
    }

    #[cfg(feature = "soxr")]
    #[test]
    fn soxr_is_aligned_at_common_rates() {
        for rate in [8_000, 16_000, 44_100, 48_000, 96_000] {
            let x = sine(rate, 1234.5, 1.0);
            let y = resample(&x, rate, Resampler::Soxr).unwrap();
            assert_eq!(y.len(), output_len(x.len(), rate));
            let d = delay(&y, 1234.5);
            assert!(d.abs() < 1e-3, "{rate} Hz: delay of {d} samples");
        }
    }

    #[cfg(not(feature = "soxr"))]
    #[test]
    fn soxr_without_the_feature_is_a_clear_error() {
        let err = resample(&[0.0; 10], 44_100, Resampler::Soxr).unwrap_err();
        assert!(matches!(err, AudioError::SoxrUnavailable));
        assert!(err.to_string().contains("--features soxr"));
    }
}

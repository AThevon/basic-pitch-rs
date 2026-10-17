// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Audio loading: decoding, downmixing and resampling to mono 22 050 Hz.
//!
//! [`load`] follows what Basic Pitch does with `librosa.load(path, sr=22050, mono=True)`:
//!
//! 1. decode every sample to `f32`, scaling integers like libsndfile (`i16 / 2^15`,
//!    `i24 / 2^23`, `i32 / 2^31`);
//! 2. average the channels in `f32`, *before* resampling (`np.mean(y, axis=0)`);
//! 3. for MP4, M4A, MOV and CAF files, drop the encoder's priming and padding samples,
//!    as ffmpeg and Core Audio do (see [`crate::mp4`] and [`crate::caf`]);
//! 4. reject NaN and infinite samples, like `librosa.load`;
//! 5. resample to 22 050 Hz, unless the file is already at that rate, and pad or cut the
//!    result to `ceil(n * 22050 / sr)` samples.
//!
//! The gain is never normalized: the model output depends on the absolute level.

use std::fs::File;
use std::path::{Path, PathBuf};

use basic_pitch::constants::AUDIO_SAMPLE_RATE;
use symphonia::core::audio::Channels;
use symphonia::core::codecs::audio::{AudioCodecParameters, AudioDecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::resample::{Resampler, resample};
use crate::{caf, mp4};

/// File extensions the decoder recognizes, for error messages and `--help`.
pub const SUPPORTED_FORMATS: &str = "WAV, AIFF, CAF, FLAC, MP3, Ogg Vorbis, MP4/M4A/MOV (AAC, ALAC) and Matroska/WebM (Vorbis, FLAC, PCM)";

/// Formats that Python Basic Pitch reads (through libsndfile) but this tool does not.
pub const UNSUPPORTED_FORMATS: &str = "RF64 and Sony Wave64 (W64)";

/// Why an audio file could not be loaded.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AudioError {
    /// The file could not be opened or read.
    #[error("cannot read {}", path.display())]
    Io {
        /// The input file.
        path: PathBuf,
        /// The I/O error.
        source: std::io::Error,
    },
    /// The path is a directory.
    #[error("{} is a directory, not an audio file", path.display())]
    IsDirectory {
        /// The input path.
        path: PathBuf,
    },
    /// The file is empty.
    #[error("{} is empty (0 bytes)", path.display())]
    Empty {
        /// The input file.
        path: PathBuf,
    },
    /// The container format is not recognized.
    #[error("{}: unsupported or unrecognized audio format (supported: {SUPPORTED_FORMATS})", path.display())]
    UnsupportedFormat {
        /// The input file.
        path: PathBuf,
    },
    /// The file has no audio track.
    #[error("{}: no audio track found", path.display())]
    NoAudioTrack {
        /// The input file.
        path: PathBuf,
    },
    /// The audio codec is not supported.
    #[error("{}: unsupported audio codec ({detail})", path.display())]
    UnsupportedCodec {
        /// The input file.
        path: PathBuf,
        /// What the decoder reported.
        detail: String,
    },
    /// The stream is corrupt.
    #[error("{}: cannot decode audio: {detail}", path.display())]
    Decode {
        /// The input file.
        path: PathBuf,
        /// What the decoder reported.
        detail: String,
    },
    /// The audio contains a NaN or infinite sample.
    #[error("{}: sample {index} (at {time_s:.3} s) is not finite ({value}); the file is corrupt or not audio", path.display())]
    NonFinite {
        /// The input file.
        path: PathBuf,
        /// Index of the first offending frame, at the sample rate of the file.
        index: usize,
        /// Its time in the file, in seconds.
        time_s: f64,
        /// Its value, after the downmix to mono.
        value: f32,
    },
    /// The resampler failed.
    #[error("resampling failed: {0}")]
    Resample(String),
    /// The requested resampler is not compiled in.
    #[error(
        "this build does not include the soxr resampler; reinstall with `--features soxr`, or use `--resampler rubato`"
    )]
    SoxrUnavailable,
}

/// Decoded audio, before resampling: one `f32` channel per plane.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    /// Sample rate of the file, in Hz.
    pub sample_rate: u32,
    /// Number of channels of the file.
    pub channels: usize,
    /// The channels averaged to mono, in `f32` (see [`downmix`]).
    pub mono: Vec<f32>,
    /// Short name of the codec, such as `pcm_s16le`, `flac` or `aac`.
    pub codec: String,
    /// Packets that could not be decoded and were skipped.
    pub skipped_packets: usize,
}

impl Decoded {
    /// Duration of the file, in seconds.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "audio lengths are far below 2^52 samples"
    )]
    pub fn duration_s(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.mono.len() as f64 / f64::from(self.sample_rate)
        }
    }
}

/// Audio ready for the model: mono, 22 050 Hz.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedAudio {
    /// Mono samples at 22 050 Hz.
    pub samples: Vec<f32>,
    /// The resampler used, or `None` when the file was already at 22 050 Hz.
    pub resampler: Option<Resampler>,
    /// The decoded file.
    pub source: SourceInfo,
}

/// What the input file contained.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceInfo {
    /// Sample rate of the file, in Hz.
    pub sample_rate: u32,
    /// Number of channels of the file.
    pub channels: usize,
    /// Number of frames (samples per channel) of the file.
    pub frames: usize,
    /// Duration of the file, in seconds.
    pub duration_s: f64,
    /// Short name of the codec.
    pub codec: String,
    /// Packets that could not be decoded and were skipped.
    pub skipped_packets: usize,
}

/// Decodes `path`, downmixes it to mono and resamples it to 22 050 Hz.
///
/// # Errors
///
/// See [`AudioError`].
pub fn load(path: &Path, resampler: Resampler) -> Result<LoadedAudio, AudioError> {
    let decoded = decode(path)?;
    // Checked before resampling: the resamplers would spread a NaN over its neighbours,
    // or fail.
    if let Some((index, &value)) = decoded
        .mono
        .iter()
        .enumerate()
        .find(|(_, x)| !x.is_finite())
    {
        #[allow(
            clippy::cast_precision_loss,
            reason = "sample indices are far below 2^52"
        )]
        let time_s = index as f64 / f64::from(decoded.sample_rate);
        return Err(AudioError::NonFinite {
            path: path.to_owned(),
            index,
            time_s,
            value,
        });
    }
    let source = SourceInfo {
        sample_rate: decoded.sample_rate,
        channels: decoded.channels,
        frames: decoded.mono.len(),
        duration_s: decoded.duration_s(),
        codec: decoded.codec.clone(),
        skipped_packets: decoded.skipped_packets,
    };
    let (samples, used) = if decoded.sample_rate == AUDIO_SAMPLE_RATE {
        (decoded.mono, None)
    } else {
        let samples = resample(&decoded.mono, decoded.sample_rate, resampler)?;
        (samples, Some(resampler))
    };
    Ok(LoadedAudio {
        samples,
        resampler: used,
        source,
    })
}

/// Decodes the first audio track of `path` and averages its channels.
///
/// Packets that fail to decode are skipped and counted in
/// [`Decoded::skipped_packets`], like most players do.
///
/// # Errors
///
/// See [`AudioError`].
pub fn decode(path: &Path) -> Result<Decoded, AudioError> {
    let decode_error = |detail: String| AudioError::Decode {
        path: path.to_owned(),
        detail,
    };
    let (mut format, tracks) = open(path)?;

    // The default audio track first, then the others: an iPhone .mov recorded with
    // spatial audio can list an APAC track that has no decoder before its AAC track.
    let mut last_error = None;
    let mut chosen = None;
    for (track_id, params) in tracks {
        match symphonia::default::get_codecs()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
        {
            Ok(decoder) => {
                chosen = Some((track_id, params, decoder));
                break;
            }
            Err(err) => last_error = Some(err.to_string()),
        }
    }
    let (track_id, params, mut decoder) = chosen.ok_or_else(|| AudioError::UnsupportedCodec {
        path: path.to_owned(),
        detail: last_error.unwrap_or_default(),
    })?;
    let codec = decoder.codec_info().short_name.to_owned();

    let mut sample_rate = params.sample_rate;
    let mut channels = params.channels.as_ref().map(Channels::count);
    let mut mono = Vec::new();
    let mut planes: Vec<Vec<f32>> = Vec::new();
    let mut skipped_packets = 0;

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            // A chained stream (a new Ogg stream after the first one): stop at the first.
            Ok(None) | Err(SymphoniaError::ResetRequired) => break,
            // Some containers end without a clean end-of-stream marker.
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(err) => return Err(decode_error(err.to_string())),
        };
        if packet.track_id != track_id {
            continue;
        }
        let buffer = match decoder.decode(&packet) {
            Ok(buffer) => buffer,
            Err(SymphoniaError::DecodeError(_)) => {
                skipped_packets += 1;
                continue;
            }
            Err(err) => return Err(decode_error(err.to_string())),
        };
        let spec = buffer.spec();
        let n_channels = spec.channels().count();
        if *channels.get_or_insert(n_channels) != n_channels {
            return Err(decode_error(format!(
                "the channel count changes from {} to {n_channels}",
                channels.unwrap_or_default()
            )));
        }
        if *sample_rate.get_or_insert(spec.rate()) != spec.rate() {
            return Err(decode_error("the sample rate changes in the stream".into()));
        }
        buffer.copy_to_vecs_planar::<f32>(&mut planes);
        downmix_into(&planes, &mut mono);
    }

    let sample_rate = sample_rate
        .filter(|&rate| rate > 0)
        .ok_or_else(|| decode_error("unknown sample rate".into()))?;
    let gapless = match format.format_info().short_name {
        "isomp4" => mp4::gapless(path, sample_rate),
        "caf" => caf::gapless(path),
        _ => None,
    };
    if let Some(gapless) = gapless {
        gapless.apply(&mut mono);
    }
    Ok(Decoded {
        sample_rate,
        channels: channels.unwrap_or(1),
        mono,
        codec,
        skipped_packets,
    })
}

/// Track id and codec parameters of an audio track.
type AudioTrack = (u32, AudioCodecParameters);

/// Opens `path` and lists its audio tracks, the default one first.
fn open(path: &Path) -> Result<(Box<dyn FormatReader>, Vec<AudioTrack>), AudioError> {
    let io_error = |source| AudioError::Io {
        path: path.to_owned(),
        source,
    };
    let metadata = std::fs::metadata(path).map_err(io_error)?;
    if metadata.is_dir() {
        return Err(AudioError::IsDirectory {
            path: path.to_owned(),
        });
    }
    if metadata.len() == 0 {
        return Err(AudioError::Empty {
            path: path.to_owned(),
        });
    }
    let file = File::open(path).map_err(io_error)?;
    let stream = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|err| match err {
            SymphoniaError::IoError(source)
                if source.kind() != std::io::ErrorKind::UnexpectedEof =>
            {
                AudioError::Io {
                    path: path.to_owned(),
                    source,
                }
            }
            _ => AudioError::UnsupportedFormat {
                path: path.to_owned(),
            },
        })?;

    let no_track = || AudioError::NoAudioTrack {
        path: path.to_owned(),
    };
    let default_id = format.default_track(TrackType::Audio).map(|track| track.id);
    let mut tracks: Vec<AudioTrack> = format
        .tracks()
        .iter()
        .filter_map(|track| {
            let params = track.codec_params.as_ref()?.audio()?;
            Some((track.id, params.clone()))
        })
        .collect();
    if tracks.is_empty() {
        return Err(no_track());
    }
    tracks.sort_by_key(|(id, _)| Some(*id) != default_id);
    Ok((format, tracks))
}

/// Averages `planes` (one slice per channel) into one mono channel.
///
/// Reproduces `np.mean(y, axis=0)` on a `float32` array: the channels are summed in
/// order in `f32`, then divided by the channel count in `f32`.
///
/// # Panics
///
/// If the planes have different lengths.
#[must_use]
pub fn downmix<P: AsRef<[f32]>>(planes: &[P]) -> Vec<f32> {
    let mut out = Vec::new();
    downmix_into(planes, &mut out);
    out
}

#[allow(
    clippy::cast_precision_loss,
    reason = "channel counts are small integers, exact in f32"
)]
fn downmix_into<P: AsRef<[f32]>>(planes: &[P], out: &mut Vec<f32>) {
    let Some((first, rest)) = planes.split_first() else {
        return;
    };
    let first = first.as_ref();
    if rest.is_empty() {
        out.extend_from_slice(first);
        return;
    }
    let start = out.len();
    out.extend_from_slice(first);
    let acc = &mut out[start..];
    for plane in rest {
        let plane = plane.as_ref();
        assert_eq!(plane.len(), acc.len(), "audio planes differ in length");
        for (a, &x) in acc.iter_mut().zip(plane) {
            *a += x;
        }
    }
    let n = planes.len() as f32;
    for a in acc {
        *a /= n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_is_passed_through() {
        assert_eq!(downmix(&[vec![0.1_f32, -0.5]]), vec![0.1, -0.5]);
    }

    #[test]
    fn stereo_is_averaged_in_f32() {
        let left = [0.1_f32, 1.0, f32::MAX];
        let right = [0.2_f32, -1.0, f32::MAX];
        let mono = downmix(&[left, right]);
        assert_eq!(mono[0], (0.1_f32 + 0.2_f32) / 2.0);
        assert_eq!(mono[1], 0.0);
        // Summing in f32 overflows exactly like numpy's float32 mean.
        assert_eq!(mono[2], f32::INFINITY);
    }

    #[test]
    fn channels_are_summed_in_order() {
        let planes = [[1.0e8_f32], [1.0], [-1.0e8]];
        // ((1e8 + 1) - 1e8) / 3 in f32: the 1 is lost, as in numpy.
        assert_eq!(downmix(&planes), vec![0.0]);
    }

    #[test]
    fn no_planes_gives_no_samples() {
        assert!(downmix::<Vec<f32>>(&[]).is_empty());
    }

    #[test]
    fn directories_and_empty_files_are_named_as_such() {
        let dir = tempfile::tempdir().unwrap();
        let err = decode(dir.path()).unwrap_err();
        assert!(matches!(err, AudioError::IsDirectory { .. }), "{err}");
        let empty = dir.path().join("empty.wav");
        std::fs::write(&empty, b"").unwrap();
        let err = decode(&empty).unwrap_err();
        assert!(matches!(err, AudioError::Empty { .. }), "{err}");
        assert!(err.to_string().contains("is empty"));
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let err = decode(Path::new("does/not/exist.wav")).unwrap_err();
        assert!(matches!(err, AudioError::Io { .. }), "{err}");
        assert!(err.to_string().contains("does/not/exist.wav"));
    }
}

// Copyright 2022 Spotify AB (original: basic_pitch/predict.py)
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
// Ported from spotify/basic-pitch v0.4.0. Changes: the options of `basic-pitch` and the
// output file names of `build_output_path` (inference.py) are kept, with new options for
// the output formats, the trim mode, the pitch bend offset, the resampler and the
// inference backend; the rest is rewritten in Rust.

//! Command-line arguments and the transcription loop.

use std::collections::HashMap;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use basic_pitch::constants::{
    AUDIO_SAMPLE_RATE, DEFAULT_FRAME_THRESHOLD, DEFAULT_MIDI_TEMPO, DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    DEFAULT_ONSET_THRESHOLD, MIDI_OFFSET, N_FREQ_BINS_NOTES, hz_to_midi,
};
use basic_pitch::model;
use basic_pitch::{BasicPitch, MidiParams, NoteParams, Transcription, TrimMode};
use clap::{CommandFactory, Parser, ValueEnum, error::ErrorKind};

use crate::audio::{self, LoadedAudio, SUPPORTED_FORMATS, UNSUPPORTED_FORMATS};
use crate::output::{self, JsonDocument, JsonNote};
use crate::progress::{Progress, ProgressBackend};
use crate::resample::Resampler;

const ABOUT: &str =
    "Polyphonic audio-to-MIDI transcription with Spotify's Basic Pitch model, in Rust.";

const LONG_ABOUT: &str = "\
Polyphonic audio-to-MIDI transcription with Spotify's Basic Pitch model, in Rust.

Transcribes recordings of any instrument or voice into notes with pitch bends, and writes
them as MIDI, JSON or CSV. From the same samples, the notes are the same as the Python
basic-pitch package gives; files that are not at 22 050 Hz are resampled first, which can
move a note that sits right at a detection threshold. Unofficial port, not affiliated
with Spotify.";

const AFTER_HELP: &str = "\
Examples:
  basic-pitch-rs song.wav                  Write song_basic_pitch.mid in the current directory
  basic-pitch-rs -o midi/ takes/*.flac     Transcribe several files into midi/
  basic-pitch-rs riff.m4a --midi --json    Write both the MIDI file and the JSON notes
  basic-pitch-rs riff.m4a --json --output - | jq '.notes[].note_name'
  basic-pitch-rs bass.wav --minimum-frequency 30 --maximum-frequency 400

Output files are named <input stem>_basic_pitch.<ext>, like the Python tool's, and are
never overwritten without --force. Unlike the Python tool, the output directory is not a
positional argument: use -o DIR.";

/// How the model output is trimmed to the length of the audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Trim {
    /// Like spotify/basic-pitch main: keeps the end of the file.
    Main,
    /// Like the basic-pitch 0.4.0 release on PyPI: drops the last ~0.7 % of the frames (about 0.4 s per minute).
    #[value(name = "v0.4.0")]
    V0_4_0,
}

impl Trim {
    fn mode(self) -> TrimMode {
        match self {
            Self::Main => TrimMode::Main,
            Self::V0_4_0 => TrimMode::V0_4_0,
        }
    }
}

/// The inference engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum BackendKind {
    /// tract, pure Rust (always available).
    Tract,
    /// ONNX Runtime, about 3 times faster (needs the `ort` feature).
    Ort,
}

impl BackendKind {
    fn name(self) -> &'static str {
        match self {
            Self::Tract => "tract",
            Self::Ort => "ort",
        }
    }
}

/// The command-line arguments of `basic-pitch-rs`.
#[derive(Debug, Parser)]
#[command(
    name = "basic-pitch-rs",
    version,
    about = ABOUT,
    long_about = LONG_ABOUT,
    after_help = AFTER_HELP,
    max_term_width = 100
)]
#[allow(clippy::struct_excessive_bools, reason = "command-line flags")]
pub struct Cli {
    /// Audio files to transcribe.
    #[arg(
        required = true,
        value_name = "INPUT",
        long_help = format!("Audio files to transcribe.\n\nSupported: {SUPPORTED_FORMATS}. Not supported: {UNSUPPORTED_FORMATS}.")
    )]
    pub inputs: Vec<PathBuf>,

    /// Directory for the output files [default: the current directory].
    #[arg(short = 'o', long, value_name = "DIR", help_heading = "Output")]
    pub output_dir: Option<PathBuf>,

    /// Write to FILE instead ("-" for stdout). Needs a single input and a single format;
    /// the format follows the extension (.mid, .json, .csv, .npz) unless a format flag is
    /// given.
    #[arg(
        long,
        value_name = "FILE",
        conflicts_with = "output_dir",
        help_heading = "Output"
    )]
    pub output: Option<PathBuf>,

    /// Write a MIDI file (the default when no other format is chosen).
    #[arg(long, alias = "save-midi", help_heading = "Output")]
    pub midi: bool,

    /// Write the notes as JSON: times, frames, pitch, note name, velocity and pitch bends,
    /// with the parameters and model used.
    #[arg(long, help_heading = "Output")]
    pub json: bool,

    /// Write the notes as CSV, in the format of Basic Pitch's --save-note-events.
    #[arg(long, alias = "save-note-events", help_heading = "Output")]
    pub csv: bool,

    /// Write the raw model outputs (note, onset and contour posteriorgrams) as a NumPy .npz
    /// archive.
    #[arg(
        long,
        alias = "save-model-outputs",
        help_heading = "Output",
        long_help = "Write the raw model outputs (note, onset and contour posteriorgrams) as a NumPy .npz archive.\n\nThe archive holds three float32 arrays named note (frames x 88), onset (frames x 88) and contour (frames x 264), readable with numpy.load without pickle. This differs from the .npz of the Python tool's --save-model-outputs, which stores a pickled dict under the key basic_pitch_model_output, with note and onset already cut to the frequency range."
    )]
    pub posteriorgrams: bool,

    /// Overwrite existing output files.
    #[arg(short, long, help_heading = "Output")]
    pub force: bool,

    /// Do not print progress or summaries on stderr (errors are still printed).
    #[arg(short, long, help_heading = "Output")]
    pub quiet: bool,

    /// Minimum onset activation to start a note, between 0 and 1.
    #[arg(long, value_name = "P", default_value_t = DEFAULT_ONSET_THRESHOLD, value_parser = probability, allow_negative_numbers = true, help_heading = "Note detection")]
    pub onset_threshold: f64,

    /// Minimum note activation to sustain a note, between 0 and 1.
    #[arg(long, value_name = "P", default_value_t = DEFAULT_FRAME_THRESHOLD, value_parser = probability, allow_negative_numbers = true, help_heading = "Note detection")]
    pub frame_threshold: f64,

    /// Minimum note length, in milliseconds.
    #[arg(long, value_name = "MS", default_value_t = DEFAULT_MINIMUM_NOTE_LENGTH_MS, value_parser = non_negative, allow_negative_numbers = true, help_heading = "Note detection")]
    pub minimum_note_length: f64,

    /// Ignore notes below this frequency, in Hz. The model covers A0 (27.5 Hz) to C8
    /// (4186 Hz): a lower value means no limit.
    #[arg(long, value_name = "HZ", value_parser = positive, allow_negative_numbers = true, help_heading = "Note detection")]
    pub minimum_frequency: Option<f64>,

    /// Ignore notes above this frequency, in Hz. The model covers A0 (27.5 Hz) to C8
    /// (4186 Hz): a higher value means no limit.
    #[arg(long, value_name = "HZ", value_parser = positive, allow_negative_numbers = true, help_heading = "Note detection")]
    pub maximum_frequency: Option<f64>,

    /// Skip the melodia trick, which recovers notes without a clear onset.
    #[arg(long, help_heading = "Note detection")]
    pub no_melodia: bool,

    /// Only start notes on onsets from the model, without inferring them from the note
    /// activations.
    #[arg(long, help_heading = "Note detection")]
    pub no_infer_onsets: bool,

    /// Do not estimate pitch bends.
    #[arg(long, help_heading = "Note detection")]
    pub no_pitch_bends: bool,

    /// Add N thirds of a semitone to every pitch bend, from -6 to 6. Basic Pitch reports a
    /// bend of +1 on in-tune notes (spotify/basic-pitch#87); -1 compensates. The default
    /// keeps the output identical to Python's.
    #[arg(
        long,
        value_name = "N",
        default_value_t = 0,
        allow_negative_numbers = true,
        value_parser = clap::value_parser!(i32).range(-6..=6),
        help_heading = "Note detection"
    )]
    pub bend_offset: i32,

    /// Keep the pitch bends of overlapping notes, by putting each pitch on its own MIDI
    /// track and channel.
    #[arg(long, help_heading = "MIDI")]
    pub multiple_pitch_bends: bool,

    /// Tempo of the MIDI file, in beats per minute, from 4 to 1000. Note times do not
    /// depend on it.
    #[arg(long, value_name = "BPM", default_value_t = DEFAULT_MIDI_TEMPO, value_parser = tempo, allow_negative_numbers = true, help_heading = "MIDI")]
    pub midi_tempo: f64,

    /// How the model output is trimmed at the end of the file: "main" keeps the end, like
    /// spotify/basic-pitch main; "v0.4.0" reproduces the PyPI release exactly.
    #[arg(long, value_enum, default_value_t = Trim::Main, help_heading = "Processing")]
    pub trim: Trim,

    /// Resampler for files that are not at 22 050 Hz. soxr is the one Python Basic Pitch
    /// uses (through librosa). rubato is pure Rust: from 44.1 or 48 kHz it stays within
    /// about 1e-6 of soxr on music, but below 22 050 Hz (8 to 16 kHz recordings) it
    /// differs by up to about 5e-3, which can change a few notes.
    #[arg(long, value_enum, default_value_t = Resampler::DEFAULT, help_heading = "Processing")]
    pub resampler: Resampler,

    /// Inference engine.
    #[arg(long, value_enum, default_value_t = BackendKind::Tract, help_heading = "Processing")]
    pub backend: BackendKind,

    /// Number of threads running the model with the tract backend [default: one per
    /// core]. The results do not depend on it. ONNX Runtime manages its own threads.
    #[arg(short = 'j', long, value_name = "N", help_heading = "Processing")]
    pub threads: Option<NonZeroUsize>,
}

fn probability(s: &str) -> Result<f64, String> {
    let v = finite(s)?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err("must be between 0 and 1".into())
    }
}

/// MIDI stores the tempo as microseconds per quarter note on 24 bits, so the slowest
/// tempo is about 3.6 BPM; above 1000 BPM, a file of a few hours exceeds the largest tick
/// count.
fn tempo(s: &str) -> Result<f64, String> {
    let v = finite(s)?;
    if (4.0..=1000.0).contains(&v) {
        Ok(v)
    } else {
        Err("must be between 4 and 1000 BPM".into())
    }
}

fn finite(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err("must be a finite number".into())
    }
}

fn non_negative(s: &str) -> Result<f64, String> {
    let v = finite(s)?;
    if v >= 0.0 {
        Ok(v)
    } else {
        Err("must be zero or more".into())
    }
}

fn positive(s: &str) -> Result<f64, String> {
    let v = finite(s)?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err("must be greater than zero".into())
    }
}

/// An output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    /// Standard MIDI File.
    Midi,
    /// [`JsonDocument`].
    Json,
    /// Basic Pitch's note CSV.
    Csv,
    /// NumPy archive of the posteriorgrams.
    Npz,
}

impl Format {
    /// The file extension, without the dot.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Midi => "mid",
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Npz => "npz",
        }
    }

    fn from_extension(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "mid" | "midi" | "smf" => Some(Self::Midi),
            "json" => Some(Self::Json),
            "csv" => Some(Self::Csv),
            "npz" => Some(Self::Npz),
            _ => None,
        }
    }

    fn is_binary(self) -> bool {
        matches!(self, Self::Midi | Self::Npz)
    }

    /// The command-line flag that requests this format.
    fn flag(self) -> &'static str {
        match self {
            Self::Midi => "midi",
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Npz => "posteriorgrams",
        }
    }
}

/// Number of keys of the model, from A0 (MIDI 21) to C8 (MIDI 108).
#[allow(clippy::cast_possible_wrap, reason = "88 fits in i64")]
const KEYS: i64 = N_FREQ_BINS_NOTES as i64;

/// The key index Basic Pitch derives from a frequency bound:
/// `round(hz_to_midi(hz) - 21)`, which is negative below about 26.7 Hz.
#[allow(
    clippy::cast_possible_truncation,
    reason = "saturating conversion of a rounded float"
)]
fn key_index(hz: f64) -> i64 {
    (hz_to_midi(hz) - f64::from(MIDI_OFFSET)).round_ties_even() as i64
}

/// Raised when stdout is closed by the reader (`| head`): the run then stops quietly.
#[derive(Debug, thiserror::Error)]
#[error("stdout was closed")]
struct StdoutClosed;

/// Where the outputs of one input go.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Destination {
    File(PathBuf),
    Stdout,
}

impl Destination {
    fn describe(&self) -> String {
        match self {
            Self::File(path) => path.display().to_string(),
            Self::Stdout => "stdout".into(),
        }
    }
}

/// The work planned for one input file.
#[derive(Debug)]
struct Job {
    input: PathBuf,
    outputs: Vec<(Format, Destination)>,
}

impl Cli {
    /// The formats requested with flags, in a fixed order.
    fn formats(&self) -> Vec<Format> {
        [
            (self.midi, Format::Midi),
            (self.json, Format::Json),
            (self.csv, Format::Csv),
            (self.posteriorgrams, Format::Npz),
        ]
        .into_iter()
        .filter_map(|(on, f)| on.then_some(f))
        .collect()
    }

    /// The note decoding parameters.
    ///
    /// A minimum frequency below the lowest key is dropped: Basic Pitch would turn it into
    /// a negative index and remove nearly every note (see [`Cli::frequency_warning`]).
    #[must_use]
    pub fn note_params(&self) -> NoteParams {
        NoteParams {
            onset_threshold: self.onset_threshold,
            frame_threshold: self.frame_threshold,
            minimum_note_length_ms: self.minimum_note_length,
            minimum_frequency: self.minimum_frequency.filter(|&hz| key_index(hz) >= 0),
            maximum_frequency: self.maximum_frequency,
            infer_onsets: !self.no_infer_onsets,
            melodia_trick: !self.no_melodia,
            include_pitch_bends: !self.no_pitch_bends,
            bend_offset: self.bend_offset,
            ..NoteParams::default()
        }
    }

    /// The MIDI export parameters.
    #[must_use]
    pub fn midi_params(&self) -> MidiParams {
        MidiParams {
            multiple_pitch_bends: self.multiple_pitch_bends,
            midi_tempo: self.midi_tempo,
        }
    }

    /// The warning to print when `--minimum-frequency` is below the lowest key.
    #[must_use]
    pub fn frequency_warning(&self) -> Option<String> {
        let hz = self.minimum_frequency?;
        (key_index(hz) < 0).then(|| {
            format!(
                "--minimum-frequency {hz} Hz is below A0 (27.5 Hz), the lowest note of the model: no lower limit applied (the Python tool would drop nearly every note)"
            )
        })
    }

    /// Rejects frequency bounds that leave no key to detect.
    fn check_frequencies(&self) -> Result<(), String> {
        let lowest = self.minimum_frequency.map_or(0, |hz| key_index(hz).max(0));
        let highest = self.maximum_frequency.map_or(i64::MAX, key_index);
        if lowest >= KEYS {
            return Err(format!(
                "--minimum-frequency {} Hz is above C8 (4186 Hz), the highest note of the model: no note could be detected",
                self.minimum_frequency.unwrap_or_default()
            ));
        }
        if highest <= 0 {
            return Err(format!(
                "--maximum-frequency {} Hz is at or below A0 (27.5 Hz), the lowest note of the model: no note could be detected",
                self.maximum_frequency.unwrap_or_default()
            ));
        }
        if lowest >= highest {
            return Err(format!(
                "--minimum-frequency {} Hz and --maximum-frequency {} Hz leave no note between them (the maximum key itself is excluded, as in Basic Pitch)",
                self.minimum_frequency.unwrap_or_default(),
                self.maximum_frequency.unwrap_or_default()
            ));
        }
        Ok(())
    }

    /// Checks the combination of arguments and plans the output files.
    fn plan(&self) -> Result<Vec<Job>, clap::Error> {
        let usage = |kind, msg: String| Cli::command().error(kind, msg);

        self.check_frequencies()
            .map_err(|msg| usage(ErrorKind::InvalidValue, msg))?;
        for (i, input) in self.inputs.iter().enumerate() {
            if input.is_dir() {
                let hint = if i == 0 && self.inputs.len() > 1 && self.output_dir.is_none() {
                    format!(
                        "; to write the outputs into it, use -o {} (the Python tool takes the output directory first)",
                        input.display()
                    )
                } else {
                    String::new()
                };
                return Err(usage(
                    ErrorKind::InvalidValue,
                    format!(
                        "{} is a directory, not an audio file{hint}",
                        input.display()
                    ),
                ));
            }
        }

        if !self.resampler.is_available() {
            return Err(usage(
                ErrorKind::InvalidValue,
                "this build does not include the soxr resampler; reinstall with `--features soxr`, or use `--resampler rubato`".into(),
            ));
        }
        if self.backend == BackendKind::Ort && !cfg!(feature = "ort") {
            return Err(usage(
                ErrorKind::InvalidValue,
                "this build does not include ONNX Runtime; reinstall with `--features ort`, or use `--backend tract`".into(),
            ));
        }

        let mut formats = self.formats();
        if let Some(output) = &self.output {
            if self.inputs.len() > 1 {
                return Err(usage(
                    ErrorKind::ArgumentConflict,
                    "--output takes a single input; use --output-dir for several files".into(),
                ));
            }
            let stdout = output.as_os_str() == "-";
            if formats.is_empty() {
                formats.push(if stdout {
                    Format::Midi
                } else {
                    Format::from_extension(output).ok_or_else(|| {
                        usage(
                            ErrorKind::InvalidValue,
                            format!(
                                "cannot tell the format of {} from its extension; add --midi, --json, --csv or --posteriorgrams",
                                output.display()
                            ),
                        )
                    })?
                });
            }
            if formats.len() > 1 {
                return Err(usage(
                    ErrorKind::ArgumentConflict,
                    "--output writes a single format; use --output-dir for several".into(),
                ));
            }
            let format = formats[0];
            if !stdout
                && let Some(implied) = Format::from_extension(output)
                && implied != format
            {
                return Err(usage(
                    ErrorKind::ArgumentConflict,
                    format!(
                        "--output {} has the extension of a .{} file, but --{} was requested; rename it to .{} or drop the format flag",
                        output.display(),
                        implied.extension(),
                        format.flag(),
                        format.extension()
                    ),
                ));
            }
            let destination = if stdout {
                if format.is_binary() && io::stdout().is_terminal() {
                    return Err(usage(
                        ErrorKind::InvalidValue,
                        format!(
                            "refusing to write a binary .{} file to the terminal; redirect stdout or use --json or --csv",
                            format.extension()
                        ),
                    ));
                }
                Destination::Stdout
            } else {
                Destination::File(output.clone())
            };
            return Ok(vec![Job {
                input: self.inputs[0].clone(),
                outputs: vec![(format, destination)],
            }]);
        }

        if formats.is_empty() {
            formats.push(Format::Midi);
        }
        let dir = self.output_dir.clone().unwrap_or_default();
        let mut owners: HashMap<PathBuf, &Path> = HashMap::new();
        let mut jobs = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            let stem = input
                .file_stem()
                .map_or_else(|| "output".into(), |s| s.to_string_lossy().into_owned());
            let mut outputs = Vec::with_capacity(formats.len());
            for &format in &formats {
                let path = dir.join(format!("{stem}_basic_pitch.{}", format.extension()));
                if let Some(other) = owners.insert(path.clone(), input) {
                    return Err(usage(
                        ErrorKind::ArgumentConflict,
                        format!(
                            "{} and {} would both write {}; transcribe them in separate runs or rename one",
                            other.display(),
                            input.display(),
                            path.display()
                        ),
                    ));
                }
                outputs.push((format, Destination::File(path)));
            }
            jobs.push(Job {
                input: input.clone(),
                outputs,
            });
        }
        Ok(jobs)
    }
}

/// Parses the command line, runs it and returns the exit code: 0 on success, 1 if any
/// input failed, 2 for invalid arguments.
#[must_use]
pub fn main() -> ExitCode {
    let cli = Cli::parse();
    let jobs = match cli.plan() {
        Ok(jobs) => jobs,
        Err(err) => err.exit(),
    };
    if !cli.force {
        let existing: Vec<_> = jobs
            .iter()
            .flat_map(|job| &job.outputs)
            .filter_map(|(_, dest)| match dest {
                Destination::File(path) if path.exists() => Some(path.display().to_string()),
                _ => None,
            })
            .collect();
        if !existing.is_empty() {
            eprintln!(
                "error: {} already exist{}, use --force to overwrite:\n  {}",
                if existing.len() == 1 {
                    "this output file"
                } else {
                    "these output files"
                },
                if existing.len() == 1 { "s" } else { "" },
                existing.join("\n  ")
            );
            return ExitCode::FAILURE;
        }
    }

    match run(&cli, &jobs) {
        Ok(0) => ExitCode::SUCCESS,
        Ok(failed) => {
            if jobs.len() > 1 {
                eprintln!("error: {failed} of {} inputs failed", jobs.len());
            }
            ExitCode::FAILURE
        }
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

/// Transcribes every job and returns the number of failed inputs.
fn run(cli: &Cli, jobs: &[Job]) -> Result<usize> {
    let progress = Progress::new(cli.quiet);
    if let Some(warning) = cli.frequency_warning()
        && !cli.quiet
    {
        eprintln!("warning: {warning}");
    }
    let transcriber = make_transcriber(cli, &progress)?;
    let note_params = cli.note_params();
    note_params.validate()?;

    if let Some(dir) = &cli.output_dir {
        fs::create_dir_all(dir)
            .with_context(|| format!("cannot create the output directory {}", dir.display()))?;
    }

    let mut failed = 0;
    for (i, job) in jobs.iter().enumerate() {
        let label = if jobs.len() > 1 {
            format!("[{}/{}] {}", i + 1, jobs.len(), file_name(&job.input))
        } else {
            file_name(&job.input)
        };
        match transcribe_job(cli, &transcriber, &note_params, &progress, &label, job) {
            Ok(summary) => {
                if !cli.quiet {
                    progress.println(&summary);
                }
            }
            Err(err) if err.is::<StdoutClosed>() => {
                progress.finish();
                break;
            }
            Err(err) => {
                progress.finish();
                progress.println(&format!("error: {err:#}"));
                failed += 1;
            }
        }
    }
    Ok(failed)
}

fn make_transcriber(cli: &Cli, progress: &Progress) -> Result<BasicPitch> {
    let cores = || std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let transcriber = match cli.backend {
        BackendKind::Tract => BasicPitch::with_backend(ProgressBackend {
            inner: model::TractBackend::new().context("cannot load the model")?,
            progress: progress.clone(),
        })
        .with_threads(cli.threads.unwrap_or_else(cores)),
        #[cfg(feature = "ort")]
        BackendKind::Ort => BasicPitch::with_backend(ProgressBackend {
            inner: model::OrtBackend::new().context("cannot load the model")?,
            progress: progress.clone(),
        }),
        #[cfg(not(feature = "ort"))]
        BackendKind::Ort => anyhow::bail!("this build does not include ONNX Runtime"),
    };
    Ok(transcriber.with_trim_mode(cli.trim.mode()))
}

fn transcribe_job(
    cli: &Cli,
    transcriber: &BasicPitch,
    note_params: &NoteParams,
    progress: &Progress,
    label: &str,
    job: &Job,
) -> Result<String> {
    progress.decoding(label);
    let audio = audio::load(&job.input, cli.resampler)?;
    if audio.source.skipped_packets > 0 && !cli.quiet {
        progress.println(&format!(
            "warning: {}: skipped {} corrupt packet(s)",
            job.input.display(),
            audio.source.skipped_packets
        ));
    }

    let windows = if audio.samples.is_empty() {
        0
    } else {
        basic_pitch::window::window_count(audio.samples.len())
    };
    progress.start(label, windows);
    let transcription = transcriber
        .transcribe_with(&audio.samples, note_params)
        .with_context(|| format!("cannot transcribe {}", job.input.display()));
    progress.finish();
    let transcription = transcription?;

    for (format, destination) in &job.outputs {
        let bytes = render(cli, *format, job, &audio, &transcription)?;
        match destination {
            Destination::File(path) => fs::write(path, &bytes)
                .with_context(|| format!("cannot write {}", path.display()))?,
            Destination::Stdout => {
                let mut stdout = io::stdout().lock();
                match stdout.write_all(&bytes).and_then(|()| stdout.flush()) {
                    Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {
                        return Err(StdoutClosed.into());
                    }
                    result => result.context("cannot write to stdout")?,
                }
            }
        }
    }

    let n = transcription.notes.len();
    Ok(format!(
        "{} -> {} ({n} note{}, {:.1} s)",
        job.input.display(),
        describe_outputs(&job.outputs),
        if n == 1 { "" } else { "s" },
        audio.source.duration_s
    ))
}

fn render(
    cli: &Cli,
    format: Format,
    job: &Job,
    audio: &LoadedAudio,
    transcription: &Transcription,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    match format {
        Format::Midi => bytes = transcription.to_midi(&cli.midi_params())?,
        Format::Json => {
            output::write_json(&json_document(cli, job, audio, transcription), &mut bytes)?;
        }
        Format::Csv => output::write_csv(&transcription.notes, &mut bytes)?,
        Format::Npz => output::write_npz(&transcription.posteriorgrams, &mut bytes)?,
    }
    Ok(bytes)
}

fn json_document(
    cli: &Cli,
    job: &Job,
    audio: &LoadedAudio,
    transcription: &Transcription,
) -> JsonDocument {
    let params = cli.note_params();
    JsonDocument {
        schema: output::JSON_SCHEMA_URL.into(),
        schema_version: output::JSON_SCHEMA_VERSION,
        generator: output::Generator {
            name: "basic-pitch-rs".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        model: output::ModelInfo {
            name: "basic-pitch-icassp-2022".into(),
            sha256: model::MODEL_SHA256.into(),
            backend: cli.backend.name().into(),
        },
        input: output::InputInfo {
            path: job.input.display().to_string(),
            sample_rate: audio.source.sample_rate,
            channels: audio.source.channels,
            frames: audio.source.frames,
            duration_s: audio.source.duration_s,
            codec: audio.source.codec.clone(),
        },
        analysis: output::AnalysisInfo {
            sample_rate: AUDIO_SAMPLE_RATE,
            samples: audio.samples.len(),
            resampler: audio.resampler.map(|r| r.name().into()),
            trim_mode: cli.trim.mode().to_string(),
            frames: transcription.posteriorgrams.n_frames(),
        },
        params: output::Params {
            onset_threshold: params.onset_threshold,
            frame_threshold: params.frame_threshold,
            minimum_note_length_ms: params.minimum_note_length_ms,
            minimum_frequency: params.minimum_frequency,
            maximum_frequency: params.maximum_frequency,
            infer_onsets: params.infer_onsets,
            melodia_trick: params.melodia_trick,
            include_pitch_bends: params.include_pitch_bends,
            bend_offset: params.bend_offset,
            multiple_pitch_bends: cli.multiple_pitch_bends,
            midi_tempo: cli.midi_tempo,
        },
        notes: transcription
            .sorted_notes()
            .iter()
            .map(JsonNote::from)
            .collect(),
    }
}

/// Names the outputs of a job compactly: `out/song_basic_pitch.{mid,json}`.
fn describe_outputs(outputs: &[(Format, Destination)]) -> String {
    let files: Vec<&PathBuf> = outputs
        .iter()
        .filter_map(|(_, d)| match d {
            Destination::File(path) => Some(path),
            Destination::Stdout => None,
        })
        .collect();
    match files.as_slice() {
        [first, rest @ ..]
            if !rest.is_empty()
                && rest
                    .iter()
                    .all(|p| p.with_extension("") == first.with_extension("")) =>
        {
            let extensions: Vec<_> = outputs.iter().map(|(f, _)| f.extension()).collect();
            format!(
                "{}.{{{}}}",
                first.with_extension("").display(),
                extensions.join(",")
            )
        }
        _ => outputs
            .iter()
            .map(|(_, d)| d.describe())
            .collect::<Vec<_>>()
            .join(", "),
    }
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("basic-pitch-rs").chain(args.iter().copied())).unwrap()
    }

    #[test]
    fn command_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn defaults_match_python() {
        let cli = parse(&["a.wav"]);
        assert_eq!(cli.note_params(), NoteParams::default());
        assert_eq!(cli.midi_params(), MidiParams::default());
        assert_eq!(cli.trim, Trim::Main);
    }

    #[test]
    fn flags_map_to_parameters() {
        let cli = parse(&[
            "a.wav",
            "--onset-threshold",
            "0.6",
            "--frame-threshold",
            "0.2",
            "--minimum-note-length",
            "58",
            "--minimum-frequency",
            "80",
            "--maximum-frequency",
            "1000",
            "--no-melodia",
            "--no-infer-onsets",
            "--no-pitch-bends",
            "--bend-offset",
            "-1",
            "--multiple-pitch-bends",
            "--midi-tempo",
            "90",
            "--trim",
            "v0.4.0",
        ]);
        let p = cli.note_params();
        assert_eq!(p.onset_threshold, 0.6);
        assert_eq!(p.frame_threshold, 0.2);
        assert_eq!(p.minimum_note_length_ms, 58.0);
        assert_eq!(p.minimum_frequency, Some(80.0));
        assert_eq!(p.maximum_frequency, Some(1000.0));
        assert!(!p.melodia_trick && !p.infer_onsets && !p.include_pitch_bends);
        assert_eq!(p.bend_offset, -1);
        assert!(cli.midi_params().multiple_pitch_bends);
        assert_eq!(cli.midi_params().midi_tempo, 90.0);
        assert_eq!(cli.trim.mode(), TrimMode::V0_4_0);
    }

    #[test]
    fn invalid_numbers_are_rejected() {
        for args in [
            ["a.wav", "--onset-threshold", "nan"],
            ["a.wav", "--minimum-note-length", "-1"],
            ["a.wav", "--minimum-frequency", "0"],
            ["a.wav", "--midi-tempo", "inf"],
        ] {
            let full = std::iter::once("basic-pitch-rs").chain(args);
            assert!(Cli::try_parse_from(full).is_err(), "{args:?}");
        }
    }

    #[test]
    fn default_output_is_midi_next_to_the_working_directory() {
        let jobs = parse(&["dir/song.wav"]).plan().unwrap();
        assert_eq!(
            jobs[0].outputs,
            [(
                Format::Midi,
                Destination::File("song_basic_pitch.mid".into())
            )]
        );
    }

    #[test]
    fn every_requested_format_is_planned() {
        let jobs = parse(&["a.flac", "b.mp3", "-o", "out", "--json", "--midi", "--csv"])
            .plan()
            .unwrap();
        let names: Vec<_> = jobs
            .iter()
            .flat_map(|j| &j.outputs)
            .map(|(_, d)| d.describe())
            .collect();
        let sep = std::path::MAIN_SEPARATOR;
        assert_eq!(
            names,
            [
                format!("out{sep}a_basic_pitch.mid"),
                format!("out{sep}a_basic_pitch.json"),
                format!("out{sep}a_basic_pitch.csv"),
                format!("out{sep}b_basic_pitch.mid"),
                format!("out{sep}b_basic_pitch.json"),
                format!("out{sep}b_basic_pitch.csv"),
            ]
        );
    }

    #[test]
    fn output_format_follows_the_extension() {
        let jobs = parse(&["a.wav", "--output", "notes.JSON"]).plan().unwrap();
        assert_eq!(jobs[0].outputs[0].0, Format::Json);
        let err = parse(&["a.wav", "--output", "notes.txt"])
            .plan()
            .unwrap_err();
        assert!(err.to_string().contains("extension"), "{err}");
    }

    #[test]
    fn output_needs_one_input_and_one_format() {
        assert!(
            parse(&["a.wav", "b.wav", "--output", "x.mid"])
                .plan()
                .is_err()
        );
        assert!(
            parse(&["a.wav", "--output", "x.mid", "--json", "--csv"])
                .plan()
                .is_err()
        );
        assert!(
            Cli::try_parse_from(["basic-pitch-rs", "a.wav", "--output", "x.mid", "-o", "d"])
                .is_err()
        );
    }

    #[test]
    fn colliding_outputs_are_rejected() {
        let err = parse(&["a/song.wav", "b/song.flac"]).plan().unwrap_err();
        assert!(err.to_string().contains("would both write"), "{err}");
    }

    #[test]
    fn npz_help_explains_the_difference_with_python() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("basic_pitch_model_output"), "{help}");
        // The Python flag name still works.
        assert!(parse(&["a.wav", "--save-model-outputs"]).posteriorgrams);
    }

    #[test]
    fn frequency_bounds_follow_the_piano_keys() {
        let below = parse(&["a.wav", "--minimum-frequency", "20"]);
        assert_eq!(below.note_params().minimum_frequency, None);
        assert!(below.frequency_warning().is_some());
        assert!(below.check_frequencies().is_ok());
        let a0 = parse(&["a.wav", "--minimum-frequency", "27.5"]);
        assert_eq!(a0.note_params().minimum_frequency, Some(27.5));
        assert!(a0.frequency_warning().is_none());
        for args in [
            &["a.wav", "--minimum-frequency", "5000"][..],
            &["a.wav", "--maximum-frequency", "27.5"][..],
            &[
                "a.wav",
                "--minimum-frequency",
                "440",
                "--maximum-frequency",
                "440",
            ][..],
        ] {
            assert!(parse(args).check_frequencies().is_err(), "{args:?}");
        }
        let range = parse(&[
            "a.wav",
            "--minimum-frequency",
            "80",
            "--maximum-frequency",
            "1000",
        ]);
        assert!(range.check_frequencies().is_ok());
    }

    #[test]
    fn missing_features_are_reported() {
        if !cfg!(feature = "soxr") {
            let err = parse(&["a.wav", "--resampler", "soxr"]).plan().unwrap_err();
            assert!(err.to_string().contains("--features soxr"), "{err}");
        }
        if !cfg!(feature = "ort") {
            let err = parse(&["a.wav", "--backend", "ort"]).plan().unwrap_err();
            assert!(err.to_string().contains("--features ort"), "{err}");
        }
    }
}

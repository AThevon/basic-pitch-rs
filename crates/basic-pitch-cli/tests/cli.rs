// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Behavior of the `basic-pitch-rs` binary: outputs, error messages, exit codes, and
//! decoding of the WAV sample formats.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use basic_pitch_cli::audio;
use ndarray_npy::ReadNpyExt;
use parity::fixtures_dir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_basic-pitch-rs"))
}

fn short_wav() -> PathBuf {
    fixtures_dir().join("audio/short.wav")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn writes_a_midi_file_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let out = bin()
        .arg(short_wav())
        .arg("-o")
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(out.stdout.is_empty(), "nothing on stdout unless asked");
    let midi = std::fs::read(dir.path().join("short_basic_pitch.mid")).unwrap();
    assert_eq!(&midi[..4], b"MThd");
    let log = stderr(&out);
    assert!(
        log.contains("short_basic_pitch.mid (1 note, 0.6 s)"),
        "{log}"
    );
    assert!(
        !log.contains('\r') && !log.contains('\u{1b}'),
        "no progress bar off a terminal"
    );
}

#[test]
fn quiet_prints_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let out = bin()
        .arg(short_wav())
        .arg("-o")
        .arg(dir.path())
        .arg("-q")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(out.stdout.is_empty() && out.stderr.is_empty());
}

#[test]
fn writes_every_requested_format() {
    let dir = tempfile::tempdir().unwrap();
    let out = bin()
        .arg(short_wav())
        .arg("-o")
        .arg(dir.path().join("new/sub"))
        .args(["--midi", "--json", "--csv", "--posteriorgrams"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("short_basic_pitch.{mid,json,csv,npz}"),
        "{}",
        stderr(&out)
    );
    let sub = dir.path().join("new/sub");
    for ext in ["mid", "json", "csv", "npz"] {
        assert!(
            sub.join(format!("short_basic_pitch.{ext}")).is_file(),
            "{ext}"
        );
    }

    let npz = std::fs::read(sub.join("short_basic_pitch.npz")).unwrap();
    let entries = basic_pitch_cli::npz::read_npz(&npz).unwrap();
    let shapes: Vec<(&str, (usize, usize))> = entries
        .iter()
        .map(|(name, npy)| {
            let a = ndarray::Array2::<f32>::read_npy(npy.as_slice()).unwrap();
            (name.as_str(), a.dim())
        })
        .collect();
    assert_eq!(
        shapes,
        [
            ("note.npy", (51, 88)),
            ("onset.npy", (51, 88)),
            ("contour.npy", (51, 264))
        ]
    );

    let csv = std::fs::read_to_string(sub.join("short_basic_pitch.csv")).unwrap();
    assert!(csv.starts_with("start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\r\n"));
}

#[test]
fn output_dash_writes_to_stdout() {
    let out = bin()
        .arg(short_wav())
        .args(["--csv", "--output", "-", "-q"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    let csv = String::from_utf8(out.stdout).unwrap();
    assert_eq!(csv.lines().count(), 2, "header and one note: {csv}");

    // Binary formats go to stdout too when it is not a terminal.
    let out = bin()
        .arg(short_wav())
        .args(["--output", "-", "-q"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(&out.stdout[..4], b"MThd");
}

#[test]
fn output_file_format_follows_its_extension() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("notes.json");
    let out = bin()
        .arg(short_wav())
        .arg("--output")
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(doc["schema_version"], 1);
}

#[test]
fn existing_outputs_need_force() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("short_basic_pitch.mid");
    std::fs::write(&target, b"keep me").unwrap();

    let out = bin()
        .arg(short_wav())
        .arg("-o")
        .arg(dir.path())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("--force"), "{}", stderr(&out));
    assert_eq!(std::fs::read(&target).unwrap(), b"keep me");

    let out = bin()
        .arg(short_wav())
        .arg("-o")
        .arg(dir.path())
        .arg("--force")
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(&std::fs::read(&target).unwrap()[..4], b"MThd");
}

#[test]
fn missing_file_is_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let out = bin()
        .arg("no/such/take.wav")
        .arg("-o")
        .arg(dir.path())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let log = stderr(&out);
    assert!(
        log.starts_with("error: cannot read no/such/take.wav: "),
        "{log}"
    );
    assert!(!log.contains("panicked"), "{log}");
}

#[test]
fn unsupported_format_is_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let fake = dir.path().join("notes.wav");
    std::fs::write(&fake, "this is not audio").unwrap();
    let out = bin().arg(&fake).arg("-o").arg(dir.path()).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let log = stderr(&out);
    assert!(
        log.contains("unsupported or unrecognized audio format"),
        "{log}"
    );
    assert!(log.contains("WAV, AIFF"), "{log}");
}

#[test]
fn one_failed_input_does_not_stop_the_others() {
    let dir = tempfile::tempdir().unwrap();
    let out = bin()
        .arg("missing.flac")
        .arg(short_wav())
        .arg("-o")
        .arg(dir.path())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(dir.path().join("short_basic_pitch.mid").is_file());
    assert!(
        stderr(&out).contains("1 of 2 inputs failed"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn usage_errors_exit_with_2() {
    for args in [
        &["a.wav", "b.wav", "--output", "x.mid"][..],
        &["a.wav", "--output", "x.txt"][..],
        &["a.wav", "--onset-threshold", "nan"][..],
        &["a.wav", "--trim", "v0.3"][..],
        &[][..],
    ] {
        let out = bin().args(args).output().unwrap();
        assert_eq!(out.status.code(), Some(2), "{args:?}: {}", stderr(&out));
        assert!(stderr(&out).starts_with("error: "), "{args:?}");
    }
}

#[test]
fn help_has_examples() {
    let out = bin().arg("--help").output().unwrap();
    assert!(out.status.success());
    let help = String::from_utf8(out.stdout).unwrap();
    for needle in [
        "Examples:",
        "--onset-threshold",
        "--resampler",
        "--trim",
        "v0.4.0",
        "--bend-offset",
    ] {
        assert!(help.contains(needle), "{needle}");
    }
}

fn write_wav<S: hound::Sample + Copy>(path: &Path, spec: hound::WavSpec, frames: &[Vec<S>]) {
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for frame in frames {
        for &s in frame {
            writer.write_sample(s).unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn spec(channels: u16, rate: u32, bits: u16, format: hound::SampleFormat) -> hound::WavSpec {
    hound::WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: bits,
        sample_format: format,
    }
}

/// Integer samples become floats exactly like libsndfile (`soundfile`, which librosa
/// reads with) makes them: divided by 2^(bits - 1). Channels are averaged in f32.
#[test]
#[allow(clippy::cast_precision_loss, reason = "test values are exact in f32")]
fn wav_sample_formats_decode_like_libsndfile() {
    let dir = tempfile::tempdir().unwrap();
    let int = hound::SampleFormat::Int;

    let path = dir.path().join("s16.wav");
    let frames = vec![vec![i16::MIN, i16::MAX], vec![1, -3], vec![12_345, 0]];
    write_wav(&path, spec(2, 48_000, 16, int), &frames);
    let decoded = audio::decode(&path).unwrap();
    assert_eq!((decoded.sample_rate, decoded.channels), (48_000, 2));
    assert_eq!(decoded.codec, "pcm_s16le");
    let expected: Vec<f32> = frames
        .iter()
        .map(|f| (f32::from(f[0]) / 32_768.0 + f32::from(f[1]) / 32_768.0) / 2.0)
        .collect();
    assert_eq!(decoded.mono, expected);

    let path = dir.path().join("s24.wav");
    let values = [-8_388_608_i32, 8_388_607, 1, -77];
    let frames: Vec<Vec<i32>> = values.iter().map(|&v| vec![v]).collect();
    write_wav(&path, spec(1, 44_100, 24, int), &frames);
    let decoded = audio::decode(&path).unwrap();
    let expected: Vec<f32> = values.iter().map(|&v| v as f32 / 8_388_608.0).collect();
    assert_eq!(decoded.mono, expected);

    let path = dir.path().join("s32.wav");
    let values = [i32::MIN, i32::MAX, 1, -123_456_789];
    let frames: Vec<Vec<i32>> = values.iter().map(|&v| vec![v]).collect();
    write_wav(&path, spec(1, 96_000, 32, int), &frames);
    let decoded = audio::decode(&path).unwrap();
    // libsndfile rounds the integer to f32, then scales by 2^-31.
    let expected: Vec<f32> = values.iter().map(|&v| v as f32 / 2_147_483_648.0).collect();
    assert_eq!(decoded.mono, expected);

    let path = dir.path().join("f32.wav");
    let frames = vec![
        vec![0.25_f32, -0.5, 1.5],
        vec![f32::MIN_POSITIVE, 0.0, -1.0],
    ];
    write_wav(
        &path,
        spec(3, 16_000, 32, hound::SampleFormat::Float),
        &frames,
    );
    let decoded = audio::decode(&path).unwrap();
    assert_eq!(decoded.channels, 3);
    let expected: Vec<f32> = frames.iter().map(|f| (f[0] + f[1] + f[2]) / 3.0).collect();
    assert_eq!(decoded.mono, expected);
}

#[test]
fn empty_wav_gives_no_notes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.wav");
    write_wav::<i16>(&path, spec(1, 44_100, 16, hound::SampleFormat::Int), &[]);
    let out = bin()
        .arg(&path)
        .args(["--json", "--output", "-", "-q"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(doc["notes"], serde_json::json!([]));
    assert_eq!(doc["analysis"]["frames"], 0);
}

#[test]
fn non_finite_samples_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nan.wav");
    let frames = vec![vec![0.0_f32], vec![f32::NAN]];
    write_wav(
        &path,
        spec(1, 22_050, 32, hound::SampleFormat::Float),
        &frames,
    );
    let out = bin().arg(&path).arg("-o").arg(dir.path()).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("not finite"), "{}", stderr(&out));
}

/// AAC starts with 2112 priming samples; they must be dropped so that notes are not
/// late. `afconvert` (macOS) records them in an `iTunSMPB` tag in MP4, and in the packet
/// table of CAF files.
#[cfg(target_os = "macos")]
#[test]
fn aac_priming_samples_are_dropped() {
    check_aac_priming("m4af", "take.m4a");
    check_aac_priming("caff", "take.caf");
}

#[cfg(target_os = "macos")]
fn check_aac_priming(container: &str, name: &str) {
    let dir = tempfile::tempdir().unwrap();
    let wav = fixtures_dir().join("audio/vocadito_14.wav");
    let encoded = dir.path().join(name);
    let converted = Command::new("afconvert")
        .args(["-f", container, "-d", "aac"])
        .arg(&wav)
        .arg(&encoded)
        .status();
    if !converted.is_ok_and(|s| s.success()) {
        eprintln!("afconvert is not available, skipping");
        return;
    }
    let reference = audio::decode(&wav).unwrap().mono;
    let decoded = audio::decode(&encoded).unwrap();
    assert_eq!(decoded.codec, "aac");
    assert_eq!(
        decoded.mono.len(),
        reference.len(),
        "priming and padding are dropped"
    );

    // The decoded audio lines up with the original: no lag beats lag 0.
    let correlation = |lag: usize| -> f64 {
        (20_000..120_000)
            .map(|i| f64::from(decoded.mono[i + lag]) * f64::from(reference[i]))
            .sum()
    };
    let aligned = correlation(0);
    let reverse = |lag: usize| -> f64 {
        (20_000..120_000)
            .map(|i| f64::from(decoded.mono[i]) * f64::from(reference[i + lag]))
            .sum()
    };
    for lag in [1, 2, 64, 1024, 2112] {
        assert!(
            aligned > correlation(lag) && aligned > reverse(lag),
            "{name}: lag {lag}"
        );
    }
}

/// A NaN in a file that needs resampling is reported in the file's own time, before the
/// resampler sees it (rubato would panic, soxr would smear it).
#[test]
fn non_finite_samples_are_rejected_before_resampling() {
    let dir = tempfile::tempdir().unwrap();
    for (name, bad) in [("nan_48k.wav", f32::NAN), ("inf_48k.wav", f32::INFINITY)] {
        let path = dir.path().join(name);
        let mut frames = vec![vec![0.1_f32, 0.1]; 48_000];
        frames[1000][1] = bad;
        write_wav(
            &path,
            spec(2, 48_000, 32, hound::SampleFormat::Float),
            &frames,
        );
        let out = bin().arg(&path).arg("-o").arg(dir.path()).output().unwrap();
        let log = stderr(&out);
        assert_eq!(out.status.code(), Some(1), "{name}: {log}");
        assert!(!log.contains("panicked"), "{name}: {log}");
        assert!(
            log.contains("sample 1000 (at 0.021 s) is not finite"),
            "{name}: {log}"
        );
    }
}

#[test]
fn negative_and_out_of_range_values_are_usage_errors() {
    for args in [
        &["--frame-threshold", "-0.1"][..],
        &["--frame-threshold=-0.5"][..],
        &["--onset-threshold", "2"][..],
        &["--bend-offset", "7"][..],
        &["--midi-tempo", "1e9"][..],
        &["--midi-tempo", "3"][..],
        &["--minimum-frequency", "1000", "--maximum-frequency", "100"][..],
        &["--minimum-frequency", "5000"][..],
        &["--maximum-frequency", "20"][..],
    ] {
        let out = bin().arg(short_wav()).args(args).output().unwrap();
        let log = stderr(&out);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {log}");
        assert!(!log.contains("unexpected argument"), "{args:?}: {log}");
    }
    let out = bin()
        .arg(short_wav())
        .args(["--frame-threshold", "-0.1"])
        .output()
        .unwrap();
    assert!(
        stderr(&out).contains("must be between 0 and 1"),
        "{}",
        stderr(&out)
    );
}

/// A minimum frequency below A0 would remove nearly every note in Python (a negative
/// numpy index); the tool drops it with a warning.
#[test]
fn minimum_frequency_below_the_piano_is_ignored() {
    let pluck = fixtures_dir().join("audio/pluck.wav");
    let notes = |extra: &[&str]| {
        let out = bin()
            .arg(&pluck)
            .args(["--json", "--output", "-"])
            .args(extra)
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", stderr(&out));
        let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        (doc["notes"].clone(), doc["params"].clone(), stderr(&out))
    };
    let (reference, _, _) = notes(&[]);
    let (low, params, log) = notes(&["--minimum-frequency", "20"]);
    assert_eq!(low, reference);
    assert_eq!(params["minimum_frequency"], serde_json::Value::Null);
    assert!(
        log.contains("warning: --minimum-frequency 20 Hz is below A0"),
        "{log}"
    );
}

#[test]
fn directories_are_not_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let out = bin().arg(dir.path()).arg(short_wav()).output().unwrap();
    let log = stderr(&out);
    assert_eq!(out.status.code(), Some(2), "{log}");
    assert!(log.contains("is a directory, not an audio file"), "{log}");
    assert!(log.contains("use -o"), "{log}");
    assert!(!dir.path().join("short_basic_pitch.mid").exists());
}

#[test]
fn empty_files_are_named_as_such() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zero.wav");
    std::fs::write(&path, b"").unwrap();
    let out = bin().arg(&path).arg("-o").arg(dir.path()).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("is empty (0 bytes)"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn output_extension_must_match_the_format() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("x.mid");
    let out = bin()
        .arg(short_wav())
        .arg("--output")
        .arg(&target)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert!(stderr(&out).contains("extension"), "{}", stderr(&out));
    assert!(!target.exists());
    // An extension that names no format is fine.
    let other = dir.path().join("notes.txt");
    let out = bin()
        .arg(short_wav())
        .arg("--output")
        .arg(&other)
        .arg("--json")
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
}

#[test]
fn python_flag_names_are_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let out = bin()
        .arg(short_wav())
        .args(["--save-midi", "--save-note-events", "-q", "-o"])
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(dir.path().join("short_basic_pitch.mid").is_file());
    assert!(dir.path().join("short_basic_pitch.csv").is_file());
}

/// `basic-pitch-rs ... --output - | head` stops quietly when the reader goes away.
#[test]
fn closed_stdout_is_not_an_error() {
    use std::process::Stdio;
    let mut child = bin()
        .arg(fixtures_dir().join("audio/vocadito_14.wav"))
        .args(["--json", "--output", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Close the read end before the tool writes anything.
    drop(child.stdout.take());
    let out = child.wait_with_output().unwrap();
    let log = stderr(&out);
    assert!(out.status.success(), "{log}");
    assert!(!log.contains("error"), "{log}");
}

/// The thread count does not change the result.
#[test]
fn threads_give_identical_notes() {
    let csv = |threads: &str| {
        let out = bin()
            .arg(fixtures_dir().join("audio/vocadito_14.wav"))
            .args(["--csv", "--output", "-", "-q", "--threads", threads])
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", stderr(&out));
        out.stdout
    };
    let one = csv("1");
    assert!(!one.is_empty());
    assert_eq!(csv("3"), one);
    assert_eq!(csv("16"), one);
}

/// Both backends give the same notes.
#[cfg(feature = "ort")]
#[test]
fn ort_backend_gives_the_same_notes() {
    let notes = |backend: &str| {
        let out = bin()
            .arg(fixtures_dir().join("audio/vocadito_10.wav"))
            .args(["--csv", "--output", "-", "-q", "--backend", backend])
            .output()
            .unwrap();
        assert!(out.status.success(), "{backend}: {}", stderr(&out));
        String::from_utf8(out.stdout).unwrap()
    };
    assert_eq!(notes("ort"), notes("tract"));
}

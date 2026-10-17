"""Writes the synthetic test signals used by the parity fixtures.

Every file is deterministic, so regenerating them gives the same bytes.

  synth.wav     44.1 kHz mono PCM 24: sine A3, band-limited saw C4, C major triad,
                E3 -> G3 glide, short saw A4.
  pluck.wav     48 kHz stereo PCM 24: decaying harmonic tones shaped like guitar notes,
                with a whole-tone bend, a vibrato, a fast run, a held note and a power chord.
  silence.wav   44.1 kHz mono PCM 16, 3 s of digital silence (exercises the NaN path of
                get_infered_onsets).
  short.wav     22.05 kHz mono PCM 16, 0.6 s: shorter than one model window, no resampling.

Usage: python make_synth.py OUT_DIR
"""

import pathlib
import sys

import numpy as np
import soundfile as sf


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def envelope(n, sr, attack=0.01, release=0.05):
    env = np.ones(n)
    a = max(1, int(attack * sr))
    r = max(1, int(release * sr))
    env[:a] = np.linspace(0.0, 1.0, a)
    env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def osc(freq_hz, n, sr, harmonics=1, decay=None):
    """Additive oscillator following an instantaneous frequency curve."""
    f = np.broadcast_to(np.asarray(freq_hz, dtype=np.float64), (n,))
    phase = 2.0 * np.pi * np.cumsum(f) / sr
    out = np.zeros(n)
    for k in range(1, harmonics + 1):
        if np.max(f) * k >= sr / 2:
            break
        out += np.sin(k * phase) / k
    if decay is not None:
        out *= np.exp(-np.arange(n) / (decay * sr))
    return out


def place(buf, sig, start, sr, gain, attack=0.01, release=0.05):
    i = int(round(start * sr))
    sig = sig * envelope(len(sig), sr, attack, release) * gain
    buf[i : i + len(sig)] += sig


def synth(sr=44100):
    buf = np.zeros(int(7.5 * sr))
    place(buf, osc(midi_to_hz(57), sr, sr), 0.50, sr, 0.5)
    place(buf, osc(midi_to_hz(60), sr, sr, harmonics=40) * (2 / np.pi), 1.75, sr, 0.35)
    for m in (60, 64, 67):
        place(buf, osc(midi_to_hz(m), int(1.5 * sr), sr), 3.00, sr, 0.25)
    n = int(1.5 * sr)
    glide = np.concatenate([np.full(n // 3, 52.0), np.linspace(52.0, 55.0, n // 3), np.full(n - 2 * (n // 3), 55.0)])
    place(buf, osc(midi_to_hz(glide), n, sr), 4.75, sr, 0.5)
    place(buf, osc(midi_to_hz(69), int(0.5 * sr), sr, harmonics=40) * (2 / np.pi), 6.50, sr, 0.35)
    return buf / max(1.0, np.max(np.abs(buf)) / 0.9)


def pluck(sr=48000):
    rng = np.random.default_rng(7)
    buf = np.zeros(int(10.0 * sr))

    def note(start, dur, pitch_curve, gain=0.4, decay=1.2):
        n = int(dur * sr)
        curve = np.broadcast_to(np.asarray(pitch_curve, dtype=np.float64), (n,))
        place(buf, osc(midi_to_hz(curve), n, sr, harmonics=12, decay=decay), start, sr, gain, attack=0.003)

    # isolated notes
    for i, m in enumerate((64, 67, 69)):
        note(0.3 + 0.45 * i, 0.4, m)
    # whole-tone bend on B3 -> C#4, held at the top
    n = int(1.4 * sr)
    bend = np.concatenate([np.full(int(0.25 * sr), 59.0), np.linspace(59.0, 61.0, int(0.25 * sr))])
    bend = np.concatenate([bend, np.full(n - len(bend), 61.0)])
    note(1.8, 1.4, bend, decay=2.0)
    # vibrato on A4, +/- 0.3 semitone at 6 Hz
    n = int(1.5 * sr)
    vib = 69.0 + 0.3 * np.sin(2 * np.pi * 6.0 * np.arange(n) / sr) * np.clip(np.arange(n) / (0.4 * sr), 0, 1)
    note(3.5, 1.5, vib, decay=2.5)
    # fast run, 16th notes at 140 bpm
    for i, m in enumerate((69, 67, 64, 62, 64, 67, 69, 72)):
        note(5.3 + i * 0.107, 0.12, m, gain=0.35, decay=0.4)
    # long held note
    note(6.4, 2.0, 57, decay=3.0)
    # power chord E2 + B2 + E3
    for m in (40, 47, 52):
        note(8.6, 1.2, m, gain=0.25, decay=1.5)

    buf += rng.normal(0.0, 1e-4, buf.shape)  # a little noise floor, like a real recording
    buf /= max(1.0, np.max(np.abs(buf)) / 0.9)
    right = np.roll(buf, 12) * 0.9  # slightly different right channel, so the downmix matters
    return np.stack([buf, right], axis=1)


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "synth.wav", synth().astype(np.float32), 44100, subtype="PCM_24")
    sf.write(out / "pluck.wav", pluck().astype(np.float32), 48000, subtype="PCM_24")
    sf.write(out / "silence.wav", np.zeros(3 * 44100, dtype=np.float32), 44100, subtype="PCM_16")
    short = synth(22050)[int(0.5 * 22050) : int(1.1 * 22050)]
    sf.write(out / "short.wav", short.astype(np.float32), 22050, subtype="PCM_16")
    for p in sorted(out.glob("*.wav")):
        info = sf.info(p)
        print(f"{p.name}: {info.samplerate} Hz, {info.channels} ch, {info.frames} frames, {info.subtype}")


if __name__ == "__main__":
    main()

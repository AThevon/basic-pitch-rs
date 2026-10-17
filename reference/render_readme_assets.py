"""Renders the README visuals from real basic-pitch-rs output.

  assets/hero.svg            The contour posteriorgram of tests/fixtures/audio/pluck.wav
                             in gray, with the notes and pitch bends that basic-pitch-rs
                             detects drawn over it.
  assets/social-preview.png  The 1280x640 image for the GitHub social preview.

Nothing is drawn by hand: the script runs the release build of the command-line tool on
pluck.wav with the default parameters (`--json --posteriorgrams`) and draws what it
returns. The text is converted to paths with usvg, so the SVG looks the same everywhere
and needs no web font.

Requirements, besides uv:
  - cargo (the script builds and runs the CLI from this workspace);
  - resvg and usvg: `cargo install resvg usvg --locked`;
  - the static Familjen Grotesk fonts (SIL Open Font License), FamiljenGrotesk-Regular.ttf
    and FamiljenGrotesk-Medium.ttf, in the directory given with --fonts. The static
    instances are linked from https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@400;500
    (fetched with a plain HTTP client, the CSS lists the .ttf files).

Usage, from this directory:
  uv run --locked python -I render_readme_assets.py --fonts DIR
"""

import argparse
import base64
import json
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUDIO = ROOT / "tests" / "fixtures" / "audio" / "pluck.wav"
ASSETS = ROOT / "assets"

BG = (0x13, 0x15, 0x18)
INK = (0xC9, 0xCF, 0xD8)  # brightest posteriorgram value
ACCENT = "#6fcf7c"
TEXT = "#e6e9ee"
MUTED = "#8b929c"
GRID = "#262a30"
FONT = "Familjen Grotesk"

CONTOUR_BINS_PER_SEMITONE = 3
LOWEST_MIDI = 21  # contour bin 0 and note bin 0 are A0
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FRAMES_PER_SECOND = 22050 / 256


def note_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def transcribe(out_dir):
    cmd = [
        "cargo", "run", "--release", "--locked", "--quiet", "-p", "basic-pitch-cli", "--",
        str(AUDIO), "--json", "--posteriorgrams", "--quiet", "--force", "-o", str(out_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    stem = AUDIO.stem + "_basic_pitch"
    doc = json.loads((out_dir / f"{stem}.json").read_text())
    contour = np.load(out_dir / f"{stem}.npz")["contour"]
    return doc, contour


def png_rgb(pixels):
    """Encodes an (h, w, 3) uint8 array as a PNG, without dependencies."""
    h, w, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(h))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def heatmap(contour, lo_midi, hi_midi, frames):
    """The contour activations between two MIDI pitches, highest pitch on top."""
    lo = (lo_midi - LOWEST_MIDI) * CONTOUR_BINS_PER_SEMITONE
    hi = (hi_midi - LOWEST_MIDI) * CONTOUR_BINS_PER_SEMITONE + 1
    # The sigmoid outputs rest around 0.1: the gray scale spans 0.1 to 0.9.
    values = np.clip((contour[frames[0] : frames[1], lo:hi].T[::-1] - 0.1) / 0.8, 0.0, 1.0) ** 0.7
    bg = np.array(BG, dtype=np.float64)
    ink = np.array(INK, dtype=np.float64)
    rgb = bg + values[..., None] * (ink - bg)
    return png_rgb(np.round(rgb).astype(np.uint8))


class Plot:
    """Maps frames and MIDI pitches (bin centers) to SVG coordinates."""

    def __init__(self, x, y, w, h, frames, lo_midi, hi_midi):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.f0, self.f1 = frames
        self.lo, self.hi = lo_midi, hi_midi
        # One row per contour bin; bin centers sit half a row inside the edges.
        self.rows = (hi_midi - lo_midi) * CONTOUR_BINS_PER_SEMITONE + 1
        self.row_h = h / self.rows

    def fx(self, frame):
        return self.x + (frame - self.f0) * self.w / (self.f1 - self.f0)

    def py(self, midi):
        row = (self.hi - midi) * CONTOUR_BINS_PER_SEMITONE
        return self.y + (row + 0.5) * self.row_h

    @property
    def semitone(self):
        return self.row_h * CONTOUR_BINS_PER_SEMITONE


def draw_plot(plot, notes, contour, stroke=3.0):
    out = []
    png = heatmap(contour, plot.lo, plot.hi, (plot.f0, plot.f1))
    href = "data:image/png;base64," + base64.b64encode(png).decode()
    out.append(
        f'<image x="{plot.x}" y="{plot.y}" width="{plot.w}" height="{plot.h}" '
        f'preserveAspectRatio="none" href="{href}"/>'
    )
    for note in notes:
        s, e = note["start_frame"], note["end_frame"]
        if e <= plot.f0 or s >= plot.f1:
            continue
        p = note["pitch"]
        x0, x1 = plot.fx(max(s, plot.f0)), plot.fx(min(e, plot.f1))
        half = plot.semitone * 0.4
        y = plot.py(p)
        out.append(
            f'<rect x="{x0:.1f}" y="{y - half:.1f}" width="{x1 - x0:.1f}" height="{2 * half:.1f}" rx="2" '
            f'fill="{ACCENT}" fill-opacity="0.3"/>'
        )
        # Pitch bends are one integer per frame, in thirds of a semitone: drawn as steps,
        # joined only between neighboring bins so that isolated jumps stay visible as such.
        path, last = [], None
        for i, bend in enumerate(note["pitch_bends"]):
            frame = s + i
            if frame < plot.f0 or frame + 1 > plot.f1:
                continue
            by = plot.py(p + bend / CONTOUR_BINS_PER_SEMITONE)
            if last is not None and abs(bend - last) <= 1:
                path.append(f"V{by:.1f}")
            else:
                path.append(f"M{plot.fx(frame):.1f},{by:.1f}")
            path.append(f"H{plot.fx(frame + 1):.1f}")
            last = bend
        if path:
            out.append(
                f'<path d="{"".join(path)}" fill="none" stroke="{ACCENT}" '
                f'stroke-width="{stroke}" stroke-linejoin="round" stroke-linecap="round"/>'
            )
    return out


def hero_svg(doc, contour):
    width, height = 1600, 820
    lo, hi = 38, 76
    plot = Plot(118, 112, 1430, 600, (0, contour.shape[0]), lo, hi)
    notes = doc["notes"]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
        f'<rect width="{width}" height="{height}" rx="20" fill="rgb{BG}"/>',
        f'<text x="{plot.x}" y="64" font-size="30"><tspan font-weight="500" fill="{TEXT}">pluck.wav</tspan>'
        f'<tspan dx="16" fill="{MUTED}">model contour in gray, the {len(notes)} notes and their pitch bends in green</tspan></text>',
    ]
    # Octave lines and labels on every C, plus E2, the lowest note of the file.
    for midi in range(lo, hi + 1):
        if midi % 12 == 0:
            y = plot.py(midi)
            out.append(f'<line x1="{plot.x}" x2="{plot.x + plot.w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1.5"/>')
        if midi % 12 == 0 or midi == 40:
            out.append(
                f'<text x="{plot.x - 18}" y="{plot.py(midi) + 8:.1f}" font-size="26" fill="{MUTED}" '
                f'text-anchor="end">{note_name(midi)}</text>'
            )
    out += draw_plot(plot, notes, contour)
    seconds = int((plot.f1 - 1) / FRAMES_PER_SECOND)
    for t in range(0, seconds + 1):
        x = plot.fx(t * FRAMES_PER_SECOND)
        label = f"{t} s" if t in (0, seconds) else str(t)
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{plot.y + plot.h + 6}" y2="{plot.y + plot.h + 16}" '
            f'stroke="{MUTED}" stroke-width="1.5"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{plot.y + plot.h + 48}" font-size="26" fill="{MUTED}" '
            f'text-anchor="middle">{label}</text>'
        )
    labels = [
        (0.75, "single notes"),
        (2.5, "whole-tone bend"),
        (4.25, "vibrato"),
        (5.75, "fast run"),
        (7.4, "held note"),
        (9.2, "power chord"),
    ]
    for t, text in labels:
        out.append(
            f'<text x="{plot.fx(t * FRAMES_PER_SECOND):.1f}" y="{height - 30}" font-size="26" '
            f'fill="{TEXT}" text-anchor="middle">{text}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def social_svg(doc, contour):
    width, height = 1280, 640
    lo, hi = 38, 76
    plot = Plot(80, 318, 1120, 250, (290, contour.shape[0]), lo, hi)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
        f'<rect width="{width}" height="{height}" fill="rgb{BG}"/>',
        f'<text x="76" y="150" font-size="88" font-weight="500" fill="{TEXT}">basic-pitch-rs</text>',
        f'<text x="80" y="214" font-size="38" fill="{TEXT}">polyphonic audio to MIDI, with pitch bends, in Rust</text>',
        f'<text x="80" y="264" font-size="28" fill="{MUTED}">an unofficial, parity-tested port of Spotify\'s Basic Pitch</text>',
    ]
    out += draw_plot(plot, doc["notes"], contour, stroke=2.4)
    out.append("</svg>")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fonts", type=pathlib.Path, required=True, help="directory with the Familjen Grotesk .ttf files")
    args = parser.parse_args()

    fonts = [args.fonts / "FamiljenGrotesk-Regular.ttf", args.fonts / "FamiljenGrotesk-Medium.ttf"]
    missing = [str(f) for f in fonts if not f.is_file()]
    missing += [tool for tool in ("cargo", "usvg", "resvg") if shutil.which(tool) is None]
    if missing:
        sys.exit("missing: " + ", ".join(missing) + " (see the docstring)")
    font_args = ["--skip-system-fonts"]
    for f in fonts:
        font_args += ["--use-font-file", str(f)]

    ASSETS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        doc, contour = transcribe(tmp)
        (tmp / "hero.svg").write_text(hero_svg(doc, contour))
        (tmp / "social.svg").write_text(social_svg(doc, contour))
        subprocess.run(["usvg", *font_args, str(tmp / "hero.svg"), str(ASSETS / "hero.svg")], check=True)
        subprocess.run(["resvg", *font_args, str(tmp / "social.svg"), str(ASSETS / "social-preview.png")], check=True)
    for name in ("hero.svg", "social-preview.png"):
        print(f"assets/{name}: {(ASSETS / name).stat().st_size} bytes")


if __name__ == "__main__":
    main()

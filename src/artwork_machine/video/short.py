"""
30-second vertical short generator.

Produces a 1080×1920 (9:16) MP4 — ready for YouTube Shorts, Instagram Reels,
and TikTok.  Uses the first 30 seconds of audio (or full track if shorter).

Visual layout
─────────────
┌─────────────────────────────┐
│                             │
│  ARTIST NAME                │  ← top 15%
│  Track Title                │
│                             │
│  ┌───────────────────────┐  │
│  │                       │  │
│  │    ALBUM ART          │  │  ← centre 55% (with Ken-Burns)
│  │    (square, centred)  │  │
│  │                       │  │
│  └───────────────────────┘  │
│                             │
│  ▁▂▃▅▇▆▄▃▂▁  spectrum      │  ← bottom 30%
│  ─────────────  progress    │
└─────────────────────────────┘

Background: blurred + darkened version of album art.
Frames piped directly to ffmpeg — no temp disk writes.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from artwork_machine.ai.prompt_generator import CreativeDirection
from artwork_machine.audio.analyzer import AudioFeatures
from artwork_machine.video.effects import bloom, colour_grade, vignette
from artwork_machine.artwork.utils import hex_to_rgba, load_font

WIDTH    = 1080
HEIGHT   = 1920
FPS      = 30
DURATION = 30   # seconds (capped to track length if shorter)


def generate(
    album_art_path: Path,
    audio_path: Path,
    direction: CreativeDirection,
    features: AudioFeatures,
    artist: str,
    album: str,
    output_path: Path,
    *,
    draft: bool = False,
) -> Path:
    """
    Render a 30-second vertical short and mux with audio.

    Frames are streamed directly to ffmpeg via pipe.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quality_divisor = 2 if draft else 1
    w, h   = WIDTH // quality_divisor, HEIGHT // quality_divisor
    fps    = 20 if draft else FPS
    dur    = min(DURATION, int(features.duration))

    accent_colour  = _hex_to_f3(direction.palette_accent)
    primary_colour = _hex_to_f3(direction.palette_primary)
    secondary_colour = _hex_to_f3(direction.palette_secondary)

    # Prepare assets
    bg_blur  = _make_blurred_bg(album_art_path, w, h)
    art_np   = _prepare_art(album_art_path, w)

    total_frames  = fps * dur
    spectrum_data = _resample_frames(features.spectrum_frames, total_frames)
    rms_data      = _resample_curve(features.rms_curve, total_frames)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise RuntimeError("ffmpeg not found.")

    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "pipe:0",
        # Trim audio to dur seconds
        "-ss", "0", "-t", str(dur), "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "256k",
        "-shortest", "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for fi in range(total_frames):
            t        = fi / fps
            energy   = rms_data[fi] if fi < len(rms_data) else 0.5
            spectrum = np.array(spectrum_data[fi]) if fi < len(spectrum_data) else np.zeros(64)

            frame = _render_frame(
                t=t, energy=energy, spectrum=spectrum,
                bg_blur=bg_blur, art_np=art_np,
                direction=direction, features=features,
                artist=artist, album=album, dur=dur,
                accent_colour=accent_colour, primary_colour=primary_colour,
                secondary_colour=secondary_colour,
                w=w, h=h, draft=draft,
            )

            rgb = (np.clip(frame, 0, 1) * 255).astype(np.uint8)[:, :, :3]
            proc.stdin.write(rgb.tobytes())

    finally:
        proc.stdin.close()

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr.read().decode()}")

    return output_path


# ── Frame renderer ─────────────────────────────────────────────────────────────

def _render_frame(
    *, t, energy, spectrum, bg_blur, art_np, direction, features,
    artist, album, dur, accent_colour, primary_colour, secondary_colour,
    w, h, draft,
) -> np.ndarray:

    # Background
    frame = bg_blur.copy()

    # Album art centred
    frame = _composite_art(frame, art_np, t, features.bpm, w, h)

    # Artist / album text at top
    frame = _draw_top_text(frame, artist, album, direction, w, h)

    # Spectrum bars at bottom
    frame = _draw_spectrum_bars(frame, spectrum, energy, accent_colour, secondary_colour, w, h)

    # Progress bar
    frame = _draw_progress_bar(frame, t / max(dur, 1), accent_colour, primary_colour, w, h)

    if not draft:
        frame = bloom(frame, radius=6, strength=0.25)
        frame = colour_grade(
            frame,
            shadows=tuple(c * 0.08 for c in primary_colour),  # type: ignore
            highlights=(1.03, 1.01, 1.0),
        )

    frame = vignette(frame, strength=0.55)
    return frame


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _composite_art(frame, art, t, bpm, w, h):
    ah, aw = art.shape[:2]
    scale  = 1.0 + 0.010 * math.sin(math.tau * bpm / 60 * t)

    art_pil  = Image.fromarray((art * 255).astype(np.uint8), "RGBA")
    scaled_w = int(aw * scale)
    scaled_h = int(ah * scale)
    art_pil  = art_pil.resize((scaled_w, scaled_h), Image.BILINEAR)
    sx = (scaled_w - aw) // 2
    sy = (scaled_h - ah) // 2
    art_pil  = art_pil.crop((sx, sy, sx + aw, sy + ah))
    art_arr  = np.array(art_pil, dtype=np.float32) / 255.0

    x0 = (w - aw) // 2
    y0 = int(h * 0.22)

    x0c, x1c = max(0, x0), min(w, x0 + aw)
    y0c, y1c = max(0, y0), min(h, y0 + ah)
    if x1c > x0c and y1c > y0c:
        alpha = art_arr[: y1c-y0c, : x1c-x0c, 3:4]
        frame[y0c:y1c, x0c:x1c, :3] = (
            frame[y0c:y1c, x0c:x1c, :3] * (1-alpha)
            + art_arr[: y1c-y0c, : x1c-x0c, :3] * alpha
        )
    return frame


def _draw_top_text(frame, artist, album, direction, w, h):
    pil    = Image.fromarray((frame * 255).astype(np.uint8), "RGBA")
    draw   = ImageDraw.Draw(pil)
    accent = hex_to_rgba(direction.palette_accent, alpha=230)
    white  = (255, 255, 255, 200)
    margin = int(w * 0.06)
    draw.text((margin, int(h * 0.05)), artist, font=load_font(size=max(24, w // 15), bold=True), fill=accent)
    draw.text((margin, int(h * 0.05) + w // 12), album,  font=load_font(size=max(18, w // 22)),            fill=white)
    return np.array(pil, dtype=np.float32) / 255.0


def _draw_spectrum_bars(frame, spectrum, energy, accent, secondary, w, h):
    n_bars     = len(spectrum)
    bar_area_h = int(h * 0.18)
    bar_area_y = int(h * 0.76)
    bar_margin = 3
    bar_w      = max(1, (w - bar_margin * (n_bars + 1)) // n_bars)

    pil  = Image.fromarray((frame * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)

    for i, val in enumerate(spectrum):
        bh  = int(val * bar_area_h * (0.6 + energy * 0.7))
        bx  = bar_margin + i * (bar_w + bar_margin)
        by  = bar_area_y + bar_area_h - bh
        by2 = bar_area_y + bar_area_h
        blend  = i / n_bars
        colour = tuple(accent[j] * (1-blend) + secondary[j] * blend for j in range(3))
        draw.rectangle([bx, by, bx + bar_w, by2], fill=(*_u8(colour), 200))  # type: ignore
        if bh > 4:
            draw.rectangle([bx, by, bx + bar_w, by + 3], fill=(*_u8(accent), 255))

    return np.array(pil, dtype=np.float32) / 255.0


def _draw_progress_bar(frame, progress, accent, primary, w, h):
    bar_y    = int(h * 0.96)
    bx0, bx1 = int(w * 0.05), int(w * 0.95)
    bh       = max(3, h // 320)
    pil      = Image.fromarray((frame * 255).astype(np.uint8), "RGBA")
    draw     = ImageDraw.Draw(pil)
    draw.rectangle([bx0, bar_y, bx1, bar_y + bh], fill=(*_u8(primary), 80))
    fill_x = int(bx0 + (bx1 - bx0) * progress)
    draw.rectangle([bx0, bar_y, fill_x, bar_y + bh], fill=(*_u8(accent), 220))
    draw.ellipse([fill_x - bh*2, bar_y - bh, fill_x + bh*2, bar_y + bh*2], fill=(*_u8(accent), 255))
    return np.array(pil, dtype=np.float32) / 255.0


# ── Asset preparation ─────────────────────────────────────────────────────────

def _make_blurred_bg(path: Path, w: int, h: int) -> np.ndarray:
    """Load album art, fill the 9:16 frame by scaling + cropping, blur and darken."""
    img = Image.open(str(path)).convert("RGBA")

    # Scale to fill the vertical frame
    scale = max(w / img.width, h / img.height)
    nw, nh = int(img.width * scale), int(img.height * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    img = img.crop((x0, y0, x0 + w, y0 + h))

    # Heavy blur + darken for background feel
    img  = img.filter(ImageFilter.GaussianBlur(radius=30))
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr[:, :, :3] *= 0.35   # darken significantly so art pops
    return arr


def _prepare_art(path: Path, w: int) -> np.ndarray:
    """Scale album art to 75% of frame width (square)."""
    img = Image.open(str(path)).convert("RGBA")
    target = int(w * 0.75)
    img = img.resize((target, target), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


# ── Utilities ──────────────────────────────────────────────────────────────────

def _hex_to_f3(h: str) -> tuple:
    r, g, b, _ = hex_to_rgba(h)
    return (r/255, g/255, b/255)


def _u8(c) -> tuple:
    return tuple(int(x * 255) for x in c[:3])  # type: ignore


def _resample_curve(curve: list, n: int) -> list:
    if not curve:
        return [0.0] * n
    src = np.array(curve, dtype=np.float32)
    return np.interp(np.linspace(0, len(src)-1, n), np.arange(len(src)), src).tolist()


def _resample_frames(frames: list, n: int) -> list:
    if not frames:
        return [[0.0] * 64] * n
    ns = len(frames)
    result = []
    for i in range(n):
        idx = i * (ns-1) / max(n-1, 1)
        lo, hi = int(idx), min(int(idx)+1, ns-1)
        t = idx - lo
        result.append([frames[lo][j]*(1-t) + frames[hi][j]*t for j in range(len(frames[0]))])
    return result

"""
YouTube Visualizer engine.

Produces a full HD (1920x1080, 60fps) video synchronized to the input audio.

Visual layout
-------------
+-----------------------------------------------------------------------------+
|                                                                             |
|   Artist Name                                          [album cover]        |
|   Album Title                                                               |
|                                                                             |
|   --------------------------------------------------  <- waveform          |
|                                                                             |
|   spectrum bars / radial waveform / particle field                          |
|                                                                             |
|   o  ---------------------------------------------- <- progress bar        |
|       0:00                                                                  |
+-----------------------------------------------------------------------------+

Three visualiser styles (selected by Claude based on mood):
  "spectrum bars"    -- classic equaliser bar graph
  "radial waveform"  -- circular waveform pulse (ala Apple Music)
  "particle field"   -- frequency-driven particle fountain
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from artwork_machine.ai.prompt_generator import CreativeDirection
from artwork_machine.audio.analyzer import AudioFeatures
from artwork_machine.video.effects import (
    ParticleSystem,
    bloom,
    colour_grade,
    vignette,
)
from artwork_machine.artwork.utils import hex_to_rgba, load_font

WIDTH = 1920
HEIGHT = 1080
FPS = 60


def generate(
    album_cover_path: Path,
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
    Render the YouTube visualiser and mux with the source audio.

    Parameters
    ----------
    album_cover_path:
        3000x3000 album cover PNG (will be cropped to a right-side panel).
    audio_path:
        Source audio file (any ffmpeg-supported format).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quality_divisor = 2 if draft else 1
    w, h = WIDTH // quality_divisor, HEIGHT // quality_divisor
    fps = 24 if draft else FPS

    # ── Colour palette ────────────────────────────────────────────────────────
    bg_colour = _hex_to_float3(direction.palette_background)
    primary_colour = _hex_to_float3(direction.palette_primary)
    accent_colour = _hex_to_float3(direction.palette_accent)
    secondary_colour = _hex_to_float3(direction.palette_secondary)

    # ── Prepare album cover panel ─────────────────────────────────────────────
    cass_panel = _prepare_art_panel(album_cover_path, w, h)

    # ── Particle system (used for "particle field" style) ─────────────────────
    particles = ParticleSystem(w, h, n_particles=300 if not draft else 80)

    # ── Spectrum / waveform data ──────────────────────────────────────────────
    total_frames = fps * int(features.duration)
    spectrum_data = _resample_frames(features.spectrum_frames, total_frames)  # [N, 64]
    waveform_data = _resample_curve(features.waveform_normalised, total_frames * (w // 4))
    rms_data = _resample_curve(features.rms_curve, total_frames)

    style = direction.visualiser_style.lower()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for fi in range(total_frames):
            t = fi / fps
            energy = rms_data[fi] if fi < len(rms_data) else 0.5
            spectrum = np.array(spectrum_data[fi]) if fi < len(spectrum_data) else np.zeros(64)
            # Waveform slice for this frame
            wv_start = fi * (w // 4)
            wv_end = wv_start + w
            waveform_slice = np.array(waveform_data[wv_start:wv_end]) if wv_end <= len(waveform_data) else np.zeros(w)

            frame = _render_frame(
                t=t,
                fi=fi,
                energy=energy,
                spectrum=spectrum,
                waveform=waveform_slice,
                cass_panel=cass_panel,
                particles=particles,
                style=style,
                direction=direction,
                features=features,
                artist=artist,
                album=album,
                bg_colour=bg_colour,
                primary_colour=primary_colour,
                accent_colour=accent_colour,
                secondary_colour=secondary_colour,
                w=w,
                h=h,
                draft=draft,
            )

            pil = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
            pil.convert("RGB").save(str(tmp_path / f"frame_{fi:06d}.png"))

        _encode_with_audio(tmp_path, audio_path, output_path, fps)

    return output_path


# ── Frame renderer ─────────────────────────────────────────────────────────────

def _render_frame(
    *,
    t: float,
    fi: int,
    energy: float,
    spectrum: np.ndarray,
    waveform: np.ndarray,
    cass_panel: np.ndarray,
    particles: ParticleSystem,
    style: str,
    direction: CreativeDirection,
    features: AudioFeatures,
    artist: str,
    album: str,
    bg_colour: tuple,
    primary_colour: tuple,
    accent_colour: tuple,
    secondary_colour: tuple,
    w: int,
    h: int,
    draft: bool,
) -> np.ndarray:
    # ── Base background ───────────────────────────────────────────────────────
    frame = np.zeros((h, w, 4), dtype=np.float32)
    frame[:, :, :3] = bg_colour
    frame[:, :, 3] = 1.0

    # ── Album cover panel (right side) ───────────────────────────────────────
    frame = _composite_panel(frame, cass_panel, w, h)

    # ── Main visualiser (left 65% of frame) ──────────────────────────────────
    vis_w = int(w * 0.65)

    if "spectrum" in style or "bar" in style:
        frame = _draw_spectrum_bars(frame, spectrum, energy, accent_colour, secondary_colour, vis_w, h)
        frame = _draw_waveform(frame, waveform, energy, primary_colour, vis_w, h)
    elif "radial" in style:
        frame = _draw_radial_waveform(frame, waveform, spectrum, energy, accent_colour, primary_colour, vis_w, h)
    else:
        # particle field
        particles.step(t, energy)
        frame = particles.render(frame, accent_colour)
        frame = _draw_spectrum_bars(frame, spectrum, energy, accent_colour, secondary_colour, vis_w, h)

    # ── Progress bar ─────────────────────────────────────────────────────────
    progress = t / features.duration
    frame = _draw_progress_bar(frame, progress, t, accent_colour, primary_colour, w, h)

    # ── Text (artist / album) ─────────────────────────────────────────────────
    frame = _draw_text(frame, artist, album, direction, w, h)

    # ── Post-processing ───────────────────────────────────────────────────────
    if not draft:
        frame = bloom(frame, radius=8, strength=0.3)
        frame = colour_grade(
            frame,
            shadows=tuple(c * 0.1 for c in primary_colour),  # type: ignore
            highlights=(1.03, 1.01, 1.0),
        )

    frame = vignette(frame, strength=0.45)
    return frame


# ── Visualiser drawing ────────────────────────────────────────────────────────

def _draw_spectrum_bars(
    frame: np.ndarray,
    spectrum: np.ndarray,
    energy: float,
    accent: tuple,
    secondary: tuple,
    vis_w: int,
    h: int,
) -> np.ndarray:
    """Draw 64-band equaliser bars in the lower half of the visualiser area."""
    n_bars = len(spectrum)
    bar_area_h = int(h * 0.38)
    bar_area_y = int(h * 0.58)
    bar_margin = 4
    bar_w = max(1, (vis_w - bar_margin * (n_bars + 1)) // n_bars)
    max_bar_h = bar_area_h

    pil = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)

    for i, val in enumerate(spectrum):
        bh = int(val * max_bar_h * (0.7 + energy * 0.6))
        bx = bar_margin + i * (bar_w + bar_margin)
        by = bar_area_y + max_bar_h - bh
        by2 = bar_area_y + max_bar_h

        # Gradient fill: accent at top → secondary at bottom
        blend = i / n_bars
        bar_colour = _blend_colours(accent, secondary, blend)

        draw.rectangle([bx, by, bx + bar_w, by2], fill=(*_f3_to_u8(bar_colour), 200))

        # Bright top cap
        if bh > 4:
            draw.rectangle([bx, by, bx + bar_w, by + 3], fill=(*_f3_to_u8(accent), 255))

    return np.array(pil, dtype=np.float32) / 255.0


def _draw_waveform(
    frame: np.ndarray,
    waveform: np.ndarray,
    energy: float,
    colour: tuple,
    vis_w: int,
    h: int,
) -> np.ndarray:
    """Draw a horizontal waveform in the centre of the visualiser area."""
    if len(waveform) == 0:
        return frame

    pil = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)

    cy = int(h * 0.42)
    amplitude = int(h * 0.08 * (0.5 + energy))

    xs = np.linspace(20, vis_w - 20, min(len(waveform), vis_w - 40)).astype(int)
    wv_ds = np.interp(np.linspace(0, len(waveform) - 1, len(xs)), np.arange(len(waveform)), waveform)

    points = [(int(xs[i]), int(cy + wv_ds[i] * amplitude)) for i in range(len(xs))]
    if len(points) > 1:
        draw.line(points, fill=(*_f3_to_u8(colour), 180), width=2)

    return np.array(pil, dtype=np.float32) / 255.0


def _draw_radial_waveform(
    frame: np.ndarray,
    waveform: np.ndarray,
    spectrum: np.ndarray,
    energy: float,
    accent: tuple,
    primary: tuple,
    vis_w: int,
    h: int,
) -> np.ndarray:
    """Draw a circular waveform radiating from the centre-left."""
    pil = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)

    cx, cy = vis_w // 2, h // 2
    base_r = min(vis_w, h) // 5
    n_points = 360

    angles = np.linspace(0, math.tau, n_points, endpoint=False)
    wv_indices = np.linspace(0, max(len(waveform) - 1, 1), n_points)
    wv_vals = np.interp(wv_indices, np.arange(len(waveform)), waveform) if len(waveform) > 1 else np.zeros(n_points)
    sp_indices = np.linspace(0, len(spectrum) - 1, n_points)
    sp_vals = np.interp(sp_indices, np.arange(len(spectrum)), spectrum)

    amplitude = base_r * 0.5 * (0.5 + energy)
    radii = base_r + wv_vals * amplitude + sp_vals * base_r * 0.3

    pts = [
        (cx + r * math.cos(a), cy + r * math.sin(a))
        for r, a in zip(radii, angles)
    ]
    pts_int = [(int(x), int(y)) for x, y in pts]
    if len(pts_int) > 2:
        pts_int.append(pts_int[0])  # close the loop
        draw.line(pts_int, fill=(*_f3_to_u8(accent), 220), width=3)

    # Inner solid circle
    draw.ellipse(
        [cx - base_r + 4, cy - base_r + 4, cx + base_r - 4, cy + base_r - 4],
        fill=(*_f3_to_u8(primary), 40),
        outline=(*_f3_to_u8(accent), 180),
        width=2,
    )

    return np.array(pil, dtype=np.float32) / 255.0


def _draw_progress_bar(
    frame: np.ndarray,
    progress: float,
    t: float,
    accent: tuple,
    primary: tuple,
    w: int,
    h: int,
) -> np.ndarray:
    """Minimal seek bar at the bottom of the frame."""
    bar_y = int(h * 0.92)
    bar_x0, bar_x1 = int(w * 0.04), int(w * 0.96)
    bar_h = max(2, h // 240)

    pil = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)

    # Track
    draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + bar_h], fill=(*_f3_to_u8(primary), 80))
    # Progress fill
    fill_x = int(bar_x0 + (bar_x1 - bar_x0) * progress)
    draw.rectangle([bar_x0, bar_y, fill_x, bar_y + bar_h], fill=(*_f3_to_u8(accent), 220))
    # Playhead dot
    draw.ellipse(
        [fill_x - bar_h * 2, bar_y - bar_h, fill_x + bar_h * 2, bar_y + bar_h * 2],
        fill=(*_f3_to_u8(accent), 255),
    )

    # Timestamp
    font = load_font(size=max(12, h // 48))
    mins, secs = divmod(int(t), 60)
    draw.text((bar_x0, bar_y + bar_h + 6), f"{mins}:{secs:02d}", font=font, fill=(*_f3_to_u8(primary), 160))

    return np.array(pil, dtype=np.float32) / 255.0


def _draw_text(
    frame: np.ndarray,
    artist: str,
    album: str,
    direction: CreativeDirection,
    w: int,
    h: int,
) -> np.ndarray:
    pil = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)

    accent = hex_to_rgba(direction.palette_accent, alpha=240)
    white = (240, 240, 240, 200)
    margin = int(w * 0.035)

    artist_font = load_font(size=max(18, h // 22), bold=True)
    album_font = load_font(size=max(14, h // 32))

    draw.text((margin, int(h * 0.08)), artist, font=artist_font, fill=accent)
    draw.text((margin, int(h * 0.08) + h // 18), album, font=album_font, fill=white)

    return np.array(pil, dtype=np.float32) / 255.0


# ── Art panel helper ───────────────────────────────────────────────────────────

def _composite_panel(frame: np.ndarray, panel: np.ndarray, w: int, h: int) -> np.ndarray:
    ph, pw = panel.shape[:2]
    x0 = int(w * 0.67)
    y0 = int(h * 0.10)
    x1 = min(w, x0 + pw)
    y1 = min(h, y0 + ph)
    pw_clip = x1 - x0
    ph_clip = y1 - y0

    if pw_clip > 0 and ph_clip > 0:
        alpha = panel[:ph_clip, :pw_clip, 3:4]
        frame[y0:y1, x0:x1, :3] = (
            frame[y0:y1, x0:x1, :3] * (1 - alpha) + panel[:ph_clip, :pw_clip, :3] * alpha
        )
        frame[y0:y1, x0:x1, 3] = np.maximum(frame[y0:y1, x0:x1, 3], panel[:ph_clip, :pw_clip, 3])

    return frame


def _prepare_art_panel(path: Path, w: int, h: int) -> np.ndarray:
    """Load album cover, scale to ~28% frame width (square crop), return float32 RGBA."""
    img = Image.open(str(path)).convert("RGBA")
    target_w = int(w * 0.28)
    img = img.resize((target_w, target_w), Image.LANCZOS)  # square panel
    return np.array(img, dtype=np.float32) / 255.0


# ── Video encoding ─────────────────────────────────────────────────────────────

def _encode_with_audio(frames_dir: Path, audio_path: Path, output_path: Path, fps: int) -> None:
    """Encode frames to H.264 and mux with source audio."""
    import shutil as _shutil
    ffmpeg_bin = _shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`."
            )
    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "320k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


# ── Small utilities ────────────────────────────────────────────────────────────

def _hex_to_float3(hex_colour: str) -> tuple[float, float, float]:
    r, g, b, _ = hex_to_rgba(hex_colour)
    return (r / 255, g / 255, b / 255)


def _f3_to_u8(colour: tuple) -> tuple[int, int, int]:
    return tuple(int(c * 255) for c in colour[:3])  # type: ignore


def _blend_colours(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))  # type: ignore


def _resample_curve(curve: list[float], target_len: int) -> list[float]:
    if not curve:
        return [0.0] * target_len
    src = np.array(curve, dtype=np.float32)
    indices = np.linspace(0, len(src) - 1, target_len)
    return np.interp(indices, np.arange(len(src)), src).tolist()


def _resample_frames(frames: list[list[float]], target_n: int) -> list[list[float]]:
    """Resample a list of spectrum frames to target_n frames."""
    if not frames:
        return [[0.0] * 64] * target_n
    n_src = len(frames)
    result = []
    for i in range(target_n):
        src_idx = i * (n_src - 1) / max(target_n - 1, 1)
        lo = int(src_idx)
        hi = min(lo + 1, n_src - 1)
        t = src_idx - lo
        interp = [frames[lo][j] * (1 - t) + frames[hi][j] * t for j in range(len(frames[0]))]
        result.append(interp)
    return result

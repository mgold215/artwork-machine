"""
YouTube Visualizer engine.

Produces a full HD (1920×1080, 60fps) video synchronized to the input audio.
Frames are piped directly to ffmpeg — no temp disk writes, no disk space issues.

Visual layout
─────────────
┌──────────────────────────────────────────────────────────────────────────┐
│  Artist Name                                       │                     │
│  Album Title                                       │  Album Art          │
│                                                    │  (right panel)      │
│  ──────────────────  waveform  ──────────────────  │                     │
│                                                    │                     │
│  ▁▂▃▅▇▆▄  spectrum bars (64 bands)  ▄▃▂▁          │                     │
│                                                    │                     │
│  ●  ───────────────────────────────  progress bar  │                     │
└──────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from artwork_machine.ai.prompt_generator import CreativeDirection
from artwork_machine.audio.analyzer import AudioFeatures
from artwork_machine.video.effects import ParticleSystem, bloom, colour_grade, vignette
from artwork_machine.artwork.utils import hex_to_rgba, load_font

WIDTH  = 1920
HEIGHT = 1080
FPS    = 60


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
    Render the YouTube visualizer and mux with source audio.

    Frames are streamed directly to ffmpeg via pipe — no temp files written,
    no disk space required beyond the final output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quality_divisor = 2 if draft else 1
    w, h = WIDTH // quality_divisor, HEIGHT // quality_divisor
    fps  = 24 if draft else FPS

    bg_colour      = _hex_to_f3(direction.palette_background)
    primary_colour = _hex_to_f3(direction.palette_primary)
    accent_colour  = _hex_to_f3(direction.palette_accent)
    secondary_colour = _hex_to_f3(direction.palette_secondary)

    art_panel  = _prepare_art_panel(album_art_path, w, h)
    particles  = ParticleSystem(w, h, n_particles=300 if not draft else 80)

    total_frames  = fps * int(features.duration)
    spectrum_data = _resample_frames(features.spectrum_frames, total_frames)
    waveform_data = _resample_curve(features.waveform_normalised, total_frames * (w // 4))
    rms_data      = _resample_curve(features.rms_curve, total_frames)
    style         = direction.visualiser_style.lower()

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
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k",
        "-shortest", "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for fi in range(total_frames):
            t        = fi / fps
            energy   = rms_data[fi] if fi < len(rms_data) else 0.5
            spectrum = np.array(spectrum_data[fi]) if fi < len(spectrum_data) else np.zeros(64)
            wv_start = fi * (w // 4)
            wv_end   = wv_start + w
            waveform = np.array(waveform_data[wv_start:wv_end]) if wv_end <= len(waveform_data) else np.zeros(w)

            frame = _render_frame(
                t=t, fi=fi, energy=energy, spectrum=spectrum, waveform=waveform,
                art_panel=art_panel, particles=particles, style=style,
                direction=direction, features=features, artist=artist, album=album,
                bg_colour=bg_colour, primary_colour=primary_colour,
                accent_colour=accent_colour, secondary_colour=secondary_colour,
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
    *, t, fi, energy, spectrum, waveform, art_panel, particles, style,
    direction, features, artist, album, bg_colour, primary_colour,
    accent_colour, secondary_colour, w, h, draft,
) -> np.ndarray:

    frame = np.zeros((h, w, 4), dtype=np.float32)
    frame[:, :, :3] = bg_colour
    frame[:, :, 3]  = 1.0

    frame = _composite_panel(frame, art_panel, w, h)

    vis_w = int(w * 0.65)

    if "spectrum" in style or "bar" in style:
        frame = _draw_spectrum_bars(frame, spectrum, energy, accent_colour, secondary_colour, vis_w, h)
        frame = _draw_waveform(frame, waveform, energy, primary_colour, vis_w, h)
    elif "radial" in style:
        frame = _draw_radial_waveform(frame, waveform, spectrum, energy, accent_colour, primary_colour, vis_w, h)
    else:
        particles.step(t, energy)
        frame = particles.render(frame, accent_colour)
        frame = _draw_spectrum_bars(frame, spectrum, energy, accent_colour, secondary_colour, vis_w, h)

    frame = _draw_progress_bar(frame, t / features.duration, t, accent_colour, primary_colour, w, h)
    frame = _draw_text(frame, artist, album, direction, w, h)

    if not draft:
        frame = bloom(frame, radius=8, strength=0.28)
        frame = colour_grade(
            frame,
            shadows=tuple(c * 0.1 for c in primary_colour),  # type: ignore
            highlights=(1.03, 1.01, 1.0),
        )

    frame = vignette(frame, strength=0.40)
    return frame


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_spectrum_bars(frame, spectrum, energy, accent, secondary, vis_w, h):
    n_bars     = len(spectrum)
    bar_area_h = int(h * 0.38)
    bar_area_y = int(h * 0.58)
    bar_margin = 4
    bar_w      = max(1, (vis_w - bar_margin * (n_bars + 1)) // n_bars)

    pil  = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)

    for i, val in enumerate(spectrum):
        bh  = int(val * bar_area_h * (0.7 + energy * 0.6))
        bx  = bar_margin + i * (bar_w + bar_margin)
        by  = bar_area_y + bar_area_h - bh
        by2 = bar_area_y + bar_area_h
        blend = i / n_bars
        colour = _blend(accent, secondary, blend)
        draw.rectangle([bx, by, bx + bar_w, by2], fill=(*_u8(colour), 200))
        if bh > 4:
            draw.rectangle([bx, by, bx + bar_w, by + 3], fill=(*_u8(accent), 255))

    return np.array(pil, dtype=np.float32) / 255.0


def _draw_waveform(frame, waveform, energy, colour, vis_w, h):
    if not len(waveform):
        return frame
    pil  = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)
    cy        = int(h * 0.42)
    amplitude = int(h * 0.08 * (0.5 + energy))
    xs   = np.linspace(20, vis_w - 20, min(len(waveform), vis_w - 40)).astype(int)
    wv   = np.interp(np.linspace(0, len(waveform) - 1, len(xs)), np.arange(len(waveform)), waveform)
    pts  = [(int(xs[i]), int(cy + wv[i] * amplitude)) for i in range(len(xs))]
    if len(pts) > 1:
        draw.line(pts, fill=(*_u8(colour), 180), width=2)
    return np.array(pil, dtype=np.float32) / 255.0


def _draw_radial_waveform(frame, waveform, spectrum, energy, accent, primary, vis_w, h):
    pil  = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)
    cx, cy  = vis_w // 2, h // 2
    base_r  = min(vis_w, h) // 5
    n_pts   = 360
    angles  = np.linspace(0, math.tau, n_pts, endpoint=False)
    wv_vals = np.interp(np.linspace(0, max(len(waveform)-1,1), n_pts), np.arange(len(waveform)), waveform) if len(waveform) > 1 else np.zeros(n_pts)
    sp_vals = np.interp(np.linspace(0, len(spectrum)-1, n_pts), np.arange(len(spectrum)), spectrum)
    radii   = base_r + wv_vals * base_r * 0.5 * (0.5 + energy) + sp_vals * base_r * 0.3
    pts     = [(int(cx + r * math.cos(a)), int(cy + r * math.sin(a))) for r, a in zip(radii, angles)]
    pts.append(pts[0])
    if len(pts) > 2:
        draw.line(pts, fill=(*_u8(accent), 220), width=3)
    draw.ellipse([cx-base_r+4, cy-base_r+4, cx+base_r-4, cy+base_r-4], fill=(*_u8(primary), 40), outline=(*_u8(accent), 180), width=2)
    return np.array(pil, dtype=np.float32) / 255.0


def _draw_progress_bar(frame, progress, t, accent, primary, w, h):
    bar_y  = int(h * 0.92)
    bx0, bx1 = int(w * 0.04), int(w * 0.96)
    bh     = max(2, h // 240)
    pil    = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw   = ImageDraw.Draw(pil)
    draw.rectangle([bx0, bar_y, bx1, bar_y + bh], fill=(*_u8(primary), 80))
    fill_x = int(bx0 + (bx1 - bx0) * progress)
    draw.rectangle([bx0, bar_y, fill_x, bar_y + bh], fill=(*_u8(accent), 220))
    draw.ellipse([fill_x - bh*2, bar_y - bh, fill_x + bh*2, bar_y + bh*2], fill=(*_u8(accent), 255))
    font = load_font(size=max(12, h // 48))
    mins, secs = divmod(int(t), 60)
    draw.text((bx0, bar_y + bh + 6), f"{mins}:{secs:02d}", font=font, fill=(*_u8(primary), 160))
    return np.array(pil, dtype=np.float32) / 255.0


def _draw_text(frame, artist, album, direction, w, h):
    pil    = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw   = ImageDraw.Draw(pil)
    accent = hex_to_rgba(direction.palette_accent, alpha=240)
    white  = (240, 240, 240, 200)
    margin = int(w * 0.035)
    draw.text((margin, int(h * 0.08)),              artist, font=load_font(size=max(18, h // 22), bold=True), fill=accent)
    draw.text((margin, int(h * 0.08) + h // 18),   album,  font=load_font(size=max(14, h // 32)),            fill=white)
    return np.array(pil, dtype=np.float32) / 255.0


# ── Panel helpers ──────────────────────────────────────────────────────────────

def _prepare_art_panel(path: Path, w: int, h: int) -> np.ndarray:
    img = Image.open(str(path)).convert("RGBA")
    target_w = int(w * 0.30)
    img = img.resize((target_w, target_w), Image.LANCZOS)   # square
    return np.array(img, dtype=np.float32) / 255.0


def _composite_panel(frame: np.ndarray, panel: np.ndarray, w: int, h: int) -> np.ndarray:
    ph, pw = panel.shape[:2]
    x0, y0 = int(w * 0.67), int(h * 0.10)
    x1, y1 = min(w, x0 + pw), min(h, y0 + ph)
    pw_c, ph_c = x1 - x0, y1 - y0
    if pw_c > 0 and ph_c > 0:
        alpha = panel[:ph_c, :pw_c, 3:4]
        frame[y0:y1, x0:x1, :3] = frame[y0:y1, x0:x1, :3] * (1-alpha) + panel[:ph_c, :pw_c, :3] * alpha
    return frame


# ── Utilities ──────────────────────────────────────────────────────────────────

def _hex_to_f3(h: str) -> tuple:
    r, g, b, _ = hex_to_rgba(h)
    return (r/255, g/255, b/255)


def _u8(c: tuple) -> tuple:
    return tuple(int(x * 255) for x in c[:3])  # type: ignore


def _blend(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(a[i] * (1-t) + b[i] * t for i in range(3))  # type: ignore


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

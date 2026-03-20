"""
Spotify Canvas video generator.

Produces a seamlessly looping 9:16 video (720×1280, 30fps, 8 seconds).
Spotify's canvas spec: MP4 H.264, max 8 s, 720×1280, ≤ 200 MB.

Visual structure
----------------
Layer 0 (bottom): AI-generated atmospheric background image
Layer 1:          Slow Ken-Burns pan/zoom on background
Layer 2:          Ambient particle system (colour-matched to palette)
Layer 3:          Rotating cassette reel overlaid lower-centre
Layer 4:          Bloom + colour-grade pass
Layer 5 (top):    Subtle scanlines + vignette

The loop is made seamless by easing the Ken-Burns motion in/out and using
a sine-based particle phase that completes a full period in exactly 8 s.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from artwork_machine.ai.prompt_generator import CreativeDirection
from artwork_machine.audio.analyzer import AudioFeatures
from artwork_machine.video.effects import (
    ParticleSystem,
    bloom,
    chromatic_aberration,
    colour_grade,
    scanlines,
    vignette,
    reel_angle,
)
from artwork_machine.artwork.utils import hex_to_rgba, load_font


WIDTH = 720
HEIGHT = 1280
FPS = 30
DURATION = 8          # seconds
N_FRAMES = FPS * DURATION


def generate(
    bg_image_path: Path,
    cassette_art_path: Path,
    direction: CreativeDirection,
    features: AudioFeatures,
    artist: str,
    album: str,
    output_path: Path,
    *,
    draft: bool = False,
) -> Path:
    """
    Render the Spotify Canvas video.

    Parameters
    ----------
    bg_image_path:
        720×1280 AI-generated atmospheric background.
    cassette_art_path:
        3000×3000 cassette art (will be cropped and scaled).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quality_divisor = 2 if draft else 1
    w, h = WIDTH // quality_divisor, HEIGHT // quality_divisor
    fps = 15 if draft else FPS

    # ── Pre-load and prepare images ───────────────────────────────────────────
    bg = _prepare_background(bg_image_path, w, h)
    cassette_thumb = _prepare_cassette_thumb(cassette_art_path, w, direction)

    # ── Particle system ────────────────────────────────────────────────────────
    accent = _hex_to_float3(direction.palette_accent)
    particles = ParticleSystem(w, h, n_particles=150 if not draft else 40)

    # ── RMS energy curve (resampled to number of video frames) ────────────────
    energy_curve = _resample_curve(features.rms_curve, fps * DURATION)

    # ── Render frames to temp directory then encode ────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n_frames = fps * DURATION

        for fi in range(n_frames):
            t = fi / fps
            energy = energy_curve[fi] if fi < len(energy_curve) else 0.5

            frame = _render_frame(
                t=t,
                energy=energy,
                bg=bg,
                cassette_thumb=cassette_thumb,
                particles=particles,
                direction=direction,
                features=features,
                artist=artist,
                album=album,
                accent=accent,
                w=w,
                h=h,
                draft=draft,
            )

            # Save frame as PNG
            frame_img = Image.fromarray((frame * 255).astype(np.uint8), "RGBA")
            frame_img.convert("RGB").save(str(tmp_path / f"frame_{fi:05d}.png"))

        _encode_video(tmp_path, output_path, fps)

    return output_path


# ── Frame renderer ─────────────────────────────────────────────────────────────

def _render_frame(
    *,
    t: float,
    energy: float,
    bg: np.ndarray,
    cassette_thumb: np.ndarray,
    particles: ParticleSystem,
    direction: CreativeDirection,
    features: AudioFeatures,
    artist: str,
    album: str,
    accent: tuple,
    w: int,
    h: int,
    draft: bool,
) -> np.ndarray:
    """Compose all layers for a single frame. Returns H×W×4 float32 [0,1]."""

    # Layer 0: Ken-Burns pan/zoom on background
    frame = _ken_burns(bg, t, total_duration=DURATION)   # H×W×4

    # Layer 1: particles
    particles.step(t, energy)
    frame = particles.render(frame, accent)

    # Layer 2: cassette thumbnail (lower centre, rotating reel inside)
    frame = _composite_cassette(frame, cassette_thumb, t, features.bpm, w, h, energy)

    # Layer 3: text overlay (artist / album at top)
    frame = _draw_text_overlay(frame, artist, album, direction, w, h, t)

    # Layer 4: post-processing
    if not draft:
        frame = bloom(frame, radius=6, strength=0.35)
        frame = colour_grade(
            frame,
            shadows=_hex_to_float3(direction.palette_primary, scale=0.15),
            highlights=(1.05, 1.02, 1.0),
        )
        frame = chromatic_aberration(frame, shift=1.5 + energy * 1.5)

    frame = vignette(frame, strength=0.6)
    frame = scanlines(frame, alpha=0.05)

    return np.clip(frame, 0, 1)


# ── Layer helpers ──────────────────────────────────────────────────────────────

def _ken_burns(bg: np.ndarray, t: float, total_duration: float) -> np.ndarray:
    """
    Slow zoom + pan that completes a perfect loop.
    Uses a sine wave so start == end frame.
    """
    h, w = bg.shape[:2]
    # Zoom oscillates between 1.0 and 1.08
    zoom = 1.0 + 0.04 * (1 - math.cos(math.tau * t / total_duration))
    # Pan oscillates left↔right
    pan_x = 0.015 * math.sin(math.tau * t / total_duration)

    new_w = int(w / zoom)
    new_h = int(h / zoom)
    cx = w // 2 + int(pan_x * w)
    cy = h // 2

    x0 = max(0, cx - new_w // 2)
    y0 = max(0, cy - new_h // 2)
    x1 = min(w, x0 + new_w)
    y1 = min(h, y0 + new_h)

    crop = bg[y0:y1, x0:x1, :]
    # Resize back to full frame
    pil = Image.fromarray((crop * 255).astype(np.uint8), "RGBA")
    pil = pil.resize((w, h), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def _composite_cassette(
    frame: np.ndarray,
    cassette: np.ndarray,
    t: float,
    bpm: float,
    w: int,
    h: int,
    energy: float,
) -> np.ndarray:
    """Place the cassette art in the lower portion of the frame."""
    ch, cw = cassette.shape[:2]
    x0 = (w - cw) // 2
    y0 = int(h * 0.55)  # start 55% down
    y1 = y0 + ch
    x1 = x0 + cw

    # Subtle breathing scale driven by energy
    scale = 1.0 + 0.015 * math.sin(math.tau * bpm / 60 * t)

    cass_pil = Image.fromarray((cassette * 255).astype(np.uint8), "RGBA")
    scaled_w = int(cw * scale)
    scaled_h = int(ch * scale)
    cass_pil = cass_pil.resize((scaled_w, scaled_h), Image.BILINEAR)

    sx = (scaled_w - cw) // 2
    sy = (scaled_h - ch) // 2
    cass_pil = cass_pil.crop((sx, sy, sx + cw, sy + ch))
    cass_arr = np.array(cass_pil, dtype=np.float32) / 255.0

    # Clamp to frame bounds
    y0c = max(0, y0)
    y1c = min(h, y1)
    x0c = max(0, x0)
    x1c = min(w, x1)

    if y1c > y0c and x1c > x0c:
        alpha = cass_arr[:y1c - y0c, :x1c - x0c, 3:4]
        frame[y0c:y1c, x0c:x1c, :3] = (
            frame[y0c:y1c, x0c:x1c, :3] * (1 - alpha)
            + cass_arr[:y1c - y0c, :x1c - x0c, :3] * alpha
        )
        frame[y0c:y1c, x0c:x1c, 3] = np.maximum(
            frame[y0c:y1c, x0c:x1c, 3],
            cass_arr[:y1c - y0c, :x1c - x0c, 3],
        )

    return frame


def _draw_text_overlay(
    frame: np.ndarray,
    artist: str,
    album: str,
    direction: CreativeDirection,
    w: int,
    h: int,
    t: float,
) -> np.ndarray:
    """Render artist + album title as floating text at the top of the frame."""
    pil = Image.fromarray((frame * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)

    # Fade in during first 0.5 s, fade out during last 0.5 s
    alpha_mult = min(1.0, t / 0.5) * min(1.0, (DURATION - t) / 0.5)
    text_alpha = int(220 * alpha_mult)

    accent = hex_to_rgba(direction.palette_accent, alpha=text_alpha)
    white = (255, 255, 255, text_alpha)

    font_artist = load_font(size=max(20, w // 12), bold=True)
    font_album = load_font(size=max(14, w // 18))

    margin = w // 20
    draw.text((margin, h // 16), artist, font=font_artist, fill=accent)
    draw.text((margin, h // 16 + w // 10), album, font=font_album, fill=white)

    return np.array(pil, dtype=np.float32) / 255.0


# ── Encode ─────────────────────────────────────────────────────────────────────

def _encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    """Use ffmpeg to encode PNG frames into an H.264 MP4."""
    import subprocess

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


# ── Image prep ─────────────────────────────────────────────────────────────────

def _prepare_background(path: Path, w: int, h: int) -> np.ndarray:
    """Load background image, ensure RGBA, resize to (w, h), return float32."""
    img = Image.open(str(path)).convert("RGBA")
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _prepare_cassette_thumb(path: Path, w: int, direction: CreativeDirection) -> np.ndarray:
    """
    Load cassette art and scale it to fit ~60% of frame width.
    Returns RGBA float32 array.
    """
    img = Image.open(str(path)).convert("RGBA")
    target_w = int(w * 0.60)
    ratio = target_w / img.width
    target_h = int(img.height * ratio)
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _hex_to_float3(hex_colour: str, scale: float = 1.0) -> tuple[float, float, float]:
    r, g, b, _ = hex_to_rgba(hex_colour)
    return (r / 255 * scale, g / 255 * scale, b / 255 * scale)


def _resample_curve(curve: list[float], target_len: int) -> list[float]:
    """Linearly resample a 1-D curve to a different length."""
    if not curve:
        return [0.5] * target_len
    src = np.array(curve, dtype=np.float32)
    indices = np.linspace(0, len(src) - 1, target_len)
    return np.interp(indices, np.arange(len(src)), src).tolist()

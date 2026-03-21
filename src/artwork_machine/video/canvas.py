"""
Spotify Canvas video generator.

Produces a seamlessly looping 9:16 video (720×1280, 30fps, 8 seconds).
Spotify spec: MP4 H.264, max 8s, 720×1280, ≤ 200 MB.

Visual structure
----------------
Layer 0: AI-generated 9:16 background image (Ken-Burns pan/zoom)
Layer 1: Ambient particle system (colour-matched to palette)
Layer 2: Album art overlay (centred, subtle breathing pulse)
Layer 3: Bloom + colour-grade
Layer 4: Vignette + scanlines
Layer 5: Artist / album text (fade in/out)
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

from artwork_machine.ai.prompt_generator import CreativeDirection
from artwork_machine.audio.analyzer import AudioFeatures
from artwork_machine.video.effects import (
    ParticleSystem,
    bloom,
    chromatic_aberration,
    colour_grade,
    scanlines,
    vignette,
)
from artwork_machine.artwork.utils import hex_to_rgba, load_font


WIDTH    = 720
HEIGHT   = 1280
FPS      = 30
DURATION = 8   # seconds


# ── Ken-Burns camera ──────────────────────────────────────────────────────────

class CameraState(NamedTuple):
    pan_x: float
    pan_y: float
    zoom:  float


def _motion_camera(t: float, style: str, energy: float) -> CameraState:
    """Compute Ken-Burns camera state at time t (all functions period = DURATION)."""
    phase = math.tau * t / DURATION
    s = style.lower()

    if "liquid" in s or "bloom" in s:
        pan_x = 0.04 * math.sin(phase)
        pan_y = 0.025 * math.cos(phase)
        zoom  = 1.0 + 0.05 * (1 - math.cos(phase))

    elif "glitch" in s or "pulse" in s:
        kick  = 0.03 * max(0.0, math.sin(phase * 4) ** 3)
        sign  = 1 if math.sin(phase * 2) > 0 else -1
        pan_x = 0.02 * math.sin(phase) + kick * sign
        pan_y = 0.015 * math.cos(phase)
        zoom  = 1.03 + 0.04 * energy * abs(math.sin(phase * 4))

    elif "drift" in s or "cinematic" in s or "reveal" in s:
        pan_x = 0.05 * math.sin(phase * 0.5)
        pan_y = 0.01 * math.sin(phase)
        zoom  = 1.02 + 0.03 * (1 - math.cos(phase))

    elif "swirl" in s or "hypnotic" in s or "fluid" in s:
        pan_x = 0.05 * math.cos(phase)
        pan_y = 0.03 * math.sin(phase)
        zoom  = 1.03 + 0.02 * math.cos(phase * 2)

    elif "surge" in s or "electric" in s or "parallax" in s:
        pan_x = 0.06 * math.sin(phase * 2)
        pan_y = 0.04 * math.cos(phase * 2)
        zoom  = 1.0 + 0.06 * energy + 0.02 * math.cos(phase)

    else:
        pan_x = 0.03 * math.sin(phase)
        pan_y = 0.02 * math.cos(phase)
        zoom  = 1.0 + 0.04 * (1 - math.cos(phase))

    return CameraState(pan_x=pan_x, pan_y=pan_y, zoom=zoom)


def _apply_ken_burns(
    layer: np.ndarray, cam: CameraState, out_w: int, out_h: int
) -> np.ndarray:
    """Crop + resize layer using camera state to produce out_h×out_w viewport."""
    lh, lw = layer.shape[:2]

    vp_w = max(1, int(out_w / cam.zoom))
    vp_h = max(1, int(out_h / cam.zoom))

    cx = lw / 2 + cam.pan_x * lw
    cy = lh / 2 + cam.pan_y * lh

    x0 = int(cx - vp_w / 2)
    y0 = int(cy - vp_h / 2)
    x0 = max(0, min(lw - vp_w, x0))
    y0 = max(0, min(lh - vp_h, y0))

    crop = layer[y0 : y0 + vp_h, x0 : x0 + vp_w, :]
    pil  = Image.fromarray((crop * 255).astype(np.uint8), "RGBA")
    pil  = pil.resize((out_w, out_h), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


# ── Public generate() ─────────────────────────────────────────────────────────

def generate(
    bg_image_path: Path,
    album_art_path: Path,
    direction: CreativeDirection,
    features: AudioFeatures,
    artist: str,
    album: str,
    output_path: Path,
    *,
    draft: bool = False,
) -> Path:
    """
    Render the Spotify Canvas with Ken-Burns animation.

    Parameters
    ----------
    bg_image_path:
        9:16 background image (810×1440 oversize PNG) from generate_canvas_image().
    album_art_path:
        Square album art to overlay in the centre of the frame.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quality_divisor = 2 if draft else 1
    w, h = WIDTH // quality_divisor, HEIGHT // quality_divisor
    fps  = 15 if draft else FPS

    # Load background
    bg = _load_image(bg_image_path, None, None)   # keep native oversize res

    # Prepare album art overlay (~50% of frame width, centred)
    art_overlay = _prepare_art_overlay(album_art_path, w)

    accent    = _hex_to_float3(direction.palette_accent)
    particles = ParticleSystem(w, h, n_particles=120 if not draft else 30)
    energy_curve = _resample_curve(features.rms_curve, fps * DURATION)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n_frames = fps * DURATION

        for fi in range(n_frames):
            t      = fi / fps
            energy = energy_curve[fi] if fi < len(energy_curve) else 0.5

            cam   = _motion_camera(t, direction.canvas_motion_style, energy)
            frame = _apply_ken_burns(bg, cam, w, h)

            # Particles
            particles.step(t, energy)
            frame = particles.render(frame, accent)

            # Album art overlay (breathing pulse at BPM)
            frame = _composite_art(frame, art_overlay, t, features.bpm, w, h)

            # Text
            frame = _draw_text(frame, artist, album, direction, w, h, t)

            # Post-processing
            if not draft:
                frame = bloom(frame, radius=6, strength=0.30)
                frame = colour_grade(
                    frame,
                    shadows=_hex_to_float3(direction.palette_primary, scale=0.12),
                    highlights=(1.04, 1.01, 1.0),
                )
                frame = chromatic_aberration(frame, shift=1.0)

            frame = vignette(frame, strength=0.55)
            frame = scanlines(frame, alpha=0.04)

            pil = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "RGBA")
            pil.convert("RGB").save(str(tmp_path / f"frame_{fi:05d}.png"))

        _encode_video(tmp_path, output_path, fps)

    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image(path: Path, out_w: int | None, out_h: int | None) -> np.ndarray:
    img = Image.open(str(path)).convert("RGBA")
    if out_w and out_h:
        img = img.resize((out_w, out_h), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _prepare_art_overlay(path: Path, frame_w: int) -> np.ndarray:
    """Scale album art to 50% of frame width, keep square."""
    img = Image.open(str(path)).convert("RGBA")
    target = int(frame_w * 0.50)
    img = img.resize((target, target), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _composite_art(
    frame: np.ndarray,
    art: np.ndarray,
    t: float,
    bpm: float,
    w: int,
    h: int,
) -> np.ndarray:
    """Composite album art centred in the frame with a subtle BPM breathing pulse."""
    ah, aw = art.shape[:2]
    scale  = 1.0 + 0.012 * math.sin(math.tau * bpm / 60 * t)

    art_pil  = Image.fromarray((art * 255).astype(np.uint8), "RGBA")
    scaled_w = int(aw * scale)
    scaled_h = int(ah * scale)
    art_pil  = art_pil.resize((scaled_w, scaled_h), Image.BILINEAR)

    sx = (scaled_w - aw) // 2
    sy = (scaled_h - ah) // 2
    art_pil  = art_pil.crop((sx, sy, sx + aw, sy + ah))
    art_arr  = np.array(art_pil, dtype=np.float32) / 255.0

    # Centre vertically at 45% of frame height
    x0 = (w - aw) // 2
    y0 = int(h * 0.45) - ah // 2

    x0c = max(0, x0);  x1c = min(w, x0 + aw)
    y0c = max(0, y0);  y1c = min(h, y0 + ah)

    if x1c > x0c and y1c > y0c:
        alpha = art_arr[: y1c - y0c, : x1c - x0c, 3:4]
        frame[y0c:y1c, x0c:x1c, :3] = (
            frame[y0c:y1c, x0c:x1c, :3] * (1 - alpha)
            + art_arr[: y1c - y0c, : x1c - x0c, :3] * alpha
        )

    return frame


def _draw_text(
    frame: np.ndarray,
    artist: str,
    album: str,
    direction: CreativeDirection,
    w: int,
    h: int,
    t: float,
) -> np.ndarray:
    pil  = Image.fromarray((frame * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)

    alpha_mult = min(1.0, t / 0.4) * min(1.0, (DURATION - t) / 0.4)
    text_alpha = int(220 * alpha_mult)

    accent = hex_to_rgba(direction.palette_accent, alpha=text_alpha)
    white  = (255, 255, 255, text_alpha)

    margin       = w // 20
    artist_font  = load_font(size=max(18, w // 12), bold=True)
    album_font   = load_font(size=max(13, w // 18))

    draw.text((margin, h // 14), artist, font=artist_font, fill=accent)
    draw.text((margin, h // 14 + w // 10), album, font=album_font, fill=white)

    return np.array(pil, dtype=np.float32) / 255.0


def _encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    import shutil, subprocess
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def _hex_to_float3(hex_colour: str, scale: float = 1.0) -> tuple:
    r, g, b, _ = hex_to_rgba(hex_colour)
    return (r / 255 * scale, g / 255 * scale, b / 255 * scale)


def _resample_curve(curve: list[float], target_len: int) -> list[float]:
    if not curve:
        return [0.5] * target_len
    src     = np.array(curve, dtype=np.float32)
    indices = np.linspace(0, len(src) - 1, target_len)
    return np.interp(indices, np.arange(len(src)), src).tolist()

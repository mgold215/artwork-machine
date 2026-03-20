"""
Spotify Canvas video generator — Drone Parallax Edition.

Produces a seamlessly looping 9:16 video (720×1280, 30fps, 8 seconds).
Spotify's canvas spec: MP4 H.264, max 8 s, 720×1280, ≤ 200 MB.

Visual structure
----------------
Layer 0 (bottom): AI-generated far aerial background   (depth 0.15 — barely moves)
Layer 1:          AI-generated mid cassette scene       (depth 0.50 — moderate parallax)
Layer 2:          AI-generated near macro tape texture  (depth 0.90 — maximum movement)
Layer 3:          Ambient particle system (colour-matched to palette)
Layer 4:          Rotating cassette reel overlaid lower-centre
Layer 5:          Bloom + colour-grade pass
Layer 6 (top):    Subtle scanlines + vignette

DroneCamera
-----------
Simulates real drone footage: altitude-driven zoom, pan + tilt, subtle roll.
Each canvas layer is offset proportionally to its depth factor so nearer
layers travel faster than far ones, creating convincing parallax.

Motion style (from CreativeDirection.canvas_motion_style) selects one of
several pre-baked camera animation presets.  All camera functions use
sin/cos with period = DURATION so the 8-second loop is perfectly seamless.
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


WIDTH = 720
HEIGHT = 1280
FPS = 30
DURATION = 8          # seconds

# Depth factors per layer (controls parallax travel distance)
_DEPTH_FAR  = 0.15   # barely moves — aerial background
_DEPTH_MID  = 0.50   # moderate parallax — overhead cassette scene
_DEPTH_NEAR = 0.90   # maximum movement — macro tape texture

# Alpha blending weights so the aerial background always shows through
_ALPHA_MID  = 0.70
_ALPHA_NEAR = 0.55


# ── DroneCamera ────────────────────────────────────────────────────────────────

class CameraState(NamedTuple):
    pan_x: float   # normalised horizontal offset  [-1, 1]
    pan_y: float   # normalised vertical offset    [-1, 1]
    zoom:  float   # scale factor (1.0 = no zoom)
    roll:  float   # radians of rotation (kept tiny for realism)


def _motion_camera(t: float, style: str, energy: float) -> CameraState:
    """
    Compute drone camera state at time *t*.

    All functions complete exactly one period in DURATION seconds so the
    8-second video loops without a visible seam.
    """
    phase = math.tau * t / DURATION
    s = style.lower()

    if "liquid" in s or "bloom" in s:
        # Slow graceful elliptical orbit + gentle breath zoom
        pan_x = 0.04 * math.sin(phase)
        pan_y = 0.025 * math.cos(phase * 0.7)
        zoom  = 1.0 + 0.05 * (1 - math.cos(phase))
        roll  = 0.008 * math.sin(phase * 0.5)

    elif "glitch" in s or "pulse" in s:
        # Mostly static; sharp stutter kicks at quarter-period marks
        base_x = 0.02 * math.sin(phase)
        kick   = 0.03 * max(0.0, math.sin(phase * 4) ** 3)
        sign   = 1 if math.sin(phase * 2) > 0 else -1
        pan_x  = base_x + kick * sign
        pan_y  = 0.015 * math.cos(phase * 1.3)
        zoom   = 1.03 + 0.04 * energy * abs(math.sin(phase * 4))
        roll   = 0.005 * math.sin(phase * 3)

    elif "drift" in s or "cinematic" in s or "reveal" in s:
        # Slow linear pan that resets smoothly each cycle
        pan_x = 0.05 * math.sin(phase * 0.5)        # half-speed sine → smooth drift
        pan_y = 0.01 * math.sin(phase)
        zoom  = 1.02 + 0.03 * (1 - math.cos(phase))
        roll  = 0.0

    elif "swirl" in s or "hypnotic" in s or "fluid" in s:
        # Circular orbit — classic hypnotic drone loop
        pan_x = 0.05 * math.cos(phase)
        pan_y = 0.03 * math.sin(phase)
        zoom  = 1.03 + 0.02 * math.cos(phase * 2)
        roll  = 0.015 * math.sin(phase)

    elif "surge" in s or "electric" in s or "parallax" in s:
        # Fast oscillation; zoom reacts to audio energy
        pan_x = 0.06 * math.sin(phase * 2)
        pan_y = 0.04 * math.cos(phase * 1.5)
        zoom  = 1.0 + 0.06 * energy + 0.02 * math.cos(phase)
        roll  = 0.01 * math.sin(phase * 2)

    else:
        # Generic gentle drift
        pan_x = 0.03 * math.sin(phase)
        pan_y = 0.02 * math.cos(phase * 0.8)
        zoom  = 1.0 + 0.04 * (1 - math.cos(phase))
        roll  = 0.0

    return CameraState(pan_x=pan_x, pan_y=pan_y, zoom=zoom, roll=roll)


def _sample_layer(
    layer: np.ndarray,
    cam: CameraState,
    depth: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    Extract an (out_h × out_w) viewport from *layer* using the camera state
    scaled by *depth*.

    *layer* is the oversize AI image (810×1440) in float32 RGBA.
    Nearer layers (higher depth) travel farther, producing parallax.
    """
    lh, lw = layer.shape[:2]

    # Scale all camera motions by depth factor
    px   = cam.pan_x * depth
    py   = cam.pan_y * depth
    zoom = 1.0 + (cam.zoom - 1.0) * depth

    # Viewport size inside the layer
    vp_w = max(1, int(out_w / zoom))
    vp_h = max(1, int(out_h / zoom))

    # Layer centre + offset
    cx = lw / 2 + px * lw
    cy = lh / 2 + py * lh

    x0 = int(cx - vp_w / 2)
    y0 = int(cy - vp_h / 2)

    # Clamp — never reveal the edge of the oversize image
    x0 = max(0, min(lw - vp_w, x0))
    y0 = max(0, min(lh - vp_h, y0))

    crop = layer[y0 : y0 + vp_h, x0 : x0 + vp_w, :]
    pil  = Image.fromarray((crop * 255).astype(np.uint8), "RGBA")
    pil  = pil.resize((out_w, out_h), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def _composite_layers(
    far: np.ndarray,
    mid: np.ndarray,
    near: np.ndarray,
) -> np.ndarray:
    """
    Alpha-composite three RGBA float32 layers bottom → top.
    Far layer is the opaque base; mid and near are blended on top.
    """
    def over(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
        a = fg[:, :, 3:4]
        out = np.empty_like(bg)
        out[:, :, :3] = fg[:, :, :3] * a + bg[:, :, :3] * (1.0 - a)
        out[:, :, 3]  = np.maximum(bg[:, :, 3], fg[:, :, 3])
        return out

    return over(over(far, mid), near)


# ── Public generate() ─────────────────────────────────────────────────────────

def generate(
    canvas_layers: dict[str, Path] | None,
    cassette_art_path: Path,
    direction: CreativeDirection,
    features: AudioFeatures,
    artist: str,
    album: str,
    output_path: Path,
    *,
    draft: bool = False,
    # Legacy fallback for callers that still pass a single background image
    bg_image_path: Path | None = None,
) -> Path:
    """
    Render the Spotify Canvas video with drone parallax movement.

    Parameters
    ----------
    canvas_layers:
        Dict ``{"far": Path, "mid": Path, "near": Path}`` from
        :func:`image_generator.generate_canvas_layers`.  Each image is
        810×1440 (oversize 9:16) so the drone camera has pixel headroom.
    cassette_art_path:
        Cassette artwork composited over the lower portion of the frame.
    bg_image_path:
        Legacy single-background fallback (all three layers will use it).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quality_divisor = 2 if draft else 1
    w, h = WIDTH // quality_divisor, HEIGHT // quality_divisor
    fps  = 15 if draft else FPS

    # ── Load canvas layers ────────────────────────────────────────────────────
    if canvas_layers:
        far_layer  = _load_layer(canvas_layers["far"],  w, h, oversample=not draft)
        mid_layer  = _load_layer(canvas_layers["mid"],  w, h, oversample=not draft)
        near_layer = _load_layer(canvas_layers["near"], w, h, oversample=not draft)
    elif bg_image_path:
        bg = _prepare_background(bg_image_path, w, h)
        far_layer = mid_layer = near_layer = bg
    else:
        raise ValueError("Provide canvas_layers or bg_image_path")

    # ── Other assets ──────────────────────────────────────────────────────────
    cassette_thumb = _prepare_cassette_thumb(cassette_art_path, w)
    accent         = _hex_to_float3(direction.palette_accent)
    particles      = ParticleSystem(w, h, n_particles=150 if not draft else 40)
    energy_curve   = _resample_curve(features.rms_curve, fps * DURATION)

    # ── Render frames ─────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        n_frames = fps * DURATION

        for fi in range(n_frames):
            t      = fi / fps
            energy = energy_curve[fi] if fi < len(energy_curve) else 0.5

            frame = _render_frame(
                t=t,
                energy=energy,
                far_layer=far_layer,
                mid_layer=mid_layer,
                near_layer=near_layer,
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

            frame_img = Image.fromarray((frame * 255).astype(np.uint8), "RGBA")
            frame_img.convert("RGB").save(str(tmp_path / f"frame_{fi:05d}.png"))

        _encode_video(tmp_path, output_path, fps)

    return output_path


# ── Frame renderer ─────────────────────────────────────────────────────────────

def _render_frame(
    *,
    t: float,
    energy: float,
    far_layer: np.ndarray,
    mid_layer: np.ndarray,
    near_layer: np.ndarray,
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

    # ── Drone camera ──────────────────────────────────────────────────────────
    cam = _motion_camera(t, direction.canvas_motion_style, energy)

    # ── Sample each layer at its parallax depth ───────────────────────────────
    far_view  = _sample_layer(far_layer,  cam, _DEPTH_FAR,  w, h)
    mid_view  = _sample_layer(mid_layer,  cam, _DEPTH_MID,  w, h)
    near_view = _sample_layer(near_layer, cam, _DEPTH_NEAR, w, h)

    # Reduce alpha on mid + near so the aerial base always shows through
    mid_view[:, :, 3]  *= _ALPHA_MID
    near_view[:, :, 3] *= _ALPHA_NEAR

    # ── Blend depth layers ────────────────────────────────────────────────────
    frame = _composite_layers(far_view, mid_view, near_view)

    # ── Particles ─────────────────────────────────────────────────────────────
    particles.step(t, energy)
    frame = particles.render(frame, accent)

    # ── Cassette overlay ──────────────────────────────────────────────────────
    frame = _composite_cassette(frame, cassette_thumb, t, features.bpm, w, h, energy)

    # ── Text overlay ──────────────────────────────────────────────────────────
    frame = _draw_text_overlay(frame, artist, album, direction, w, h, t)

    # ── Post-processing ───────────────────────────────────────────────────────
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

def _load_layer(path: Path, out_w: int, out_h: int, *, oversample: bool) -> np.ndarray:
    """
    Load a canvas layer PNG.

    In full-quality mode we keep the image at its native oversize resolution
    (810×1440) so the DroneCamera has pixel headroom to travel.
    In draft mode we scale down to the output size (acceptable for previews).
    """
    img = Image.open(str(path)).convert("RGBA")
    if not oversample:
        img = img.resize((out_w, out_h), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _composite_cassette(
    frame: np.ndarray,
    cassette: np.ndarray,
    t: float,
    bpm: float,
    w: int,
    h: int,
    energy: float,
) -> np.ndarray:
    """Place the cassette art in the lower portion of the frame with a subtle breathing scale."""
    ch, cw = cassette.shape[:2]
    x0 = (w - cw) // 2
    y0 = int(h * 0.55)
    y1 = y0 + ch
    x1 = x0 + cw

    scale    = 1.0 + 0.015 * math.sin(math.tau * bpm / 60 * t)
    cass_pil = Image.fromarray((cassette * 255).astype(np.uint8), "RGBA")
    scaled_w = int(cw * scale)
    scaled_h = int(ch * scale)
    cass_pil = cass_pil.resize((scaled_w, scaled_h), Image.BILINEAR)

    sx = (scaled_w - cw) // 2
    sy = (scaled_h - ch) // 2
    cass_pil = cass_pil.crop((sx, sy, sx + cw, sy + ch))
    cass_arr = np.array(cass_pil, dtype=np.float32) / 255.0

    y0c = max(0, y0)
    y1c = min(h, y1)
    x0c = max(0, x0)
    x1c = min(w, x1)

    if y1c > y0c and x1c > x0c:
        alpha = cass_arr[: y1c - y0c, : x1c - x0c, 3:4]
        frame[y0c:y1c, x0c:x1c, :3] = (
            frame[y0c:y1c, x0c:x1c, :3] * (1 - alpha)
            + cass_arr[: y1c - y0c, : x1c - x0c, :3] * alpha
        )
        frame[y0c:y1c, x0c:x1c, 3] = np.maximum(
            frame[y0c:y1c, x0c:x1c, 3],
            cass_arr[: y1c - y0c, : x1c - x0c, 3],
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
    """Render artist + album title floating at the top; fades in/out."""
    pil  = Image.fromarray((frame * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(pil)

    alpha_mult = min(1.0, t / 0.5) * min(1.0, (DURATION - t) / 0.5)
    text_alpha = int(220 * alpha_mult)

    accent = hex_to_rgba(direction.palette_accent, alpha=text_alpha)
    white  = (255, 255, 255, text_alpha)

    font_artist = load_font(size=max(20, w // 12), bold=True)
    font_album  = load_font(size=max(14, w // 18))

    margin = w // 20
    draw.text((margin, h // 16), artist, font=font_artist, fill=accent)
    draw.text((margin, h // 16 + w // 10), album, font=font_album, fill=white)

    return np.array(pil, dtype=np.float32) / 255.0


# ── Video encode ───────────────────────────────────────────────────────────────

def _encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    """Encode PNG frames to H.264 MP4 via ffmpeg."""
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
    """Legacy: load a single background image as a flat layer."""
    img = Image.open(str(path)).convert("RGBA")
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _prepare_cassette_thumb(path: Path, w: int) -> np.ndarray:
    """Scale cassette art to ~60% of frame width."""
    img      = Image.open(str(path)).convert("RGBA")
    target_w = int(w * 0.60)
    target_h = int(img.height * target_w / img.width)
    img      = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _hex_to_float3(hex_colour: str, scale: float = 1.0) -> tuple[float, float, float]:
    r, g, b, _ = hex_to_rgba(hex_colour)
    return (r / 255 * scale, g / 255 * scale, b / 255 * scale)


def _resample_curve(curve: list[float], target_len: int) -> list[float]:
    if not curve:
        return [0.5] * target_len
    src     = np.array(curve, dtype=np.float32)
    indices = np.linspace(0, len(src) - 1, target_len)
    return np.interp(indices, np.arange(len(src)), src).tolist()

"""
Procedural image generator — no external image API required.

Generates label art and canvas parallax layers entirely with PIL + NumPy,
driven by the colour palette and style descriptors in CreativeDirection.

Label art  : abstract layered composition at 1024×1024 (or 512×512 draft)
Canvas layers: three depth layers at 810×1440 (or 405×720 draft), each with a
               distinct visual register (far / mid / near).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

from artwork_machine.ai.prompt_generator import CreativeDirection
from artwork_machine.artwork.utils import add_grain, hex_to_rgba


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_to_f32(hex_colour: str) -> np.ndarray:
    """Return a (3,) float32 array in [0, 1] from a CSS hex string."""
    r, g, b, _ = hex_to_rgba(hex_colour)
    return np.array([r, g, b], dtype=np.float32) / 255.0


def _make_canvas(h: int, w: int, colour: np.ndarray) -> np.ndarray:
    """Solid-fill float32 RGBA canvas (H×W×4)."""
    c = np.ones((h, w, 4), dtype=np.float32)
    c[:, :, :3] = colour
    return c


def _blend(base: np.ndarray, over: np.ndarray, opacity: float) -> np.ndarray:
    """Alpha-composite 'over' onto 'base' at the given opacity."""
    a = over[:, :, 3:4] * opacity
    return base * (1 - a) + over * a


def _radial_gradient(h: int, w: int) -> np.ndarray:
    """Return a (H×W) float32 gradient: 1.0 at centre, 0.0 at corners."""
    cy, cx = h / 2, w / 2
    y = np.linspace(0, h - 1, h, dtype=np.float32)
    x = np.linspace(0, w - 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(1.0 - dist / (max(h, w) * 0.7), 0, 1)


def _linear_gradient(h: int, w: int, angle_deg: float = 135.0) -> np.ndarray:
    """Return a (H×W) float32 linear gradient in [0, 1]."""
    rad = np.deg2rad(angle_deg)
    y = np.linspace(-1, 1, h, dtype=np.float32)
    x = np.linspace(-1, 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    proj = xx * np.cos(rad) + yy * np.sin(rad)
    return (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)


def _plasma_noise(h: int, w: int, scale: float = 0.3, seed: int = 0) -> np.ndarray:
    """Smooth turbulent noise field in [0, 1] via layered gaussian smoothing."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)
    smooth = gaussian_filter(noise, sigma=max(h, w) * scale)
    lo, hi = smooth.min(), smooth.max()
    return (smooth - lo) / (hi - lo + 1e-8)


def _vignette(h: int, w: int, strength: float = 0.6) -> np.ndarray:
    """Return a (H×W) vignette mask: 1.0 centre → (1-strength) edges."""
    r = _radial_gradient(h, w)
    return 1.0 - strength * (1.0 - r)


def _colour_layer(
    h: int, w: int,
    col: np.ndarray,
    mask: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Build an RGBA layer (H×W×4): colour filled, mask as alpha."""
    layer = np.zeros((h, w, 4), dtype=np.float32)
    layer[:, :, :3] = col
    layer[:, :, 3] = mask * alpha
    return layer


def _apply_grain(arr_f32: np.ndarray, intensity: float = 0.025, seed: int = 42) -> np.ndarray:
    """Add subtle luminance grain to a float32 H×W×3 or H×W×4 array."""
    rng = np.random.default_rng(seed)
    h, w = arr_f32.shape[:2]
    g = rng.standard_normal((h, w)).astype(np.float32) * intensity
    g = gaussian_filter(g, sigma=0.5)
    out = arr_f32.copy()
    out[:, :, :3] = np.clip(out[:, :, :3] + g[:, :, np.newaxis], 0, 1)
    return out


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGBA")


# ── Label art ────────────────────────────────────────────────────────────────

def generate_label_art(
    direction: CreativeDirection,
    output_path: Path,
    model: str = "local",          # ignored, kept for API compatibility
    *,
    draft: bool = False,
) -> Path:
    """
    Generate abstract label art at 1024×1024 (512×512 in draft mode).

    Layers (bottom → top):
      0  Background fill — palette_background
      1  Radial plasma bloom — palette_primary
      2  Diagonal gradient wash — palette_secondary
      3  Accent corona — palette_accent
      4  Dark vignette frame
      5  Film grain
    """
    size = 512 if draft else 1024
    h = w = size
    rng_seed = hash(direction.cassette_brand_name) & 0xFFFF

    bg  = _hex_to_f32(direction.palette_background)
    pri = _hex_to_f32(direction.palette_primary)
    sec = _hex_to_f32(direction.palette_secondary)
    acc = _hex_to_f32(direction.palette_accent)

    # Layer 0 — background
    canvas = _make_canvas(h, w, bg)

    # Layer 1 — plasma bloom in primary colour
    plasma = _plasma_noise(h, w, scale=0.25, seed=rng_seed)
    radial = _radial_gradient(h, w)
    mask1  = np.clip(plasma * radial * 1.8, 0, 1)
    canvas = _blend(canvas, _colour_layer(h, w, pri, mask1, 0.9), 1.0)

    # Layer 2 — secondary diagonal wash
    grad = _linear_gradient(h, w, angle_deg=135)
    plasma2 = _plasma_noise(h, w, scale=0.15, seed=rng_seed + 1)
    mask2 = np.clip(grad * plasma2 * 1.5, 0, 1)
    canvas = _blend(canvas, _colour_layer(h, w, sec, mask2, 0.6), 1.0)

    # Layer 3 — accent corona
    plasma3 = _plasma_noise(h, w, scale=0.08, seed=rng_seed + 2)
    mask3 = np.clip((plasma3 - 0.55) * 3, 0, 1)
    canvas = _blend(canvas, _colour_layer(h, w, acc, mask3, 0.55), 1.0)

    # Layer 4 — vignette
    vig = _vignette(h, w, strength=0.55)
    canvas[:, :, :3] *= vig[:, :, np.newaxis]

    # Layer 5 — grain
    canvas = _apply_grain(canvas, intensity=0.018, seed=rng_seed + 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = _to_pil(canvas).convert("RGB")
    img.save(str(output_path), "PNG", dpi=(300, 300))
    return output_path


# ── Canvas parallax layers ────────────────────────────────────────────────────

def generate_canvas_layers(
    direction: CreativeDirection,
    work_dir: Path,
    model: str = "local",
    *,
    draft: bool = False,
) -> dict[str, Path]:
    """
    Generate three depth layers for the drone parallax compositor.

    far  — Aerial-style: dark field of swirling plasma, minimal detail.
    mid  — Overhead: brighter gradient with cassette-tape ribbon traces.
    near — Macro: extreme close-up turbulence, high contrast.
    """
    lw, lh = (405, 720) if draft else (810, 1440)
    work_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "far":  work_dir / "canvas_layer_far.png",
        "mid":  work_dir / "canvas_layer_mid.png",
        "near": work_dir / "canvas_layer_near.png",
    }

    seed = hash(direction.cassette_brand_name) & 0xFFFF
    bg  = _hex_to_f32(direction.palette_background)
    pri = _hex_to_f32(direction.palette_primary)
    sec = _hex_to_f32(direction.palette_secondary)
    acc = _hex_to_f32(direction.palette_accent)

    # ── Far layer: aerial dark field ──────────────────────────────────────────
    far = _make_canvas(lh, lw, bg * 0.3)  # very dark
    p1  = _plasma_noise(lh, lw, scale=0.35, seed=seed)
    p2  = _plasma_noise(lh, lw, scale=0.15, seed=seed + 10)
    far = _blend(far, _colour_layer(lh, lw, pri * 0.6, p1 * p2 * 2.5, 0.8), 1.0)
    far = _blend(far, _colour_layer(lh, lw, acc, np.clip((p2 - 0.7) * 4, 0, 1), 0.4), 1.0)
    far[:, :, :3] *= _vignette(lh, lw, 0.7)[:, :, np.newaxis]
    far = _apply_grain(far, 0.012, seed)
    _to_pil(far).convert("RGB").save(str(paths["far"]), "PNG")

    # ── Mid layer: overhead cassette ribbon scene ──────────────────────────────
    mid = _make_canvas(lh, lw, bg * 0.6 + pri * 0.4)
    p3  = _plasma_noise(lh, lw, scale=0.20, seed=seed + 20)
    grad = _linear_gradient(lh, lw, angle_deg=70)
    mid = _blend(mid, _colour_layer(lh, lw, sec, grad * p3 * 1.8, 0.75), 1.0)
    # Tape ribbon traces: narrow horizontal plasma streaks
    ribbon_mask = np.clip(np.sin(np.linspace(0, 30, lh))[:, np.newaxis]
                          * p3 * 2, 0, 1).astype(np.float32)
    mid = _blend(mid, _colour_layer(lh, lw, pri, ribbon_mask, 0.5), 1.0)
    mid = _blend(mid, _colour_layer(lh, lw, acc, np.clip((p3 - 0.6) * 3, 0, 1), 0.35), 1.0)
    mid[:, :, :3] *= _vignette(lh, lw, 0.45)[:, :, np.newaxis]
    mid = _apply_grain(mid, 0.014, seed + 1)
    _to_pil(mid).convert("RGB").save(str(paths["mid"]), "PNG")

    # ── Near layer: extreme macro tape texture ─────────────────────────────────
    near = _make_canvas(lh, lw, bg * 0.15)
    # High-frequency micro-texture
    p4 = _plasma_noise(lh, lw, scale=0.06, seed=seed + 30)
    p5 = _plasma_noise(lh, lw, scale=0.18, seed=seed + 31)
    near = _blend(near, _colour_layer(lh, lw, pri, p4 * p5 * 3.0, 0.9), 1.0)
    near = _blend(near, _colour_layer(lh, lw, acc, np.clip((p4 - 0.4) * 4, 0, 1), 0.65), 1.0)
    # Central hot-spot (bokeh simulation)
    spot = _radial_gradient(lh, lw) ** 2
    near = _blend(near, _colour_layer(lh, lw, sec, spot * 0.6, 0.4), 1.0)
    near[:, :, :3] *= _vignette(lh, lw, 0.8)[:, :, np.newaxis]
    near = _apply_grain(near, 0.030, seed + 2)  # more grain = tactile feel
    _to_pil(near).convert("RGB").save(str(paths["near"]), "PNG")

    return paths

"""
Album art generator using Replicate Flux 1.1 Pro.

Sends the creative direction prompt to Replicate's Flux 1.1 Pro model,
downloads the result, and upscales it to a print-ready 3000x3000 PNG.

In draft mode a lighter procedural fallback is used so you can iterate
without burning API credits.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import replicate
from PIL import Image
from scipy.ndimage import gaussian_filter
from tenacity import retry, stop_after_attempt, wait_exponential

from artwork_machine.ai.prompt_generator import CreativeDirection
from artwork_machine.artwork.utils import hex_to_rgba


# ── Replicate / Flux ──────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
def generate_album_art(
    direction: CreativeDirection,
    output_path: Path,
    *,
    draft: bool = False,
) -> Path:
    """
    Generate a 3000x3000 album cover PNG.

    In production mode calls Replicate Flux 1.1 Pro at 1024x1024 and upscales.
    In draft mode uses the fast procedural fallback (no API call).

    Parameters
    ----------
    direction:
        Creative direction produced by :func:`artwork_machine.ai.prompt_generator.generate`.
    output_path:
        Where to save the final 3000x3000 PNG.
    draft:
        If True, skip Replicate and use fast procedural generation instead.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_path = output_path.parent / "album_cover_1024.png"

    if draft:
        return _generate_procedural(direction, output_path, work_path, size=512)

    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        # No token — fall back to procedural so the pipeline still works
        return _generate_procedural(direction, output_path, work_path, size=1024)

    # ── Call Replicate Flux 1.1 Pro ───────────────────────────────────────────
    output = replicate.run(
        "black-forest-labs/flux-1.1-pro",
        input={
            "prompt": direction.album_art_prompt,
            "negative_prompt": direction.negative_prompt,
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "output_format": "png",
            "output_quality": 100,
        },
    )

    # replicate SDK >= 0.25 returns a FileOutput; cast to str to get the URL
    url = str(output)
    response = httpx.get(url, timeout=120.0, follow_redirects=True)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGB")

    # Save 1024px work copy
    img.save(str(work_path), "PNG")

    # Upscale to 3000x3000 (300 DPI print-ready)
    final = img.resize((3000, 3000), Image.LANCZOS)
    final = _apply_grain(final, intensity=0.012)
    final.save(str(output_path), "PNG", dpi=(300, 300))

    return output_path


# ── Procedural fallback (draft mode / no API token) ───────────────────────────

def _hex_to_f32(hex_colour: str) -> np.ndarray:
    r, g, b, _ = hex_to_rgba(hex_colour)
    return np.array([r, g, b], dtype=np.float32) / 255.0


def _plasma_noise(h: int, w: int, scale: float = 0.25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)
    smooth = gaussian_filter(noise, sigma=max(h, w) * scale)
    lo, hi = smooth.min(), smooth.max()
    return (smooth - lo) / (hi - lo + 1e-8)


def _radial_gradient(h: int, w: int) -> np.ndarray:
    cy, cx = h / 2, w / 2
    y = np.linspace(0, h - 1, h, dtype=np.float32)
    x = np.linspace(0, w - 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(1.0 - dist / (max(h, w) * 0.7), 0, 1)


def _linear_gradient(h: int, w: int, angle_deg: float = 135.0) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    y = np.linspace(-1, 1, h, dtype=np.float32)
    x = np.linspace(-1, 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    proj = xx * np.cos(rad) + yy * np.sin(rad)
    return (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)


def _generate_procedural(
    direction: CreativeDirection,
    output_path: Path,
    work_path: Path,
    size: int = 1024,
) -> Path:
    """Fast palette-driven abstract image — no API required."""
    h = w = size
    seed = hash(direction.tagline) & 0xFFFF

    bg  = _hex_to_f32(direction.palette_background)
    pri = _hex_to_f32(direction.palette_primary)
    sec = _hex_to_f32(direction.palette_secondary)
    acc = _hex_to_f32(direction.palette_accent)

    canvas = np.ones((h, w, 3), dtype=np.float32) * bg

    # Radial plasma bloom (primary)
    plasma1 = _plasma_noise(h, w, scale=0.25, seed=seed)
    radial  = _radial_gradient(h, w)
    mask1   = np.clip(plasma1 * radial * 1.8, 0, 1)[:, :, np.newaxis]
    canvas  = canvas * (1 - mask1 * 0.9) + pri * (mask1 * 0.9)

    # Diagonal wash (secondary)
    grad   = _linear_gradient(h, w, angle_deg=135)
    plasma2 = _plasma_noise(h, w, scale=0.15, seed=seed + 1)
    mask2  = np.clip(grad * plasma2 * 1.5, 0, 1)[:, :, np.newaxis]
    canvas  = canvas * (1 - mask2 * 0.6) + sec * (mask2 * 0.6)

    # Accent corona
    plasma3 = _plasma_noise(h, w, scale=0.08, seed=seed + 2)
    mask3   = np.clip((plasma3 - 0.55) * 3, 0, 1)[:, :, np.newaxis]
    canvas  = canvas * (1 - mask3 * 0.55) + acc * (mask3 * 0.55)

    # Vignette
    vig    = _radial_gradient(h, w)
    canvas *= (0.45 + 0.55 * vig)[:, :, np.newaxis]

    img_1024 = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGB")
    img_1024.save(str(work_path), "PNG")

    final = img_1024.resize((3000, 3000), Image.LANCZOS)
    final = _apply_grain(final, intensity=0.015)
    final.save(str(output_path), "PNG", dpi=(300, 300))
    return output_path


def _apply_grain(img: Image.Image, intensity: float = 0.012) -> Image.Image:
    """Add subtle luminance film grain to a PIL RGB image."""
    arr = np.array(img, dtype=np.float32) / 255.0
    rng = np.random.default_rng(42)
    h, w = arr.shape[:2]
    grain = rng.standard_normal((h, w)).astype(np.float32) * intensity
    grain = gaussian_filter(grain, sigma=0.5)
    arr = np.clip(arr + grain[:, :, np.newaxis], 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")

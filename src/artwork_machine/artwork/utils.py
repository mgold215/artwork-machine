"""
Shared image-processing utilities for the artwork module.
"""

from __future__ import annotations

import importlib.resources
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont


# ── Colour helpers ─────────────────────────────────────────────────────────────

def hex_to_rgba(hex_colour: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Convert a CSS hex colour (with or without #) to an RGBA tuple."""
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


# ── Font loader ────────────────────────────────────────────────────────────────

_FONT_CACHE: dict[tuple, ImageFont.FreeTypeFont] = {}
_FALLBACK_FONT_PATHS = [
    # Common system font paths
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",                    # macOS
    "C:/Windows/Fonts/arial.ttf",                             # Windows
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a system font at the given size, falling back to PIL default."""
    cache_key = (size, bold)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    for path in _FALLBACK_FONT_PATHS:
        if "Bold" in path and not bold:
            continue
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[cache_key] = font
                return font
            except OSError:
                continue

    # Absolute fallback — PIL's built-in bitmap font (no size control)
    font = ImageFont.load_default()
    _FONT_CACHE[cache_key] = font  # type: ignore
    return font  # type: ignore


# ── Film-grain / texture helpers ───────────────────────────────────────────────

def add_noise(img: Image.Image, intensity: float = 0.03) -> Image.Image:
    """Add subtle luminance noise to give the image an analogue feel."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape[:2])
    if arr.ndim == 3:
        # Apply to all channels but not alpha
        for c in range(min(3, arr.shape[2])):
            arr[:, :, c] = np.clip(arr[:, :, c] + noise, 0, 255)
    result = Image.fromarray(arr.astype(np.uint8), mode=img.mode)
    return result


def add_grain(img: Image.Image, intensity: float = 0.04) -> Image.Image:
    """
    Add photographic film grain using high-frequency Gaussian noise.
    Keeps a more filmic character than simple uniform noise.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    grain = np.random.normal(0, 1, (h, w)).astype(np.float32)

    # High-pass: subtract a blurred version (simulates grain not affecting large areas)
    from scipy.ndimage import gaussian_filter
    low = gaussian_filter(grain, sigma=2)
    grain = (grain - low) * intensity * 255

    for c in range(min(3, arr.shape[2])):
        arr[:, :, c] = np.clip(arr[:, :, c] + grain, 0, 255)

    return Image.fromarray(arr.astype(np.uint8), mode=img.mode)

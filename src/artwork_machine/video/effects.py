"""
Reusable visual effects for both the Canvas and Visualiser video engines.

All functions operate on numpy float32 RGBA arrays (H×W×4) normalised to [0, 1].
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter


# ── Particle system ────────────────────────────────────────────────────────────

class ParticleSystem:
    """
    A lightweight CPU particle system for ambient floating particles.

    Particles drift upward with slight Brownian motion, wrap around edges,
    and pulse in opacity with the music energy level.
    """

    def __init__(
        self,
        width: int,
        height: int,
        n_particles: int = 200,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.w = width
        self.h = height

        # Particle state arrays (all float32, shape [N])
        self.x = rng.uniform(0, width, n_particles).astype(np.float32)
        self.y = rng.uniform(0, height, n_particles).astype(np.float32)
        self.vx = rng.normal(0, 0.3, n_particles).astype(np.float32)
        self.vy = rng.uniform(-0.8, -0.2, n_particles).astype(np.float32)   # drift up
        self.radius = rng.uniform(1.5, 5.0, n_particles).astype(np.float32)
        self.alpha = rng.uniform(0.1, 0.6, n_particles).astype(np.float32)
        self.phase = rng.uniform(0, math.tau, n_particles).astype(np.float32)

    def step(self, t: float, energy: float = 0.5) -> None:
        """Advance the simulation by one frame."""
        # Brownian jitter scales with energy
        jitter = 0.5 + energy * 1.5
        self.vx += np.random.normal(0, jitter * 0.15, len(self.x)).astype(np.float32)
        self.vy += np.random.normal(0, jitter * 0.08, len(self.y)).astype(np.float32)

        # Terminal velocity
        self.vx = np.clip(self.vx, -2, 2)
        self.vy = np.clip(self.vy, -3, 0.5)

        self.x = (self.x + self.vx) % self.w
        self.y = (self.y + self.vy) % self.h

        # Pulse alpha with time and energy
        self.alpha = 0.15 + 0.45 * (0.5 + 0.5 * np.sin(t * 2 + self.phase)) * (0.5 + energy * 0.5)

    def render(self, frame: np.ndarray, colour: tuple[float, float, float]) -> np.ndarray:
        """
        Render particles as soft glowing circles onto a float32 RGBA frame (in-place).

        Parameters
        ----------
        frame:
            H×W×4 float32 array, values in [0, 1].
        colour:
            RGB tuple in [0, 1] used for all particles.
        """
        h, w = frame.shape[:2]
        r, g, b = colour

        for i in range(len(self.x)):
            px, py = int(self.x[i]), int(self.y[i])
            rad = max(1, int(self.radius[i]))
            alpha = float(self.alpha[i])

            # Bounding box
            x0, x1 = max(0, px - rad * 2), min(w, px + rad * 2 + 1)
            y0, y1 = max(0, py - rad * 2), min(h, py + rad * 2 + 1)

            if x0 >= x1 or y0 >= y1:
                continue

            # Gaussian soft dot
            xs = np.arange(x0, x1) - px
            ys = np.arange(y0, y1) - py
            xx, yy = np.meshgrid(xs, ys)
            dist2 = xx**2 + yy**2
            blob = np.exp(-dist2 / (2 * (rad * 0.7) ** 2)) * alpha

            frame[y0:y1, x0:x1, 0] = np.clip(frame[y0:y1, x0:x1, 0] + r * blob, 0, 1)
            frame[y0:y1, x0:x1, 1] = np.clip(frame[y0:y1, x0:x1, 1] + g * blob, 0, 1)
            frame[y0:y1, x0:x1, 2] = np.clip(frame[y0:y1, x0:x1, 2] + b * blob, 0, 1)
            frame[y0:y1, x0:x1, 3] = np.clip(frame[y0:y1, x0:x1, 3] + blob * 0.6, 0, 1)

        return frame


# ── Bloom / glow ───────────────────────────────────────────────────────────────

def bloom(frame: np.ndarray, radius: float = 8.0, strength: float = 0.4) -> np.ndarray:
    """
    Screen-blend a blurred copy of the frame to simulate lens bloom.

    Parameters
    ----------
    frame:
        H×W×4 float32 array in [0, 1].
    """
    blurred = gaussian_filter(frame[:, :, :3], sigma=radius)
    # Screen blend:  1 - (1-a)(1-b)
    rgb = frame[:, :, :3]
    screen = 1.0 - (1.0 - rgb) * (1.0 - blurred * strength)
    result = frame.copy()
    result[:, :, :3] = np.clip(screen, 0, 1)
    return result


# ── Chromatic aberration ───────────────────────────────────────────────────────

def chromatic_aberration(frame: np.ndarray, shift: float = 2.0) -> np.ndarray:
    """Shift the R and B channels slightly apart for a lo-fi VHS look."""
    result = frame.copy()
    s = int(round(shift))
    if s <= 0:
        return result
    h, w = frame.shape[:2]
    # Shift red channel left
    result[:, :w - s, 0] = frame[:, s:, 0]
    # Shift blue channel right
    result[:, s:, 2] = frame[:, :w - s, 2]
    return result


# ── VHS scanline overlay ───────────────────────────────────────────────────────

def scanlines(frame: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    """Overlay alternating dark horizontal lines (CRT / VHS aesthetic)."""
    result = frame.copy()
    h = frame.shape[0]
    mask = np.zeros(h, dtype=np.float32)
    mask[::2] = alpha
    result[:, :, :3] *= (1.0 - mask[:, None, None])
    return result


# ── Colour grading ─────────────────────────────────────────────────────────────

def colour_grade(
    frame: np.ndarray,
    shadows: tuple[float, float, float] = (0.0, 0.0, 0.0),
    midtones: tuple[float, float, float] = (1.0, 1.0, 1.0),
    highlights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Basic three-way colour grade (shadows / midtones / highlights lift + gain).
    """
    result = frame.copy()
    rgb = result[:, :, :3]

    # Shadows: additive lift in dark areas
    shadow_mask = 1.0 - rgb  # bright = 0, dark = 1
    for c, s in enumerate(shadows):
        rgb[:, :, c] += shadow_mask[:, :, c] * s

    # Highlights: multiplicative gain in bright areas
    for c, h in enumerate(highlights):
        rgb[:, :, c] = rgb[:, :, c] ** (1.0 / max(h, 0.01))

    result[:, :, :3] = np.clip(rgb, 0, 1)
    return result


# ── Vignette ───────────────────────────────────────────────────────────────────

def vignette(frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply a radial vignette (darkening at edges)."""
    h, w = frame.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = 1.0 - np.clip(dist * strength, 0, 1)
    result = frame.copy()
    result[:, :, :3] *= mask[:, :, None]
    return result


# ── Cassette reel animation helper ────────────────────────────────────────────

def reel_angle(t: float, bpm: float) -> float:
    """Return the rotation angle (degrees) for an animated reel at time t."""
    rps = bpm / 60.0 * 0.5   # reels spin at half the music tempo visually
    return (t * rps * 360.0) % 360.0

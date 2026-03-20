"""Tests for the video effects module."""

from __future__ import annotations

import numpy as np
import pytest

from artwork_machine.video.effects import (
    ParticleSystem,
    bloom,
    chromatic_aberration,
    colour_grade,
    scanlines,
    vignette,
    reel_angle,
)


@pytest.fixture
def blank_frame():
    """128×128 RGBA float32 frame, all zeros (black)."""
    return np.zeros((128, 128, 4), dtype=np.float32)


@pytest.fixture
def white_frame():
    return np.ones((128, 128, 4), dtype=np.float32)


class TestParticleSystem:
    def test_init_creates_particles(self):
        ps = ParticleSystem(320, 240, n_particles=50)
        assert len(ps.x) == 50
        assert len(ps.y) == 50

    def test_step_does_not_crash(self):
        ps = ParticleSystem(320, 240, n_particles=20)
        ps.step(0.5, energy=0.7)

    def test_render_modifies_frame(self, blank_frame):
        ps = ParticleSystem(128, 128, n_particles=20)
        result = ps.render(blank_frame.copy(), colour=(1.0, 0.5, 0.0))
        assert not np.allclose(result, blank_frame)

    def test_render_stays_in_range(self, blank_frame):
        ps = ParticleSystem(128, 128, n_particles=30)
        result = ps.render(blank_frame.copy(), colour=(1.0, 1.0, 1.0))
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6


class TestBloom:
    def test_output_shape_unchanged(self, white_frame):
        result = bloom(white_frame)
        assert result.shape == white_frame.shape

    def test_black_frame_stays_black(self, blank_frame):
        result = bloom(blank_frame)
        assert np.allclose(result[:, :, :3], 0.0, atol=1e-5)

    def test_values_in_range(self, white_frame):
        result = bloom(white_frame)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6


class TestChromaticAberration:
    def test_output_shape_unchanged(self, white_frame):
        result = chromatic_aberration(white_frame, shift=3.0)
        assert result.shape == white_frame.shape

    def test_zero_shift_is_identity(self, white_frame):
        result = chromatic_aberration(white_frame, shift=0.0)
        assert np.allclose(result, white_frame)


class TestColourGrade:
    def test_output_shape_unchanged(self, white_frame):
        result = colour_grade(white_frame)
        assert result.shape == white_frame.shape

    def test_values_in_range(self, white_frame):
        result = colour_grade(white_frame, shadows=(0.1, 0.0, 0.2))
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6


class TestVignette:
    def test_centre_brighter_than_corner(self, white_frame):
        result = vignette(white_frame, strength=0.8)
        centre = result[64, 64, 0]
        corner = result[0, 0, 0]
        assert centre > corner

    def test_values_in_range(self, white_frame):
        result = vignette(white_frame)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6


class TestScanlines:
    def test_alternating_rows_darker(self, white_frame):
        result = scanlines(white_frame, alpha=0.5)
        # Every even row should be darker
        assert result[0, 0, 0] < result[1, 0, 0]

    def test_output_shape_unchanged(self, white_frame):
        result = scanlines(white_frame)
        assert result.shape == white_frame.shape


class TestReelAngle:
    def test_returns_float(self):
        assert isinstance(reel_angle(1.0, 120.0), float)

    def test_wraps_at_360(self):
        angle = reel_angle(100.0, 120.0)
        assert 0.0 <= angle < 360.0

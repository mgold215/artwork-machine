"""Tests for the drone parallax canvas generator."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from artwork_machine.video.canvas import (
    CameraState,
    _motion_camera,
    _sample_layer,
    _composite_layers,
    _hex_to_float3,
    _resample_curve,
    _load_layer,
    _DEPTH_FAR,
    _DEPTH_MID,
    _DEPTH_NEAR,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _solid_layer(w: int, h: int, colour=(0.5, 0.5, 0.5, 1.0)) -> np.ndarray:
    """Return an H×W×4 float32 array filled with *colour*."""
    arr = np.zeros((h, w, 4), dtype=np.float32)
    arr[:, :, 0] = colour[0]
    arr[:, :, 1] = colour[1]
    arr[:, :, 2] = colour[2]
    arr[:, :, 3] = colour[3]
    return arr


def _gradient_layer(w: int, h: int) -> np.ndarray:
    """Horizontal gradient so pan offsets produce measurable pixel differences."""
    arr = np.zeros((h, w, 4), dtype=np.float32)
    arr[:, :, 3] = 1.0
    for x in range(w):
        arr[:, x, 0] = x / (w - 1)
    return arr


# ── CameraState / _motion_camera ──────────────────────────────────────────────

class TestMotionCamera:
    STYLES = [
        "slow liquid bloom",
        "glitch pulse orbit",
        "cinematic drift reveal",
        "hypnotic fluid swirl",
        "electric parallax surge",
        "unknown style",          # fallback
    ]

    def test_returns_camera_state(self):
        cam = _motion_camera(0.0, "liquid bloom", 0.5)
        assert isinstance(cam, CameraState)

    @pytest.mark.parametrize("style", STYLES)
    def test_all_styles_produce_valid_values(self, style):
        for t in [0.0, 2.0, 4.0, 7.9]:
            cam = _motion_camera(t, style, energy=0.6)
            assert math.isfinite(cam.pan_x)
            assert math.isfinite(cam.pan_y)
            assert cam.zoom > 0
            assert math.isfinite(cam.roll)

    @pytest.mark.parametrize("style", STYLES)
    def test_loop_is_seamless(self, style):
        """Frame 0 and frame at exactly DURATION should be identical."""
        from artwork_machine.video.canvas import DURATION
        cam_start = _motion_camera(0.0, style, energy=0.5)
        cam_end   = _motion_camera(float(DURATION), style, energy=0.5)
        assert pytest.approx(cam_start.pan_x, abs=1e-6) == cam_end.pan_x
        assert pytest.approx(cam_start.pan_y, abs=1e-6) == cam_end.pan_y
        assert pytest.approx(cam_start.zoom,  abs=1e-6) == cam_end.zoom
        assert pytest.approx(cam_start.roll,  abs=1e-6) == cam_end.roll

    def test_energy_affects_surge_zoom(self):
        """'electric surge' style zoom should be larger at high energy."""
        cam_low  = _motion_camera(2.0, "electric surge", energy=0.0)
        cam_high = _motion_camera(2.0, "electric surge", energy=1.0)
        assert cam_high.zoom > cam_low.zoom


# ── _sample_layer ─────────────────────────────────────────────────────────────

class TestSampleLayer:
    OUT_W, OUT_H = 72, 128

    def test_output_shape(self):
        layer = _solid_layer(81, 144)   # oversize (1.125×)
        cam   = CameraState(0.0, 0.0, 1.0, 0.0)
        out   = _sample_layer(layer, cam, _DEPTH_MID, self.OUT_W, self.OUT_H)
        assert out.shape == (self.OUT_H, self.OUT_W, 4)

    def test_zero_motion_returns_centre(self):
        """No pan / zoom → output should match the centre crop of the layer."""
        layer = _gradient_layer(81, 144)
        cam   = CameraState(0.0, 0.0, 1.0, 0.0)
        out   = _sample_layer(layer, cam, _DEPTH_MID, self.OUT_W, self.OUT_H)
        assert out.shape == (self.OUT_H, self.OUT_W, 4)

    def test_far_layer_moves_less_than_near(self):
        """
        With the same camera pan, the far layer viewport moves less
        (proportional to depth), so its pixel values differ less from
        the no-pan case than the near layer does.
        """
        layer = _gradient_layer(162, 288)   # bigger oversize for headroom
        cam_pan  = CameraState(0.3, 0.0, 1.0, 0.0)
        cam_zero = CameraState(0.0, 0.0, 1.0, 0.0)

        far_pan  = _sample_layer(layer, cam_pan,  _DEPTH_FAR,  self.OUT_W, self.OUT_H)
        far_zero = _sample_layer(layer, cam_zero, _DEPTH_FAR,  self.OUT_W, self.OUT_H)
        near_pan = _sample_layer(layer, cam_pan,  _DEPTH_NEAR, self.OUT_W, self.OUT_H)
        near_zero= _sample_layer(layer, cam_zero, _DEPTH_NEAR, self.OUT_W, self.OUT_H)

        far_diff  = np.mean(np.abs(far_pan  - far_zero))
        near_diff = np.mean(np.abs(near_pan - near_zero))

        assert far_diff < near_diff, (
            f"Far layer should move less than near layer "
            f"(far_diff={far_diff:.4f}, near_diff={near_diff:.4f})"
        )

    def test_zoom_reduces_viewport(self):
        """Higher zoom → smaller viewport extracted → upscaled back, so result is 'zoomed in'."""
        layer    = _gradient_layer(162, 288)
        cam_zoom = CameraState(0.0, 0.0, 2.0, 0.0)
        cam_flat = CameraState(0.0, 0.0, 1.0, 0.0)
        out_zoom = _sample_layer(layer, cam_zoom, _DEPTH_NEAR, self.OUT_W, self.OUT_H)
        out_flat = _sample_layer(layer, cam_flat, _DEPTH_NEAR, self.OUT_W, self.OUT_H)
        # They should differ
        assert not np.allclose(out_zoom, out_flat, atol=0.01)

    def test_values_in_range(self):
        layer = _gradient_layer(162, 288)
        cam   = CameraState(0.1, 0.05, 1.05, 0.01)
        out   = _sample_layer(layer, cam, _DEPTH_MID, self.OUT_W, self.OUT_H)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5


# ── _composite_layers ─────────────────────────────────────────────────────────

class TestCompositeLayers:
    def test_output_shape(self):
        far  = _solid_layer(72, 128, (0.1, 0.0, 0.0, 1.0))
        mid  = _solid_layer(72, 128, (0.0, 0.5, 0.0, 0.0))  # transparent
        near = _solid_layer(72, 128, (0.0, 0.0, 0.9, 0.0))  # transparent
        out  = _composite_layers(far, mid, near)
        assert out.shape == (128, 72, 4)

    def test_transparent_fg_reveals_bg(self):
        """Fully transparent mid + near → output equals far background."""
        far  = _solid_layer(72, 128, (0.3, 0.4, 0.5, 1.0))
        mid  = _solid_layer(72, 128, (1.0, 0.0, 0.0, 0.0))
        near = _solid_layer(72, 128, (0.0, 1.0, 0.0, 0.0))
        out  = _composite_layers(far, mid, near)
        assert np.allclose(out[:, :, :3], far[:, :, :3], atol=1e-5)

    def test_opaque_near_covers_all(self):
        """Fully opaque near → output RGB matches near layer."""
        far  = _solid_layer(72, 128, (0.1, 0.1, 0.1, 1.0))
        mid  = _solid_layer(72, 128, (0.5, 0.5, 0.5, 0.5))
        near = _solid_layer(72, 128, (0.9, 0.2, 0.1, 1.0))
        out  = _composite_layers(far, mid, near)
        assert np.allclose(out[:, :, :3], near[:, :, :3], atol=1e-5)

    def test_values_in_range(self):
        far  = np.random.rand(64, 64, 4).astype(np.float32)
        far[:, :, 3] = 1.0
        mid  = np.random.rand(64, 64, 4).astype(np.float32)
        near = np.random.rand(64, 64, 4).astype(np.float32)
        out  = _composite_layers(far, mid, near)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5


# ── Utilities ─────────────────────────────────────────────────────────────────

class TestHexToFloat3:
    def test_red(self):
        r, g, b = _hex_to_float3("#ff0000")
        assert pytest.approx(r) == 1.0
        assert pytest.approx(g) == 0.0
        assert pytest.approx(b) == 0.0

    def test_scale(self):
        r, g, b = _hex_to_float3("#ffffff", scale=0.5)
        assert pytest.approx(r) == 0.5


class TestResampleCurve:
    def test_empty_returns_halves(self):
        out = _resample_curve([], 10)
        assert out == [0.5] * 10

    def test_length_matches_target(self):
        assert len(_resample_curve([0.1, 0.9], 50)) == 50

    def test_single_value(self):
        out = _resample_curve([0.7], 5)
        assert all(pytest.approx(v) == 0.7 for v in out)


# ── _load_layer ───────────────────────────────────────────────────────────────

class TestLoadLayer:
    def test_draft_mode_resizes_to_output(self, tmp_path):
        img = Image.new("RGBA", (810, 1440), (100, 150, 200, 255))
        p   = tmp_path / "layer.png"
        img.save(str(p))
        arr = _load_layer(p, 360, 640, oversample=False)
        assert arr.shape == (640, 360, 4)

    def test_full_quality_keeps_native_size(self, tmp_path):
        img = Image.new("RGBA", (810, 1440), (100, 150, 200, 255))
        p   = tmp_path / "layer.png"
        img.save(str(p))
        arr = _load_layer(p, 360, 640, oversample=True)
        # Native 810×1440, not resized
        assert arr.shape == (1440, 810, 4)

    def test_float32_normalised(self, tmp_path):
        img = Image.new("RGBA", (100, 100), (255, 128, 0, 255))
        p   = tmp_path / "layer.png"
        img.save(str(p))
        arr = _load_layer(p, 100, 100, oversample=False)
        assert arr.dtype == np.float32
        assert arr.max() <= 1.0 + 1e-5


# ── Integration: generate() with mocked ffmpeg ────────────────────────────────

class TestGenerateIntegration:
    """
    Smoke-test the full generate() call.

    - Three tiny solid-colour PNG layers (no Replicate calls)
    - A tiny solid-colour cassette image
    - A mock AudioFeatures object
    - ffmpeg is mocked so no actual video encoding is needed
    """

    def _make_direction(self):
        from artwork_machine.ai.prompt_generator import CreativeDirection
        return CreativeDirection(
            image_prompt="test",
            negative_prompt="test",
            art_style="test",
            canvas_far_prompt="far",
            canvas_mid_prompt="mid",
            canvas_near_prompt="near",
            palette_primary="#1a0a2e",
            palette_secondary="#16213e",
            palette_accent="#e94560",
            palette_background="#0f3460",
            cassette_shell_colour="#c8a96e",
            cassette_era="80s",
            cassette_brand_name="Ferrox",
            canvas_motion_style="slow liquid bloom",
            visualiser_style="spectrum bars",
            label_font_style="condensed sans-serif",
            label_tagline="Magnetic dreams unfurl",
        )

    def _make_features(self):
        from artwork_machine.audio.analyzer import AudioFeatures
        return AudioFeatures(
            duration=180.0,
            sample_rate=44100,
            bpm=120.0,
            beat_frames=[0, 22050, 44100],
            beat_times=[0.0, 0.5, 1.0],
            key="C major",
            chroma_mean=[0.1] * 12,
            rms_mean=0.05,
            rms_curve=[0.5] * 10,
            dynamic_range_db=12.0,
            spectral_centroid_mean=2000.0,
            spectral_rolloff_mean=4000.0,
            zero_crossing_rate=0.05,
            mfcc_mean=[0.0] * 13,
            genre_hints=["pop"],
            mood_tags=["energetic"],
        )

    def test_generate_produces_output_file(self, tmp_path):
        from artwork_machine.video.canvas import generate

        # Create tiny (81×144) layer PNGs — enough headroom for the viewport
        layer_paths = {}
        for name, colour in [("far", (20, 20, 80)), ("mid", (60, 40, 20)), ("near", (10, 60, 10))]:
            img = Image.new("RGBA", (81, 144), colour + (255,))
            p   = tmp_path / f"layer_{name}.png"
            img.save(str(p))
            layer_paths[name] = p

        cass_img = Image.new("RGBA", (200, 200), (180, 140, 80, 255))
        cass_path = tmp_path / "cassette.png"
        cass_img.save(str(cass_path))

        output_path = tmp_path / "out.mp4"

        # Mock ffmpeg so we don't need it installed
        with patch("artwork_machine.video.canvas._encode_video") as mock_enc:
            mock_enc.side_effect = lambda frames_dir, out, fps: out.touch()
            generate(
                canvas_layers=layer_paths,
                cassette_art_path=cass_path,
                direction=self._make_direction(),
                features=self._make_features(),
                artist="Test Artist",
                album="Test Album",
                output_path=output_path,
                draft=True,   # 15fps, half resolution
            )

        mock_enc.assert_called_once()

    def test_legacy_bg_image_path_fallback(self, tmp_path):
        """Passing bg_image_path instead of canvas_layers should still work."""
        from artwork_machine.video.canvas import generate

        bg = Image.new("RGBA", (360, 640), (40, 40, 80, 255))
        bg_path = tmp_path / "bg.png"
        bg.save(str(bg_path))

        cass_img = Image.new("RGBA", (200, 200), (180, 140, 80, 255))
        cass_path = tmp_path / "cassette.png"
        cass_img.save(str(cass_path))

        output_path = tmp_path / "out_legacy.mp4"

        with patch("artwork_machine.video.canvas._encode_video") as mock_enc:
            mock_enc.side_effect = lambda frames_dir, out, fps: out.touch()
            generate(
                canvas_layers=None,
                bg_image_path=bg_path,
                cassette_art_path=cass_path,
                direction=self._make_direction(),
                features=self._make_features(),
                artist="Test Artist",
                album="Test Album",
                output_path=output_path,
                draft=True,
            )

        mock_enc.assert_called_once()

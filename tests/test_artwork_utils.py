"""Tests for artwork utilities."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from artwork_machine.artwork.utils import hex_to_rgba, add_noise, add_grain


class TestHexToRgba:
    def test_standard_hex(self):
        assert hex_to_rgba("#ff0000") == (255, 0, 0, 255)

    def test_lowercase(self):
        assert hex_to_rgba("#00ff88") == (0, 255, 136, 255)

    def test_no_hash(self):
        assert hex_to_rgba("1a0a2e") == (26, 10, 46, 255)

    def test_shorthand(self):
        assert hex_to_rgba("#fff") == (255, 255, 255, 255)

    def test_custom_alpha(self):
        r, g, b, a = hex_to_rgba("#000000", alpha=128)
        assert a == 128

    def test_black(self):
        assert hex_to_rgba("#000000") == (0, 0, 0, 255)


class TestAddNoise:
    def test_output_same_mode(self):
        img = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
        result = add_noise(img, intensity=0.05)
        assert result.mode == "RGBA"

    def test_output_same_size(self):
        img = Image.new("RGB", (100, 100), (50, 100, 150))
        result = add_noise(img)
        assert result.size == img.size

    def test_modifies_pixels(self):
        img = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
        result = add_noise(img, intensity=0.10)
        # At least some pixels should differ from the flat grey
        arr_in = np.array(img)
        arr_out = np.array(result)
        assert not np.all(arr_in[:, :, :3] == arr_out[:, :, :3])


class TestAddGrain:
    def test_output_same_mode(self):
        img = Image.new("RGBA", (64, 64), (200, 180, 160, 255))
        result = add_grain(img)
        assert result.mode == "RGBA"

    def test_values_in_valid_range(self):
        img = Image.new("RGBA", (64, 64), (200, 180, 160, 255))
        result = add_grain(img, intensity=0.1)
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255

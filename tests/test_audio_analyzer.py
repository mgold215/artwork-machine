"""
Tests for the audio analysis engine.

Uses a synthetically generated sine-wave tone so no real audio files are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path

from artwork_machine.audio.analyzer import analyse, _detect_key


def _generate_sine(freq: float = 440.0, duration: float = 5.0, sr: int = 22_050) -> Path:
    """Write a short sine-wave WAV to a temp file and return the path."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    return Path(tmp.name)


@pytest.fixture(scope="module")
def sine_wav():
    path = _generate_sine()
    yield path
    path.unlink(missing_ok=True)


class TestDetectKey:
    def test_returns_string(self):
        chroma = np.random.rand(12)
        key = _detect_key(chroma)
        assert isinstance(key, str)
        assert "major" in key or "minor" in key

    def test_known_c_major(self):
        # C major chord tones: C, E, G (indices 0, 4, 7) boosted
        chroma = np.zeros(12)
        chroma[[0, 4, 7]] = 5.0
        key = _detect_key(chroma)
        assert key == "C major"


class TestAnalyse:
    def test_returns_features(self, sine_wav):
        features = analyse(sine_wav)
        assert features.duration > 0
        assert features.bpm > 0
        assert isinstance(features.key, str)

    def test_bpm_plausible(self, sine_wav):
        features = analyse(sine_wav)
        # A sine wave doesn't have strong beats; librosa will guess some BPM
        assert 30 <= features.bpm <= 300

    def test_rms_positive(self, sine_wav):
        features = analyse(sine_wav)
        assert features.rms_mean > 0

    def test_waveform_not_empty(self, sine_wav):
        features = analyse(sine_wav)
        assert len(features.waveform_normalised) > 0
        assert max(abs(v) for v in features.waveform_normalised) <= 1.0 + 1e-6

    def test_spectrum_frames_not_empty(self, sine_wav):
        features = analyse(sine_wav)
        assert len(features.spectrum_frames) > 0
        assert len(features.spectrum_frames[0]) == 64

    def test_mood_tags_populated(self, sine_wav):
        features = analyse(sine_wav)
        assert len(features.mood_tags) > 0

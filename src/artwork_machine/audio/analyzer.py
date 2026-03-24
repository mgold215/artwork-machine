"""
Audio analysis engine.

Extracts rich perceptual features from an audio file that are used to:
  - Drive creative prompt generation (mood, energy, genre cues)
  - Animate video elements in sync with the music (BPM, beat frames, waveform)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np


@dataclass
class AudioFeatures:
    # ── Temporal ─────────────────────────────────────────────────────────────
    duration: float                   # seconds
    sample_rate: int

    # ── Rhythm ───────────────────────────────────────────────────────────────
    bpm: float
    beat_frames: list[int]            # sample indices of detected beats
    beat_times: list[float]           # seconds of each beat

    # ── Pitch / tonality ─────────────────────────────────────────────────────
    key: str                          # e.g. "C major", "F# minor"
    chroma_mean: list[float]          # 12-bin chroma centroid

    # ── Energy / dynamics ────────────────────────────────────────────────────
    rms_mean: float                   # overall loudness (0–1 normalised)
    rms_curve: list[float]            # per-frame RMS (downsampled to ~10 fps)
    dynamic_range_db: float           # peak minus floor in dB

    # ── Timbre / texture ─────────────────────────────────────────────────────
    spectral_centroid_mean: float     # brightness (Hz)
    spectral_rolloff_mean: float      # high-frequency rolloff (Hz)
    zero_crossing_rate: float         # noisiness / transient density
    mfcc_mean: list[float]            # 13 MFCCs — genre/timbre fingerprint

    # ── Perceptual tags ───────────────────────────────────────────────────────
    mood_tags: list[str] = field(default_factory=list)
    genre_hints: list[str] = field(default_factory=list)

    # ── Waveform for visualiser ───────────────────────────────────────────────
    waveform_normalised: list[float] = field(default_factory=list)   # ~44100 samples/min
    spectrum_frames: list[list[float]] = field(default_factory=list)  # STFT magnitude frames


# ── Key detection ─────────────────────────────────────────────────────────────

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl–Schmuckler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _detect_key(chroma_mean: np.ndarray) -> str:
    """Return the most likely musical key using Krumhansl–Schmuckler profiles."""
    best_key, best_mode, best_r = 0, "major", -np.inf
    for root in range(12):
        rotated = np.roll(chroma_mean, -root)
        r_maj = float(np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1])
        r_min = float(np.corrcoef(rotated, _MINOR_PROFILE)[0, 1])
        if r_maj > best_r:
            best_r, best_key, best_mode = r_maj, root, "major"
        if r_min > best_r:
            best_r, best_key, best_mode = r_min, root, "minor"
    return f"{_NOTE_NAMES[best_key]} {best_mode}"


# ── Mood / genre heuristics ────────────────────────────────────────────────────

def _infer_mood(
    bpm: float,
    rms: float,
    centroid: float,
    zcr: float,
    key: str,
) -> tuple[list[str], list[str]]:
    """Rule-based perceptual tag inference — supplements Claude's creative judgement."""
    mood: list[str] = []
    genre: list[str] = []

    # Energy axis
    if rms > 0.25:
        mood.append("energetic")
    elif rms > 0.10:
        mood.append("moderate energy")
    else:
        mood.append("calm")

    # Tempo axis
    if bpm < 70:
        mood.append("slow")
        genre.append("ambient" if rms < 0.1 else "ballad")
    elif bpm < 100:
        mood.append("relaxed")
    elif bpm < 130:
        mood.append("upbeat")
    elif bpm < 160:
        mood.append("driving")
        genre.append("rock")
    else:
        mood.append("frenetic")
        genre.append("electronic")

    # Brightness axis
    if centroid > 4000:
        mood.append("bright")
        genre.append("pop" if bpm > 100 else "folk")
    elif centroid < 1500:
        mood.append("dark")
        genre.append("metal" if zcr > 0.15 else "jazz")

    # Modality
    if "minor" in key:
        mood.append("melancholic")
    else:
        mood.append("optimistic")

    # Noise / distortion proxy
    if zcr > 0.20:
        genre.append("rock")

    return mood, list(dict.fromkeys(genre))  # deduplicate genres


# ── Public API ────────────────────────────────────────────────────────────────

def analyse(audio_path: Path, *, visualiser_downsample: int = 512) -> AudioFeatures:
    """
    Load and analyse an audio file.  Returns a rich :class:`AudioFeatures` object.

    Parameters
    ----------
    audio_path:
        Path to any format supported by librosa (MP3, WAV, FLAC, AIFF, …).
    visualiser_downsample:
        Hop size used when building the waveform array for the visualiser.
        512 @ 22 050 Hz ≈ 43 fps — plenty for smooth animation.
    """
    y, sr = librosa.load(str(audio_path), sr=22_050, mono=True)

    # ── Rhythm ───────────────────────────────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if float(tempo) >= 30 else 120.0  # fallback for unpitched/silent audio
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # ── Chroma / key ─────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key = _detect_key(chroma_mean)

    # ── Energy ───────────────────────────────────────────────────────────────
    rms_frames = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_mean = float(np.mean(rms_frames))
    rms_max = float(np.max(rms_frames)) or 1.0
    rms_normalised = (rms_frames / rms_max).tolist()

    # Downsample RMS to ~10 fps for lightweight curve storage
    target_fps = 10
    hop_t = 512 / sr  # seconds per frame
    ds = max(1, round(1 / (hop_t * target_fps)))
    rms_curve = rms_normalised[::ds]

    rms_db = librosa.amplitude_to_db(rms_frames, ref=np.max)
    dynamic_range = float(np.max(rms_db) - np.percentile(rms_db, 5))

    # ── Timbre ───────────────────────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # ── Waveform for visualiser ───────────────────────────────────────────────
    waveform = y[::visualiser_downsample]
    waveform_max = float(np.max(np.abs(waveform))) or 1.0
    waveform_normalised = (waveform / waveform_max).tolist()

    # ── Spectrum frames (for frequency visualiser bars) ───────────────────────
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    # Compress to 64 mel-like frequency bins, downsample time to ~24 fps
    mel = librosa.feature.melspectrogram(S=D**2, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalise 0-1
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    time_ds = max(1, round(sr / (512 * 24)))
    spectrum_frames = mel_norm[:, ::time_ds].T.tolist()

    # ── Perceptual tags ───────────────────────────────────────────────────────
    mood_tags, genre_hints = _infer_mood(
        bpm=bpm,
        rms=rms_mean,
        centroid=float(np.mean(centroid)),
        zcr=float(np.mean(zcr)),
        key=key,
    )

    return AudioFeatures(
        duration=float(librosa.get_duration(y=y, sr=sr)),
        sample_rate=sr,
        bpm=bpm,
        beat_frames=beat_frames.tolist(),
        beat_times=beat_times,
        key=key,
        chroma_mean=chroma_mean.tolist(),
        rms_mean=rms_mean,
        rms_curve=rms_curve,
        dynamic_range_db=dynamic_range,
        spectral_centroid_mean=float(np.mean(centroid)),
        spectral_rolloff_mean=float(np.mean(rolloff)),
        zero_crossing_rate=float(np.mean(zcr)),
        mfcc_mean=mfcc.mean(axis=1).tolist(),
        mood_tags=mood_tags,
        genre_hints=genre_hints,
        waveform_normalised=waveform_normalised,
        spectrum_frames=spectrum_frames,
    )

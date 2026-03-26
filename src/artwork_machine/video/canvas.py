"""
Spotify Canvas video generator.

Produces a seamlessly looping 9:16 video (720x1280, 30fps, 8 seconds).
Spotify's canvas spec: MP4 H.264, max 8s, 720x1280, <=200 MB.

Primary path  — Runway Gen-3 Alpha:
  Sends the album cover to Runway's image-to-video API with the
  creative direction motion prompt.  Downloads a 10s clip, trims it
  to 8s, and crossfades the tail into the head for a seamless loop.

Draft / fallback path — Procedural Ken-Burns:
  Applies a sine-based zoom + slow pan to a 9:16 crop of the album
  cover, overlays particles and post-processing effects.  No API call.
  Period = 8s guarantees frame 0 == frame N (seamless loop).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from artwork_machine.audio.analyzer import AudioFeatures
from artwork_machine.ai.prompt_generator import CreativeDirection
from artwork_machine.video.effects import (
    ParticleSystem,
    bloom,
    colour_grade,
    vignette,
)
from artwork_machine.artwork.utils import hex_to_rgba


# ── Constants ─────────────────────────────────────────────────────────────────

CANVAS_W = 720
CANVAS_H = 1280
_RUNWAY_BASE = "https://api.dev.runwayml.com/v1"
_RUNWAY_VERSION = "2024-11-06"
_RUNWAY_MODEL = "gen3a_turbo"
_RUNWAY_RATIO = "768:1280"
_RUNWAY_DURATION = 10  # seconds — will be trimmed to 8s + crossfade loop


# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    album_cover_path: Path,
    direction: CreativeDirection,
    features: AudioFeatures,
    output_path: Path,
    *,
    draft: bool = False,
) -> Path:
    """
    Generate the Spotify Canvas video.

    Parameters
    ----------
    album_cover_path:
        Path to the 3000x3000 album cover PNG.
    direction:
        Creative direction (provides motion prompt and palette).
    features:
        Audio features (provides BPM for particle sync).
    output_path:
        Where to write the final MP4.
    draft:
        If True, skip Runway and use Ken-Burns fallback.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("RUNWAY_API_TOKEN", "")

    if not draft and token:
        try:
            return _generate_runway(
                album_cover_path, direction, output_path, token
            )
        except Exception as exc:
            # If Runway fails for any reason, fall back gracefully
            print(f"[canvas] Runway failed ({exc}), falling back to Ken-Burns")

    return _generate_ken_burns(
        album_cover_path, direction, features, output_path, draft=draft
    )


# ── Runway Gen-3 path ─────────────────────────────────────────────────────────

def _runway_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-Runway-Version": _RUNWAY_VERSION,
        "Content-Type": "application/json",
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _submit_runway_task(
    cover_path: Path,
    motion_prompt: str,
    token: str,
) -> str:
    """Submit image-to-video task to Runway, return task ID."""
    import base64

    img_data = base64.standard_b64encode(cover_path.read_bytes()).decode()
    suffix = cover_path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    prompt_image = f"data:{media_type};base64,{img_data}"

    payload = {
        "model": _RUNWAY_MODEL,
        "promptImage": prompt_image,
        "promptText": motion_prompt,
        "duration": _RUNWAY_DURATION,
        "ratio": _RUNWAY_RATIO,
    }

    resp = httpx.post(
        f"{_RUNWAY_BASE}/image_to_video",
        headers=_runway_headers(token),
        json=payload,
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def _poll_runway_task(task_id: str, token: str, timeout: int = 300) -> str:
    """Poll until task succeeds, return the MP4 download URL."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = httpx.get(
            f"{_RUNWAY_BASE}/tasks/{task_id}",
            headers=_runway_headers(token),
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        if status == "SUCCEEDED":
            return data["output"][0]
        if status == "FAILED":
            raise RuntimeError(f"Runway task {task_id} failed: {data}")
        time.sleep(3)
    raise TimeoutError(f"Runway task {task_id} timed out after {timeout}s")


def _generate_runway(
    cover_path: Path,
    direction: CreativeDirection,
    output_path: Path,
    token: str,
) -> Path:
    """Full Runway Gen-3 path: submit → poll → download → loop."""
    work_dir = output_path.parent / "work"
    work_dir.mkdir(exist_ok=True)
    raw_path = work_dir / "runway_raw.mp4"

    # Submit and wait
    task_id = _submit_runway_task(cover_path, direction.canvas_motion_prompt, token)
    mp4_url = _poll_runway_task(task_id, token)

    # Download raw clip
    resp = httpx.get(mp4_url, timeout=120.0, follow_redirects=True)
    resp.raise_for_status()
    raw_path.write_bytes(resp.content)

    # Trim to 8s + seamless crossfade loop via ffmpeg
    _make_seamless_loop(raw_path, output_path, duration=8, crossfade=1)

    return output_path


def _make_seamless_loop(
    src: Path,
    dst: Path,
    duration: int = 8,
    crossfade: int = 1,
) -> None:
    """
    Trim src to `duration` seconds and crossfade the tail into the head
    for a perfectly seamless loop.

      [0 .. duration-crossfade]  +  xfade([duration-crossfade .. duration], [0 .. crossfade])
    """
    trim_end = duration
    xf_start = duration - crossfade

    filter_complex = (
        f"[0:v]trim=0:{xf_start},setpts=PTS-STARTPTS[a];"
        f"[0:v]trim={xf_start}:{trim_end},setpts=PTS-STARTPTS[b];"
        f"[0:v]trim=0:{crossfade},setpts=PTS-STARTPTS[c];"
        f"[b][c]xfade=fade:duration={crossfade}:offset=0[x];"
        f"[a][x]concat=n=2:v=1:a=0[out]"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


# ── Ken-Burns fallback ────────────────────────────────────────────────────────

def _generate_ken_burns(
    cover_path: Path,
    direction: CreativeDirection,
    features: AudioFeatures,
    output_path: Path,
    *,
    draft: bool = False,
) -> Path:
    """
    Procedural Ken-Burns animation on the album cover.

    Crops the square cover to 9:16, applies a seamless sine-based zoom
    and slow pan, overlays particles, and encodes to MP4.
    """
    fps = 15 if draft else 30
    duration = 8  # seconds
    total_frames = fps * duration

    # Prepare 9:16 base image with extra headroom for zoom
    # Scale the cover so height = CANVAS_H * 1.25 (25% zoom headroom)
    headroom = 1.20
    base_h = int(CANVAS_H * headroom)
    base_w = int(CANVAS_W * headroom)

    cover = Image.open(cover_path).convert("RGB")
    # Crop to 9:16 first (center crop), then scale to headroom size
    orig_w, orig_h = cover.size
    crop_h = orig_h
    crop_w = int(orig_h * CANVAS_W / CANVAS_H)
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(orig_w * CANVAS_H / CANVAS_W)
    x0 = (orig_w - crop_w) // 2
    y0 = (orig_h - crop_h) // 2
    cover_916 = cover.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    base_img = cover_916.resize((base_w, base_h), Image.LANCZOS)
    base_arr = np.array(base_img, dtype=np.float32) / 255.0

    # Palette for particles
    pri_rgb = hex_to_rgba(direction.palette_accent)[:3]

    particles = ParticleSystem(
        width=CANVAS_W,
        height=CANVAS_H,
        n_particles=120 if not draft else 60,
        colour=pri_rgb,
    )

    frames_dir = Path(tempfile.mkdtemp())

    for i in range(total_frames):
        t = i / fps  # seconds

        # Zoom: 1.0 -> 1.12 -> 1.0  (one full cycle, seamless)
        zoom = 1.0 + 0.06 * (1.0 - np.cos(2 * np.pi * t / duration))

        # Pan: gentle sine drift
        pan_x = np.sin(2 * np.pi * t / duration) * 0.04
        pan_y = np.sin(2 * np.pi * t / duration + np.pi / 3) * 0.02

        # Sample window from base_arr
        sample_h = int(CANVAS_H / zoom)
        sample_w = int(CANVAS_W / zoom)

        cx = base_w / 2 + pan_x * base_w
        cy = base_h / 2 + pan_y * base_h

        x1 = int(cx - sample_w / 2)
        y1 = int(cy - sample_h / 2)
        x1 = max(0, min(x1, base_w - sample_w))
        y1 = max(0, min(y1, base_h - sample_h))

        crop_arr = base_arr[y1:y1 + sample_h, x1:x1 + sample_w]
        frame = Image.fromarray(
            (np.clip(crop_arr, 0, 1) * 255).astype(np.uint8)
        ).resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
        frame_arr = np.array(frame, dtype=np.float32) / 255.0

        # Add RGBA channel for effects
        frame_rgba = np.ones((CANVAS_H, CANVAS_W, 4), dtype=np.float32)
        frame_rgba[:, :, :3] = frame_arr

        # Particles
        energy = float(np.clip(features.rms_mean * 10, 0, 1))
        frame_rgba = particles.update(frame_rgba, energy=energy)

        # Bloom
        frame_rgba = bloom(frame_rgba, strength=0.3)

        # Colour grade
        bg = hex_to_rgba(direction.palette_background)[:3]
        acc = hex_to_rgba(direction.palette_accent)[:3]
        shadows = tuple(c / 255.0 * 0.15 for c in bg)
        highlights = tuple(c / 255.0 * 0.12 for c in acc)
        frame_rgba = colour_grade(
            frame_rgba,
            shadows=shadows,      # type: ignore[arg-type]
            highlights=highlights, # type: ignore[arg-type]
        )

        # Vignette
        frame_rgba = vignette(frame_rgba, strength=0.5)

        # Save frame
        out_frame = (np.clip(frame_rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(out_frame).save(str(frames_dir / f"frame_{i:05d}.png"))

    # Encode with ffmpeg
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    # Clean up temp frames
    for f in frames_dir.iterdir():
        f.unlink()
    frames_dir.rmdir()

    return output_path

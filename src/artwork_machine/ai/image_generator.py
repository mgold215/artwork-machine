"""
Image generation client.

Wraps Replicate's API to generate high-quality artwork from prompts produced by
the :mod:`prompt_generator` module.  Supports Flux 1.1 Pro (default) and SDXL
with automatic retry / exponential back-off.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import httpx
import replicate
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from artwork_machine.ai.prompt_generator import CreativeDirection


# ── Model parameter presets ────────────────────────────────────────────────────

_FLUX_PARAMS = {
    "width": 1024,
    "height": 1024,
    "output_format": "png",
    "output_quality": 100,
    "safety_tolerance": 2,
    "prompt_upsampling": True,
}

_SDXL_PARAMS = {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "high_noise_frac": 0.8,
    "apply_watermark": False,
}


# ── Public API ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_label_art(
    direction: CreativeDirection,
    output_path: Path,
    model: str = "black-forest-labs/flux-1.1-pro",
    *,
    draft: bool = False,
) -> Path:
    """
    Generate the primary artwork for the cassette label.

    The image is square (1024×1024) and returned as a PNG.  The compositor
    will crop and warp it to fit the label area of the cassette template.

    Parameters
    ----------
    direction:
        Creative direction produced by Claude.
    output_path:
        Where to save the PNG.
    model:
        Replicate model string.  Flux 1.1 Pro gives the best results.
    draft:
        If True, use fewer steps / lower resolution for fast iteration.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a richly qualified prompt
    full_prompt = (
        f"{direction.image_prompt}  "
        f"Style: {direction.art_style}.  "
        f"Colour palette dominated by {direction.palette_primary} and "
        f"{direction.palette_secondary} with {direction.palette_accent} accents.  "
        "Ultra-detailed, high fidelity, award-winning album cover art, "
        "no text, no typography."
    )

    params: dict = {"prompt": full_prompt}

    if "flux" in model.lower():
        params.update(_FLUX_PARAMS)
        if draft:
            params["width"] = 512
            params["height"] = 512
            params["output_quality"] = 80
            params["prompt_upsampling"] = False
    elif "sdxl" in model.lower():
        params.update(_SDXL_PARAMS)
        params["negative_prompt"] = direction.negative_prompt
        if draft:
            params["num_inference_steps"] = 25
            params["width"] = 768
            params["height"] = 768
    else:
        # Generic fallback — just pass prompt + dimensions
        params["negative_prompt"] = direction.negative_prompt

    output = replicate.run(model, input=params)

    # Replicate returns a URL or file-like — normalise to bytes
    img_bytes = _fetch_output(output)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    img.save(str(output_path), "PNG")
    return output_path


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_canvas_background(
    direction: CreativeDirection,
    output_path: Path,
    model: str = "black-forest-labs/flux-1.1-pro",
    *,
    draft: bool = False,
) -> Path:
    """
    Generate a 9:16 background frame for the Spotify Canvas video.

    The canvas video layers animated elements on top of this static background.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"Atmospheric, ambient visual for a music video loop.  "
        f"{direction.image_prompt}  "
        f"Style: {direction.art_style}, deep focus, cinematic.  "
        f"Dominated by {direction.palette_background} and {direction.palette_primary}.  "
        "No text, no people, no faces, seamlessly loopable aesthetic, "
        "designed for a 9:16 vertical phone screen."
    )

    params: dict = {"prompt": prompt}

    if "flux" in model.lower():
        params.update(_FLUX_PARAMS)
        params["width"] = 720
        params["height"] = 1280
        if draft:
            params["width"] = 360
            params["height"] = 640
            params["output_quality"] = 75
            params["prompt_upsampling"] = False
    else:
        params.update(_SDXL_PARAMS)
        params["width"] = 768
        params["height"] = 1344  # closest SDXL 9:16

    output = replicate.run(model, input=params)
    img_bytes = _fetch_output(output)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    img.save(str(output_path), "PNG")
    return output_path


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fetch_output(output) -> bytes:  # noqa: ANN001
    """Normalise Replicate output (URL string, list of URLs, or file object) to bytes."""
    if isinstance(output, list):
        output = output[0]
    if hasattr(output, "read"):
        return output.read()
    if isinstance(output, str) and output.startswith("http"):
        with httpx.Client(timeout=60) as client:
            r = client.get(output)
            r.raise_for_status()
            return r.content
    raise ValueError(f"Unexpected Replicate output type: {type(output)}")

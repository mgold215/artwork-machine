"""
Image generation client.

Uses the Hugging Face Inference API (free tier) to generate artwork from prompts
produced by the :mod:`prompt_generator` module.

Default model: black-forest-labs/FLUX.1-schnell
  - Free with a Hugging Face account (huggingface.co)
  - Requires agreeing to the model licence on the HF model page once

Outputs
-------
Album art   — generated at 1024×1024, upscaled to 3000×3000 with LANCZOS
Thumbnail   — generated at 1024×576, upscaled to 1280×720 (16:9)
Canvas bg   — generated at 810×1440 (9:16, oversize for Ken-Burns headroom)
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import InferenceClient
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from artwork_machine.ai.prompt_generator import CreativeDirection


_DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"

# Final output sizes
_ALBUM_ART_FINAL = (3000, 3000)
_THUMB_FINAL     = (1280, 720)

# HF generation sizes (within API limits)
_ALBUM_ART_GEN       = (1024, 1024)
_ALBUM_ART_GEN_DRAFT = (512, 512)
_THUMB_GEN           = (1024, 576)
_THUMB_GEN_DRAFT     = (512, 288)
_CANVAS_GEN          = (810, 1440)   # oversize 9:16 for Ken-Burns headroom
_CANVAS_GEN_DRAFT    = (405, 720)

# Quality suffix appended to every prompt
_QUALITY_SUFFIX = (
    "Shot on camera, 4K cinema lens, photorealistic, hyperdetailed, "
    "no CGI, no render, no text, no watermarks."
)


# ── Client ────────────────────────────────────────────────────────────────────

def _get_client() -> InferenceClient:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN is not set. "
            "Get a free token at huggingface.co → Settings → Access Tokens."
        )
    return InferenceClient(token=token)


# ── Album art ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_album_art(
    direction: CreativeDirection,
    output_path: Path,
    model: str = _DEFAULT_MODEL,
    *,
    draft: bool = False,
) -> Path:
    """
    Generate 3000×3000 album art for streaming services.

    Generates at 1024×1024 (HF API limit) then upscales to 3000×3000 with
    LANCZOS resampling — industry standard for print/streaming submissions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gen_w, gen_h = _ALBUM_ART_GEN_DRAFT if draft else _ALBUM_ART_GEN

    prompt = (
        f"{direction.album_art_prompt}  "
        f"Style: {direction.art_style}.  "
        f"Colour palette: {direction.palette_primary}, "
        f"{direction.palette_secondary}, {direction.palette_accent}.  "
        f"{_QUALITY_SUFFIX}"
    )

    image = _generate(model, prompt, gen_w, gen_h)

    # Upscale to final delivery size
    final_w, final_h = (_ALBUM_ART_GEN_DRAFT if draft else _ALBUM_ART_FINAL)
    if (gen_w, gen_h) != (final_w, final_h):
        image = image.resize((final_w, final_h), Image.LANCZOS)

    image.convert("RGB").save(str(output_path), "PNG")
    return output_path


# ── YouTube thumbnail ─────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_thumbnail(
    direction: CreativeDirection,
    output_path: Path,
    model: str = _DEFAULT_MODEL,
    *,
    draft: bool = False,
) -> Path:
    """Generate 1280×720 YouTube thumbnail (16:9)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gen_w, gen_h = _THUMB_GEN_DRAFT if draft else _THUMB_GEN

    prompt = (
        f"{direction.thumbnail_prompt}  "
        f"Style: {direction.art_style}.  "
        f"Colour palette: {direction.palette_primary}, "
        f"{direction.palette_secondary}, {direction.palette_accent}.  "
        f"{_QUALITY_SUFFIX}"
    )

    image = _generate(model, prompt, gen_w, gen_h)
    image = image.resize(_THUMB_FINAL, Image.LANCZOS)
    image.convert("RGB").save(str(output_path), "PNG")
    return output_path


# ── Spotify Canvas background ─────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_canvas_image(
    direction: CreativeDirection,
    output_path: Path,
    model: str = _DEFAULT_MODEL,
    *,
    draft: bool = False,
) -> Path:
    """
    Generate a 9:16 background image for the Spotify Canvas.

    Generated at 810×1440 (oversize) so the Ken-Burns pan/zoom compositor
    has ~12% extra pixels to travel without revealing edges.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gen_w, gen_h = _CANVAS_GEN_DRAFT if draft else _CANVAS_GEN

    prompt = (
        f"{direction.canvas_prompt}  "
        f"Style: {direction.art_style}.  "
        f"Colour palette: {direction.palette_primary}, "
        f"{direction.palette_secondary}, {direction.palette_accent}.  "
        f"{_QUALITY_SUFFIX}"
    )

    image = _generate(model, prompt, gen_w, gen_h)
    image.convert("RGBA").save(str(output_path), "PNG")
    return output_path


# ── Shared helper ─────────────────────────────────────────────────────────────

def _generate(model: str, prompt: str, width: int, height: int) -> Image.Image:
    """Call the HF Inference API and return a PIL Image."""
    client = _get_client()
    return client.text_to_image(prompt=prompt, model=model, width=width, height=height)

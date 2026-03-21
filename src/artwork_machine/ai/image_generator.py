"""
Image generation client.

Uses the Hugging Face Inference API (free tier) to generate artwork from prompts
produced by the :mod:`prompt_generator` module.

Default model: black-forest-labs/FLUX.1-schnell
  - Same Flux family as the original Flux 1.1 Pro
  - Free with a Hugging Face account (huggingface.co)
  - Requires agreeing to the model licence on the HF model page once

Canvas layers are generated at OVERSIZE resolution (810×1440 — 9:16) so the
drone parallax compositor has ~12 % extra pixels in all directions.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import InferenceClient
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from artwork_machine.ai.prompt_generator import CreativeDirection


# ── Size constants ────────────────────────────────────────────────────────────

# Label art — square
_LABEL_W, _LABEL_H = 1024, 1024
_LABEL_W_DRAFT, _LABEL_H_DRAFT = 512, 512

# Canvas layers — 9:16, oversized to give parallax compositor headroom
_LAYER_W, _LAYER_H = 810, 1440
_LAYER_W_DRAFT, _LAYER_H_DRAFT = 405, 720

# Default free model — FLUX.1-schnell on Hugging Face
_DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"


# ── Client ────────────────────────────────────────────────────────────────────

def _get_client() -> InferenceClient:
    """Return an InferenceClient using HF_TOKEN from environment."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN is not set. "
            "Get a free token at huggingface.co → Settings → Access Tokens, "
            "then add it to your .env file."
        )
    return InferenceClient(token=token)


# ── Label art ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_label_art(
    direction: CreativeDirection,
    output_path: Path,
    model: str = _DEFAULT_MODEL,
    *,
    draft: bool = False,
) -> Path:
    """
    Generate the primary cassette label artwork (square, 1024×1024).

    The prompt foregrounds photorealistic cassette structures as the hero
    subject, as specified in the creative direction.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w = _LABEL_W_DRAFT if draft else _LABEL_W
    h = _LABEL_H_DRAFT if draft else _LABEL_H

    prompt = (
        f"{direction.image_prompt}  "
        f"Style: {direction.art_style}.  "
        f"Colour palette: {direction.palette_primary}, "
        f"{direction.palette_secondary}, {direction.palette_accent}.  "
        "Shot on camera, 4K cinema lens, photorealistic, hyperdetailed, "
        "no CGI, no render, no text, no watermarks."
    )

    return _run_and_save(model, prompt, w, h, output_path)


# ── Canvas parallax layers ────────────────────────────────────────────────────

def generate_canvas_layers(
    direction: CreativeDirection,
    work_dir: Path,
    model: str = _DEFAULT_MODEL,
    *,
    draft: bool = False,
) -> dict[str, Path]:
    """
    Generate the three depth layers used by the drone parallax compositor.

    Returns a dict with keys ``"far"``, ``"mid"``, ``"near"`` mapping to PNG paths.

    Layer semantics
    ───────────────
    far  (depth 0.15) — Aerial drone background.  Barely moves.
    mid  (depth 0.50) — Overhead cassette scene.  Moderate parallax.
    near (depth 0.90) — Macro tape texture.  Maximum parallax, most movement.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "far":  work_dir / "canvas_layer_far.png",
        "mid":  work_dir / "canvas_layer_mid.png",
        "near": work_dir / "canvas_layer_near.png",
    }
    prompts = {
        "far":  direction.canvas_far_prompt,
        "mid":  direction.canvas_mid_prompt,
        "near": direction.canvas_near_prompt,
    }

    for key, path in paths.items():
        _generate_canvas_layer(
            prompt=prompts[key],
            output_path=path,
            model=model,
            draft=draft,
        )

    return paths


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _generate_canvas_layer(
    prompt: str,
    output_path: Path,
    model: str,
    *,
    draft: bool,
) -> Path:
    """Generate one canvas parallax layer at the correct oversize resolution."""
    w = _LAYER_W_DRAFT if draft else _LAYER_W
    h = _LAYER_H_DRAFT if draft else _LAYER_H

    full_prompt = (
        f"{prompt}  "
        "Shot on camera, 4K cinema lens, photorealistic, hyperdetailed, "
        "no CGI, no render, no text, no watermarks, no logos."
    )

    return _run_and_save(model, full_prompt, w, h, output_path)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _run_and_save(model: str, prompt: str, width: int, height: int, output_path: Path) -> Path:
    """Call the HF Inference API, get a PIL Image, save it as PNG."""
    client = _get_client()
    # text_to_image returns a PIL Image directly — no URL fetching needed
    image: Image.Image = client.text_to_image(
        prompt=prompt,
        model=model,
        width=width,
        height=height,
    )
    image = image.convert("RGBA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path), "PNG")
    return output_path

"""
Image generation client.

Wraps Replicate's API to generate high-quality artwork from prompts produced by
the :mod:`prompt_generator` module.  Supports Flux 1.1 Pro (default) and SDXL
with automatic retry / exponential back-off.

Canvas layers are generated at OVERSIZE resolution (810×1440 — 9:16 at max
Flux height) so the drone parallax compositor has ~12 % extra pixels to travel
across in all directions without revealing edges.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import replicate
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from artwork_machine.ai.prompt_generator import CreativeDirection


# ── Flux/SDXL defaults ────────────────────────────────────────────────────────

_FLUX_BASE = {
    "output_format": "png",
    "output_quality": 100,
    "safety_tolerance": 2,
    "prompt_upsampling": True,
}

_SDXL_BASE = {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "high_noise_frac": 0.8,
    "apply_watermark": False,
}

# Canvas layer sizes — oversized to give the parallax compositor headroom.
# 810×1440 is the maximum 9:16 that fits within Flux's 1440-px limit.
_LAYER_W, _LAYER_H = 810, 1440
_LAYER_W_DRAFT, _LAYER_H_DRAFT = 405, 720


# ── Label art ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_label_art(
    direction: CreativeDirection,
    output_path: Path,
    model: str = "black-forest-labs/flux-1.1-pro",
    *,
    draft: bool = False,
) -> Path:
    """
    Generate the primary cassette label artwork (square, 1024×1024).

    The prompt always foregrounds photorealistic cassette structures as the
    hero subject, as specified in the creative direction.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_prompt = (
        f"{direction.image_prompt}  "
        f"Style: {direction.art_style}.  "
        f"Colour palette: {direction.palette_primary}, "
        f"{direction.palette_secondary}, {direction.palette_accent}.  "
        "Hyper-detailed, photorealistic, award-winning album cover, "
        "no text, no typography, no watermarks."
    )

    params: dict = {"prompt": full_prompt}

    if "flux" in model.lower():
        params.update(_FLUX_BASE)
        params["width"], params["height"] = (512, 512) if draft else (1024, 1024)
        if draft:
            params["output_quality"] = 80
            params["prompt_upsampling"] = False
    else:
        params.update(_SDXL_BASE)
        params["negative_prompt"] = direction.negative_prompt
        params["width"], params["height"] = (768, 768) if draft else (1024, 1024)
        if draft:
            params["num_inference_steps"] = 25

    return _run_and_save(model, params, output_path)


# ── Canvas parallax layers ────────────────────────────────────────────────────

def generate_canvas_layers(
    direction: CreativeDirection,
    work_dir: Path,
    model: str = "black-forest-labs/flux-1.1-pro",
    *,
    draft: bool = False,
) -> dict[str, Path]:
    """
    Generate the three depth layers used by the drone parallax compositor.

    Returns a dict with keys ``"far"``, ``"mid"``, ``"near"`` mapping to PNG paths.

    Each image is generated at 810×1440 (9:16, max Flux resolution) — ~12 %
    larger than the 720×1280 output frame — giving the drone camera enough
    pixel headroom to pan and zoom without edge bleed.

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
            negative_prompt=direction.negative_prompt,
            output_path=path,
            model=model,
            draft=draft,
        )

    return paths


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _generate_canvas_layer(
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    model: str,
    *,
    draft: bool,
) -> Path:
    """Generate one canvas parallax layer at the correct oversize resolution."""
    lw = _LAYER_W_DRAFT if draft else _LAYER_W
    lh = _LAYER_H_DRAFT if draft else _LAYER_H

    # Suffix the prompt with quality directives
    full_prompt = (
        f"{prompt}  "
        "Photorealistic, ultra-detailed, cinematic colour grade, "
        "no text, no watermarks, no logos."
    )

    params: dict = {"prompt": full_prompt}

    if "flux" in model.lower():
        params.update(_FLUX_BASE)
        params["width"] = lw
        params["height"] = lh
        if draft:
            params["output_quality"] = 78
            params["prompt_upsampling"] = False
    else:
        params.update(_SDXL_BASE)
        params["negative_prompt"] = negative_prompt
        # SDXL nearest 9:16 within its limits
        params["width"] = 768
        params["height"] = 1344
        if draft:
            params["num_inference_steps"] = 25

    return _run_and_save(model, params, output_path)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _run_and_save(model: str, params: dict, output_path: Path) -> Path:
    output = replicate.run(model, input=params)
    img_bytes = _fetch_output(output)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    img.save(str(output_path), "PNG")
    return output_path


def _fetch_output(output) -> bytes:  # noqa: ANN001
    """Normalise Replicate output (URL, list, or file-like) to bytes."""
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

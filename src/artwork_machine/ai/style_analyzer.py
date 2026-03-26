"""
Claude vision-based style analyzer.

Accepts 1–5 reference album cover images and uses Claude's vision capability
to extract a coherent aesthetic profile across all of them.  The resulting
:class:`StyleAnalysis` is injected into the creative direction prompt so that
the generated album art matches the user's taste.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import anthropic


@dataclass
class StyleAnalysis:
    dominant_colors: list[str]      # hex strings, most prominent first
    art_style: str                  # e.g. "minimalist photography", "abstract painting"
    composition: str                # e.g. "centered subject", "rule of thirds", "full bleed"
    mood: str                       # e.g. "dark and brooding", "bright and euphoric"
    era: str                        # e.g. "90s grunge", "modern minimalism", "80s synth"
    visual_themes: list[str]        # e.g. ["nature", "urban", "faces", "abstract geometry"]
    texture: str                    # e.g. "gritty/raw", "clean/polished", "dreamy/soft"
    color_treatment: str            # e.g. "high contrast", "muted/desaturated", "vibrant"
    typography_style: str           # e.g. "bold serif", "clean sans-serif", "hand-lettered"
    summary: str                    # 2–3 sentence style brief for injecting into prompts


_SYSTEM_PROMPT = """\
You are an expert art director and music packaging designer with deep knowledge of
visual aesthetics across all genres and eras of recorded music.

Your job: analyse a set of album cover images that a user likes, identify the common
visual threads that unite them, and produce a precise style brief that can guide the
creation of a new, original album cover in a compatible aesthetic.

Rules:
- Output must be a single valid JSON object — no markdown, no prose outside JSON.
- Focus on what is SHARED across the images, not what is unique to each one.
- Be specific and visual: avoid vague adjectives like "beautiful" or "interesting".
- All hex colour values must be valid 6-digit strings starting with #.
- dominant_colors: list the 3–6 most prevalent colours across all covers (hex).
- visual_themes: list 2–5 concrete subject/theme tags.
- summary: write 2–3 sentences that a designer could use as a brief — mention the
  dominant medium, palette feel, mood, and any recurring compositional motifs.
"""

_USER_TEMPLATE = """\
I'm sharing {n} album cover image(s) that represent my visual taste.
Analyse them as a collection and identify the shared aesthetic.

Return a JSON object with EXACTLY these keys:

{{
  "dominant_colors": ["#rrggbb", ...],
  "art_style": "...",
  "composition": "...",
  "mood": "...",
  "era": "...",
  "visual_themes": ["...", ...],
  "texture": "...",
  "color_treatment": "...",
  "typography_style": "...",
  "summary": "..."
}}
"""


def _encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for a given image file."""
    suffix = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


def analyse_reference_covers(
    image_paths: list[Path],
    client: anthropic.Anthropic | None = None,
) -> StyleAnalysis:
    """
    Use Claude vision to analyse 1–5 reference album covers and return a
    :class:`StyleAnalysis` describing the shared aesthetic.

    Parameters
    ----------
    image_paths:
        Local paths to the reference cover images (JPEG/PNG/WebP).
    client:
        Optional pre-constructed Anthropic client.  If omitted, one is
        created using the ``ANTHROPIC_API_KEY`` environment variable.
    """
    if not image_paths:
        raise ValueError("At least one reference cover image is required.")
    if len(image_paths) > 5:
        image_paths = image_paths[:5]

    if client is None:
        client = anthropic.Anthropic()

    # Build multi-image message content
    content: list[dict] = []
    for path in image_paths:
        data, media_type = _encode_image(path)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        })

    content.append({
        "type": "text",
        "text": _USER_TEMPLATE.format(n=len(image_paths)),
    })

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    raw = message.content[0].text.strip()
    # Strip any accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rstrip("`").strip()

    data = json.loads(raw)

    return StyleAnalysis(
        dominant_colors=data.get("dominant_colors", []),
        art_style=data.get("art_style", ""),
        composition=data.get("composition", ""),
        mood=data.get("mood", ""),
        era=data.get("era", ""),
        visual_themes=data.get("visual_themes", []),
        texture=data.get("texture", ""),
        color_treatment=data.get("color_treatment", ""),
        typography_style=data.get("typography_style", ""),
        summary=data.get("summary", ""),
    )

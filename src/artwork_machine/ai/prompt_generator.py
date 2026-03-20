"""
Claude-powered creative prompt generator.

Given audio features + artist metadata, Claude produces:
  1. A rich image-generation prompt for the cassette label art
  2. A style descriptor used by the compositor (colour palette, era, texture)
  3. A motion descriptor used by the video engines (particle style, animation mood)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

from artwork_machine.audio.analyzer import AudioFeatures


@dataclass
class CreativeDirection:
    # ── Image generation ─────────────────────────────────────────────────────
    image_prompt: str          # Full positive prompt for Flux / SDXL
    negative_prompt: str       # Things to avoid
    art_style: str             # e.g. "vaporwave oil painting", "lo-fi watercolour"

    # ── Colour palette (hex strings) ─────────────────────────────────────────
    palette_primary: str       # e.g. "#1a0a2e"
    palette_secondary: str     # e.g. "#c084fc"
    palette_accent: str        # e.g. "#f0abfc"
    palette_background: str    # e.g. "#0d0118"

    # ── Cassette physical style ───────────────────────────────────────────────
    cassette_shell_colour: str   # hex colour for the plastic shell
    cassette_era: str            # "70s", "80s", "90s", "modern"
    cassette_brand_name: str     # fictional brand to print on shell

    # ── Motion / video feel ───────────────────────────────────────────────────
    canvas_motion_style: str     # "slow drift", "glitch pulse", "liquid bloom"
    visualiser_style: str        # "spectrum bars", "radial waveform", "particle field"

    # ── Typography ───────────────────────────────────────────────────────────
    label_font_style: str        # "bold serif", "hand-lettered", "futuristic mono"
    label_tagline: str           # short evocative phrase for label (≤8 words)


_SYSTEM_PROMPT = """\
You are an award-winning creative director who specialises in music packaging and
visual identity.  You produce precise, vivid creative briefs for AI image generators
and motion graphics artists.

Your output must be a single valid JSON object — no markdown, no prose outside JSON.
All hex colour values must be valid 6-digit hex strings starting with #.
Keep every prompt under 300 tokens.
"""

_USER_TEMPLATE = """\
Generate a complete visual creative direction for an album with the following profile:

ARTIST: {artist}
ALBUM TITLE: {album}
GENRE HINTS: {genre_hints}
MOOD TAGS: {mood_tags}
MUSICAL KEY: {key}
BPM: {bpm:.0f}
ENERGY (0-1): {energy:.2f}
BRIGHTNESS (spectral centroid Hz): {brightness:.0f}
DYNAMIC RANGE (dB): {dynamic_range:.1f}

Return a JSON object with exactly these keys:
{{
  "image_prompt": "...",
  "negative_prompt": "...",
  "art_style": "...",
  "palette_primary": "#rrggbb",
  "palette_secondary": "#rrggbb",
  "palette_accent": "#rrggbb",
  "palette_background": "#rrggbb",
  "cassette_shell_colour": "#rrggbb",
  "cassette_era": "70s|80s|90s|modern",
  "cassette_brand_name": "...",
  "canvas_motion_style": "...",
  "visualiser_style": "...",
  "label_font_style": "...",
  "label_tagline": "..."
}}

Guidelines:
- image_prompt: painterly, cinematic scene that emotionally captures the music.
  Mention specific textures, lighting, mood.  Do NOT mention cassettes.
- negative_prompt: list artefacts, styles, and quality issues to avoid.
- art_style: 3-5 word label (used for compositor style-matching).
- palette_*: colours that form a cohesive, striking palette fitting the mood.
- cassette_shell_colour: the physical plastic colour (can be translucent, chrome, etc).
- cassette_brand_name: invented retro-sounding brand (e.g. "Sonara", "Velvetone").
- canvas_motion_style: one motion concept for the Spotify Canvas loop.
- visualiser_style: one visualisation concept for the YouTube visualiser.
- label_font_style: typography direction for the cassette label text.
- label_tagline: ≤8-word evocative phrase to print on the cassette label.
"""


def generate(
    features: AudioFeatures,
    artist: str,
    album: str,
    client: anthropic.Anthropic | None = None,
) -> CreativeDirection:
    """
    Call Claude to generate a full :class:`CreativeDirection` for the album.

    Parameters
    ----------
    features:
        Analysed audio features.
    artist, album:
        Metadata provided by the user.
    client:
        Optional pre-built Anthropic client (useful for testing / DI).
    """
    if client is None:
        client = anthropic.Anthropic()

    prompt = _USER_TEMPLATE.format(
        artist=artist,
        album=album,
        genre_hints=", ".join(features.genre_hints) or "unknown",
        mood_tags=", ".join(features.mood_tags) or "unknown",
        key=features.key,
        bpm=features.bpm,
        energy=min(features.rms_mean * 10, 1.0),
        brightness=features.spectral_centroid_mean,
        dynamic_range=features.dynamic_range_db,
    )

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if the model returns them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)

    return CreativeDirection(
        image_prompt=data["image_prompt"],
        negative_prompt=data["negative_prompt"],
        art_style=data["art_style"],
        palette_primary=data["palette_primary"],
        palette_secondary=data["palette_secondary"],
        palette_accent=data["palette_accent"],
        palette_background=data["palette_background"],
        cassette_shell_colour=data["cassette_shell_colour"],
        cassette_era=data["cassette_era"],
        cassette_brand_name=data["cassette_brand_name"],
        canvas_motion_style=data["canvas_motion_style"],
        visualiser_style=data["visualiser_style"],
        label_font_style=data["label_font_style"],
        label_tagline=data["label_tagline"],
    )

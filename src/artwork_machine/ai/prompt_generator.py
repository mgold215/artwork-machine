"""
Claude-powered creative prompt generator.

Given audio features + artist metadata, Claude produces:
  1. Three image prompts: album art (1:1), thumbnail (16:9), canvas background (9:16)
  2. Colour palette, motion style, and text descriptors
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

from artwork_machine.audio.analyzer import AudioFeatures


@dataclass
class CreativeDirection:
    # ── Image prompts ─────────────────────────────────────────────────────────
    album_art_prompt: str      # 1:1 square — primary streaming artwork
    thumbnail_prompt: str      # 16:9 — YouTube thumbnail
    canvas_prompt: str         # 9:16 — Spotify Canvas background
    negative_prompt: str
    art_style: str             # 3-5 word aesthetic label

    # ── Colour palette (hex strings) ─────────────────────────────────────────
    palette_primary: str
    palette_secondary: str
    palette_accent: str
    palette_background: str

    # ── Motion / video ────────────────────────────────────────────────────────
    canvas_motion_style: str   # drives the Ken-Burns animation preset
    visualiser_style: str      # "spectrum bars" | "radial waveform" | "particle field"

    # ── Text ─────────────────────────────────────────────────────────────────
    tagline: str               # ≤8 words, shown on canvas and shorts


_SYSTEM_PROMPT = """\
You are an award-winning creative director specialising in music packaging and
editorial photography.  Your visual language is informed by labels like KREAM
and Afterlife, and films like Inception and Blade Runner 2049 — ominous,
architectural, bleak, and hauntingly beautiful.

RULES:
- Output must be a single valid JSON object — no markdown, no prose outside JSON.
- All hex colour values: valid 6-digit strings starting with #.
- All imagery must look like it was SHOT ON A REAL CAMERA — 4K cinema lens,
  natural or practical lighting only.  Never CGI, never render, never digital art.
- The same colour palette must carry through all three prompts as one visual world.
- Prompts must be vivid, specific, and under 280 tokens each.
"""

_USER_TEMPLATE = """\
Generate a complete visual creative direction for a streaming release:

ARTIST: {artist}
ALBUM TITLE: {album}
GENRE HINTS: {genre_hints}
MOOD TAGS: {mood_tags}
MUSICAL KEY: {key}
BPM: {bpm:.0f}
ENERGY (0-1): {energy:.2f}
BRIGHTNESS (spectral centroid Hz): {brightness:.0f}
DYNAMIC RANGE (dB): {dynamic_range:.1f}

Return a JSON object with EXACTLY these keys:

{{
  "album_art_prompt": "...",
  "thumbnail_prompt": "...",
  "canvas_prompt": "...",
  "negative_prompt": "...",
  "art_style": "...",

  "palette_primary": "#rrggbb",
  "palette_secondary": "#rrggbb",
  "palette_accent": "#rrggbb",
  "palette_background": "#rrggbb",

  "canvas_motion_style": "...",
  "visualiser_style": "...",
  "tagline": "..."
}}

FIELD GUIDELINES:

album_art_prompt:
  A REAL CAMERA photograph — never CGI, never a render.  A CASSETTE TAPE is the
  hero subject, physically embedded in or consumed by an ominous structure or
  environment: pressed into crumbling brutalist concrete, half-submerged in a
  rain-flooded tunnel, lodged in rusted industrial grating, lying on fog-covered
  rooftop asphalt, or cemented into an overpass wall.  The scene should feel
  bleak, cinematic, and slightly unsettling — the aesthetic of KREAM or the
  hallway sequence from Inception.  The cassette shell material, colour, and
  tape texture must be clearly visible.  Describe: the exact structure/surface
  the cassette inhabits, the ambient light source (overcast skylight, sodium
  vapour streetlight, single bare bulb), and camera lens language ("Sony A7R V,
  85mm f/1.4", "4K anamorphic", "shallow DOF, foreground blur").
  Palette: {palette_primary_hint} and {palette_secondary_hint}.
  No text, no typography, no people.

thumbnail_prompt:
  A REAL CAMERA photograph, 16:9 cinematic frame.  A CASSETTE TAPE embedded in
  or resting against a dramatic piece of architecture or infrastructure — wider
  composition showing more of the surrounding ominous environment.  Designed to
  read at small size on YouTube: bold contrast, strong visual anchor (the
  cassette), dark atmospheric surroundings.  Same colour palette and mood.
  Shot on camera, 4K.  No text, no people.

canvas_prompt:
  A REAL CAMERA photograph, vertical 9:16 composition for a phone screen.
  A CASSETTE TAPE integrated into a powerful vertical environment — a narrow
  fog-filled stairwell, a brutalist tower seen from below, an aerial shot
  straight down over fog-covered infrastructure, rain-soaked concrete walkways,
  or a dimly lit underground passage.  The cassette may be smaller in frame
  but must be present.  Strong vertical geometry dominates.  Same colour
  palette.  Shot on camera, 4K.  No text, no people.

negative_prompt:
  CGI, 3D render, digital art, Octane render, computer generated, studio
  lighting, clean backgrounds, colourful, bright, cheerful, cartoonish,
  watermarks, logos, text, deformations, artefacts, oversaturated colours,
  people, faces.

art_style:
  3-5 word label describing the dominant photographic aesthetic (e.g.
  "ominous brutalist location photography", "bleak architectural noir",
  "cinematic desolate documentary").

canvas_motion_style:
  One vivid phrase describing the camera animation feel for the Canvas loop.
  Examples: "slow liquid bloom", "glitch pulse orbit", "cinematic drift reveal",
  "hypnotic fluid swirl", "electric parallax surge".

visualiser_style:
  "spectrum bars" | "radial waveform" | "particle field"
  Choose based on mood: high-energy → bars, atmospheric/minor → radial,
  complex/noisy → particles.

tagline:
  ≤8 evocative words — a line that captures the emotional core of the music.
"""


def generate(
    features: AudioFeatures,
    artist: str,
    album: str,
    client: anthropic.Anthropic | None = None,
) -> CreativeDirection:
    """Call Claude to generate a full :class:`CreativeDirection`."""
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
        palette_primary_hint="the primary palette colour",
        palette_secondary_hint="the secondary palette colour",
    )

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1536,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)

    return CreativeDirection(
        album_art_prompt=data["album_art_prompt"],
        thumbnail_prompt=data["thumbnail_prompt"],
        canvas_prompt=data["canvas_prompt"],
        negative_prompt=data["negative_prompt"],
        art_style=data["art_style"],
        palette_primary=data["palette_primary"],
        palette_secondary=data["palette_secondary"],
        palette_accent=data["palette_accent"],
        palette_background=data["palette_background"],
        canvas_motion_style=data["canvas_motion_style"],
        visualiser_style=data["visualiser_style"],
        tagline=data["tagline"],
    )

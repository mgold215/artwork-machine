"""
Claude-powered creative prompt generator.

Given audio features + artist metadata, Claude produces:
  1. A photorealistic cassette-structure image prompt for the label art
  2. Three layered canvas prompts for the drone parallax system
     (far aerial · mid cassette overhead · near macro tape)
  3. Style, palette, motion, and typography descriptors
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

from artwork_machine.audio.analyzer import AudioFeatures


@dataclass
class CreativeDirection:
    # ── Primary label art ─────────────────────────────────────────────────────
    image_prompt: str          # Positive prompt — always features cassette structures
    negative_prompt: str       # Things to avoid
    art_style: str             # e.g. "hyperrealistic macro photography"

    # ── Canvas parallax layer prompts ─────────────────────────────────────────
    canvas_far_prompt: str     # Aerial/drone shot background (depth layer 0)
    canvas_mid_prompt: str     # Overhead cassette scene (depth layer 1)
    canvas_near_prompt: str    # Extreme macro tape/shell texture (depth layer 2)

    # ── Colour palette (hex strings) ─────────────────────────────────────────
    palette_primary: str
    palette_secondary: str
    palette_accent: str
    palette_background: str

    # ── Cassette physical style ───────────────────────────────────────────────
    cassette_shell_colour: str
    cassette_era: str            # "70s", "80s", "90s", "modern"
    cassette_brand_name: str

    # ── Motion / video feel ───────────────────────────────────────────────────
    canvas_motion_style: str     # drives animation preset, e.g. "liquid bloom"
    visualiser_style: str        # "spectrum bars", "radial waveform", "particle field"

    # ── Typography ───────────────────────────────────────────────────────────
    label_font_style: str
    label_tagline: str           # ≤8 words


_SYSTEM_PROMPT = """\
You are an award-winning creative director specialising in music packaging and
editorial photography.  Your visual language is informed by labels like KREAM
and Afterlife, and films like Inception and Blade Runner 2049 — ominous,
architectural, bleak, and hauntingly beautiful.

RULES:
- Output must be a single valid JSON object — no markdown, no prose outside JSON.
- All hex colour values: valid 6-digit strings starting with #.
- Every prompt must foreground CASSETTE TAPE physically integrated into or placed
  within ominous structures, brutalist architecture, or desolate environments.
- All imagery must look like it was SHOT ON A REAL CAMERA — 4K cinema lens,
  natural or practical lighting only.  Never CGI, never render, never digital art.
- The same colour palette must carry through all three canvas layers as a
  cohesive, unbroken visual world.
- Prompts must be vivid, specific, and under 280 tokens each.
"""

_USER_TEMPLATE = """\
Generate a complete visual and motion creative direction for an album:

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
  "image_prompt": "...",
  "negative_prompt": "...",
  "art_style": "...",

  "canvas_far_prompt": "...",
  "canvas_mid_prompt": "...",
  "canvas_near_prompt": "...",

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

FIELD GUIDELINES:

image_prompt:
  A REAL CAMERA photograph — never CGI, never a render.  A cassette tape is
  physically embedded in, resting against, or consumed by an ominous structure:
  crumbling brutalist concrete, rain-soaked industrial flooring, a fog-covered
  rooftop, a dimly lit stairwell, or a flooded underground space.  The scene
  should feel bleak, cinematic, and slightly unsettling — the aesthetic of
  KREAM or the Inception hallway sequence.  Specify: the cassette shell material
  and colour, the exact surface or structure it inhabits, the ambient light
  source (overcast skylight, single bare bulb, sodium vapour streetlight),
  and camera/lens language ("shot on a Sony A7R V with a 85mm f/1.4",
  "4K anamorphic lens flare", "shallow DOF, foreground blur").
  Palette: {palette_primary_hint} and {palette_secondary_hint}.
  No text, no typography.

negative_prompt:
  CGI, 3D render, digital art, Octane render, computer generated, studio
  lighting, clean backgrounds, colourful, bright, cheerful, cartoonish,
  watermarks, logos, text, deformations, artefacts, oversaturated colours.

art_style:
  3-5 word label describing the dominant photographic aesthetic (e.g.
  "ominous brutalist location photography", "bleak architectural macro",
  "cinematic noir documentary").

canvas_far_prompt:
  AERIAL DRONE PHOTOGRAPHY shot straight down from 300-500 m.  The landscape
  below is bleak and ominous — empty highways at night, fog-covered industrial
  zones, brutalist housing blocks seen from above, or desolate terrain with
  geometric shadows.  No cassettes at this scale.  The scene should feel like
  the opening shot of a thriller film.  Colour grade: muted, desaturated,
  {palette_bg_hint} tones.  Shot on a cinema drone camera, 4K, hyperrealistic.

canvas_mid_prompt:
  OVERHEAD CAMERA shot looking straight down.  A cassette tape ({era} era,
  {shell_hint} colour shell) rests on an ominous surface — cracked concrete,
  wet asphalt, rusted metal grating, or aged tile.  The magnetic tape has been
  partially unspooled, flowing across the surface in dark ribbons.  The light
  source is a single overhead practical lamp, casting hard shadows.
  Colour palette: {palette_primary_hint} / {palette_secondary_hint}.
  Shot on camera, 4K, photorealistic.  No CGI.

canvas_near_prompt:
  EXTREME MACRO PHOTOGRAPHY — camera pressed against the cassette shell where
  it meets the surrounding structure (concrete, rust, wet stone).  Fill the
  frame with the boundary between cassette and material: the cassette edge,
  ferric tape texture, cracks or debris from the environment pressing in.
  Light: a single raking practical source revealing micro-texture.
  Palette: {palette_accent_hint} highlights, {palette_bg_hint} deep shadows.
  Shot on a 100mm macro lens, f/2.8, natural or practical light only.
  No text, no CGI.

canvas_motion_style:
  One vivid phrase describing the camera/animation mood for the canvas loop.
  Choose a style that fits the music energy.
  Examples: "slow liquid bloom", "glitch pulse orbit", "cinematic drift reveal",
  "hypnotic fluid swirl", "electric parallax surge".

visualiser_style:
  "spectrum bars" | "radial waveform" | "particle field"

label_font_style:
  Typography direction for the cassette label: e.g. "condensed sans-serif
  industrial", "hand-lettered retro italic", "futuristic mono stencil".

label_tagline:
  ≤8 evocative words to emboss on the cassette label.

cassette_brand_name:
  Invented retro-sounding tape brand (e.g. "Ferrox", "Chromalon", "Velvetone").
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

    # Pre-compute palette hints for the template (Claude sets them in JSON;
    # these hints are for the human-readable guidelines section only)
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
        # Inline hints so Claude sees them in the field docs
        palette_primary_hint="the primary palette colour",
        palette_secondary_hint="the secondary palette colour",
        palette_bg_hint="the background palette colour",
        palette_accent_hint="the accent colour",
        era="appropriate cassette era",
        shell_hint="chosen shell colour",
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
        image_prompt=data["image_prompt"],
        negative_prompt=data["negative_prompt"],
        art_style=data["art_style"],
        canvas_far_prompt=data["canvas_far_prompt"],
        canvas_mid_prompt=data["canvas_mid_prompt"],
        canvas_near_prompt=data["canvas_near_prompt"],
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

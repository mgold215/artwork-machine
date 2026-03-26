"""
Claude-powered creative direction generator.

Given audio features + an optional style analysis from the user's reference
covers, Claude produces a complete creative brief for generating:

  1. A square album cover image prompt (Flux 1.1 Pro)
  2. A canvas animation motion prompt (Runway Gen-3)
  3. Colour palette, visualiser style, and typography descriptors
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

from artwork_machine.audio.analyzer import AudioFeatures
from artwork_machine.ai.style_analyzer import StyleAnalysis


@dataclass
class CreativeDirection:
    # ── Album cover art ───────────────────────────────────────────────────────
    album_art_prompt: str       # Flux prompt for 3000×3000 square cover
    negative_prompt: str        # Things to avoid
    art_style: str              # e.g. "cinematic macro photography"

    # ── Colour palette (hex) ─────────────────────────────────────────────────
    palette_primary: str
    palette_secondary: str
    palette_accent: str
    palette_background: str

    # ── Canvas animation (Runway Gen-3) ──────────────────────────────────────
    canvas_motion_style: str    # Short preset name: "atmospheric drift" etc.
    canvas_motion_prompt: str   # Full Runway Gen-3 motion description

    # ── YouTube visualiser ───────────────────────────────────────────────────
    visualiser_style: str       # "spectrum bars" | "radial waveform" | "particle field"

    # ── Typography / text ────────────────────────────────────────────────────
    typography_style: str       # e.g. "condensed sans-serif industrial"
    tagline: str                # ≤8 evocative words


_SYSTEM_PROMPT = """\
You are an award-winning art director specialising in music packaging and
album cover design.  You translate musical and aesthetic signals into precise
creative briefs for AI image and video generation tools.

RULES:
- Output must be a single valid JSON object — no markdown, no prose outside JSON.
- All hex colour values: valid 6-digit strings starting with #.
- album_art_prompt must describe a SQUARE (1:1) image — this is crucial for
  album cover format.  Do NOT mention cassette tapes.  Focus on the visual
  concept, mood, and subject matter that fits the music.
- Prompts must be vivid, specific, sensory, and under 300 tokens each.
- canvas_motion_prompt must be written as a Runway Gen-3 motion instruction:
  short, vivid, describing MOVEMENT in the scene (not what the scene looks like).
"""

_USER_TEMPLATE = """\
Generate a complete visual creative direction for this album:

ARTIST: {artist}
ALBUM TITLE: {album}
GENRE HINTS: {genre_hints}
MOOD TAGS: {mood_tags}
MUSICAL KEY: {key}
BPM: {bpm:.0f}
ENERGY (0-1): {energy:.2f}
BRIGHTNESS (spectral centroid Hz): {brightness:.0f}
DYNAMIC RANGE (dB): {dynamic_range:.1f}

{style_section}

Return a JSON object with EXACTLY these keys:

{{
  "album_art_prompt": "...",
  "negative_prompt": "...",
  "art_style": "...",

  "palette_primary": "#rrggbb",
  "palette_secondary": "#rrggbb",
  "palette_accent": "#rrggbb",
  "palette_background": "#rrggbb",

  "canvas_motion_style": "...",
  "canvas_motion_prompt": "...",

  "visualiser_style": "...",
  "typography_style": "...",
  "tagline": "..."
}}

FIELD GUIDELINES:

album_art_prompt:
  A square album cover concept that matches the music's emotional character.
  {style_prompt_guidance}
  Include: subject matter, colour palette, lighting, medium/technique, mood.
  Camera/render language if photographic: lens, aperture, lighting setup.
  No text, no typography in the image.  Make it feel like a real, iconic cover.

negative_prompt:
  Artefacts, blur, distortion, watermarks, text, logos, cassette tapes,
  cartoonish rendering, oversaturated colours, and any visual issues to avoid.

art_style:
  3-5 word label for the dominant aesthetic (e.g. "cinematic macro photography",
  "abstract expressionist oil painting", "brutalist graphic design").

palette_primary / palette_secondary / palette_accent / palette_background:
  Cohesive 4-colour palette that fits both the music mood and {style_palette_hint}.

canvas_motion_style:
  2-4 word preset name for the animation mood, e.g.:
  "slow atmospheric drift", "rhythmic pulse zoom", "liquid chromatic bloom",
  "hypnotic spiral pull", "electric parallax surge".

canvas_motion_prompt:
  A Runway Gen-3 motion instruction (15-40 words).  Describe MOVEMENT only -
  camera moves, element animations, atmospheric changes.  The scene itself
  is already established by the album cover; this prompt animates it.
  Example: "Camera slowly drifts forward, depth of field shifts, particles
  drift upward, colours breathe in and out with a subtle pulse."

visualiser_style:
  "spectrum bars" - use for energetic, rhythmically complex music
  "radial waveform" - use for atmospheric, ambient, or minor-key music
  "particle field" - use for experimental, noise-driven, or intricate texture

typography_style:
  Font direction for album title / artist name overlays in the visualiser,
  e.g. "bold condensed sans-serif", "elegant thin serif", "hand-lettered script".

tagline:
  <=8 evocative words that could serve as a subtitle or strapline for this album.
"""

_STYLE_SECTION_TEMPLATE = """\
USER STYLE REFERENCE (extracted from covers they like):
  Summary: {summary}
  Art style: {art_style}
  Mood: {mood}
  Era / aesthetic: {era}
  Visual themes: {visual_themes}
  Color treatment: {color_treatment}
  Texture: {texture}
  Dominant colors: {dominant_colors}

The new cover should feel like it belongs in this user's collection -- same DNA,
but original and tailored to THIS specific track's musical character.
"""


def generate(
    features: AudioFeatures,
    artist: str,
    album: str,
    style: StyleAnalysis | None = None,
    client: anthropic.Anthropic | None = None,
) -> CreativeDirection:
    """
    Call Claude to generate a :class:`CreativeDirection` for the given track.

    Parameters
    ----------
    features:
        Audio features extracted by :func:`artwork_machine.audio.analyzer.analyse`.
    artist:
        Artist name.
    album:
        Album / track title.
    style:
        Optional style analysis from user's reference covers.  When provided,
        Claude will tailor the creative direction to match their taste.
    client:
        Optional pre-constructed Anthropic client.
    """
    if client is None:
        client = anthropic.Anthropic()

    if style is not None:
        style_section = _STYLE_SECTION_TEMPLATE.format(
            summary=style.summary,
            art_style=style.art_style,
            mood=style.mood,
            era=style.era,
            visual_themes=", ".join(style.visual_themes),
            color_treatment=style.color_treatment,
            texture=style.texture,
            dominant_colors=", ".join(style.dominant_colors),
        )
        style_prompt_guidance = (
            f"Strongly draw from the user's aesthetic: {style.summary} "
            f"Visual themes: {', '.join(style.visual_themes)}. "
            f"Color treatment: {style.color_treatment}."
        )
        style_palette_hint = (
            f"the user's preferred dominant colors ({', '.join(style.dominant_colors[:3])})"
        )
    else:
        style_section = "No style reference provided -- use music features alone."
        style_prompt_guidance = "Base the visual concept purely on the musical character."
        style_palette_hint = "the music's mood"

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
        style_section=style_section,
        style_prompt_guidance=style_prompt_guidance,
        style_palette_hint=style_palette_hint,
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
        raw = raw.rstrip("`").strip()

    data = json.loads(raw)

    return CreativeDirection(
        album_art_prompt=data["album_art_prompt"],
        negative_prompt=data["negative_prompt"],
        art_style=data["art_style"],
        palette_primary=data["palette_primary"],
        palette_secondary=data["palette_secondary"],
        palette_accent=data["palette_accent"],
        palette_background=data["palette_background"],
        canvas_motion_style=data["canvas_motion_style"],
        canvas_motion_prompt=data["canvas_motion_prompt"],
        visualiser_style=data["visualiser_style"],
        typography_style=data["typography_style"],
        tagline=data["tagline"],
    )

"""
Cog predictor for artwork-machine on Replicate.

Takes an audio file + artist/album info and generates:
  - Album art (3000x3000 PNG)
  - YouTube thumbnail (1280x720 PNG)
  - Spotify Canvas background (810x1440 PNG)

All image generation is powered by Claude (creative direction) + FLUX (image gen).
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Add the src directory so our artwork_machine package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cog import BasePredictor, Input, Path as CogPath

import anthropic
from huggingface_hub import InferenceClient

from artwork_machine.audio.analyzer import analyse
from artwork_machine.ai.prompt_generator import generate as generate_direction
from artwork_machine.ai import image_generator


class Predictor(BasePredictor):
    """Replicate predictor that generates album artwork from an audio file."""

    def setup(self):
        """
        Called once when the model container starts.
        We don't need to load any large model weights — Claude and FLUX
        are called via API, so setup is lightweight.
        """
        pass

    def predict(
        self,
        audio: CogPath = Input(
            description="Audio file (MP3, WAV, FLAC, AIFF, or M4A)"
        ),
        artist: str = Input(
            description="Artist name (e.g. 'moodmixformat')"
        ),
        album: str = Input(
            description="Album or track title (e.g. 'BLOOM')"
        ),
        output_type: str = Input(
            description="Which artwork to generate",
            choices=["album_art", "thumbnail", "canvas_bg", "all"],
            default="all",
        ),
        draft_mode: bool = Input(
            description="Generate at lower resolution for faster previews",
            default=False,
        ),
        image_model: str = Input(
            description="Hugging Face model ID for image generation",
            default="black-forest-labs/FLUX.1-schnell",
        ),
    ) -> list[CogPath]:
        """
        Run the artwork-machine pipeline and return generated images.

        Pipeline steps:
        1. Analyse the audio (BPM, key, energy, mood tags)
        2. Generate creative direction with Claude (prompts, palette, style)
        3. Generate images with FLUX via Hugging Face
        """

        # -- Create a temp directory for all outputs --
        work_dir = Path(tempfile.mkdtemp(prefix="artwork-machine-"))

        # -- Step 1: Analyse audio --
        # Extracts BPM, key, mood tags, energy level, etc. from the audio file
        print("Step 1/3: Analysing audio...", flush=True)
        features = analyse(Path(audio))
        print(
            f"  -> {features.duration:.0f}s, {features.bpm:.0f} BPM, "
            f"{features.key}, mood: {', '.join(features.mood_tags)}",
            flush=True,
        )

        # -- Step 2: Creative direction via Claude --
        # Claude picks art style, colour palette, image prompts, and a tagline
        print("Step 2/3: Generating creative direction with Claude...", flush=True)
        client = anthropic.Anthropic()
        direction = generate_direction(features, artist, album, client=client)
        print(f"  -> Style: {direction.art_style}", flush=True)
        print(
            f"  -> Palette: {direction.palette_primary} / "
            f"{direction.palette_secondary} / {direction.palette_accent}",
            flush=True,
        )
        print(f'  -> Tagline: "{direction.tagline}"', flush=True)

        # -- Step 3: Generate images with FLUX --
        print("Step 3/3: Generating images...", flush=True)
        output_paths = []

        # Album art — 3000x3000 square for streaming platforms
        if output_type in ("album_art", "all"):
            print("  -> Generating album art (3000x3000)...", flush=True)
            art_path = work_dir / "album_art.png"
            image_generator.generate_album_art(
                direction, art_path, model=image_model, draft=draft_mode
            )
            output_paths.append(art_path)

        # YouTube thumbnail — 1280x720 landscape
        if output_type in ("thumbnail", "all"):
            print("  -> Generating thumbnail (1280x720)...", flush=True)
            thumb_path = work_dir / "thumbnail.png"
            image_generator.generate_thumbnail(
                direction, thumb_path, model=image_model, draft=draft_mode
            )
            output_paths.append(thumb_path)

        # Spotify Canvas background — 810x1440 portrait
        if output_type in ("canvas_bg", "all"):
            print("  -> Generating canvas background (810x1440)...", flush=True)
            canvas_path = work_dir / "canvas_bg.png"
            image_generator.generate_canvas_image(
                direction, canvas_path, model=image_model, draft=draft_mode
            )
            output_paths.append(canvas_path)

        # -- Write the creative direction as a JSON sidecar --
        # This gives you the style info, palette, and prompts Claude chose
        meta_path = work_dir / "creative_direction.json"
        meta_path.write_text(
            json.dumps(
                {
                    "artist": artist,
                    "album": album,
                    "art_style": direction.art_style,
                    "palette": {
                        "primary": direction.palette_primary,
                        "secondary": direction.palette_secondary,
                        "accent": direction.palette_accent,
                        "background": direction.palette_background,
                    },
                    "tagline": direction.tagline,
                    "canvas_motion_style": direction.canvas_motion_style,
                    "visualiser_style": direction.visualiser_style,
                    "audio": {
                        "bpm": round(features.bpm),
                        "key": features.key,
                        "duration_seconds": round(features.duration, 1),
                        "mood_tags": features.mood_tags,
                        "genre_hints": features.genre_hints,
                    },
                },
                indent=2,
            )
        )
        output_paths.append(meta_path)

        print(f"Done — {len(output_paths)} files generated.", flush=True)

        # Return all generated files
        return [CogPath(p) for p in output_paths]

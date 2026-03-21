"""
Main orchestration pipeline.

Pipeline stages
───────────────
1.  Audio analysis          → AudioFeatures
2.  Creative direction      → CreativeDirection   (Claude)
3a. Album art               → album_art.png       (1024px → upscaled 3000×3000)
3b. YouTube thumbnail       → thumbnail.png       (1024×576 → 1280×720)
3c. Canvas background       → canvas_bg.png       (810×1440 oversize 9:16)
4.  Spotify Canvas          → spotify_canvas.mp4  (720×1280, 8s loop)
5.  30-second short         → short.mp4           (1080×1920, 30s)
6.  YouTube visualizer      → youtube_visualizer.mp4  (1920×1080, full length)
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from artwork_machine.audio.analyzer import analyse, AudioFeatures
from artwork_machine.ai.prompt_generator import generate as generate_direction, CreativeDirection
from artwork_machine.ai import image_generator
from artwork_machine.video import canvas as canvas_gen
from artwork_machine.video import visualizer as viz_gen
from artwork_machine.video import short as short_gen

console = Console()


@dataclass
class PipelineResult:
    album_art:          Path
    thumbnail:          Path
    spotify_canvas:     Path
    short_video:        Path
    youtube_visualizer: Path
    output_dir:         Path
    features:           AudioFeatures
    direction:          CreativeDirection
    elapsed_seconds:    float = 0.0


@dataclass
class PipelineOptions:
    artist: str
    album:  str
    audio_path:    Path
    output_dir:    Path
    draft:         bool = False
    skip_canvas:   bool = False
    skip_short:    bool = False
    skip_visualizer: bool = False
    image_model:   str = "black-forest-labs/FLUX.1-schnell"


def run(opts: PipelineOptions) -> PipelineResult:
    t0 = time.perf_counter()

    safe_name = _safe_slug(f"{opts.artist}_{opts.album}")
    out  = opts.output_dir / safe_name
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:

        # ── 1. Audio analysis ─────────────────────────────────────────────────
        t = progress.add_task("Analysing audio …", total=None)
        features = analyse(opts.audio_path)
        progress.update(t, description=f"[green]✓ Audio analysed  ({features.duration:.0f}s, {features.bpm:.0f} BPM, {features.key})")

        # ── 2. Creative direction ─────────────────────────────────────────────
        t2 = progress.add_task("Generating creative direction with Claude …", total=None)
        client    = anthropic.Anthropic()
        direction = generate_direction(features, opts.artist, opts.album, client=client)
        progress.update(t2, description=f"[green]✓ Direction: {direction.art_style}")

        # ── 3a. Album art (3000×3000) ─────────────────────────────────────────
        t3a = progress.add_task("Generating album art …", total=None)
        album_art_path = work / "album_art.png"
        image_generator.generate_album_art(
            direction, album_art_path, model=opts.image_model, draft=opts.draft,
        )
        progress.update(t3a, description="[green]✓ Album art generated (3000×3000)")

        # ── 3b. YouTube thumbnail (1280×720) ──────────────────────────────────
        t3b = progress.add_task("Generating YouTube thumbnail …", total=None)
        thumbnail_path = work / "thumbnail.png"
        image_generator.generate_thumbnail(
            direction, thumbnail_path, model=opts.image_model, draft=opts.draft,
        )
        progress.update(t3b, description="[green]✓ Thumbnail generated (1280×720)")

        # ── 3c. Canvas background (810×1440) ──────────────────────────────────
        canvas_bg_path = work / "canvas_bg.png"
        if not opts.skip_canvas:
            t3c = progress.add_task("Generating Spotify Canvas background …", total=None)
            image_generator.generate_canvas_image(
                direction, canvas_bg_path, model=opts.image_model, draft=opts.draft,
            )
            progress.update(t3c, description="[green]✓ Canvas background generated (9:16)")

        # Copy deliverables to output root
        shutil.copy(album_art_path, out / "album_art.png")
        shutil.copy(thumbnail_path, out / "thumbnail.png")

        # ── 4. Spotify Canvas ─────────────────────────────────────────────────
        canvas_output = out / "spotify_canvas.mp4"
        if not opts.skip_canvas:
            t4 = progress.add_task("Rendering Spotify Canvas …", total=None)
            canvas_gen.generate(
                bg_image_path=canvas_bg_path,
                album_art_path=album_art_path,
                direction=direction,
                features=features,
                artist=opts.artist,
                album=opts.album,
                output_path=canvas_output,
                draft=opts.draft,
            )
            progress.update(t4, description="[green]✓ Spotify Canvas rendered (8s loop)")

        # ── 5. 30-second short ────────────────────────────────────────────────
        short_output = out / "short.mp4"
        if not opts.skip_short:
            t5 = progress.add_task("Rendering 30-second short …", total=None)
            short_gen.generate(
                album_art_path=album_art_path,
                audio_path=opts.audio_path,
                direction=direction,
                features=features,
                artist=opts.artist,
                album=opts.album,
                output_path=short_output,
                draft=opts.draft,
            )
            progress.update(t5, description="[green]✓ 30-second short rendered (1080×1920)")

        # ── 6. YouTube visualizer ─────────────────────────────────────────────
        viz_output = out / "youtube_visualizer.mp4"
        if not opts.skip_visualizer:
            t6 = progress.add_task("Rendering YouTube visualizer …", total=None)
            viz_gen.generate(
                album_art_path=album_art_path,
                audio_path=opts.audio_path,
                direction=direction,
                features=features,
                artist=opts.artist,
                album=opts.album,
                output_path=viz_output,
                draft=opts.draft,
            )
            progress.update(t6, description="[green]✓ YouTube visualizer rendered (1920×1080)")

    elapsed = time.perf_counter() - t0
    console.print(f"\n[bold green]✓ Pipeline complete in {elapsed:.1f}s[/bold green]")
    console.print(f"  Output: [cyan]{out}[/cyan]")

    return PipelineResult(
        album_art=out / "album_art.png",
        thumbnail=out / "thumbnail.png",
        spotify_canvas=canvas_output,
        short_video=short_output,
        youtube_visualizer=viz_output,
        output_dir=out,
        features=features,
        direction=direction,
        elapsed_seconds=elapsed,
    )


def _safe_slug(text: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text).strip("_")[:80]

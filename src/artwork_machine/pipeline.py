"""
Main orchestration pipeline.

Wires together every module in the correct order and exposes a single
:func:`run` function consumed by the CLI.

Pipeline stages
───────────────
1.  Audio analysis          → AudioFeatures
2.  Creative direction      → CreativeDirection   (Claude)
3.  Image generation        → label_art.png       (Flux / SDXL)
                            → canvas_bg.png
4.  Cassette composition    → cassette_A.png
                            → cassette_B.png
5.  Spotify Canvas          → canvas.mp4
6.  YouTube Visualiser      → visualizer.mp4
7.  Export package          → <artist>_<album>/
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

import anthropic

from artwork_machine.audio.analyzer import analyse, AudioFeatures
from artwork_machine.ai.prompt_generator import generate as generate_direction, CreativeDirection
from artwork_machine.ai import image_generator
from artwork_machine.artwork import cassette
from artwork_machine.video import canvas as canvas_gen
from artwork_machine.video import visualizer as viz_gen

console = Console()


@dataclass
class PipelineResult:
    cassette_side_a: Path
    cassette_side_b: Path
    spotify_canvas: Path
    youtube_visualizer: Path
    output_dir: Path
    features: AudioFeatures
    direction: CreativeDirection
    elapsed_seconds: float = 0.0


@dataclass
class PipelineOptions:
    artist: str
    album: str
    audio_path: Path
    output_dir: Path
    draft: bool = False
    skip_canvas: bool = False
    skip_visualizer: bool = False
    image_model: str = "black-forest-labs/flux-1.1-pro"


def run(opts: PipelineOptions) -> PipelineResult:
    """
    Execute the full artwork generation pipeline.

    All intermediate files are placed in a ``work/`` subdirectory and the
    final deliverables are copied to the output directory root.
    """
    t0 = time.perf_counter()

    # Sanitise output directory
    safe_name = _safe_slug(f"{opts.artist}_{opts.album}")
    out = opts.output_dir / safe_name
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:

        # ── Stage 1: Audio Analysis ───────────────────────────────────────────
        task = progress.add_task("Analysing audio …", total=None)
        features = analyse(opts.audio_path)
        progress.update(task, description=f"[green]✓ Audio analysed  ({features.duration:.0f}s, {features.bpm:.0f} BPM, {features.key})")

        # ── Stage 2: Claude creative direction ────────────────────────────────
        task2 = progress.add_task("Generating creative direction with Claude …", total=None)
        client = anthropic.Anthropic()
        direction = generate_direction(features, opts.artist, opts.album, client=client)
        progress.update(task2, description=f"[green]✓ Direction: {direction.art_style} · {direction.cassette_era} · {direction.cassette_brand_name}")

        # ── Stage 3a: Label art generation ────────────────────────────────────
        task3 = progress.add_task("Generating label artwork (Flux) …", total=None)
        label_art_path = work / "label_art.png"
        image_generator.generate_label_art(
            direction,
            label_art_path,
            model=opts.image_model,
            draft=opts.draft,
        )
        progress.update(task3, description="[green]✓ Label artwork generated")

        # ── Stage 3b: Canvas background ───────────────────────────────────────
        if not opts.skip_canvas:
            task3b = progress.add_task("Generating canvas background …", total=None)
            canvas_bg_path = work / "canvas_bg.png"
            image_generator.generate_canvas_background(
                direction,
                canvas_bg_path,
                model=opts.image_model,
                draft=opts.draft,
            )
            progress.update(task3b, description="[green]✓ Canvas background generated")

        # ── Stage 4: Cassette composition ─────────────────────────────────────
        task4 = progress.add_task("Compositing cassette artwork …", total=None)
        cassette_a_path = work / "cassette_A.png"
        cassette_b_path = work / "cassette_B.png"
        cassette.compose(label_art_path, direction, opts.artist, opts.album, cassette_a_path, side="A")
        cassette.compose(label_art_path, direction, opts.artist, opts.album, cassette_b_path, side="B")
        progress.update(task4, description="[green]✓ Cassette Side A + B rendered")

        # Copy to output root
        shutil.copy(cassette_a_path, out / "cassette_side_A.png")
        shutil.copy(cassette_b_path, out / "cassette_side_B.png")

        # ── Stage 5: Spotify Canvas ───────────────────────────────────────────
        canvas_output = out / "spotify_canvas.mp4"
        if not opts.skip_canvas:
            task5 = progress.add_task("Rendering Spotify Canvas …", total=None)
            canvas_gen.generate(
                bg_image_path=canvas_bg_path,
                cassette_art_path=cassette_a_path,
                direction=direction,
                features=features,
                artist=opts.artist,
                album=opts.album,
                output_path=canvas_output,
                draft=opts.draft,
            )
            progress.update(task5, description="[green]✓ Spotify Canvas rendered (8s loop)")

        # ── Stage 6: YouTube Visualiser ───────────────────────────────────────
        viz_output = out / "youtube_visualizer.mp4"
        if not opts.skip_visualizer:
            task6 = progress.add_task("Rendering YouTube visualiser …", total=None)
            viz_gen.generate(
                cassette_art_path=cassette_a_path,
                audio_path=opts.audio_path,
                direction=direction,
                features=features,
                artist=opts.artist,
                album=opts.album,
                output_path=viz_output,
                draft=opts.draft,
            )
            progress.update(task6, description="[green]✓ YouTube visualiser rendered")

    elapsed = time.perf_counter() - t0

    console.print(f"\n[bold green]✓ Pipeline complete in {elapsed:.1f}s[/bold green]")
    console.print(f"  Output: [cyan]{out}[/cyan]")

    return PipelineResult(
        cassette_side_a=out / "cassette_side_A.png",
        cassette_side_b=out / "cassette_side_B.png",
        spotify_canvas=canvas_output,
        youtube_visualizer=viz_output,
        output_dir=out,
        features=features,
        direction=direction,
        elapsed_seconds=elapsed,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_slug(text: str) -> str:
    """Turn a string into a safe directory name."""
    import re
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text).strip("_")[:80]

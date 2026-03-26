"""
Main orchestration pipeline.

Wires together every module in the correct order and exposes a single
:func:`run` function consumed by the CLI and the Flask web UI.

Pipeline stages
───────────────
1.  Audio analysis          -> AudioFeatures
2.  Style analysis          -> StyleAnalysis   (Claude vision on reference covers)
3.  Creative direction      -> CreativeDirection (Claude)
4.  Album art generation    -> album_cover.png  (Replicate Flux 1.1 Pro)
5.  Spotify Canvas          -> spotify_canvas.mp4 (Runway Gen-3)
6.  YouTube Visualiser      -> youtube_visualizer.mp4 (procedural)
7.  Export package          -> <artist>_<album>/
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from artwork_machine.audio.analyzer import analyse, AudioFeatures
from artwork_machine.ai.style_analyzer import analyse_reference_covers, StyleAnalysis
from artwork_machine.ai.prompt_generator import generate as generate_direction, CreativeDirection
from artwork_machine.ai import image_generator
from artwork_machine.video import canvas as canvas_gen
from artwork_machine.video import visualizer as viz_gen

console = Console()


@dataclass
class PipelineResult:
    album_cover: Path
    spotify_canvas: Path
    youtube_visualizer: Path
    output_dir: Path
    features: AudioFeatures
    direction: CreativeDirection
    style: StyleAnalysis | None
    elapsed_seconds: float = 0.0


@dataclass
class PipelineOptions:
    artist: str
    album: str
    audio_path: Path
    output_dir: Path
    reference_cover_paths: list[Path] = field(default_factory=list)
    draft: bool = False
    skip_canvas: bool = False
    skip_visualizer: bool = False
    # Optional callback for real-time status updates (used by Flask UI)
    progress_callback: Callable[[str, str], None] | None = None


def run(opts: PipelineOptions) -> PipelineResult:
    """
    Execute the full artwork generation pipeline.

    All intermediate files land in a ``work/`` subdirectory; final
    deliverables are copied to the output directory root.
    """
    t0 = time.perf_counter()

    def _status(stage: str, msg: str) -> None:
        if opts.progress_callback:
            opts.progress_callback(stage, msg)

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
        _status("analyzing_audio", "Analysing audio…")
        task = progress.add_task("Analysing audio…", total=None)
        features = analyse(opts.audio_path)
        progress.update(
            task,
            description=f"[green]✓ Audio: {features.duration:.0f}s · {features.bpm:.0f} BPM · {features.key}",
        )

        # ── Stage 2: Style Analysis ───────────────────────────────────────────
        style: StyleAnalysis | None = None
        if opts.reference_cover_paths:
            _status("analyzing_style", f"Analysing {len(opts.reference_cover_paths)} reference cover(s)…")
            task2 = progress.add_task(
                f"Analysing {len(opts.reference_cover_paths)} reference cover(s)…", total=None
            )
            client = anthropic.Anthropic()
            style = analyse_reference_covers(opts.reference_cover_paths, client=client)
            progress.update(task2, description=f"[green]✓ Style: {style.art_style} · {style.mood}")
        else:
            client = anthropic.Anthropic()

        # ── Stage 3: Creative Direction ───────────────────────────────────────
        _status("generating_direction", "Generating creative direction with Claude…")
        task3 = progress.add_task("Generating creative direction…", total=None)
        direction = generate_direction(features, opts.artist, opts.album, style=style, client=client)
        progress.update(task3, description=f"[green]✓ Direction: {direction.art_style} · {direction.canvas_motion_style}")

        # ── Stage 4: Album Art Generation ─────────────────────────────────────
        _status("generating_cover", "Generating album art…")
        task4 = progress.add_task("Generating album art (Flux 1.1 Pro)…", total=None)
        cover_path = out / "album_cover.png"
        image_generator.generate_album_art(direction, cover_path, draft=opts.draft)
        progress.update(task4, description="[green]✓ Album art generated (3000×3000)")

        # ── Stage 5: Spotify Canvas ───────────────────────────────────────────
        canvas_output = out / "spotify_canvas.mp4"
        if not opts.skip_canvas:
            _status("animating_canvas", "Animating canvas (Runway Gen-3)…")
            task5 = progress.add_task("Animating Spotify Canvas (Runway Gen-3)…", total=None)
            canvas_gen.generate(
                album_cover_path=cover_path,
                direction=direction,
                features=features,
                output_path=canvas_output,
                draft=opts.draft,
            )
            progress.update(task5, description="[green]✓ Spotify Canvas rendered (8s seamless loop)")

        # ── Stage 6: YouTube Visualiser ───────────────────────────────────────
        viz_output = out / "youtube_visualizer.mp4"
        if not opts.skip_visualizer:
            _status("rendering_visualizer", "Rendering YouTube visualiser…")
            task6 = progress.add_task("Rendering YouTube visualiser…", total=None)
            viz_gen.generate(
                album_cover_path=cover_path,
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
    _status("done", f"Complete in {elapsed:.1f}s")

    console.print(f"\n[bold green]✓ Pipeline complete in {elapsed:.1f}s[/bold green]")
    console.print(f"  Output: [cyan]{out}[/cyan]")

    return PipelineResult(
        album_cover=cover_path,
        spotify_canvas=canvas_output,
        youtube_visualizer=viz_output,
        output_dir=out,
        features=features,
        direction=direction,
        style=style,
        elapsed_seconds=elapsed,
    )


def _safe_slug(text: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text).strip("_")[:80]

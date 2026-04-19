"""
artwork-machine CLI
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()


@click.group()
@click.version_option("0.2.0", prog_name="artwork-machine")
def main() -> None:
    """Automated album art, Spotify Canvas, YouTube thumbnail + visualizer generator."""


@main.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("--artist", "-a", required=True, help="Artist name")
@click.option("--album",  "-l", required=True, help="Album / track title")
@click.option("--output", "-o", default="./output", show_default=True,
              help="Output directory", type=click.Path(path_type=Path))
@click.option("--model",  "-m", default="black-forest-labs/FLUX.1-schnell",
              show_default=True, help="Hugging Face image model ID")
@click.option("--draft", is_flag=True, default=False, help="Draft quality (fast, low-res)")
@click.option("--no-canvas",     is_flag=True, default=False, help="Skip Spotify Canvas")
@click.option("--no-short",      is_flag=True, default=False, help="Skip 30-second short")
@click.option("--no-visualizer", is_flag=True, default=False, help="Skip YouTube visualizer")
@click.option("--youtube-upload", is_flag=True, default=False,
              help="Upload visualizer to YouTube after generation")
def generate(
    audio_file: Path, artist: str, album: str, output: Path,
    model: str, draft: bool, no_canvas: bool, no_short: bool, no_visualizer: bool,
    youtube_upload: bool,
) -> None:
    """Run the full artwork generation pipeline for AUDIO_FILE."""
    from artwork_machine.pipeline import PipelineOptions, run

    _validate_env()

    if draft:
        console.print("[yellow]⚡ Draft mode — faster, lower resolution[/yellow]")

    opts = PipelineOptions(
        artist=artist, album=album, audio_path=audio_file, output_dir=output,
        draft=draft, skip_canvas=no_canvas, skip_short=no_short,
        skip_visualizer=no_visualizer, youtube_upload=youtube_upload,
        image_model=model,
    )

    try:
        result = run(opts)
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    table = Table(title="Generated Deliverables", show_header=True, header_style="bold cyan")
    table.add_column("File", style="white")
    table.add_column("Path", style="green")

    deliverables = [
        ("Album Art (3000×3000)",    result.album_art),
        ("YouTube Thumbnail",        result.thumbnail),
        ("Spotify Canvas",           result.spotify_canvas),
        ("30-second Short",          result.short_video),
        ("YouTube Visualizer",       result.youtube_visualizer),
    ]
    for label, path in deliverables:
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            table.add_row(label, f"{path}  ({size_mb:.1f} MB)")

    console.print(table)
    console.print(f"\n[bold]Creative direction:[/bold]")
    console.print(f"  Style:    {result.direction.art_style}")
    console.print(f"  Palette:  {result.direction.palette_primary} · {result.direction.palette_secondary} · {result.direction.palette_accent}")
    console.print(f"  Motion:   {result.direction.canvas_motion_style}")
    console.print(f"  Tagline:  \"{result.direction.tagline}\"")


@main.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("--json-out", is_flag=True, help="Output JSON instead of table")
def analyse(audio_file: Path, json_out: bool) -> None:
    """Analyse AUDIO_FILE and display its musical features."""
    from artwork_machine.audio.analyzer import analyse as do_analyse

    console.print(f"Analysing [cyan]{audio_file}[/cyan] …")
    try:
        features = do_analyse(audio_file)
    except Exception as exc:
        console.print(f"[red]Analysis failed:[/red] {exc}")
        raise SystemExit(1) from exc

    if json_out:
        click.echo(json.dumps({
            "duration": features.duration, "bpm": features.bpm, "key": features.key,
            "mood_tags": features.mood_tags, "genre_hints": features.genre_hints,
            "rms_mean": features.rms_mean, "spectral_centroid_mean": features.spectral_centroid_mean,
            "dynamic_range_db": features.dynamic_range_db,
        }, indent=2))
        return

    table = Table(title=str(audio_file.name), show_header=True, header_style="bold cyan")
    table.add_column("Feature", style="white")
    table.add_column("Value", style="green")
    for row in [
        ("Duration", f"{features.duration:.1f}s"),
        ("BPM", f"{features.bpm:.1f}"),
        ("Key", features.key),
        ("Mood", ", ".join(features.mood_tags)),
        ("Genre hints", ", ".join(features.genre_hints) or "—"),
        ("Energy (RMS)", f"{features.rms_mean:.4f}"),
        ("Brightness", f"{features.spectral_centroid_mean:.0f} Hz"),
        ("Dynamic range", f"{features.dynamic_range_db:.1f} dB"),
        ("Beat count", str(len(features.beat_times))),
    ]:
        table.add_row(*row)
    console.print(table)


def _validate_env() -> None:
    import os
    missing = [k for k in ("ANTHROPIC_API_KEY", "HF_TOKEN") if not os.getenv(k)]
    if missing:
        console.print(
            f"[bold red]Missing environment variables:[/bold red] {', '.join(missing)}\n"
            "Add them to your .env file."
        )
        raise SystemExit(1)

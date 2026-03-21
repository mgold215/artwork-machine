"""
artwork-machine CLI

Usage examples:

  # Full pipeline (production quality)
  artwork-machine generate song.mp3 --artist "Neon Drift" --album "Glass Veins"

  # Fast draft render (skips full resolution, fewer AI steps)
  artwork-machine generate song.flac --artist "The Hollow" --album "Dust" --draft

  # Skip the visualiser (faster, great for singles)
  artwork-machine generate track.wav --artist "Solaris" --album "Orbit" --no-visualizer

  # Use a different image generation model
  artwork-machine generate song.mp3 --artist "X" --album "Y" \\
      --model stability-ai/sdxl:latest

  # Analyse audio only (no generation)
  artwork-machine analyse song.mp3
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
@click.version_option("0.1.0", prog_name="artwork-machine")
def main() -> None:
    """Automated cassette art, Spotify Canvas, and YouTube visualizer generator."""


@main.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("--artist", "-a", required=True, help="Artist name")
@click.option("--album", "-l", required=True, help="Album / track title")
@click.option(
    "--output", "-o",
    default="./output",
    show_default=True,
    help="Output directory",
    type=click.Path(path_type=Path),
)
@click.option(
    "--model", "-m",
    default="black-forest-labs/FLUX.1-schnell",
    show_default=True,
    help="Hugging Face model ID for image generation",
)
@click.option("--draft", is_flag=True, default=False, help="Draft quality (fast, low-res)")
@click.option("--no-canvas", is_flag=True, default=False, help="Skip Spotify Canvas video")
@click.option("--no-visualizer", is_flag=True, default=False, help="Skip YouTube visualizer")
def generate(
    audio_file: Path,
    artist: str,
    album: str,
    output: Path,
    model: str,
    draft: bool,
    no_canvas: bool,
    no_visualizer: bool,
) -> None:
    """
    Run the full artwork generation pipeline for AUDIO_FILE.

    Requires ANTHROPIC_API_KEY and REPLICATE_API_TOKEN in environment or .env.
    """
    from artwork_machine.pipeline import PipelineOptions, run

    _validate_env()

    if draft:
        console.print("[yellow]⚡ Draft mode — lower resolution, faster generation[/yellow]")

    opts = PipelineOptions(
        artist=artist,
        album=album,
        audio_path=audio_file,
        output_dir=output,
        draft=draft,
        skip_canvas=no_canvas,
        skip_visualizer=no_visualizer,
        image_model=model,
    )

    try:
        result = run(opts)
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    # ── Print summary ─────────────────────────────────────────────────────────
    table = Table(title="Generated Deliverables", show_header=True, header_style="bold cyan")
    table.add_column("File", style="white")
    table.add_column("Path", style="green")

    deliverables = [
        ("Cassette Side A", result.cassette_side_a),
        ("Cassette Side B", result.cassette_side_b),
        ("Spotify Canvas", result.spotify_canvas),
        ("YouTube Visualizer", result.youtube_visualizer),
    ]
    for label, path in deliverables:
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            table.add_row(label, f"{path}  ({size_mb:.1f} MB)")

    console.print(table)
    console.print(f"\n[bold]Creative direction:[/bold]")
    console.print(f"  Style:   {result.direction.art_style}")
    console.print(f"  Palette: {result.direction.palette_primary} · {result.direction.palette_secondary} · {result.direction.palette_accent}")
    console.print(f"  Motion:  {result.direction.canvas_motion_style}")
    console.print(f"  Brand:   {result.direction.cassette_brand_name} ({result.direction.cassette_era})")
    console.print(f"  Tagline: \"{result.direction.label_tagline}\"")


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
        data = {
            "duration": features.duration,
            "bpm": features.bpm,
            "key": features.key,
            "mood_tags": features.mood_tags,
            "genre_hints": features.genre_hints,
            "rms_mean": features.rms_mean,
            "spectral_centroid_mean": features.spectral_centroid_mean,
            "dynamic_range_db": features.dynamic_range_db,
        }
        click.echo(json.dumps(data, indent=2))
        return

    table = Table(title=str(audio_file.name), show_header=True, header_style="bold cyan")
    table.add_column("Feature", style="white")
    table.add_column("Value", style="green")

    rows = [
        ("Duration", f"{features.duration:.1f}s"),
        ("BPM", f"{features.bpm:.1f}"),
        ("Key", features.key),
        ("Mood", ", ".join(features.mood_tags)),
        ("Genre hints", ", ".join(features.genre_hints) or "—"),
        ("Energy (RMS)", f"{features.rms_mean:.4f}"),
        ("Brightness (centroid)", f"{features.spectral_centroid_mean:.0f} Hz"),
        ("Dynamic range", f"{features.dynamic_range_db:.1f} dB"),
        ("Beat count", str(len(features.beat_times))),
    ]
    for feature, value in rows:
        table.add_row(feature, value)

    console.print(table)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_env() -> None:
    """Check required API keys are set."""
    import os
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.getenv("HF_TOKEN"):
        missing.append("HF_TOKEN")

    if missing:
        console.print(
            f"[bold red]Missing environment variables:[/bold red] {', '.join(missing)}\n"
            "Copy [cyan].env.example[/cyan] → [cyan].env[/cyan] and add your keys.\n"
            "Get a free HF_TOKEN at: huggingface.co → Settings → Access Tokens"
        )
        raise SystemExit(1)

"""
Demo: render a canvas preview video without any API keys.

Generates 3 synthetic parallax layers (aerial gradient, cassette mid,
macro texture) and a placeholder cassette image, then runs the full
drone-parallax canvas renderer to produce an 8-second MP4 loop.

Usage:
    python scripts/demo_canvas.py [--style STYLE] [--output OUTPUT]

Styles:
    liquid bloom (default) | glitch pulse | cinematic drift |
    hypnotic swirl | electric surge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Make sure the package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from artwork_machine.ai.prompt_generator import CreativeDirection
from artwork_machine.audio.analyzer import AudioFeatures
from artwork_machine.video.canvas import generate


# ── Synthetic layer factories ──────────────────────────────────────────────────

def _make_far_layer(w: int = 810, h: int = 1440) -> Image.Image:
    """
    Aerial background: deep indigo-to-teal gradient with subtle cloud wisps.
    Represents a drone shot of city lights / ocean at dusk.
    """
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        # Dark indigo at top → deep teal at bottom
        arr[y, :, 0] = int(10  + t * 30)
        arr[y, :, 1] = int(10  + t * 60)
        arr[y, :, 2] = int(80  + t * 120)

    img = Image.fromarray(arr, "RGB").convert("RGBA")

    # Add some soft horizontal light bands (city grid abstraction)
    draw = ImageDraw.Draw(img)
    rng  = np.random.default_rng(42)
    for _ in range(18):
        y_pos  = int(rng.uniform(0, h))
        bright = int(rng.uniform(60, 140))
        alpha  = int(rng.uniform(30, 80))
        draw.line([(0, y_pos), (w, y_pos)], fill=(bright, bright + 20, bright + 40, alpha), width=2)

    return img.filter(ImageFilter.GaussianBlur(radius=4))


def _make_mid_layer(w: int = 810, h: int = 1440) -> Image.Image:
    """
    Mid layer: overhead cassette / unspooled tape scene.
    Warm amber ribbons on a dark surface.
    """
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    # Soft dark surface
    arr[:, :, 0] = 18
    arr[:, :, 1] = 12
    arr[:, :, 2] = 24
    arr[:, :, 3] = 200   # mostly opaque, slight transparency

    img = Image.fromarray(arr, "RGBA")
    draw = ImageDraw.Draw(img)

    rng = np.random.default_rng(7)

    # Draw sinuous tape ribbons across the surface
    for ribbon in range(8):
        points = []
        x = int(rng.uniform(0, w))
        for seg in range(60):
            y = int(seg / 60 * h)
            x = int(x + rng.uniform(-30, 30))
            x = max(0, min(w, x))
            points.append((x, y))

        for i in range(len(points) - 1):
            r = int(180 + rng.uniform(-20, 20))
            g = int(120 + rng.uniform(-20, 20))
            b = int(40  + rng.uniform(-10, 10))
            draw.line([points[i], points[i + 1]], fill=(r, g, b, 220), width=3)

    # Cassette shell silhouette, top-left corner
    draw.rectangle([(30, 40), (220, 140)], fill=(60, 50, 40, 230), outline=(200, 160, 80, 255), width=2)
    draw.ellipse([(60, 65), (110, 115)], fill=(30, 25, 20, 230), outline=(160, 130, 70, 255), width=1)
    draw.ellipse([(130, 65), (180, 115)], fill=(30, 25, 20, 230), outline=(160, 130, 70, 255), width=1)

    return img.filter(ImageFilter.GaussianBlur(radius=1))


def _make_near_layer(w: int = 810, h: int = 1440) -> Image.Image:
    """
    Near layer: extreme macro tape texture.
    Brown ferric oxide surface, fine horizontal striations, reflective edge.
    """
    arr = np.zeros((h, w, 4), dtype=np.uint8)

    rng = np.random.default_rng(13)

    # Base: warm brown ferric oxide
    noise = rng.integers(0, 25, (h, w), dtype=np.uint8)
    arr[:, :, 0] = np.clip(110 + noise, 0, 255)
    arr[:, :, 1] = np.clip(60  + noise // 2, 0, 255)
    arr[:, :, 2] = np.clip(20  + noise // 4, 0, 255)
    arr[:, :, 3] = 140   # semi-transparent — reveals far + mid through it

    img = Image.fromarray(arr, "RGBA")
    draw = ImageDraw.Draw(img)

    # Fine horizontal striations (tape grain)
    for y in range(0, h, 4):
        alpha = int(rng.uniform(20, 60))
        bright = int(rng.uniform(140, 200))
        draw.line([(0, y), (w, y)], fill=(bright, bright // 2, bright // 4, alpha), width=1)

    # Bright reflective sheen band diagonally across frame
    for offset in range(0, 80, 2):
        ya = max(0, h // 3 - 40 + offset)
        yb = min(h, h // 3 + 40 + offset)
        alpha = int(80 * (1 - abs(offset - 40) / 40))
        draw.line([(0, ya), (w, yb)], fill=(220, 170, 100, alpha), width=2)

    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def _make_cassette_art(w: int = 400, h: int = 400) -> Image.Image:
    """Simple placeholder cassette artwork."""
    img  = Image.new("RGBA", (w, h), (25, 20, 40, 255))
    draw = ImageDraw.Draw(img)

    # Cassette body
    draw.rounded_rectangle([(20, 60), (w - 20, h - 60)], radius=12,
                            fill=(55, 45, 70, 255), outline=(180, 140, 80, 255), width=3)
    # Reels
    for cx in [w // 3, 2 * w // 3]:
        draw.ellipse([(cx - 55, h // 2 - 55), (cx + 55, h // 2 + 55)],
                     fill=(30, 25, 35, 255), outline=(160, 120, 60, 255), width=2)
        draw.ellipse([(cx - 25, h // 2 - 25), (cx + 25, h // 2 + 25)],
                     fill=(50, 40, 60, 255), outline=(140, 110, 55, 255), width=1)
    # Label
    draw.rectangle([(60, 90), (w - 60, h // 2 - 20)],
                   fill=(200, 60, 80, 255), outline=(255, 200, 80, 255), width=2)
    # Tape window
    draw.rectangle([(w // 4, h // 2 + 20), (3 * w // 4, h - 80)],
                   fill=(15, 12, 20, 255), outline=(100, 80, 50, 255), width=2)

    return img


# ── Direction / features stubs ────────────────────────────────────────────────

def _direction(style: str) -> CreativeDirection:
    return CreativeDirection(
        image_prompt="demo",
        negative_prompt="blur, artefacts",
        art_style="hyperrealistic macro CGI",
        canvas_far_prompt="demo far",
        canvas_mid_prompt="demo mid",
        canvas_near_prompt="demo near",
        palette_primary="#1a0a2e",
        palette_secondary="#16213e",
        palette_accent="#e94560",
        palette_background="#0f3460",
        cassette_shell_colour="#c8a96e",
        cassette_era="80s",
        cassette_brand_name="Ferrox",
        canvas_motion_style=style,
        visualiser_style="spectrum bars",
        label_font_style="condensed sans-serif industrial",
        label_tagline="Magnetic dreams unfurl",
    )


def _features() -> AudioFeatures:
    # Synthesise a simple 0→1→0 energy curve that mimics a track with a build
    n = 240
    t  = np.linspace(0, 1, n)
    curve = (np.sin(np.pi * t) * 0.6 + 0.2 + np.random.default_rng(0).uniform(-0.05, 0.05, n)).tolist()
    return AudioFeatures(
        duration=180.0,
        sample_rate=44100,
        bpm=128.0,
        beat_frames=list(range(0, 44100 * 8, 44100 * 60 // 128)),
        beat_times=[i * 60 / 128 for i in range(17)],
        key="A minor",
        chroma_mean=[0.08] * 12,
        rms_mean=0.07,
        rms_curve=curve,
        dynamic_range_db=14.0,
        spectral_centroid_mean=3200.0,
        spectral_rolloff_mean=6400.0,
        zero_crossing_rate=0.06,
        mfcc_mean=[0.0] * 13,
        genre_hints=["electronic", "synthwave"],
        mood_tags=["hypnotic", "cinematic"],
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Render a demo canvas video")
    parser.add_argument("--style", default="slow liquid bloom",
                        help="Motion style (default: 'slow liquid bloom')")
    parser.add_argument("--output", default="/tmp/demo_canvas.mp4",
                        help="Output MP4 path (default: /tmp/demo_canvas.mp4)")
    parser.add_argument("--draft", action="store_true",
                        help="Half-resolution, 15fps — much faster render")
    args = parser.parse_args()

    out = Path(args.output)
    tmp = out.parent / "_demo_layers"
    tmp.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic layers …")
    layers = {}
    for name, fn in [("far", _make_far_layer), ("mid", _make_mid_layer), ("near", _make_near_layer)]:
        p = tmp / f"layer_{name}.png"
        fn().save(str(p))
        layers[name] = p
        print(f"  {name:4s} → {p}")

    cass_path = tmp / "cassette.png"
    _make_cassette_art().save(str(cass_path))
    print(f"  cassette → {cass_path}")

    print(f"\nRendering 8-second canvas loop  (style: '{args.style}') …")
    generate(
        canvas_layers=layers,
        cassette_art_path=cass_path,
        direction=_direction(args.style),
        features=_features(),
        artist="Demo Artist",
        album="Demo Album",
        output_path=out,
        draft=args.draft,
    )

    print(f"\n✓ Done → {out}")
    print("  Play with:  mpv " + str(out) + "   or open it in any video player.")


if __name__ == "__main__":
    main()

"""
Photorealistic cassette tape compositor.

Builds a true-to-life cassette artwork image from:
  - An AI-generated label art image
  - A procedurally rendered cassette shell (plastic body, reels, window, screws)
  - Typography (artist, album, brand, side A/B)

Output is a 3000×3000 px PNG suitable for album cover, merch, and print.

Cassette anatomy (all measurements in a normalised 1000×600 unit grid):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Brand                   [LABEL ART]                        Side A  │
  │ ┌──────────────────────────────────────────────────────────────────┐│
  │ │ ╔════════╗   Artist Name                         ╔════════╗     ││
  │ │ ║ REEL L ║   Album Title                         ║ REEL R ║     ││
  │ │ ╚════════╝   • • • • • •                         ╚════════╝     ││
  │ └──────────────────────────────────────────────────────────────────┘│
  │  Tagline                                               60 min / TDK  │
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from artwork_machine.ai.prompt_generator import CreativeDirection
from artwork_machine.artwork.utils import hex_to_rgba, load_font, add_noise, add_grain


# ── Layout constants (final canvas = 3000×3000) ──────────────────────────────

CANVAS = 3000        # square canvas
SHELL_W = 2700       # cassette shell width
SHELL_H = 1620       # cassette shell height  (≈ C-60 aspect ratio 1.667:1 at 3:5)
SHELL_X = (CANVAS - SHELL_W) // 2
SHELL_Y = (CANVAS - SHELL_H) // 2

CORNER_R = 90        # shell corner radius
WINDOW_W = 980       # magnetic tape window width
WINDOW_H = 360       # magnetic tape window height
WINDOW_Y = SHELL_Y + (SHELL_H - WINDOW_H) // 2  # vertically centred

LABEL_PAD = 48       # padding inside shell edge before label
LABEL_X = SHELL_X + LABEL_PAD
LABEL_Y = SHELL_Y + LABEL_PAD
LABEL_W = SHELL_W - LABEL_PAD * 2
LABEL_H = SHELL_H - LABEL_PAD * 2

REEL_R = 260         # reel outer radius
REEL_HUB_R = 90      # central hub radius
REEL_SPOKE = 5       # number of spokes
REEL_L_CX = SHELL_X + 540
REEL_R_CX = SHELL_X + SHELL_W - 540
REEL_CY = SHELL_Y + SHELL_H // 2

SCREW_R = 28
SCREW_POSITIONS = [
    (SHELL_X + 120, SHELL_Y + 120),
    (SHELL_X + SHELL_W - 120, SHELL_Y + 120),
    (SHELL_X + 120, SHELL_Y + SHELL_H - 120),
    (SHELL_X + SHELL_W - 120, SHELL_Y + SHELL_H - 120),
]

GUIDE_POST_R = 18    # the small posts that guide tape through the window
GUIDE_L_X = SHELL_X + SHELL_W // 2 - WINDOW_W // 2 + 30
GUIDE_R_X = SHELL_X + SHELL_W // 2 + WINDOW_W // 2 - 30
GUIDE_Y = SHELL_Y + SHELL_H // 2


# ── Main compositor ────────────────────────────────────────────────────────────

def compose(
    label_art_path: Path,
    direction: CreativeDirection,
    artist: str,
    album: str,
    output_path: Path,
    *,
    side: str = "A",
) -> Path:
    """
    Compose the full cassette artwork.

    Parameters
    ----------
    label_art_path:
        1024×1024 PNG from the image generator.
    direction:
        Creative direction from Claude.
    artist, album:
        Metadata strings.
    output_path:
        Where to write the final PNG.
    side:
        "A" or "B" — printed on the label.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # 1. Background gradient
    _draw_background(canvas, direction)

    # 2. Cassette shell (plastic body)
    shell_colour = hex_to_rgba(direction.cassette_shell_colour, alpha=245)
    _draw_shell(draw, shell_colour, direction)

    # 3. Label art — fills the label area behind the window
    _draw_label(canvas, label_art_path, direction, artist, album, side)

    # 4. Tape window (dark cutout showing tape)
    _draw_window(draw, direction)

    # 5. Reels
    _draw_reel(draw, REEL_L_CX, REEL_CY, direction, fill_ratio=0.6)
    _draw_reel(draw, REEL_R_CX, REEL_CY, direction, fill_ratio=0.4)

    # 6. Guide posts
    _draw_guide_posts(draw, direction)

    # 7. Corner screws
    _draw_screws(draw, direction)

    # 8. Shell highlights / reflections
    _draw_shell_highlights(canvas, direction)

    # 9. Grain, dust, and subtle chromatic aberration for realism
    canvas = add_grain(canvas, intensity=0.04)
    canvas = _add_subtle_vignette(canvas)

    canvas.save(str(output_path), "PNG", optimize=False, compress_level=1)
    return output_path


# ── Drawing primitives ─────────────────────────────────────────────────────────

def _draw_background(canvas: Image.Image, direction: CreativeDirection) -> None:
    """Radial gradient background."""
    bg_colour = hex_to_rgba(direction.palette_background)
    sec_colour = hex_to_rgba(direction.palette_primary, alpha=120)

    bg = Image.new("RGBA", (CANVAS, CANVAS), bg_colour[:3] + (255,))
    # Vignette-style radial gradient overlay
    gradient = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    for r in range(CANVAS // 2, 0, -10):
        alpha = int(180 * (1 - r / (CANVAS // 2)))
        ellipse_box = [
            CANVAS // 2 - r, CANVAS // 2 - r,
            CANVAS // 2 + r, CANVAS // 2 + r,
        ]
        ImageDraw.Draw(gradient).ellipse(
            ellipse_box,
            fill=sec_colour[:3] + (alpha,),
        )

    canvas.alpha_composite(bg)
    canvas.alpha_composite(gradient)


def _draw_shell(
    draw: ImageDraw.ImageDraw,
    shell_colour: tuple,
    direction: CreativeDirection,
) -> None:
    """Draw the main plastic cassette body with rounded corners."""
    box = [SHELL_X, SHELL_Y, SHELL_X + SHELL_W, SHELL_Y + SHELL_H]

    # Shadow
    shadow_offset = 24
    shadow_box = [b + shadow_offset for b in box[:2]] + [b + shadow_offset for b in box[2:]]
    draw.rounded_rectangle(shadow_box, radius=CORNER_R, fill=(0, 0, 0, 120))

    # Shell body
    draw.rounded_rectangle(box, radius=CORNER_R, fill=shell_colour)

    # Inner bevel (slightly lighter top/left edge)
    bevel_col = _lighten(shell_colour, 40)
    bevel_box = [box[0] + 4, box[1] + 4, box[2] - 4, box[3] - 4]
    draw.rounded_rectangle(bevel_box, radius=CORNER_R - 4, outline=bevel_col, width=3)


def _draw_label(
    canvas: Image.Image,
    label_art_path: Path,
    direction: CreativeDirection,
    artist: str,
    album: str,
    side: str,
) -> None:
    """Composite the AI label art and typography onto the label area."""
    art = Image.open(str(label_art_path)).convert("RGBA")
    art = art.resize((LABEL_W, LABEL_H), Image.LANCZOS)

    # Paste label art
    canvas.alpha_composite(art, dest=(LABEL_X, LABEL_Y))

    # Gradient overlay so text is legible
    overlay = Image.new("RGBA", (LABEL_W, LABEL_H), (0, 0, 0, 0))
    ovl_draw = ImageDraw.Draw(overlay)
    # Top strip — semi-transparent for brand + side
    ovl_draw.rectangle([0, 0, LABEL_W, 110], fill=(0, 0, 0, 120))
    # Bottom strip — for tagline / info
    ovl_draw.rectangle([0, LABEL_H - 100, LABEL_W, LABEL_H], fill=(0, 0, 0, 120))
    canvas.alpha_composite(overlay, dest=(LABEL_X, LABEL_Y))

    # ── Typography ────────────────────────────────────────────────────────────
    draw = ImageDraw.Draw(canvas)

    primary_c = hex_to_rgba(direction.palette_accent)[:3]
    secondary_c = (255, 255, 255)

    # Brand name — top left
    brand_font = load_font(size=52, bold=True)
    draw.text(
        (LABEL_X + 40, LABEL_Y + 28),
        direction.cassette_brand_name.upper(),
        font=brand_font,
        fill=primary_c,
    )

    # Side indicator — top right
    side_font = load_font(size=64, bold=True)
    side_text = f"SIDE {side}"
    side_bbox = draw.textbbox((0, 0), side_text, font=side_font)
    side_w = side_bbox[2] - side_bbox[0]
    draw.text(
        (LABEL_X + LABEL_W - side_w - 40, LABEL_Y + 22),
        side_text,
        font=side_font,
        fill=primary_c,
    )

    # Artist name — vertically centred left
    artist_font = load_font(size=88, bold=True)
    draw.text(
        (LABEL_X + 60, LABEL_Y + LABEL_H // 2 - 90),
        artist,
        font=artist_font,
        fill=secondary_c,
    )

    # Album title
    album_font = load_font(size=60)
    draw.text(
        (LABEL_X + 60, LABEL_Y + LABEL_H // 2 + 20),
        album,
        font=album_font,
        fill=secondary_c,
    )

    # Tagline — bottom left
    tag_font = load_font(size=40)
    draw.text(
        (LABEL_X + 40, LABEL_Y + LABEL_H - 76),
        direction.label_tagline.upper(),
        font=tag_font,
        fill=(*primary_c, 200),
    )

    # Era + format info — bottom right
    era_text = f"C-60  ✦  {direction.cassette_era}  ✦  TYPE I"
    era_bbox = draw.textbbox((0, 0), era_text, font=tag_font)
    era_w = era_bbox[2] - era_bbox[0]
    draw.text(
        (LABEL_X + LABEL_W - era_w - 40, LABEL_Y + LABEL_H - 76),
        era_text,
        font=tag_font,
        fill=(200, 200, 200, 180),
    )


def _draw_window(draw: ImageDraw.ImageDraw, direction: CreativeDirection) -> None:
    """Draw the tape window cutout with leader tape visible."""
    cx = SHELL_X + SHELL_W // 2
    win_x = cx - WINDOW_W // 2
    win_y = WINDOW_Y
    win_box = [win_x, win_y, win_x + WINDOW_W, win_y + WINDOW_H]

    # Window opening (dark, showing tape)
    draw.rounded_rectangle(win_box, radius=24, fill=(10, 8, 6, 240))

    # Leader tape — thin horizontal dark-brown strip
    tape_y = SHELL_Y + SHELL_H // 2 - 8
    draw.rectangle([win_x + 20, tape_y, win_x + WINDOW_W - 20, tape_y + 16], fill=(40, 20, 10, 220))

    # Window rim / border (plastic bevel)
    draw.rounded_rectangle(win_box, radius=24, outline=(80, 70, 60, 200), width=6)

    # Subtle reflection across top of window
    reflect_box = [win_x + 12, win_y + 8, win_x + WINDOW_W - 12, win_y + 40]
    draw.rounded_rectangle(reflect_box, radius=10, fill=(255, 255, 255, 18))


def _draw_reel(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    direction: CreativeDirection,
    fill_ratio: float = 0.5,
) -> None:
    """
    Draw a single reel with hub, spokes, and tape wrap.

    Parameters
    ----------
    fill_ratio:
        0 = empty (beginning of tape), 1 = full.
    """
    shell_c = hex_to_rgba(direction.cassette_shell_colour)
    dark_c = (15, 12, 10, 255)
    tape_c = (40, 25, 12, 230)
    spoke_c = _lighten(shell_c, 20)
    hub_c = _darken(shell_c, 30)

    # Tape wrap (outermost ring, dark brown)
    tape_r = int(REEL_HUB_R + (REEL_R - REEL_HUB_R) * fill_ratio)
    _circle(draw, cx, cy, REEL_R, fill=(0, 0, 0, 80))         # outer shadow
    _circle(draw, cx, cy, tape_r, fill=tape_c)

    # Tape wrap edge highlight
    _circle(draw, cx, cy, tape_r, outline=(70, 45, 20, 180), width=3)

    # Hub
    _circle(draw, cx, cy, REEL_HUB_R, fill=hub_c)
    _circle(draw, cx, cy, REEL_HUB_R, outline=_lighten(hub_c, 25), width=2)

    # Spokes
    for i in range(REEL_SPOKE):
        angle = math.radians(i * 360 / REEL_SPOKE)
        sx = cx + int(math.cos(angle) * REEL_HUB_R)
        sy = cy + int(math.sin(angle) * REEL_HUB_R)
        ex = cx + int(math.cos(angle) * (tape_r - 8))
        ey = cy + int(math.sin(angle) * (tape_r - 8))
        draw.line([(sx, sy), (ex, ey)], fill=spoke_c, width=8)

    # Centre hole
    _circle(draw, cx, cy, 22, fill=dark_c)
    _circle(draw, cx, cy, 10, fill=(5, 5, 5, 255))

    # Subtle reel reflection
    _circle(draw, cx - 30, cy - 30, 18, fill=(255, 255, 255, 25))


def _draw_guide_posts(draw: ImageDraw.ImageDraw, direction: CreativeDirection) -> None:
    """Tiny cylindrical posts that route tape through the window."""
    col = _darken(hex_to_rgba(direction.cassette_shell_colour), 40)
    for gx in [GUIDE_L_X, GUIDE_R_X]:
        _circle(draw, gx, GUIDE_Y, GUIDE_POST_R, fill=col)
        _circle(draw, gx, GUIDE_Y, GUIDE_POST_R, outline=_lighten(col, 30), width=2)
        # Highlight dot
        _circle(draw, gx - 4, GUIDE_Y - 4, 4, fill=(255, 255, 255, 80))


def _draw_screws(draw: ImageDraw.ImageDraw, direction: CreativeDirection) -> None:
    """Phillips head screws at the four corners."""
    col = _darken(hex_to_rgba(direction.cassette_shell_colour), 50)
    highlight = _lighten(col, 40)

    for sx, sy in SCREW_POSITIONS:
        _circle(draw, sx, sy, SCREW_R, fill=col)
        _circle(draw, sx, sy, SCREW_R, outline=highlight, width=2)
        # Phillips cross
        cross_len = SCREW_R - 6
        draw.line([(sx - cross_len, sy), (sx + cross_len, sy)], fill=highlight, width=4)
        draw.line([(sx, sy - cross_len), (sx, sy + cross_len)], fill=highlight, width=4)
        _circle(draw, sx - 2, sy - 2, 4, fill=(255, 255, 255, 60))


def _draw_shell_highlights(canvas: Image.Image, direction: CreativeDirection) -> None:
    """Add a subtle specular highlight along the top edge of the shell."""
    highlight = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    h_draw = ImageDraw.Draw(highlight)

    # Top-edge highlight strip
    h_draw.rounded_rectangle(
        [SHELL_X + 20, SHELL_Y + 8, SHELL_X + SHELL_W - 20, SHELL_Y + 60],
        radius=40,
        fill=(255, 255, 255, 40),
    )
    # Bottom-edge shadow
    h_draw.rounded_rectangle(
        [SHELL_X + 40, SHELL_Y + SHELL_H - 50, SHELL_X + SHELL_W - 40, SHELL_Y + SHELL_H - 10],
        radius=40,
        fill=(0, 0, 0, 60),
    )

    highlight = highlight.filter(ImageFilter.GaussianBlur(radius=12))
    canvas.alpha_composite(highlight)


def _add_subtle_vignette(canvas: Image.Image) -> Image.Image:
    vignette = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette)
    steps = 30
    for i in range(steps):
        r = int(CANVAS // 2 * (1 - i / steps))
        alpha = int(80 * (i / steps) ** 2)
        vd.ellipse(
            [CANVAS // 2 - r, CANVAS // 2 - r, CANVAS // 2 + r, CANVAS // 2 + r],
            fill=(0, 0, 0, alpha),
        )
    canvas.alpha_composite(vignette)
    return canvas


# ── Small helpers ──────────────────────────────────────────────────────────────

def _circle(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    r: int,
    fill=None,
    outline=None,
    width: int = 1,
) -> None:
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=fill,
        outline=outline,
        width=width,
    )


def _lighten(colour: tuple, amount: int) -> tuple:
    return tuple(min(255, c + amount) for c in colour[:3]) + (colour[3],)  # type: ignore


def _darken(colour: tuple, amount: int) -> tuple:
    return tuple(max(0, c - amount) for c in colour[:3]) + (colour[3],)  # type: ignore

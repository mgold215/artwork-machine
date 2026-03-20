# artwork-machine

**Fully automated generator for photorealistic cassette album art, Spotify Canvas videos, and YouTube visualizers.**

Drop in an audio file, point it at your API keys, and the machine handles everything — from audio analysis through AI-driven creative direction to final platform-ready exports.

---

## What it generates

| Deliverable | Spec | Platform |
|---|---|---|
| **Cassette Side A + B** | 3000×3000 PNG, 300 DPI, print-ready | Album stores, merch, social |
| **Spotify Canvas** | 720×1280 MP4, H.264, 8 s seamless loop | Spotify for Artists |
| **YouTube Visualizer** | 1920×1080 MP4, H.264 + AAC 320k, full-length | YouTube |

---

## Pipeline architecture

```
Audio file (MP3/WAV/FLAC/…)
        │
        ▼
┌───────────────────────┐
│   Audio Analyser      │  librosa — BPM, key, mood, energy,
│   (audio/analyzer.py) │  spectrum frames, waveform
└───────────┬───────────┘
            │ AudioFeatures
            ▼
┌───────────────────────┐
│   Creative Director   │  Claude Opus 4.6 — converts musical features
│   (ai/prompt_         │  into a cohesive visual identity:
│    generator.py)      │  colour palette, art style, cassette era,
└───────────┬───────────┘  brand name, motion descriptor
            │ CreativeDirection
            ├──────────────────────────────────────────┐
            ▼                                          ▼
┌───────────────────────┐              ┌───────────────────────────┐
│   Image Generator     │              │   Image Generator         │
│   Label Art           │              │   Canvas Background       │
│   (ai/image_          │              │   (1024×1024 → 720×1280)  │
│    generator.py)      │              └──────────────┬────────────┘
│   Flux 1.1 Pro /      │                             │
│   SDXL via Replicate  │                             │
└───────────┬───────────┘                             │
            │ label_art.png                           │
            ▼                                          │
┌───────────────────────┐                             │
│   Cassette Compositor │◄────────────────────────────┘
│   (artwork/cassette   │  Procedurally rendered shell:
│    .py)               │  reels, screws, window, leader tape,
│                       │  typography, film grain
└──────┬────────────────┘
       │ cassette_side_A.png
       │ cassette_side_B.png
       ├─────────────────────────────────────────────────┐
       ▼                                                 ▼
┌──────────────────────┐                   ┌────────────────────────┐
│  Spotify Canvas      │                   │  YouTube Visualizer    │
│  (video/canvas.py)   │                   │  (video/visualizer.py) │
│                      │                   │                        │
│  Ken-Burns pan/zoom  │                   │  Spectrum bars OR      │
│  Particle system     │                   │  Radial waveform OR    │
│  Animated reel       │                   │  Particle field        │
│  Bloom + grade       │                   │  (style chosen by AI)  │
│  8 s seamless loop   │                   │  Full-length + audio   │
└──────────────────────┘                   └────────────────────────┘
```

---

## Installation

### Requirements

- Python 3.11+
- ffmpeg (`brew install ffmpeg` / `apt install ffmpeg`)
- API keys: [Anthropic](https://console.anthropic.com) + [Replicate](https://replicate.com)

```bash
git clone https://github.com/your-org/artwork-machine
cd artwork-machine
pip install -e .
cp .env.example .env
# Edit .env and add your API keys
```

---

## Usage

### Full pipeline

```bash
artwork-machine generate song.mp3 \
  --artist "Neon Drift" \
  --album  "Glass Veins"
```

Output lands in `./output/Neon_Drift_Glass_Veins/`.

### Draft mode (fast iteration)

```bash
artwork-machine generate song.mp3 \
  --artist "Neon Drift" \
  --album  "Glass Veins" \
  --draft
```

Half resolution, fewer AI steps — renders in a fraction of the time.

### Skip specific outputs

```bash
# Cassette + Canvas only (no visualizer)
artwork-machine generate song.mp3 -a "X" -l "Y" --no-visualizer

# Cassette only (no video)
artwork-machine generate song.mp3 -a "X" -l "Y" --no-canvas --no-visualizer
```

### Audio analysis only

```bash
artwork-machine analyse song.mp3

# JSON output (pipe-friendly)
artwork-machine analyse song.mp3 --json-out | jq .key
```

### Alternative image models

```bash
# SDXL (faster, slightly less detailed)
artwork-machine generate song.mp3 -a "X" -l "Y" \
  --model stability-ai/sdxl:latest

# Realistic Vision (photorealistic style)
artwork-machine generate song.mp3 -a "X" -l "Y" \
  --model lucataco/realistic-vision-v5:latest
```

---

## Output structure

```
output/
└── Artist_Album/
    ├── cassette_side_A.png      ← 3000×3000, print-ready
    ├── cassette_side_B.png
    ├── spotify_canvas.mp4       ← 720×1280, 8 s loop
    ├── youtube_visualizer.mp4   ← 1920×1080, full-length
    └── work/                    ← intermediate files
        ├── label_art.png
        ├── canvas_bg.png
        ├── cassette_A.png
        └── cassette_B.png
```

---

## Design decisions

### Why Claude for creative direction?

Audio features alone (BPM, key, timbre) are rich signals but they're numbers.
Claude translates them into a coherent *visual language* — matching the emotional
register of the music to specific colour palettes, art movements, and motion
aesthetics. It also generates the fictional cassette brand name and label tagline,
adding the kind of authentic detail that makes the output feel curated rather than
generated.

### Why Flux 1.1 Pro?

Flux consistently outperforms SDXL on photorealistic texture and colour fidelity —
exactly what you need for label art that has to survive printing on actual cassette
labels. SDXL is still available as a fallback for speed.

### Why CPU-rendered cassette shells?

AI-generated cassette images tend to hallucinate the wrong number of screws, wonky
reels, and impossible geometry. By procedurally compositing the shell in PIL we get:
- Pixel-perfect anatomy every time
- Full control over reel fill ratio, brand placement, era styling
- No prompt-engineering gymnastics

### Seamless Canvas loop

The Spotify Canvas loop uses a sine-based Ken-Burns pan/zoom whose period equals
exactly the video duration, so the first and last frame are identical. Particle
phases are also initialised to complete full cycles in 8 seconds.

### Three visualiser styles

Claude picks the style (`spectrum bars`, `radial waveform`, or `particle field`)
based on the music's mood. High-energy / bright music → bars. Atmospheric / minor
key → radial. Noise / complex texture → particles.

---

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude API key | required |
| `REPLICATE_API_TOKEN` | Replicate token | required |
| `IMAGE_MODEL` | Replicate model string | `black-forest-labs/flux-1.1-pro` |
| `OUTPUT_DIR` | Where to write deliverables | `./output` |
| `RENDER_QUALITY` | `production` or `draft` | `production` |
| `CANVAS_FPS` | Canvas video frame rate | `30` |
| `CANVAS_DURATION_SECONDS` | Canvas loop length | `8` |
| `VISUALIZER_FPS` | Visualizer frame rate | `60` |
| `VISUALIZER_WIDTH` | Visualizer width px | `1920` |
| `VISUALIZER_HEIGHT` | Visualizer height px | `1080` |

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use synthetically generated audio (sine waves) — no real audio files or API
keys required.

---

## Module reference

| Module | Responsibility |
|---|---|
| `audio/analyzer.py` | Librosa-based feature extraction |
| `ai/prompt_generator.py` | Claude creative direction via structured JSON |
| `ai/image_generator.py` | Replicate image generation with retry |
| `artwork/cassette.py` | Full cassette compositor (shell, reels, label, typography) |
| `artwork/utils.py` | Shared image utilities (colour, fonts, grain) |
| `video/canvas.py` | Spotify Canvas renderer |
| `video/visualizer.py` | YouTube visualizer renderer |
| `video/effects.py` | Reusable visual effects (particles, bloom, grade, vignette) |
| `export/packager.py` | Zip archive builder |
| `pipeline.py` | End-to-end orchestrator |
| `cli.py` | Click CLI (`generate`, `analyse`) |

# artwork-machine — Project Standards

## What This Is

A web app that takes a WAV track + reference album covers → generates:
1. **3000×3000 HD album cover** (Replicate Flux 1.1 Pro)
2. **Spotify Canvas** 720×1280 MP4, 8s seamless loop (Runway Gen-3)
3. **YouTube Visualizer** 1920×1080 MP4, full-length + audio (procedural)

The LLM (Claude) analyzes uploaded reference covers via vision to understand the user's taste, then combines that with audio features to generate a matching cover prompt.

---

## Architecture

```
WAV + reference covers
        │
        ▼
┌──────────────────┐
│ Audio Analysis   │  librosa — BPM, key, mood, energy, spectrum
└────────┬─────────┘
         │ AudioFeatures
         ▼
┌──────────────────┐
│ Style Analysis   │  Claude vision — analyzes reference covers
└────────┬─────────┘  (skipped if no refs; uses saved profile)
         │ StyleAnalysis
         ▼
┌──────────────────┐
│ Creative Dir.    │  Claude Opus 4.6 — fuses audio + style
└────────┬─────────┘  → CreativeDirection JSON
         │
    ┌────┴─────────────────────────┐
    ▼                              ▼
┌──────────────┐        ┌──────────────────────┐
│ Album Art    │        │ Canvas Animation     │
│ Replicate    │        │ Runway Gen-3 Alpha   │
│ Flux 1.1 Pro │        │ 720×1280, 8s loop    │
│ 3000×3000    │        └──────────────────────┘
└──────┬───────┘
       │ album_cover.png
       ▼
┌──────────────────────┐
│ YouTube Visualizer   │  Procedural, synced to audio
│ 1920×1080 full-len   │
└──────────────────────┘
```

---

## Key Files

| File | Role |
|------|------|
| `app.py` | Flask web UI — all routes, job management, profile management |
| `src/artwork_machine/pipeline.py` | Orchestrates all 6 stages |
| `src/artwork_machine/ai/style_analyzer.py` | Claude vision on reference covers |
| `src/artwork_machine/ai/prompt_generator.py` | Claude creative direction → `CreativeDirection` |
| `src/artwork_machine/ai/image_generator.py` | Replicate Flux API → album_cover.png |
| `src/artwork_machine/audio/analyzer.py` | librosa audio feature extraction (do not modify) |
| `src/artwork_machine/video/canvas.py` | Runway Gen-3 image-to-video API |
| `src/artwork_machine/video/visualizer.py` | Procedural YouTube visualizer |
| `src/artwork_machine/video/effects.py` | Reusable effects (do not modify) |
| `src/artwork_machine/config.py` | Pydantic settings from env |

---

## Environment Variables

```
ANTHROPIC_API_KEY=sk-ant-...          # Required — Claude
REPLICATE_API_TOKEN=r8_...            # Required — Flux image generation
RUNWAY_API_TOKEN=...                  # Required — Canvas animation
OUTPUT_DIR=./output                   # Where deliverables land
PROFILES_DIR=./profiles               # Saved style profiles
RENDER_QUALITY=production             # "production" or "draft"
CANVAS_FPS=30
CANVAS_DURATION_SECONDS=8
VISUALIZER_FPS=60
VISUALIZER_WIDTH=1920
VISUALIZER_HEIGHT=1080
PORT=7860
```

---

## Code Conventions

- Python 3.11+, type hints everywhere
- `from __future__ import annotations` at top of every module
- `@dataclass` for data structures (not Pydantic models in src/)
- Pydantic `BaseSettings` only in `config.py`
- No `print()` — use `rich.console.Console()` in pipeline, nothing in library code
- All paths as `pathlib.Path`, never raw strings
- External API calls wrapped in `tenacity.retry` with exponential backoff
- Draft mode: halve resolution, reduce inference steps, skip Runway (use Ken-Burns)
- `httpx` for all HTTP (not `requests`)
- `replicate` SDK for Flux (not raw HTTP)

---

## Output Structure

```
output/<Artist>_<Album>/
├── album_cover.png          ← 3000×3000 PNG, 300 DPI
├── spotify_canvas.mp4       ← 720×1280, 8s seamless loop
├── youtube_visualizer.mp4   ← 1920×1080, full-length + audio
└── work/
    ├── album_cover_1024.png ← Flux native output (before upscale)
    └── runway_raw.mp4       ← Runway 10s output (before trim/loop)
```

---

## Style Profiles

Stored in `profiles/` (configurable via `PROFILES_DIR`):

```
profiles/
├── <uuid>.json              # {"id", "name", "created_at", "style_analysis", "cover_count"}
└── covers/
    └── <uuid>_<n>.jpg       # Thumbnail copies of reference covers
```

The `style_analysis` field in the JSON is the cached `StyleAnalysis` — no need to re-run Claude vision on every generation if the profile hasn't changed.

---

## Runway Gen-3 API Notes

- Base URL: `https://api.dev.runwayml.com/v1`
- Headers: `Authorization: Bearer <token>`, `X-Runway-Version: 2024-11-06`
- `POST /image_to_video` with model `gen3a_turbo`, ratio `768:1280`, duration `10`
- Poll `GET /tasks/<id>` every 3s until `status == "SUCCEEDED"`
- Download MP4, trim to 8s, crossfade for seamless loop via ffmpeg
- In draft mode: skip Runway entirely, use procedural Ken-Burns fallback

---

## Replicate / Flux Notes

- Use `replicate` Python SDK (>= 0.25)
- Model: `black-forest-labs/flux-1.1-pro`
- Output at 1024×1024, then upscale to 3000×3000 via `Image.LANCZOS`
- Set `REPLICATE_API_TOKEN` env var — SDK reads it automatically

---

## Deployment

- Railway via Docker (`railway.toml` + `Dockerfile`)
- Start command: `gunicorn app:app --bind 0.0.0.0:8080 --workers 1 --timeout 600`
- `profiles/` and `output/` dirs must be writable (Railway ephemeral disk or mount)

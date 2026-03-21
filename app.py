"""
artwork-machine web UI

Run locally:  .venv/bin/python app.py
Then open:    http://localhost:7860
"""

import os
import sys
from pathlib import Path

print("Starting artwork-machine...", flush=True)

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

print("Importing pipeline...", flush=True)
from artwork_machine.pipeline import PipelineOptions, run as run_pipeline
print("Pipeline ready.", flush=True)


def generate(audio_file, artist, album, skip_canvas, skip_short, skip_visualizer, draft_mode, progress=gr.Progress()):
    if not audio_file:
        raise gr.Error("Please upload an audio file.")
    if not artist.strip():
        raise gr.Error("Artist name is required.")
    if not album.strip():
        raise gr.Error("Album / track title is required.")

    progress(0.05, desc="Starting…")

    opts = PipelineOptions(
        artist=artist.strip(),
        album=album.strip(),
        audio_path=Path(audio_file),
        output_dir=Path("./output"),
        draft=draft_mode,
        skip_canvas=skip_canvas,
        skip_short=skip_short,
        skip_visualizer=skip_visualizer,
    )

    try:
        result = run_pipeline(opts)
    except Exception as exc:
        raise gr.Error(f"Pipeline failed: {exc}") from exc

    progress(1.0, desc="Done.")

    album_art  = str(result.album_art)          if result.album_art.exists()          else None
    thumbnail  = str(result.thumbnail)          if result.thumbnail.exists()          else None
    canvas     = str(result.spotify_canvas)     if result.spotify_canvas.exists()     else None
    short      = str(result.short_video)        if result.short_video.exists()        else None
    visualizer = str(result.youtube_visualizer) if result.youtube_visualizer.exists() else None

    summary = (
        f"Style:    {result.direction.art_style}\n"
        f"Palette:  {result.direction.palette_primary}  ·  "
        f"{result.direction.palette_secondary}  ·  {result.direction.palette_accent}\n"
        f"Motion:   {result.direction.canvas_motion_style}\n"
        f"Tagline:  \"{result.direction.tagline}\""
    )

    return album_art, thumbnail, canvas, short, visualizer, summary


with gr.Blocks(title="artwork-machine") as app:

    gr.Markdown("# artwork-machine")
    gr.Markdown("Drop in a track. Get album art, thumbnail, Spotify Canvas, a short, and a YouTube visualizer.")

    with gr.Row(equal_height=False):

        # ── Inputs ────────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            audio_input  = gr.Audio(type="filepath", label="Audio file", sources=["upload"])
            artist_input = gr.Textbox(label="Artist name",        placeholder="e.g. moodmixformat")
            album_input  = gr.Textbox(label="Album / track title", placeholder="e.g. BLOOM")

            gr.Markdown("**Skip options** (check to speed things up)")
            with gr.Row():
                skip_canvas     = gr.Checkbox(label="Skip Canvas",      value=False)
                skip_short      = gr.Checkbox(label="Skip Short",        value=False)
                skip_visualizer = gr.Checkbox(label="Skip Visualizer",   value=True,
                                              info="Needs ~80 GB temp space")
            draft_mode = gr.Checkbox(label="Draft mode (faster, lower res)", value=False)

            generate_btn = gr.Button("Generate", variant="primary")

        # ── Outputs ───────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            album_art_out  = gr.Image(label="Album Art (3000×3000)")
            thumbnail_out  = gr.Image(label="YouTube Thumbnail (1280×720)")
            canvas_out     = gr.Video(label="Spotify Canvas (8s loop)")
            short_out      = gr.Video(label="30-second Short")
            visualizer_out = gr.Video(label="YouTube Visualizer (full length)")
            summary_out    = gr.Textbox(label="Creative direction", lines=4, interactive=False)

    generate_btn.click(
        fn=generate,
        inputs=[audio_input, artist_input, album_input,
                skip_canvas, skip_short, skip_visualizer, draft_mode],
        outputs=[album_art_out, thumbnail_out, canvas_out, short_out, visualizer_out, summary_out],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print(f"Launching on port {port}", flush=True)
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )

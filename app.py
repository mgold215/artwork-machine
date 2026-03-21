"""
artwork-machine web UI

Run locally:  .venv/bin/python app.py
Then open:    http://localhost:7860
"""

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from artwork_machine.pipeline import PipelineOptions, run as run_pipeline


def generate(audio_file, artist, album, skip_visualizer, draft_mode, progress=gr.Progress()):
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not audio_file:
        raise gr.Error("Please upload an audio file.")
    if not artist.strip():
        raise gr.Error("Artist name is required.")
    if not album.strip():
        raise gr.Error("Album / track title is required.")

    progress(0.05, desc="Starting pipeline…")

    opts = PipelineOptions(
        artist=artist.strip(),
        album=album.strip(),
        audio_path=Path(audio_file),
        output_dir=Path("./output"),
        draft=draft_mode,
        skip_canvas=False,
        skip_visualizer=skip_visualizer,
    )

    try:
        progress(0.10, desc="Analysing audio…")
        result = run_pipeline(opts)
    except Exception as exc:
        raise gr.Error(f"Pipeline failed: {exc}") from exc

    progress(1.0, desc="Done.")

    cassette_a = str(result.cassette_side_a) if result.cassette_side_a.exists() else None
    cassette_b = str(result.cassette_side_b) if result.cassette_side_b.exists() else None
    canvas     = str(result.spotify_canvas)  if result.spotify_canvas.exists()  else None

    direction_summary = (
        f"Style:    {result.direction.art_style}\n"
        f"Palette:  {result.direction.palette_primary}  ·  "
        f"{result.direction.palette_secondary}  ·  {result.direction.palette_accent}\n"
        f"Motion:   {result.direction.canvas_motion_style}\n"
        f"Brand:    {result.direction.cassette_brand_name} ({result.direction.cassette_era})\n"
        f"Tagline:  \"{result.direction.label_tagline}\""
    )

    return cassette_a, cassette_b, canvas, direction_summary


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="artwork-machine") as app:

    gr.Markdown("# artwork-machine", elem_id="title")
    gr.Markdown(
        "Drop in a track. Get cassette art, a Spotify Canvas loop, and a YouTube visualizer.",
        elem_id="subtitle",
    )

    with gr.Row(equal_height=False):

        # ── Left column — inputs ──────────────────────────────────────────────
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath",
                label="Audio file",
                sources=["upload"],
            )
            artist_input = gr.Textbox(label="Artist name", placeholder="e.g. moodmixformat")
            album_input  = gr.Textbox(label="Album / track title", placeholder="e.g. BLOOM")

            with gr.Row():
                skip_viz   = gr.Checkbox(label="Skip YouTube visualizer", value=True,
                                         info="Saves ~80 GB of temp disk space")
                draft_mode = gr.Checkbox(label="Draft mode", value=False,
                                         info="Faster, lower resolution preview")

            generate_btn = gr.Button("Generate", variant="primary", elem_id="generate-btn")

        # ── Right column — outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            cassette_a_out = gr.Image(label="Cassette Side A")
            cassette_b_out = gr.Image(label="Cassette Side B")
            canvas_out     = gr.Video(label="Spotify Canvas (8s loop)")
            direction_out  = gr.Textbox(label="Creative direction", lines=5, interactive=False)

    generate_btn.click(
        fn=generate,
        inputs=[audio_input, artist_input, album_input, skip_viz, draft_mode],
        outputs=[cassette_a_out, cassette_b_out, canvas_out, direction_out],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
    )

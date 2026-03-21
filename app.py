"""
artwork-machine web UI — Flask edition

Standard HTTP upload, no WebSockets, works reliably behind any proxy.

Run locally:  .venv/bin/python app.py
"""

import os
import threading
import uuid
from pathlib import Path

print("Starting artwork-machine...", flush=True)

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

load_dotenv()

print("Importing pipeline...", flush=True)
from artwork_machine.pipeline import PipelineOptions, run as run_pipeline
print("Pipeline ready.", flush=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

# In-memory job store: job_id -> { status, message, result, error }
JOBS: dict = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from flask import make_response
    resp = make_response(HTML)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/generate", methods=["POST"])
def generate():
    audio = request.files.get("audio")
    artist = request.form.get("artist", "").strip()
    album  = request.form.get("album",  "").strip()

    if not audio or not audio.filename:
        return jsonify({"error": "No audio file uploaded."}), 400
    if not artist:
        return jsonify({"error": "Artist name is required."}), 400
    if not album:
        return jsonify({"error": "Album / track title is required."}), 400

    job_id    = str(uuid.uuid4())
    audio_dir = Path("/tmp/artwork-machine")
    audio_dir.mkdir(exist_ok=True)
    audio_path = audio_dir / f"{job_id}_{audio.filename}"
    audio.save(audio_path)

    skip_canvas     = request.form.get("skip_canvas")     == "true"
    skip_short      = request.form.get("skip_short")      == "true"
    skip_visualizer = request.form.get("skip_visualizer") == "true"
    draft_mode      = request.form.get("draft_mode")      == "true"

    JOBS[job_id] = {"status": "running", "message": "Starting pipeline…", "result": None, "error": None}

    t = threading.Thread(
        target=_run_job,
        args=(job_id, audio_path, artist, album, skip_canvas, skip_short, skip_visualizer, draft_mode),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(job)


@app.route("/output/<path:filename>")
def output(filename):
    return send_from_directory("./output", filename)


# ── Background job ────────────────────────────────────────────────────────────

def _run_job(job_id, audio_path, artist, album, skip_canvas, skip_short, skip_visualizer, draft_mode):
    try:
        def update_status(msg):
            JOBS[job_id]["message"] = msg

        opts = PipelineOptions(
            artist=artist,
            album=album,
            audio_path=audio_path,
            output_dir=Path("./output"),
            draft=draft_mode,
            skip_canvas=skip_canvas,
            skip_short=skip_short,
            skip_visualizer=skip_visualizer,
            status_callback=update_status,
        )
        result = run_pipeline(opts)

        files = {}
        for key, path in [
            ("album_art",  result.album_art),
            ("thumbnail",  result.thumbnail),
            ("canvas",     result.spotify_canvas),
            ("short",      result.short_video),
            ("visualizer", result.youtube_visualizer),
        ]:
            if path.exists():
                # Make path relative to ./output for serving
                rel = str(path.relative_to(Path("./output")))
                files[key] = f"/output/{rel}"

        JOBS[job_id].update({
            "status":  "done",
            "message": "Complete.",
            "result":  {
                "files":   files,
                "style":   result.direction.art_style,
                "palette": f"{result.direction.palette_primary} · {result.direction.palette_secondary} · {result.direction.palette_accent}",
                "motion":  result.direction.canvas_motion_style,
                "tagline": result.direction.tagline,
            },
        })
    except Exception as exc:
        JOBS[job_id].update({"status": "error", "error": str(exc)})
    finally:
        try:
            audio_path.unlink()
        except Exception:
            pass


# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>artwork-machine</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0e0e0e; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; min-height: 100vh; padding: 40px 20px; }
    h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 6px; }
    p.sub { color: #888; margin-bottom: 32px; font-size: 0.95rem; }
    .container { max-width: 900px; margin: 0 auto; }
    .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 28px; margin-bottom: 24px; }
    label { display: block; font-size: 0.85rem; color: #aaa; margin-bottom: 6px; margin-top: 16px; }
    label:first-child { margin-top: 0; }
    input[type="text"], input[type="file"] { width: 100%; background: #111; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; color: #e0e0e0; font-size: 0.95rem; outline: none; }
    input[type="file"] { padding: 8px; cursor: pointer; }
    input[type="text"]:focus { border-color: #555; }
    .checkboxes { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 16px; }
    .checkboxes label { display: flex; align-items: center; gap: 8px; cursor: pointer; margin-top: 0; font-size: 0.9rem; color: #ccc; }
    .checkboxes input[type="checkbox"] { width: 16px; height: 16px; accent-color: #7c6af5; }
    button { width: 100%; margin-top: 20px; background: #7c6af5; color: #fff; border: none; border-radius: 8px; padding: 14px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #6a57e0; }
    button:disabled { background: #333; color: #666; cursor: not-allowed; }
    #status { margin-top: 16px; padding: 12px 16px; border-radius: 8px; font-size: 0.9rem; display: none; }
    #status.running { background: #1c2040; border: 1px solid #3a4080; color: #9ab; }
    #status.error   { background: #2a1010; border: 1px solid #6a2020; color: #f88; }
    #status.done    { background: #102a10; border: 1px solid #206a20; color: #8f8; }
    .results { margin-top: 24px; }
    .results h2 { font-size: 1.1rem; margin-bottom: 16px; color: #ccc; }
    .results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .output-item { background: #111; border: 1px solid #2a2a2a; border-radius: 10px; overflow: hidden; }
    .output-item .label { padding: 10px 14px; font-size: 0.8rem; color: #888; border-bottom: 1px solid #222; }
    .output-item img { width: 100%; display: block; }
    .output-item video { width: 100%; display: block; }
    .output-item a { display: block; padding: 12px 14px; color: #9ab4f5; font-size: 0.85rem; text-decoration: none; }
    .output-item a:hover { color: #bcd; }
    .direction { margin-top: 16px; background: #111; border: 1px solid #2a2a2a; border-radius: 10px; padding: 16px; font-size: 0.85rem; color: #999; line-height: 1.8; }
    .direction span { color: #ccc; }
  </style>
</head>
<body>
  <div class="container">
    <h1>artwork-machine</h1>
    <p class="sub">Upload a track. Get album art, thumbnail, Spotify Canvas, a short, and a YouTube visualizer.</p>

    <div class="card">
      <form id="form">
        <label>Audio file (MP3 / WAV / FLAC)</label>
        <input type="file" name="audio" accept="audio/*,.mp3,.wav,.flac,.aiff,.m4a" required>

        <label>Artist name</label>
        <input type="text" name="artist" placeholder="e.g. moodmixformat" required>

        <label>Album / track title</label>
        <input type="text" name="album" placeholder="e.g. BLOOM" required>

        <div class="checkboxes">
          <label><input type="checkbox" name="skip_canvas">     Skip Canvas</label>
          <label><input type="checkbox" name="skip_short">      Skip Short</label>
          <label><input type="checkbox" name="skip_visualizer" checked> Skip Visualizer</label>
          <label><input type="checkbox" name="draft_mode">      Draft mode</label>
        </div>

        <button type="submit" id="btn">Generate</button>
      </form>
      <div id="status"></div>
    </div>

    <div class="results" id="results" style="display:none"></div>
  </div>

  <script>
    let pollTimer = null;

    document.getElementById('form').addEventListener('submit', function(e) {
      e.preventDefault();
      const btn = document.getElementById('btn');
      const statusEl = document.getElementById('status');
      const resultsEl = document.getElementById('results');

      btn.disabled = true;
      btn.textContent = 'Generating…';
      resultsEl.style.display = 'none';
      statusEl.style.display = 'block';
      statusEl.className = 'running';
      statusEl.textContent = 'Uploading audio file…';

      const form = e.target;
      const data = new FormData(form);

      // Convert checkboxes to true/false strings
      ['skip_canvas','skip_short','skip_visualizer','draft_mode'].forEach(function(k) {
        data.set(k, form.querySelector('[name="' + k + '"]').checked ? 'true' : 'false');
      });

      // Read the file first via FileReader to confirm it's accessible
      // (iCloud Drive files can appear local but be unreadable by the browser sandbox)
      const audioFile = form.querySelector('[name="audio"]').files[0];
      if (!audioFile) { showError('No file selected.'); return; }

      statusEl.textContent = 'Reading file…';

      const reader = new FileReader();
      reader.onerror = function() {
        showError('Cannot read this file. If it is stored in iCloud Drive, open Finder, right-click the file, and choose "Download Now" — then try again.');
        btn.disabled = false;
        btn.textContent = 'Generate';
      };

      reader.onload = function(ev) {
        // File is readable — rebuild FormData with the raw bytes so XHR can send it reliably
        const blob = new Blob([ev.target.result], { type: audioFile.type || 'audio/wav' });
        const fd = new FormData();
        fd.append('audio', blob, audioFile.name);
        ['artist','album'].forEach(function(k) {
          fd.append(k, data.get(k));
        });
        ['skip_canvas','skip_short','skip_visualizer','draft_mode'].forEach(function(k) {
          fd.append(k, data.get(k));
        });

        statusEl.textContent = 'Uploading… 0%';

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/generate');

        xhr.upload.onprogress = function(ev) {
          if (ev.lengthComputable) {
            const pct = Math.round((ev.loaded / ev.total) * 100);
            statusEl.textContent = 'Uploading… ' + pct + '%';
          }
        };

        xhr.onload = function() {
          if (xhr.status !== 200) {
            showError('Server error: ' + xhr.status);
            return;
          }
          let json;
          try { json = JSON.parse(xhr.responseText); } catch(ex) {
            showError('Bad response from server');
            return;
          }
          if (json.error) { showError(json.error); return; }
          statusEl.textContent = 'Pipeline running — this takes a few minutes…';
          pollTimer = setInterval(function() { poll(json.job_id); }, 4000);
        };

        xhr.onerror = function() {
          showError('Network error during upload. Check your connection and try again.');
        };

        xhr.send(fd);
      };

      // Trigger the file read — this will fail fast if the file is an iCloud stub
      reader.readAsArrayBuffer(audioFile);
    });

    async function poll(jobId) {
      try {
        const res = await fetch(`/status/${jobId}`);
        if (res.status === 404) {
          clearInterval(pollTimer);
          showError('Server restarted mid-job — please re-upload your file.');
          return;
        }
        const job = await res.json();
        const statusEl = document.getElementById('status');

        if (job.status === 'running') {
          statusEl.textContent = job.message || 'Running…';
        } else if (job.status === 'error') {
          clearInterval(pollTimer);
          showError(job.error);
        } else if (job.status === 'done') {
          clearInterval(pollTimer);
          statusEl.className = 'done';
          statusEl.textContent = '✓ Done.';
          document.getElementById('btn').disabled = false;
          document.getElementById('btn').textContent = 'Generate';
          showResults(job.result);
        }
      } catch (err) {
        // network blip — keep polling
      }
    }

    function showError(msg) {
      const statusEl = document.getElementById('status');
      statusEl.className = 'error';
      statusEl.textContent = '✗ ' + msg;
      document.getElementById('btn').disabled = false;
      document.getElementById('btn').textContent = 'Generate';
    }

    function showResults(result) {
      const el = document.getElementById('results');
      const f  = result.files;

      const items = [
        { key: 'album_art',  label: 'Album Art (3000×3000)',    type: 'image' },
        { key: 'thumbnail',  label: 'YouTube Thumbnail',         type: 'image' },
        { key: 'canvas',     label: 'Spotify Canvas (8s loop)',  type: 'video' },
        { key: 'short',      label: '30-second Short',           type: 'video' },
        { key: 'visualizer', label: 'YouTube Visualizer',        type: 'video' },
      ];

      let grid = '<div class="results-grid">';
      items.forEach(item => {
        if (!f[item.key]) return;
        grid += `<div class="output-item">
          <div class="label">${item.label}</div>
          ${item.type === 'image'
            ? `<img src="${f[item.key]}" loading="lazy">`
            : `<video src="${f[item.key]}" controls loop playsinline></video>`}
          <a href="${f[item.key]}" download>↓ Download</a>
        </div>`;
      });
      grid += '</div>';

      el.innerHTML = `
        <h2>Generated deliverables</h2>
        ${grid}
        <div class="direction">
          <span>Style:</span>   ${result.style}<br>
          <span>Palette:</span> ${result.palette}<br>
          <span>Motion:</span>  ${result.motion}<br>
          <span>Tagline:</span> "${result.tagline}"
        </div>`;

      el.style.display = 'block';
    }
  </script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print(f"Launching on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

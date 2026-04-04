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


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff", ".m4a", ".ogg"}

@app.route("/tracks")
def list_tracks():
    output_dir = Path("./output")
    tracks = []
    if output_dir.exists():
        for folder in sorted(output_dir.iterdir()):
            if not folder.is_dir():
                continue
            art = folder / "album_art.png"
            if not art.exists():
                continue
            audio_file = next(
                (f for f in folder.iterdir() if f.stem == "audio" and f.suffix in AUDIO_EXTENSIONS),
                None,
            )
            if not audio_file:
                continue
            parts = folder.name.split("_", 1)
            artist = parts[0].replace("-", " ") if len(parts) > 1 else "Unknown"
            title  = parts[1].replace("-", " ") if len(parts) > 1 else folder.name.replace("-", " ")
            tracks.append({
                "id":          folder.name,
                "title":       title,
                "artist":      artist,
                "artwork_url": f"/output/{folder.name}/album_art.png",
                "audio_url":   f"/output/{folder.name}/{audio_file.name}",
                "uploaded_at": int(folder.stat().st_mtime),
            })
    return jsonify(tracks)


@app.route("/player")
def player():
    return PLAYER_HTML


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
    <div style="text-align:right; margin-bottom:12px;">
      <a href="/player" style="color:#7c6af5; text-decoration:none; font-size:0.88rem; font-weight:500;">&#9835; All Tracks</a>
    </div>
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


PLAYER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>All Tracks — artwork-machine</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --accent: #7c6af5;
      --accent-dim: rgba(124,106,245,0.35);
      --bg: #0a0a0a;
      --surface: rgba(255,255,255,0.04);
      --border: rgba(255,255,255,0.08);
      --text: #e8e8e8;
      --subtext: #888;
      --sidebar: 280px;
      --controls: 96px;
    }

    html, body { height: 100%; overflow: hidden; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      display: flex;
      flex-direction: column;
    }

    /* ── Background artwork blur ── */
    #bg-blur {
      position: fixed; inset: 0; z-index: 0;
      background-size: cover; background-position: center;
      filter: blur(90px) saturate(1.5) brightness(0.25);
      transform: scale(1.15);
      transition: background-image 0.6s ease;
    }
    #bg-overlay {
      position: fixed; inset: 0; z-index: 1;
      background: linear-gradient(to bottom, rgba(10,10,10,0.55) 0%, rgba(10,10,10,0.72) 100%);
    }

    /* ── Nav ── */
    #topnav {
      position: relative; z-index: 10;
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 24px;
      border-bottom: 1px solid var(--border);
      background: rgba(10,10,10,0.6);
      backdrop-filter: blur(20px);
      flex-shrink: 0;
    }
    #topnav a {
      color: var(--subtext); text-decoration: none; font-size: 0.85rem;
      transition: color 0.2s;
    }
    #topnav a:hover { color: var(--text); }
    #topnav .title { font-size: 0.95rem; font-weight: 600; color: var(--text); }

    /* ── Main layout ── */
    #main {
      position: relative; z-index: 5;
      display: flex; flex: 1; overflow: hidden;
    }

    /* ── Sidebar ── */
    #sidebar {
      width: var(--sidebar); flex-shrink: 0;
      display: flex; flex-direction: column;
      border-right: 1px solid var(--border);
      background: rgba(10,10,10,0.5);
      backdrop-filter: blur(20px);
      overflow: hidden;
    }
    #sidebar-header {
      padding: 16px 16px 10px;
      flex-shrink: 0;
    }
    #sidebar-header h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--subtext); margin-bottom: 10px; }
    .sort-tabs { display: flex; gap: 6px; }
    .sort-tab {
      flex: 1; padding: 6px 0; text-align: center; font-size: 0.78rem;
      border-radius: 6px; cursor: pointer; border: 1px solid var(--border);
      color: var(--subtext); background: transparent; transition: all 0.15s;
    }
    .sort-tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
    .sort-tab:hover:not(.active) { border-color: #555; color: var(--text); }

    #track-list {
      flex: 1; overflow-y: auto;
      padding: 6px 0;
    }
    #track-list::-webkit-scrollbar { width: 4px; }
    #track-list::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

    .track-item {
      display: flex; align-items: center; gap: 12px;
      padding: 8px 14px; cursor: pointer;
      transition: background 0.15s;
      border-left: 3px solid transparent;
    }
    .track-item:hover { background: rgba(255,255,255,0.05); }
    .track-item.active {
      border-left-color: var(--accent);
      background: rgba(124,106,245,0.1);
    }
    .track-thumb {
      width: 42px; height: 42px; border-radius: 6px;
      object-fit: cover; flex-shrink: 0;
      background: #222;
    }
    .track-meta { overflow: hidden; }
    .track-name { font-size: 0.85rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .track-artist { font-size: 0.75rem; color: var(--subtext); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 2px; }
    .track-item.active .track-name { color: var(--accent); }

    /* ── Center stage ── */
    #stage {
      flex: 1; display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      padding: 32px 24px 24px;
      gap: 24px;
    }

    #artwork-wrap {
      position: relative;
      transition: transform 0.4s cubic-bezier(0.34,1.56,0.64,1), opacity 0.3s ease;
    }
    #artwork-wrap.transitioning { opacity: 0; transform: scale(0.93); }

    #artwork {
      width: min(460px, calc(100vw - var(--sidebar) - 80px));
      height: min(460px, calc(100vw - var(--sidebar) - 80px));
      border-radius: 14px;
      object-fit: cover;
      display: block;
      box-shadow: 0 0 0 0 var(--accent-dim);
      transition: box-shadow 0.4s ease;
    }
    body.playing #artwork {
      box-shadow:
        0 0 60px 8px var(--accent-dim),
        0 24px 80px rgba(0,0,0,0.8);
      animation: artglow 3s ease-in-out infinite;
    }
    @keyframes artglow {
      0%, 100% { box-shadow: 0 0 55px 6px var(--accent-dim), 0 24px 80px rgba(0,0,0,0.8); }
      50%       { box-shadow: 0 0 80px 18px rgba(124,106,245,0.5), 0 24px 80px rgba(0,0,0,0.8); }
    }

    #now-playing {
      text-align: center;
      transition: opacity 0.3s ease;
    }
    #now-playing.transitioning { opacity: 0; }
    #np-title  { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
    #np-artist { font-size: 0.95rem; color: var(--subtext); }

    #empty-state {
      text-align: center; color: var(--subtext);
    }
    #empty-state p { font-size: 1rem; margin-bottom: 8px; }
    #empty-state a { color: var(--accent); text-decoration: none; }

    /* ── Controls bar ── */
    #controls {
      position: relative; z-index: 10;
      height: var(--controls);
      background: rgba(10,10,10,0.72);
      backdrop-filter: blur(24px) saturate(1.2);
      border-top: 1px solid var(--border);
      flex-shrink: 0;
      display: flex; flex-direction: column;
      justify-content: center;
      padding: 0 28px;
      gap: 8px;
    }

    .ctrl-row {
      display: flex; align-items: center; gap: 16px;
    }
    .ctrl-row.center { justify-content: center; }

    .ctrl-btn {
      background: none; border: none; cursor: pointer;
      color: var(--subtext); padding: 6px;
      border-radius: 50%; display: flex; align-items: center; justify-content: center;
      transition: color 0.2s, background 0.2s, transform 0.1s;
      flex-shrink: 0;
    }
    .ctrl-btn:hover { color: var(--text); background: rgba(255,255,255,0.07); }
    .ctrl-btn:active { transform: scale(0.9); }
    .ctrl-btn.active { color: var(--accent); }
    .ctrl-btn svg { display: block; }

    #btn-play {
      width: 52px; height: 52px;
      background: #fff; color: #000;
      border-radius: 50%;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
      transition: transform 0.15s, background 0.2s;
    }
    #btn-play:hover { transform: scale(1.07); background: #e8e8ff; }
    #btn-play:active { transform: scale(0.95); }

    /* Progress */
    .progress-group {
      flex: 1; display: flex; align-items: center; gap: 10px;
      font-size: 0.75rem; color: var(--subtext);
      min-width: 0;
    }
    .time-label { flex-shrink: 0; min-width: 36px; text-align: center; font-variant-numeric: tabular-nums; }

    input[type="range"] {
      -webkit-appearance: none; appearance: none;
      height: 3px; border-radius: 2px;
      background: var(--border);
      outline: none; cursor: pointer;
      flex: 1;
      transition: height 0.15s;
    }
    input[type="range"]:hover { height: 5px; }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 13px; height: 13px; border-radius: 50%;
      background: #fff; cursor: pointer;
      box-shadow: 0 0 4px rgba(0,0,0,0.4);
      transition: transform 0.1s;
    }
    input[type="range"]::-webkit-slider-thumb:active { transform: scale(1.3); }
    input[type="range"].vol { flex: 0 0 80px; }

    #progress-bar {
      background: linear-gradient(to right, var(--accent) var(--pct, 0%), var(--border) var(--pct, 0%));
    }
    #vol-bar {
      background: linear-gradient(to right, #ccc var(--vol, 80%), var(--border) var(--vol, 80%));
    }

    /* Responsive: hide sidebar label at narrow widths */
    @media (max-width: 640px) {
      :root { --sidebar: 200px; }
      #np-title { font-size: 1.1rem; }
    }
  </style>
</head>
<body>
  <!-- Blurred background layers -->
  <div id="bg-blur"></div>
  <div id="bg-overlay"></div>

  <!-- Top nav -->
  <nav id="topnav">
    <a href="/">&#8592; Generate</a>
    <span class="title">All Tracks</span>
    <span></span>
  </nav>

  <!-- Main area -->
  <div id="main">
    <!-- Sidebar: track list -->
    <aside id="sidebar">
      <div id="sidebar-header">
        <h2>Library</h2>
        <div class="sort-tabs">
          <button class="sort-tab active" data-sort="title">Title</button>
          <button class="sort-tab" data-sort="date">Date</button>
        </div>
      </div>
      <div id="track-list"></div>
    </aside>

    <!-- Center: artwork + info -->
    <section id="stage">
      <div id="empty-state" style="display:none">
        <p>No tracks found.</p>
        <a href="/">Generate your first track &#8594;</a>
      </div>

      <div id="artwork-wrap">
        <img id="artwork" src="" alt="Album art">
      </div>

      <div id="now-playing">
        <div id="np-title">—</div>
        <div id="np-artist">—</div>
      </div>
    </section>
  </div>

  <!-- Controls bar -->
  <div id="controls">
    <div class="ctrl-row center">
      <!-- Shuffle -->
      <button class="ctrl-btn" id="btn-shuffle" title="Shuffle">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
          <path d="M16.47 5.47a.75.75 0 011.06 0l2 2a.75.75 0 010 1.06l-2 2a.75.75 0 11-1.06-1.06l.72-.72H15c-.92 0-1.6.6-2.35 1.67l-.29.42C11.4 11.97 10.3 13.5 8.5 13.5H3a.75.75 0 010-1.5h5.5c.92 0 1.6-.6 2.35-1.67l.29-.42C12.1 8.53 13.2 7 15 7h2.19l-.72-.72a.75.75 0 010-1.06zm0 9a.75.75 0 011.06-1.06l2 2a.75.75 0 010 1.06l-2 2a.75.75 0 11-1.06-1.06l.72-.72H15c-1.8 0-2.9-1.53-3.85-2.91l-.29-.42C10.1 13.1 9.42 12.5 8.5 12.5H3a.75.75 0 010-1.5h5.5c1.8 0 2.9 1.53 3.85 2.91l.29.42C13.4 15.4 14.08 16 15 16h2.19l-.72-.72z"/>
        </svg>
      </button>
      <!-- Prev -->
      <button class="ctrl-btn" id="btn-prev" title="Previous">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
          <path d="M6 6a1 1 0 011 1v10a1 1 0 11-2 0V7a1 1 0 011-1zm3.43 1.534a1 1 0 011.538-.844l8 5a1 1 0 010 1.688l-8 5A1 1 0 019.43 18V7.534z"/>
        </svg>
      </button>
      <!-- Play/Pause -->
      <button class="ctrl-btn" id="btn-play" title="Play/Pause">
        <svg id="icon-play" width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
          <path d="M8 5.14v14l11-7-11-7z"/>
        </svg>
        <svg id="icon-pause" width="22" height="22" viewBox="0 0 24 24" fill="currentColor" style="display:none">
          <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
        </svg>
      </button>
      <!-- Next -->
      <button class="ctrl-btn" id="btn-next" title="Next">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
          <path d="M18 6a1 1 0 011 1v10a1 1 0 11-2 0V7a1 1 0 011-1zm-2.43 1.534l-8 5a1 1 0 000 1.688l8 5A1 1 0 0017 18V7.534a1 1 0 00-1.43-.9z"/>
        </svg>
      </button>

      <!-- Progress + time -->
      <div class="progress-group">
        <span class="time-label" id="time-cur">0:00</span>
        <input type="range" id="progress-bar" min="0" max="100" value="0" step="0.1">
        <span class="time-label" id="time-dur">0:00</span>
      </div>

      <!-- Volume -->
      <svg width="16" height="16" viewBox="0 0 24 24" fill="#888" flex-shrink="0">
        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0014 7.97v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
      </svg>
      <input type="range" id="vol-bar" class="vol" min="0" max="100" value="80" step="1">
    </div>
  </div>

  <audio id="audio"></audio>

  <script>
    const audio      = document.getElementById('audio');
    const bgBlur     = document.getElementById('bg-blur');
    const artImg     = document.getElementById('artwork');
    const artWrap    = document.getElementById('artwork-wrap');
    const npTitle    = document.getElementById('np-title');
    const npArtist   = document.getElementById('np-artist');
    const npBlock    = document.getElementById('now-playing');
    const trackList  = document.getElementById('track-list');
    const emptyState = document.getElementById('empty-state');
    const btnPlay    = document.getElementById('btn-play');
    const iconPlay   = document.getElementById('icon-play');
    const iconPause  = document.getElementById('icon-pause');
    const btnPrev    = document.getElementById('btn-prev');
    const btnNext    = document.getElementById('btn-next');
    const btnShuffle = document.getElementById('btn-shuffle');
    const progressBar = document.getElementById('progress-bar');
    const volBar     = document.getElementById('vol-bar');
    const timeCur    = document.getElementById('time-cur');
    const timeDur    = document.getElementById('time-dur');

    let allTracks    = [];  // original order from API
    let queue        = [];  // current playback order
    let currentIdx   = -1;
    let shuffled     = false;
    let sortMode     = 'title';  // 'title' | 'date'
    let isSeeking    = false;

    // ── Utilities ──────────────────────────────────────────────────────────────
    function fmt(secs) {
      if (!isFinite(secs)) return '0:00';
      const m = Math.floor(secs / 60);
      const s = Math.floor(secs % 60).toString().padStart(2, '0');
      return m + ':' + s;
    }

    function shuffle(arr) {
      const a = [...arr];
      for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
      }
      return a;
    }

    function sortedTracks() {
      const copy = [...allTracks];
      if (sortMode === 'title') {
        copy.sort((a, b) => a.title.localeCompare(b.title));
      } else {
        copy.sort((a, b) => b.uploaded_at - a.uploaded_at);
      }
      return copy;
    }

    // ── Render track list ──────────────────────────────────────────────────────
    function renderList() {
      const sorted = sortedTracks();
      trackList.innerHTML = '';
      sorted.forEach((track, i) => {
        const div = document.createElement('div');
        div.className = 'track-item';
        div.dataset.id = track.id;
        div.innerHTML = `
          <img class="track-thumb" src="${track.artwork_url}" loading="lazy" alt="">
          <div class="track-meta">
            <div class="track-name">${escHtml(track.title)}</div>
            <div class="track-artist">${escHtml(track.artist)}</div>
          </div>`;
        div.addEventListener('click', () => loadFromSorted(sorted, i));
        trackList.appendChild(div);
      });
      highlightActive();
    }

    function escHtml(s) {
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function highlightActive() {
      document.querySelectorAll('.track-item').forEach(el => {
        el.classList.toggle('active', queue[currentIdx] && el.dataset.id === queue[currentIdx].id);
      });
    }

    // ── Load a track ──────────────────────────────────────────────────────────
    function loadFromSorted(sorted, idx) {
      // When clicking a track: set queue = sorted order starting at that track
      queue = sorted;
      currentIdx = idx;
      loadCurrent();
    }

    function loadCurrent(autoplay = true) {
      if (currentIdx < 0 || currentIdx >= queue.length) return;
      const track = queue[currentIdx];

      // Artwork transition
      artWrap.classList.add('transitioning');
      npBlock.classList.add('transitioning');

      setTimeout(() => {
        artImg.src = track.artwork_url;
        bgBlur.style.backgroundImage = `url('${track.artwork_url}')`;
        npTitle.textContent  = track.title;
        npArtist.textContent = track.artist;
        artWrap.classList.remove('transitioning');
        npBlock.classList.remove('transitioning');
      }, 280);

      audio.src = track.audio_url;
      audio.volume = volBar.value / 100;
      progressBar.value = 0;
      timeCur.textContent = '0:00';
      timeDur.textContent = '0:00';

      if (autoplay) {
        audio.play().catch(() => {});
      }
      highlightActive();
    }

    // ── Playback controls ─────────────────────────────────────────────────────
    btnPlay.addEventListener('click', () => {
      if (currentIdx === -1 && queue.length > 0) {
        currentIdx = 0; loadCurrent(); return;
      }
      if (audio.paused) { audio.play().catch(() => {}); }
      else { audio.pause(); }
    });

    btnNext.addEventListener('click', playNext);
    btnPrev.addEventListener('click', playPrev);

    function playNext() {
      if (!queue.length) return;
      currentIdx = (currentIdx + 1) % queue.length;
      loadCurrent();
    }
    function playPrev() {
      if (!queue.length) return;
      if (audio.currentTime > 3) { audio.currentTime = 0; return; }
      currentIdx = (currentIdx - 1 + queue.length) % queue.length;
      loadCurrent();
    }

    btnShuffle.addEventListener('click', () => {
      shuffled = !shuffled;
      btnShuffle.classList.toggle('active', shuffled);
      if (shuffled) {
        // Rebuild queue shuffled, keep current track at front
        const current = queue[currentIdx];
        const rest = shuffle(queue.filter((_, i) => i !== currentIdx));
        queue = current ? [current, ...rest] : rest;
        currentIdx = 0;
      } else {
        // Restore sorted order, find current track
        const current = queue[currentIdx];
        queue = sortedTracks();
        currentIdx = current ? queue.findIndex(t => t.id === current.id) : 0;
        if (currentIdx === -1) currentIdx = 0;
      }
    });

    audio.addEventListener('ended', playNext);

    // ── Play/pause state sync ─────────────────────────────────────────────────
    audio.addEventListener('play',  () => {
      iconPlay.style.display  = 'none';
      iconPause.style.display = 'block';
      document.body.classList.add('playing');
    });
    audio.addEventListener('pause', () => {
      iconPlay.style.display  = 'block';
      iconPause.style.display = 'none';
      document.body.classList.remove('playing');
    });

    // ── Progress ──────────────────────────────────────────────────────────────
    audio.addEventListener('timeupdate', () => {
      if (isSeeking || !isFinite(audio.duration)) return;
      const pct = (audio.currentTime / audio.duration) * 100;
      progressBar.value = pct;
      progressBar.style.setProperty('--pct', pct + '%');
      timeCur.textContent = fmt(audio.currentTime);
    });
    audio.addEventListener('loadedmetadata', () => {
      timeDur.textContent = fmt(audio.duration);
    });
    audio.addEventListener('durationchange', () => {
      timeDur.textContent = fmt(audio.duration);
    });

    progressBar.addEventListener('mousedown', () => { isSeeking = true; });
    progressBar.addEventListener('input', () => {
      const pct = progressBar.value;
      progressBar.style.setProperty('--pct', pct + '%');
      timeCur.textContent = fmt((pct / 100) * (audio.duration || 0));
    });
    progressBar.addEventListener('change', () => {
      if (isFinite(audio.duration)) {
        audio.currentTime = (progressBar.value / 100) * audio.duration;
      }
      isSeeking = false;
    });

    // ── Volume ────────────────────────────────────────────────────────────────
    volBar.style.setProperty('--vol', '80%');
    volBar.addEventListener('input', () => {
      audio.volume = volBar.value / 100;
      volBar.style.setProperty('--vol', volBar.value + '%');
    });

    // ── Sort tabs ─────────────────────────────────────────────────────────────
    document.querySelectorAll('.sort-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.sort-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        sortMode = btn.dataset.sort;
        const currentId = queue[currentIdx]?.id;
        renderList();
        if (!shuffled) {
          queue = sortedTracks();
          currentIdx = currentId ? queue.findIndex(t => t.id === currentId) : 0;
          if (currentIdx === -1) currentIdx = 0;
        }
      });
    });

    // ── Keyboard shortcuts ────────────────────────────────────────────────────
    document.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT') return;
      if (e.code === 'Space') { e.preventDefault(); btnPlay.click(); }
      if (e.code === 'ArrowRight') { e.preventDefault(); playNext(); }
      if (e.code === 'ArrowLeft')  { e.preventDefault(); playPrev(); }
    });

    // ── Load tracks from API ──────────────────────────────────────────────────
    async function init() {
      try {
        const res = await fetch('/tracks');
        allTracks = await res.json();
      } catch (e) {
        allTracks = [];
      }

      if (!allTracks.length) {
        document.getElementById('artwork-wrap').style.display = 'none';
        document.getElementById('now-playing').style.display = 'none';
        emptyState.style.display = 'block';
        return;
      }

      queue = sortedTracks();
      currentIdx = 0;
      renderList();
      // Pre-load first track (no autoplay; wait for user interaction)
      loadCurrent(false);
    }

    init();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print(f"Launching on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

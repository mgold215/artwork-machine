"""
artwork-machine Flask web UI.

Routes
──────
GET  /                      Single-page web app
GET  /profiles              List saved style profiles (JSON)
POST /profiles              Create profile: upload reference covers
DELETE /profiles/<id>       Delete a profile
POST /generate              Start generation job
GET  /status/<job_id>       Poll job status
GET  /output/<filename>     Download a generated file
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file, abort
from dotenv import load_dotenv

load_dotenv()

from artwork_machine.pipeline import run as run_pipeline, PipelineOptions

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./output"))
PROFILES_DIR = Path(os.environ.get("PROFILES_DIR", "./profiles"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
(PROFILES_DIR / "covers").mkdir(exist_ok=True)

ALLOWED_AUDIO = {".wav", ".mp3", ".flac", ".aiff", ".ogg", ".m4a"}
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
MAX_AUDIO_MB = 500
MAX_IMAGE_MB = 20

# ── App + job store ───────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = (MAX_AUDIO_MB + MAX_IMAGE_MB * 5) * 1024 * 1024

_jobs: dict[str, dict] = {}   # job_id -> {status, stage, message, result, error}
_jobs_lock = threading.Lock()


# ── Helper: save uploaded file safely ────────────────────────────────────────

def _save_upload(file_storage, dest_dir: Path, allowed_exts: set[str]) -> Path:
    orig = Path(file_storage.filename or "upload")
    if orig.suffix.lower() not in allowed_exts:
        abort(400, f"File type {orig.suffix!r} not allowed. Allowed: {allowed_exts}")
    dest = dest_dir / f"{uuid.uuid4().hex}{orig.suffix.lower()}"
    file_storage.save(str(dest))
    return dest


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return _HTML, 200, {"Content-Type": "text/html; charset=utf-8",
                        "Cache-Control": "no-store"}


# ── Profile management ────────────────────────────────────────────────────────

@app.route("/profiles", methods=["GET"])
def list_profiles():
    profiles = []
    for p in sorted(PROFILES_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            profiles.append(json.loads(p.read_text()))
        except Exception:
            pass
    return jsonify(profiles)


@app.route("/profiles", methods=["POST"])
def create_profile():
    name = request.form.get("name", "My Style").strip() or "My Style"
    covers = request.files.getlist("covers")
    if not covers or all(c.filename == "" for c in covers):
        abort(400, "At least one reference cover image is required.")

    profile_id = uuid.uuid4().hex
    tmp_dir = Path(f"/tmp/am_profile_{profile_id}")
    tmp_dir.mkdir(parents=True)

    cover_paths: list[Path] = []
    saved_thumbs: list[str] = []
    try:
        for i, cover in enumerate(covers[:5]):
            if not cover.filename:
                continue
            tmp_path = _save_upload(cover, tmp_dir, ALLOWED_IMAGE)
            cover_paths.append(tmp_path)
            # Save thumbnail copy to profiles/covers/
            thumb_name = f"{profile_id}_{i}{tmp_path.suffix}"
            thumb_dest = PROFILES_DIR / "covers" / thumb_name
            shutil.copy(tmp_path, thumb_dest)
            saved_thumbs.append(str(thumb_dest))

        # Run Claude vision analysis
        from artwork_machine.ai.style_analyzer import analyse_reference_covers
        style = analyse_reference_covers(cover_paths)
        style_dict = {
            "dominant_colors": style.dominant_colors,
            "art_style": style.art_style,
            "composition": style.composition,
            "mood": style.mood,
            "era": style.era,
            "visual_themes": style.visual_themes,
            "texture": style.texture,
            "color_treatment": style.color_treatment,
            "typography_style": style.typography_style,
            "summary": style.summary,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    profile = {
        "id": profile_id,
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        "cover_paths": saved_thumbs,
        "cover_count": len(saved_thumbs),
        "style_analysis": style_dict,
    }
    (PROFILES_DIR / f"{profile_id}.json").write_text(json.dumps(profile, indent=2))

    return jsonify(profile), 201


@app.route("/profiles/<profile_id>", methods=["DELETE"])
def delete_profile(profile_id: str):
    profile_file = PROFILES_DIR / f"{profile_id}.json"
    if not profile_file.exists():
        abort(404)
    try:
        profile = json.loads(profile_file.read_text())
        for cp in profile.get("cover_paths", []):
            Path(cp).unlink(missing_ok=True)
    except Exception:
        pass
    profile_file.unlink(missing_ok=True)
    return jsonify({"deleted": profile_id})


# ── Generation ────────────────────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
def generate():
    artist = request.form.get("artist", "").strip()
    album = request.form.get("album", "").strip()
    if not artist or not album:
        abort(400, "Artist and album name are required.")

    audio_file = request.files.get("audio")
    if not audio_file or not audio_file.filename:
        abort(400, "Audio file is required.")

    draft = request.form.get("draft_mode") == "true"
    skip_canvas = request.form.get("skip_canvas") == "true"
    skip_visualizer = request.form.get("skip_visualizer") == "true"
    profile_id = request.form.get("profile_id", "").strip() or None

    # Save audio to temp location
    tmp_dir = Path(f"/tmp/am_job_{uuid.uuid4().hex}")
    tmp_dir.mkdir(parents=True)
    audio_path = _save_upload(audio_file, tmp_dir, ALLOWED_AUDIO)

    # Collect reference covers (either fresh uploads or from saved profile)
    ref_paths: list[Path] = []

    fresh_covers = request.files.getlist("covers")
    if fresh_covers and any(c.filename for c in fresh_covers):
        for c in fresh_covers[:5]:
            if c.filename:
                ref_paths.append(_save_upload(c, tmp_dir, ALLOWED_IMAGE))
    elif profile_id:
        profile_file = PROFILES_DIR / f"{profile_id}.json"
        if profile_file.exists():
            profile = json.loads(profile_file.read_text())
            ref_paths = [Path(p) for p in profile.get("cover_paths", []) if Path(p).exists()]

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "stage": "queued",
            "message": "Job queued…",
            "result": None,
            "error": None,
        }

    t = threading.Thread(
        target=_run_job,
        args=(job_id, artist, album, audio_path, ref_paths, tmp_dir, draft, skip_canvas, skip_visualizer),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


def _run_job(
    job_id: str,
    artist: str,
    album: str,
    audio_path: Path,
    ref_paths: list[Path],
    tmp_dir: Path,
    draft: bool,
    skip_canvas: bool,
    skip_visualizer: bool,
) -> None:
    def _cb(stage: str, msg: str) -> None:
        with _jobs_lock:
            _jobs[job_id]["stage"] = stage
            _jobs[job_id]["message"] = msg

    try:
        opts = PipelineOptions(
            artist=artist,
            album=album,
            audio_path=audio_path,
            output_dir=OUTPUT_DIR,
            reference_cover_paths=ref_paths,
            draft=draft,
            skip_canvas=skip_canvas,
            skip_visualizer=skip_visualizer,
            progress_callback=_cb,
        )
        result = run_pipeline(opts)

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["stage"] = "done"
            _jobs[job_id]["message"] = "Complete!"
            _jobs[job_id]["result"] = {
                "album_cover": str(result.album_cover.relative_to(OUTPUT_DIR.parent)),
                "spotify_canvas": str(result.spotify_canvas.relative_to(OUTPUT_DIR.parent)) if result.spotify_canvas.exists() else None,
                "youtube_visualizer": str(result.youtube_visualizer.relative_to(OUTPUT_DIR.parent)) if result.youtube_visualizer.exists() else None,
                "art_style": result.direction.art_style,
                "tagline": result.direction.tagline,
                "bpm": round(result.features.bpm),
                "key": result.features.key,
                "elapsed": round(result.elapsed_seconds),
            }
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["stage"] = "error"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["message"] = f"Error: {e}"
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/status/<job_id>")
def status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        abort(404)
    return jsonify(job), 200, {"Cache-Control": "no-store"}


@app.route("/output/<path:filename>")
def output_file(filename: str):
    # Prevent directory traversal
    safe = OUTPUT_DIR / filename
    try:
        safe.relative_to(OUTPUT_DIR)
    except ValueError:
        abort(400)
    if not safe.exists():
        abort(404)
    return send_file(str(safe), as_attachment=True)


# ── Single-page HTML ──────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>artwork-machine</title>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1c1c28;
    --border: #2a2a3a;
    --text: #e8e8f0;
    --muted: #7070a0;
    --accent: #7c6bff;
    --accent2: #ff6b9d;
    --success: #4ade80;
    --error: #f87171;
    --radius: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px;
    line-height: 1.6;
    min-height: 100vh;
  }
  .container { max-width: 860px; margin: 0 auto; padding: 32px 20px 80px; }
  h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px;
  }
  .subtitle { color: var(--muted); margin-bottom: 36px; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 20px;
  }
  .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 16px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
  label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
  input[type="text"], select {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 15px;
    padding: 10px 14px;
    outline: none;
    transition: border-color .2s;
  }
  input[type="text"]:focus, select:focus { border-color: var(--accent); }

  /* Drop zones */
  .drop-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 28px;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s, background .2s;
    position: relative;
  }
  .drop-zone:hover, .drop-zone.drag-over { border-color: var(--accent); background: rgba(124,107,255,.06); }
  .drop-zone input[type="file"] { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  .drop-zone .icon { font-size: 2rem; margin-bottom: 8px; }
  .drop-zone .label { color: var(--muted); font-size: 13px; }
  .drop-zone .filename { color: var(--success); font-size: 13px; margin-top: 6px; display: none; }

  /* Cover grid */
  .cover-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-top: 14px;
  }
  .cover-slot {
    aspect-ratio: 1;
    border: 2px dashed var(--border);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    cursor: pointer;
    position: relative;
    transition: border-color .2s;
  }
  .cover-slot:hover { border-color: var(--accent); }
  .cover-slot input[type="file"] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .cover-slot img { width: 100%; height: 100%; object-fit: cover; }
  .cover-slot .plus { color: var(--muted); font-size: 1.5rem; }

  /* Profile selector */
  .profile-row { display: flex; gap: 10px; align-items: flex-end; }
  .profile-row select { flex: 1; }
  .btn-sm {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    cursor: pointer;
    font-size: 13px;
    padding: 10px 14px;
    white-space: nowrap;
    transition: border-color .2s;
  }
  .btn-sm:hover { border-color: var(--accent); }

  /* Row / grid fields */
  .field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .field { margin-bottom: 16px; }

  /* Toggle switches */
  .toggles { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 4px; }
  .toggle { display: flex; align-items: center; gap: 8px; cursor: pointer; }
  .toggle input { width: 38px; height: 22px; accent-color: var(--accent); cursor: pointer; }
  .toggle span { font-size: 13px; color: var(--muted); }

  /* Generate button */
  .btn-generate {
    width: 100%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none;
    border-radius: var(--radius);
    color: #fff;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 600;
    padding: 16px;
    transition: opacity .2s, transform .1s;
    margin-top: 8px;
  }
  .btn-generate:hover { opacity: .9; }
  .btn-generate:active { transform: scale(.98); }
  .btn-generate:disabled { opacity: .45; cursor: not-allowed; }

  /* Progress */
  #progress-section { display: none; }
  .stage-list { list-style: none; padding: 0; }
  .stage-list li {
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14px;
  }
  .stage-list li:last-child { border-bottom: none; }
  .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--border); flex-shrink: 0;
  }
  .dot.active { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
  .dot.done { background: var(--success); }
  .dot.error { background: var(--error); }

  /* Results */
  #results-section { display: none; }
  .results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
  .result-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .result-card .thumb {
    width: 100%;
    aspect-ratio: 1;
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .result-card .thumb img { width: 100%; height: 100%; object-fit: cover; }
  .result-card .thumb .icon { font-size: 3rem; }
  .result-card .info { padding: 14px; }
  .result-card .info h3 { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
  .result-card .info p { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
  .btn-dl {
    display: block;
    background: var(--accent);
    border: none;
    border-radius: 6px;
    color: #fff;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    padding: 8px 12px;
    text-align: center;
    text-decoration: none;
    transition: opacity .2s;
  }
  .btn-dl:hover { opacity: .85; }
  .meta-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  .meta-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-size: 12px;
    padding: 4px 12px;
    color: var(--muted);
  }
  .meta-chip span { color: var(--text); }

  /* Toast */
  #toast {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: var(--error); color: #fff;
    border-radius: 8px; padding: 10px 20px; font-size: 14px;
    display: none; z-index: 999; max-width: 420px; text-align: center;
  }
</style>
</head>
<body>
<div class="container">

  <h1>artwork-machine</h1>
  <p class="subtitle">Drop a track. Get a 3000×3000 album cover, Spotify Canvas, and YouTube visualizer.</p>

  <!-- Audio upload -->
  <div class="card">
    <h2>Track</h2>
    <div class="drop-zone" id="audio-drop">
      <input type="file" id="audio-input" accept=".wav,.mp3,.flac,.aiff,.ogg,.m4a">
      <div class="icon">🎵</div>
      <div class="label">Drop WAV / MP3 / FLAC here, or click to browse</div>
      <div class="filename" id="audio-filename"></div>
    </div>
    <div class="field-row" style="margin-top:16px">
      <div>
        <label for="artist">Artist name</label>
        <input type="text" id="artist" placeholder="e.g. Neon Drift">
      </div>
      <div>
        <label for="album">Album / track title</label>
        <input type="text" id="album" placeholder="e.g. Glass Veins">
      </div>
    </div>
  </div>

  <!-- Style profile -->
  <div class="card">
    <h2>Style Reference</h2>
    <p style="font-size:13px;color:var(--muted);margin-bottom:14px">
      Upload covers you love — Claude analyzes your taste and generates art that matches your aesthetic.
    </p>

    <!-- Saved profiles -->
    <div class="field">
      <label>Saved profile</label>
      <div class="profile-row">
        <select id="profile-select">
          <option value="">No saved profile — use covers below</option>
        </select>
        <button class="btn-sm" onclick="deleteProfile()">Delete</button>
        <button class="btn-sm" onclick="saveProfile()">Save as new profile</button>
      </div>
    </div>

    <!-- Fresh cover uploads -->
    <label>Or upload reference covers (up to 5)</label>
    <div class="cover-grid" id="cover-grid">
      <div class="cover-slot" onclick="triggerCoverPick(0)">
        <input type="file" id="cover-0" accept=".jpg,.jpeg,.png,.webp" onchange="previewCover(0,this)">
        <span class="plus">+</span>
      </div>
      <div class="cover-slot" onclick="triggerCoverPick(1)">
        <input type="file" id="cover-1" accept=".jpg,.jpeg,.png,.webp" onchange="previewCover(1,this)">
        <span class="plus">+</span>
      </div>
      <div class="cover-slot" onclick="triggerCoverPick(2)">
        <input type="file" id="cover-2" accept=".jpg,.jpeg,.png,.webp" onchange="previewCover(2,this)">
        <span class="plus">+</span>
      </div>
      <div class="cover-slot" onclick="triggerCoverPick(3)">
        <input type="file" id="cover-3" accept=".jpg,.jpeg,.png,.webp" onchange="previewCover(3,this)">
        <span class="plus">+</span>
      </div>
      <div class="cover-slot" onclick="triggerCoverPick(4)">
        <input type="file" id="cover-4" accept=".jpg,.jpeg,.png,.webp" onchange="previewCover(4,this)">
        <span class="plus">+</span>
      </div>
    </div>
  </div>

  <!-- Options -->
  <div class="card">
    <h2>Options</h2>
    <div class="toggles">
      <label class="toggle">
        <input type="checkbox" id="draft-mode">
        <span>Draft mode (fast preview, skips Runway)</span>
      </label>
      <label class="toggle">
        <input type="checkbox" id="skip-canvas">
        <span>Skip Spotify Canvas</span>
      </label>
      <label class="toggle">
        <input type="checkbox" id="skip-visualizer">
        <span>Skip YouTube Visualizer</span>
      </label>
    </div>
  </div>

  <!-- Generate -->
  <button class="btn-generate" id="generate-btn" onclick="startGeneration()">
    Generate
  </button>

  <!-- Progress -->
  <div class="card" id="progress-section" style="margin-top:24px">
    <h2>Progress</h2>
    <ul class="stage-list" id="stage-list">
      <li><span class="dot" id="dot-analyzing_audio"></span> Analysing audio</li>
      <li><span class="dot" id="dot-analyzing_style"></span> Analysing style references</li>
      <li><span class="dot" id="dot-generating_direction"></span> Generating creative direction</li>
      <li><span class="dot" id="dot-generating_cover"></span> Generating album art (Flux 1.1 Pro)</li>
      <li><span class="dot" id="dot-animating_canvas"></span> Animating canvas (Runway Gen-3)</li>
      <li><span class="dot" id="dot-rendering_visualizer"></span> Rendering YouTube visualizer</li>
      <li><span class="dot" id="dot-done"></span> Done</li>
    </ul>
    <p id="progress-msg" style="margin-top:12px;font-size:13px;color:var(--muted)"></p>
  </div>

  <!-- Results -->
  <div class="card" id="results-section" style="margin-top:24px">
    <h2>Results</h2>
    <div class="meta-row" id="meta-row"></div>
    <div class="results-grid" id="results-grid"></div>
  </div>

</div>

<div id="toast"></div>

<script>
// ── Profiles ──────────────────────────────────────────────────────────────────
async function loadProfiles() {
  const res = await fetch('/profiles');
  const profiles = await res.json();
  const sel = document.getElementById('profile-select');
  // Clear existing options (keep default)
  while (sel.options.length > 1) sel.remove(1);
  profiles.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = `${p.name} (${p.cover_count} cover${p.cover_count !== 1 ? 's' : ''})`;
    sel.appendChild(opt);
  });
}

async function saveProfile() {
  const covers = getSelectedCovers();
  if (covers.length === 0) {
    showToast('Upload at least one reference cover before saving a profile.');
    return;
  }
  const name = prompt('Profile name:', 'My Style') || 'My Style';
  const fd = new FormData();
  fd.append('name', name);
  covers.forEach(f => fd.append('covers', f));

  const btn = document.querySelector('.btn-sm');
  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    const res = await fetch('/profiles', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    await loadProfiles();
    // Select new profile
    const data = await res.json().catch(() => ({}));
  } catch(e) {
    showToast('Failed to save profile: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save as new profile';
  }
}

async function deleteProfile() {
  const sel = document.getElementById('profile-select');
  const id = sel.value;
  if (!id) { showToast('No profile selected.'); return; }
  if (!confirm('Delete this profile?')) return;
  await fetch(`/profiles/${id}`, { method: 'DELETE' });
  await loadProfiles();
}

// ── Audio drop zone ──────────────────────────────────────────────────────────
document.getElementById('audio-drop').addEventListener('dragover', e => {
  e.preventDefault();
  e.currentTarget.classList.add('drag-over');
});
document.getElementById('audio-drop').addEventListener('dragleave', e => {
  e.currentTarget.classList.remove('drag-over');
});
document.getElementById('audio-drop').addEventListener('drop', e => {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {
    document.getElementById('audio-input').files = e.dataTransfer.files;
    updateAudioLabel(file.name);
  }
});
document.getElementById('audio-input').addEventListener('change', function() {
  if (this.files[0]) updateAudioLabel(this.files[0].name);
});
function updateAudioLabel(name) {
  const el = document.getElementById('audio-filename');
  el.textContent = name;
  el.style.display = 'block';
}

// ── Cover slots ──────────────────────────────────────────────────────────────
function triggerCoverPick(i) {
  document.getElementById(`cover-${i}`).click();
}
function previewCover(i, input) {
  const slot = input.parentElement;
  if (input.files[0]) {
    const url = URL.createObjectURL(input.files[0]);
    const existing = slot.querySelector('img');
    if (existing) existing.remove();
    const img = document.createElement('img');
    img.src = url;
    slot.querySelector('.plus').style.display = 'none';
    slot.insertBefore(img, slot.querySelector('input').nextSibling);
  }
}
function getSelectedCovers() {
  const files = [];
  for (let i = 0; i < 5; i++) {
    const inp = document.getElementById(`cover-${i}`);
    if (inp && inp.files[0]) files.push(inp.files[0]);
  }
  return files;
}

// ── Generation ───────────────────────────────────────────────────────────────
const STAGE_ORDER = [
  'analyzing_audio','analyzing_style','generating_direction',
  'generating_cover','animating_canvas','rendering_visualizer','done'
];
let _pollTimer = null;

async function startGeneration() {
  const audio = document.getElementById('audio-input').files[0];
  const artist = document.getElementById('artist').value.trim();
  const album  = document.getElementById('album').value.trim();
  if (!audio)  { showToast('Please select an audio file.'); return; }
  if (!artist) { showToast('Please enter the artist name.'); return; }
  if (!album)  { showToast('Please enter the album/track title.'); return; }

  // Reset UI
  document.getElementById('generate-btn').disabled = true;
  document.getElementById('progress-section').style.display = 'block';
  document.getElementById('results-section').style.display = 'none';
  document.querySelectorAll('.dot').forEach(d => { d.className = 'dot'; });
  document.getElementById('progress-msg').textContent = 'Starting…';

  const fd = new FormData();
  fd.append('audio', audio);
  fd.append('artist', artist);
  fd.append('album', album);
  fd.append('draft_mode', document.getElementById('draft-mode').checked ? 'true' : 'false');
  fd.append('skip_canvas', document.getElementById('skip-canvas').checked ? 'true' : 'false');
  fd.append('skip_visualizer', document.getElementById('skip-visualizer').checked ? 'true' : 'false');

  const profileId = document.getElementById('profile-select').value;
  if (profileId) {
    fd.append('profile_id', profileId);
  } else {
    getSelectedCovers().forEach(f => fd.append('covers', f));
  }

  const res = await fetch('/generate', { method: 'POST', body: fd });
  if (!res.ok) {
    const text = await res.text();
    showToast('Error: ' + text);
    document.getElementById('generate-btn').disabled = false;
    return;
  }
  const { job_id } = await res.json();
  _pollTimer = setInterval(() => pollStatus(job_id), 3000);
}

async function pollStatus(jobId) {
  try {
    const res = await fetch(`/status/${jobId}`, { cache: 'no-store' });
    if (!res.ok) return;
    const data = await res.json();

    // Update dots
    const stageIdx = STAGE_ORDER.indexOf(data.stage);
    STAGE_ORDER.forEach((s, i) => {
      const dot = document.getElementById(`dot-${s}`);
      if (!dot) return;
      if (i < stageIdx) dot.className = 'dot done';
      else if (i === stageIdx) dot.className = data.status === 'error' ? 'dot error' : 'dot active';
      else dot.className = 'dot';
    });
    document.getElementById('progress-msg').textContent = data.message || '';

    if (data.status === 'done') {
      clearInterval(_pollTimer);
      showResults(data.result);
      document.getElementById('generate-btn').disabled = false;
    } else if (data.status === 'error') {
      clearInterval(_pollTimer);
      showToast(data.error || 'An error occurred.');
      document.getElementById('generate-btn').disabled = false;
    }
  } catch(e) {
    console.error('Poll error:', e);
  }
}

function showResults(result) {
  document.getElementById('results-section').style.display = 'block';

  // Meta chips
  const metaRow = document.getElementById('meta-row');
  metaRow.innerHTML = '';
  const chips = [
    ['Style', result.art_style],
    ['BPM', result.bpm],
    ['Key', result.key],
    ['Time', result.elapsed + 's'],
  ];
  chips.forEach(([k, v]) => {
    const chip = document.createElement('div');
    chip.className = 'meta-chip';
    chip.innerHTML = `${k}: <span>${v}</span>`;
    metaRow.appendChild(chip);
  });

  // Tagline
  if (result.tagline) {
    const tag = document.createElement('p');
    tag.style.cssText = 'font-style:italic;color:var(--muted);margin-bottom:16px;font-size:13px';
    tag.textContent = '"' + result.tagline + '"';
    metaRow.parentElement.insertBefore(tag, document.getElementById('results-grid'));
  }

  const grid = document.getElementById('results-grid');
  grid.innerHTML = '';
  const files = [
    { key: 'album_cover',       label: 'Album Cover',         spec: '3000×3000 · 300 DPI', icon: '🎨', ext: 'png' },
    { key: 'spotify_canvas',    label: 'Spotify Canvas',      spec: '720×1280 · 8s loop',  icon: '📱', ext: 'mp4' },
    { key: 'youtube_visualizer',label: 'YouTube Visualizer',  spec: '1920×1080 · Full length', icon: '▶️', ext: 'mp4' },
  ];
  files.forEach(f => {
    const path = result[f.key];
    if (!path) return;
    const card = document.createElement('div');
    card.className = 'result-card';
    const isImg = f.ext === 'png';
    const thumbHtml = isImg
      ? `<img src="/output/${path}" alt="${f.label}" loading="lazy">`
      : `<div class="icon">${f.icon}</div>`;
    card.innerHTML = `
      <div class="thumb">${thumbHtml}</div>
      <div class="info">
        <h3>${f.label}</h3>
        <p>${f.spec}</p>
        <a class="btn-dl" href="/output/${path}" download>Download</a>
      </div>
    `;
    grid.appendChild(card);
  });
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(() => { t.style.display = 'none'; }, 5000);
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadProfiles();
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

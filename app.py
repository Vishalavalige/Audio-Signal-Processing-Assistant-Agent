"""
app.py
------
Flask web application front-end for the Audio Signal Processing Agent.
Problem Statement 32

Routes
------
  GET  /                  -- Home page: upload form + batch-analysis button
  POST /analyse           -- Upload a WAV file and return analysis results
  GET  /batch             -- Run batch analysis on all files in sample_audio/
  GET  /generate          -- (Re)generate synthetic test tones
  GET  /api/analyse/<f>   -- JSON API: analyse a file in sample_audio/

Usage
-----
  pip install flask
  python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    url_for,
)

import generate_test_tones
import feature_extractor
import classifier

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB upload limit
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg"}

SAMPLE_DIR = generate_test_tones.OUTPUT_DIR


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _analyse(audio_path: str) -> dict:
    """Extract features and classify; return a serialisable result dict."""
    features = feature_extractor.extract_features(audio_path)
    result   = classifier.classify(features)
    return {
        "filename":        os.path.basename(audio_path),
        "label":           result["label"],
        "urgency":         result["urgency"],
        "reason":          result["reason"],
        "sc_mean_hz":      round(features["sc_mean"], 2),
        "rms_mean":        round(features["rms_mean"], 6),
        "mfcc0_mean":      round(float(features["mfcc_mean"][0]), 3),
    }


# ---------------------------------------------------------------------------
# HTML template (inline – no separate templates/ folder needed)
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Audio Signal Processing Agent</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body   { font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
           background: #f7f8fa; color: #1f2328; font-size: 15px; line-height: 1.6; }
  header { background: #1f2328; color: #fff; padding: 18px 32px; }
  header h1 { font-size: 1.3rem; font-weight: 600; }
  header p  { font-size: 0.85rem; color: #8b949e; margin-top: 2px; }
  main   { max-width: 860px; margin: 32px auto; padding: 0 24px 48px; }
  h2     { font-size: 1.05rem; font-weight: 600; margin-bottom: 12px; color: #1f2328; }
  .card  { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
           padding: 24px; margin-bottom: 24px; }
  label  { display: block; font-size: 0.88rem; color: #57606a; margin-bottom: 6px; }
  input[type=file] { font-size: 0.9rem; }
  .btn   { display: inline-block; padding: 8px 18px; border-radius: 6px;
           font-size: 0.9rem; cursor: pointer; border: none; text-decoration: none; }
  .btn-primary  { background: #3b82d4; color: #fff; }
  .btn-secondary{ background: #e5e7eb; color: #1f2328; margin-left: 8px; }
  .btn-danger   { background: #e5484d; color: #fff; margin-left: 8px; }
  .btn:hover { opacity: 0.88; }
  table  { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th     { text-align: left; padding: 8px 12px; background: #f7f8fa;
           border-bottom: 2px solid #e5e7eb; font-weight: 600; color: #57606a; }
  td     { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
           font-size: 0.8rem; font-weight: 600; }
  .high   { background: #ffe0e0; color: #c0392b; }
  .medium { background: #fff3cd; color: #856404; }
  .low    { background: #d4edda; color: #155724; }
  .reason { font-size: 0.82rem; color: #57606a; }
  .flash  { padding: 12px 16px; border-radius: 6px; margin-bottom: 20px;
            border: 1px solid; }
  .flash.error { background: #ffe0e0; border-color: #f5c6cb; color: #721c24; }
  .flash.info  { background: #d1ecf1; border-color: #bee5eb; color: #0c5460; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
  footer { text-align: center; font-size: 0.78rem; color: #8b949e;
           border-top: 1px solid #e5e7eb; padding: 18px 0; margin-top: 40px; }
</style>
</head>
<body>

<header>
  <h1>Audio Signal Processing Agent</h1>
  <p>Problem Statement 32 &mdash; Acoustic event detection &amp; classification</p>
</header>

<main>

  {% if flash_msg %}
  <div class="flash {{ flash_type }}">{{ flash_msg }}</div>
  {% endif %}

  <!-- Upload card -->
  <div class="card">
    <h2>Analyse an Audio File</h2>
    <form method="POST" action="/analyse" enctype="multipart/form-data">
      <label>Upload a WAV / MP3 / FLAC / OGG file (max 20 MB)</label>
      <input type="file" name="audio_file" accept=".wav,.mp3,.flac,.ogg" required>
      <div class="actions" style="margin-top:14px;">
        <button class="btn btn-primary" type="submit">Analyse</button>
        <a class="btn btn-secondary" href="/batch">Batch — all test tones</a>
        <a class="btn btn-danger"    href="/generate">Regenerate test tones</a>
      </div>
    </form>
  </div>

  <!-- Results card -->
  {% if results %}
  <div class="card">
    <h2>Results</h2>
    <table>
      <thead>
        <tr>
          <th>File</th>
          <th>Event</th>
          <th>Urgency</th>
          <th>SC mean (Hz)</th>
          <th>RMS mean</th>
          <th>MFCC[0] mean</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {% for r in results %}
        <tr>
          <td>{{ r.filename }}</td>
          <td><strong>{{ r.label }}</strong></td>
          <td><span class="badge {{ r.urgency }}">{{ r.urgency|upper }}</span></td>
          <td>{{ r.sc_mean_hz }}</td>
          <td>{{ r.rms_mean }}</td>
          <td>{{ r.mfcc0_mean }}</td>
          <td class="reason">{{ r.reason }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

</main>

<footer>Made with IBM Bob</footer>
</body>
</html>
"""


def _render(results=None, flash_msg=None, flash_type="info"):
    return render_template_string(
        _HTML,
        results=results or [],
        flash_msg=flash_msg,
        flash_type=flash_type,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return _render()


@app.route("/analyse", methods=["POST"])
def analyse_upload():
    """Accept a file upload, run analysis, return results page."""
    file = request.files.get("audio_file")
    if not file or file.filename == "":
        return _render(flash_msg="No file selected.", flash_type="error")

    if not _allowed(file.filename):
        return _render(
            flash_msg=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            flash_type="error",
        )

    # Save to a temp file so librosa can read it
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        result = _analyse(tmp_path)
        # Show the original filename, not the temp name
        result["filename"] = file.filename
    except Exception as exc:
        return _render(flash_msg=f"Analysis failed: {exc}", flash_type="error")
    finally:
        os.unlink(tmp_path)

    return _render(results=[result])


@app.route("/batch")
def batch():
    """Analyse every WAV file in the sample_audio directory."""
    if not os.path.isdir(SAMPLE_DIR):
        return _render(
            flash_msg="sample_audio/ directory not found. Click 'Regenerate test tones' first.",
            flash_type="error",
        )

    wav_files = sorted(
        f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".wav")
    )
    if not wav_files:
        return _render(
            flash_msg="No WAV files found in sample_audio/. Click 'Regenerate test tones' first.",
            flash_type="error",
        )

    results = []
    errors  = []
    for fname in wav_files:
        path = os.path.join(SAMPLE_DIR, fname)
        try:
            results.append(_analyse(path))
        except Exception as exc:
            errors.append(f"{fname}: {exc}")

    flash = f"Analysed {len(results)} file(s)."
    if errors:
        flash += " Errors: " + "; ".join(errors)

    return _render(results=results, flash_msg=flash)


@app.route("/generate")
def generate():
    """Regenerate synthetic test tones and redirect to batch analysis."""
    try:
        generate_test_tones.main()
    except Exception as exc:
        return _render(flash_msg=f"Generation failed: {exc}", flash_type="error")
    return redirect(url_for("batch"))


@app.route("/api/analyse/<filename>")
def api_analyse(filename: str):
    """JSON API: analyse a named file inside sample_audio/."""
    path = os.path.join(SAMPLE_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({"error": f"File not found: {filename}"}), 404
    try:
        result = _analyse(path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Auto-generate test tones if the sample directory is empty / missing
    has_tones = os.path.isdir(SAMPLE_DIR) and any(
        f.endswith(".wav") for f in os.listdir(SAMPLE_DIR)
    )
    if not has_tones:
        print("[app] Generating test tones …")
        generate_test_tones.main()

    print("[app] Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

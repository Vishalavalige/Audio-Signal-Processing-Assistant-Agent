"""
app.py
------
Flask web-application dashboard for the Audio Signal Processing Agent.
Problem Statement 32

Routes
------
  GET  /                  -- Dashboard (stats + history + upload form)
  POST /analyse           -- Upload a WAV/MP3/FLAC/OGG file → analyse + redirect
  GET  /batch             -- Run batch analysis on all files in sample_audio/
  GET  /generate          -- (Re)generate synthetic test tones → redirect to batch
  GET  /api/analyse/<f>   -- JSON API: analyse a file in sample_audio/
  GET  /api/history       -- JSON API: return current in-memory history

Usage
-----
  pip install flask
  python app.py
  → open http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from collections import deque

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

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

# In-memory history (newest first, capped at 50 entries)
_history: deque[dict] = deque(maxlen=50)


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _analyse(audio_path: str) -> dict:
    """Extract features and classify; return a serialisable result dict."""
    features = feature_extractor.extract_features(audio_path)
    result   = classifier.classify(features)
    return {
        "filename":   os.path.basename(audio_path),
        "label":      result["label"],
        "urgency":    result["urgency"],
        "reason":     result["reason"],
        "sc_mean_hz": round(features["sc_mean"], 2),
        "rms_mean":   round(features["rms_mean"], 6),
        "mfcc0_mean": round(float(features["mfcc_mean"][0]), 3),
    }


# ---------------------------------------------------------------------------
# HTML Dashboard (single inline template)
# ---------------------------------------------------------------------------

_DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Audio Signal Processing — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body   { font-family: -apple-system,"Segoe UI",system-ui,sans-serif;
           background: #f0f2f5; color: #1f2328; font-size: 14px; line-height: 1.6; }

  /* ── Header ───────────────────────────────────────────────────── */
  header { background: #1f2328; color: #fff; padding: 14px 28px;
           display: flex; align-items: center; gap: 14px; }
  header .logo { font-size: 1.5rem; }
  header h1   { font-size: 1.15rem; font-weight: 700; }
  header p    { font-size: 0.8rem; color: #8b949e; }
  header nav  { margin-left: auto; display: flex; gap: 6px; }
  header nav a { color: #cdd9e5; font-size: 0.82rem; text-decoration: none;
                 padding: 5px 12px; border-radius: 5px; border: 1px solid #444d56; }
  header nav a:hover { background: #2d333b; }

  /* ── Layout ───────────────────────────────────────────────────── */
  .page   { max-width: 1100px; margin: 0 auto; padding: 24px 20px 56px; }
  .grid2  { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid3  { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }
  .grid4  { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }
  @media(max-width:750px){ .grid2,.grid3,.grid4{ grid-template-columns:1fr; } }

  /* ── Card ─────────────────────────────────────────────────────── */
  .card   { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
            padding: 20px; }
  .card-title { font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
                letter-spacing: .06em; color: #57606a; margin-bottom: 10px; }

  /* ── Stat cards ───────────────────────────────────────────────── */
  .stat-val   { font-size: 2rem; font-weight: 700; line-height: 1.1; }
  .stat-sub   { font-size: 0.78rem; color: #57606a; margin-top: 3px; }
  .stat-high  { color: #c0392b; }
  .stat-med   { color: #856404; }
  .stat-low   { color: #155724; }
  .stat-total { color: #3b82d4; }

  /* ── Flash ────────────────────────────────────────────────────── */
  .flash { padding: 10px 16px; border-radius: 6px; margin-bottom: 18px;
           border: 1px solid; font-size: 0.88rem; }
  .flash.error { background:#ffe0e0; border-color:#f5c6cb; color:#721c24; }
  .flash.info  { background:#d1ecf1; border-color:#bee5eb; color:#0c5460; }

  /* ── Upload form ──────────────────────────────────────────────── */
  .upload-area { border: 2px dashed #d0d7de; border-radius: 8px;
                 padding: 24px; text-align: center; cursor: pointer;
                 transition: border-color .2s; }
  .upload-area:hover { border-color: #3b82d4; }
  .upload-area input[type=file] { display:none; }
  .upload-area .upload-icon { font-size: 2rem; color: #8b949e; }
  .upload-area p { font-size: 0.85rem; color: #57606a; margin-top: 6px; }
  #chosen-file  { font-size: 0.82rem; color: #3b82d4; margin-top: 6px; font-weight:600; }
  .btn-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
  .btn     { display:inline-block; padding:7px 18px; border-radius:6px;
             font-size:0.88rem; cursor:pointer; border:none; text-decoration:none;
             font-weight:500; }
  .btn-primary   { background:#3b82d4; color:#fff; }
  .btn-secondary { background:#e5e7eb; color:#1f2328; }
  .btn-danger    { background:#e5484d; color:#fff; }
  .btn:hover     { opacity:.85; }

  /* ── Results table ────────────────────────────────────────────── */
  .tbl-wrap { overflow-x:auto; }
  table  { width:100%; border-collapse:collapse; font-size:0.88rem; }
  th     { text-align:left; padding:8px 10px; background:#f7f8fa;
           border-bottom:2px solid #e5e7eb; font-weight:600; color:#57606a;
           white-space:nowrap; }
  td     { padding:8px 10px; border-bottom:1px solid #f0f0f0; vertical-align:top; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:#fafbfc; }
  .badge { display:inline-block; padding:2px 9px; border-radius:10px;
           font-size:0.78rem; font-weight:700; letter-spacing:.03em; }
  .badge.high   { background:#ffe0e0; color:#c0392b; }
  .badge.medium { background:#fff3cd; color:#856404; }
  .badge.low    { background:#d4edda; color:#155724; }
  .reason-cell  { font-size:0.78rem; color:#57606a; max-width:320px; }
  .num          { font-family: "SF Mono", Consolas, monospace; font-size:0.82rem; }

  /* ── Chart containers ────────────────────────────────────────── */
  .chart-wrap { position:relative; height:220px; }

  /* ── History empty state ─────────────────────────────────────── */
  .empty { text-align:center; color:#8b949e; padding:30px 0; font-size:0.88rem; }

  /* ── Footer ──────────────────────────────────────────────────── */
  footer { text-align:center; font-size:0.75rem; color:#8b949e;
           border-top:1px solid #e5e7eb; padding:16px 0; margin-top:40px; }
  .section-label { font-size:0.92rem; font-weight:700; margin:28px 0 12px;
                   color:#1f2328; }
</style>
</head>
<body>

<header>
  <span class="logo">🎵</span>
  <div>
    <h1>Audio Signal Processing Agent</h1>
    <p>Acoustic event detection &amp; classification — Problem Statement 32</p>
  </div>
  <nav>
    <a href="/">Dashboard</a>
    <a href="/batch">Batch Analyse</a>
    <a href="/generate">Regenerate Tones</a>
  </nav>
</header>

<div class="page">

  {% if flash_msg %}
  <div class="flash {{ flash_type }}">{{ flash_msg }}</div>
  {% endif %}

  <!-- ── Stat Cards ──────────────────────────────────────────────── -->
  <div class="grid4" style="margin-bottom:16px;">
    <div class="card">
      <div class="card-title">Total Analysed</div>
      <div class="stat-val stat-total">{{ stats.total }}</div>
      <div class="stat-sub">files in session</div>
    </div>
    <div class="card">
      <div class="card-title">🔴 High Urgency</div>
      <div class="stat-val stat-high">{{ stats.high }}</div>
      <div class="stat-sub">glass_breaking</div>
    </div>
    <div class="card">
      <div class="card-title">🟡 Medium Urgency</div>
      <div class="stat-val stat-med">{{ stats.medium }}</div>
      <div class="stat-sub">dog_barking</div>
    </div>
    <div class="card">
      <div class="card-title">🟢 Low Urgency</div>
      <div class="stat-val stat-low">{{ stats.low }}</div>
      <div class="stat-sub">background_hum</div>
    </div>
  </div>

  <!-- ── Charts ──────────────────────────────────────────────────── -->
  <div class="grid2" style="margin-bottom:16px;">
    <div class="card">
      <div class="card-title">Event Distribution</div>
      <div class="chart-wrap"><canvas id="pieChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Spectral Centroid (Hz) — Last 10</div>
      <div class="chart-wrap"><canvas id="scChart"></canvas></div>
    </div>
  </div>

  <!-- ── Upload ──────────────────────────────────────────────────── -->
  <div class="section-label">Analyse Audio</div>
  <div class="card" style="margin-bottom:16px;">
    <form method="POST" action="/analyse" enctype="multipart/form-data" id="uploadForm">
      <div class="upload-area" onclick="document.getElementById('fileInput').click()">
        <div class="upload-icon">📂</div>
        <p>Click to select a WAV / MP3 / FLAC / OGG file (max 20 MB)</p>
        <div id="chosen-file"></div>
        <input type="file" name="audio_file" id="fileInput"
               accept=".wav,.mp3,.flac,.ogg"
               onchange="document.getElementById('chosen-file').textContent=this.files[0]?.name||''; document.getElementById('uploadForm').submit()">
      </div>
      <div class="btn-row">
        <a class="btn btn-secondary" href="/batch">⚡ Batch — all test tones</a>
        <a class="btn btn-danger"    href="/generate">🔄 Regenerate test tones</a>
      </div>
    </form>
  </div>

  <!-- ── Latest Result ───────────────────────────────────────────── -->
  {% if results %}
  <div class="section-label">Latest Result</div>
  <div class="card" style="margin-bottom:16px;">
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>File</th><th>Event</th><th>Urgency</th>
            <th>SC mean (Hz)</th><th>RMS mean</th><th>MFCC[0]</th><th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {% for r in results %}
          <tr>
            <td>{{ r.filename }}</td>
            <td><strong>{{ r.label }}</strong></td>
            <td><span class="badge {{ r.urgency }}">{{ r.urgency|upper }}</span></td>
            <td class="num">{{ r.sc_mean_hz }}</td>
            <td class="num">{{ r.rms_mean }}</td>
            <td class="num">{{ r.mfcc0_mean }}</td>
            <td class="reason-cell">{{ r.reason }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
  {% endif %}

  <!-- ── Session History ─────────────────────────────────────────── -->
  <div class="section-label">Session History</div>
  <div class="card">
    {% if history %}
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>File</th><th>Event</th><th>Urgency</th>
            <th>SC mean (Hz)</th><th>RMS mean</th><th>MFCC[0]</th>
          </tr>
        </thead>
        <tbody>
          {% for r in history %}
          <tr>
            <td class="num" style="color:#8b949e;">{{ loop.index }}</td>
            <td>{{ r.filename }}</td>
            <td>{{ r.label }}</td>
            <td><span class="badge {{ r.urgency }}">{{ r.urgency|upper }}</span></td>
            <td class="num">{{ r.sc_mean_hz }}</td>
            <td class="num">{{ r.rms_mean }}</td>
            <td class="num">{{ r.mfcc0_mean }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% else %}
    <div class="empty">No analyses yet — upload a file or run batch analysis to get started.</div>
    {% endif %}
  </div>

</div><!-- /.page -->

<footer>Made with IBM Bob</footer>

<script>
// ── Chart data injected by Flask ──────────────────────────────────
const pieData    = {{ pie_data   | tojson }};
const scLabels   = {{ sc_labels  | tojson }};
const scValues   = {{ sc_values  | tojson }};

// ── Pie / Doughnut — event distribution ──────────────────────────
(function() {
  const ctx = document.getElementById('pieChart');
  if (!ctx) return;
  const total = pieData.high + pieData.medium + pieData.low;
  if (total === 0) {
    ctx.parentElement.innerHTML =
      '<div class="empty" style="height:220px;display:flex;align-items:center;justify-content:center;">No data yet</div>';
    return;
  }
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['glass_breaking', 'dog_barking', 'background_hum'],
      datasets: [{
        data: [pieData.high, pieData.medium, pieData.low],
        backgroundColor: ['#e5484d','#f59f00','#30a46c'],
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { font: { size: 11 }, padding: 14 } }
      }
    }
  });
})();

// ── Line — spectral centroid history ─────────────────────────────
(function() {
  const ctx = document.getElementById('scChart');
  if (!ctx) return;
  if (scValues.length === 0) {
    ctx.parentElement.innerHTML =
      '<div class="empty" style="height:220px;display:flex;align-items:center;justify-content:center;">No data yet</div>';
    return;
  }
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: scLabels,
      datasets: [{
        label: 'SC mean (Hz)',
        data: scValues,
        backgroundColor: scValues.map(v =>
          v > 2000 ? 'rgba(229,72,77,0.7)' :
          v >  500 ? 'rgba(245,159,0,0.7)' :
                     'rgba(48,164,108,0.7)'
        ),
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { ticks: { font: { size: 10 }, maxRotation: 35 } },
        y: { beginAtZero: true,
             title: { display: true, text: 'Hz', font: { size: 11 } } }
      },
      plugins: { legend: { display: false } }
    }
  });
})();
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats() -> dict:
    h = list(_history)
    return {
        "total":  len(h),
        "high":   sum(1 for r in h if r["urgency"] == "high"),
        "medium": sum(1 for r in h if r["urgency"] == "medium"),
        "low":    sum(1 for r in h if r["urgency"] == "low"),
    }


def _chart_data() -> tuple[dict, list, list]:
    """Return pie_data dict and last-10 SC label/value lists."""
    s = _stats()
    pie = {"high": s["high"], "medium": s["medium"], "low": s["low"]}
    last10 = list(_history)[-10:]
    labels = [r["filename"] for r in last10]
    values = [r["sc_mean_hz"] for r in last10]
    return pie, labels, values


def _render(results=None, flash_msg=None, flash_type="info"):
    pie, sc_labels, sc_values = _chart_data()
    return render_template_string(
        _DASHBOARD,
        results=results or [],
        history=list(_history),
        stats=_stats(),
        pie_data=pie,
        sc_labels=sc_labels,
        sc_values=sc_values,
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
    file = request.files.get("audio_file")
    if not file or file.filename == "":
        return _render(flash_msg="No file selected.", flash_type="error")

    if not _allowed(file.filename):
        return _render(
            flash_msg=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            flash_type="error",
        )

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        result = _analyse(tmp_path)
        result["filename"] = file.filename
    except Exception as exc:
        return _render(flash_msg=f"Analysis failed: {exc}", flash_type="error")
    finally:
        os.unlink(tmp_path)

    _history.appendleft(result)
    return _render(results=[result], flash_msg=f"✅ Analysed: {result['filename']}")


@app.route("/batch")
def batch():
    if not os.path.isdir(SAMPLE_DIR):
        return _render(
            flash_msg="sample_audio/ not found. Click 'Regenerate test tones' first.",
            flash_type="error",
        )

    wav_files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".wav"))
    if not wav_files:
        return _render(
            flash_msg="No WAV files found in sample_audio/. Click 'Regenerate test tones' first.",
            flash_type="error",
        )

    results, errors = [], []
    for fname in wav_files:
        path = os.path.join(SAMPLE_DIR, fname)
        try:
            r = _analyse(path)
            results.append(r)
            _history.appendleft(r)
        except Exception as exc:
            errors.append(f"{fname}: {exc}")

    flash = f"✅ Analysed {len(results)} file(s)."
    if errors:
        flash += "  ⚠️ Errors: " + "; ".join(errors)

    return _render(results=results, flash_msg=flash)


@app.route("/generate")
def generate():
    try:
        generate_test_tones.main()
    except Exception as exc:
        return _render(flash_msg=f"Generation failed: {exc}", flash_type="error")
    return redirect(url_for("batch"))


@app.route("/api/analyse/<filename>")
def api_analyse(filename: str):
    path = os.path.join(SAMPLE_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({"error": f"File not found: {filename}"}), 404
    try:
        result = _analyse(path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    _history.appendleft(result)
    return jsonify(result)


@app.route("/api/history")
def api_history():
    return jsonify(list(_history))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    has_tones = os.path.isdir(SAMPLE_DIR) and any(
        f.endswith(".wav") for f in os.listdir(SAMPLE_DIR)
    )
    if not has_tones:
        print("[app] Generating test tones …")
        generate_test_tones.main()

    print("[app] Dashboard at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

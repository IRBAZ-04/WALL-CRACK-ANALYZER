"""
app.py
------
Self-contained Flask web application for Wall Crack Analyzer.
Classical Image Processing: Pure mathematical algorithms from scratch (NumPy, Scipy, Pillow).
Contains embedded HTML, CSS, and JavaScript in a single template string.
"""
import os
import io
import base64
import time
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

import source

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ndarray_to_base64_png(img_array: np.ndarray) -> str:
    """Converts a 2D or 3D NumPy array into a base64 PNG data URL."""
    if len(img_array.shape) == 2:
        pil_img = Image.fromarray(img_array.astype(np.uint8), mode="L")
    else:
        pil_img = Image.fromarray(img_array.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Wall Crack Analyzer - Classical Image Processing</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0b0f17;
      --bg-card: rgba(18, 24, 38, 0.75);
      --bg-card-hover: rgba(26, 34, 54, 0.85);
      --border-color: rgba(56, 189, 248, 0.15);
      --border-highlight: rgba(56, 189, 248, 0.4);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-glow: rgba(56, 189, 248, 0.25);
      --accent: #818cf8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --radius-lg: 16px;
      --radius-md: 10px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at 15% 15%, rgba(30, 41, 59, 0.8) 0%, var(--bg-dark) 50%, #05080e 100%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 2rem 1.5rem 4rem;
      position: relative;
      overflow-x: hidden;
    }

    /* Ambient background glows */
    body::before {
      content: "";
      position: fixed;
      top: -150px;
      left: 20%;
      width: 500px;
      height: 500px;
      background: radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, transparent 70%);
      pointer-events: none;
      z-index: 0;
    }

    .container {
      max-width: 1320px;
      margin: 0 auto;
      position: relative;
      z-index: 1;
    }

    /* Header */
    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }

    .badge-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(56, 189, 248, 0.1);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: var(--primary);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 0.9rem;
    }

    .badge-pill span.dot {
      width: 7px;
      height: 7px;
      background: var(--primary);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--primary);
    }

    h1 {
      font-size: 2.6rem;
      font-weight: 800;
      background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
      letter-spacing: -0.02em;
    }

    .subtitle {
      font-size: 1.12rem;
      color: var(--text-muted);
      font-weight: 500;
    }

    .subtitle strong {
      color: var(--primary);
      font-weight: 600;
    }

    /* Glass Card */
    .card {
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 1.75rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }

    .card:hover {
      border-color: var(--border-highlight);
    }

    /* Upload & Configuration Form */
    .controls-grid {
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 1.75rem;
      margin-bottom: 2rem;
    }

    @media (max-width: 900px) {
      .controls-grid {
        grid-template-columns: 1fr;
      }
    }

    /* Drop Zone */
    .dropzone {
      border: 2px dashed rgba(56, 189, 248, 0.3);
      border-radius: var(--radius-md);
      padding: 2.2rem 1.5rem;
      text-align: center;
      cursor: pointer;
      background: rgba(15, 23, 42, 0.4);
      transition: all 0.2s ease;
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 220px;
    }

    .dropzone:hover, .dropzone.dragover {
      border-color: var(--primary);
      background: rgba(56, 189, 248, 0.05);
      box-shadow: inset 0 0 20px rgba(56, 189, 248, 0.1);
    }

    .dropzone input[type="file"] {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
      width: 100%;
      height: 100%;
    }

    .dropzone-icon {
      font-size: 2.8rem;
      margin-bottom: 0.6rem;
      color: var(--primary);
    }

    .dropzone-text {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 0.25rem;
    }

    .dropzone-sub {
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .preview-box {
      margin-top: 1rem;
      max-height: 160px;
      display: none;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border-color);
    }

    .preview-box img {
      max-width: 100%;
      max-height: 160px;
      object-fit: contain;
      display: block;
    }

    /* Parameters & Action panel */
    .form-group {
      margin-bottom: 1.25rem;
    }

    .form-label {
      display: block;
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 0.45rem;
    }

    .form-hint {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.3rem;
    }

    select, input[type="number"], input[type="text"] {
      width: 100%;
      background: rgba(11, 15, 23, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: var(--radius-md);
      padding: 0.75rem 1rem;
      color: #fff;
      font-family: inherit;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    select:focus, input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--primary-glow);
    }

    /* Sample Image Quick Buttons */
    .sample-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 0.5rem;
    }

    .chip-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: var(--text-muted);
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .chip-btn:hover {
      background: rgba(56, 189, 248, 0.15);
      border-color: var(--primary);
      color: #fff;
    }

    /* Analyze Button */
    .btn-analyze {
      width: 100%;
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      color: #ffffff;
      border: 1px solid rgba(56, 189, 248, 0.5);
      padding: 0.95rem 1.5rem;
      border-radius: var(--radius-md);
      font-family: inherit;
      font-size: 1.05rem;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 4px 18px rgba(2, 132, 199, 0.35);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: all 0.2s ease;
      margin-top: 1.5rem;
    }

    .btn-analyze:hover:not(:disabled) {
      background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
      box-shadow: 0 6px 24px rgba(14, 165, 233, 0.5);
      transform: translateY(-1px);
    }

    .btn-analyze:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    .spinner {
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255, 255, 255, 0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: none;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Warning Banner */
    .warning-banner {
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.3);
      border-left: 4px solid var(--warning);
      padding: 1rem 1.25rem;
      border-radius: var(--radius-md);
      color: #fde68a;
      font-size: 0.9rem;
      line-height: 1.45;
      margin-bottom: 2rem;
      display: flex;
      align-items: flex-start;
      gap: 12px;
    }

    .warning-banner svg {
      flex-shrink: 0;
      margin-top: 2px;
    }

    /* Metrics Grid */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 2.25rem;
    }

    .metric-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1.1rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }

    .metric-title {
      font-size: 0.78rem;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--text-muted);
      letter-spacing: 0.05em;
      margin-bottom: 0.4rem;
    }

    .metric-value {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.4rem;
      font-weight: 700;
      color: #fff;
    }

    .metric-sub {
      font-size: 0.78rem;
      color: var(--primary);
      margin-top: 0.25rem;
      font-family: 'JetBrains Mono', monospace;
    }

    .severity-badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.95rem;
      font-weight: 800;
      letter-spacing: 0.04em;
    }

    .severity-CRITICAL { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .severity-HIGH { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid #f97316; }
    .severity-MEDIUM { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .severity-LOW { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .severity-NONE { background: rgba(148, 163, 184, 0.2); color: #94a3b8; border: 1px solid #64748b; }

    /* Dashboard Gallery Panels */
    .section-title {
      font-size: 1.45rem;
      font-weight: 700;
      margin-bottom: 1.25rem;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1.25rem;
    }

    @media (max-width: 1100px) {
      .gallery-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }

    @media (max-width: 600px) {
      .gallery-grid {
        grid-template-columns: 1fr;
      }
    }

    .panel-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.2s ease, border-color 0.2s ease;
      cursor: pointer;
    }

    .panel-card:hover {
      transform: translateY(-3px);
      border-color: var(--primary);
    }

    .panel-header {
      padding: 0.75rem 1rem;
      background: rgba(15, 23, 42, 0.7);
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .panel-name {
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--text-main);
    }

    .panel-tag {
      font-size: 0.72rem;
      padding: 2px 7px;
      border-radius: 4px;
      background: rgba(56, 189, 248, 0.12);
      color: var(--primary);
      font-family: 'JetBrains Mono', monospace;
    }

    .panel-img-wrap {
      background: #000;
      aspect-ratio: 4 / 3;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .panel-img-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    /* Modal for Full Image View */
    .modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.88);
      backdrop-filter: blur(8px);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }

    .modal.active {
      display: flex;
    }

    .modal-content {
      max-width: 90vw;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .modal-content img {
      max-width: 88vw;
      max-height: 80vh;
      border-radius: 8px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.8);
      border: 1px solid var(--border-highlight);
    }

    .modal-close {
      position: absolute;
      top: 20px;
      right: 25px;
      font-size: 2rem;
      color: #fff;
      cursor: pointer;
      opacity: 0.8;
      transition: opacity 0.2s;
    }

    .modal-close:hover {
      opacity: 1;
    }

    /* Error and Info Notices */
    .alert {
      padding: 1rem 1.25rem;
      border-radius: var(--radius-md);
      margin-bottom: 1.5rem;
      font-size: 0.95rem;
      display: none;
    }

    .alert-error {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #fca5a5;
    }

    .alert-info {
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid rgba(56, 189, 248, 0.4);
      color: #bae6fd;
    }

    footer {
      text-align: center;
      margin-top: 4rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge-pill">
        <span class="dot"></span> 100% From-Scratch Algorithms &bull; Zero OpenCV
      </div>
      <h1>Wall Crack Analyzer</h1>
      <p class="subtitle"><strong>Wall Crack Analyzer</strong>, classical image processing, not AI</p>
    </header>

    <!-- Error/Notice Display -->
    <div id="errorAlert" class="alert alert-error"></div>
    <div id="infoAlert" class="alert alert-info"></div>

    <!-- Main Input Form & Settings -->
    <div class="controls-grid">
      <!-- Upload Box -->
      <div class="card">
        <label class="form-label">Wall Photograph (JPG / PNG)</label>
        <div class="dropzone" id="dropzone">
          <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.bmp,.webp">
          <div class="dropzone-icon">📷</div>
          <div class="dropzone-text" id="dropzoneText">Drag and drop wall photo here</div>
          <div class="dropzone-sub">or click to browse from device</div>
          <div class="preview-box" id="previewBox">
            <img id="previewImg" src="" alt="Upload preview">
          </div>
        </div>

        <div style="margin-top: 1rem;">
          <div class="form-label" style="font-size: 0.8rem; margin-bottom: 0.3rem;">Or try campus samples:</div>
          <div class="sample-chips">
            <button class="chip-btn" onclick="loadSample('CAMPUS_ACADEMIC_001.jpg', 'Academic Block Fissure')">Campus Academic Block</button>
            <button class="chip-btn" onclick="loadSample('CAMPUS_HOSTEL_002.jpg', 'Hostel Corridor Wall')">Hostel Wall</button>
            <button class="chip-btn" onclick="loadSample('CAMPUS_CLEAN_007.jpg', 'Library Sound Wall')">Clean Sound Wall</button>
          </div>
        </div>
      </div>

      <!-- Parameter Selection -->
      <div class="card" style="display: flex; flex-direction: column; justify-content: space-between;">
        <div>
          <div class="form-group">
            <label class="form-label" for="algoSelect">Edge Detection Method</label>
            <select id="algoSelect">
              <option value="canny" selected>Canny (Textbook 5-Stage with Hysteresis)</option>
              <option value="sobel">Sobel Operator (Discrete Gx, Gy Magnitude)</option>
              <option value="prewitt">Prewitt Operator (Smoothed Difference)</option>
              <option value="laplacian">Laplacian of Gaussian (2nd Spatial Derivative)</option>
              <option value="fusion">Mathematical Multi-Operator Fusion</option>
            </select>
            <div class="form-hint">Implemented strictly with pure 2D convolution and NumPy array matrix math.</div>
          </div>

          <div class="form-group">
            <label class="form-label" for="calibInput">Centimeters Per Pixel (Optional Calibration)</label>
            <input type="number" step="0.001" min="0.0001" id="calibInput" placeholder="e.g. 0.05 (leave blank if uncalibrated)">
            <div class="form-hint" id="calibHint">If empty, physical measurements will show "Calibration not provided".</div>
          </div>
        </div>

        <button class="btn-analyze" id="analyzeBtn" onclick="runAnalysis()">
          <div class="spinner" id="btnSpinner"></div>
          <span id="btnText">Analyze Crack Structure</span>
        </button>
      </div>
    </div>

    <!-- Mandatory Safety Warning Banner -->
    <div class="warning-banner">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
      </svg>
      <div>
        <strong>Visual Support Notice:</strong> Severity is a visual decision-support label, not a structural safety rating. Pixel values are not physical units without calibration.
      </div>
    </div>

    <!-- Results Section (hidden until analyzed) -->
    <div id="resultsContainer" style="display: none;">
      <!-- Metrics Dashboard -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-title">Structural Severity</div>
          <div class="metric-value"><span id="metricSeverity" class="severity-badge severity-NONE">---</span></div>
          <div class="metric-sub" id="metricOrientation">Orientation: ---</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Crack Length</div>
          <div class="metric-value" id="metricLengthPx">0 px</div>
          <div class="metric-sub" id="metricLengthCm">Calibration not provided</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Average Width</div>
          <div class="metric-value" id="metricAvgWidthPx">0 px</div>
          <div class="metric-sub" id="metricAvgWidthCm">Calibration not provided</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Maximum Width</div>
          <div class="metric-value" id="metricMaxWidthPx">0 px</div>
          <div class="metric-sub" id="metricMaxWidthCm">Calibration not provided</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Mask Coverage / Damage</div>
          <div class="metric-value" id="metricDamagePct">0.00%</div>
          <div class="metric-sub" id="metricCrackAreaPx">0 pixels</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Lighting Detected</div>
          <div class="metric-value" id="metricLighting" style="font-size: 1.15rem; text-transform: capitalize;">---</div>
          <div class="metric-sub" id="metricSegmentation">Method: ---</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Classical Confidence</div>
          <div class="metric-value" id="metricConfidence">0.0%</div>
          <div class="metric-sub">Pure Geometric PCA</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Pipeline Runtime</div>
          <div class="metric-value" id="metricRuntime">0.000s</div>
          <div class="metric-sub" id="metricOperator">Operator: Canny</div>
        </div>
      </div>

      <!-- 8-Panel Image Dashboard Gallery -->
      <div class="section-title">
        <span>Dashboard Inspection Panels</span>
        <span style="font-size: 0.85rem; font-weight: 500; color: var(--text-muted);">(Click any panel for high-resolution view)</span>
      </div>

      <div class="gallery-grid">
        <div class="panel-card" onclick="openModal('Original Image', 'panelOriginal')">
          <div class="panel-header">
            <span class="panel-name">1. Original Image</span>
            <span class="panel-tag">RAW INPUT</span>
          </div>
          <div class="panel-img-wrap"><img id="panelOriginal" src="" alt="Original"></div>
        </div>

        <div class="panel-card" onclick="openModal('Contrast Enhanced', 'panelEnhanced')">
          <div class="panel-header">
            <span class="panel-name">2. Enhanced</span>
            <span class="panel-tag">HISTOGRAM/CLAHE</span>
          </div>
          <div class="panel-img-wrap"><img id="panelEnhanced" src="" alt="Enhanced"></div>
        </div>

        <div class="panel-card" onclick="openModal('Selected Edge Map', 'panelEdgeMap')">
          <div class="panel-header">
            <span class="panel-name" id="edgeMapTitle">3. Edge Map</span>
            <span class="panel-tag" id="edgeMapTag">CANNY</span>
          </div>
          <div class="panel-img-wrap"><img id="panelEdgeMap" src="" alt="Edge Map"></div>
        </div>

        <div class="panel-card" onclick="openModal('Morphologically Cleaned Mask', 'panelCleanedMask')">
          <div class="panel-header">
            <span class="panel-name">4. Cleaned Mask</span>
            <span class="panel-tag">OTSU + MORPHOLOGY</span>
          </div>
          <div class="panel-img-wrap"><img id="panelCleanedMask" src="" alt="Cleaned Mask"></div>
        </div>

        <div class="panel-card" onclick="openModal('Crack Skeleton Centerline', 'panelSkeleton')">
          <div class="panel-header">
            <span class="panel-name">5. Skeleton Centerline</span>
            <span class="panel-tag">1-PX TOPOLOGY</span>
          </div>
          <div class="panel-img-wrap"><img id="panelSkeleton" src="" alt="Skeleton"></div>
        </div>

        <div class="panel-card" onclick="openModal('Width Intensity Heatmap', 'panelHeatmap')">
          <div class="panel-header">
            <span class="panel-name">6. Width Heatmap</span>
            <span class="panel-tag">EUCLIDEAN DISTANCE</span>
          </div>
          <div class="panel-img-wrap"><img id="panelHeatmap" src="" alt="Heatmap"></div>
        </div>

        <div class="panel-card" onclick="openModal('Multi-Operator Fusion Map', 'panelFusion')">
          <div class="panel-header">
            <span class="panel-name">7. Fusion Map</span>
            <span class="panel-tag">GRADIENT WEIGHTED</span>
          </div>
          <div class="panel-img-wrap"><img id="panelFusion" src="" alt="Fusion Map"></div>
        </div>

        <div class="panel-card" onclick="openModal('Final Annotated Overlay', 'panelAnnotated')">
          <div class="panel-header">
            <span class="panel-name">8. Annotated Overlay</span>
            <span class="panel-tag">FISSURE + HUD</span>
          </div>
          <div class="panel-img-wrap"><img id="panelAnnotated" src="" alt="Annotated"></div>
        </div>
      </div>
    </div>

    <!-- Modal for Full-Resolution Preview -->
    <div class="modal" id="imageModal" onclick="closeModal()">
      <div class="modal-close">&times;</div>
      <div class="modal-content" onclick="event.stopPropagation()">
        <h3 id="modalTitle" style="color: #fff; margin-bottom: 0.75rem; font-size: 1.1rem;"></h3>
        <img id="modalImg" src="" alt="Full view">
        <a id="downloadLink" href="" download="crack_analysis.png" class="chip-btn" style="margin-top: 1rem; text-decoration: none; display: inline-block;">Download PNG</a>
      </div>
    </div>

    <footer>
      Wall Crack Analyzer &bull; Pure Mathematical Image Processing Pipeline &bull; Semester 5 Project
    </footer>
  </div>

  <script>
    let currentFile = null;

    const fileInput = document.getElementById('fileInput');
    const dropzone = document.getElementById('dropzone');
    const dropzoneText = document.getElementById('dropzoneText');
    const previewBox = document.getElementById('previewBox');
    const previewImg = document.getElementById('previewImg');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnText = document.getElementById('btnText');
    const errorAlert = document.getElementById('errorAlert');
    const infoAlert = document.getElementById('infoAlert');
    const resultsContainer = document.getElementById('resultsContainer');

    // Drag and Drop Handling
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });

    function handleFileSelect(file) {
      currentFile = file;
      dropzoneText.textContent = file.name;
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewBox.style.display = 'block';
      };
      reader.readAsDataURL(file);
      hideAlerts();
    }

    function hideAlerts() {
      errorAlert.style.display = 'none';
      infoAlert.style.display = 'none';
    }

    function showError(msg) {
      errorAlert.textContent = msg;
      errorAlert.style.display = 'block';
      infoAlert.style.display = 'none';
    }

    function showInfo(msg) {
      infoAlert.textContent = msg;
      infoAlert.style.display = 'block';
      errorAlert.style.display = 'none';
    }

    // Load sample image
    async function loadSample(filename, label) {
      hideAlerts();
      dropzoneText.textContent = `Loading sample: ${label}...`;
      try {
        const resp = await fetch(`/api/sample/${filename}`);
        if (!resp.ok) throw new Error("Could not load sample file.");
        const blob = await resp.blob();
        currentFile = new File([blob], filename, { type: "image/jpeg" });
        previewImg.src = URL.createObjectURL(blob);
        previewBox.style.display = 'block';
        dropzoneText.textContent = `${label} (${filename})`;
      } catch (err) {
        showError("Failed to fetch sample image: " + err.message);
      }
    }

    async function runAnalysis() {
      hideAlerts();
      if (!currentFile) {
        showError("Please select or drop a wall image file (JPG/PNG) to analyze.");
        return;
      }

      const algo = document.getElementById('algoSelect').value;
      const calib = document.getElementById('calibInput').value.trim();

      const formData = new FormData();
      formData.append('image', currentFile);
      formData.append('algorithm', algo);
      if (calib) {
        formData.append('cm_per_pixel', calib);
      }

      analyzeBtn.disabled = true;
      btnSpinner.style.display = 'inline-block';
      btnText.textContent = 'Processing Mathematical Algorithms...';

      try {
        const resp = await fetch('/analyze', {
          method: 'POST',
          body: formData
        });

        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.error || "Analysis failed on server.");
        }

        renderResults(data);
      } catch (err) {
        showError("Error: " + err.message);
      } finally {
        analyzeBtn.disabled = false;
        btnSpinner.style.display = 'none';
        btnText.textContent = 'Analyze Crack Structure';
      }
    }

    function renderResults(data) {
      const m = data.measurements;
      const imgs = data.images;

      if (!m.crack_detected) {
        showInfo("Analysis complete: No significant crack fracture detected on this wall surface.");
      }

      // Severity Badge
      const sevEl = document.getElementById('metricSeverity');
      const sevLevel = (m.width && m.width.severity_level) ? m.width.severity_level.split(" ")[0].toUpperCase() : (m.crack_detected ? "DETECTED" : "NONE");
      sevEl.textContent = sevLevel;
      sevEl.className = 'severity-badge severity-' + sevLevel;

      // Orientation
      const orientType = (m.width && m.width.orientation_type) ? m.width.orientation_type : "N/A";
      const orientDeg = (m.width && m.width.orientation_deg !== undefined) ? m.width.orientation_deg : 0;
      document.getElementById('metricOrientation').textContent = `Orientation: ${orientType} (${orientDeg}°)`;

      // Length
      document.getElementById('metricLengthPx').textContent = `${Math.round(m.length.length_px * 10) / 10} px`;
      document.getElementById('metricLengthCm').textContent = m.calibrated ? `${m.length.length_cm} cm` : "Calibration not provided";

      // Widths
      document.getElementById('metricAvgWidthPx').textContent = `${m.width.avg_width_px} px`;
      document.getElementById('metricAvgWidthCm').textContent = m.calibrated ? `${m.width.avg_width_cm} cm` : "Calibration not provided";
      document.getElementById('metricMaxWidthPx').textContent = `${m.width.max_width_px} px`;
      document.getElementById('metricMaxWidthCm').textContent = m.calibrated ? `${m.width.max_width_cm} cm` : "Calibration not provided";

      // Damage / Area
      document.getElementById('metricDamagePct').textContent = `${m.area.damage_percentage}%`;
      document.getElementById('metricCrackAreaPx').textContent = `${m.area.crack_area_px.toLocaleString()} px² (${m.area.wall_area_px.toLocaleString()} total)`;

      // Environment & Runtime
      document.getElementById('metricLighting').textContent = m.lighting_label;
      document.getElementById('metricSegmentation').textContent = `Method: ${m.segmentation_method}`;
      document.getElementById('metricConfidence').textContent = `${m.confidence}%`;
      document.getElementById('metricRuntime').textContent = `${data.runtime}s`;
      document.getElementById('metricOperator').textContent = `Operator: ${data.algorithm.toUpperCase()}`;

      // Image Panels
      document.getElementById('panelOriginal').src = imgs.original;
      document.getElementById('panelEnhanced').src = imgs.enhanced;
      document.getElementById('panelEdgeMap').src = imgs.edge_map;
      document.getElementById('edgeMapTag').textContent = data.algorithm.toUpperCase();
      document.getElementById('edgeMapTitle').textContent = `3. ${data.algorithm.toUpperCase()} Edge Map`;
      document.getElementById('panelCleanedMask').src = imgs.cleaned_mask;
      document.getElementById('panelSkeleton').src = imgs.skeleton;
      document.getElementById('panelHeatmap').src = imgs.width_heatmap;
      document.getElementById('panelFusion').src = imgs.fusion_map;
      document.getElementById('panelAnnotated').src = imgs.annotated_overlay;

      resultsContainer.style.display = 'block';
      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function openModal(title, imgElementId) {
      const src = document.getElementById(imgElementId).src;
      if (!src) return;
      document.getElementById('modalTitle').textContent = title;
      document.getElementById('modalImg').src = src;
      document.getElementById('downloadLink').href = src;
      document.getElementById('imageModal').classList.add('active');
    }

    function closeModal() {
      document.getElementById('imageModal').classList.remove('active');
    }
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "app": "Wall Crack Analyzer",
        "engine": "Classical Image Processing (From-Scratch)",
        "zero_opencv_algorithms": True,
    })


@app.route("/api/sample/<filename>", methods=["GET"])
def get_sample(filename):
    safe_name = os.path.basename(filename)
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "raw", safe_name)
    if not os.path.exists(sample_path):
        return jsonify({"error": f"Sample image {safe_name} not found."}), 404
    from flask import send_file
    return send_file(sample_path, mimetype="image/jpeg")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Please upload JPG, PNG, or WEBP."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{int(time.time())}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    chosen_algo = request.form.get("algorithm", "canny").lower().strip()
    cm_per_px_str = request.form.get("cm_per_pixel", "").strip()
    cm_per_pixel = None
    if cm_per_px_str:
        try:
            val = float(cm_per_px_str)
            if val > 0:
                cm_per_pixel = val
        except ValueError:
            pass

    try:
        images_dict, measurements = source.analyze_image_classical(
            image_path=save_path,
            chosen_algo=chosen_algo,
            cm_per_pixel=cm_per_pixel
        )

        # Convert images to base64 data URLs for fast browser rendering
        encoded_images = {}
        for key, arr in images_dict.items():
            encoded_images[key] = ndarray_to_base64_png(arr)

        return jsonify({
            "success": True,
            "algorithm": chosen_algo,
            "runtime": measurements["runtime_sec"],
            "measurements": measurements,
            "images": encoded_images,
        })
    except Exception as exc:
        return jsonify({"error": f"Error during pipeline execution: {str(exc)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting Wall Crack Analyzer Web Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

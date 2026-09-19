"""
app.py
Adaptive Hybrid Framework for Low-Light Image Enhancement

A simple local web app: upload an image in your browser, see the
original and Zero-DCE-enhanced result side by side.

Setup (one time):
    pip install flask torch torchvision pillow numpy

Run:
    python app.py

Then open in your browser:
    http://127.0.0.1:5000
"""

import os
import io
import time
import base64

import numpy as np
import torch
import torchvision
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, render_template_string

import model

# ---------------------------------------------------------------------------
# Configuration - adjust these if your paths differ
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = r"D:\College\Projects\ICML-Image\snapshots_SICE\best_model.pth"
UPLOAD_FOLDER = "web_uploads"
RESULT_FOLDER = "web_results"
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Info] Using device: {device}")

dce_net = model.enhance_net_nopool().to(device)
state = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(state, dict) and 'model_state_dict' in state:
    dce_net.load_state_dict(state['model_state_dict'])
    print(f"[Info] Loaded checkpoint (epoch={state.get('epoch')}, "
          f"best_loss={state.get('best_loss')}) from {CHECKPOINT_PATH}")
else:
    dce_net.load_state_dict(state)
    print(f"[Info] Loaded raw state_dict from {CHECKPOINT_PATH}")
dce_net.eval()


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Low-Light Image Enhancement</title>
    <style>
        body {
            font-family: -apple-system, Segoe UI, Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            background: #111;
            color: #eee;
        }
        h1 { font-size: 22px; }
        .upload-box {
            border: 2px dashed #555;
            border-radius: 8px;
            padding: 24px;
            text-align: center;
            margin-bottom: 24px;
        }
        input[type=submit] {
            background: #3d7eff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 15px;
            cursor: pointer;
            margin-top: 12px;
        }
        input[type=submit]:hover { background: #2f66d9; }
        .results {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .result-col {
            flex: 1;
            min-width: 300px;
        }
        .result-col h3 { margin-bottom: 8px; font-size: 15px; color: #aaa; }
        .result-col img {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #333;
        }
        .error { color: #ff6b6b; margin-top: 12px; }
        .meta { color: #888; font-size: 13px; margin-top: 8px; }
    </style>
</head>
<body>
    <h1>Adaptive Hybrid Framework - Low-Light Image Enhancement</h1>
    <div class="upload-box">
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept=".jpg,.jpeg,.png" required>
            <br>
            <input type="submit" value="Enhance Image">
        </form>
    </div>

    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}

    {% if original_b64 %}
    <div class="results">
        <div class="result-col">
            <h3>Original</h3>
            <img src="data:image/png;base64,{{ original_b64 }}">
        </div>
        <div class="result-col">
            <h3>Enhanced</h3>
            <img src="data:image/png;base64,{{ enhanced_b64 }}">
        </div>
    </div>
    <p class="meta">Inference time: {{ elapsed_ms }} ms &nbsp;|&nbsp; Device: {{ device }}</p>
    {% endif %}
</body>
</html>
"""


def image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def enhance(pil_image):
    img_np = np.asarray(pil_image, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.time()
        _, enhanced_tensor, _ = dce_net(img_tensor)
        elapsed = time.time() - start

    enhanced_tensor = enhanced_tensor.squeeze(0).clamp(0, 1).cpu()
    enhanced_np = (enhanced_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    enhanced_pil = Image.fromarray(enhanced_np)

    return enhanced_pil, elapsed


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template_string(PAGE_TEMPLATE, error=None, original_b64=None)

    uploaded_file = request.files.get('image')
    if not uploaded_file or uploaded_file.filename == '':
        return render_template_string(PAGE_TEMPLATE, error="No file selected.", original_b64=None)

    ext = os.path.splitext(uploaded_file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return render_template_string(
            PAGE_TEMPLATE,
            error=f"Unsupported file type '{ext}'. Please upload a .jpg, .jpeg, or .png image.",
            original_b64=None,
        )

    try:
        pil_image = Image.open(uploaded_file.stream).convert('RGB')
    except (UnidentifiedImageError, OSError) as err:
        return render_template_string(
            PAGE_TEMPLATE,
            error=f"Could not read this image ({err}). It may be corrupted — try another file.",
            original_b64=None,
        )

    try:
        enhanced_pil, elapsed = enhance(pil_image)
    except Exception as err:
        return render_template_string(
            PAGE_TEMPLATE,
            error=f"Enhancement failed: {err}",
            original_b64=None,
        )

    return render_template_string(
        PAGE_TEMPLATE,
        error=None,
        original_b64=image_to_base64(pil_image),
        enhanced_b64=image_to_base64(enhanced_pil),
        elapsed_ms=f"{elapsed * 1000:.1f}",
        device=str(device),
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

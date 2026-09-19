"""
Low-Light Image Enhancement — Comparison Web App (v6: adds weighted fusion)
==========================================================================
  POST /api/start        -> upload image, get a session_id back
  POST /api/run_model    -> run ONE model, return its result
  POST /api/run_fusion   -> quality-weighted fusion of ALL 5 model outputs
  POST /api/run_hybrid   -> post-process a chosen base image (fusion or single model)
"""

import os
import time
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory

from models.zero_dce import ZeroDCE
from models.zero_dce_pp import ZeroDCEPP
from models.retinexnet import RetinexNet
from models.mirnet import MIRNet
from models.sci import SCI
from metrics import score_no_reference, compute_psnr_ssim
from hybrid import apply_hybrid_postprocessing
from fusion import fuse_all_models

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

print("[Startup] Loading models...")
MODELS = {
    "zero_dce": ZeroDCE(),
    "zero_dce_pp": ZeroDCEPP(),
    "retinexnet": RetinexNet(),
    "mirnet": MIRNet(),
    "sci": SCI(),
}
print("[Startup] All models loaded.")

DISPLAY_NAMES = {
    "zero_dce": "Zero-DCE",
    "zero_dce_pp": "Zero-DCE++",
    "retinexnet": "RetinexNet",
    "mirnet": "MIRNet",
    "sci": "SCI",
}
MODEL_ORDER = list(MODELS.keys())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/start", methods=["POST"])
def start():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    ground_truth_file = request.files.get("ground_truth")

    session_id = uuid.uuid4().hex[:10]
    in_filename = f"{session_id}_input.jpg"
    in_path = os.path.join(UPLOAD_DIR, in_filename)
    file.save(in_path)

    image = cv2.imread(in_path)
    if image is None:
        return jsonify({"error": "Could not read image"}), 400

    has_ground_truth = False
    if ground_truth_file and ground_truth_file.filename:
        gt_path = os.path.join(UPLOAD_DIR, f"{session_id}_gt.jpg")
        ground_truth_file.save(gt_path)
        has_ground_truth = True

    return jsonify({
        "session_id": session_id,
        "input_url": f"/uploads/{in_filename}",
        "has_ground_truth": has_ground_truth,
        "model_order": MODEL_ORDER,
        "model_names": DISPLAY_NAMES,
    })


def _load_session_image(session_id):
    return cv2.imread(os.path.join(UPLOAD_DIR, f"{session_id}_input.jpg"))


def _load_ground_truth(session_id):
    gt_path = os.path.join(UPLOAD_DIR, f"{session_id}_gt.jpg")
    if os.path.exists(gt_path):
        return cv2.imread(gt_path)
    return None


@app.route("/api/run_model", methods=["POST"])
def run_model():
    data = request.get_json()
    session_id = data.get("session_id")
    key = data.get("model_key")

    if key not in MODELS:
        return jsonify({"error": f"Unknown model: {key}"}), 400

    image = _load_session_image(session_id)
    if image is None:
        return jsonify({"error": "Session image not found"}), 400

    model = MODELS[key]
    start_time = time.time()
    try:
        enhanced = model.enhance(image)
    except Exception as e:
        return jsonify({"error": str(e), "key": key, "name": DISPLAY_NAMES[key]}), 500
    elapsed = time.time() - start_time

    out_filename = f"{session_id}_{key}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, enhanced)

    entry = {
        "key": key,
        "name": DISPLAY_NAMES[key],
        "image_url": f"/outputs/{out_filename}",
        "time_seconds": round(elapsed, 3),
        "quality": score_no_reference(enhanced),
    }

    ground_truth = _load_ground_truth(session_id)
    if ground_truth is not None:
        psnr, ssim = compute_psnr_ssim(enhanced, ground_truth)
        entry["psnr"] = round(psnr, 3)
        entry["ssim"] = round(ssim, 4)

    return jsonify(entry)


@app.route("/api/run_fusion", methods=["POST"])
def run_fusion():
    """
    Quality-weighted fusion of all 5 already-computed model outputs.
    Weights are computed from each model's composite no-reference score
    via a temperature-scaled softmax (see fusion.py) — deterministic,
    not random, and reproducible for the same inputs.
    """
    data = request.get_json()
    session_id = data.get("session_id")

    images = []
    scores = []
    names = []

    for key in MODEL_ORDER:
        path = os.path.join(OUTPUT_DIR, f"{session_id}_{key}.jpg")
        img = cv2.imread(path)
        if img is None:
            continue
        images.append(img)
        scores.append(score_no_reference(img)["composite"])
        names.append(DISPLAY_NAMES[key])

    if len(images) < 2:
        return jsonify({"error": "Not enough model outputs found to fuse"}), 400

    start_time = time.time()
    fused_image, weights = fuse_all_models(images, scores, temperature=10.0)
    elapsed = time.time() - start_time

    out_filename = f"{session_id}_fusion.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, fused_image)

    weight_breakdown = [
        {"name": name, "weight_percent": round(float(w) * 100, 1)}
        for name, w in zip(names, weights)
    ]
    weight_breakdown.sort(key=lambda x: x["weight_percent"], reverse=True)

    entry = {
        "key": "fusion",
        "name": "Quality-weighted fusion (all 5 models)",
        "image_url": f"/outputs/{out_filename}",
        "time_seconds": round(elapsed, 3),
        "quality": score_no_reference(fused_image),
        "weights": weight_breakdown,
    }

    ground_truth = _load_ground_truth(session_id)
    if ground_truth is not None:
        psnr, ssim = compute_psnr_ssim(fused_image, ground_truth)
        entry["psnr"] = round(psnr, 3)
        entry["ssim"] = round(ssim, 4)

    return jsonify(entry)


@app.route("/api/run_hybrid", methods=["POST"])
def run_hybrid():
    data = request.get_json()
    session_id = data.get("session_id")
    base_key = data.get("base_key")  # e.g. "fusion" or a single model key

    base_path = os.path.join(OUTPUT_DIR, f"{session_id}_{base_key}.jpg")
    base_image = cv2.imread(base_path)
    if base_image is None:
        return jsonify({"error": "Base image not found"}), 400

    start_time = time.time()
    try:
        hybrid_image = apply_hybrid_postprocessing(base_image)
    except Exception as e:
        return jsonify({"error": str(e), "key": "hybrid", "name": "Hybrid"}), 500
    elapsed = time.time() - start_time

    out_filename = f"{session_id}_hybrid.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, hybrid_image)

    base_label = "fusion" if base_key == "fusion" else DISPLAY_NAMES.get(base_key, base_key)
    entry = {
        "key": "hybrid",
        "name": f"Hybrid (post-processed {base_label})",
        "image_url": f"/outputs/{out_filename}",
        "time_seconds": round(elapsed, 3),
        "quality": score_no_reference(hybrid_image),
    }

    ground_truth = _load_ground_truth(session_id)
    if ground_truth is not None:
        psnr, ssim = compute_psnr_ssim(hybrid_image, ground_truth)
        entry["psnr"] = round(psnr, 3)
        entry["ssim"] = round(ssim, 4)

    return jsonify(entry)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)

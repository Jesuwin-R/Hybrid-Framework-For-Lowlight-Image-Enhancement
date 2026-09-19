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
import gc

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
)

import cv2
import numpy as np

from models.zero_dce import ZeroDCE
from models.zero_dce_pp import ZeroDCEPP
from models.retinexnet import RetinexNet
from models.mirnet import MIRNet
from models.sci import SCI

from metrics import score_no_reference, compute_psnr_ssim
from hybrid import apply_hybrid_postprocessing
from fusion import fuse_all_models


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ============================================================
# MODEL INFORMATION
# ============================================================

DISPLAY_NAMES = {
    "zero_dce": "Zero-DCE",
    "zero_dce_pp": "Zero-DCE++",
    "retinexnet": "RetinexNet",
    "mirnet": "MIRNet",
    "sci": "SCI",
}


# IMPORTANT:
# We store the MODEL CLASSES instead of creating all models
# during application startup.
#
# This saves RAM on low-memory hosting such as Render Free.
MODEL_CLASSES = {
    "zero_dce": ZeroDCE,
    "zero_dce_pp": ZeroDCEPP,
    "retinexnet": RetinexNet,
    "mirnet": MIRNet,
    "sci": SCI,
}


MODEL_ORDER = list(MODEL_CLASSES.keys())


# ============================================================
# MAIN PAGE
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# ============================================================
# SERVE OUTPUT IMAGES
# ============================================================

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ============================================================
# START SESSION / UPLOAD IMAGE
# ============================================================

@app.route("/api/start", methods=["POST"])
def start():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    ground_truth_file = request.files.get("ground_truth")

    session_id = uuid.uuid4().hex[:10]

    in_filename = f"{session_id}_input.jpg"

    in_path = os.path.join(
        UPLOAD_DIR,
        in_filename
    )

    file.save(in_path)

    image = cv2.imread(in_path)

    if image is None:
        return jsonify({
            "error": "Could not read image"
        }), 400

    has_ground_truth = False

    if ground_truth_file and ground_truth_file.filename:

        gt_path = os.path.join(
            UPLOAD_DIR,
            f"{session_id}_gt.jpg"
        )

        ground_truth_file.save(gt_path)

        has_ground_truth = True

    return jsonify({
        "session_id": session_id,
        "input_url": f"/uploads/{in_filename}",
        "has_ground_truth": has_ground_truth,
        "model_order": MODEL_ORDER,
        "model_names": DISPLAY_NAMES,
    })


# ============================================================
# LOAD SESSION IMAGE
# ============================================================

def _load_session_image(session_id):

    image_path = os.path.join(
        UPLOAD_DIR,
        f"{session_id}_input.jpg"
    )

    return cv2.imread(image_path)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def _load_ground_truth(session_id):

    gt_path = os.path.join(
        UPLOAD_DIR,
        f"{session_id}_gt.jpg"
    )

    if os.path.exists(gt_path):
        return cv2.imread(gt_path)

    return None


# ============================================================
# RUN ONE MODEL
# ============================================================

@app.route("/api/run_model", methods=["POST"])
def run_model():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No JSON data received"
        }), 400

    session_id = data.get("session_id")

    key = data.get("model_key")

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if key not in MODEL_CLASSES:

        return jsonify({
            "error": f"Unknown model: {key}"
        }), 400

    # --------------------------------------------------------
    # Load uploaded image
    # --------------------------------------------------------

    image = _load_session_image(session_id)

    if image is None:

        return jsonify({
            "error": "Session image not found"
        }), 400

    # --------------------------------------------------------
    # Load ONLY the requested model
    # --------------------------------------------------------

    model = None

    start_time = time.time()

    try:

        print(
            f"[Model] Loading {DISPLAY_NAMES[key]}..."
        )

        model_class = MODEL_CLASSES[key]

        model = model_class()

        print(
            f"[Model] Running {DISPLAY_NAMES[key]}..."
        )

        enhanced = model.enhance(image)

        elapsed = time.time() - start_time

        print(
            f"[Model] {DISPLAY_NAMES[key]} completed "
            f"in {elapsed:.2f} seconds"
        )

    except Exception as e:

        print(
            f"[Model] Error in {DISPLAY_NAMES[key]}: {e}"
        )

        return jsonify({
            "error": str(e),
            "key": key,
            "name": DISPLAY_NAMES[key]
        }), 500

    finally:

        # ----------------------------------------------------
        # Release model memory
        # ----------------------------------------------------

        if model is not None:

            del model

        gc.collect()

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    out_filename = f"{session_id}_{key}.jpg"

    out_path = os.path.join(
        OUTPUT_DIR,
        out_filename
    )

    cv2.imwrite(
        out_path,
        enhanced
    )

    # ========================================================
    # QUALITY INFORMATION
    # ========================================================

    entry = {
        "key": key,
        "name": DISPLAY_NAMES[key],
        "image_url": f"/outputs/{out_filename}",
        "time_seconds": round(elapsed, 3),
        "quality": score_no_reference(enhanced),
    }

    # ========================================================
    # OPTIONAL GROUND TRUTH METRICS
    # ========================================================

    ground_truth = _load_ground_truth(session_id)

    if ground_truth is not None:

        psnr, ssim = compute_psnr_ssim(
            enhanced,
            ground_truth
        )

        entry["psnr"] = round(
            psnr,
            3
        )

        entry["ssim"] = round(
            ssim,
            4
        )

    return jsonify(entry)


# ============================================================
# RUN WEIGHTED FUSION
# ============================================================

@app.route("/api/run_fusion", methods=["POST"])
def run_fusion():

    """
    Quality-weighted fusion of all already-computed model outputs.

    Weights are computed from each model's composite
    no-reference score via a temperature-scaled softmax.

    Deterministic and reproducible for the same inputs.
    """

    data = request.get_json()

    if not data:

        return jsonify({
            "error": "No JSON data received"
        }), 400

    session_id = data.get("session_id")

    images = []
    scores = []
    names = []

    # --------------------------------------------------------
    # Load outputs generated by individual models
    # --------------------------------------------------------

    for key in MODEL_ORDER:

        path = os.path.join(
            OUTPUT_DIR,
            f"{session_id}_{key}.jpg"
        )

        img = cv2.imread(path)

        if img is None:
            continue

        images.append(img)

        scores.append(
            score_no_reference(img)["composite"]
        )

        names.append(
            DISPLAY_NAMES[key]
        )

    # --------------------------------------------------------
    # Need at least two model outputs
    # --------------------------------------------------------

    if len(images) < 2:

        return jsonify({
            "error": "Not enough model outputs found to fuse"
        }), 400

    # --------------------------------------------------------
    # Perform fusion
    # --------------------------------------------------------

    start_time = time.time()

    fused_image, weights = fuse_all_models(
        images,
        scores,
        temperature=10.0
    )

    elapsed = time.time() - start_time

    # ========================================================
    # SAVE FUSION RESULT
    # ========================================================

    out_filename = f"{session_id}_fusion.jpg"

    out_path = os.path.join(
        OUTPUT_DIR,
        out_filename
    )

    cv2.imwrite(
        out_path,
        fused_image
    )

    # ========================================================
    # WEIGHT BREAKDOWN
    # ========================================================

    weight_breakdown = [

        {
            "name": name,
            "weight_percent": round(
                float(weight) * 100,
                1
            )
        }

        for name, weight in zip(
            names,
            weights
        )
    ]

    weight_breakdown.sort(
        key=lambda x: x["weight_percent"],
        reverse=True
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    entry = {

        "key": "fusion",

        "name": (
            "Quality-weighted fusion "
            "(all 5 models)"
        ),

        "image_url": (
            f"/outputs/{out_filename}"
        ),

        "time_seconds": round(
            elapsed,
            3
        ),

        "quality": score_no_reference(
            fused_image
        ),

        "weights": weight_breakdown,
    }

    # ========================================================
    # OPTIONAL GROUND TRUTH
    # ========================================================

    ground_truth = _load_ground_truth(
        session_id
    )

    if ground_truth is not None:

        psnr, ssim = compute_psnr_ssim(
            fused_image,
            ground_truth
        )

        entry["psnr"] = round(
            psnr,
            3
        )

        entry["ssim"] = round(
            ssim,
            4
        )

    return jsonify(entry)


# ============================================================
# RUN HYBRID POST-PROCESSING
# ============================================================

@app.route("/api/run_hybrid", methods=["POST"])
def run_hybrid():

    data = request.get_json()

    if not data:

        return jsonify({
            "error": "No JSON data received"
        }), 400

    session_id = data.get("session_id")

    base_key = data.get(
        "base_key"
    )

    # --------------------------------------------------------
    # Find base image
    # --------------------------------------------------------

    base_path = os.path.join(
        OUTPUT_DIR,
        f"{session_id}_{base_key}.jpg"
    )

    base_image = cv2.imread(
        base_path
    )

    if base_image is None:

        return jsonify({
            "error": "Base image not found"
        }), 400

    # --------------------------------------------------------
    # Run hybrid processing
    # --------------------------------------------------------

    start_time = time.time()

    try:

        hybrid_image = apply_hybrid_postprocessing(
            base_image
        )

    except Exception as e:

        return jsonify({
            "error": str(e),
            "key": "hybrid",
            "name": "Hybrid"
        }), 500

    elapsed = time.time() - start_time

    # ========================================================
    # SAVE HYBRID RESULT
    # ========================================================

    out_filename = (
        f"{session_id}_hybrid.jpg"
    )

    out_path = os.path.join(
        OUTPUT_DIR,
        out_filename
    )

    cv2.imwrite(
        out_path,
        hybrid_image
    )

    # ========================================================
    # BASE LABEL
    # ========================================================

    if base_key == "fusion":

        base_label = "fusion"

    else:

        base_label = DISPLAY_NAMES.get(
            base_key,
            base_key
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    entry = {

        "key": "hybrid",

        "name": (
            f"Hybrid "
            f"(post-processed {base_label})"
        ),

        "image_url": (
            f"/outputs/{out_filename}"
        ),

        "time_seconds": round(
            elapsed,
            3
        ),

        "quality": score_no_reference(
            hybrid_image
        ),
    }

    # ========================================================
    # OPTIONAL GROUND TRUTH
    # ========================================================

    ground_truth = _load_ground_truth(
        session_id
    )

    if ground_truth is not None:

        psnr, ssim = compute_psnr_ssim(
            hybrid_image,
            ground_truth
        )

        entry["psnr"] = round(
            psnr,
            3
        )

        entry["ssim"] = round(
            ssim,
            4
        )

    return jsonify(entry)


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
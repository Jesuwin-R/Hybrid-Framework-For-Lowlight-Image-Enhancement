"""
Quality metrics for comparing enhanced images.

PSNR/SSIM are full-reference metrics requiring ground truth. For live app
use with arbitrary uploads (no ground truth), score_no_reference() is used
instead for ranking/fusion weighting.
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def _contrast(gray: np.ndarray) -> float:
    return float(gray.std())


def _colorfulness(image_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(image_bgr.astype("float"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))


def _mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def _noise_estimate(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def score_no_reference(image_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    brightness = _mean_brightness(gray)
    contrast = _contrast(gray)
    entropy = _entropy(gray)
    colorfulness = _colorfulness(image_bgr)
    sharpness = _noise_estimate(gray)

    target_brightness = 130.0
    brightness_score = 1.0 - min(abs(brightness - target_brightness) / target_brightness, 1.0)
    contrast_score = min(contrast / 80.0, 1.0)
    entropy_score = min(entropy / 8.0, 1.0)
    colorfulness_score = min(colorfulness / 60.0, 1.0)
    sharpness_score = min(sharpness / 500.0, 1.0)

    composite = (
        0.30 * brightness_score
        + 0.25 * contrast_score
        + 0.20 * entropy_score
        + 0.15 * sharpness_score
        + 0.10 * colorfulness_score
    )

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "entropy": round(entropy, 3),
        "colorfulness": round(colorfulness, 2),
        "sharpness": round(sharpness, 2),
        "composite": round(composite, 4),
    }


def compute_psnr_ssim(enhanced_bgr: np.ndarray, ground_truth_bgr: np.ndarray):
    if enhanced_bgr.shape != ground_truth_bgr.shape:
        ground_truth_bgr = cv2.resize(
            ground_truth_bgr, (enhanced_bgr.shape[1], enhanced_bgr.shape[0])
        )
    psnr = peak_signal_noise_ratio(ground_truth_bgr, enhanced_bgr, data_range=255)
    ssim = structural_similarity(ground_truth_bgr, enhanced_bgr, channel_axis=2, data_range=255)
    return psnr, ssim

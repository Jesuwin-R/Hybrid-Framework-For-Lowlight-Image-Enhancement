"""
Hybrid post-processing: white balance + adaptive contrast + denoising,
applied on top of a base enhanced image (either the single best model's
output, or the quality-weighted fusion of all 5).
"""

import cv2
import numpy as np


def gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    result = img_bgr.astype(np.float32)
    b, g, r = cv2.split(result)
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    b = b * (mean_gray / (mean_b + 1e-6))
    g = g * (mean_gray / (mean_g + 1e-6))
    r = r * (mean_gray / (mean_r + 1e-6))
    balanced = cv2.merge([b, g, r])
    return np.clip(balanced, 0, 255).astype(np.uint8)


def adaptive_contrast(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        img_bgr, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
    )


def apply_hybrid_postprocessing(base_image_bgr: np.ndarray) -> np.ndarray:
    step1 = gray_world_white_balance(base_image_bgr)
    step2 = adaptive_contrast(step1)
    step3 = denoise(step2)
    return step3

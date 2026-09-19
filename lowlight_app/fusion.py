"""
Quality-Weighted Fusion

Instead of picking one "best" model and discarding the rest, this module
fuses ALL 5 model outputs into a single image, weighted by each model's
no-reference quality score. Weights are computed deterministically via a
temperature-scaled softmax over the quality scores — NOT random, and NOT
manually tuned per-image. The same formula runs on every image, so the
results are reproducible and defensible in a paper.

    weight_i = exp(score_i * temperature) / sum(exp(score_j * temperature))

Higher temperature -> more winner-takes-all (closer to pure selection).
Lower temperature -> more even blending across all 5 models.
"""

import cv2
import numpy as np


def compute_fusion_weights(scores: list, temperature: float = 10.0) -> np.ndarray:
    """
    scores: list of composite no-reference quality scores (one per model),
            in the same order as the images being fused.
    Returns: array of weights, same length, summing to 1.0.
    """
    scores = np.array(scores, dtype=np.float64)
    scaled = scores * temperature
    scaled -= scaled.max()  # numerical stability, doesn't change the result
    exp_scores = np.exp(scaled)
    weights = exp_scores / exp_scores.sum()
    return weights


def weighted_fuse(images: list, weights: np.ndarray) -> np.ndarray:
    """
    images: list of BGR numpy arrays (may differ slightly in size across
            models due to each model's own resizing constraints).
    weights: array from compute_fusion_weights(), same length as images.
    Returns: single fused BGR image at the size of the first input image.
    """
    target_h, target_w = images[0].shape[:2]

    fused = np.zeros((target_h, target_w, 3), dtype=np.float64)
    for img, w in zip(images, weights):
        resized = cv2.resize(img, (target_w, target_h)).astype(np.float64)
        fused += resized * w

    return np.clip(fused, 0, 255).astype(np.uint8)


def fuse_all_models(images: list, quality_scores: list, temperature: float = 10.0):
    """
    Convenience wrapper: computes weights and fuses in one call.
    Returns (fused_image, weights) so the caller can display the
    percentage contribution of each model.
    """
    weights = compute_fusion_weights(quality_scores, temperature=temperature)
    fused = weighted_fuse(images, weights)
    return fused, weights

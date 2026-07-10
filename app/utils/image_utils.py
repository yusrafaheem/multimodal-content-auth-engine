"""Small image helpers shared by the detectors. Only depends on Pillow/NumPy/OpenCV,
which are cheap, common dependencies -- kept separate from anything touching torch.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image


def load_image(image_bytes: bytes) -> Image.Image:
    """Load raw bytes into a PIL image, normalized to RGB."""
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.float32)


def error_level_analysis(img: Image.Image, quality: int = 90) -> Tuple[np.ndarray, float]:
    """Classic ELA: re-compress the image as JPEG at a fixed quality and diff it
    against the original. Regions that were pasted/edited after the last real
    save tend to show a different error level than untouched regions, since
    they haven't been through the same number of JPEG compression cycles.

    Returns the per-pixel error map and a single scalar "anomaly score" in
    [0, 1] where higher means more evidence of localized tampering.
    """
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)
    resaved.load()

    orig = to_numpy(img)
    resave = to_numpy(resaved)
    diff = np.abs(orig - resave)

    # Normalize per-channel error map to 0-255 for visualization/inspection.
    error_map = diff.mean(axis=2)

    # Heuristic anomaly score: tampered regions produce a bimodal error
    # distribution (a "hot" edited patch next to a "cold" untouched background).
    # We approximate that with the ratio of the 95th percentile error to the
    # median error -- a large gap suggests localized inconsistency rather than
    # uniform, whole-image compression noise.
    median = float(np.median(error_map)) + 1e-6
    p95 = float(np.percentile(error_map, 95))
    ratio = p95 / median
    # Squash into [0, 1] with a soft cap; empirically tuned constants, documented
    # here rather than buried as magic numbers.
    anomaly_score = 1.0 - np.exp(-max(ratio - 4.0, 0.0) / 6.0)
    return error_map, float(np.clip(anomaly_score, 0.0, 1.0))


def noise_residual_score(img: Image.Image) -> float:
    """Adversarial-perturbation heuristic: high-frequency noise residual analysis.

    Adversarial perturbations (FGSM/PGD-style) and GAN upsampling artifacts both
    leave a distinctive high-frequency signature that's largely invisible to the
    eye but shows up once you subtract a denoised version of the image from the
    original. We use a simple Gaussian-blur residual as a cheap, dependency-light
    stand-in for a learned noise-print detector.
    """
    import cv2

    arr = to_numpy(img)
    blurred = cv2.GaussianBlur(arr, (5, 5), 0)
    residual = arr - blurred
    # Energy of the high-frequency residual, normalized by image energy.
    residual_energy = float(np.mean(residual ** 2))
    signal_energy = float(np.mean(arr ** 2)) + 1e-6
    ratio = residual_energy / signal_energy
    # Empirically (see benchmarking/attack_fixtures.py), clean synthetic photos
    # land around ratio ~0.0007 while perturbed ones run an order of magnitude
    # higher. This constant is a starting point, not a calibrated threshold --
    # re-tune it against `benchmarking/run_benchmark.py` on real photos before
    # relying on it for anything beyond a demo.
    score = np.clip(ratio / 0.01, 0.0, 1.0)
    return float(score)

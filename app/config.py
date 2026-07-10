"""Central configuration for the authentication engine.

Everything that touches a pretrained model or a threshold lives here so the
rest of the codebase never hardcodes magic numbers or model names.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # --- Vision transformer backbone -------------------------------------------------
    # A real, publicly available ViT checkpoint used as a feature backbone. The
    # classification head on top of it (authentic vs. manipulated) ships randomly
    # initialized -- see app/detectors/image_detector.py and
    # benchmarking/README.md for how to fine-tune it on a labeled deepfake dataset.
    vit_backbone: str = os.getenv("VIT_BACKBONE", "google/vit-base-patch16-224-in21k")

    # --- Cross-modal (image/text) similarity model --------------------------------
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

    # --- Behavior flags -------------------------------------------------------------
    # When True (default), detectors try to load torch/transformers models.
    # When False, or when those libraries aren't installed, detectors fall back to
    # deterministic, dependency-light heuristics (ELA, noise-residual stats, EXIF
    # rules, text-burstiness). This keeps the API usable on a laptop with no GPU
    # and no internet, and keeps the test suite runnable without heavy deps.
    use_pretrained_models: bool = os.getenv("USE_PRETRAINED_MODELS", "1") not in ("0", "false", "False")

    # --- Fusion weights for the cross-modal consistency pipeline --------------------
    weights: dict = field(default_factory=lambda: {
        "image": 0.5,
        "text": 0.25,
        "metadata": 0.25,
    })

    # Verdict thresholds (unified authenticity score is in [0, 1], 1 = authentic)
    suspicious_threshold: float = float(os.getenv("SUSPICIOUS_THRESHOLD", "0.6"))
    fake_threshold: float = float(os.getenv("FAKE_THRESHOLD", "0.35"))


settings = Settings()

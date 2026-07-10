"""Image authenticity detector.

Two independent signal sources are fused here:

1. A ViT backbone + classification head (`_ModelBackedScorer`), used when
   `settings.use_pretrained_models` is True and torch/transformers are
   installed. The backbone (google/vit-base-patch16-224-in21k) is a real,
   publicly hosted checkpoint; the classification head is a small linear
   layer that starts randomly initialized. It is architecturally correct but
   NOT trained out of the box -- see benchmarking/README.md for a fine-tuning
   recipe against a labeled deepfake dataset (e.g. FaceForensics++, DFDC).

2. Classical, dependency-light heuristics (`_HeuristicScorer`): JPEG error
   level analysis (manipulation/splicing) and high-frequency noise-residual
   analysis (adversarial perturbations / GAN upsampling artifacts). These run
   on CPU with only Pillow/NumPy/OpenCV and require no model download, so the
   engine still produces a real signal even with `use_pretrained_models=False`
   or when torch isn't available at all.

The public `ImageAuthDetector.analyze()` always returns a `DetectorResult`
with a `method` field telling you which path actually ran.
"""
from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from app.config import settings
from app.schemas import DetectorResult
from app.utils.image_utils import error_level_analysis, load_image, noise_residual_score

logger = logging.getLogger(__name__)


class _HeuristicScorer:
    """No-torch-required fallback. See app/utils/image_utils.py for the math."""

    def score(self, img: Image.Image) -> DetectorResult:
        _, ela_anomaly = error_level_analysis(img)
        noise_anomaly = noise_residual_score(img)

        # ELA catches splicing/pasting; noise-residual catches adversarial
        # perturbations -- they target different attack types, so we take the
        # max rather than averaging (a weighted average would let a strong hit
        # on one detector get diluted by a quiet reading on the other). Then
        # invert so the returned score follows the project-wide convention of
        # 1.0 = authentic, 0.0 = fake/manipulated.
        combined_anomaly = max(ela_anomaly, noise_anomaly)
        authenticity = float(max(0.0, 1.0 - combined_anomaly))

        label = "authentic" if authenticity >= settings.suspicious_threshold else (
            "suspicious" if authenticity >= settings.fake_threshold else "likely_manipulated"
        )
        return DetectorResult(
            score=authenticity,
            label=label,
            method="heuristic_ela_noise_residual",
            details={
                "ela_anomaly": round(ela_anomaly, 4),
                "noise_residual_anomaly": round(noise_anomaly, 4),
            },
        )


class _ModelBackedScorer:
    """ViT backbone + linear head. Lazily imports torch/transformers so this
    module can be imported (and the heuristic path exercised) even when those
    packages aren't installed.
    """

    def __init__(self, backbone_name: str):
        self.backbone_name = backbone_name
        self._model = None
        self._processor = None
        self._head = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from torch import nn
        from transformers import ViTImageProcessor, ViTModel

        self._processor = ViTImageProcessor.from_pretrained(self.backbone_name)
        self._model = ViTModel.from_pretrained(self.backbone_name)
        self._model.eval()
        # Binary head: authentic vs. manipulated. Randomly initialized -- fine-tune
        # this against labeled data before trusting it in production. Loading a
        # fine-tuned checkpoint: pass `head_state_dict_path` to `analyze()`.
        self._head = nn.Linear(self._model.config.hidden_size, 2)
        self._torch = torch

    def score(self, img: Image.Image, head_state_dict_path: Optional[str] = None) -> DetectorResult:
        self._ensure_loaded()
        torch = self._torch

        if head_state_dict_path:
            state = torch.load(head_state_dict_path, map_location="cpu")
            self._head.load_state_dict(state)

        inputs = self._processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
            logits = self._head(pooled)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            authenticity = float(probs[0].item())  # index 0 = "authentic" class

        label = "authentic" if authenticity >= settings.suspicious_threshold else (
            "suspicious" if authenticity >= settings.fake_threshold else "likely_manipulated"
        )
        return DetectorResult(
            score=authenticity,
            label=label,
            method=f"vit_backbone:{self.backbone_name}",
            details={"note": "classification head is untrained unless a fine-tuned checkpoint was supplied"},
        )


class ImageAuthDetector:
    def __init__(self, use_pretrained_models: Optional[bool] = None):
        self.use_pretrained_models = (
            settings.use_pretrained_models if use_pretrained_models is None else use_pretrained_models
        )
        self._heuristic = _HeuristicScorer()
        self._model_scorer: Optional[_ModelBackedScorer] = None

    def analyze(self, image_bytes: bytes, head_state_dict_path: Optional[str] = None) -> DetectorResult:
        img = load_image(image_bytes)

        if self.use_pretrained_models:
            try:
                if self._model_scorer is None:
                    self._model_scorer = _ModelBackedScorer(settings.vit_backbone)
                return self._model_scorer.score(img, head_state_dict_path=head_state_dict_path)
            except ImportError:
                logger.warning("torch/transformers not available, falling back to heuristic image scorer")
            except Exception:
                logger.exception("ViT backbone scoring failed, falling back to heuristic image scorer")

        return self._heuristic.score(img)

"""Text/claim authenticity + cross-modal (image<->text) consistency detector.

Two signal sources, same pattern as the image detector:

1. Model-backed: CLIP image-text similarity (`openai/clip-vit-base-patch32`)
   scores how well a caption/claim actually matches the image content -- the
   core of "cross-modal consistency" checking. Requires torch/transformers.

2. Heuristic fallback: lightweight statistical text analysis (repetition /
   burstiness / vocabulary-richness) that flags text with the low-perplexity,
   overly-uniform cadence characteristic of raw LLM output. This does NOT
   require an image and does NOT require torch, so it's useful as a standalone
   "does this caption look machine-generated" signal even offline.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from PIL import Image

from app.config import settings
from app.schemas import DetectorResult

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


class _HeuristicTextScorer:
    """No image, no torch: pure text statistics.

    - Vocabulary richness (type-token ratio): very low richness on longer
      passages can indicate templated/generated text.
    - Sentence-length burstiness: human writing tends to vary sentence length
      more than raw LLM completions sampled at low temperature.
    """

    def score(self, text: str) -> DetectorResult:
        tokens = _tokenize(text)
        if len(tokens) < 5:
            return DetectorResult(
                score=0.5,
                label="insufficient_text",
                method="heuristic_text_stats",
                details={"note": "text too short for a reliable statistical signal"},
            )

        ttr = len(set(tokens)) / len(tokens)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        lengths = [len(_tokenize(s)) for s in sentences] or [len(tokens)]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        burstiness = (variance ** 0.5) / (mean_len + 1e-6)

        # Low vocabulary richness + low burstiness => more "generated-sounding".
        richness_signal = min(ttr / 0.65, 1.0)  # human text on short passages is often >0.65
        burstiness_signal = min(burstiness / 0.5, 1.0)
        authenticity = float(0.5 * richness_signal + 0.5 * burstiness_signal)

        label = "likely_human" if authenticity >= settings.suspicious_threshold else (
            "uncertain" if authenticity >= settings.fake_threshold else "likely_generated"
        )
        return DetectorResult(
            score=authenticity,
            label=label,
            method="heuristic_text_stats",
            details={"type_token_ratio": round(ttr, 4), "sentence_burstiness": round(burstiness, 4)},
        )


class _CLIPConsistencyScorer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.eval()
        self._torch = torch

    def score(self, text: str, image: Image.Image) -> DetectorResult:
        self._ensure_loaded()
        torch = self._torch

        inputs = self._processor(text=[text], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
            similarity = torch.nn.functional.cosine_similarity(
                outputs.image_embeds, outputs.text_embeds
            ).item()

        # CLIP cosine similarities for genuinely matching image/caption pairs
        # typically land ~0.25-0.35; mismatched pairs are lower. Rescale to [0, 1].
        authenticity = float((similarity + 1) / 2)
        label = "consistent" if authenticity >= settings.suspicious_threshold else (
            "uncertain" if authenticity >= settings.fake_threshold else "inconsistent"
        )
        return DetectorResult(
            score=authenticity,
            label=label,
            method=f"clip_cross_modal:{self.model_name}",
            details={"cosine_similarity": round(similarity, 4)},
        )


class TextConsistencyDetector:
    def __init__(self, use_pretrained_models: Optional[bool] = None):
        self.use_pretrained_models = (
            settings.use_pretrained_models if use_pretrained_models is None else use_pretrained_models
        )
        self._heuristic = _HeuristicTextScorer()
        self._clip_scorer: Optional[_CLIPConsistencyScorer] = None

    def analyze(self, text: str, image_bytes: Optional[bytes] = None) -> DetectorResult:
        if image_bytes and self.use_pretrained_models:
            try:
                from app.utils.image_utils import load_image

                img = load_image(image_bytes)
                if self._clip_scorer is None:
                    self._clip_scorer = _CLIPConsistencyScorer(settings.clip_model)
                return self._clip_scorer.score(text, img)
            except ImportError:
                logger.warning("torch/transformers not available, falling back to text-only heuristics")
            except Exception:
                logger.exception("CLIP cross-modal scoring failed, falling back to text-only heuristics")

        return self._heuristic.score(text)

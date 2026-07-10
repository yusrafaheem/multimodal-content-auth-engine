"""Cross-modal consistency pipeline: fuses image, text, and metadata detector
outputs into a single explainable verdict.

Fusion is a simple weighted average (weights in app/config.py) rather than a
learned fusion model -- deliberately, so the result stays interpretable for a
learning project. A natural extension (see README "Roadmap") is to replace
this with a small learned fusion head once you have labeled multi-modal data.
"""
from __future__ import annotations

from typing import Optional

from app.config import settings
from app.detectors.image_detector import ImageAuthDetector
from app.detectors.metadata_detector import MetadataAuthDetector
from app.detectors.text_detector import TextConsistencyDetector
from app.schemas import UnifiedVerdict


class CrossModalConsistencyPipeline:
    def __init__(self, use_pretrained_models: Optional[bool] = None):
        self.image_detector = ImageAuthDetector(use_pretrained_models=use_pretrained_models)
        self.text_detector = TextConsistencyDetector(use_pretrained_models=use_pretrained_models)
        self.metadata_detector = MetadataAuthDetector()

    def run(self, image_bytes: bytes, caption: Optional[str] = None) -> UnifiedVerdict:
        image_result = self.image_detector.analyze(image_bytes)
        metadata_result = self.metadata_detector.analyze(image_bytes)
        text_result = self.text_detector.analyze(caption, image_bytes=image_bytes) if caption else None

        weights = dict(settings.weights)
        if text_result is None:
            # Redistribute the text weight across the remaining modalities.
            redistributed = weights.pop("text") / 2
            weights["image"] += redistributed
            weights["metadata"] += redistributed

        unified = weights["image"] * image_result.score + weights["metadata"] * metadata_result.score
        if text_result is not None:
            unified += weights["text"] * text_result.score

        # "Weakest link" rule: a weighted average can let one strongly-flagged
        # modality get diluted by two clean ones (e.g. blatant EXIF spoofing
        # hidden behind a visually clean image). The verdict is driven by the
        # worse of (a) the overall weighted score and (b) the worst *reliable*
        # component score. Only image and metadata participate in the veto --
        # both are deterministic/rule-based and low-noise. The text heuristic
        # fallback is comparatively noisy on short captions (burstiness is
        # nearly undefined for a single sentence), so it stays part of the
        # weighted average but is deliberately not allowed to single-handedly
        # veto an otherwise-clean verdict. The full `unified_score` (which
        # does include text) is still reported for transparency.
        verdict_score = min(unified, image_result.score, metadata_result.score)

        if verdict_score >= settings.suspicious_threshold:
            verdict = "authentic"
        elif verdict_score >= settings.fake_threshold:
            verdict = "suspicious"
        else:
            verdict = "likely_fake"

        reasons = []
        if image_result.label != "authentic":
            reasons.append(f"image signal: {image_result.label} ({image_result.method})")
        if metadata_result.label != "consistent":
            reasons.append(f"metadata signal: {metadata_result.label} ({', '.join(metadata_result.details.get('findings', [])) or 'n/a'})")
        if text_result is not None and text_result.label not in ("authentic", "consistent", "likely_human"):
            reasons.append(f"text signal: {text_result.label} ({text_result.method})")
        explanation = "; ".join(reasons) if reasons else "all modalities consistent with an authentic, unmodified asset"

        return UnifiedVerdict(
            verdict=verdict,
            unified_score=round(float(unified), 4),
            image=image_result,
            text=text_result,
            metadata=metadata_result,
            explanation=explanation,
        )

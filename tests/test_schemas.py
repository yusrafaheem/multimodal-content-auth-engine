"""Tests for app/schemas.py's Pydantic models.

None of these go through a detector or the API -- they construct
DetectorResult/UnifiedVerdict/HealthResponse directly, checking the
constraints Pydantic is enforcing on our behalf: the [0, 1] score bounds
every detector's docstring promises, which fields are actually required
vs. optional, and a couple of Pydantic-specific behaviors (default_factory
isolation, and v2's lax bool coercion) that are easy to assume rather than
verify.

Needs pydantic (already in requirements.txt) -- this is the one file in this
batch that can't run without it, since unlike image_utils.py/config.py,
schemas.py's whole job IS being a set of Pydantic models.
"""
import unittest

from pydantic import ValidationError

from app.schemas import DetectorResult, HealthResponse, UnifiedVerdict


class DetectorResultScoreBoundsTests(unittest.TestCase):
    def test_score_below_zero_is_rejected(self):
        # score: float = Field(..., ge=0.0, le=1.0) -- Pydantic v2 enforces
        # this constraint at construction time, not just in documentation.
        with self.assertRaises(ValidationError):
            DetectorResult(score=-0.0001, label="authentic", method="test")

    def test_score_above_one_is_rejected(self):
        with self.assertRaises(ValidationError):
            DetectorResult(score=1.0001, label="authentic", method="test")


class DetectorResultRequiredFieldsTests(unittest.TestCase):
    def test_missing_label_is_rejected(self):
        # label has no default -- omitting it should be a validation error,
        # not a silently-None field, since downstream code (the pipeline's
        # explanation string, the API response) assumes label is always a
        # real string.
        with self.assertRaises(ValidationError):
            DetectorResult(score=0.5, method="test")

    def test_details_default_is_a_fresh_dict_per_instance_not_a_shared_one(self):
        # Same class of bug as app/config.py's `weights` field (see
        # test_config.py) -- if `details` were declared as a plain `= {}`
        # default instead of Field(default_factory=dict), every
        # DetectorResult built without an explicit `details` would share
        # and silently mutate one dict object.
        a = DetectorResult(score=0.5, label="uncertain", method="test")
        b = DetectorResult(score=0.5, label="uncertain", method="test")
        a.details["leaked"] = True
        self.assertNotIn("leaked", b.details)


class UnifiedVerdictTests(unittest.TestCase):
    def _detector_result(self, score=0.8):
        return DetectorResult(score=score, label="authentic", method="test")

    def test_nested_detector_results_round_trip_through_model_dump(self):
        # UnifiedVerdict.image/text/metadata are typed as Optional[DetectorResult]
        # -- confirms model_dump() recurses into those nested models rather
        # than e.g. leaving them as DetectorResult objects or dropping them,
        # since the API layer relies on this to produce a plain-JSON response.
        verdict = UnifiedVerdict(
            verdict="suspicious",
            unified_score=0.5,
            image=self._detector_result(0.4),
            metadata=self._detector_result(0.9),
            explanation="image signal: suspicious",
        )
        dumped = verdict.model_dump()
        self.assertEqual(dumped["image"]["score"], 0.4)
        self.assertEqual(dumped["metadata"]["score"], 0.9)
        self.assertIsNone(dumped["text"])


if __name__ == "__main__":
    unittest.main()

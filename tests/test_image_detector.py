import unittest

from app.detectors.image_detector import ImageAuthDetector
from benchmarking.attack_fixtures import make_adversarial_fixture, make_clean_fixture, make_splice_fixture


class ImageDetectorHeuristicTests(unittest.TestCase):
    """These tests force use_pretrained_models=False so they run without torch
    or transformers installed -- the heuristic path is real, dependency-light
    signal-processing code, not a mock."""

    def setUp(self):
        self.detector = ImageAuthDetector(use_pretrained_models=False)

    def test_clean_image_scores_above_suspicious_threshold(self):
        fixture = make_clean_fixture(seed=2)
        result = self.detector.analyze(fixture.image_bytes)
        self.assertGreaterEqual(result.score, 0.6)
        self.assertEqual(result.method, "heuristic_ela_noise_residual")

    def test_adversarial_perturbation_lowers_score_vs_clean(self):
        clean = self.detector.analyze(make_clean_fixture(seed=2).image_bytes)
        adversarial = self.detector.analyze(make_adversarial_fixture(seed=2).image_bytes)
        self.assertLess(adversarial.score, clean.score)

    def test_spliced_image_flagged_by_ela(self):
        fixture = make_splice_fixture(seed=2)
        result = self.detector.analyze(fixture.image_bytes)
        self.assertGreater(result.details["ela_anomaly"], 0.5)


class ImageDetectorFallbackBehaviorTests(unittest.TestCase):
    def test_falls_back_gracefully_when_pretrained_requested_but_unavailable(self):
        # use_pretrained_models=True but torch/transformers aren't installed in
        # this environment -- the detector must not raise, it must fall back.
        detector = ImageAuthDetector(use_pretrained_models=True)
        fixture = make_clean_fixture(seed=3)
        result = detector.analyze(fixture.image_bytes)
        self.assertIsNotNone(result.score)


if __name__ == "__main__":
    unittest.main()

import unittest

from app.detectors.text_detector import TextConsistencyDetector


class TextDetectorHeuristicTests(unittest.TestCase):
    def setUp(self):
        self.detector = TextConsistencyDetector(use_pretrained_models=False)

    def test_short_text_is_marked_insufficient(self):
        result = self.detector.analyze("Nice photo")
        self.assertEqual(result.label, "insufficient_text")

    def test_repetitive_text_scores_lower_than_varied_text(self):
        repetitive = ("The cat sat on the mat. The cat sat on the mat. "
                       "The cat sat on the mat. The cat sat on the mat.")
        varied = ("Morning fog rolled off the lake while a lone heron picked "
                  "its way along the reeds, unhurried, as if the whole marsh "
                  "belonged to it alone.")
        rep_result = self.detector.analyze(repetitive)
        varied_result = self.detector.analyze(varied)
        self.assertLess(rep_result.score, varied_result.score)


if __name__ == "__main__":
    unittest.main()

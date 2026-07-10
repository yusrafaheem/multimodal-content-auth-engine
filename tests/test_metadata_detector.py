import unittest

from app.detectors.metadata_detector import MetadataAuthDetector
from benchmarking.attack_fixtures import make_clean_fixture, make_metadata_spoofed_fixture


class MetadataDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = MetadataAuthDetector()

    def test_clean_fixture_scores_high(self):
        fixture = make_clean_fixture(seed=1)
        result = self.detector.analyze(fixture.image_bytes)
        self.assertGreaterEqual(result.score, 0.9)
        self.assertEqual(result.label, "consistent")

    def test_spoofed_fixture_is_flagged(self):
        fixture = make_metadata_spoofed_fixture(seed=1)
        result = self.detector.analyze(fixture.image_bytes)
        self.assertLess(result.score, 0.6)
        self.assertIn("modify_date_before_original_date", result.details["findings"])

    def test_no_exif_is_penalized_but_not_catastrophic(self):
        # An image with genuinely no EXIF (e.g. a re-encoded screenshot) should
        # be treated as *slightly* suspicious, not automatically "fake" --
        # plenty of authentic images (screenshots, downloaded assets) lack EXIF.
        import io

        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(120, 130, 140))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        result = self.detector.analyze(buf.getvalue())
        self.assertIn("no_exif_data", result.details["findings"])
        self.assertGreater(result.score, 0.3)


if __name__ == "__main__":
    unittest.main()

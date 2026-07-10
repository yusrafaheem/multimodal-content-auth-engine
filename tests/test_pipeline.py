import unittest

from app.pipeline.consistency_pipeline import CrossModalConsistencyPipeline
from benchmarking.attack_fixtures import (
    make_adversarial_fixture,
    make_clean_fixture,
    make_metadata_spoofed_fixture,
    make_splice_fixture,
)


class PipelineFusionTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = CrossModalConsistencyPipeline(use_pretrained_models=False)

    def test_clean_fixture_is_authentic(self):
        f = make_clean_fixture(seed=7)
        verdict = self.pipeline.run(f.image_bytes, caption=f.caption)
        self.assertEqual(verdict.verdict, "authentic")

    def test_metadata_spoofing_is_not_averaged_away_by_clean_image(self):
        # Regression test for the "weakest link" fusion rule: a blatant EXIF
        # spoof must not get diluted into an "authentic" verdict just because
        # the pixel content itself looks clean.
        f = make_metadata_spoofed_fixture(seed=7)
        verdict = self.pipeline.run(f.image_bytes, caption=f.caption)
        self.assertNotEqual(verdict.verdict, "authentic")

    def test_adversarial_and_splice_are_flagged(self):
        for fixture_fn in (make_adversarial_fixture, make_splice_fixture):
            f = fixture_fn(seed=7)
            verdict = self.pipeline.run(f.image_bytes, caption=f.caption)
            self.assertNotEqual(verdict.verdict, "authentic", msg=f"{fixture_fn.__name__} was not flagged")

    def test_verdict_without_caption_still_works(self):
        f = make_clean_fixture(seed=8)
        verdict = self.pipeline.run(f.image_bytes, caption=None)
        self.assertIsNone(verdict.text)
        self.assertIsNotNone(verdict.image)
        self.assertIsNotNone(verdict.metadata)


if __name__ == "__main__":
    unittest.main()

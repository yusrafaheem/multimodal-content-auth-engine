"""Direct tests of app/utils/image_utils.py -- the pure Pillow/NumPy/OpenCV
math underneath both image_detector.py's heuristic path and
metadata_detector.py.

Every existing test file in this repo (test_image_detector.py etc.) only
exercises this module indirectly, through a detector. These tests go
straight at the functions themselves.

Deliberately import-light: this file only needs Pillow/NumPy/OpenCV (already
in requirements.txt), not pydantic/fastapi/torch -- one of the fastest and
most dependency-free files in the whole suite to run.
"""
import io
import unittest

import numpy as np
from PIL import Image

from app.utils.image_utils import error_level_analysis, load_image, noise_residual_score, to_numpy
from benchmarking.attack_fixtures import make_adversarial_fixture, make_clean_fixture, make_splice_fixture


def _solid_color_image(size=(64, 64), color=(120, 130, 140)):
    return Image.new("RGB", size, color=color)


class LoadImageTests(unittest.TestCase):
    def test_grayscale_image_is_converted_to_rgb(self):
        gray = Image.new("L", (32, 32), color=128)
        buf = io.BytesIO()
        gray.save(buf, "PNG")
        loaded = load_image(buf.getvalue())
        self.assertEqual(loaded.mode, "RGB")

    def test_garbage_bytes_raise_rather_than_returning_a_blank_image(self):
        # A silent "return something" here would be far more dangerous than
        # an exception -- a caller could end up authenticating garbage input
        # as if it were a real (blank/black) photo.
        with self.assertRaises(Exception):
            load_image(b"this is not an image, just some text bytes")


class ToNumpyTests(unittest.TestCase):
    def test_shape_matches_image_dimensions_height_width_channels(self):
        img = _solid_color_image(size=(50, 30))  # PIL size is (width, height)
        arr = to_numpy(img)
        self.assertEqual(arr.shape, (30, 50, 3))


class ErrorLevelAnalysisTests(unittest.TestCase):
    def test_anomaly_score_is_always_within_zero_one(self):
        for fixture_fn in (make_clean_fixture, make_adversarial_fixture, make_splice_fixture):
            img = load_image(fixture_fn(seed=11).image_bytes)
            _, anomaly = error_level_analysis(img)
            self.assertGreaterEqual(anomaly, 0.0)
            self.assertLessEqual(anomaly, 1.0)

    def test_a_flat_solid_color_image_has_near_zero_anomaly(self):
        # No edges or texture at all -- JPEG re-compression of a flat color
        # produces almost no error anywhere, so there's no "hot spot" for
        # ELA's percentile-vs-median ratio to catch.
        flat = _solid_color_image(size=(128, 128))
        _, anomaly = error_level_analysis(flat)
        self.assertLess(anomaly, 0.1)

    def test_spliced_image_scores_higher_than_the_clean_image_it_was_built_from(self):
        clean = load_image(make_clean_fixture(seed=11).image_bytes)
        spliced = load_image(make_splice_fixture(seed=11).image_bytes)
        _, clean_anomaly = error_level_analysis(clean)
        _, splice_anomaly = error_level_analysis(spliced)
        self.assertGreater(splice_anomaly, clean_anomaly)


class NoiseResidualScoreTests(unittest.TestCase):
    def test_score_is_always_within_zero_one_including_on_pure_random_noise(self):
        # Pure random noise is the adversarial case for this function's own
        # clipping logic -- maximally high-frequency content, the input most
        # likely to blow past an un-clamped ratio.
        rng = np.random.default_rng(0)
        noisy_arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        noisy_img = Image.fromarray(noisy_arr, mode="RGB")
        score = noise_residual_score(noisy_img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

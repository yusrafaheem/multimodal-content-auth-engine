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

from PIL import Image

from app.utils.image_utils import load_image


class LoadImageTests(unittest.TestCase):
    def test_grayscale_image_is_converted_to_rgb(self):
        gray = Image.new("L", (32, 32), color=128)
        buf = io.BytesIO()
        gray.save(buf, "PNG")
        loaded = load_image(buf.getvalue())
        self.assertEqual(loaded.mode, "RGB")


if __name__ == "__main__":
    unittest.main()

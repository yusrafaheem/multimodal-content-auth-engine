"""HTTP-level tests for app/main.py -- the FastAPI routes themselves.

Every existing test file in this repo tests a detector or the pipeline
directly, in-process. None of them go through the actual HTTP layer: request
parsing, multipart file uploads, form validation, or FastAPI's own error
responses for missing/invalid input. main.py currently has zero test
coverage of its own -- this file is the first.

Uses FastAPI's TestClient (Starlette under the hood), which drives real
ASGI request/response cycles against the app in-process -- no network socket,
but a genuinely different code path than calling a detector's .analyze()
directly, including all of FastAPI's own request validation.

Runs entirely in heuristic mode without any special setup: torch/transformers
aren't in requirements.txt (only requirements-ml.txt), so every detector's
`try: import torch ... except ImportError: fall back to heuristic` path kicks
in automatically regardless of the USE_PRETRAINED_MODELS flag -- same reason
the CI workflow's lightweight `pip install` step doesn't need to set that env
var either.
"""
import unittest

from fastapi.testclient import TestClient

from app.main import app
from benchmarking.attack_fixtures import make_clean_fixture, make_metadata_spoofed_fixture

# raise_server_exceptions=False: a later test in this file deliberately sends
# input the app doesn't handle gracefully -- the point is to observe the HTTP
# response the caller actually gets, not to have the unhandled exception
# re-raised into the test process the way TestClient does by default.
client = TestClient(app, raise_server_exceptions=False)


def _upload(fixture):
    return {"file": (f"{fixture.name}.jpg", fixture.image_bytes, "image/jpeg")}


class HealthEndpointTests(unittest.TestCase):
    def test_health_returns_ok_status_and_a_boolean_flag(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertIsInstance(body["use_pretrained_models"], bool)

    def test_cors_header_is_present_for_a_cross_origin_request(self):
        # The API is meant to be called from a browser-based client on a
        # different origin -- if CORSMiddleware were ever removed or
        # misconfigured, a cross-origin caller would get a silently-blocked
        # request in the browser (CORS failures don't show up as an HTTP
        # error status; the response has to be inspected for the header).
        resp = client.get("/health", headers={"origin": "https://example.com"})
        self.assertEqual(resp.headers.get("access-control-allow-origin"), "*")


class ImageEndpointTests(unittest.TestCase):
    def test_clean_image_scores_above_the_suspicious_threshold(self):
        # HTTP-layer confirmation of what test_image_detector.py already
        # checks in-process: a clean fixture uploaded as a real multipart
        # file, through FastAPI's own request parsing, should still score
        # above settings.suspicious_threshold and be labeled with the
        # heuristic method name (no ML deps installed in this environment).
        fixture = make_clean_fixture(seed=21)
        resp = client.post("/v1/authenticate/image", files=_upload(fixture))
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertGreaterEqual(body["score"], 0.6)
        self.assertEqual(body["method"], "heuristic_ela_noise_residual")

    def test_missing_file_is_a_422_not_a_500(self):
        # FastAPI's own request validation should reject this before any
        # application code runs -- a required UploadFile parameter with no
        # matching multipart part is a client error (422), not something
        # that should reach the route handler and blow up as a 500.
        resp = client.post("/v1/authenticate/image")
        self.assertEqual(resp.status_code, 422)


class MetadataEndpointTests(unittest.TestCase):
    def test_spoofed_metadata_is_flagged_with_a_specific_finding(self):
        # HTTP-layer regression test for metadata_detector.py's EXIF
        # rule-based checks, run through the real /v1/authenticate/metadata
        # route rather than calling the detector directly. Confirms both
        # the low score AND that the specific finding name survives request
        # parsing + response serialization intact -- not just a generic
        # "something's wrong" signal.
        fixture = make_metadata_spoofed_fixture(seed=21)
        resp = client.post("/v1/authenticate/metadata", files=_upload(fixture))
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertLess(body["score"], 0.6)
        self.assertIn("modify_date_before_original_date", body["details"]["findings"])

    def test_clean_metadata_is_labeled_consistent(self):
        # The negative-control counterpart to the spoofed-metadata test
        # above, run through the same HTTP route -- a fixture with no EXIF
        # red flags should come back "consistent", not just "not
        # spoofed"/anything-but-spoofed. Pins down the actual label string
        # the API contract promises for the clean case.
        fixture = make_clean_fixture(seed=21)
        resp = client.post("/v1/authenticate/metadata", files=_upload(fixture))
        self.assertEqual(resp.json()["label"], "consistent")


class FullPipelineEndpointTests(unittest.TestCase):
    def test_caption_is_optional_and_text_is_null_when_omitted(self):
        # /v1/authenticate's `caption` form field is declared Optional --
        # confirms it's genuinely optional at the HTTP layer (not just
        # "happens to work if you remember to send it"), and that skipping
        # it produces text: null in the response rather than a validation
        # error or a fabricated result.
        fixture = make_clean_fixture(seed=24)
        resp = client.post("/v1/authenticate", files=_upload(fixture))
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNone(body["text"])
        self.assertIsNotNone(body["image"])


class TextEndpointTests(unittest.TestCase):
    def test_valid_text_form_field_returns_a_heuristic_result(self):
        # /v1/authenticate/text takes a plain form field, not a file upload
        # -- a genuinely different FastAPI parameter type (Form(...) vs
        # UploadFile) than every other route this file tests, worth
        # confirming works over real HTTP.
        resp = client.post("/v1/authenticate/text", data={"text": (
            "Morning fog rolled off the lake while a lone heron picked its "
            "way along the reeds, unhurried, as if the whole marsh belonged "
            "to it alone."
        )})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["method"], "heuristic_text_stats")

    def test_missing_required_text_field_is_a_422(self):
        # Mirror of the missing-file test for the image endpoint: a
        # required Form(...) field with nothing supplied should fail
        # FastAPI's own validation before reaching the route handler.
        resp = client.post("/v1/authenticate/text")
        self.assertEqual(resp.status_code, 422)


class GarbageInputTests(unittest.TestCase):
    """None of the three single-modality endpoints wrap `load_image()` in a
    try/except -- a genuinely corrupt/non-image upload will raise
    PIL.UnidentifiedImageError, which FastAPI has no handler registered for,
    so it becomes an unhandled 500. Documenting that as current, real
    behavior (not something this test file silently papers over) rather
    than assuming input is always well-formed.
    """

    def test_non_image_bytes_uploaded_as_a_file_produce_a_500_not_a_crash_or_a_fake_200(self):
        # raise_server_exceptions=False on the module-level client is what
        # makes this observable at all -- without it, TestClient re-raises
        # the UnidentifiedImageError into this test process instead of
        # returning the 500 response a real HTTP client would actually see.
        resp = client.post(
            "/v1/authenticate/image",
            files={"file": ("not_a_photo.jpg", b"this is definitely not image data", "image/jpeg")},
        )
        self.assertEqual(resp.status_code, 500)


if __name__ == "__main__":
    unittest.main()

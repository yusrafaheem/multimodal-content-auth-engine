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


if __name__ == "__main__":
    unittest.main()

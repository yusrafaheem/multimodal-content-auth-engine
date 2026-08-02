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
from benchmarking.attack_fixtures import make_clean_fixture

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


if __name__ == "__main__":
    unittest.main()

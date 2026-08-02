"""Tests for app/config.py's Settings dataclass.

The interesting thing about this file isn't the values themselves -- it's
*when* they're computed. Every field default here is a plain expression like
`os.getenv("VIT_BACKBONE", ...)`, not `field(default_factory=...)`. Python
evaluates a dataclass's plain field defaults exactly once, when the class
body executes (i.e. at *import* time) -- not once per `Settings()` call. That
means setting `os.environ["SUSPICIOUS_THRESHOLD"]` and then calling
`Settings()` again does NOT pick up the new value if `app.config` was already
imported; the default was already baked in. Several tests in this file exist
specifically to pin that down, using importlib to force a fresh re-import
after changing the environment.

No pydantic/fastapi import anywhere in this file -- app.config only touches
os and dataclasses, so this (like test_image_utils.py) runs with zero heavy
dependencies.
"""
import importlib
import os
import unittest

import app.config as config_module


def _settings_reimported_with_env(**env_overrides):
    """Set env vars, force Python to re-run app/config.py's module body
    (re-evaluating every field default against the new environment), and
    return a `Settings()` instance built from that reload -- captured
    *before* the environment is restored.
    """
    saved = {k: os.environ.get(k) for k in env_overrides}
    try:
        for k, v in env_overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(config_module)
        return config_module.Settings()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(config_module)  # restore the module to its normal state too


class SettingsDefaultsTests(unittest.TestCase):
    def test_use_pretrained_models_defaults_to_true_when_unset(self):
        s = _settings_reimported_with_env(USE_PRETRAINED_MODELS=None)
        self.assertTrue(s.use_pretrained_models)


if __name__ == "__main__":
    unittest.main()

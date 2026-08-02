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

    def test_suspicious_threshold_is_always_greater_than_fake_threshold(self):
        # The pipeline's three-way verdict (authentic / suspicious /
        # likely_fake) silently breaks if this invariant is ever violated --
        # e.g. a misconfigured deployment setting FAKE_THRESHOLD above
        # SUSPICIOUS_THRESHOLD would make "suspicious" unreachable.
        # Documenting it as an explicit, checked assumption.
        s = config_module.Settings()
        self.assertGreater(s.suspicious_threshold, s.fake_threshold)


class SettingsMutableDefaultTests(unittest.TestCase):
    def test_weights_dict_uses_a_fresh_object_per_instance(self):
        # `weights` is declared with field(default_factory=lambda: {...}) --
        # if it had instead been a plain `= {...}` default (the classic
        # Python mutable-default-argument footgun), every Settings()
        # instance would share and silently mutate the SAME dict object.
        # This test would fail loudly if that regression were introduced.
        a = config_module.Settings()
        b = config_module.Settings()
        a.weights["image"] = 0.99
        self.assertNotEqual(b.weights["image"], 0.99)
        self.assertEqual(b.weights["image"], 0.5)


class SettingsEnvVarParsingTests(unittest.TestCase):
    def test_use_pretrained_models_is_NOT_recognized_when_env_is_uppercase_FALSE(self):
        # A real gotcha, not a hypothetical: the falsy-string check is
        # `not in ("0", "false", "False")` -- only two capitalization
        # variants are covered. "FALSE" (all caps) is NOT one of them, so
        # it's treated as truthy. Anyone setting USE_PRETRAINED_MODELS=FALSE
        # in a .env file or a shell export would silently get the opposite
        # of what they asked for. Documented here as current, real behavior.
        s = _settings_reimported_with_env(USE_PRETRAINED_MODELS="FALSE")
        self.assertTrue(s.use_pretrained_models)

    def test_setting_the_env_var_after_import_does_NOT_retroactively_change_defaults(self):
        # This is the payoff test for the module docstring's claim:
        # changing os.environ without reloading app.config must NOT affect
        # the already-imported module's Settings default.
        os.environ["SUSPICIOUS_THRESHOLD"] = "0.99"
        try:
            self.assertNotEqual(config_module.Settings().suspicious_threshold, 0.99)
            self.assertEqual(config_module.Settings().suspicious_threshold, 0.6)
        finally:
            del os.environ["SUSPICIOUS_THRESHOLD"]

    def test_a_non_numeric_threshold_env_var_raises_at_import_time_not_at_first_use(self):
        # suspicious_threshold's default is float(os.getenv(...)) -- that
        # float() call runs while the class body executes, so a bad value
        # blows up the *import*/reload itself with a ValueError, not later
        # when someone happens to read settings.suspicious_threshold. A
        # misconfigured deployment would fail loudly and immediately, not
        # produce a confusing error somewhere downstream.
        with self.assertRaises(ValueError):
            _settings_reimported_with_env(SUSPICIOUS_THRESHOLD="not-a-number")


if __name__ == "__main__":
    unittest.main()

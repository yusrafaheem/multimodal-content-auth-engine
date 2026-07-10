"""Synthetic benchmark fixture generator.

There's no real deepfake/adversarial-image dataset bundled with this repo (for
size and licensing reasons). Instead this module procedurally generates small,
labeled fixtures for each attack category so `run_benchmark.py` has something
concrete to score end-to-end, and so the harness itself -- not any particular
dataset -- is what gets exercised in CI.

For real evaluation numbers, point `run_benchmark.py` at a real dataset
(FaceForensics++, DFDC, CASIA) instead -- see benchmarking/README.md.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Fixture:
    name: str
    category: str  # "clean" | "adversarial_perturbation" | "metadata_spoofed" | "manipulated_splice"
    image_bytes: bytes
    caption: str
    expected_authentic: bool


def _base_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    # Smooth gradient + low-frequency blobs, roughly approximating natural-photo
    # statistics (low-frequency dominant, mild texture) far better than pure noise.
    x = np.linspace(0, 4 * np.pi, 224)
    y = np.linspace(0, 4 * np.pi, 224)
    xx, yy = np.meshgrid(x, y)
    base = 128 + 60 * np.sin(xx * 0.7 + rng.uniform(0, 3)) * np.cos(yy * 0.5 + rng.uniform(0, 3))
    texture = rng.normal(0, 4, size=base.shape)
    channel = np.clip(base + texture, 0, 255)
    arr = np.stack([channel, channel * 0.9 + 10, channel * 0.8 + 20], axis=-1).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _to_bytes(img: Image.Image, quality: int = 92, **exif_kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, **exif_kwargs)
    return buf.getvalue()


def make_clean_fixture(seed: int) -> Fixture:
    img = _base_image(seed)
    exif = Image.Exif()
    exif[271] = "SyntheticCam"       # Make
    exif[272] = "SC-1"               # Model
    exif[306] = "2026:06:01 10:00:00"  # DateTime
    exif[36867] = "2026:06:01 10:00:00"  # DateTimeOriginal
    return Fixture(
        name=f"clean_{seed}",
        category="clean",
        image_bytes=_to_bytes(img, exif=exif),
        caption="A softly lit abstract gradient photographed with a digital camera.",
        expected_authentic=True,
    )


def make_adversarial_fixture(seed: int) -> Fixture:
    img = _base_image(seed)
    arr = np.array(img).astype(np.float32)
    rng = np.random.default_rng(seed + 1000)
    # High-frequency perturbation, small in magnitude (imperceptible) but with
    # concentrated energy -- mimics an FGSM/PGD-style adversarial patch.
    perturbation = rng.normal(0, 18, size=arr.shape)
    arr = np.clip(arr + perturbation, 0, 255).astype(np.uint8)
    perturbed = Image.fromarray(arr, mode="RGB")
    return Fixture(
        name=f"adversarial_{seed}",
        category="adversarial_perturbation",
        image_bytes=_to_bytes(perturbed),
        caption="A softly lit abstract gradient photographed with a digital camera.",
        expected_authentic=False,
    )


def make_metadata_spoofed_fixture(seed: int) -> Fixture:
    img = _base_image(seed)
    exif = Image.Exif()
    exif[305] = "Adobe Photoshop 25.0"  # Software
    exif[306] = "2026:06:01 09:00:00"   # DateTime (modified BEFORE original -> inconsistent)
    exif[36867] = "2026:06:01 10:00:00"  # DateTimeOriginal
    return Fixture(
        name=f"metadata_spoofed_{seed}",
        category="metadata_spoofed",
        image_bytes=_to_bytes(img, exif=exif),
        caption="A raw, unedited photo straight from the camera.",
        expected_authentic=False,
    )


def make_splice_fixture(seed: int) -> Fixture:
    base = np.array(_base_image(seed)).astype(np.float32)
    patch = np.array(_base_image(seed + 500)).astype(np.float32)
    # Paste a differently-compressed patch into the middle of the image, then
    # save once, then re-open/re-save to create the double-compression seam
    # ELA is designed to catch.
    base[70:150, 70:150, :] = patch[70:150, 70:150, :]
    spliced = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    spliced.save(buf, "JPEG", quality=60)
    buf.seek(0)
    once = Image.open(buf)
    return Fixture(
        name=f"splice_{seed}",
        category="manipulated_splice",
        image_bytes=_to_bytes(once, quality=92),
        caption="An unedited group photo.",
        expected_authentic=False,
    )


def build_fixture_set(n_per_category: int = 5) -> list[Fixture]:
    fixtures: list[Fixture] = []
    for i in range(n_per_category):
        fixtures.append(make_clean_fixture(i))
        fixtures.append(make_adversarial_fixture(i))
        fixtures.append(make_metadata_spoofed_fixture(i))
        fixtures.append(make_splice_fixture(i))
    return fixtures

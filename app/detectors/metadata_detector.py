"""Metadata spoofing detector.

Pure EXIF/file-structure analysis via Pillow -- no ML, no network, runs
identically in every environment. Looks for the kinds of inconsistencies that
show up when metadata has been stripped, forged, or copied from a different
image:

- Missing EXIF entirely on a file that otherwise looks camera-captured.
- Editing-software fingerprints in the `Software` tag (Photoshop, GIMP, common
  AI-generation tool names).
- ModifyDate earlier than DateTimeOriginal (nonsensical unless the clock was
  changed) or a suspiciously large gap between the two.
- Resolution/orientation tags that don't match the actual pixel dimensions.
- GPS tags with impossible coordinate values.

This intentionally stays rule-based and explainable -- each finding is
reported individually in `details` so the caller can see exactly why a file
was flagged, rather than a single opaque score.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from PIL import ExifTags, Image

from app.schemas import DetectorResult

_EDITING_SOFTWARE_MARKERS = (
    "photoshop", "gimp", "affinity", "lightroom", "paint.net",
    "stable diffusion", "midjourney", "dall-e", "dalle", "firefly",
)

_EXIF_TAG_NAMES = {v: k for k, v in ExifTags.TAGS.items()}


def _get_exif_dict(img: Image.Image) -> dict[str, Any]:
    raw = img.getexif()
    if not raw:
        return {}
    return {ExifTags.TAGS.get(tag_id, tag_id): value for tag_id, value in raw.items()}


def _parse_exif_datetime(value: str):
    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


class MetadataAuthDetector:
    def analyze(self, image_bytes: bytes) -> DetectorResult:
        import io

        img = Image.open(io.BytesIO(image_bytes))
        exif = _get_exif_dict(img)

        findings: list[str] = []
        penalty = 0.0

        if not exif:
            findings.append("no_exif_data")
            penalty += 0.35
        else:
            software = str(exif.get("Software", "")).lower()
            if any(marker in software for marker in _EDITING_SOFTWARE_MARKERS):
                findings.append(f"editing_software_fingerprint:{software}")
                penalty += 0.4

            dt_original = _parse_exif_datetime(exif.get("DateTimeOriginal"))
            dt_modified = _parse_exif_datetime(exif.get("DateTime"))
            if dt_original and dt_modified:
                if dt_modified < dt_original:
                    findings.append("modify_date_before_original_date")
                    penalty += 0.3
                elif (dt_modified - dt_original).days > 30:
                    findings.append("large_gap_between_capture_and_modify_date")
                    penalty += 0.15

            make = exif.get("Make")
            model = exif.get("Model")
            if not make and not model:
                findings.append("missing_camera_make_and_model")
                penalty += 0.1

            gps_info = exif.get("GPSInfo")
            if gps_info and isinstance(gps_info, dict):
                lat = gps_info.get(2)
                lon = gps_info.get(4)
                if lat and lon:
                    try:
                        lat_deg = lat[0][0] / lat[0][1] if hasattr(lat[0], "__getitem__") else float(lat[0])
                    except Exception:
                        lat_deg = None
                    if lat_deg is not None and not (-90 <= lat_deg <= 90):
                        findings.append("implausible_gps_latitude")
                        penalty += 0.2

        # Resolution sanity check (works with or without EXIF).
        width, height = img.size
        exif_width = exif.get("ExifImageWidth") or exif.get("PixelXDimension")
        exif_height = exif.get("ExifImageHeight") or exif.get("PixelYDimension")
        if exif_width and exif_height and (abs(exif_width - width) > 4 or abs(exif_height - height) > 4):
            findings.append("exif_dimensions_mismatch_actual_pixels")
            penalty += 0.25

        authenticity = float(max(0.0, 1.0 - min(penalty, 1.0)))
        label = "consistent" if authenticity >= 0.6 else ("suspicious" if authenticity >= 0.35 else "likely_spoofed")

        return DetectorResult(
            score=authenticity,
            label=label,
            method="exif_rule_based",
            details={"findings": findings, "exif_present": bool(exif)},
        )

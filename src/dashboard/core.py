from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

CLASS_TO_CODE = {"background": 0, "no damage": 1, "minor": 2, "major": 3, "destroyed": 4}


@dataclass(frozen=True)
class TriageRecord:
    tile_id: str
    predicted_severe_area: float
    calibrated_severe_confidence: float
    priority: float


def priority_score(predicted_severe_area: float, calibrated_severe_confidence: float) -> float:
    return float(max(0.0, predicted_severe_area) * np.clip(calibrated_severe_confidence, 0.0, 1.0))


def rank_tiles(records: list[TriageRecord]) -> list[TriageRecord]:
    return sorted(records, key=lambda r: r.priority, reverse=True)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def registration_sanity_check(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    max_shift_pixels: float = 12.0,
    min_response: float = 0.05,
) -> dict[str, float | bool]:
    if pre.shape != post.shape:
        raise ValueError(f"Before/after images must have identical dimensions; got {pre.shape} vs {post.shape}")
    if pre.ndim != 3 or pre.shape[2] != 3:
        raise ValueError("Images must be RGB arrays with shape HxWx3")

    gray_pre = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_post = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY).astype(np.float32)
    max_dim = max(gray_pre.shape)
    scale = min(1.0, 512.0 / max_dim)
    if scale < 1.0:
        size = (max(16, int(gray_pre.shape[1] * scale)), max(16, int(gray_pre.shape[0] * scale)))
        gray_pre = cv2.resize(gray_pre, size, interpolation=cv2.INTER_AREA)
        gray_post = cv2.resize(gray_post, size, interpolation=cv2.INTER_AREA)

    shift, response = cv2.phaseCorrelate(gray_pre, gray_post)
    shift_x = float(shift[0] / scale)
    shift_y = float(shift[1] / scale)
    magnitude = math.hypot(shift_x, shift_y)
    ok = magnitude <= max_shift_pixels and float(response) >= min_response
    return {
        "shift_x_pixels": shift_x,
        "shift_y_pixels": shift_y,
        "shift_magnitude_pixels": magnitude,
        "response": float(response),
        "ok": bool(ok),
    }


def corrected_mask(
    building_mask: np.ndarray,
    damage_mask: np.ndarray,
    *,
    tile_override: str | None = None,
    building_override_id: int | None = None,
    building_override_class: str | None = None,
) -> np.ndarray:
    corrected = damage_mask.copy()
    if tile_override and tile_override in CLASS_TO_CODE and tile_override != "background":
        corrected[building_mask == 1] = CLASS_TO_CODE[tile_override]

    if building_override_id is not None and building_override_class in CLASS_TO_CODE:
        num, labels = cv2.connectedComponents(building_mask.astype(np.uint8), connectivity=8)
        if 0 < building_override_id < num:
            corrected[labels == building_override_id] = CLASS_TO_CODE[building_override_class]
    corrected[building_mask == 0] = 0
    return corrected.astype(np.uint8)


def mask_to_pixel_geojson(mask: np.ndarray, class_names: dict[int, str], min_area_pixels: int = 8) -> dict:
    features: list[dict] = []
    for cls in range(1, int(mask.max()) + 1):
        if cls not in class_names:
            continue
        binary = (mask == cls).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area_pixels or len(contour) < 3:
                continue
            coords = contour[:, 0, :].astype(float).tolist()
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "severity_code": cls,
                        "severity": class_names[cls],
                        "pixel_area": area,
                    },
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                }
            )
    return {
        "type": "FeatureCollection",
        "metadata": {
            "coordinate_space": "pixel",
            "note": "Pixel-space polygons; no geographic CRS transform was applied.",
        },
        "features": features,
    }

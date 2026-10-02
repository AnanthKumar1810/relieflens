from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

DAMAGE_TO_CLASS = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}
UNCLASSIFIED_LABELS = {"un-classified", "unclassified"}
BACKGROUND = 0
IGNORE_INDEX = 255


@dataclass(frozen=True)
class BuildingAnnotation:
    uid: str
    subtype: str
    polygons: tuple[np.ndarray, ...]

    @property
    def polygon(self) -> np.ndarray:
        """Backward-compatible access to the first polygon component."""
        return self.polygons[0]


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    disaster: str
    disaster_type: str
    pre_image: str
    post_image: str
    pre_label: str | None
    post_label: str | None


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_xy(obj: dict) -> Iterable[dict]:
    features = obj.get("features", {})
    for feature in features.get("xy", []):
        if feature.get("properties", {}).get("feature_type", "building") == "building":
            yield feature


def load_buildings(
    label_path: str | Path,
    *,
    post: bool = True,
    unknown_damage_policy: str = "error",
) -> list[BuildingAnnotation]:
    if unknown_damage_policy not in {"error", "ignore"}:
        raise ValueError("unknown_damage_policy must be 'error' or 'ignore'")

    obj = read_json(label_path)
    result: list[BuildingAnnotation] = []
    for feat in _iter_xy(obj):
        props = feat.get("properties", {})
        subtype = props.get("subtype", "no-damage") if post else "no-damage"
        if post and subtype not in DAMAGE_TO_CLASS and subtype not in UNCLASSIFIED_LABELS:
            message = f"Unknown xBD damage subtype {subtype!r} in {label_path}"
            if unknown_damage_policy == "error":
                raise ValueError(message)
            warnings.warn(message + "; annotation ignored", RuntimeWarning, stacklevel=2)
            continue

        polygons = _polygons_from_wkt(feat.get("wkt", ""))
        if not polygons:
            continue
        result.append(BuildingAnnotation(str(props.get("uid", "")), subtype, polygons))
    return result


def _polygons_from_wkt(wkt: str) -> tuple[np.ndarray, ...]:
    # xBD provides pixel-space polygons under features["xy"].wkt.
    try:
        from shapely import wkt as shapely_wkt

        geom = shapely_wkt.loads(wkt)
        if geom.is_empty:
            return ()
        if geom.geom_type == "Polygon":
            return (np.asarray(geom.exterior.coords, dtype=np.float32),)
        if geom.geom_type == "MultiPolygon":
            return tuple(
                np.asarray(poly.exterior.coords, dtype=np.float32)
                for poly in geom.geoms
                if not poly.is_empty and len(poly.exterior.coords) >= 4
            )
        if geom.geom_type == "GeometryCollection":
            polygons: list[np.ndarray] = []
            for child in geom.geoms:
                if child.geom_type == "Polygon" and not child.is_empty:
                    polygons.append(np.asarray(child.exterior.coords, dtype=np.float32))
                elif child.geom_type == "MultiPolygon":
                    polygons.extend(
                        np.asarray(poly.exterior.coords, dtype=np.float32)
                        for poly in child.geoms
                        if not poly.is_empty and len(poly.exterior.coords) >= 4
                    )
            return tuple(polygons)
    except Exception as exc:
        raise ValueError(f"Invalid WKT geometry: {wkt!r}") from exc
    return ()


def render_building_mask(
    label_path: str | Path,
    height: int,
    width: int,
    *,
    damage: bool,
    unknown_damage_policy: str = "error",
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    ignore = np.zeros((height, width), dtype=bool)
    for ann in load_buildings(label_path, post=damage, unknown_damage_policy=unknown_damage_policy):
        for polygon in ann.polygons:
            pts = np.round(polygon).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
            if len(pts) < 3:
                continue
            if damage and ann.subtype in UNCLASSIFIED_LABELS:
                cv2.fillPoly(ignore, [pts], 1)
            elif damage:
                cls = DAMAGE_TO_CLASS.get(ann.subtype)
                if cls is None:
                    raise ValueError(f"Unknown damage subtype {ann.subtype!r}")
                cv2.fillPoly(mask, [pts], cls)
            else:
                cv2.fillPoly(mask, [pts], 1)
    if damage:
        mask[ignore] = IGNORE_INDEX
    return mask


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.height, im.width


def disaster_from_label(path: str | Path) -> tuple[str, str]:
    meta = read_json(path).get("metadata", {})
    return str(meta.get("disaster", "unknown")), str(meta.get("disaster_type", "unknown"))


def find_named_file(root: Path, relative_candidates: list[Path]) -> Path | None:
    for rel in relative_candidates:
        p = root / rel
        if p.exists():
            return p
    return None

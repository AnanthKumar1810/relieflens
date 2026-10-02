from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.prepare import discover
from src.data.xbd import DAMAGE_TO_CLASS, render_building_mask


def _make_json(path: Path, subtype: str = "major-damage"):
    obj = {"metadata": {"disaster": "disaster-a", "disaster_type": "flooding", "width": 64, "height": 64},
           "features": {"lng_lat": [], "xy": [{"properties": {"uid": "1", "feature_type": "building", "subtype": subtype},
           "wkt": "POLYGON ((10 10, 30 10, 30 30, 10 30, 10 10))"}]}}
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_mask_rendering(tmp_path):
    label = tmp_path / "x_post_0000.json"; _make_json(label, "destroyed")
    mask = render_building_mask(label, 64, 64, damage=True)
    assert int(mask[20,20]) == DAMAGE_TO_CLASS["destroyed"]
    build = render_building_mask(label, 64, 64, damage=False)
    assert int(build[20,20]) == 1


def test_discover_pairs(tmp_path):
    root = tmp_path / "xbd"; (root/"train"/"images").mkdir(parents=True); (root/"train"/"labels").mkdir(parents=True)
    for phase in ["pre", "post"]:
        Image.fromarray(np.zeros((64,64,3), np.uint8)).save(root/"train"/"images"/f"disaster-a_00000001_{phase}_disaster.png")
        _make_json(root/"train"/"labels"/f"disaster-a_00000001_{phase}_disaster.json", "no-damage" if phase=="pre" else "major-damage")
    rows = discover(root)
    assert len(rows) == 1
    assert rows[0]["disaster"] == "disaster-a"


def test_split_rejects_disaster_assigned_to_multiple_splits():
    from src.data.prepare import split_rows
    rows = [{"sample_id":"a","disaster":"d1"}]
    try:
        split_rows(rows, {"disasters":{"train":["d1"],"val":["d1"],"test":[]},"strict_event_split":True})
    except ValueError as exc:
        assert "assigned to both" in str(exc)
    else:
        raise AssertionError("Expected duplicate disaster split assignment to fail")


def test_random_crop_repeats_each_training_image_configurably(tmp_path):
    from src.data.dataset import XBDPairDataset
    root = tmp_path / "xbd"; (root / "images").mkdir(parents=True); (root / "labels").mkdir(parents=True)
    for phase in ["pre", "post"]:
        Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(root / "images" / f"d_00000001_{phase}_disaster.png")
        _make_json(root / "labels" / f"d_00000001_{phase}_disaster.json")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,disaster,disaster_type,pre_image,post_image,pre_label,post_label\n"
        f"x,d,f,{root / 'images' / 'd_00000001_pre_disaster.png'},{root / 'images' / 'd_00000001_post_disaster.png'},{root / 'labels' / 'd_00000001_pre_disaster.json'},{root / 'labels' / 'd_00000001_post_disaster.json'}\n",
        encoding="utf-8",
    )
    ds = XBDPairDataset(manifest, patch_size=32, stride=32, training=True, normalize="none", random_crop=True, random_crops_per_image=4, augment=False)
    assert len(ds) == 4


def test_multipolygon_preserves_all_components(tmp_path):
    label = tmp_path / "multi.json"
    obj = {
        "metadata": {"disaster": "d", "disaster_type": "flooding"},
        "features": {
            "xy": [
                {
                    "properties": {"uid": "m1", "feature_type": "building", "subtype": "destroyed"},
                    "wkt": "MULTIPOLYGON (((5 5, 10 5, 10 10, 5 10, 5 5)), ((20 20, 25 20, 25 25, 20 25, 20 20)))",
                }
            ]
        },
    }
    label.write_text(json.dumps(obj), encoding="utf-8")
    mask = render_building_mask(label, 32, 32, damage=True)
    assert int(mask[7, 7]) == DAMAGE_TO_CLASS["destroyed"]
    assert int(mask[22, 22]) == DAMAGE_TO_CLASS["destroyed"]


def test_unknown_damage_label_fails_loudly(tmp_path):
    label = tmp_path / "unknown.json"
    obj = {
        "metadata": {"disaster": "d", "disaster_type": "flooding"},
        "features": {
            "xy": [
                {
                    "properties": {"uid": "u1", "feature_type": "building", "subtype": "invented-label"},
                    "wkt": "POLYGON ((5 5, 10 5, 10 10, 5 10, 5 5))",
                }
            ]
        },
    }
    label.write_text(json.dumps(obj), encoding="utf-8")
    try:
        render_building_mask(label, 32, 32, damage=True)
    except ValueError as exc:
        assert "Unknown xBD damage subtype" in str(exc)
    else:
        raise AssertionError("Unknown damage labels must not silently become background")


def test_discover_keeps_same_basename_in_different_directories_distinct(tmp_path):
    root = tmp_path / "xbd"
    for subdir in ["a", "b"]:
        (root / "train" / subdir / "images").mkdir(parents=True)
        (root / "train" / subdir / "labels").mkdir(parents=True)
        for phase in ["pre", "post"]:
            Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                root / "train" / subdir / "images" / f"same_0001_{phase}_disaster.png"
            )
            _make_json(root / "train" / subdir / "labels" / f"same_0001_{phase}_disaster.json")
    rows = discover(root)
    assert len(rows) == 2
    assert rows[0]["sample_id"] != rows[1]["sample_id"]

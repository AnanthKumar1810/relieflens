from __future__ import annotations

import numpy as np

from src.dashboard.core import corrected_mask, mask_to_pixel_geojson, priority_score, registration_sanity_check


def test_registration_sanity_on_identical_pair():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 120
    result = registration_sanity_check(image, image, max_shift_pixels=2.0, min_response=0.01)
    assert result["shift_magnitude_pixels"] < 0.1
    assert result["ok"] is True


def test_corrected_mask_applies_tile_override():
    building = np.zeros((16, 16), dtype=np.uint8)
    building[2:6, 2:6] = 1
    damage = building.copy()
    corrected = corrected_mask(building, damage, tile_override="destroyed")
    assert int(corrected[3, 3]) == 4


def test_geojson_is_explicitly_pixel_space():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:8, 2:8] = 4
    data = mask_to_pixel_geojson(mask, {0: "background", 4: "destroyed"})
    assert data["type"] == "FeatureCollection"
    assert data["metadata"]["coordinate_space"] == "pixel"
    assert len(data["features"]) == 1


def test_priority_score_is_bounded():
    assert 0.0 <= priority_score(0.7, 0.9) <= 1.0

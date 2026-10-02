from __future__ import annotations

import csv
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.xbd import PairRecord, render_building_mask

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_rows(csv_path: str | Path) -> list[PairRecord]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [PairRecord(**row) for row in csv.DictReader(f)]


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _pad_to_multiple(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    ph = max(0, size - h)
    pw = max(0, size - w)
    if arr.ndim == 3:
        return np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return np.pad(arr, ((0, ph), (0, pw)), mode="constant")


def _crop(arr: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    arr = _pad_to_multiple(arr, size)
    return arr[y : y + size, x : x + size]


def _normalize(img: np.ndarray, mode: str) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    if mode == "imagenet":
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


class XBDPairDataset(Dataset):
    """Paired xBD dataset with binary building and semantic damage targets.

    Damage target classes: 0 background, 1 no-damage, 2 minor, 3 major, 4 destroyed.
    Unclassified building pixels use 255 and are ignored by the damage loss/metrics.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        patch_size: int = 256,
        stride: int = 256,
        training: bool = True,
        normalize: str = "imagenet",
        random_crop: bool = True,
        random_crops_per_image: int = 4,
        augment: bool = True,
        unknown_damage_policy: str = "error",
        paired_rotate90: bool = True,
    ) -> None:
        self.rows = _load_rows(manifest_csv)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.training = training
        self.normalize = normalize
        self.random_crop = random_crop and training
        self.random_crops_per_image = max(1, int(random_crops_per_image))
        self.augment = augment and training
        self.unknown_damage_policy = unknown_damage_policy
        self.paired_rotate90 = paired_rotate90
        if not self.rows:
            raise ValueError(f"No samples found in {manifest_csv}")

        self.index: list[tuple[int, int, int]] = []
        for sample_idx, row in enumerate(self.rows):
            with Image.open(row.post_image) as im:
                h, w = im.height, im.width
            ys = list(range(0, max(h - self.patch_size, 0) + 1, self.stride)) or [0]
            xs = list(range(0, max(w - self.patch_size, 0) + 1, self.stride)) or [0]
            if not training:
                if ys[-1] + self.patch_size < h:
                    ys.append(h - self.patch_size)
                if xs[-1] + self.patch_size < w:
                    xs.append(w - self.patch_size)
            for y in ys:
                for x in xs:
                    self.index.append((sample_idx, y, x))

    def __len__(self) -> int:
        if self.training and self.random_crop:
            return len(self.rows) * self.random_crops_per_image
        return len(self.index)

    def _sample_coords(self, idx: int, h: int, w: int) -> tuple[int, int]:
        max_y = max(0, h - self.patch_size)
        max_x = max(0, w - self.patch_size)
        if self.training and self.random_crop:
            return random.randint(0, max_y), random.randint(0, max_x)
        _, y, x = self.index[idx]
        return min(y, max_y), min(x, max_x)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample_idx = (idx % len(self.rows)) if self.training and self.random_crop else self.index[idx][0]
        row = self.rows[sample_idx]
        pre = _load_rgb(row.pre_image)
        post = _load_rgb(row.post_image)
        h, w = post.shape[:2]
        building = render_building_mask(row.post_label, h, w, damage=False, unknown_damage_policy=self.unknown_damage_policy)
        damage = render_building_mask(row.post_label, h, w, damage=True, unknown_damage_policy=self.unknown_damage_policy)
        y, x = self._sample_coords(idx, h, w)
        pre = _crop(pre, y, x, self.patch_size)
        post = _crop(post, y, x, self.patch_size)
        building = _crop(building, y, x, self.patch_size)
        damage = _crop(damage, y, x, self.patch_size)

        if self.augment:
            if self.paired_rotate90 and random.random() < 0.5:
                k = random.randint(1, 3)
                pre = np.rot90(pre, k).copy(); post = np.rot90(post, k).copy()
                building = np.rot90(building, k).copy(); damage = np.rot90(damage, k).copy()
            if random.random() < 0.5:
                pre = np.flip(pre, axis=1).copy(); post = np.flip(post, axis=1).copy()
                building = np.flip(building, axis=1).copy(); damage = np.flip(damage, axis=1).copy()
            if random.random() < 0.5:
                pre = np.flip(pre, axis=0).copy(); post = np.flip(post, axis=0).copy()
                building = np.flip(building, axis=0).copy(); damage = np.flip(damage, axis=0).copy()

        pre_t = torch.from_numpy(_normalize(pre, self.normalize).transpose(2, 0, 1)).float()
        post_t = torch.from_numpy(_normalize(post, self.normalize).transpose(2, 0, 1)).float()
        building_t = torch.from_numpy((building > 0).astype(np.int64))
        damage_t = torch.from_numpy(damage.astype(np.int64))
        return {
            "pre": pre_t,
            "post": post_t,
            "building": building_t,
            "damage": damage_t,
            "sample_id": row.sample_id,
            "disaster": row.disaster,
        }


def save_manifest_rows(rows: list[PairRecord], path: str | Path) -> None:
    import csv
    fields = list(asdict(rows[0]).keys()) if rows else ["sample_id", "disaster", "disaster_type", "pre_image", "post_image", "pre_label", "post_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(asdict(row))

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.relieflens import ReliefLensNet
from src.models.factory import build_model

CLASS_NAMES = {0: "background", 1: "no damage", 2: "minor", 3: "major", 4: "destroyed"}


def _norm(x: np.ndarray, mode: str) -> np.ndarray:
    y = x.astype(np.float32) / 255.0
    if mode == "imagenet":
        y = (y - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return y


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    cfg.setdefault("model", {"dropout": 0.1, "num_damage_classes": 5, "pretrained": False})
    model = build_model(ckpt.get("model_type", "siamese"), cfg, pretrained_override=False).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, cfg


def checkpoint_metadata(path: str | Path, device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device)
    metrics = ckpt.get("metrics", {})
    return {
        "model_type": ckpt.get("model_type", "siamese"),
        "epoch": ckpt.get("epoch"),
        "temperature": float(ckpt.get("calibration_temperature", 1.0)),
        "damage_macro_f1": metrics.get("damage_macro_f1"),
    }


def predict_pair(
    model: ReliefLensNet,
    pre: np.ndarray,
    post: np.ndarray,
    device: str = "cpu",
    normalize: str = "imagenet",
    temperature: float = 1.0,
    allow_resize: bool = False,
    abstain_confidence_threshold: float = 0.60,
    abstain_entropy_threshold: float = 0.65,
) -> dict:
    if pre.ndim != 3 or post.ndim != 3 or pre.shape[2] != 3 or post.shape[2] != 3:
        raise ValueError("Both images must be RGB arrays with shape HxWx3")
    if pre.shape != post.shape:
        if not allow_resize:
            raise ValueError(
                f"Before/after images must have identical dimensions for aligned inference; got {pre.shape} vs {post.shape}"
            )
        h, w = post.shape[:2]
        pre = cv2.resize(pre, (w, h), interpolation=cv2.INTER_AREA)

    h, w = post.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    pre2 = np.pad(pre, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    post2 = np.pad(post, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    pre_t = torch.from_numpy(_norm(pre2, normalize).transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    post_t = torch.from_numpy(_norm(post2, normalize).transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(pre_t, post_t)
        bprob = out["building_logits"].softmax(1)[0]
        dlog = out["damage_logits"][0] / max(float(temperature), 1e-3)
        dprob = dlog.softmax(0)
    bmask = bprob.argmax(0).cpu().numpy()[:h, :w].astype(np.uint8)
    dmask = dprob.argmax(0).cpu().numpy()[:h, :w].astype(np.uint8)
    dmask[bmask == 0] = 0
    probs = dprob.cpu().numpy()[:, :h, :w]
    max_confidence = float(probs.max(axis=0).mean())
    entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=0)
    normalized_entropy = float((entropy / np.log(probs.shape[0])).mean())
    severe_conf = probs[3] + probs[4]
    severe = np.isin(dmask, [3, 4])
    area = float(severe.mean())
    conf = float(severe_conf[severe].mean()) if severe.any() else 0.0
    priority = area * conf
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bmask, connectivity=8)
    buildings = []
    for idx in range(1, num):
        area_px = int(stats[idx, cv2.CC_STAT_AREA])
        if area_px < 8:
            continue
        component_mask = labels == idx
        class_probs = probs[:, component_mask].mean(axis=1)
        cls = int(np.argmax(class_probs))
        buildings.append(
            {
                "building_id": idx,
                "area_pixels": area_px,
                "centroid_x": float(centroids[idx][0]),
                "centroid_y": float(centroids[idx][1]),
                "damage_class": CLASS_NAMES[cls],
                "confidence": float(class_probs[cls]),
            }
        )
    return {
        "building_mask": bmask,
        "damage_mask": dmask,
        "damage_probs": probs,
        "priority": float(priority),
        "logits": dlog.cpu().numpy()[:, :h, :w],
        "buildings": buildings,
        "temperature": float(temperature),
        "mean_confidence": max_confidence,
        "normalized_entropy": normalized_entropy,
        "abstain": bool(max_confidence < abstain_confidence_threshold or normalized_entropy > abstain_entropy_threshold),
    }


def overlay_damage(image: np.ndarray, damage_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    colors = {
        1: np.array([80, 180, 80], np.float32),
        2: np.array([240, 220, 40], np.float32),
        3: np.array([240, 140, 40], np.float32),
        4: np.array([220, 50, 50], np.float32),
    }
    for cls, color in colors.items():
        m = damage_mask == cls
        out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)

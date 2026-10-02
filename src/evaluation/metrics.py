from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.data.xbd import IGNORE_INDEX


def binary_scores_from_counts(intersection: int, union: int, pred_positive: int, target_positive: int) -> dict[str, float]:
    iou = intersection / union if union else 1.0
    denom = pred_positive + target_positive
    f1 = (2 * intersection / denom) if denom else 1.0
    return {"building_iou": float(iou), "building_f1": float(f1)}


def binary_iou_f1(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return binary_scores_from_counts(*update_binary_counts(pred, target))


def damage_scores(pred: np.ndarray, target: np.ndarray, num_classes: int = 5) -> dict[str, float]:
    valid = target != IGNORE_INDEX
    y_true = target[valid].ravel()
    y_pred = pred[valid].ravel()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return damage_scores_from_confusion_matrix(cm)


def damage_scores_from_confusion_matrix(cm: np.ndarray) -> dict[str, float]:
    cm = np.asarray(cm, dtype=np.int64)
    num_classes = cm.shape[0]
    scores = []
    for cls in range(1, num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    macro = float(np.mean(scores)) if scores else 0.0
    out = {"damage_macro_f1": macro}
    names = ["no_damage", "minor", "major", "destroyed"]
    for name, score in zip(names, scores):
        out[f"damage_f1_{name}"] = float(score)
    out["damage_confusion_matrix"] = cm.tolist()
    return out


def triage_recall_from_weights(
    priority: np.ndarray,
    severe_weights: np.ndarray,
    fractions: list[float],
) -> dict[str, float]:
    priority = np.asarray(priority, dtype=np.float64)
    severe_weights = np.asarray(severe_weights, dtype=np.float64)
    if priority.shape != severe_weights.shape:
        raise ValueError("priority and severe_weights must have the same shape")
    order = np.argsort(-priority)
    total = severe_weights.sum()
    out: dict[str, float] = {}
    if total == 0:
        for fraction in fractions:
            out[f"top_{int(fraction * 100)}_severe_pixel_recall"] = float("nan")
        return out
    for fraction in fractions:
        k = max(1, int(np.ceil(fraction * len(priority))))
        covered = severe_weights[order[:k]].sum()
        out[f"top_{int(fraction * 100)}_severe_pixel_recall"] = float(covered / total)
    return out


def triage_recall(
    priority: np.ndarray,
    target_damage: np.ndarray,
    fractions: list[float],
    severe_classes: tuple[int, ...] = (3, 4),
) -> dict[str, float]:
    severe = np.isin(target_damage, severe_classes)
    weights = severe.reshape(severe.shape[0], -1).sum(axis=1)
    return triage_recall_from_weights(priority, weights, fractions)


def update_binary_counts(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    p = pred.astype(bool)
    t = target.astype(bool)
    intersection = int(np.logical_and(p, t).sum())
    union = int(np.logical_or(p, t).sum())
    return intersection, union, int(p.sum()), int(t.sum())


def update_damage_confusion(
    cm: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 5,
) -> np.ndarray:
    valid = target != IGNORE_INDEX
    y_true = target[valid].ravel()
    y_pred = pred[valid].ravel()
    if y_true.size:
        cm += confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm


def summarize_predictions(
    building_logits,
    damage_logits,
    damage_target=None,
) -> dict:
    bprob = building_logits.softmax(dim=1)
    dprob = damage_logits.softmax(dim=1)
    bpred = bprob.argmax(1)
    dpred = dprob.argmax(1)
    severe_conf = dprob[:, [3, 4]].sum(1)
    severe_mask = torch.isin(dpred, torch.tensor([3, 4], device=dpred.device))
    severe_area = severe_mask.flatten(1).float().mean(1)
    mean_severe_conf = torch.where(
        severe_mask.flatten(1).any(1),
        (severe_conf * severe_mask.float()).flatten(1).sum(1) / severe_mask.flatten(1).float().sum(1).clamp_min(1.0),
        torch.zeros_like(severe_area),
    )
    priority = (severe_area * mean_severe_conf).detach().cpu().numpy()
    return {"building_pred": bpred, "damage_pred": dpred, "damage_probs": dprob, "priority": priority}

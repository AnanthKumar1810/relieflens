from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import XBDPairDataset
from src.data.xbd import IGNORE_INDEX
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.metrics import (
    binary_scores_from_counts,
    damage_scores_from_confusion_matrix,
    triage_recall_from_weights,
    update_binary_counts,
    update_damage_confusion,
)
from src.models.factory import build_model
from src.utils.config import load_config
from src.utils.seed import seed_everything


def load_model(checkpoint: str | Path, cfg: dict, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    model_type = ckpt.get("model_type", "siamese")
    model = build_model(model_type, cfg, pretrained_override=False).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt.get("calibration_temperature", 1.0), ckpt


def _batch_priority_and_weight(
    dprob: torch.Tensor,
    target_damage: torch.Tensor,
    severe_classes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    device = dprob.device
    severe_tensor = torch.tensor(severe_classes, device=device)
    predicted_severe = torch.isin(dprob.argmax(1), severe_tensor)
    severe_conf = dprob[:, list(severe_classes)].sum(1)
    flat_mask = predicted_severe.flatten(1)
    flat_conf = severe_conf.flatten(1)
    area = flat_mask.float().mean(1)
    mean_conf = torch.where(
        flat_mask.any(1),
        (flat_conf * flat_mask.float()).sum(1) / flat_mask.float().sum(1).clamp_min(1.0),
        torch.zeros_like(area),
    )
    priority = (area * mean_conf).detach().cpu().numpy()
    severe_truth = torch.isin(target_damage, severe_tensor)
    severe_weight = severe_truth.flatten(1).sum(1).detach().cpu().numpy().astype(np.float64)
    return priority, severe_weight


def run_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device, cfg: dict, temperature: float) -> dict:
    num_classes = int(cfg["model"]["num_damage_classes"])
    b_inter = b_union = b_pred_pos = b_target_pos = 0
    damage_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    priorities: list[float] = []
    severe_weights: list[float] = []
    cal_logits: list[torch.Tensor] = []
    cal_labels: list[torch.Tensor] = []
    cal_budget = int(cfg["calibration"].get("max_pixels", 100_000))
    collected = 0
    severe_classes = tuple(cfg["triage"]["severe_classes"])

    with torch.no_grad():
        for batch in loader:
            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            out = model(pre, post)
            bpred = out["building_logits"].argmax(1).cpu().numpy()
            dpred = out["damage_logits"].argmax(1).cpu().numpy()
            btgt = batch["building"].numpy()
            dtgt = batch["damage"].numpy()
            inter, union, ppos, tpos = update_binary_counts(bpred, btgt)
            b_inter += inter
            b_union += union
            b_pred_pos += ppos
            b_target_pos += tpos
            damage_cm = update_damage_confusion(damage_cm, dpred, dtgt, num_classes)

            calibrated = out["damage_logits"] / max(float(temperature), 1e-3)
            dprob = calibrated.softmax(1)
            batch_priority, batch_severe_weight = _batch_priority_and_weight(dprob, batch["damage"], severe_classes)
            priorities.extend(batch_priority.tolist())
            severe_weights.extend(batch_severe_weight.tolist())

            if collected < cal_budget:
                flat_logits = out["damage_logits"].detach().cpu().permute(0, 2, 3, 1).reshape(-1, num_classes)
                flat_labels = batch["damage"].detach().cpu().reshape(-1)
                valid_idx = torch.nonzero(flat_labels != IGNORE_INDEX, as_tuple=False).flatten()
                take = min(cal_budget - collected, int(valid_idx.numel()))
                if take:
                    if valid_idx.numel() > take:
                        positions = torch.linspace(0, valid_idx.numel() - 1, take).long()
                        valid_idx = valid_idx[positions]
                    cal_logits.append(flat_logits[valid_idx])
                    cal_labels.append(flat_labels[valid_idx])
                    collected += take

    result: dict = {}
    result.update(binary_scores_from_counts(b_inter, b_union, b_pred_pos, b_target_pos))
    result.update(damage_scores_from_confusion_matrix(damage_cm))
    if cal_logits:
        logits = torch.cat(cal_logits, dim=0)[:cal_budget]
        labels = torch.cat(cal_labels, dim=0)[:cal_budget]
        calibrated = logits / max(float(temperature), 1e-3)
        probs = calibrated.softmax(1)
        result["ece"] = expected_calibration_error(probs, labels)
        result["calibration_sample_pixels"] = int(labels.numel())
    else:
        result["ece"] = float("nan")
        result["calibration_sample_pixels"] = 0
    result["temperature"] = float(temperature)
    result.update(
        triage_recall_from_weights(
            np.asarray(priorities),
            np.asarray(severe_weights),
            cfg["triage"]["top_fractions"],
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="reports/metrics.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["inference"]["device"] != "cpu" else "cpu")
    ds = XBDPairDataset(
        args.manifest,
        patch_size=cfg["image"]["patch_size"],
        stride=cfg["image"]["stride"],
        training=False,
        normalize=cfg["image"]["normalize"],
        unknown_damage_policy=cfg.get("data", {}).get("unknown_damage_policy", "error"),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    model, temperature, _ = load_model(args.checkpoint, cfg, device)
    metrics = run_eval(model, loader, device, cfg, temperature)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import XBDPairDataset
from src.evaluation.calibration import fit_temperature
from src.evaluation.metrics import binary_scores_from_counts, damage_scores_from_confusion_matrix, update_binary_counts, update_damage_confusion
from src.models.factory import build_model
from src.training.losses import ReliefLensLoss
from src.utils.config import load_config
from src.utils.seed import seed_everything


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True, grad_clip_norm: float = 1.0):
    model.train(train)
    totals = {"total": 0.0, "building_ce": 0.0, "building_dice": 0.0, "damage_focal": 0.0}
    n = 0
    for batch in tqdm(loader, leave=False):
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        targets = {"building": batch["building"].to(device), "damage": batch["damage"].to(device)}
        with torch.set_grad_enabled(train):
            amp_enabled = scaler is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(pre, post)
                losses = criterion(outputs, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer); scaler.update()
                else:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
        bs = pre.shape[0]
        n += bs
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * bs
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def collect_validation(model, loader, device, max_calibration_pixels: int = 100_000):
    b_inter = b_union = b_pred_pos = b_target_pos = 0
    damage_cm = np.zeros((5, 5), dtype=np.int64)
    cal_logits, cal_labels = [], []
    collected = 0
    for batch in loader:
        out = model(batch["pre"].to(device), batch["post"].to(device))
        bpred = out["building_logits"].argmax(1).cpu().numpy()
        btgt = batch["building"].numpy()
        dpred = out["damage_logits"].argmax(1).cpu().numpy()
        dtgt = batch["damage"].numpy()
        inter, union, ppos, tpos = update_binary_counts(bpred, btgt)
        b_inter += inter; b_union += union; b_pred_pos += ppos; b_target_pos += tpos
        damage_cm = update_damage_confusion(damage_cm, dpred, dtgt, 5)
        if collected < max_calibration_pixels:
            flat_logits = out["damage_logits"].cpu().permute(0, 2, 3, 1).reshape(-1, out["damage_logits"].shape[1])
            flat_labels = batch["damage"].cpu().reshape(-1)
            take = min(max_calibration_pixels - collected, flat_labels.numel())
            idx = torch.linspace(0, flat_labels.numel() - 1, take).long() if take > 1 else torch.zeros(1, dtype=torch.long)
            cal_logits.append(flat_logits[idx]); cal_labels.append(flat_labels[idx]); collected += take
    result = {}
    result.update(binary_scores_from_counts(b_inter, b_union, b_pred_pos, b_target_pos))
    result.update(damage_scores_from_confusion_matrix(damage_cm))
    return result | {"logits": torch.cat(cal_logits) if cal_logits else torch.empty(0, 5), "labels": torch.cat(cal_labels) if cal_labels else torch.empty(0, dtype=torch.long)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--model", choices=["siamese", "baseline"], default="siamese")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = XBDPairDataset(
        args.train_manifest,
        patch_size=cfg["image"]["patch_size"],
        stride=cfg["image"]["stride"],
        training=True,
        normalize=cfg["image"]["normalize"],
        random_crop=bool(cfg["training"].get("random_crop", True)),
        random_crops_per_image=int(cfg["training"].get("random_crops_per_image", 4)),
        unknown_damage_policy=cfg.get("data", {}).get("unknown_damage_policy", "error"),
        paired_rotate90=bool(cfg["training"].get("paired_rotate90", True)),
    )
    val_ds = XBDPairDataset(
        args.val_manifest,
        patch_size=cfg["image"]["patch_size"],
        stride=cfg["image"]["stride"],
        training=False,
        normalize=cfg["image"]["normalize"],
        unknown_damage_policy=cfg.get("data", {}).get("unknown_damage_policy", "error"),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], pin_memory=device.type=="cuda")
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])

    model = build_model(args.model, cfg).to(device)
    criterion = ReliefLensLoss(cfg["training"]["damage_class_weights"], cfg["training"]["focal_gamma"], cfg["training"]["building_loss_weight"], cfg["training"]["damage_loss_weight"]).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda")
    best = -float("inf")
    bad_epochs = 0
    history = []
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        grad_clip = float(cfg["training"].get("grad_clip_norm", 1.0))
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True, grad_clip_norm=grad_clip)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, scaler=None, train=False, grad_clip_norm=grad_clip)
        pack = collect_validation(model, val_loader, device, max_calibration_pixels=int(cfg["calibration"].get("max_pixels", 100_000)))
        metrics = {k: v for k, v in pack.items() if k in {"building_iou", "building_f1", "damage_macro_f1", "damage_f1_no_damage", "damage_f1_minor", "damage_f1_major", "damage_f1_destroyed"}}
        # Monitor damage macro-F1, because the project is a damage-triage system.
        score = metrics["damage_macro_f1"]
        row = {"epoch": epoch + 1, "train": train_loss, "val": val_loss, "metrics": metrics}
        history.append(row)
        print(json.dumps(row, indent=2))
        torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "config": cfg, "metrics": metrics, "model_type": args.model}, out_dir / "last.pt")
        if score > best:
            best = score; bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "config": cfg, "metrics": metrics, "model_type": args.model}, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["training"]["early_stopping_patience"]:
                break
    # Fit calibration strictly on validation logits after checkpoint selection. Never use test data here.
    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    pack = collect_validation(model, val_loader, device, max_calibration_pixels=int(cfg["calibration"].get("max_pixels", 100_000)))
    cal = fit_temperature(pack["logits"], pack["labels"], **cfg["calibration"])
    temperature = float(cal.temperature.item())
    best_ckpt["calibration_temperature"] = temperature
    torch.save(best_ckpt, out_dir / "best.pt")
    (out_dir / "calibration.json").write_text(json.dumps({"temperature": temperature, "source": "validation"}, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best validation damage macro-F1: {best:.4f}; validation temperature: {temperature:.4f}")


if __name__ == "__main__":
    main()

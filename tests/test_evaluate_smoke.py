from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch


def _write_manifest(src_root: Path, path: Path) -> None:
    rows = []
    from src.data.prepare import discover

    rows.extend(discover(src_root / "tests" / "fixtures" / "xbd" / "disaster-d"))
    assert rows, "Synthetic xBD fixture must contain at least one pair"
    fields = ["sample_id", "disaster", "disaster_type", "pre_image", "post_image", "pre_label", "post_label"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_cli_evaluation_after_synthetic_checkpoint(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "smoke.yaml"
    manifest = tmp_path / "test.csv"
    _write_manifest(root, manifest)

    from src.models.factory import build_model
    from src.utils.config import load_config

    cfg = load_config(cfg_path)
    model = build_model("siamese", cfg, pretrained_override=False)
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "model_type": "siamese",
            "calibration_temperature": 1.0,
            "epoch": 0,
            "metrics": {"damage_macro_f1": 0.0},
        },
        ckpt,
    )

    out = tmp_path / "pytest_metrics.json"
    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.evaluate",
        "--config",
        str(cfg_path),
        "--checkpoint",
        str(ckpt),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
    ]
    result = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    assert result.returncode == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "damage_macro_f1" in data
    assert "ece" in data
    assert "temperature" in data

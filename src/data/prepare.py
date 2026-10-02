from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath

from src.data.xbd import disaster_from_label, image_size


def _files(root: Path, sub: str, suffix: str) -> list[Path]:
    candidates = []
    for p in root.rglob(f"*{suffix}"):
        if sub in p.parts:
            candidates.append(p)
    return sorted(candidates)


def _sample_key(root: Path, path: Path, bucket: str) -> str:
    """Return a collision-safe key shared by corresponding images/labels."""
    rel = path.relative_to(root)
    parts = list(rel.parts)
    try:
        idx = parts.index(bucket)
    except ValueError as exc:
        raise ValueError(f"Path {path} is not under expected {bucket!r} directory") from exc
    normalized = parts[:idx] + parts[idx + 1 :]
    if not normalized:
        raise ValueError(f"Could not derive sample key from {path}")
    return PurePosixPath(*normalized).as_posix()


def _sample_id_from_key(key: str) -> tuple[str, str]:
    path = PurePosixPath(key)
    filename = path.name
    stem = PurePosixPath(filename).stem
    if "_pre_" not in stem and "_post_" not in stem:
        raise ValueError(f"xBD filename has no _pre_/_post_ marker: {key}")
    phase = "pre" if "_pre_" in stem else "post"
    sample_filename = stem.replace("_pre_", "_").replace("_post_", "_")
    sample_key = PurePosixPath(path.parent, sample_filename).as_posix()
    return sample_key, phase


def _index_unique(root: Path, sub: str, suffix: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _files(root, sub, suffix):
        key = _sample_key(root, path, sub)
        sample_key, phase = _sample_id_from_key(key)
        pair_key = f"{sample_key}::{phase}"
        if pair_key in index:
            raise ValueError(
                f"Duplicate xBD {sub[:-1]} entry for {sample_key!r} ({phase}):\n"
                f"  {index[pair_key]}\n  {path}"
            )
        index[pair_key] = path
    return index


def discover(root: str | Path) -> list[dict]:
    root = Path(root)
    images = _index_unique(root, "images", ".png")
    labels = _index_unique(root, "labels", ".json")
    samples: dict[str, dict] = {}

    all_keys = set(images) | set(labels)
    for pair_key in sorted(all_keys):
        sample_key, phase = pair_key.rsplit("::", 1)
        bucket = samples.setdefault(sample_key, {})
        bucket[f"{phase}_image"] = str(images[pair_key]) if pair_key in images else None
        bucket[f"{phase}_label"] = str(labels[pair_key]) if pair_key in labels else None

    rows = []
    for sample_id, b in sorted(samples.items()):
        if not (b.get("pre_image") and b.get("post_image") and b.get("post_label")):
            continue
        if image_size(b["pre_image"]) != image_size(b["post_image"]):
            raise ValueError(
                f"Mismatched pre/post image dimensions for {sample_id}: "
                f"{b['pre_image']} vs {b['post_image']}"
            )
        disaster, disaster_type = disaster_from_label(b["post_label"])
        rows.append(
            {
                "sample_id": sample_id,
                "disaster": disaster,
                "disaster_type": disaster_type,
                "pre_image": b["pre_image"],
                "post_image": b["post_image"],
                "pre_label": b.get("pre_label") or "",
                "post_label": b["post_label"],
            }
        )
    return rows


def split_rows(rows: list[dict], split_cfg: dict) -> dict[str, list[dict]]:
    mapping = split_cfg["disasters"]
    owner: dict[str, str] = {}
    for split, disasters in mapping.items():
        for disaster in disasters:
            if disaster in owner:
                raise ValueError(f"Disaster {disaster!r} is assigned to both {owner[disaster]} and {split}")
            owner[disaster] = split

    used: set[str] = set()
    result: dict[str, list[dict]] = {}
    for split, disasters in mapping.items():
        out = []
        for row in rows:
            if row["disaster"] in disasters:
                out.append(row)
                used.add(row["sample_id"])
        result[split] = out
    if split_cfg.get("strict_event_split", True):
        leftovers = [r for r in rows if r["sample_id"] not in used]
        if leftovers:
            raise ValueError(
                f"{len(leftovers)} samples belong to disasters not assigned to train/val/test. "
                "Add those disaster names explicitly to configs/default.yaml."
            )
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "disaster", "disaster_type", "pre_image", "post_image", "pre_label", "post_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create event-disjoint xBD manifests")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rows = discover(args.raw_root)
    splits = split_rows(rows, cfg["split"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, split_rows_ in splits.items():
        write_csv(split_rows_, out / f"{split}.csv")
    (out / "summary.json").write_text(
        json.dumps({k: len(v) for k, v in splits.items()}, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: len(v) for k, v in splits.items()}, indent=2))


if __name__ == "__main__":
    main()

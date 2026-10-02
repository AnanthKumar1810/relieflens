from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_pair(out_dir: str | Path, size: int = 512) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    base = np.clip(rng.normal(110, 28, (size, size, 3)), 0, 255).astype(np.uint8)
    pre = Image.fromarray(base)
    post = Image.fromarray(base.copy())
    pre_d = ImageDraw.Draw(pre); post_d = ImageDraw.Draw(post)
    for box in [(70,80,150,150),(240,100,340,200),(90,280,180,370),(300,300,430,410)]:
        pre_d.rectangle(box, outline=(210,210,210), width=3, fill=(150,150,150))
        post_d.rectangle(box, outline=(210,210,210), width=3, fill=(150,150,150))
    post_d.rectangle((240,100,340,200), fill=(55,55,55), outline=(220,50,50), width=5)
    post_d.rectangle((300,300,430,410), fill=(85,50,50), outline=(240,120,40), width=5)
    pre.save(out/"demo_pre.png"); post.save(out/"demo_post.png")
    print(f"Wrote {out/'demo_pre.png'} and {out/'demo_post.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/demo"); args = ap.parse_args(); make_pair(args.out)

if __name__ == "__main__":
    main()

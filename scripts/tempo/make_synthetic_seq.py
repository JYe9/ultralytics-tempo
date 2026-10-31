# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
r"""Generate a tiny synthetic video dataset of uniformly moving colored rectangles in YOLO format.

Each sequence is a set of frames named `<seq>_<idx>.png` (matching seq_pattern '^(?P<seq>.+)_(?P<idx>\d+)\.png$'),
with one YOLO label file per frame. Frames get a per-sequence noisy background so that consecutive frames differ.

Usage:
    python scripts/tempo/make_synthetic_seq.py --out datasets/synth_seq
    yolo train model=yolo26n.yaml data=datasets/synth_seq/data.yaml imgsz=128 epochs=1 device=cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils import YAML

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR colors, class id = color index
NAMES = ("red", "green", "blue")
SEQ_PATTERN = r"^(?P<seq>.+)_(?P<idx>\d+)\.png$"


def make_dataset(
    out: str | Path,
    num_seqs: int = 3,
    num_frames: int = 12,
    size: int = 128,
    num_objects: int = 2,
    temporal: dict | None = None,
    seed: int = 0,
) -> Path:
    """Write sequences, labels and a data.yaml; every sequence is used for both train and val.

    Args:
        out (str | Path): Output dataset directory.
        num_seqs (int): Number of sequences.
        num_frames (int): Frames per sequence.
        size (int): Square frame size in pixels.
        num_objects (int): Rectangles per sequence, each moving with constant velocity.
        temporal (dict, optional): `temporal` section written into data.yaml, defaults to num_frames=3, stride=1.
        seed (int): Random seed.

    Returns:
        (Path): Path of the written data.yaml.
    """
    out = Path(out)
    rng = np.random.default_rng(seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for s in range(num_seqs):
        background = rng.integers(40, 90, (size, size, 3), dtype=np.uint8)
        objects = []
        for _ in range(num_objects):
            w, h = rng.integers(size // 8, size // 4, 2)
            x, y = rng.uniform(0, size - w), rng.uniform(0, size - h)
            vx, vy = rng.uniform(-3, 3, 2)
            objects.append([x, y, w, h, vx, vy, int(rng.integers(len(COLORS)))])
        for f in range(num_frames):
            img = background.copy()
            lines = []
            for o in objects:
                x, y, w, h, vx, vy, c = o
                x, y = float(np.clip(x + vx * f, 0, size - w)), float(np.clip(y + vy * f, 0, size - h))
                cv2.rectangle(img, (round(x), round(y)), (round(x + w) - 1, round(y + h) - 1), COLORS[c], -1)
                lines.append(f"{c} {(x + w / 2) / size:.6f} {(y + h / 2) / size:.6f} {w / size:.6f} {h / size:.6f}")
            name = f"seq{s}_{f:03d}"
            cv2.imwrite(str(out / "images" / f"{name}.png"), img)
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    temporal = {"num_frames": 3, "stride": 1, "seq_pattern": SEQ_PATTERN, **(temporal or {})}
    data = {"path": out.resolve().as_posix(), "train": "images", "val": "images", "names": dict(enumerate(NAMES))}
    yaml = out / "data.yaml"
    YAML.save(yaml, {**data, "temporal": temporal})
    return yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="datasets/synth_seq", help="output dataset directory")
    parser.add_argument("--seqs", type=int, default=3, help="number of sequences")
    parser.add_argument("--frames", type=int, default=12, help="frames per sequence")
    parser.add_argument("--size", type=int, default=128, help="frame size in pixels")
    parser.add_argument("--num-frames", type=int, default=3, help="temporal.num_frames written to data.yaml")
    parser.add_argument("--stride", type=int, default=1, help="temporal.stride written to data.yaml")
    parser.add_argument("--diff", action="store_true", help="temporal.diff_channel written to data.yaml")
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    cfg = {"num_frames": a.num_frames, "stride": a.stride, "diff_channel": a.diff}
    print(make_dataset(a.out, a.seqs, a.frames, a.size, temporal=cfg, seed=a.seed))

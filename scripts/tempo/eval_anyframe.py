# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate one temporal model at every (n, stride) combination and write a CSV for comparison with fixed-n models.

The model's metadata defines max_frames K, the training sets and cond; by default n runs over 1..K and stride over the
training strides. Columns match results/matrix.csv where they overlap (mAP50, mAP50-95, per-class AP, latency), so a
stage-2a any-frame model can be compared row by row with stage-1 fixed-frame models. Exported FiLM models only run at
their baked (n, stride); other combinations are recorded with an error.

Usage:
    python scripts/tempo/eval_anyframe.py --model runs/detect/train/weights/best.pt --data data/visdrone_vid_tempo.yaml \
        --imgsz 640 --batch 16 --device 0
    python scripts/tempo/eval_anyframe.py --model best.pt --data data.yaml --frames 1 3 5 --strides 1 2 --out any.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from run_matrix import latency_ms  # sibling script

from ultralytics import YOLO
from ultralytics.data.temporal import check_temporal_cfg
from ultralytics.models.yolo.detect.temporal import TemporalValidator, find_film, model_temporal_cfg
from ultralytics.utils.torch_utils import select_device

FIELDS = [
    "time",
    "model",
    "dataset",
    "max_frames",
    "cond",
    "slot_fill",
    "diff_channel",
    "num_frames",
    "stride",
    "trained",
    "mAP50",
    "mAP50-95",
    "latency_ms",
    "val_preprocess_ms",
    "val_inference_ms",
    "AP50_per_class",
    "AP50-95_per_class",
    "error",
]


def evaluate(model: str, data: str, n: int, s: int, cfg: dict, a: argparse.Namespace) -> dict:
    """Validate `model` at (n, s) and return one CSV row."""
    row = {"num_frames": n, "stride": s, "trained": n in cfg["train_frames"] and s in cfg["train_strides"]}
    m = YOLO(model, task="detect").val(
        data=data, imgsz=a.imgsz, batch=a.batch, device=a.device, validator=TemporalValidator.at(n, s), plots=False
    )
    names = m.names
    row.update(
        {
            "mAP50": round(float(m.box.map50), 5),
            "mAP50-95": round(float(m.box.map), 5),
            "AP50_per_class": json.dumps(
                {names[c]: round(float(m.box.ap50[i]), 5) for i, c in enumerate(m.ap_class_index)}
            ),
            "AP50-95_per_class": json.dumps(
                {names[c]: round(float(m.box.ap[i]), 5) for i, c in enumerate(m.ap_class_index)}
            ),
            "val_preprocess_ms": round(float(m.speed["preprocess"]), 3),
            "val_inference_ms": round(float(m.speed["inference"]), 3),
        }
    )
    if str(model).endswith(".pt"):  # forward latency of the PyTorch model at this (n, s); identical input size for all
        net = deepcopy(YOLO(model).model).fuse()
        if film := find_film(net):
            film.default = (n, s)
        row["latency_ms"] = round(
            latency_ms(net, 3 * cfg["max_frames"] + cfg["diff_channel"], a.imgsz, select_device(a.device), a.iters), 3
        )
    return row


def main(a: argparse.Namespace) -> Path:
    """Evaluate all requested (n, stride) combinations and write the CSV."""
    cfg = model_temporal_cfg(a.model)
    if not cfg:
        raise ValueError(f"{a.model} has no temporal config in its metadata")
    cfg = check_temporal_cfg({"temporal": dict(cfg)})
    frames = a.frames or list(range(1, cfg["max_frames"] + 1))
    strides = a.strides or cfg["train_strides"]
    base = {"model": str(a.model), "dataset": Path(a.data).stem}
    base.update({k: cfg[k] for k in ("max_frames", "cond", "slot_fill", "diff_channel")})
    rows = []
    for n in frames:
        for s in strides:
            try:
                row = evaluate(a.model, a.data, n, s, cfg, a)
            except Exception as e:  # noqa: BLE001 - e.g. an exported FiLM model asked for another (n, s)
                print(f"(n={n}, stride={s}) failed: {e!r}")
                row = {"num_frames": n, "stride": s, "error": repr(e)}
            rows.append({**base, **row, "time": datetime.now().isoformat(timespec="seconds")})
    out = Path(a.out or f"results/anyframe_{Path(a.model).stem}_{Path(a.data).stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="trained temporal model (*.pt or exported)")
    parser.add_argument("--data", required=True, help="data YAML with the validation images")
    parser.add_argument("--frames", type=int, nargs="+", default=None, help="n values (default 1..max_frames)")
    parser.add_argument("--strides", type=int, nargs="+", default=None, help="strides (default: training strides)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="e.g. 0 or cpu")
    parser.add_argument("--iters", type=int, default=100, help="timed iterations for latency")
    parser.add_argument("--out", default=None, help="CSV path (default results/anyframe_<model>_<data>.csv)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())

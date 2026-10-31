# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train and validate the multi-frame early-fusion experiment matrix, appending one CSV row per experiment.

Each experiment writes a copy of the base data YAML with its own `temporal:` section, trains (3-channel *.pt weights
are inflated automatically), validates best.pt, and records mAP50, mAP50-95, per-class AP, parameters, GFLOPs and
latency to results/matrix.csv. Rows that already exist for the same settings are skipped, so reruns resume.

Usage:
    python scripts/tempo/run_matrix.py --data data/stvd_tempo.yaml --imgsz 640 --epochs 100 --batch 16 --device 0
    python scripts/tempo/run_matrix.py --data data/stvd_tempo.yaml --models yolo26n --configs n1 n3s1 --epochs 1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import get_flops, get_num_params, select_device

CONFIGS = {  # name: (num_frames n, stride s, diff_channel)
    "n1": (1, 1, False),
    "n2s1": (2, 1, False),
    "n2s1d": (2, 1, True),
    "n3s1": (3, 1, False),
    "n3s2": (3, 2, False),
    "n3s3": (3, 3, False),
    "n5s2": (5, 2, False),
}
FIELDS = [
    "time",
    "model",
    "config",
    "num_frames",
    "stride",
    "diff_channel",
    "channels",
    "imgsz",
    "epochs",
    "batch",
    "seed",
    "mAP50",
    "mAP50-95",
    "params",
    "GFLOPs",
    "latency_ms",
    "val_inference_ms",
    "AP50_per_class",
    "AP50-95_per_class",
    "weights",
    "error",
]
KEY = ("model", "config", "imgsz", "epochs", "batch", "seed")


@torch.no_grad()
def latency_ms(model: torch.nn.Module, channels: int, imgsz: int, device: torch.device, iters: int) -> float:
    """Return the median batch-1 forward latency in milliseconds on `device`."""
    model = model.to(device).eval()
    x = torch.zeros(1, channels, imgsz, imgsz, device=device)
    times = []
    for i in range(iters + 10):  # 10 warmup iterations
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if i >= 10:
            times.append((time.perf_counter() - t) * 1000)
    return sorted(times)[len(times) // 2]


def run(model_name: str, config: str, a: argparse.Namespace, base: dict) -> dict:
    """Train and validate one experiment and return its CSV row."""
    n, s, diff = CONFIGS[config]
    name = f"{model_name}_{config}_e{a.epochs}_sz{a.imgsz}_seed{a.seed}"
    data_yaml = Path(a.project) / "data" / f"{name}.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(
        data_yaml,
        {**base, "temporal": {**base.get("temporal", {}), "num_frames": n, "stride": s, "diff_channel": diff}},
    )
    row = {"model": model_name, "config": config, "num_frames": n, "stride": s, "diff_channel": diff}
    row.update(channels=3 * n + diff, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, seed=a.seed)

    weights = f"{model_name}.yaml" if a.scratch else f"{model_name}.pt"
    train_args = {
        "data": str(data_yaml),
        "imgsz": a.imgsz,
        "epochs": a.epochs,
        "batch": a.batch,
        "device": a.device,
        "seed": a.seed,
    }
    trained = YOLO(weights)
    trained.train(**train_args, workers=a.workers, project=a.project, name=name, exist_ok=True, deterministic=True)
    best = trained.trainer.best
    model = YOLO(best)
    m = model.val(
        data=str(data_yaml), imgsz=a.imgsz, batch=a.batch, device=a.device, project=a.project, name=f"{name}_val"
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
            "val_inference_ms": round(float(m.speed["inference"]), 3),
            "params": get_num_params(model.model),
            "GFLOPs": round(get_flops(model.model, a.imgsz), 3),
            "latency_ms": round(
                latency_ms(model.model.fuse(), 3 * n + diff, a.imgsz, select_device(a.device), a.iters), 3
            ),
            "weights": str(best),
        }
    )
    return row


def main(a: argparse.Namespace) -> None:
    """Run every requested model x config combination and append results to the CSV."""
    base = YAML.load(a.data)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {tuple(r[k] for k in KEY) for r in csv.DictReader(f) if not r.get("error")}
    for model_name in a.models:
        for config in a.configs:
            key = (model_name, config, str(a.imgsz), str(a.epochs), str(a.batch), str(a.seed))
            if key in done:
                print(f"Skipping {key}, already in {out}")
                continue
            try:
                row = run(model_name, config, a, base)
            except Exception as e:  # noqa: BLE001 - keep the matrix running, record the failure
                print(f"Experiment {model_name} {config} failed: {e!r}")
                n, s, diff = CONFIGS[config]
                row = dict(zip(KEY, key), num_frames=n, stride=s, diff_channel=diff, error=repr(e))
            row["time"] = datetime.now().isoformat(timespec="seconds")
            new = not out.exists()
            with open(out, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)
            print(f"Appended {row.get('model')} {row.get('config')} -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="base data YAML, e.g. data/stvd_tempo.yaml")
    parser.add_argument("--models", nargs="+", default=["yolo26n", "yolo26s"], help="model names")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS), help="temporal configs")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", required=True, help="e.g. 0, 0,1 or cpu")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--scratch", action="store_true", help="train from <model>.yaml instead of pretrained <model>.pt"
    )
    parser.add_argument("--iters", type=int, default=100, help="timed iterations for latency")
    parser.add_argument("--project", default="runs/tempo", help="training output directory")
    parser.add_argument("--out", default="results/matrix.csv", help="CSV that results are appended to")
    main(parser.parse_args())

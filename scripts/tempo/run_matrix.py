# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train and validate the multi-frame early-fusion experiment matrix, appending one CSV row per experiment.

For every dataset x model x config, a copy of the dataset YAML with the config's `temporal:` section is written,
the model is trained (3-channel *.pt weights are inflated automatically) and best.pt is validated. mAP50, mAP50-95,
per-class AP, parameters, GFLOPs and latency are appended to results/matrix.csv together with the dataset name and the
fairness hyperparameters. Every config, n=1 included, trains with FAIR_HYP (no mosaic, mixup, cutmix, copy_paste,
close_mosaic); temporal datasets also disable Albumentations. Rows already in the CSV are skipped, so reruns resume,
and appends are file-locked so parallel SLURM array tasks can share one CSV.

Usage:
    python scripts/tempo/run_matrix.py --data data/stvd_tempo.yaml --imgsz 640 --epochs 100 --batch 16 --device 0 --seed 0
    python scripts/tempo/run_matrix.py --data data/stvd_tempo.yaml data/mot_tempo.yaml --models yolo26n --configs n1 n3s1 \
        --imgsz 640 --epochs 1 --batch 16 --device 0 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
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
FAIR_HYP = {"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "close_mosaic": 0}  # every config, n=1 too
KEY = ("dataset", "model", "config", "imgsz", "epochs", "batch", "seed")
FIELDS = [
    "time",
    *KEY,
    "num_frames",
    "stride",
    "diff_channel",
    "channels",
    *FAIR_HYP,
    "albumentations",
    "mAP50",
    "mAP50-95",
    "params",
    "GFLOPs",
    "latency_ms",
    "val_inference_ms",
    "AP50_per_class",
    "AP50-95_per_class",
    "data_yaml",
    "weights",
    "error",
]


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


def base_row(dataset: str, model_name: str, config: str, a: argparse.Namespace) -> dict:
    """Return the identifying and fairness columns of one experiment."""
    n, s, diff = CONFIGS[config]
    row = dict(zip(KEY, (dataset, model_name, config, a.imgsz, a.epochs, a.batch, a.seed)))
    row.update(num_frames=n, stride=s, diff_channel=diff, channels=3 * n + diff, albumentations="disabled")
    return {**row, **FAIR_HYP}


def run(data: str, model_name: str, config: str, a: argparse.Namespace) -> dict:
    """Train and validate one experiment and return its CSV row."""
    dataset = Path(data).stem
    n, s, diff = CONFIGS[config]
    row = base_row(dataset, model_name, config, a)
    name = f"{dataset}_{model_name}_{config}_e{a.epochs}_sz{a.imgsz}_seed{a.seed}"
    base = YAML.load(data)
    base["path"] = str(check_det_dataset(data)["path"])  # resolved like training would, the copy lives elsewhere
    data_yaml = Path(a.project).resolve() / "data" / f"{name}.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(
        data_yaml,
        {**base, "temporal": {**(base.get("temporal") or {}), "num_frames": n, "stride": s, "diff_channel": diff}},
    )
    row["data_yaml"] = str(data_yaml)

    weights = f"{model_name}.yaml" if a.scratch else f"{model_name}.pt"
    args = {"data": str(data_yaml), "imgsz": a.imgsz, "batch": a.batch, "device": a.device}
    trained = YOLO(weights)
    trained.train(
        **args,
        **FAIR_HYP,
        epochs=a.epochs,
        seed=a.seed,
        deterministic=True,
        workers=a.workers,
        project=a.project,
        name=name,
        exist_ok=True,
    )
    best = trained.trainer.best
    model = YOLO(best)
    m = model.val(**args, project=a.project, name=f"{name}_val", exist_ok=True)
    names = m.names
    per_class = {
        k: {names[c]: round(float(v[i]), 5) for i, c in enumerate(m.ap_class_index)}
        for k, v in (("ap50", m.box.ap50), ("ap", m.box.ap))
    }
    row.update(
        {
            "mAP50": round(float(m.box.map50), 5),
            "mAP50-95": round(float(m.box.map), 5),
            "AP50_per_class": json.dumps(per_class["ap50"]),
            "AP50-95_per_class": json.dumps(per_class["ap"]),
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


@contextmanager
def locked(path: Path):
    """Hold an exclusive lock on `path`.lock (POSIX), so concurrent jobs append whole rows."""
    with open(f"{path}.lock", "w") as lock:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def completed(out: Path) -> set:
    """Return the keys of successful experiments already in the CSV; raise if its columns are outdated."""
    if not out.exists():
        return set()
    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"{out} has different columns than this script writes; move it aside or pass --out")
        return {tuple(r[k] for k in KEY) for r in reader if not r["error"]}


def append(out: Path, row: dict) -> None:
    """Append one row to the CSV, writing the header for a new file."""
    with locked(out):
        new = not out.exists() or out.stat().st_size == 0
        with open(out, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)


def main(a: argparse.Namespace) -> None:
    """Run every requested dataset x model x config combination and append results to the CSV."""
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for data in a.data:
        for model_name in a.models:
            for config in a.configs:
                row = base_row(Path(data).stem, model_name, config, a)
                key = tuple(str(row[k]) for k in KEY)
                with locked(out):
                    done = key in completed(out)
                if done:
                    print(f"Skipping {key}, already in {out}")
                    continue
                try:
                    row = run(data, model_name, config, a)
                except Exception as e:  # noqa: BLE001 - keep the matrix running, record the failure
                    print(f"Experiment {key} failed: {e!r}")
                    row["error"] = repr(e)
                row["time"] = datetime.now().isoformat(timespec="seconds")
                append(out, row)
                print(f"Appended {key} -> {out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="one or more temporal data YAMLs")
    parser.add_argument("--models", nargs="+", default=["yolo26n", "yolo26s"], help="model names")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS), help="temporal configs")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", required=True, help="e.g. 0, 0,1,2,3 or cpu")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--scratch", action="store_true", help="train from <model>.yaml instead of pretrained <model>.pt"
    )
    parser.add_argument("--iters", type=int, default=100, help="timed iterations for latency")
    parser.add_argument("--project", default="runs/tempo", help="training output directory")
    parser.add_argument("--out", default="results/matrix.csv", help="CSV that results are appended to")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())

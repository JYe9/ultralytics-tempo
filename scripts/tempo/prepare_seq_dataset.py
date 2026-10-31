# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
r"""Convert video detection datasets into the temporal YOLO layout used by TemporalYOLODataset.

Adapters:
    stvd          STVD (arXiv:2410.13616): images with YOLO txt labels, sequences from --seq-pattern or directories
    visdrone-vid  VisDrone2019-VID: <split dir>/sequences/<seq>/*.jpg + annotations/<seq>.txt; drops ignored regions
                  (category 0), others (11) and boxes with score 0; --class-map remaps the other categories
    uavdt         UAVDT DET/MOT: UAV-benchmark-M/<M0101>/img*.jpg + GT/<M0101>_gt_whole.txt (1 car, 2 truck, 3 bus)
    mot           MOT17/MOT20: train/<seq>/{img1,gt/gt.txt,seqinfo.ini}; pedestrians (class 1) with the evaluation
                  flag set and visibility >= --min-visibility; MOT17 detector copies (DPM/FRCNN/SDP) are deduplicated

Output (written with --out-dir, default datasets/<adapter>_tempo):
    images/<split>/<sequence>/<frame:06d>.<ext>  (symlinks, or copies with --copy)
    labels/<split>/<sequence>/<frame:06d>.txt    (YOLO format, empty for frames without objects)
    <out-yaml> with train/val(/test), names and a temporal: section (one sequence per directory)

Splits: --split temporal (default, first 1-val_ratio of every sequence trains, the rest validates), --split official
(dataset split directories: VisDrone train/val/test-dev, STVD train/val/test directories), --split seqs --val-seqs ...

Step 1, inspect (prints sequences, frames per sequence, fps, camera motion, class instances; writes nothing):
    python scripts/tempo/prepare_seq_dataset.py stvd --root /data/STVD --inspect --seq-pattern '^(?P<seq>.+)_(?P<idx>\d+)\.jpg$'
    python scripts/tempo/prepare_seq_dataset.py visdrone-vid --root /data/VisDrone --inspect --estimate-motion
    python scripts/tempo/prepare_seq_dataset.py mot --root /data/MOT17 --inspect --min-visibility 0.25

Step 2, convert:
    python scripts/tempo/prepare_seq_dataset.py visdrone-vid --root /data/VisDrone --split official
    python scripts/tempo/prepare_seq_dataset.py uavdt --root /data/UAVDT --split seqs --val-seqs M0203 M0205
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ultralytics.data.temporal import check_temporal_cfg, group_sequences, sequence_key
from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import YAML

VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
UAVDT_NAMES = ["car", "truck", "bus"]
# Camera motion of MOT sequences as documented by MOTChallenge (MOT16/17 paper, MOT20 static elevated cameras)
MOT_MOVING = {"MOT17-05", "MOT17-06", "MOT17-07", "MOT17-10", "MOT17-11", "MOT17-12", "MOT17-13", "MOT17-14"}


@dataclass
class Sequence:
    """One video: sorted frames, per-frame YOLO boxes (cls, xc, yc, w, h normalized) and metadata."""

    name: str
    frames: list[Path]
    labels: list[list[tuple]] = field(default_factory=list)
    split: str | None = None  # official split if the dataset defines one
    fps: float | None = None
    motion: str = "unknown"


def images_in(d: Path) -> list[Path]:
    """Return image files directly in a directory."""
    return [p for p in d.iterdir() if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS]


def frame_number(p: Path) -> int:
    """Return the last integer in a file stem (frame number)."""
    return int(re.findall(r"\d+", p.stem)[-1])


def image_size(p: Path) -> tuple[int, int]:
    """Return (width, height) of an image without decoding it."""
    from PIL import Image

    with Image.open(p) as im:
        return im.size


def yolo_box(cls: int, left: float, top: float, w: float, h: float, iw: int, ih: int) -> tuple | None:
    """Clip a pixel ltwh box to the image and return a normalized (cls, xc, yc, w, h) tuple, or None if empty."""
    x1, y1, x2, y2 = max(left, 0), max(top, 0), min(left + w, iw), min(top + h, ih)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return cls, (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih, (x2 - x1) / iw, (y2 - y1) / ih


def class_map(opt: argparse.Namespace, default: dict[int, int]) -> dict[int, int]:
    """Return the source-category -> output-class map from --class-map JSON or the adapter default."""
    return {int(k): int(v) for k, v in json.loads(opt.class_map).items()} if opt.class_map else default


def box_sequence(name, frames, rows, cmap, split=None, fps=None, motion="unknown") -> Sequence:
    """Build a Sequence from per-row (frame number, category, left, top, w, h) pixel annotations."""
    iw, ih = image_size(frames[0])
    index = {frame_number(f): i for i, f in enumerate(frames)}
    labels = [[] for _ in frames]
    for fn, cat, left, top, w, h in rows:
        if cat in cmap and cmap[cat] >= 0 and fn in index and (b := yolo_box(cmap[cat], left, top, w, h, iw, ih)):
            labels[index[fn]].append(b)
    return Sequence(name, frames, labels, split, fps, motion)


def read_rows(file: Path, ncols: int) -> np.ndarray:
    """Read the first `ncols` columns of a comma-separated annotation file (tolerates trailing commas)."""
    rows = [[float(x) for x in line.split(",")[:ncols]] for line in file.read_text().splitlines() if line.strip()]
    return np.array(rows, dtype=float).reshape(-1, ncols)


def stvd(root: Path, opt: argparse.Namespace) -> tuple[list[Sequence], list[str]]:
    """STVD and other image datasets with YOLO txt labels; sequences from --seq-pattern or one per directory."""
    files = sorted(str(p) for p in root.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS and p.is_file())
    if opt.image_dir:
        files = [f for f in files if opt.image_dir in Path(f).parts]
    groups, unmatched = group_sequences(files, opt.seq_pattern)
    if unmatched:
        print(f"WARNING: {unmatched} images do not match the sequence rule and become 1-frame sequences")
    cmap = class_map(opt, {})
    sequences, max_cls = [], -1
    for g in groups:
        frames = [Path(files[i]) for i in g]
        labels = []
        for lb in img2label_paths([str(f) for f in frames]):
            boxes = []
            for line in Path(lb).read_text().splitlines() if Path(lb).is_file() else []:
                c, *xywh = line.split()[:5]
                c = cmap.get(int(float(c)), -1) if cmap else int(float(c))
                if c >= 0:
                    boxes.append((c, *map(float, xywh)))
                    max_cls = max(max_cls, c)
            labels.append(boxes)
        rel = frames[0].parent.relative_to(root)
        split = next(({"valid": "val"}.get(p, p) for p in rel.parts if p in {"train", "val", "valid", "test"}), None)
        key = sequence_key(str(frames[0]), opt.seq_pattern)
        name = "_".join([*rel.parts, key[0][1] if key else frames[0].stem]).strip("_") or "seq"
        sequences.append(Sequence(name, frames, labels, split))
    return sequences, opt.names or [f"class_{i}" for i in range(max_cls + 1)]


def visdrone_vid(root: Path, opt: argparse.Namespace) -> tuple[list[Sequence], list[str]]:
    """VisDrone2019-VID: split directories (*-train, *-val, *-test-dev) with sequences/ and annotations/."""
    cmap = class_map(opt, {i: i - 1 for i in range(1, 11)})
    cmap = {k: v for k, v in cmap.items() if k not in {0, 11}}  # ignored regions and 'others' are always dropped
    sequences = []
    for seq_root in sorted({p.parent for p in root.rglob("sequences") if p.is_dir()}):
        split = next((s for s in ("test", "val", "train") if s in seq_root.name.lower()), None)  # e.g. *-VID-val
        for d in sorted(p for p in (seq_root / "sequences").iterdir() if p.is_dir()):
            ann = seq_root / "annotations" / f"{d.name}.txt"
            if not ann.is_file():
                print(f"WARNING: no annotations for {d}, skipped")
                continue
            a = read_rows(ann, 8)  # frame, id, left, top, w, h, score, category, truncation, occlusion
            a = a[a[:, 6] > 0]  # score 0 marks boxes ignored in evaluation
            rows = [(int(r[0]), int(r[7]), *r[2:6]) for r in a]
            frames = sorted(images_in(d), key=frame_number)
            sequences.append(box_sequence(d.name, frames, rows, cmap, split=split))
    return sequences, opt.names or VISDRONE_NAMES


def uavdt(root: Path, opt: argparse.Namespace) -> tuple[list[Sequence], list[str]]:
    """UAVDT: M<4 digits> sequence directories of img*.jpg and <seq>_gt_whole.txt files anywhere below root."""
    cmap = class_map(opt, {1: 0, 2: 1, 3: 2})
    gts = {p.name[: -len("_gt_whole.txt")]: p for p in root.rglob("*_gt_whole.txt")}
    sequences = []
    for d in sorted(p for p in root.rglob("M[0-9][0-9][0-9][0-9]") if p.is_dir() and images_in(p)):
        if d.name not in gts:
            print(f"WARNING: no {d.name}_gt_whole.txt for {d}, skipped")
            continue
        a = read_rows(gts[d.name], 9)  # frame, id, left, top, w, h, out-of-view, occlusion, category
        rows = [(int(r[0]), int(r[8]), *r[2:6]) for r in a]
        sequences.append(box_sequence(d.name, sorted(images_in(d), key=frame_number), rows, cmap))
    return sequences, opt.names or UAVDT_NAMES


def mot(root: Path, opt: argparse.Namespace) -> tuple[list[Sequence], list[str]]:
    """MOT17/MOT20: <seq>/img1 frames with gt/gt.txt; keeps evaluated pedestrians with visibility >= --min-visibility."""
    cmap = class_map(opt, {1: 0})
    sequences, seen = [], set()
    for gt in sorted(root.rglob("gt/gt.txt")):
        d = gt.parent.parent
        base = re.sub(r"-(DPM|FRCNN|SDP)$", "", d.name)  # MOT17 ships identical frames/GT once per public detector
        if base in seen:
            continue
        seen.add(base)
        ini = configparser.ConfigParser()
        ini.read(d / "seqinfo.ini")
        fps = ini.getfloat("Sequence", "frameRate", fallback=None)
        im_dir = d / ini.get("Sequence", "imDir", fallback="img1")
        a = read_rows(gt, 9)  # frame, id, left, top, w, h, considered flag, class, visibility
        a = a[(a[:, 6] > 0) & (a[:, 8] >= opt.min_visibility)]
        rows = [(int(r[0]), int(r[7]), *r[2:6]) for r in a]
        known = base.startswith(("MOT16", "MOT17", "MOT20"))
        motion = "moving" if base in MOT_MOVING else "static" if known else "unknown"
        frames = sorted(images_in(im_dir), key=frame_number)
        sequences.append(box_sequence(base, frames, rows, cmap, fps=fps, motion=motion))
    return sequences, opt.names or ["pedestrian"]


ADAPTERS = {"stvd": stvd, "visdrone-vid": visdrone_vid, "uavdt": uavdt, "mot": mot}


def estimate_motion(frames: list[Path], pairs: int = 5, gap: int = 10, threshold: float = 0.01) -> str:
    """Estimate camera motion from the median global phase-correlation shift between frames `gap` apart."""
    import cv2

    if len(frames) < 2:
        return "unknown"
    gap = min(gap, len(frames) - 1)
    shifts = []
    for i in np.linspace(0, len(frames) - 1 - gap, min(pairs, len(frames) - gap)).astype(int):
        a, b = (cv2.imread(str(frames[j]), cv2.IMREAD_GRAYSCALE) for j in (i, i + gap))
        h = max(1, round(a.shape[0] * 256 / a.shape[1]))
        a, b = (np.float32(cv2.resize(x, (256, h))) for x in (a, b))
        (dx, dy), _ = cv2.phaseCorrelate(a, b)
        shifts.append(np.hypot(dx, dy) / 256)
    return f"{'moving' if np.median(shifts) > threshold else 'static'} (estimated)"


def assign_splits(sequences: list[Sequence], opt: argparse.Namespace) -> dict[str, list[tuple[Sequence, int, int]]]:
    """Return {split: [(sequence, first frame, end frame)]} for the requested split mode."""
    out = {}
    for s in sequences:
        if opt.split == "temporal":
            k = max(1, round(len(s.frames) * (1 - opt.val_ratio))) if len(s.frames) > 1 else 1
            parts = [("train", 0, k), ("val", k, len(s.frames))]
        elif opt.split == "seqs":
            parts = [("val" if s.name in opt.val_seqs else "train", 0, len(s.frames))]
        else:  # official
            if s.split is None:
                raise ValueError(f"{opt.adapter}: sequence {s.name} has no official split, use --split temporal/seqs")
            parts = [(s.split, 0, len(s.frames))]
        for split, a, b in parts:
            if b > a:
                out.setdefault(split, []).append((s, a, b))
    return out


def report(sequences: list[Sequence], names: list[str], splits: dict | None = None) -> None:
    """Print per-sequence and overall statistics."""
    print(f"\n{len(sequences)} sequences")
    print(f"  {'sequence':<28}{'split':<8}{'frames':>7}{'fps':>7}  {'camera':<20}instances")
    for s in sequences:
        fps = f"{s.fps:g}" if s.fps else "?"
        print(f"  {s.name:<28}{s.split or '-':<8}{len(s.frames):>7}{fps:>7}  {s.motion:<20}{sum(map(len, s.labels))}")
    lengths = sorted(len(s.frames) for s in sequences)
    if lengths:
        print(f"frames per sequence: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
    counts = Counter(int(b[0]) for s in sequences for lb in s.labels for b in lb)
    print("class instances: " + ", ".join(f"{names[c] if c < len(names) else c}={counts[c]}" for c in sorted(counts)))
    for split, parts in (splits or {}).items():
        print(f"  {split}: {len(parts)} sequences, {sum(b - a for _, a, b in parts)} frames")


def write(splits: dict, names: list[str], opt: argparse.Namespace) -> Path:
    """Write images (symlinks or copies), YOLO labels and the data YAML; return the YAML path."""
    out = Path(opt.out_dir or f"datasets/{opt.adapter}_tempo").resolve()
    for split, parts in splits.items():
        for s, a, b in parts:
            im_dir, lb_dir = out / "images" / split / s.name, out / "labels" / split / s.name
            im_dir.mkdir(parents=True, exist_ok=True)
            lb_dir.mkdir(parents=True, exist_ok=True)
            for i in range(a, b):
                src, dst = s.frames[i], im_dir / f"{i:06d}{s.frames[i].suffix.lower()}"
                if not dst.exists():
                    if opt.copy:
                        shutil.copy2(src, dst)
                    else:
                        os.symlink(src.resolve(), dst)
                lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in s.labels[i]]
                (lb_dir / f"{i:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    temporal = {"num_frames": opt.num_frames, "stride": opt.stride, "seq_pattern": None, "diff_channel": opt.diff}
    check_temporal_cfg({"temporal": dict(temporal)})
    data = {"path": str(out), **{k: f"images/{k}" for k in ("train", "val", "test") if k in splits}}
    data.update(names=dict(enumerate(names)), temporal=temporal)
    yaml = Path(opt.out_yaml or f"data/{opt.adapter.replace('-', '_')}_tempo.yaml")
    yaml.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(yaml, data)
    return yaml


def tree(root: Path, depth: int, max_entries: int = 8, prefix: str = "") -> None:
    """Print a depth-limited directory tree with file counts."""
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    files = [p for p in root.iterdir() if p.is_file()]
    exts = Counter(p.suffix.lower() for p in files)
    print(f"{prefix}{root.name}/  ({len(files)} files {dict(exts.most_common(4))}, {len(dirs)} dirs)")
    if depth > 0:
        for d in dirs[:max_entries]:
            tree(d, depth - 1, max_entries, prefix + "    ")
        if len(dirs) > max_entries:
            print(f"{prefix}    ... {len(dirs) - max_entries} more dirs")


def main(opt: argparse.Namespace) -> Path | None:
    """Inspect or convert a dataset with the selected adapter."""
    root = Path(opt.root).resolve()
    sequences, names = ADAPTERS[opt.adapter](root, opt)
    if not sequences:
        raise FileNotFoundError(f"{opt.adapter}: no sequences found under {root}")
    if opt.estimate_motion:
        for s in sequences:
            s.motion = estimate_motion(s.frames) if s.motion == "unknown" else s.motion
    if opt.inspect:
        tree(root, opt.depth)
        report(sequences, names)
        return None
    splits = assign_splits(sequences, opt)
    report(sequences, names, splits)
    yaml = write(splits, names, opt)
    print(f"Wrote {yaml}")
    return yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("adapter", choices=list(ADAPTERS), help="dataset format")
    parser.add_argument("--root", required=True, help="extracted dataset root directory")
    parser.add_argument("--inspect", action="store_true", help="print layout and sequence statistics only")
    parser.add_argument("--depth", type=int, default=2, help="directory tree depth printed by --inspect")
    parser.add_argument("--estimate-motion", action="store_true", help="estimate unknown camera motion from frames")
    parser.add_argument("--split", choices=["temporal", "official", "seqs"], default="temporal")
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="last fraction of each sequence for --split temporal"
    )
    parser.add_argument("--val-seqs", nargs="+", default=[], help="validation sequence names for --split seqs")
    parser.add_argument("--class-map", default=None, help='JSON source-category -> class id, e.g. \'{"1": 0, "4": 1}\'')
    parser.add_argument("--names", nargs="+", default=None, help="output class names in id order")
    parser.add_argument("--min-visibility", type=float, default=0.0, help="mot: minimum visibility ratio")
    parser.add_argument(
        "--seq-pattern", default=None, help="stvd: file name regex with (?P<idx>..), optional (?P<seq>..)"
    )
    parser.add_argument("--image-dir", default=None, help="stvd: only use images below directories with this name")
    parser.add_argument(
        "--out-dir", default=None, help="converted dataset directory (default datasets/<adapter>_tempo)"
    )
    parser.add_argument("--out-yaml", default=None, help="data YAML (default data/<adapter>_tempo.yaml)")
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    parser.add_argument("--num-frames", type=int, default=3, help="default temporal.num_frames in the YAML")
    parser.add_argument("--stride", type=int, default=1, help="default temporal.stride in the YAML")
    parser.add_argument("--diff", action="store_true", help="default temporal.diff_channel in the YAML")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
r"""Prepare an extracted STVD dataset (arXiv:2410.13616) for multi-frame early-fusion training.

Frames are grouped into sequences with the same rule TemporalYOLODataset uses (`--seq-pattern` regex on the file name,
or one sequence per directory), each sequence is split in time (first 80% train, last 20% val by default), and image
lists plus a data YAML with a `temporal:` section are written. Labels must already be YOLO txt files that
Ultralytics finds by replacing the last `/images/` path component with `/labels/` (see `img2label_paths`).

Step 1, inspect the real layout first (prints only, writes nothing):
    python scripts/tempo/prepare_stvd.py --root /data/STVD --inspect
    python scripts/tempo/prepare_stvd.py --root /data/STVD --inspect --seq-pattern '^(?P<seq>.+)_(?P<idx>\d+)\.jpg$'

Step 2, write the lists and data/stvd_tempo.yaml:
    python scripts/tempo/prepare_stvd.py --root /data/STVD --seq-pattern '...' --names drone bird
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from ultralytics.data.temporal import check_temporal_cfg, group_sequences
from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import YAML


def find_images(root: Path, image_dir: str | None) -> list[str]:
    """Return sorted image paths under root, optionally restricted to directories named `image_dir`."""
    files = sorted(str(p) for p in root.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS and p.is_file())
    return [f for f in files if image_dir in Path(f).parts] if image_dir else files


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


def inspect(root: Path, files: list[str], seq_pattern: str | None, depth: int) -> None:
    """Print layout, file name samples, label coverage and inferred sequence statistics."""
    tree(root, depth)
    print(f"\n{len(files)} images found. Samples:")
    rng = random.Random(0)
    for f in sorted(rng.sample(files, min(10, len(files)))):
        print(f"  {Path(f).relative_to(root)}")
    others = Counter(
        p.suffix.lower() for p in root.rglob("*") if p.is_file() and p.suffix[1:].lower() not in IMG_FORMATS
    )
    print(f"\nNon-image files by extension: {dict(others.most_common(10))}")
    labels = img2label_paths(files)
    found = [lb for lb in labels if Path(lb).is_file()]
    print(f"YOLO labels found for {len(found)}/{len(files)} images (expected at e.g. {labels[0] if labels else '-'})")
    if found:
        lines = Path(found[0]).read_text().splitlines()[:3]
        print(f"  first label file {found[0]}:\n    " + "\n    ".join(lines or ["<empty>"]))
    report_sequences(files, seq_pattern)


def report_sequences(files: list[str], seq_pattern: str | None) -> list[list[int]]:
    """Group files into sequences and print statistics."""
    sequences, unmatched = group_sequences(files, seq_pattern)
    lengths = sorted(len(s) for s in sequences)
    rule = f"seq_pattern={seq_pattern!r}" if seq_pattern else "one sequence per directory (last integer = frame index)"
    print(f"\nSequence rule: {rule}")
    print(f"  {len(sequences)} sequences, {unmatched} unmatched images (each becomes a 1-frame sequence)")
    if lengths:
        print(f"  frames per sequence: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
    for s in [s for s in sequences if len(s) > 1][:3]:
        print(f"  e.g. {len(s)} frames: {Path(files[s[0]]).name} ... {Path(files[s[-1]]).name}")
    return sequences


def max_class_id(label_files: list[str]) -> int:
    """Return the largest class id in YOLO label files, or -1 if there are none."""
    lines = (line for f in label_files if Path(f).is_file() for line in Path(f).read_text().splitlines())
    ids = [int(float(line.split()[0])) for line in lines if line.strip()]
    return max(ids, default=-1)


def main(a: argparse.Namespace) -> None:
    """Inspect or write the train/val lists and data YAML."""
    root = Path(a.root).resolve()
    files = find_images(root, a.image_dir)
    if not files:
        raise FileNotFoundError(f"No images found under {root} (image_dir={a.image_dir})")
    if a.inspect:
        return inspect(root, files, a.seq_pattern, a.depth)

    sequences = report_sequences(files, a.seq_pattern)
    train, val = [], []
    for s in sequences:
        k = max(1, round(len(s) * (1 - a.val_ratio))) if len(s) > 1 else 1  # 1-frame sequences go to train
        train += [files[i] for i in s[:k]]
        val += [files[i] for i in s[k:]]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f"{out.stem}_train.txt").write_text("\n".join(train) + "\n")
    (out.parent / f"{out.stem}_val.txt").write_text("\n".join(val) + "\n")
    names = a.names or [f"class_{i}" for i in range(max_class_id(img2label_paths(files)) + 1)]
    temporal = {"num_frames": a.num_frames, "stride": a.stride, "seq_pattern": a.seq_pattern, "diff_channel": a.diff}
    data = {
        "path": str(out.parent.resolve()),
        "train": f"{out.stem}_train.txt",
        "val": f"{out.stem}_val.txt",
        "names": dict(enumerate(names)),
        "temporal": temporal,
    }
    check_temporal_cfg({"temporal": dict(temporal)})  # validate before writing
    YAML.save(out, data)
    print(f"\n{len(train)} train / {len(val)} val images, {len(names)} classes {names}\nWrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="extracted STVD root directory")
    parser.add_argument("--inspect", action="store_true", help="only print layout, name samples and sequence stats")
    parser.add_argument("--depth", type=int, default=3, help="directory tree depth printed by --inspect")
    parser.add_argument("--image-dir", default=None, help="only use images below directories with this name")
    parser.add_argument("--seq-pattern", default=None, help="file name regex with (?P<idx>..) and optional (?P<seq>..)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="last fraction of each sequence used for val")
    parser.add_argument("--names", nargs="+", default=None, help="class names in id order (default: class_<id>)")
    parser.add_argument("--out", default="data/stvd_tempo.yaml", help="output data YAML")
    parser.add_argument("--num-frames", type=int, default=3, help="default temporal.num_frames in the YAML")
    parser.add_argument("--stride", type=int, default=1, help="default temporal.stride in the YAML")
    parser.add_argument("--diff", action="store_true", help="default temporal.diff_channel in the YAML")
    main(parser.parse_args())

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multi-frame early-fusion data support: frames t-(n-1)s, ..., t stacked on the channel axis, supervised by frame t."""

from __future__ import annotations

import random
import re
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.utils import LOGGER

from .augment import Format, RandomHSV
from .dataset import YOLODataset

TEMPORAL_DEFAULTS = {"num_frames": 1, "stride": 1, "seq_pattern": None, "diff_channel": False, "random_frames": False}
MIXING_AUGMENTATIONS = ("mosaic", "mixup", "cutmix", "copy_paste")  # combine different images, i.e. times/sequences


def check_temporal_cfg(data: dict) -> dict:
    """Validate and normalize `data['temporal']` in place and set `data['channels']` to the fused channel count.

    Args:
        data (dict): Dataset dictionary containing a `temporal` mapping.

    Returns:
        (dict): Normalized temporal configuration with all keys of `TEMPORAL_DEFAULTS`.

    Raises:
        ValueError: If a key is unknown, a value is out of range, or `channels` contradicts the temporal config.
    """
    cfg = {**TEMPORAL_DEFAULTS, **data["temporal"]}
    if unknown := set(cfg) - set(TEMPORAL_DEFAULTS):
        raise ValueError(f"Unknown 'temporal' keys {sorted(unknown)}, valid keys are {list(TEMPORAL_DEFAULTS)}")
    cfg["num_frames"], cfg["stride"] = int(cfg["num_frames"]), int(cfg["stride"])
    cfg["diff_channel"], cfg["random_frames"] = bool(cfg["diff_channel"]), bool(cfg["random_frames"])
    if cfg["num_frames"] < 1 or cfg["stride"] < 1:
        raise ValueError(f"'temporal' num_frames and stride must be >= 1, got {cfg['num_frames']}, {cfg['stride']}")
    if cfg["diff_channel"] and cfg["num_frames"] < 2:
        raise ValueError("'temporal' diff_channel=True needs num_frames >= 2 (difference between t and t-stride)")
    if cfg["seq_pattern"] is not None and "idx" not in re.compile(cfg["seq_pattern"]).groupindex:
        raise ValueError(f"'temporal' seq_pattern must define a named group (?P<idx>...), got {cfg['seq_pattern']}")
    channels = 3 * cfg["num_frames"] + cfg["diff_channel"]
    if data.get("channels", channels) != channels:
        raise ValueError(f"data 'channels: {data['channels']}' contradicts 'temporal' config requiring {channels}")
    data["temporal"], data["channels"] = cfg, channels
    return cfg


def sequence_key(path: str, seq_pattern: str | None = None) -> tuple[tuple[str, str], int] | None:
    """Return the (sequence id, frame index) of an image path, or None if the file name does not match.

    With `seq_pattern` the file name must match it; the sequence is its optional `seq` group and the frame its `idx`
    group. Without a pattern every directory is one video and the frame index is the last integer in the file stem.
    The parent directory is always part of the sequence id, so equal names in different directories never mix.

    Args:
        path (str): Image file path.
        seq_pattern (str, optional): Regex applied to the file name, with named groups `idx` and optionally `seq`.

    Returns:
        (tuple | None): ((parent directory, sequence name), frame index) or None if the name does not match.
    """
    p = Path(path)
    if seq_pattern:
        m = re.match(seq_pattern, p.name)
        return ((str(p.parent), m.groupdict().get("seq") or ""), int(m["idx"])) if m else None
    digits = re.findall(r"\d+", p.stem)
    return ((str(p.parent), ""), int(digits[-1])) if digits else None


def group_sequences(im_files: list[str], seq_pattern: str | None = None) -> tuple[list[list[int]], int]:
    """Group image files into sequences ordered by frame index.

    Args:
        im_files (list[str]): Image file paths.
        seq_pattern (str, optional): File-name regex, see `sequence_key`.

    Returns:
        sequences (list[list[int]]): Indices into `im_files` per sequence, sorted by frame index.
        unmatched (int): Number of files that did not match and became single-frame sequences.
    """
    groups, sequences = {}, []
    for i, f in enumerate(im_files):
        if key := sequence_key(f, seq_pattern):
            groups.setdefault(key[0], []).append((key[1], i))
        else:
            sequences.append([i])
    unmatched = len(sequences)
    sequences += [[i for _, i in sorted(v)] for _, v in sorted(groups.items())]
    return sequences, unmatched


def window_positions(pos: int, num_frames: int, stride: int) -> list[int]:
    """Return sequence positions [pos-(n-1)s, ..., pos], replacing positions before the start with the first frame."""
    return [max(pos - k * stride, 0) for k in range(num_frames - 1, -1, -1)]


def compose_temporal_input(
    img: np.ndarray, num_frames: int, diff_channel: bool, rgb: bool = True, num_valid: int | None = None
) -> np.ndarray:
    """Convert stacked BGR frames (H, W, 3n) into the model input layout (H, W, 3n(+1)).

    Frame order is kept (channels [0:3] earliest, last 3 channels target frame t); each frame is optionally converted
    to RGB, and the grayscale |t - (t-s)| difference is appended last when `diff_channel` is True. With `num_valid`
    below `num_frames` the earliest unused frame slots (and the difference if only t is valid) are zero.

    Args:
        img (np.ndarray): Stacked BGR frames of shape (H, W, 3 * num_frames), uint8.
        num_frames (int): Number of frame slots n.
        diff_channel (bool): Whether to append the absolute grayscale difference between the last two slots.
        rgb (bool): Whether to convert each frame from BGR to RGB.
        num_valid (int, optional): Number of valid most recent frames, defaults to all `num_frames`.

    Returns:
        (np.ndarray): Contiguous array of shape (H, W, 3 * num_frames + diff_channel).
    """
    num_valid = num_frames if num_valid is None else num_valid
    frames = [img[..., i : i + 3] for i in range(0, 3 * num_frames, 3)]
    if diff_channel:
        gray = [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_BGR2GRAY) for f in frames[-2:]]
        diff = cv2.absdiff(gray[1], gray[0]) if num_valid > 1 else np.zeros_like(gray[0])
    frames = [np.zeros_like(f) for f in frames[: num_frames - num_valid]] + frames[num_frames - num_valid :]
    if rgb:
        frames = [f[..., ::-1] for f in frames]
    if diff_channel:
        frames.append(diff[..., None])
    return np.ascontiguousarray(np.concatenate(frames, axis=2))


class TemporalHSV(RandomHSV):
    """RandomHSV that applies identical random gains to every 3-channel frame of a stacked multi-frame image."""

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply one HSV jitter draw to each 3-channel group of `labels['img']` in place."""
        img = labels["img"]
        state = np.random.get_state()
        for i in range(0, img.shape[-1] - img.shape[-1] % 3, 3):
            np.random.set_state(state)  # replay the same gains for every frame
            frame = {"img": np.ascontiguousarray(img[..., i : i + 3])}
            img[..., i : i + 3] = super().apply_image(frame)["img"]
        return labels


class TemporalFormat(Format):
    """Format that converts stacked BGR frames with `compose_temporal_input` instead of a single BGR-to-RGB flip."""

    def __init__(self, num_frames: int = 1, diff_channel: bool = False, **kwargs):
        """Initialize TemporalFormat with the frame layout and the standard Format arguments."""
        super().__init__(**kwargs)
        self.num_frames, self.diff_channel = num_frames, diff_channel

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Convert `labels['img']` (H, W, 3n) to a (3n(+1), H, W) tensor, honoring an optional `num_valid` key."""
        img, num_valid = labels.pop("img"), labels.pop("num_valid", None)
        rgb = random.uniform(0, 1) > self.bgr  # one draw for all frames, same RNG use as Format
        img = compose_temporal_input(img, self.num_frames, self.diff_channel, rgb=rgb, num_valid=num_valid)
        labels["img"] = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return labels


class TemporalYOLODataset(YOLODataset):
    """YOLO detection dataset returning n stacked frames per sample, labeled by the last (target) frame.

    Images are grouped into sequences by `data['temporal']['seq_pattern']` (or one sequence per directory) and sorted
    by frame index. Sample i stacks frames [t-(n-1)s, ..., t] of its own sequence, repeating the first frame where the
    window starts before the sequence. Frames are read one by one with `load_image`, so the regular RAM/disk image cache
    and the augmentation buffer apply per frame. Every spatial and color augmentation runs once on the stacked image, so
    all frames share identical random parameters; mosaic, mixup, cutmix and copy_paste are disabled.

    Attributes:
        temporal (dict): Normalized temporal configuration, see `check_temporal_cfg`.
        sequences (list[list[int]]): Dataset indices of each sequence sorted by frame index.
        windows (list[list[int]]): Dataset indices of the n input frames of each sample, earliest first.

    Examples:
        >>> data = {"names": {0: "obj"}, "temporal": {"num_frames": 3, "stride": 1}}
        >>> dataset = TemporalYOLODataset(img_path="path/to/images", data=data, augment=False)
        >>> dataset[0]["img"].shape  # (9, H, W)
    """

    def __init__(self, *args, data: dict, **kwargs):
        """Initialize TemporalYOLODataset; `data` must contain a `temporal` mapping."""
        self.temporal = t = check_temporal_cfg(data)
        self.format_class = partial(TemporalFormat, num_frames=t["num_frames"], diff_channel=t["diff_channel"])
        super().__init__(*args, data={**data, "channels": 3}, **kwargs)  # frames are read as 3-channel BGR images
        self.data = data
        self.sequences, unmatched = group_sequences(self.im_files, t["seq_pattern"])
        if unmatched:
            LOGGER.warning(f"{self.prefix}{unmatched} images do not match the temporal sequence rule, used as 1-frame")
        self.windows = [None] * len(self.im_files)
        for seq in self.sequences:
            for pos, i in enumerate(seq):
                self.windows[i] = [seq[p] for p in window_positions(pos, t["num_frames"], t["stride"])]
        lengths = [len(s) for s in self.sequences]
        LOGGER.info(
            f"{self.prefix}Temporal: {len(lengths)} sequences ({min(lengths)}-{max(lengths)} frames), "
            f"num_frames={t['num_frames']}, stride={t['stride']}, diff_channel={t['diff_channel']}, "
            f"input channels={data['channels']}"
        )

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Return the target frame's labels with `img` holding the stacked BGR frames (H, W, 3n), earliest first."""
        label = super().get_image_and_label(index)  # target frame image, shapes and instances
        h, w = label["img"].shape[:2]
        frames = []
        for j in self.windows[index]:
            im = label["img"] if j == index else self.load_image(j)[0]
            frames.append(im if im.shape[:2] == (h, w) else cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR))
        label["img"] = np.concatenate(frames, axis=2)
        if self.augment and self.temporal["random_frames"]:  # reserved: random n in [1, K], earliest slots zeroed
            label["num_valid"] = random.randint(1, self.temporal["num_frames"])
        return label

    def build_transforms(self, hyp):
        """Build YOLODataset transforms without cross-image mixing and with frame-synchronized HSV jitter."""
        if self.augment:
            if enabled := [k for k in MIXING_AUGMENTATIONS if getattr(hyp, k, 0.0)]:
                changes = ", ".join(f"{k}={getattr(hyp, k)}->0.0" for k in enabled)
                LOGGER.warning(f"{self.prefix}Temporal dataset sets {changes}, they mix different times or sequences")
            for k in MIXING_AUGMENTATIONS:
                setattr(hyp, k, 0.0)
        transforms = super().build_transforms(hyp)
        for i, t in enumerate(transforms.transforms):
            if type(t) is RandomHSV:
                transforms.transforms[i] = TemporalHSV(t.hgain, t.sgain, t.vgain)
        return transforms

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multi-frame early-fusion data support: frames t-(n-1)s, ..., t stacked on the channel axis, supervised by frame t."""

from __future__ import annotations

import random
import re
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.utils import LOGGER

from .augment import Albumentations, Format, RandomHSV
from .dataset import YOLODataset

# Channel layout of the fused input, stored with the config in checkpoints and export metadata
FRAME_ORDER = "rgb_frames_earliest_first_target_last_then_diff"
TEMPORAL_DEFAULTS = {
    "num_frames": None,  # n used for validation and prediction, defaults to max_frames
    "stride": 1,  # s used for validation and prediction
    "max_frames": None,  # K frame slots, input channels 3K(+1); defaults to the largest n configured
    "train_frames": None,  # n values sampled per training sample, defaults to [num_frames]
    "train_strides": None,  # s values sampled per training sample, defaults to [stride]
    "slot_fill": "zero",  # unused earlier slots when n < K: 'zero' or 'repeat' (earliest used frame)
    "cond": "none",  # 'film': condition the stem on (n, s), see ultralytics.models.yolo.detect.temporal.FrameFiLM
    "seq_pattern": None,
    "diff_channel": False,
    "frame_order": FRAME_ORDER,
}
SLOT_FILLS, CONDS = ("zero", "repeat"), ("none", "film")
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
    if cfg.pop("random_frames", False):  # stage-1 checkpoints: reserved flag, now train_frames 1..K with zero fill
        cfg["train_frames"] = cfg["train_frames"] or list(range(1, int(cfg["max_frames"] or cfg["num_frames"]) + 1))
    if unknown := set(cfg) - set(TEMPORAL_DEFAULTS):
        raise ValueError(f"Unknown 'temporal' keys {sorted(unknown)}, valid keys are {list(TEMPORAL_DEFAULTS)}")
    train_frames = sorted({int(n) for n in cfg["train_frames"] or []})
    n = None if cfg["num_frames"] is None else int(cfg["num_frames"])
    k = int(cfg["max_frames"]) if cfg["max_frames"] is not None else max([*train_frames, 1 if n is None else n])
    n, s = k if n is None else n, int(cfg["stride"])
    cfg.update(num_frames=n, stride=s, max_frames=k, train_frames=train_frames or [n])
    cfg["train_strides"] = sorted({int(x) for x in cfg["train_strides"] or [s]})
    cfg["diff_channel"] = bool(cfg["diff_channel"])
    if not all(1 <= x <= k for x in [n, *cfg["train_frames"]]):
        raise ValueError(f"'temporal' num_frames and train_frames must be in [1, max_frames={k}], got {cfg}")
    if min(s, *cfg["train_strides"]) < 1:
        raise ValueError(f"'temporal' stride and train_strides must be >= 1, got {cfg}")
    if cfg["slot_fill"] not in SLOT_FILLS or cfg["cond"] not in CONDS:
        raise ValueError(f"'temporal' slot_fill must be one of {SLOT_FILLS} and cond one of {CONDS}, got {cfg}")
    if cfg["frame_order"] != FRAME_ORDER:
        raise ValueError(f"'temporal' frame_order {cfg['frame_order']!r} is not supported, expected {FRAME_ORDER!r}")
    if cfg["seq_pattern"] is not None and "idx" not in re.compile(cfg["seq_pattern"]).groupindex:
        raise ValueError(f"'temporal' seq_pattern must define a named group (?P<idx>...), got {cfg['seq_pattern']}")
    channels = 3 * k + cfg["diff_channel"]
    if data.get("channels", channels) != channels:
        raise ValueError(f"data 'channels: {data['channels']}' contradicts 'temporal' config requiring {channels}")
    data["temporal"], data["channels"] = cfg, channels
    return cfg


def runtime_temporal_cfg(
    model_cfg: dict | None, data_cfg: dict | None = None, num_frames: int | None = None, stride: int | None = None
) -> dict:
    """Resolve the temporal config for validation or prediction with an optional runtime (n, stride).

    The model's config (checkpoint or export metadata) defines the input layout (max_frames, diff_channel, slot_fill,
    cond); the data config only fills in when the model carries none, and its `seq_pattern` names the frames to read.

    Args:
        model_cfg (dict | None): Temporal config stored with the model.
        data_cfg (dict | None): Temporal config of the data YAML.
        num_frames (int, optional): Runtime n, at most max_frames; defaults to the config's num_frames.
        stride (int, optional): Runtime stride; defaults to the config's stride.

    Returns:
        (dict): Normalized temporal config whose num_frames and stride are the ones to run with.
    """
    cfg = dict(model_cfg or data_cfg or {})
    if model_cfg and data_cfg and data_cfg.get("seq_pattern"):
        cfg["seq_pattern"] = data_cfg["seq_pattern"]
    cfg.update({k: v for k, v in (("num_frames", num_frames), ("stride", stride)) if v is not None})
    cfg = check_temporal_cfg({"temporal": cfg})
    if cfg["num_frames"] not in cfg["train_frames"] or cfg["stride"] not in cfg["train_strides"]:
        LOGGER.warning(
            f"Temporal: running n={cfg['num_frames']}, stride={cfg['stride']}, which training never sampled "
            f"(train_frames={cfg['train_frames']}, train_strides={cfg['train_strides']})"
        )
    return cfg


@lru_cache(maxsize=None)
def _log_albumentations_disabled() -> None:
    """Log once per process that temporal training does not use Albumentations."""
    LOGGER.info("Temporal dataset: Albumentations augmentations are disabled for all num_frames, including 1")


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


def slot_positions(pos: int, num_frames: int, stride: int, max_frames: int, diff_channel: bool) -> list[int]:
    """Return the sequence positions stacked for one sample: K frame slots, then t-stride if diff_channel is set.

    The n used frames fill the last slots (target t in the last one); the K-n earlier slots repeat the earliest used
    frame, which `compose_temporal_input` zeroes later for slot_fill='zero'. The extra t-stride frame only feeds the
    difference channel, so the difference is always |t - (t-stride)|, also for n=1.
    """
    window = window_positions(pos, num_frames, stride)
    return [window[0]] * (max_frames - num_frames) + window + ([max(pos - stride, 0)] if diff_channel else [])


def compose_temporal_input(
    img: np.ndarray, num_frames: int, diff_channel: bool, rgb: bool = True, num_valid: int | None = None
) -> np.ndarray:
    """Convert stacked BGR frames from `slot_positions` into the model input layout (H, W, 3K(+1)).

    Slot order is kept (channels [0:3] earliest, last 3 channels target frame t); each frame is optionally converted
    to RGB. With `diff_channel` the input holds one more frame, t-stride, after the K slots: it is replaced by the
    grayscale |t - (t-stride)| difference channel. With `num_valid` below `num_frames` the earliest K-num_valid slots
    are zero (slot_fill='zero').

    Args:
        img (np.ndarray): Stacked BGR frames of shape (H, W, 3K) or (H, W, 3K + 3) with diff_channel, uint8.
        num_frames (int): Number of frame slots K.
        diff_channel (bool): Whether the last 3 channels hold frame t-stride, turned into the difference channel.
        rgb (bool): Whether to convert each frame from BGR to RGB.
        num_valid (int, optional): Number of used most recent slots, defaults to all `num_frames`.

    Returns:
        (np.ndarray): Contiguous array of shape (H, W, 3K + diff_channel).
    """
    num_valid = num_frames if num_valid is None else num_valid
    frames = [img[..., i : i + 3] for i in range(0, 3 * num_frames, 3)]
    if diff_channel:
        gray = [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_BGR2GRAY) for f in (frames[-1], img[..., -3:])]
        diff = cv2.absdiff(gray[0], gray[1])
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
    """Format that converts stacked BGR frames with `compose_temporal_input` instead of a single BGR-to-RGB flip.

    As the last transform it runs after every geometric and photometric augmentation (including HSV), so the difference
    channel is recomputed from the augmented frames t and t-s and is itself never augmented. The sample's (n, stride)
    is returned as a `frame_cond` tensor.
    """

    def __init__(self, num_frames: int = 1, diff_channel: bool = False, slot_fill: str = "zero", **kwargs):
        """Initialize TemporalFormat with the slot count K, the slot layout and the standard Format arguments."""
        super().__init__(**kwargs)
        self.num_frames, self.diff_channel, self.slot_fill = num_frames, diff_channel, slot_fill

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Convert `labels['img']` to a (3K(+1), H, W) tensor and `labels['frame_cond']` to an (n, stride) tensor."""
        img, (n, stride) = labels.pop("img"), labels.pop("frame_cond")
        rgb = random.uniform(0, 1) > self.bgr  # one draw for all frames, same RNG use as Format
        num_valid = n if self.slot_fill == "zero" else None
        img = compose_temporal_input(img, self.num_frames, self.diff_channel, rgb=rgb, num_valid=num_valid)
        labels["img"] = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        labels["frame_cond"] = torch.tensor([n, stride])
        return labels


class TemporalYOLODataset(YOLODataset):
    """YOLO detection dataset returning K frame slots per sample, labeled by the last (target) frame.

    Images are grouped into sequences by `data['temporal']['seq_pattern']` (or one sequence per directory) and sorted
    by frame index. Sample i uses frames [t-(n-1)s, ..., t] of its own sequence, repeating the first frame where the
    window starts before the sequence, placed in the last n of K slots (`slot_positions`); earlier slots follow
    `slot_fill`. Training samples draw n from `train_frames` and s from `train_strides` per sample, evaluation uses
    `num_frames` and `stride`; the pair is returned as `frame_cond`. Frames are read one by one with `load_image`, so
    the regular RAM/disk image cache and the augmentation buffer apply per frame. Every spatial and color augmentation
    runs once on the stacked image, so all frames share identical random parameters; mosaic, mixup, cutmix, copy_paste
    and Albumentations are disabled.

    Attributes:
        temporal (dict): Normalized temporal configuration, see `check_temporal_cfg`.
        sequences (list[list[int]]): Dataset indices of each sequence sorted by frame index.
        positions (list[tuple[list[int], int]]): (sequence, position in it) of each sample.
        windows (list[list[int]]): Dataset indices of the K frame slots of each sample at the evaluation (n, stride).

    Examples:
        >>> data = {"names": {0: "obj"}, "temporal": {"max_frames": 3, "train_frames": [1, 2, 3]}}
        >>> dataset = TemporalYOLODataset(img_path="path/to/images", data=data, augment=False)
        >>> dataset[0]["img"].shape  # (9, H, W)
    """

    def __init__(self, *args, data: dict, **kwargs):
        """Initialize TemporalYOLODataset; `data` must contain a `temporal` mapping."""
        self.temporal = t = check_temporal_cfg(data)
        self.format_class = partial(
            TemporalFormat, num_frames=t["max_frames"], diff_channel=t["diff_channel"], slot_fill=t["slot_fill"]
        )
        super().__init__(*args, data={**data, "channels": 3}, **kwargs)  # frames are read as 3-channel BGR images
        self.data = data
        self.sequences, unmatched = group_sequences(self.im_files, t["seq_pattern"])
        if unmatched:
            LOGGER.warning(f"{self.prefix}{unmatched} images do not match the temporal sequence rule, used as 1-frame")
        self.positions = [None] * len(self.im_files)
        for seq in self.sequences:
            for pos, i in enumerate(seq):
                self.positions[i] = (seq, pos)
        self.windows = [
            self.frame_indices(i, t["num_frames"], t["stride"])[: t["max_frames"]] for i in range(len(self))
        ]
        lengths = [len(s) for s in self.sequences]
        LOGGER.info(
            f"{self.prefix}Temporal: {len(lengths)} sequences ({min(lengths)}-{max(lengths)} frames), "
            f"max_frames={t['max_frames']}, {'train_frames' if self.augment else 'num_frames'}="
            f"{t['train_frames'] if self.augment else t['num_frames']}, "
            f"{'train_strides' if self.augment else 'stride'}={t['train_strides'] if self.augment else t['stride']}, "
            f"slot_fill={t['slot_fill']}, diff_channel={t['diff_channel']}, input channels={data['channels']}"
        )

    def frame_indices(self, index: int, num_frames: int, stride: int) -> list[int]:
        """Return the dataset indices stacked for sample `index` at (n, stride), see `slot_positions`."""
        seq, pos = self.positions[index]
        t = self.temporal
        return [seq[p] for p in slot_positions(pos, num_frames, stride, t["max_frames"], t["diff_channel"])]

    def sample_frames(self) -> tuple[int, int]:
        """Return this sample's (n, stride): drawn from train_frames/train_strides when training, else the eval pair."""
        t = self.temporal
        if not self.augment:
            return t["num_frames"], t["stride"]
        n, s = t["train_frames"], t["train_strides"]  # no RNG draw for a single choice, so stage-1 runs are unchanged
        return (random.choice(n) if len(n) > 1 else n[0]), (random.choice(s) if len(s) > 1 else s[0])

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Return the target frame's labels with `img` holding the stacked BGR slots (H, W, 3K(+3)), earliest first."""
        label = super().get_image_and_label(index)  # target frame image, shapes and instances
        n, stride = self.sample_frames()
        h, w = label["img"].shape[:2]
        loaded = {index: label["img"]}  # repeated slots read each frame once
        for j in self.frame_indices(index, n, stride):
            if j not in loaded:
                im = self.load_image(j)[0]
                loaded[j] = im if im.shape[:2] == (h, w) else cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        frames = [loaded[j] for j in self.frame_indices(index, n, stride)]
        label["img"], label["frame_cond"] = np.concatenate(frames, axis=2), (n, stride)
        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate like YOLODataset and stack the per-sample (n, stride) into a (B, 2) `frame_cond` tensor."""
        new_batch = YOLODataset.collate_fn(batch)
        new_batch["frame_cond"] = torch.stack(new_batch["frame_cond"], 0)
        return new_batch

    def build_transforms(self, hyp):
        """Build YOLODataset transforms without cross-image mixing or Albumentations, with frame-synchronized HSV."""
        if self.augment:
            if enabled := [k for k in MIXING_AUGMENTATIONS if getattr(hyp, k, 0.0)]:
                changes = ", ".join(f"{k}={getattr(hyp, k)}->0.0" for k in enabled)
                LOGGER.warning(f"{self.prefix}Temporal dataset sets {changes}, they mix different times or sequences")
            for k in MIXING_AUGMENTATIONS:
                setattr(hyp, k, 0.0)
        transforms = super().build_transforms(hyp)
        if self.augment:  # Albumentations handles 1/3-channel images only, so drop it for every n (n=1 included)
            transforms.transforms = [t for t in transforms.transforms if not isinstance(t, Albumentations)]
            _log_albumentations_disabled()
        for i, t in enumerate(transforms.transforms):
            if type(t) is RandomHSV:
                transforms.transforms[i] = TemporalHSV(t.hgain, t.sgain, t.vgain)
        return transforms

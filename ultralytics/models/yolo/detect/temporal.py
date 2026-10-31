# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multi-frame early-fusion model utilities: first-conv inflation and streaming temporal prediction."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ultralytics.data.temporal import check_temporal_cfg, compose_temporal_input, sequence_key, window_positions
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import LOGGER, YAML


def inflate_first_conv(model: nn.Module, n: int, diff_channel: bool = False) -> nn.Module:
    """Inflate the first convolution of a 3-channel model in place to accept 3n (+1 difference) input channels.

    The RGB filters are repeated for each of the n frames and scaled by 1/n, and the difference-channel filter is zero,
    so for n identical frames and a zero difference channel the first layer output equals the original model's.

    Args:
        model (nn.Module): Model whose first `nn.Conv2d` takes 3 input channels, e.g. `YOLO("yolo26n.pt").model`.
        n (int): Number of stacked frames.
        diff_channel (bool): Whether one extra difference channel follows the frames.

    Returns:
        (nn.Module): The same model, with its first convolution and `yaml['channels']` updated.

    Examples:
        >>> from ultralytics.nn.tasks import DetectionModel
        >>> model = inflate_first_conv(DetectionModel("yolo26n.yaml", verbose=False), n=3, diff_channel=True)
        >>> model.yaml["channels"]
        10
    """
    parent, name, conv = next(
        (p, k, c) for p in model.modules() for k, c in p.named_children() if isinstance(c, nn.Conv2d)
    )
    channels = 3 * n + int(diff_channel)
    if conv.in_channels == channels:
        return model
    if conv.in_channels != 3 or conv.groups != 1:
        raise ValueError(f"inflate_first_conv expects a 3-channel first conv, got in_channels={conv.in_channels}")
    new = nn.Conv2d(
        channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(conv.weight.device, conv.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : 3 * n] = conv.weight.repeat(1, n, 1, 1) / n
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    new.weight.requires_grad_(conv.weight.requires_grad)
    setattr(parent, name, new)
    if isinstance(getattr(model, "yaml", None), dict):
        model.yaml["channels"] = channels
    return model


class TemporalPredictor(DetectionPredictor):
    """Streaming predictor for multi-frame early-fusion detection models.

    Keeps a ring buffer of the last (n-1)s+1 frames per stream and stacks frames [t-(n-1)s, ..., t] exactly like
    `TemporalYOLODataset`, repeating the first frame until the buffer is full; results belong to the newest frame t.
    The buffer restarts when the source path changes for videos and streams, or when the sequence id changes for image
    files (using `seq_pattern`, or one sequence per directory). Image names without a sequence id, such as in-memory
    numpy frames, continue the current stream, so calling predict() once per frame keeps the buffer. The temporal config is read from the trained model's
    `yaml['temporal']`, or from the `data` YAML for exported models.

    Examples:
        >>> from ultralytics import YOLO
        >>> from ultralytics.models.yolo.detect.temporal import TemporalPredictor
        >>> results = YOLO("runs/detect/train/weights/best.pt").predict("video.mp4", predictor=TemporalPredictor)
    """

    def setup_model(self, model, verbose: bool = True):
        """Set up the model and resolve the temporal frame layout it expects."""
        super().setup_model(model, verbose)
        cfg = getattr(getattr(self.model, "model", None), "yaml", {}).get("temporal")
        if cfg is None and self.args.data and Path(self.args.data).suffix in {".yaml", ".yml"}:
            cfg = YAML.load(self.args.data).get("temporal")  # exported models carry no temporal config
        data = {"temporal": cfg or {}, "channels": self.model.channels}
        self.temporal = check_temporal_cfg(data)
        self.buffer = deque(maxlen=(self.temporal["num_frames"] - 1) * self.temporal["stride"] + 1)
        self.buffer_key = None
        if verbose:
            LOGGER.info(f"TemporalPredictor: {self.temporal}")

    def _frame_key(self, path: str):
        """Return the stream identity of a frame; a change restarts the ring buffer."""
        if getattr(self.dataset, "mode", None) != "image":  # video or stream frames share their source path
            return path
        key = sequence_key(path, self.temporal["seq_pattern"])  # image files: sequence id from the file name
        return key[0] if key else None  # unmatched names, e.g. in-memory 'image0.jpg', continue the current stream

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Push each BGR frame into the ring buffer and return letterboxed multi-frame inputs (H, W, 3n(+1)), RGB."""
        n, s = self.temporal["num_frames"], self.temporal["stride"]
        paths = self.batch[0] if self.batch else [None] * len(im)
        stacked = []
        for frame, path in zip(im, paths):
            if (key := self._frame_key(path)) != self.buffer_key:
                self.buffer.clear()
                self.buffer_key = key
            frame = np.atleast_3d(frame)  # loaders read with the fused channel count (IMREAD_UNCHANGED): make BGR
            if frame.shape[2] in {1, 4}:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR if frame.shape[2] == 1 else cv2.COLOR_BGRA2BGR)
            self.buffer.append(frame[..., :3])
            stacked.append(np.concatenate([self.buffer[p] for p in window_positions(len(self.buffer) - 1, n, s)], 2))
        return [compose_temporal_input(x, n, self.temporal["diff_channel"]) for x in super().pre_transform(stacked)]

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """Convert a list of BGR frames into a normalized (B, 3n(+1), H, W) tensor; tensors pass through unchanged."""
        if isinstance(im, torch.Tensor):
            return super().preprocess(im)
        im = torch.from_numpy(np.stack(self.pre_transform(im))).to(self.device).permute(0, 3, 1, 2).contiguous()
        return (im.half() if self.model.fp16 else im.float()).div_(255)

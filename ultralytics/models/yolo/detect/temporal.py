# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multi-frame early-fusion models: first-conv inflation, (n, stride) FiLM, runtime-(n, stride) predict and val."""

from __future__ import annotations

import ast
from collections import deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ultralytics.data.temporal import compose_temporal_input, runtime_temporal_cfg, sequence_key, slot_positions
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.backends.base import BaseBackend
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


class FrameFiLM(nn.Module):
    """FiLM conditioning of stem features on the input's frame count n and stride s.

    An MLP maps the normalized features (n/K, s/S, (n-1)s / ((K-1)S)) to per-channel (scale, shift), applied as
    x * (1 + scale) + shift. The last layer starts at zero, so the module is an exact identity until trained and loading
    weights without FiLM (e.g. stage 1) keeps their behavior. Per-sample (n, s) comes from `cond` (set from the batch
    `frame_cond` during training, or by the caller); without it the `default` pair is used, which is also the pair
    baked into exported models.

    Attributes:
        max_frames (int): Slot count K used for normalization.
        max_stride (int): Largest training stride S used for normalization.
        default (tuple[int, int]): (n, s) used when no `cond` is set.
        cond (torch.Tensor | None): (B, 2) or (1, 2) tensor of (n, s) for the next forward passes.
    """

    def __init__(self, channels: int, max_frames: int, max_stride: int, default: tuple[int, int], hidden: int = 32):
        """Initialize an identity FiLM for `channels` feature channels."""
        super().__init__()
        self.max_frames, self.max_stride, self.default, self.cond = max_frames, max_stride, tuple(default), None
        self.mlp = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Scale and shift (B, C, H, W) features by the (n, s) embedding of `cond`, `self.cond` or `self.default`."""
        if cond is None:
            cond = self.cond if self.cond is not None else torch.tensor([self.default], device=x.device)
        n, s = cond.to(x.device, torch.float32).unbind(1)
        span = max((self.max_frames - 1) * self.max_stride, 1)
        feats = torch.stack([n / self.max_frames, s / self.max_stride, (n - 1) * s / span], 1)
        scale, shift = self.mlp(feats.to(self.mlp[0].weight.dtype)).to(x.dtype)[..., None, None].chunk(2, 1)
        return x * (1 + scale) + shift

    def capture(self, module: nn.Module, args: tuple) -> None:
        """Model forward pre-hook: take the per-sample (n, s) from a training batch dict."""
        if isinstance(args[0], dict) and "frame_cond" in args[0]:
            self.cond = args[0]["frame_cond"]

    def release(self, module: nn.Module, args: tuple, output) -> None:
        """Model forward hook: drop a batch's (n, s) after its loss pass, so evaluation falls back to `default`."""
        if isinstance(args[0], dict):
            self.cond = None

    def condition(self, module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
        """Stem forward hook: condition the stem output."""
        return self(output)


def attach_film(model: nn.Module, temporal: dict, layer: int = 1) -> nn.Module:
    """Add an identity `FrameFiLM` on the output of stem layer `layer` (default: after the first two convolutions).

    The module lives at `model.film`, so every existing state-dict key is unchanged and stage-1 weights load as before;
    it is applied through forward hooks, which tracing-based exports (e.g. ONNX) record like any other operation.

    Args:
        model (nn.Module): YOLO DetectionModel.
        temporal (dict): Normalized temporal config (max_frames, train_strides, num_frames, stride).
        layer (int): Index of the stem layer whose output is conditioned.

    Returns:
        (nn.Module): The same model.
    """
    stem = model.model[layer]
    channels = getattr(stem, "conv", stem).out_channels
    default = (temporal["num_frames"], temporal["stride"])
    model.film = FrameFiLM(channels, temporal["max_frames"], max(temporal["train_strides"]), default)
    stem.register_forward_hook(model.film.condition)
    model.register_forward_pre_hook(model.film.capture)
    model.register_forward_hook(model.film.release)
    return model


def find_film(model) -> FrameFiLM | None:
    """Return the FrameFiLM of a DetectionModel or an AutoBackend-wrapped PyTorch model, else None."""
    for m in (model, getattr(model, "model", None)):
        if isinstance(m, nn.Module) and (film := next((x for x in m.modules() if isinstance(x, FrameFiLM)), None)):
            return film
    return None


def set_frame_cond(model: nn.Module, num_frames: int, stride: int) -> nn.Module:
    """Set a model's default (n, stride), e.g. before export: FiLM models bake it in, and metadata records it.

    Args:
        model (nn.Module): Temporal DetectionModel.
        num_frames (int): n, at most max_frames.
        stride (int): Frame stride.

    Returns:
        (nn.Module): The same model.

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO("best.pt")
        >>> set_frame_cond(model.model, 2, 1)
        >>> model.export(format="onnx")  # a FiLM model now runs at n=2, stride=1
    """
    cfg = runtime_temporal_cfg(model.yaml["temporal"], num_frames=num_frames, stride=stride)
    model.yaml["temporal"] = cfg
    if film := find_film(model):
        film.default = (cfg["num_frames"], cfg["stride"])
    return model


def model_temporal_cfg(model) -> dict | None:
    """Return the temporal config stored with a model module, an AutoBackend, a *.pt path or an exported model path."""
    if isinstance(model, (str, Path)):
        if str(model).endswith(".pt"):
            from ultralytics.nn.tasks import load_checkpoint

            return model_temporal_cfg(load_checkpoint(model)[0])
        cfg = BaseBackend.read_metadata(model).get("temporal")
        return ast.literal_eval(cfg) if isinstance(cfg, str) else cfg
    for m in (model, getattr(model, "model", None)):
        if isinstance(getattr(m, "yaml", None), dict) and m.yaml.get("temporal"):
            return m.yaml["temporal"]
    return getattr(model, "temporal", None)  # AutoBackend export metadata


def check_baked_cond(cfg: dict, model_cfg: dict | None, film: FrameFiLM | None) -> None:
    """Raise if an exported FiLM model (no FrameFiLM module to re-condition) is asked for another (n, stride)."""
    if model_cfg and model_cfg.get("cond") == "film" and film is None:
        baked = (model_cfg["num_frames"], model_cfg["stride"])
        if (cfg["num_frames"], cfg["stride"]) != baked:
            raise ValueError(
                f"This exported FiLM model has (n, stride)={baked} baked in; re-export with set_frame_cond() for "
                f"({cfg['num_frames']}, {cfg['stride']}), or use the *.pt model to switch at runtime"
            )


@lru_cache(maxsize=None)
def _variant(cls: type, num_frames: int | None, stride: int | None) -> type:
    """Return a cached subclass of `cls` fixed to a runtime (n, stride); the same arguments give the same class."""
    return type(f"{cls.__name__}_n{num_frames}_s{stride}", (cls,), {"frames": num_frames, "frame_stride": stride})


class TemporalPredictor(DetectionPredictor):
    """Streaming predictor for multi-frame early-fusion detection models at a runtime-selectable (n, stride).

    Keeps a ring buffer of recent frames per stream and stacks K slots exactly like `TemporalYOLODataset`
    (`slot_positions`): frames [t-(n-1)s, ..., t] in the last n slots, earlier slots per `slot_fill`, repeating the
    first frame until the buffer is full; results belong to the newest frame t. The buffer restarts when the source
    path changes for videos and streams, or when the sequence id changes for image files (using `seq_pattern`, or one
    sequence per directory). Image names without a sequence id, such as in-memory numpy frames, continue the current
    stream, so calling predict() once per frame keeps the buffer.

    The temporal config comes from the model metadata first (`yaml['temporal']` of a checkpoint, or the `temporal` entry
    of exported-model metadata) and from the `data` YAML only when the model carries none. (n, stride) default to the
    config's num_frames and stride; `TemporalPredictor.at(n, stride)` selects others (n <= max_frames). PyTorch FiLM
    models are conditioned on the selected pair; exported FiLM models only run at the pair baked in at export.

    Examples:
        >>> from ultralytics import YOLO
        >>> from ultralytics.models.yolo.detect.temporal import TemporalPredictor
        >>> model = YOLO("runs/detect/train/weights/best.pt")
        >>> results = model.predict("video.mp4", predictor=TemporalPredictor)
        >>> results = model.predict("video.mp4", predictor=TemporalPredictor.at(2, 3))  # n=2, stride=3
    """

    frames: int | None = None  # runtime n, None uses the config's num_frames
    frame_stride: int | None = None  # runtime stride, None uses the config's stride

    @classmethod
    def at(cls, num_frames: int | None = None, stride: int | None = None) -> type:
        """Return this predictor class fixed to a runtime (n, stride), cached so repeated predict() calls reuse it."""
        return _variant(cls, num_frames, stride)

    def setup_model(self, model, verbose: bool = True):
        """Set up the model and resolve the temporal frame layout and the (n, stride) to run with."""
        super().setup_model(model, verbose)
        model_cfg = model_temporal_cfg(self.model)
        data_yaml = self.args.data if self.args.data and Path(self.args.data).suffix in {".yaml", ".yml"} else None
        data_cfg = YAML.load(data_yaml).get("temporal") if data_yaml and Path(data_yaml).is_file() else None
        self.temporal = runtime_temporal_cfg(model_cfg, data_cfg, self.frames, self.frame_stride)
        t = self.temporal
        if self.model.channels != 3 * t["max_frames"] + t["diff_channel"]:
            raise ValueError(f"TemporalPredictor: model has {self.model.channels} input channels, config {t} does not")
        if model_cfg and data_cfg and data_cfg.get("max_frames", t["max_frames"]) != t["max_frames"]:
            LOGGER.warning(f"TemporalPredictor: data={data_yaml} temporal config differs, using model metadata")
        self.film = find_film(self.model)
        check_baked_cond(t, model_cfg, self.film)
        n, s = t["num_frames"], t["stride"]
        self.buffer = deque(maxlen=max((n - 1) * s, s if t["diff_channel"] else 0) + 1)
        self.buffer_key = None
        if verbose:
            LOGGER.info(f"TemporalPredictor: {t}")

    def _frame_key(self, path: str):
        """Return the stream identity of a frame; a change restarts the ring buffer."""
        if getattr(self.dataset, "mode", None) != "image":  # video or stream frames share their source path
            return path
        key = sequence_key(path, self.temporal["seq_pattern"])  # image files: sequence id from the file name
        return key[0] if key else None  # unmatched names, e.g. in-memory 'image0.jpg', continue the current stream

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Push each BGR frame into the ring buffer and return letterboxed K-slot inputs (H, W, 3K(+1)), RGB."""
        t = self.temporal
        n, s, k, diff = t["num_frames"], t["stride"], t["max_frames"], t["diff_channel"]
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
            positions = slot_positions(len(self.buffer) - 1, n, s, k, diff)
            stacked.append(np.concatenate([self.buffer[p] for p in positions], 2))
        num_valid = n if t["slot_fill"] == "zero" else None
        return [compose_temporal_input(x, k, diff, num_valid=num_valid) for x in super().pre_transform(stacked)]

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """Convert a list of BGR frames into a normalized (B, 3K(+1), H, W) tensor; tensors pass through unchanged."""
        if isinstance(im, torch.Tensor):
            return super().preprocess(im)
        im = torch.from_numpy(np.stack(self.pre_transform(im))).to(self.device).permute(0, 3, 1, 2).contiguous()
        return (im.half() if self.model.fp16 else im.float()).div_(255)

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """Run inference with a PyTorch FiLM model conditioned on the selected (n, stride)."""
        if self.film is None:
            return super().inference(im, *args, **kwargs)
        self.film.cond = torch.tensor([[self.temporal["num_frames"], self.temporal["stride"]]], device=im.device)
        try:
            return super().inference(im, *args, **kwargs)
        finally:
            self.film.cond = None


class TemporalValidator(DetectionValidator):
    """Validator for temporal models at a runtime-selectable (n, stride), reading the layout from model metadata.

    The model's stored temporal config (checkpoint `yaml['temporal']` or export metadata) defines max_frames,
    diff_channel, slot_fill and cond; the data YAML provides the images and its `seq_pattern`. Training-time validation
    keeps using the trainer's dataloader and DetectionValidator.

    Examples:
        >>> from ultralytics import YOLO
        >>> from ultralytics.models.yolo.detect.temporal import TemporalValidator
        >>> metrics = YOLO("best.pt").val(data="data.yaml", validator=TemporalValidator.at(2, 3))
    """

    frames: int | None = None  # runtime n, None uses the model config's num_frames
    frame_stride: int | None = None  # runtime stride, None uses the model config's stride

    @classmethod
    def at(cls, num_frames: int | None = None, stride: int | None = None) -> type:
        """Return this validator class fixed to a runtime (n, stride), see `TemporalPredictor.at`."""
        return _variant(cls, num_frames, stride)

    def get_model(self, model, trainer=None):
        """Return the model to validate, conditioning a FiLM model's copy on the selected (n, stride)."""
        model = super().get_model(model, trainer)
        if trainer is not None:
            return model
        if model is None and str(self.args.model).endswith(".pt"):
            from ultralytics.nn.tasks import load_checkpoint

            model = load_checkpoint(self.args.model)[0]
        self.model_temporal = model_temporal_cfg(model if model is not None else self.args.model)
        cfg = runtime_temporal_cfg(self.model_temporal, None, self.frames, self.frame_stride)
        film = find_film(model) if model is not None else None
        check_baked_cond(cfg, self.model_temporal, film)
        if film is not None:  # this is the validator's own copy
            film.default = (cfg["num_frames"], cfg["stride"])
        return model

    def get_dataloader(self, dataset_path: str, batch_size: int):
        """Build the dataloader with the model's temporal layout and the selected (n, stride)."""
        if not self.training:
            data_cfg = self.data.get("temporal")
            cfg = runtime_temporal_cfg(self.model_temporal, data_cfg, self.frames, self.frame_stride)
            self.data = {**self.data, "temporal": cfg, "channels": 3 * cfg["max_frames"] + cfg["diff_channel"]}
        return super().get_dataloader(dataset_path, batch_size)

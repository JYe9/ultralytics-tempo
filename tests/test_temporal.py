# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CPU tests for multi-frame early fusion on synthetic sequences (no network, no pretrained weights)."""

import importlib.util
import random
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import BaseMixTransform
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.temporal import TemporalYOLODataset, check_temporal_cfg, group_sequences, sequence_key
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.temporal import TemporalPredictor, inflate_first_conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_requirements

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tempo" / "make_synthetic_seq.py"
_spec = importlib.util.spec_from_file_location("make_synthetic_seq", SCRIPT)
make_synthetic_seq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_synthetic_seq)

IMGSZ = 128
AUG = {
    "hsv_h": 0.1,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.2,
    "scale": 0.5,
    "shear": 2.0,
    "fliplr": 0.5,
    "flipud": 0.5,
}


def make_data(tmp_path, name="synth", **temporal):
    """Generate the synthetic dataset and return its checked data dict."""
    return check_det_dataset(str(make_synthetic_seq.make_dataset(tmp_path / name, temporal=temporal)))


def build(data, mode="val", **overrides):
    """Build the dataset that the trainer/validator would build for `data`."""
    cfg = get_cfg(overrides={"imgsz": IMGSZ, **overrides})
    return build_yolo_dataset(cfg, data["train"], 4, data, mode=mode)


def seeded(dataset, i, seed=0):
    """Return dataset[i] with all RNGs seeded."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return dataset[i]


def frame_id(path):
    """Return (sequence name, frame index) of a synthetic frame path."""
    key, idx = sequence_key(path, make_synthetic_seq.SEQ_PATTERN)
    return key[1], idx


def test_check_temporal_cfg():
    """Channels follow num_frames/diff_channel and invalid configs raise."""
    data = {"temporal": {"num_frames": 3, "diff_channel": True}}
    assert check_temporal_cfg(data)["stride"] == 1 and data["channels"] == 10
    assert check_temporal_cfg(data) == data["temporal"]  # idempotent
    for bad in ({"num_frames": 0}, {"num_frames": 1, "diff_channel": True}, {"seq_pattern": r"(\d+)"}, {"nframes": 2}):
        with pytest.raises(ValueError):
            check_temporal_cfg({"temporal": bad})
    with pytest.raises(ValueError):
        check_temporal_cfg({"temporal": {"num_frames": 2}, "channels": 3})


def test_group_sequences():
    """Frames group by sequence (and directory), sort numerically, and unmatched names become single frames."""
    files = ["d/b_10.png", "d/a_2.png", "d/b_9.png", "d/a_10.png", "e/a_1.png", "d/readme.png"]
    sequences, unmatched = group_sequences(files, r"^(?P<seq>[a-z]+)_(?P<idx>\d+)\.png$")
    assert unmatched == 1
    assert [[files[i] for i in s] for s in sequences] == [
        ["d/readme.png"],
        ["d/a_2.png", "d/a_10.png"],
        ["d/b_9.png", "d/b_10.png"],
        ["e/a_1.png"],
    ]
    sequences, _ = group_sequences(["v1/f10.jpg", "v1/f2.jpg", "v2/f1.jpg"])  # directory mode
    assert sequences == [[1, 0], [2]]


def test_windows_padding_and_boundaries(tmp_path):
    """Windows are [t-(n-1)s, ..., t] within one sequence, padded with the first frame at the start."""
    n, s = 3, 2
    dataset = build(make_data(tmp_path, num_frames=n, stride=s))
    assert isinstance(dataset, TemporalYOLODataset) and len(dataset.sequences) == 3
    for i, window in enumerate(dataset.windows):
        seq, t = frame_id(dataset.im_files[i])
        ids = [frame_id(dataset.im_files[j]) for j in window]
        assert all(q == seq for q, _ in ids)  # never crosses sequences
        assert [k for _, k in ids] == [max(t - k * s, 0) for k in range(n - 1, -1, -1)]  # order and padding


@pytest.mark.parametrize("num_frames,diff_channel", [(1, False), (2, True), (3, False), (3, True)])
def test_output_shape(tmp_path, num_frames, diff_channel):
    """Samples are (3n(+1), H, W) for train and val, and data channels match."""
    data = make_data(tmp_path, num_frames=num_frames, diff_channel=diff_channel)
    c = 3 * num_frames + diff_channel
    assert data["channels"] == c
    for mode in ("train", "val"):
        assert build(data, mode)[5]["img"].shape == (c, IMGSZ, IMGSZ)


def test_frame_layout(tmp_path):
    """Channels are RGB frames earliest-first with the target last, then the |t - (t-s)| grayscale difference."""
    import cv2

    dataset = build(make_data(tmp_path, num_frames=2, stride=3, diff_channel=True))
    i = 7
    img = dataset[i]["img"].numpy()
    frames = [cv2.imread(dataset.im_files[j]) for j in dataset.windows[i]]  # 128x128, so no letterbox resize
    assert frame_id(dataset.im_files[dataset.windows[i][0]])[1] == frame_id(dataset.im_files[i])[1] - 3
    np.testing.assert_array_equal(img[0:3], frames[0][..., ::-1].transpose(2, 0, 1))
    np.testing.assert_array_equal(img[3:6], frames[1][..., ::-1].transpose(2, 0, 1))
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(int) for f in frames]
    np.testing.assert_array_equal(img[6], np.abs(gray[1] - gray[0]))


def test_augmentation_shared_across_frames(tmp_path):
    """One random draw augments all frames: identical frames stay identical, the target matches a 1-frame dataset."""
    data = make_data(tmp_path, num_frames=3, stride=1)
    static = tmp_path / "static" / "images"  # one sequence whose frames are all the same image
    shutil.copytree(tmp_path / "synth" / "labels", static.parent / "labels")
    static.mkdir(parents=True)
    for f in sorted((tmp_path / "synth" / "images").glob("seq0_*.png")):
        shutil.copy(tmp_path / "synth" / "images" / "seq0_000.png", static / f.name)
    shutil.copy(tmp_path / "synth" / "labels" / "seq0_000.txt", static.parent / "labels" / "seq0_005.txt")
    static_data = {**data, "train": str(static), "temporal": dict(data["temporal"])}
    dataset = build(static_data, "train", **AUG)
    for seed in range(5):
        img = seeded(dataset, 5, seed)["img"]
        assert not torch.equal(img[6:9], seeded(dataset, 5, seed + 100)["img"][6:9])  # augmentation is active
        assert torch.equal(img[0:3], img[3:6]) and torch.equal(img[3:6], img[6:9])

    # target frame of n=3 is augmented exactly like the same frame in a 1-frame dataset with the same seed
    single = build(make_data(tmp_path, "single", num_frames=1), "train", **AUG)
    multi = build(data, "train", **AUG)
    for i in (3, 20):
        a, b = seeded(multi, i, i), seeded(single, i, i)
        assert torch.equal(a["img"][6:9], b["img"]) and torch.equal(a["bboxes"], b["bboxes"])


def test_mixing_augmentations_disabled(tmp_path):
    """Mosaic, mixup, cutmix and copy_paste are forced off for training."""
    dataset = build(make_data(tmp_path), "train", mosaic=1.0, mixup=0.5, cutmix=0.5, copy_paste=0.5)
    pre, mixup, cutmix = dataset.transforms.transforms[:3]
    mixers = [t for t in [*pre.transforms, mixup, cutmix] if isinstance(t, BaseMixTransform)]
    assert len(mixers) == 4 and all(t.p == 0.0 for t in mixers)


def test_random_frames_zero_slots(tmp_path):
    """Reserved random_frames mode samples n in [1, K] and zeroes the earliest unused slots."""
    data = make_data(tmp_path, num_frames=3, random_frames=True)
    dataset = build(data, "train", **{k: 0.0 for k in AUG})
    zero_slots = set()
    for seed in range(20):
        img = seeded(dataset, 6, seed)["img"]
        zero_slots.add(sum(int(img[k : k + 3].max() == 0) for k in (0, 3, 6)))
        assert img[6:9].max() > 0  # target frame always present
    assert zero_slots == {0, 1, 2}
    assert build(data, "val")[6]["img"][0:3].max() > 0  # all frames at evaluation


def test_n1_matches_yolodataset(tmp_path):
    """With num_frames=1 the temporal dataset reproduces YOLODataset samples bit for bit (train and val)."""
    data = make_data(tmp_path, num_frames=1)
    assert data["channels"] == 3
    plain = {k: v for k, v in data.items() if k != "temporal"}
    for mode, hyp in (("train", {**AUG, "mosaic": 0.0}), ("val", {})):
        a, b = build(data, mode, **hyp), build(plain, mode, **hyp)
        assert type(a) is TemporalYOLODataset and type(b) is YOLODataset
        for i in (0, 11, 30):
            x, y = seeded(a, i, i), seeded(b, i, i)
            assert x.keys() == y.keys()
            for k in ("img", "cls", "bboxes", "batch_idx"):
                assert torch.equal(x[k], y[k]), k


@pytest.mark.parametrize("n,diff_channel", [(1, False), (3, False), (3, True)])
def test_inflate_first_conv(n, diff_channel):
    """Inflated first conv (and whole model) matches the 3-channel model on n identical frames and zero difference."""
    torch.manual_seed(0)
    src = DetectionModel("yolo26n.yaml", nc=3, verbose=False).eval()
    for m in src.modules():  # non-trivial BN statistics
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5), m.running_var.uniform_(0.5, 1.5)
    dst = inflate_first_conv(deepcopy(src), n, diff_channel)
    assert inflate_first_conv(dst, n, diff_channel) is dst  # no-op once inflated
    c = 3 * n + diff_channel
    assert dst.yaml["channels"] == c and dst.model[0].conv.in_channels == c
    x = torch.rand(2, 3, 64, 64)
    xn = torch.cat([x] * n + [torch.zeros(2, int(diff_channel), 64, 64)], 1)
    with torch.no_grad():
        torch.testing.assert_close(dst.model[0](xn), src.model[0](x), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(dst(xn)[0], src(x)[0], rtol=1e-4, atol=1e-4)
    if c != 3:
        with pytest.raises(ValueError):
            inflate_first_conv(dst, n + 1)


def test_trainer_inflates_pretrained_weights(tmp_path):
    """DetectionTrainer.get_model inflates 3-channel weights for temporal data and records the config."""
    yaml = make_synthetic_seq.make_dataset(tmp_path / "synth", temporal={"num_frames": 3, "diff_channel": True})
    trainer = DetectionTrainer(overrides={"model": "yolo26n.yaml", "data": str(yaml), "project": str(tmp_path)})
    src = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
    model = trainer.get_model(cfg="yolo26n.yaml", weights=src, verbose=False)
    w, w0 = model.model[0].conv.weight, src.model[0].conv.weight
    assert src.model[0].conv.in_channels == 3  # caller's weights untouched
    torch.testing.assert_close(w[:, :9], w0.repeat(1, 3, 1, 1) / 3)
    assert w[:, 9].abs().sum() == 0 and model.yaml["temporal"]["num_frames"] == 3


def test_predictor_matches_dataset(tmp_path):
    """TemporalPredictor ring buffer builds exactly the val dataset inputs, restarting per sequence."""
    data = make_data(tmp_path, num_frames=3, stride=2, diff_channel=True)
    dataset = build(data, "val")
    model = DetectionModel("yolo26n.yaml", nc=3, ch=data["channels"], verbose=False)
    model.yaml["temporal"] = data["temporal"]
    predictor = TemporalPredictor(overrides={"imgsz": IMGSZ, "device": "cpu", "save": False, "verbose": False})
    inputs = []
    predictor.preprocess = lambda im, f=predictor.preprocess: inputs.append(f(im)) or inputs[-1]
    files = sorted(dataset.im_files, key=frame_id)  # 3 sequences, frames in temporal order
    results = predictor(source=files, model=model)
    assert len(results) == len(files) and results[0].orig_img.shape == (IMGSZ, IMGSZ, 3)
    x = torch.cat(inputs)
    for k, f in enumerate(files):
        expected = dataset[dataset.im_files.index(f)]["img"].float() / 255
        torch.testing.assert_close(x[k], expected, rtol=0, atol=0)


def test_train_val_predict_export(tmp_path):
    """Train n=3 for 1 epoch on CPU, then validate, stream-predict and export to ONNX."""
    yaml = make_synthetic_seq.make_dataset(tmp_path / "synth", temporal={"num_frames": 3, "stride": 1})
    model = YOLO("yolo26n.yaml")
    model.train(
        data=str(yaml), imgsz=IMGSZ, epochs=1, batch=4, device="cpu", workers=0, project=str(tmp_path), name="t"
    )
    best = tmp_path / "t" / "weights" / "best.pt"
    trained = YOLO(best)
    assert trained.model.yaml["channels"] == 9 and trained.model.yaml["temporal"]["num_frames"] == 3

    metrics = trained.val(data=str(yaml), imgsz=IMGSZ, batch=4, device="cpu", workers=0, project=str(tmp_path))
    assert 0.0 <= metrics.box.map50 <= 1.0 and len(metrics.box.ap_class_index) > 0

    frames = sorted((tmp_path / "synth" / "images").glob("seq1_*.png"))
    results = trained.predict(frames, predictor=TemporalPredictor, imgsz=IMGSZ, device="cpu", conf=0.001)
    assert len(results) == len(frames) and all(r.orig_img.shape == (IMGSZ, IMGSZ, 3) for r in results)

    import cv2

    stream = YOLO(best)  # in-memory frames, one predict() call per frame, keeps the ring buffer
    for f in frames[:4]:
        stream.predict(cv2.imread(str(f)), predictor=TemporalPredictor, imgsz=IMGSZ, device="cpu", verbose=False)
    assert len(stream.predictor.buffer) == 3

    if not check_requirements("onnx", install=False):
        return
    onnx = trained.export(format="onnx", imgsz=IMGSZ, device="cpu")
    assert YAML.load(yaml)["temporal"]["num_frames"] == 3
    if check_requirements("onnxruntime", install=False):
        results = YOLO(onnx, task="detect").predict(
            frames, predictor=TemporalPredictor, data=str(yaml), imgsz=IMGSZ, device="cpu"
        )
        assert len(results) == len(frames)

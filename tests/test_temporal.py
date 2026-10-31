# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CPU tests for multi-frame early fusion on synthetic sequences (no network, no pretrained weights)."""

import importlib.util
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Albumentations, BaseMixTransform
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.temporal import (
    FRAME_ORDER,
    TemporalYOLODataset,
    check_temporal_cfg,
    group_sequences,
    sequence_key,
)
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.temporal import TemporalPredictor, inflate_first_conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_requirements

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "tempo"


def load_script(name):
    """Import a scripts/tempo module by file name."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = sys.modules[name] = importlib.util.module_from_spec(spec)  # registered first for dataclasses
    spec.loader.exec_module(module)
    return module


make_synthetic_seq = load_script("make_synthetic_seq")
prepare = load_script("prepare_seq_dataset")
run_matrix = load_script("run_matrix")

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
    """With num_frames=1 the temporal dataset reproduces YOLODataset (without Albumentations) bit for bit."""
    data = make_data(tmp_path, num_frames=1)
    assert data["channels"] == 3
    plain = {k: v for k, v in data.items() if k != "temporal"}
    for mode, hyp in (("train", {**AUG, "mosaic": 0.0}), ("val", {})):
        a, b = build(data, mode, **hyp), build(plain, mode, **hyp)
        assert type(a) is TemporalYOLODataset and type(b) is YOLODataset
        b.transforms.transforms = [t for t in b.transforms.transforms if not isinstance(t, Albumentations)]
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


def test_trainer_inflates_pretrained_pt(tmp_path):
    """A 3-channel .pt loaded through the real trainer path is inflated by inflate_first_conv, not BaseModel.load."""
    yaml = make_synthetic_seq.make_dataset(tmp_path / "synth", temporal={"num_frames": 3, "diff_channel": True})
    torch.manual_seed(0)
    src = DetectionModel("yolo26n.yaml", nc=3, verbose=False)  # random init, so every filter is distinct
    pt = tmp_path / "random3ch.pt"
    torch.save({"model": deepcopy(src), "train_args": {}}, pt)

    trainer = DetectionTrainer(overrides={"model": str(pt), "data": str(yaml), "project": str(tmp_path)})
    trainer.setup_model()  # load_checkpoint -> DetectionTrainer.get_model -> inflate_first_conv -> BaseModel.load
    model, w0 = trainer.model, src.model[0].conv.weight
    w = model.model[0].conv.weight
    assert w.shape[1] == 10 and model.yaml["channels"] == 10 and model.yaml["temporal"]["num_frames"] == 3
    torch.testing.assert_close(w[:, :9], w0.repeat(1, 3, 1, 1) / 3, rtol=0, atol=0)  # RGB x3, scaled by 1/3
    assert w[:, 9].abs().sum() == 0  # difference channel
    sd, src_sd = model.state_dict(), src.state_dict()
    for k, v in src_sd.items():  # every other tensor transferred unchanged
        if k != "model.0.conv.weight" and sd[k].shape == v.shape:
            torch.testing.assert_close(sd[k], v, rtol=0, atol=0)

    # BaseModel.load alone would copy the RGB filters unscaled into channels 0:3 and keep random init elsewhere
    plain = DetectionModel("yolo26n.yaml", nc=3, ch=10, verbose=False)
    plain.load(deepcopy(src), verbose=False)
    torch.testing.assert_close(plain.model[0].conv.weight[:, :3], w0)
    assert not torch.equal(plain.model[0].conv.weight, w)

    # Model.train passes the in-memory module: it is inflated on a copy, the caller's model keeps 3 channels
    get_model = trainer.get_model(cfg="yolo26n.yaml", weights=src, verbose=False)
    assert src.model[0].conv.in_channels == 3 and torch.equal(get_model.model[0].conv.weight, w)


def test_diff_channel_after_photometric_augmentation(tmp_path):
    """The difference channel is recomputed from the HSV-augmented frames t and t-s and is not HSV-augmented itself."""
    import cv2

    data = make_data(tmp_path, num_frames=2, stride=2, diff_channel=True)
    hsv_only = {**{k: 0.0 for k in AUG}, "hsv_h": 0.5, "hsv_s": 0.9, "hsv_v": 0.9}
    dataset = build(data, "train", **hsv_only)
    i = 7
    raw = [cv2.imread(dataset.im_files[j]) for j in dataset.windows[i]]  # no geometric change with these settings
    raw_diff = cv2.absdiff(*(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in raw[::-1]))
    changed = 0
    for seed in range(5):
        img = seeded(dataset, i, seed)["img"].numpy().transpose(1, 2, 0)  # (H, W, 7): RGB t-s, RGB t, diff
        gray = [cv2.cvtColor(np.ascontiguousarray(img[..., k : k + 3]), cv2.COLOR_RGB2GRAY) for k in (0, 3)]
        np.testing.assert_array_equal(img[..., 6], cv2.absdiff(gray[1], gray[0]))  # from the augmented frames
        assert not np.array_equal(img[..., 3:6], raw[1][..., ::-1])  # HSV changed the frames
        changed += not np.array_equal(img[..., 6], raw_diff)
    assert changed  # so the difference differs from the one of the raw frames


def test_albumentations_disabled(tmp_path):
    """Temporal training pipelines contain no Albumentations transform for any num_frames, including 1."""
    from ultralytics.data.augment import Albumentations

    for n in (1, 3):
        dataset = build(make_data(tmp_path, f"n{n}", num_frames=n), "train")
        assert not any(isinstance(t, Albumentations) for t in dataset.transforms.transforms)


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
    ckpt = torch.load(best, map_location="cpu", weights_only=False)  # metadata saved in the checkpoint
    assert (ckpt.get("ema") or ckpt["model"]).yaml["temporal"]["frame_order"] == FRAME_ORDER

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

    other = tmp_path / "n2.yaml"  # model metadata wins over a conflicting data= YAML
    YAML.save(other, {**YAML.load(yaml), "temporal": {"num_frames": 2}})
    model = YOLO(best)
    model.predict(frames[:1], predictor=TemporalPredictor, data=str(other), imgsz=IMGSZ, device="cpu")
    assert model.predictor.temporal["num_frames"] == 3

    if not check_requirements("onnx", install=False):
        return
    import ast

    import onnx

    exported = trained.export(format="onnx", imgsz=IMGSZ, device="cpu")
    meta = {p.key: p.value for p in onnx.load(exported).metadata_props}
    assert ast.literal_eval(meta["temporal"]) == trained.model.yaml["temporal"]
    if check_requirements("onnxruntime", install=False):
        model = YOLO(exported, task="detect")  # no data=: the temporal config comes from the export metadata
        results = model.predict(frames, predictor=TemporalPredictor, imgsz=IMGSZ, device="cpu")
        assert len(results) == len(frames) and model.predictor.temporal == trained.model.yaml["temporal"]


# Dataset adapters (scripts/tempo/prepare_seq_dataset.py) on tiny synthetic layouts ------------------------------------
def write_frames(d, names, w=64, h=48):
    """Write small gray frames with the given file names into directory d."""
    import cv2

    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        cv2.imwrite(str(d / n), np.full((h, w, 3), 100, np.uint8))


def adapter(name, root, *args):
    """Run an adapter on root and return (sequences, names, options)."""
    opt = prepare.parse_args([name, "--root", str(root), *args])
    return (*prepare.ADAPTERS[name](Path(root), opt), opt)


def convert(name, root, tmp_path, *args):
    """Convert a dataset and return the checked data dict of the written YAML."""
    out = tmp_path / f"{name}_out"
    opt = prepare.parse_args(
        [name, "--root", str(root), "--out-dir", str(out), "--out-yaml", str(out / "d.yaml"), *args]
    )
    return check_det_dataset(str(prepare.main(opt)))


def test_adapter_stvd(tmp_path, capsys):
    """STVD: sequences from seq_pattern, YOLO labels copied, temporal 80/20 split, inspect statistics."""
    root = make_synthetic_seq.make_dataset(tmp_path / "stvd").parent
    pattern = make_synthetic_seq.SEQ_PATTERN
    seqs, names, _ = adapter("stvd", root, "--seq-pattern", pattern, "--names", "red", "green", "blue")
    assert len(seqs) == 3 and [len(s.frames) for s in seqs] == [12] * 3 and names == ["red", "green", "blue"]
    lines = (root / "labels" / "seq0_000.txt").read_text().splitlines()
    assert seqs[0].labels[0] == [(int(c), *map(float, xywh)) for c, *xywh in (ln.split() for ln in lines)]
    prepare.main(prepare.parse_args(["stvd", "--root", str(root), "--seq-pattern", pattern, "--inspect"]))
    out = capsys.readouterr().out
    assert "3 sequences" in out and "unknown" in out and "frames per sequence: min 12" in out

    data = convert("stvd", root, tmp_path, "--seq-pattern", pattern, "--names", "red", "green", "blue")
    train, val = build(data, "train"), build({**data, "train": data["val"]}, "val")
    assert len(train.sequences) == len(val.sequences) == 3 and len(train) == 30 and len(val) == 6


def test_adapter_visdrone_vid(tmp_path):
    """VisDrone-VID: ignored regions, 'others' and score-0 boxes dropped, class map, official split."""
    train = tmp_path / "vd" / "VisDrone2019-VID-train"
    write_frames(train / "sequences" / "uav0001_v", [f"{i:07d}.jpg" for i in range(1, 5)])
    (train / "annotations").mkdir()
    (train / "annotations" / "uav0001_v.txt").write_text(
        "1,0,10,10,20,10,1,1,0,0\n"  # pedestrian
        "1,1,0,0,30,30,0,0,0,0\n"  # ignored region
        "2,2,5,5,10,10,1,11,0,0\n"  # others
        "2,3,5,5,10,10,0,4,0,0\n"  # score 0
        "3,4,40,30,40,40,1,4,0,0,\n"  # car, clipped to the image, trailing comma
    )
    val = tmp_path / "vd" / "VisDrone2019-VID-val"
    write_frames(val / "sequences" / "uav0002_v", [f"{i:07d}.jpg" for i in range(1, 4)])
    (val / "annotations").mkdir()
    (val / "annotations" / "uav0002_v.txt").write_text("1,0,1,1,5,5,1,9,0,0\n")  # bus

    seqs, names, _ = adapter("visdrone-vid", tmp_path / "vd")
    assert [(s.name, s.split) for s in seqs] == [("uav0001_v", "train"), ("uav0002_v", "val")] and len(names) == 10
    assert [len(lb) for lb in seqs[0].labels] == [1, 0, 1, 0] and seqs[0].labels[0][0][0] == 0
    np.testing.assert_allclose(seqs[0].labels[2][0], (3, 52 / 64, 39 / 48, 24 / 64, 18 / 48))
    assert seqs[1].labels[0][0][0] == 8

    seqs, names, _ = adapter(
        "visdrone-vid", tmp_path / "vd", "--class-map", '{"1": 0, "4": 1}', "--names", "person", "car"
    )
    assert (
        [b[0] for lb in seqs[0].labels for b in lb] == [0, 1] and not any(seqs[1].labels) and names == ["person", "car"]
    )

    data = convert("visdrone-vid", tmp_path / "vd", tmp_path, "--split", "official")
    assert len(build(data, "train")) == 4 and len(build({**data, "train": data["val"]}, "val")) == 3


def test_adapter_uavdt(tmp_path):
    """UAVDT: M#### sequences with *_gt_whole.txt (ignore files unused), category map, --split seqs."""
    for seq in ("M0101", "M0201"):
        write_frames(tmp_path / "uavdt" / "UAV-benchmark-M" / seq, [f"img{i:06d}.jpg" for i in range(1, 4)])
    gt = tmp_path / "uavdt" / "UAV-benchmark-MOTD_v1.0" / "GT"
    gt.mkdir(parents=True)
    (gt / "M0101_gt_whole.txt").write_text("1,1,10,10,10,10,1,1,1\n2,1,10,10,10,10,1,1,3\n")
    (gt / "M0101_gt_ignore.txt").write_text("1,-1,0,0,60,40,-1,-1,-1\n")
    (gt / "M0201_gt_whole.txt").write_text("3,5,0,0,8,8,1,1,2\n")
    seqs, names, _ = adapter("uavdt", tmp_path / "uavdt")
    assert [s.name for s in seqs] == ["M0101", "M0201"] and names == ["car", "truck", "bus"]
    assert [[b[0] for b in lb] for lb in seqs[0].labels] == [[0], [2], []] and seqs[1].labels[2][0][0] == 1
    data = convert("uavdt", tmp_path / "uavdt", tmp_path, "--split", "seqs", "--val-seqs", "M0201")
    assert Path(data["val"]).name == "val" and (Path(data["val"]) / "M0201").is_dir()
    assert len(build({**data, "train": data["val"]}, "val")) == 3
    with pytest.raises(ValueError):  # UAVDT ships no split directories
        convert("uavdt", tmp_path / "uavdt", tmp_path, "--split", "official")


@pytest.mark.parametrize("bench", ["MOT17", "MOT20"])
def test_adapter_mot(tmp_path, capsys, bench):
    """MOT17/20: pedestrians only, evaluation flag and visibility filters, detector copies deduplicated, fps/motion."""
    root = tmp_path / bench
    seqs = ["MOT17-02-DPM", "MOT17-02-FRCNN", "MOT17-05-FRCNN"] if bench == "MOT17" else ["MOT20-01", "MOT20-02"]
    for name in seqs:
        d = root / "train" / name
        write_frames(d / "img1", [f"{i:06d}.jpg" for i in range(1, 4)])
        (d / "gt").mkdir()
        (d / "gt" / "gt.txt").write_text(
            "1,1,10,10,10,20,1,1,0.9\n"  # kept
            "1,2,30,10,10,20,1,1,0.1\n"  # low visibility
            "2,3,10,10,10,20,0,1,1.0\n"  # not evaluated
            "2,4,10,10,10,20,1,7,1.0\n"  # static person, not a pedestrian
        )
        (d / "seqinfo.ini").write_text("[Sequence]\nname=x\nimDir=img1\nframeRate=25\nseqLength=3\n")
    write_frames(root / "test" / seqs[0] / "img1", ["000001.jpg"])  # no gt.txt: skipped

    found, names, _ = adapter("mot", root, "--min-visibility", "0.5")
    assert len(found) == 2 and names == ["pedestrian"] and all(s.fps == 25 for s in found)
    assert [len(lb) for lb in found[0].labels] == [1, 0, 0]
    expected = ["static", "moving"] if bench == "MOT17" else ["static", "static"]
    assert [s.motion for s in found] == expected
    assert [s.name for s in found] == (["MOT17-02", "MOT17-05"] if bench == "MOT17" else seqs)
    assert [len(lb) for lb in adapter("mot", root)[0][0].labels] == [2, 0, 0]  # visibility 0.1 kept by default

    prepare.main(prepare.parse_args(["mot", "--root", str(root), "--inspect"]))
    out = capsys.readouterr().out
    assert "2 sequences" in out and "pedestrian=" in out and " 25 " in out
    data = convert("mot", root, tmp_path, "--val-ratio", "0.34")
    assert len(build(data, "train")) == 4 and len(build({**data, "train": data["val"]}, "val")) == 2


def test_estimate_motion(tmp_path):
    """Phase-correlation motion estimate separates a panning camera from a static one."""
    import cv2

    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (200, 600), dtype=np.uint8), (5, 5), 0)
    for kind, step in (("static", 0), ("moving", 6)):
        frames = []
        for i in range(12):
            frames.append(tmp_path / f"{kind}_{i:02d}.png")
            cv2.imwrite(str(frames[-1]), scene[:, i * step : i * step + 256])
        assert prepare.estimate_motion(frames) == f"{kind} (estimated)"


def test_run_matrix_fair_hyperparameters(tmp_path):
    """run_matrix trains every config with mosaic/mixup/cutmix/copy_paste/close_mosaic off and records them."""
    import csv

    yaml = make_synthetic_seq.make_dataset(tmp_path / "synth", num_seqs=2, num_frames=6)
    out = tmp_path / "matrix.csv"
    args = ["--data", str(yaml), "--models", "yolo26n", "--configs", "n1", "--imgsz", "64", "--epochs", "1"]
    args += ["--batch", "4", "--device", "cpu", "--seed", "0", "--workers", "0", "--scratch", "--iters", "1"]
    run_matrix.main(run_matrix.parse_args([*args, "--project", str(tmp_path / "runs"), "--out", str(out)]))
    with open(out) as f:
        (row,) = list(csv.DictReader(f))
    assert row["error"] == "" and row["dataset"] == "data" and row["channels"] == "3"
    assert all(float(row[k]) == 0 for k in run_matrix.FAIR_HYP) and row["albumentations"] == "disabled"
    train_args = YAML.load(Path(row["weights"]).parents[1] / "args.yaml")
    assert all(train_args[k] == v for k, v in run_matrix.FAIR_HYP.items())
    run_matrix.main(run_matrix.parse_args([*args, "--project", str(tmp_path / "runs"), "--out", str(out)]))
    with open(out) as f:
        assert len(list(csv.DictReader(f))) == 1  # completed experiments are skipped

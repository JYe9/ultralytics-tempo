# Tempo Stage 1: 多帧早期融合（Multi-frame Early Fusion）基础设施

本阶段让 YOLO26（以及未来同接口的 YOLO27）接受"多帧早期融合"输入：把 n 帧连续视频帧在通道维拼接为 `(3n, H, W)`（可选再加 1 个差分通道），只用最后一帧（目标帧 t）的标签监督。Head、loss、标签分配完全不变，仍是 YOLO26 端到端 NMS-free head。

参考：arXiv:2410.13616（STVD 数据集）、arXiv:2506.20550（多帧早期融合 YOLO）。

## 设计

```
data.yaml (temporal:) ──check_det_dataset──> data["channels"] = 3n(+1)
        │                                         │
        ▼                                         ▼
build_yolo_dataset ──> TemporalYOLODataset   DetectionModel(ch=3n(+1))  ← 已有的 channels 机制
        │  按序列分组/排序，窗口 [t-(n-1)s … t]    │  .pt 权重先 inflate_first_conv 再 load
        │  load_image() 逐帧读取（复用 RAM/disk 缓存）│
        │  拼成 (H, W, 3n) 后整体做一次增强           │
        ▼                                         ▼
 TemporalFormat: BGR→RGB(逐帧) + 差分通道 ───> 训练 / 验证 (mAP 不变)
                                                  │
                         TemporalPredictor: 环形缓冲区 (n-1)s+1 帧 → 同一组帧规则 → 目标帧结果
```

- **channels 复用**：Ultralytics 已原生支持数据集 YAML 的 `channels` 字段（`check_det_dataset` → `DetectionTrainer.get_model(ch=data["channels"])` → `model.yaml["channels"]` → AutoBackend / exporter 都读它）。因此只需在 `check_det_dataset` 中由 `temporal` 推导 `channels = 3n + diff`，无需改模型构建。
- **增强一致性**：所有帧先拼成一张 `(H, W, 3n)` 的多通道图，再走原有的 `RandomPerspective`（缩放/平移/旋转/剪切）、`RandomFlip`、`LetterBox`，因此几何参数天然完全相同。HSV 原实现只处理 3 通道，`TemporalHSV` 对每一帧重放同一个随机状态，保证所有帧色彩增益一致。
- **禁止跨时刻/跨序列混合**：训练时强制 `mosaic=mixup=cutmix=copy_paste=0` 并打印警告。
- **Albumentations 统一禁用**：temporal 模式下（包括 n=1）训练管线不含 Albumentations，进程内打印一次提示，不需要手动卸载。
- **差分通道**由最后一个变换 `TemporalFormat` 计算：此时所有几何与光度增强（含 HSV）都已作用于各帧，差分取增强后的 `|gray(t) - gray(t-s)|`，差分通道本身不经过 HSV 或任何增强；letterbox/仿射填充区域的差分为 0，训练与推理一致（`test_diff_channel_after_photometric_augmentation`）。

## 配置字段（数据集 YAML 的 `temporal:`）

```yaml
temporal:
    max_frames: 5 # K：槽位数，输入通道固定为 3K(+1)
    train_frames: [1, 2, 3, 5] # 训练时每个样本随机采样的 n
    train_strides: [1, 2, 3] # 训练时每个样本随机采样的 stride
    num_frames: 3 # 验证/推理默认的 n（运行时可改，<= K）
    stride: 1 # 验证/推理默认的 stride，按"序列内可用帧的位置"计数
    slot_fill: zero # n < K 时较早槽位的填充：zero（置零）或 repeat（重复最早使用的帧）
    cond: none # film：用 (n, stride) 对 stem 输出做 FiLM
    seq_pattern: '^(?P<seq>.+)_(?P<idx>\d+)\.(png|jpg)$' # 文件名正则，必须有 idx 组；seq 组可选
    diff_channel: false # true 时追加 1 个 |gray(t) - gray(t-stride)| 差分通道
    # frame_order: rgb_frames_earliest_first_target_last_then_diff  # 自动写入，标识下方的通道布局
```

| 字段            | 默认           | 含义                                                                                                                            |
| --------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `max_frames`    | 所配置的最大 n | 槽位数 K；输入通道 `3K(+1)`                                                                                                     |
| `train_frames`  | `[num_frames]` | 训练时每个样本均匀随机采样的 n（同一 batch 内可不同）                                                                           |
| `train_strides` | `[stride]`     | 训练时每个样本均匀随机采样的 stride，与 n 独立                                                                                  |
| `num_frames`    | `max_frames`   | 验证/推理默认的 n；只配置 `num_frames` 时即阶段 1 的固定帧模式（K=n），`n=1` 时与原版 YOLO 行为逐位一致                         |
| `stride`        | `1`            | 验证/推理默认的 stride；窗口为 `[t-(n-1)s, …, t-s, t]`                                                                          |
| `slot_fill`     | `zero`         | n<K 时前 K-n 个槽位：`zero` 在全部增强之后置零；`repeat` 重复最早使用的帧                                                       |
| `cond`          | `none`         | `film`：FiLM 条件编码，见下文                                                                                                   |
| `seq_pattern`   | `null`         | 文件名正则；`null` 表示"每个子目录是一段视频"，帧号取文件名中最后一个整数。序列 ID 总是包含父目录，因此不同目录永远不会混在一起 |
| `diff_channel`  | `false`        | 追加差分通道，放在最末；任何 n（包括 1）都用 t 与 t-stride 计算                                                                 |

阶段 1 的 `random_frames: true` 仍可读取，等价于 `train_frames: [1..K]`（置零填充）；该字段已被上面的字段取代。

- 不要再手写 `channels:`；若写了且与 `3n(+1)` 不一致会报错。
- 序列开头不足 `(n-1)s` 帧时，用该序列最早可用帧补齐；不会跨序列取帧。不匹配 `seq_pattern` 的图片当作 1 帧序列并打印警告数量。

**帧顺序约定**：K 个槽位，通道 `[0:3]` 是最早的槽位，`[3(K-1):3K]` 是目标帧 t（始终在最后一个槽位），n 帧占据最后 n 个槽位，开启差分时最后 1 个通道是差分；每帧均为 RGB。该约定以 `frame_order` 字段随配置保存，布局不一致的配置会被拒绝。

## 模型

- `inflate_first_conv(model, n, diff_channel)`（`ultralytics/models/yolo/detect/temporal.py`）：RGB 权重沿输入通道复制 n 次再乘 `1/n`，差分通道权重为 0。n 帧完全相同且差分为 0 时，第一层（乃至整个模型）的输出与原 3 通道模型一致（`tests/test_temporal.py::test_inflate_first_conv`）。
- 训练入口：`DetectionTrainer.get_model` 发现 `data["temporal"]` 时，会先对 3 通道预训练权重（`yolo26n.pt` 等）的深拷贝调用 `inflate_first_conv`，再交给 `BaseModel.load`。此时首层形状已经一致，`BaseModel.load` 自带的"首层通道不匹配"分支（只把 RGB 滤波器原样拷进前 3 个通道、其余保持随机初始化）不会触发，最终生效的是 `inflate_first_conv` 的结果（`test_trainer_inflates_pretrained_pt` 从随机初始化 3 通道模型保存的 .pt 出发，走真实的 `setup_model → get_model` 路径验证）。
- 元数据：`temporal` 配置（max_frames、num_frames、stride、train_frames、train_strides、slot_fill、cond、diff_channel、seq_pattern、frame_order）写进 `model.yaml["temporal"]`，随 checkpoint 保存；导出时写入导出模型 metadata 的 `temporal` 字段（ONNX 等格式均适用）。

## 阶段 2a：任意帧数（一次训练，运行时切换 n 与 stride）

- **训练**：每个样本独立采样 `n ∈ train_frames`、`stride ∈ train_strides`，batch 中以 `frame_cond`（B×2 张量）携带。为了让 diff 对任何 n 都是 `|t − (t−stride)|`，t−stride 这一帧作为额外 3 通道随其它帧一起做同一次增强，在 `TemporalFormat` 中算出差分后丢弃。`train_frames`/`train_strides` 只有一个取值时不消耗随机数，阶段 1 的训练逐位不变。
- **FiLM（`cond: film`）**：`FrameFiLM` 把 `(n/K, s/S, (n−1)s/((K−1)S))` 经 MLP 映射为逐通道 `(scale, shift)`，作用在 stem 第二层卷积（`model.model[1]`）的输出上：`x·(1+scale)+shift`。最后一层零初始化，所以初始时严格恒等（逐位相等），从阶段 1 权重（相同 K）加载后行为不变。模块挂在 `model.film`，现有 state-dict 键名不变，通过 forward hook 生效：训练时模型的 pre-hook 从 batch 读取 `frame_cond`，loss 计算后释放；没有 `cond` 时使用 `default`（即 `num_frames, stride`）。
- **导出：(n, stride) 作为编译期常量**。理由：
    1. Ultralytics 的 AutoBackend 和 20 多种导出后端都假设单一图像输入，增加第二个输入需要改动每个后端以及 metadata 处理；
    2. TensorRT/TFLite/CoreML 等运行时对单输入图最简单、最稳定；
    3. 部署通常固定在一个工作点上，切换只需重新导出（秒级），不需要重训；
    4. `cond: none` 的模型本身就能在导出后运行时切换 (n, stride)（输入通道恒为 3K），只有 FiLM 模型受此限制。

用 `set_frame_cond(model, n, s)` 选择要固化的 (n, s)，它同时写入导出 metadata；导出后的 FiLM 模型被要求以其它 (n, s) 运行时会直接报错。

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect.temporal import TemporalPredictor, TemporalValidator, set_frame_cond

model = YOLO("best.pt")
model.val(data="data.yaml", validator=TemporalValidator.at(2, 3))  # n=2, stride=3
model.predict("video.mp4", predictor=TemporalPredictor.at(1, 1))
set_frame_cond(model.model, 2, 1)
model.export(format="onnx")  # FiLM 固化为 (2, 1)
```

`TemporalPredictor.at(n, s)` / `TemporalValidator.at(n, s)` 返回按参数缓存的子类，因此逐帧多次调用 `predict()` 时会复用同一个 predictor 及其环形缓冲区（`functools.partial` 做不到这一点：`Model.predict` 遇到类型不同的 predictor 会重建）。max_frames、diff_channel、slot_fill、cond 等布局参数总是从模型 metadata 读取。

`scripts/tempo/eval_anyframe.py` 对一个模型遍历所有 (n, stride)（默认 n=1..K、stride 取训练集合），输出 CSV：mAP50、mAP50-95、每类 AP、延迟，以及该组合是否在训练中出现过（`trained`）；列与 `matrix.csv` 一致，可与阶段 1 的固定帧模型逐项对比。

```bash
python scripts/tempo/eval_anyframe.py --model runs/detect/train/weights/best.pt --data data/visdrone_vid_tempo.yaml --imgsz 640 --device 0
```

## 推理与导出

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect.temporal import TemporalPredictor

model = YOLO("runs/detect/train/weights/best.pt")
for r in model.predict("video.mp4", predictor=TemporalPredictor, stream=True):
    ...  # r 对应当前（目标）帧

# 逐帧在线推理：每次传入一帧 numpy 图像，环形缓冲区在多次调用之间保留
for frame in frames:
    r = model.predict(frame, predictor=TemporalPredictor, verbose=False)[0]

# 导出模型的 metadata 带 temporal 配置，无需 data=
YOLO("best.onnx", task="detect").predict("video.mp4", predictor=TemporalPredictor)
```

- `TemporalPredictor` 维护长度 `(n-1)s+1` 的环形缓冲区，组帧规则与 `TemporalYOLODataset` 完全一致（`test_predictor_matches_dataset` 逐像素验证）。视频/流在源路径变化时重置；图片文件按 `seq_pattern`（或目录）判断序列变化并重置；无法识别序列的名称（例如内存中的 numpy 帧）视为同一个连续流。
- 配置来源优先级：模型 metadata（checkpoint 的 `model.yaml["temporal"]` 或导出模型 metadata）> `data=` YAML。只有模型不带配置时才读 `data=`；两者的 num_frames/stride/diff_channel 不一致时打印警告并使用模型 metadata。
- ONNX 导出：`exporter` 已使用 `model.yaml["channels"]` 生成 `(1, 3n(+1), H, W)` 的 dummy 输入，只额外写入 `temporal` metadata；导出后用 ONNX Runtime + `TemporalPredictor`（不传 `data=`）推理已在 CPU 上验证。

## 本地运行步骤（GPU 机器）

```bash
# 0. 安装（仓库根目录，激活虚拟环境）
pip install -e ".[dev]"

# 1. CPU 自测（合成数据，几十秒）
pytest tests/test_temporal.py -v

# 2. 先 inspect 数据集：序列数、每序列帧数、帧率、相机是否运动、各类实例数（只打印，不写文件）
python scripts/tempo/prepare_seq_dataset.py stvd --root /path/to/STVD --inspect --seq-pattern '^(?P<seq>.+)_(?P<idx>\d+)\.jpg$'
python scripts/tempo/prepare_seq_dataset.py visdrone-vid --root /path/to/VisDrone --inspect --estimate-motion
python scripts/tempo/prepare_seq_dataset.py uavdt --root /path/to/UAVDT --inspect
python scripts/tempo/prepare_seq_dataset.py mot --root /path/to/MOT17 --inspect --min-visibility 0.25

# 3. 转换（默认每段序列前 80% 帧训练、后 20% 验证），生成 data/<adapter>_tempo.yaml
python scripts/tempo/prepare_seq_dataset.py stvd --root /path/to/STVD --seq-pattern '<确认的正则>' --names <类别名...>
python scripts/tempo/prepare_seq_dataset.py visdrone-vid --root /path/to/VisDrone --split official
python scripts/tempo/prepare_seq_dataset.py uavdt --root /path/to/UAVDT --split seqs --val-seqs <测试序列...>
python scripts/tempo/prepare_seq_dataset.py mot --root /path/to/MOT17 --min-visibility 0.25 --out-yaml data/mot17_tempo.yaml

# 4. 实验矩阵（数据集 × yolo26n/yolo26s × 7 种配置），结果追加到 results/matrix.csv，已完成的实验自动跳过
python scripts/tempo/run_matrix.py --data data/stvd_tempo.yaml data/mot17_tempo.yaml --imgsz 640 --epochs 100 --batch 16 --device 0 --seed 0

# 4'. SLURM：每个数组任务跑一个配置，共享同一个（加锁的）CSV
sbatch --array=0-6 --gres=gpu:1 --export=ALL,GPUS=1,DATASETS="data/stvd_tempo.yaml" scripts/tempo/slurm_matrix.sbatch
```

### 数据集适配器（`scripts/tempo/prepare_seq_dataset.py`）

| 适配器         | 输入布局                                                                                | 规则                                                                                                                            |
| -------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `stvd`         | 图片 + YOLO txt 标签（`images/`↔`labels/`）                                             | 序列按 `--seq-pattern` 或目录识别；`train/val/test` 目录作为官方划分                                                            |
| `visdrone-vid` | `VisDrone2019-VID-{train,val,test-dev}/sequences/<seq>/*.jpg` + `annotations/<seq>.txt` | 丢弃 ignored regions（类别 0）、others（类别 11）和 score=0 的框；`--class-map` 可配置类别映射；官方划分来自目录名              |
| `uavdt`        | `UAV-benchmark-M/M####/img*.jpg` + `GT/M####_gt_whole.txt`                              | 类别 1 car、2 truck、3 bus；`_gt_ignore.txt` 不使用；官方 train/test 列表随 UAVDT 工具包发布，用 `--split seqs --val-seqs` 指定 |
| `mot`          | `train/<seq>/{img1, gt/gt.txt, seqinfo.ini}`（MOT17/MOT20）                             | 只取行人类（class 1）、评估标志为 1、`visibility >= --min-visibility`；MOT17 的 DPM/FRCNN/SDP 副本去重；帧率读自 seqinfo.ini    |

- 输出：`<out-dir>/images/<split>/<seq>/<帧号:06d>.<ext>`（默认符号链接，`--copy` 复制）和对应的 `labels/` YOLO txt，YAML 中 `seq_pattern: null`（每个目录一段序列）。
- 划分：`--split temporal`（默认，按 `--val-ratio`）、`official`（数据集自带的划分目录）、`seqs`（`--val-seqs` 指定的整段序列作验证）。
- 相机运动：MOT17/MOT20 用 MOTChallenge 公布的信息（MOT17-05/06/07/10/11/12/13/14 为运动相机，其余及 MOT20 为静止），其它数据集标为 `unknown`；`--estimate-motion` 用帧间相位相关估计未知项，标注为 `(estimated)`。

### 公平性（`run_matrix.py`）

所有配置（包括 n=1）统一使用 `mosaic=0, mixup=0, cutmix=0, copy_paste=0, close_mosaic=0`，temporal 数据集同时禁用 Albumentations；这些超参与 `albumentations=disabled` 一起写入 `matrix.csv` 的每一行。`--data` 可以给多个 YAML，`dataset` 列记录 YAML 文件名（不含扩展名）。

`run_matrix.py` 配置名：`n1`、`n2s1`、`n2s1d`（+差分）、`n3s1`、`n3s2`、`n3s3`、`n5s2`；`--configs`/`--models` 可选子集，`--scratch` 从 YAML 随机初始化训练。CSV 其它列：mAP50、mAP50-95、每类 AP50 和 AP50-95（JSON）、参数量、GFLOPs、batch=1 前向延迟（median，ms）、验证时每张图推理耗时。CSV 列与脚本不一致（例如旧版本生成的文件）时会报错，请换一个 `--out`。

`slurm_matrix.sbatch` 通过环境变量参数化：`DATASETS`、`MODELS`、`CONFIGS`、`GPUS`（需与 `--gres=gpu:N` 一致，>1 时 DDP）、`IMGSZ`、`EPOCHS`、`BATCH`、`SEED`、`WORKERS`、`PROJECT`、`OUT`、`EXTRA`、`VENV`；作为数组任务时第 i 个任务只跑 `CONFIGS` 的第 i 项。

单次训练也可以直接用 CLI：

```bash
yolo train model=yolo26n.pt data=data/stvd_tempo.yaml imgsz=640 epochs=100 device=0
yolo val model=runs/detect/train/weights/best.pt data=data/stvd_tempo.yaml
```

## 已知限制

1. **步长按序列内位置计数**：`stride` 按排序后序列中的位置计算，不按帧号做算术；帧号有缺口时（抽帧数据）窗口跨越的真实时间会变大。
2. **验证段开头补齐**：`prepare_seq_dataset.py` 按时间切分后，验证段是独立的序列，它的前 `(n-1)s` 帧用验证段第一帧补齐，不会从训练段借上下文帧。
3. **训练可视化**：`train_batch*.jpg` 等图只画前 3 个通道，即最早帧，而框属于目标帧；仅影响可视化。
4. **推理需显式指定 `predictor=TemporalPredictor`**：默认 `DetectionPredictor` 不会组帧，会因通道数不匹配报错。对图片目录推理时，加载器按文件名字典序排序，帧号未补零时请传入已排序的文件列表。
5. **只接入了 detect 任务**：`segment/pose/obb` 的 Trainer 覆盖了 `get_model`，不会自动 inflate 预训练权重（数据集本身可用）。
6. **FiLM 通过 hook 和模块状态（`film.cond`）传递 (n, s)**：同一个模型对象不要在多个线程里同时以不同的 (n, s) 推理；TorchScript `script` 不支持（导出使用 tracing，已验证 ONNX）。
7. **缓存**：RAM/disk 缓存按帧生效；训练时的图像缓冲区（`max_buffer_length`）按帧计数，多帧时实际覆盖的样本更少。
8. **未在 GPU / DDP / SLURM 上运行过**：所有验证都在 CPU、合成数据上完成；四个适配器只在按公开格式构造的小样本上测试过，真实数据请先跑 `--inspect` 核对。
9. **UAVDT 官方划分**未内置（需 `--val-seqs`）；VisDrone 的 ignored regions 默认按帧用灰度 114 遮盖（`--mask-value`、`--no-mask-ignored` 可调），被遮盖的帧会重新编码为 JPEG（质量 95），其余帧仍是符号链接；UAVDT 的 `_gt_ignore.txt` 没有遮盖。
10. **延迟与 n 无关**：输入通道固定为 3K，不同 n 的模型前向计算量相同；`eval_anyframe.py` 里不同 n 的延迟差异只来自测量噪声和预处理。
11. **K 必须一致才能从阶段 1 权重加载**：`inflate_first_conv` 只扩展 3 通道权重；阶段 1 的 3n 通道权重需要 `max_frames = n`。

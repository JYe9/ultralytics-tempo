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
- **差分通道**在增强之后、BGR→RGB 之前计算（`|gray(t) - gray(t-s)|`），因此 letterbox/仿射填充区域的差分为 0，训练与推理一致。

## 配置字段（数据集 YAML 的 `temporal:`）

```yaml
temporal:
    num_frames: 3 # n，同时也是槽位数上限 K
    stride: 1 # s，按"序列内可用帧的位置"计数
    seq_pattern: '^(?P<seq>.+)_(?P<idx>\d+)\.(png|jpg)$' # 文件名正则，必须有 idx 组；seq 组可选
    diff_channel: false # true 时追加 1 个 t 与 t-s 的灰度绝对差通道（需要 n >= 2）
    random_frames: false # 预留：训练时随机 n∈[1,K]，最早的未用槽位置零（目标帧始终在最后）
```

| 字段            | 默认    | 含义                                                                                                                            |
| --------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `num_frames`    | `1`     | 输入帧数 n；`n=1` 时与原版 YOLO 行为逐位一致                                                                                    |
| `stride`        | `1`     | 帧步长 s；窗口为 `[t-(n-1)s, …, t-s, t]`                                                                                        |
| `seq_pattern`   | `null`  | 文件名正则；`null` 表示"每个子目录是一段视频"，帧号取文件名中最后一个整数。序列 ID 总是包含父目录，因此不同目录永远不会混在一起 |
| `diff_channel`  | `false` | 追加差分通道，输入通道 `3n+1`，差分在最末                                                                                       |
| `random_frames` | `false` | 为后续"每个样本帧数可变"预留；只在训练时生效，验证/推理总是用满 K 帧                                                            |

- 不要再手写 `channels:`；若写了且与 `3n(+1)` 不一致会报错。
- 序列开头不足 `(n-1)s` 帧时，用该序列最早可用帧补齐；不会跨序列取帧。不匹配 `seq_pattern` 的图片当作 1 帧序列并打印警告数量。

**帧顺序约定**：通道 `[0:3]` 是最早帧，`[3(n-1):3n]` 是目标帧 t，开启差分时最后 1 个通道是差分；每帧均为 RGB。

## 模型

- `inflate_first_conv(model, n, diff_channel)`（`ultralytics/models/yolo/detect/temporal.py`）：RGB 权重沿输入通道复制 n 次再乘 `1/n`，差分通道权重为 0。n 帧完全相同且差分为 0 时，第一层（乃至整个模型）的输出与原 3 通道模型一致（`tests/test_temporal.py::test_inflate_first_conv`）。
- 训练入口：`DetectionTrainer.get_model` 发现 `data["temporal"]` 时，会先对 3 通道预训练权重（`yolo26n.pt` 等）的深拷贝调用 `inflate_first_conv` 再加载，并把 `temporal` 配置写进 `model.yaml["temporal"]`（随 checkpoint 保存，供推理使用）。

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

# 导出的模型不带 temporal 配置，需要通过 data= 指定
YOLO("best.onnx", task="detect").predict("video.mp4", predictor=TemporalPredictor, data="data/stvd_tempo.yaml")
```

- `TemporalPredictor` 维护长度 `(n-1)s+1` 的环形缓冲区，组帧规则与 `TemporalYOLODataset` 完全一致（`test_predictor_matches_dataset` 逐像素验证）。视频/流在源路径变化时重置；图片文件按 `seq_pattern`（或目录）判断序列变化并重置；无法识别序列的名称（例如内存中的 numpy 帧）视为同一个连续流。
- ONNX 导出：无需改动，`exporter` 已使用 `model.yaml["channels"]` 生成 `(1, 3n(+1), H, W)` 的 dummy 输入；导出后用 ONNX Runtime + `TemporalPredictor` 推理已在 CPU 上验证。

## 本地运行步骤（GPU 机器）

```bash
# 0. 安装（仓库根目录，激活虚拟环境）
pip install -e ".[dev]"

# 1. CPU 自测（合成数据，几十秒）
pytest tests/test_temporal.py -v

# 2. 先检查 STVD 目录结构和命名，确认序列识别规则
python scripts/tempo/prepare_stvd.py --root /path/to/STVD --inspect
python scripts/tempo/prepare_stvd.py --root /path/to/STVD --inspect --seq-pattern '^(?P<seq>.+)_(?P<idx>\d+)\.jpg$'

# 3. 生成 data/stvd_tempo.yaml（每段序列前 80% 帧训练、后 20% 验证）
python scripts/tempo/prepare_stvd.py --root /path/to/STVD --seq-pattern '<上一步确认的正则>' --names <类别名...>

# 4. 跑实验矩阵（yolo26n/yolo26s × 7 种配置），结果追加到 results/matrix.csv，已完成的实验会自动跳过
python scripts/tempo/run_matrix.py --data data/stvd_tempo.yaml --imgsz 640 --epochs 100 --batch 16 --device 0 --seed 0
```

`prepare_stvd.py` 要求标签已经是 YOLO txt，且路径满足 Ultralytics 规则（把路径中最后一个 `/images/` 换成 `/labels/`，扩展名换成 `.txt`）；`--inspect` 会打印标签覆盖率。如果 STVD 原始标注是其它格式，需要先转换。`--image-dir` 可以把搜索限制在某个名字的目录下。

`run_matrix.py` 配置名：`n1`、`n2s1`、`n2s1d`（+差分）、`n3s1`、`n3s2`、`n3s3`、`n5s2`；`--configs`/`--models` 可选子集，`--scratch` 从 YAML 随机初始化训练。CSV 列：mAP50、mAP50-95、每类 AP50 和 AP50-95（JSON）、参数量、GFLOPs、batch=1 前向延迟（median，ms）、验证时每张图推理耗时等。

单次训练也可以直接用 CLI：

```bash
yolo train model=yolo26n.pt data=data/stvd_tempo.yaml imgsz=640 epochs=100 device=0
yolo val model=runs/detect/train/weights/best.pt data=data/stvd_tempo.yaml
```

## 已知限制

1. **步长按序列内位置计数**：`stride` 按排序后序列中的位置计算，不按帧号做算术；帧号有缺口时（抽帧数据）窗口跨越的真实时间会变大。
2. **验证段开头补齐**：`prepare_stvd.py` 按时间切分后，验证段是独立的序列，它的前 `(n-1)s` 帧用验证段第一帧补齐，不会从训练段借上下文帧。
3. **Albumentations**：若环境装了 `albumentations`，其默认增强（概率均为 1%）只支持 1/3 通道，`n>1` 时会被跳过，而 `n=1` 会执行。对比实验如需严格一致，可卸载 `albumentations`。
4. **训练可视化**：`train_batch*.jpg` 等图只画前 3 个通道，即最早帧，而框属于目标帧；仅影响可视化。
5. **推理需显式指定 `predictor=TemporalPredictor`**：默认 `DetectionPredictor` 不会组帧，会因通道数不匹配报错。对图片目录推理时，加载器按文件名字典序排序，帧号未补零时请传入已排序的文件列表。
6. **只接入了 detect 任务**：`segment/pose/obb` 的 Trainer 覆盖了 `get_model`，不会自动 inflate 预训练权重（数据集本身可用）。
7. **`random_frames`** 只实现了"槽位置零"的数据接口；模型侧尚无显式的帧数输入，留给后续阶段。
8. **缓存**：RAM/disk 缓存按帧生效；训练时的图像缓冲区（`max_buffer_length`）按帧计数，多帧时实际覆盖的样本更少。
9. **未在 GPU / DDP 上运行过**：所有验证都在 CPU、合成数据上完成。

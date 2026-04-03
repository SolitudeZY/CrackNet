# U-Net 裂缝语义分割

基于 EfficientNet-B0 + U-Net 的像素级裂缝分割方案，替代 YOLO-seg 粗糙的 proto mask，用于精确裂缝宽度测量。

## 为什么不用 YOLO-seg

YOLO-seg 的分割原理是在 `imgsz/4` 分辨率（416 输入下仅 104x104）生成 32 个原型掩码，再线性组合。对于宽度仅 1-5 像素的裂缝，这种低分辨率 mask **无法描绘精细轮廓**，导致掩码"画胖"。

本方案使用 U-Net **直接输出 416x416 全分辨率掩码**，通过 skip connection 保留每一级空间细节。

## 环境依赖

已有环境（PyTorch 2.10+、torchvision 0.25+、Pillow、numpy、tqdm）之外，仅需安装：

```bash
pip install scipy
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `convert_masks.py` | 将 YOLO 多边形标注转为二值 mask PNG（一次性预处理） |
| `model.py` | U-Net 模型定义（EfficientNet-B0 encoder + 4 级 decoder） |
| `train.py` | 训练脚本（Dataset、BCE+Dice loss、验证、checkpoint） |
| `predict.py` | 推理、可视化叠加、裂缝宽度测量 |

## 快速开始

### 1. 生成二值 mask（仅首次运行）

```bash
cd /home/fs-ai/YOLO-crack-detection/scripts/unet_seg
python convert_masks.py
```

读取 `dataset/ultralytics/labels/` 下的 YOLO 多边形标注，输出到 `dataset/ultralytics/masks/`。约 30 秒完成，生成 4029 张 PNG。

### 2. 训练

```bash
python train.py
```

训练日志实时打印到终端，格式如下：

```
Epoch | Train Loss |   Val Loss |  Val IoU |         LR
------------------------------------------------------------
    1 |     1.2345 |     1.1234 |   0.1500 |   0.000100
    2 |     0.9876 |     0.8765 |   0.2800 |   0.000200
   ...
```

训练产出：
- `checkpoints/best.pt` — 最佳模型（按 val IoU）
- `checkpoints/last.pt` — 最新 checkpoint（可用于断点续训）
- `training_log.csv` — 完整训练日志

**主要超参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 100 | 带 early stopping |
| batch_size | 16 | RTX 5060 Ti 16GB 安全值 |
| encoder_lr | 1e-4 | 预训练权重微调，慢 10 倍 |
| decoder_lr | 1e-3 | 随机初始化，正常学习率 |
| patience | 20 | 连续 20 epoch 无提升则停止 |
| loss | BCE + Dice | 处理裂缝像素占比低的类不平衡 |
| AMP | 开启 | 混合精度，节省显存加速训练 |

预计训练时间约 30-40 分钟（RTX 5060 Ti）。

### 3. 测试集评估

```bash
python predict.py --test
```

对 112 张测试图计算 IoU 并生成可视化叠加图，输出到 `predictions/`。

### 4. 单张推理

```bash
python predict.py --image /path/to/image.jpg
```

输出 `predictions/` 下两个文件：
- `{name}_mask.png` — 二值掩码（黑白）
- `{name}_overlay.jpg` — 红色半透明叠加可视化

### 5. 裂缝宽度测量

```bash
python predict.py --image /path/to/image.jpg --measure
```

输出示例：

```
--- Crack Width (pixels) ---
  Max:    12.4 px
  Mean:   5.2 px
  Median: 4.8 px
```

测量原理：对二值 mask 做距离变换（每个像素到最近边界的距离），再骨架化提取裂缝中心线，骨架上每点的距离值 × 2 即为该点的局部裂缝宽度。

> 如需转换为物理尺寸（mm），将像素宽度乘以标定系数（mm/pixel）即可。

## 模型架构

```
输入 (B, 3, 416, 416)
    │
    ▼ EfficientNet-B0 Encoder (pretrained ImageNet)
    ├── features[1]: 16ch,  208×208  ──── skip ────┐
    ├── features[2]: 24ch,  104×104  ──── skip ──┐ │
    ├── features[3]: 40ch,   52×52   ──── skip ┐ │ │
    ├── features[5]: 112ch,  26×26   ── skip ┐ │ │ │
    └── features[7]: 320ch,  13×13 (bottleneck)│ │ │ │
                                               │ │ │ │
    ▼ Decoder                                  │ │ │ │
    ├── d4: 320→128, 13→26   ◄── concat ──────┘ │ │ │
    ├── d3: 128→64,  26→52   ◄── concat ────────┘ │ │
    ├── d2: 64→32,   52→104  ◄── concat ──────────┘ │
    ├── d1: 32→16,  104→208  ◄── concat ────────────┘
    └── head: 16→1, 208→416
    │
    ▼
输出 (B, 1, 416, 416) logits
```

总参数：4,357,453（encoder 3.6M + decoder 0.76M）

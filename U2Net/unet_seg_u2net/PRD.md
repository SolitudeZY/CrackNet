# PRD: U2Net 裂缝语义分割

## 1. 项目概述

### 1.1 背景
基于 DeepCrack 论文的 CrackTree260 数据增强方法，已获得 35,100 张训练数据（512x512）。需要使用 U2-Net（Qin et al., Pattern Recognition 2020）架构进行路面裂缝的二值语义分割，并部署到 GPU 受限环境。

### 1.2 目标
- 使用 U2NETP（轻量版，~4.7MB）进行裂缝语义分割训练
- 提供完整的训练、验证、测试和 ONNX 导出流程
- 模型需适配 GPU 受限的推理环境

### 1.3 参考资源
- **U2Net 代码**: `/home/fs-ai/YOLO-crack-detection/U-2-Net/`
- **数据集**: `/home/fs-ai/YOLO-crack-detection/dataset/deepcrack/CrackTree260_augmented/`
- **测试集**: CRKWH100, CrackLS315, Stone331（位于 deepcrack 目录下）
- **现有 v3 脚本**: `/home/fs-ai/YOLO-crack-detection/scripts/unet_seg_v3/`（参考模式）

---

## 2. 数据规格

### 2.1 训练数据
| 属性 | 值 |
|------|------|
| 总量 | 35,100 张（260 原始 x 135 增强） |
| 图像格式 | JPG, 512x512, RGB |
| 掩码格式 | PNG, 512x512, 8-bit 灰度 |
| 目录结构 | `images/` + `masks/` |
| 命名规则 | `{源ID}_r{角度}_f{翻转}__{裁剪位置}.jpg/png` |
| 总大小 | ~5.8GB (images) + ~149MB (masks) |

### 2.2 增强策略（已完成）
- 9 个旋转角度（0°-80°, 间隔10°）
- 3 种翻转（水平/垂直/无翻转）
- 5 个裁剪位置（4 角 + 中心），大小 512x512

### 2.3 测试集
- **CRKWH100**: 100 张线阵相机路面图像
- **CrackLS315**: 315 张激光照明路面图像
- **Stone331**: 331 张石材表面图像

### 2.4 数据划分方案
**决策**: 由于 35,100 张全部来自 260 张原始图像的增强，同一原始图像的不同增强版本之间高度相关。

**方案**: 按原始图像划分（非按增强图像随机划分），避免数据泄露。
- 训练集: 234 张原始图像 → ~31,590 张增强 (90%)
- 验证集: 26 张原始图像 → ~3,510 张增强 (10%)
- 测试集: 使用外部测试集（CRKWH100, CrackLS315, Stone331）

---

## 3. 模型选择

### 3.1 决策分析：U2NET vs U2NETP

| 特性 | U2NET (Full) | U2NETP (Lite) |
|------|-------------|---------------|
| 模型大小 | 173.6 MB | **4.7 MB** |
| 通道配置 | 64→128→256→512 | 64→64→64→64 |
| 参数量 | ~44M | **~1.1M** |
| 适用场景 | 高精度服务器端 | **边缘/受限GPU** |

**选择: U2NETP**（轻量版），原因：
1. GPU 受限环境要求小模型
2. 4.7MB 模型大小适合部署
3. 裂缝分割是二值任务，不需要超大模型容量
4. 可通过训练策略弥补模型容量不足

### 3.2 U2NETP 架构概要
```
输入(3ch) → Encoder(6 stages with RSU blocks) → Decoder(5 stages)
         → 6 个 Side Output + 1 个 Fused Output
         → Sigmoid → 二值掩码
```

- 所有 stage 均使用 64 通道（轻量化关键）
- RSU 模块提供多尺度特征提取（嵌套 U 结构）
- RSU4F 使用空洞卷积替代池化（最深层）

---

## 4. 技术决策（Grill-Me 审问）

### 4.1 输入分辨率
**问题**: U2Net 默认使用 320x320，但数据是 512x512。选择哪个？

**分析**:
- 320x320: 原始 U2Net 设计分辨率，计算开销小，内存占用低
- 512x512: 保留全部细节，但计算量增加 2.56 倍
- 裂缝是细线状结构，分辨率降低可能丢失关键细节

**决策**: 使用 **320x320** 作为默认训练分辨率。
- 理由：GPU 受限环境下需要控制计算量；U2NETP 的多尺度 RSU 结构本身就能捕获不同尺度的特征
- 推理时可通过滑动窗口处理更大图像

### 4.2 损失函数
**问题**: 使用原始 U2Net 的 Multi-BCE 还是 v3 中的 Focal Tversky + OHEM + Boundary？

**分析**:
- Multi-BCE: U2Net 原始设计，简单有效，对 7 个输出分别计算 BCE
- Focal Tversky: 更适合处理类别不平衡（裂缝像素远少于背景）
- Boundary Loss: 有助于改善边缘质量

**决策**: 使用 **混合损失 = Multi-scale BCE + Dice Loss**
- 理由：BCE 保持 U2Net 多尺度监督的原始设计；加入 Dice Loss 处理正负样本不平衡；不过度复杂化，保持训练稳定性
- 权重：`total_loss = Σ(BCE_i + Dice_i) for i in [d0..d6]`，d0 权重加倍

### 4.3 优化器与学习率
**问题**: 使用 U2Net 原始 Adam(lr=0.001) 还是 v3 的 AdamW + Cosine？

**决策**: 使用 **AdamW + Cosine Annealing with Warmup**
- 初始 lr=1e-3, weight_decay=1e-4
- Warmup 3 epochs
- 理由：AdamW 的权重衰减解耦更稳定；Cosine 调度器已在 v3 中验证有效

### 4.4 数据增强（在线）
**问题**: 数据已经离线增强了 135 倍，还需要在线增强吗？

**决策**: **轻度在线增强**
- 保留：RandomBrightnessContrast, CLAHE, GaussNoise（像素级，不改变几何）
- 不使用几何增强（已经离线做了旋转/翻转/裁剪）
- 使用 albumentations 库（与 v3 一致）

### 4.5 训练 Epoch 数
**问题**: U2Net 原始训练 100K epoch，v3 使用 300 epoch。

**决策**: **100 epochs**，Early Stopping patience=15
- 理由：35,100 张图片足够大，无需过多 epoch；Early Stopping 防止过拟合
- Batch size = 16（U2NETP 模型小，可以用更大 batch）

### 4.6 评估指标
**决策**: 与 v3 保持一致
- **主指标**: IoU (Jaccard), Dice (F1-Score)
- **辅助**: Precision, Recall, F-measure（ODS/OIS）
- 保存最佳模型依据：best_iou, best_dice

### 4.7 ONNX 导出规格
**问题**: 导出时输入尺寸固定还是动态？

**决策**: **固定输入 320x320 + 可选动态轴**
- 默认固定尺寸（推理优化）
- 提供动态 batch 轴选项
- 仅导出 d0（fused output），不导出中间 side outputs
- 简化模型：移除多余 sigmoid（ONNX 内置）

### 4.8 推理策略
**决策**: 滑动窗口 + 重叠拼接
- 窗口大小：320x320
- 重叠：1/3（与 v3 一致）
- 支持 TTA（水平翻转）
- 后处理：阈值二值化 + 可选形态学去噪

---

## 5. 文件结构设计

```
unet_seg_u2net/
├── model.py              # U2NETP 模型定义（从 U-2-Net 适配）
├── dataset.py            # CrackDataset 数据加载器
├── losses.py             # 多尺度 BCE + Dice 损失
├── train.py              # 训练脚本（主入口）
├── test.py               # 测试脚本（PyTorch 模型评估）
├── export_onnx.py        # ONNX 导出
├── predict_onnx.py       # ONNX 推理脚本
├── dataset_stats.json    # 数据集统计量（mean/std）
└── runs/                 # 训练输出目录
    └── exp_{timestamp}/
        ├── best_iou.pt
        ├── best_dice.pt
        ├── last.pt
        └── training_log.csv
```

---

## 6. 功能需求

### 6.1 train.py
- [x] 按原始图像划分 train/val（避免数据泄露）
- [x] U2NETP 模型加载与初始化
- [x] 多尺度损失（7 个输出的 BCE + Dice）
- [x] AdamW + Cosine Annealing + Warmup
- [x] Mixed Precision Training (AMP)
- [x] 自动保存 best_iou, best_dice, last checkpoint
- [x] 训练日志 CSV 记录
- [x] Early Stopping
- [x] 支持断点续训（--resume）

### 6.2 test.py
- [x] 加载 PyTorch checkpoint 进行评估
- [x] 支持多个测试集（CRKWH100, CrackLS315, Stone331, 验证集）
- [x] 滑动窗口推理（处理任意尺寸图像）
- [x] 计算 Precision, Recall, IoU, Dice, F-measure
- [x] 可视化输出（原图 + 预测叠加）
- [x] 支持 TTA

### 6.3 export_onnx.py
- [x] 从 checkpoint 加载模型
- [x] 导出固定 320x320 输入的 ONNX 模型
- [x] 仅导出 fused output (d0)
- [x] 验证 ONNX 输出一致性
- [x] 打印模型大小和参数统计

### 6.4 predict_onnx.py
- [x] ONNX Runtime 推理
- [x] 支持 CUDA / CPU provider
- [x] 滑动窗口处理大图
- [x] 结果可视化与保存

---

## 7. 非功能需求

| 项目 | 要求 |
|------|------|
| 模型大小 | < 10MB (ONNX) |
| 推理延迟 | < 50ms/frame @320x320 (GPU) |
| 训练显存 | < 8GB @batch_size=16 |
| 代码风格 | 与 v3 保持一致 |
| 依赖 | torch, torchvision, albumentations, onnx, onnxruntime, opencv-python, numpy, tqdm |

---

## 8. 里程碑

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| M1 | model.py + dataset.py + losses.py | P0 |
| M2 | train.py（完整训练流程） | P0 |
| M3 | test.py（评估+可视化） | P0 |
| M4 | export_onnx.py + predict_onnx.py | P0 |
| M5 | 数据集统计量计算 | P1 |

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| U2NETP 容量不足 | 精度不够 | 增大通道数（如64→128），或使用知识蒸馏 |
| 数据泄露 | 验证指标虚高 | 严格按原始图像划分 |
| 320x320 丢失细节 | 漏检细裂缝 | 滑动窗口推理；可试验更大输入 |
| 正负样本不平衡 | 训练不稳定 | Dice Loss + 在线难例挖掘 |

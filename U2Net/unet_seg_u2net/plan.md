# 执行计划: U2Net 裂缝语义分割

本文档根据 [PRD.md](PRD.md) 的需求，梳理并规划了 U2Net 裂缝语义分割项目的执行步骤。项目旨在实现一个轻量级的裂缝分割模型 (U2NETP)，并确保其能在 GPU 受限的环境中高效部署。

---

## 阶段 1：环境与数据准备 (M5 & 前置任务)

**目标：** 确保运行环境依赖安装完毕，且数据集可以被正确读取与预处理。

- [ ] **依赖确认**
  - [ ] 检查并安装所需的 Python 库：`torch`, `torchvision`, `albumentations`, `onnx`, `onnxruntime`, `opencv-python`, `numpy`, `tqdm` 等。
- [ ] **数据集审查与划分逻辑**
  - [ ] 编写或调整数据加载器代码，实现按 **原始图像** 划分训练集 (90%) 和验证集 (10%)，避免同源增强数据导致的泄露。
  - [ ] 确认测试集 (CRKWH100, CrackLS315, Stone331) 路径配置正确。
- [ ] **数据集统计量计算 (M5)**
  - [ ] 编写/运行脚本计算训练数据集的 RGB 均值 (mean) 和标准差 (std)。
  - [ ] 保存结果至 `dataset_stats.json` 以供后续脚本加载使用。

---

## 阶段 2：模型与基础组件实现 (M1)

**目标：** 完成模型定义、数据加载流水线和自定义损失函数。

- [ ] **编写 `model.py`**
  - [ ] 从原始 U-2-Net 代码适配 `U2NETP` (轻量版) 架构。
  - [ ] 确保网络输出为 7 个未经过 sigmoid 激活的 logits 预测图 (d0~d6)。
- [ ] **编写 `dataset.py`**
  - [ ] 实现 `CrackDataset` 数据加载器。
  - [ ] 集成 `albumentations`，配置轻度在线数据增强 (如 RandomBrightnessContrast, CLAHE, GaussNoise)，关闭会改变几何特征的增强操作。
  - [ ] 实现将图片统一 resize 为 320x320 的预处理逻辑。
- [ ] **编写 `losses.py`**
  - [ ] 实现 **多尺度混合损失 (Multi-scale BCE + Dice Loss)**。
  - [ ] 编写损失计算函数，将 d0~d6 各个输出的 BCE 和 Dice loss 求和，其中对融合输出 (d0) 给予双倍权重惩罚。

---

## 阶段 3：训练流程搭建 (M2)

**目标：** 实现完整的模型训练流水线，支持自动保存和日志记录。

- [ ] **编写 `train.py`**
  - [ ] 加载 `dataset_stats.json` 中的均值和方差初始化 Dataloader。
  - [ ] 实例化 `U2NETP` 模型和损失函数。
  - [ ] 配置 `AdamW` 优化器与 `Cosine Annealing with Warmup` 学习率调度器。
  - [ ] 引入 `torch.cuda.amp` 实现混合精度训练 (AMP) 以降低显存占用。
  - [ ] 实现训练循环 (最大 100 epochs，Batch size 设为 16)。
  - [ ] 添加 Early Stopping 机制 (patience=15)。
  - [ ] 实现验证逻辑，计算 IoU、Dice 等指标，并在表现提升时保存 `best_iou.pt` 和 `best_dice.pt`。
  - [ ] 实现断点续训功能 (`--resume`)。
  - [ ] 将每轮训练和验证的指标记录至 `runs/exp_{timestamp}/training_log.csv`。

---

## 阶段 4：评估与可视化实现 (M3)

**目标：** 编写测试脚本，评估 PyTorch 模型在各个测试集上的表现。

- [ ] **编写 `test.py`**
  - [ ] 编写模型权重加载逻辑 (读取 `.pt` 文件)。
  - [ ] 实现指标计算函数：Precision, Recall, IoU, Dice, F-measure。
  - [ ] 实现 **滑动窗口推理 (Sliding Window Inference)** 机制，支持重叠拼接 (窗口 320x320，重叠 1/3)，以支持任意分辨率大图测试。
  - [ ] 加入测试时增强 (TTA) 功能，例如水平翻转后取平均。
  - [ ] 编写可视化代码：将二值预测掩码以半透明颜色叠加至原图并保存。
  - [ ] 编写入口函数，支持在多个外部测试集上批量运行评估并打印结果。

---

## 阶段 5：ONNX 导出与部署验证 (M4)

**目标：** 将训练好的 PyTorch 模型导出为轻量级 ONNX 格式，并验证其推理效果。

- [ ] **编写 `export_onnx.py`**
  - [ ] 加载训练得到的 `best_iou.pt` 或 `best_dice.pt`。
  - [ ] 使用 `torch.onnx.export` 导出模型，默认输入尺寸固定为 320x320。
  - [ ] 仅保留 `d0` 融合输出，并通过包裹 `nn.Module` 在计算图末尾融合 `Sigmoid` 操作。
  - [ ] 配置动态批处理轴 (dynamic batch axis) 以备未来使用。
  - [ ] 验证生成的 ONNX 文件大小 (预期 < 10MB)。
- [ ] **编写 `predict_onnx.py`**
  - [ ] 使用 `onnxruntime` 库加载生成的 ONNX 模型。
  - [ ] 移植滑动窗口推理逻辑到 ONNX 推理版本，确保对任意尺寸图片的处理能力一致。
  - [ ] 测试单图推理速度，验证在 CPU/GPU 下的延迟是否满足要求 (< 50ms/frame)。
  - [ ] 验证 ONNX 推理生成的掩码和可视化结果与 PyTorch 版本完全一致。

---

## 阶段 6：系统联调与验收

**目标：** 整体串联测试，确保代码鲁棒性。

- [ ] **联调测试**
  - [ ] 从头到尾跑通一遍小型测试集，确认数据流无报错。
  - [ ] 检查 GPU 显存占用，确保满足受限环境要求 (< 8GB)。
- [ ] **风险排查**
  - [ ] 确认数据划分逻辑不存在数据泄露。
  - [ ] 若发现 320x320 导致细裂缝漏检严重，考虑在滑窗推理时适当调小 stride 或是增加形态学后处理逻辑。
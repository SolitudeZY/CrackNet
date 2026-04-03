# 裂缝语义分割模型改进空间分析报告

> 基于 `unet_seg_v2` 技术报告与完整代码库的深度审查
> 生成日期：2026-03-25

---

## 当前性能基线

| 指标 | 最佳值 | 对应 Epoch | 备注 |
|------|--------|-----------|------|
| IoU | **0.6508** | 119 | 主要瓶颈指标 |
| Dice/F1 | **0.7803** | 81 | 与 IoU 最佳 epoch 不一致 |
| Precision | 0.8708 | 16 | 早期即达到，后续下降 |
| Recall | 0.8050 | 10 | 同上 |

**核心问题**：模型在 IoU ~0.65 / Dice ~0.78 处形成了明显的天花板，后 100 个 epoch 几乎无提升。最佳 Precision 和最佳 IoU 出现在截然不同的 epoch，说明模型无法同时优化两者。

---

## 一、数据层面（Data-Centric）— 收益最高

### 1.1 数据增强严重不足（当前最大短板）

当前增强仅有：翻转、旋转（±30°）、亮度/饱和度/色相调整。这对裂缝这类细长、拓扑结构丰富的目标远远不够。

**建议补充的增强策略（按优先级排序）**：

| 增强方法 | 原因 | 预期收益 |
|---------|------|---------|
| **弹性变形 (Elastic Transform)** | 裂缝形态千变万化，弹性变形能模拟不同弯曲走向，是裂缝分割中收益最大的单一增强 | 高 |
| **高斯噪声 / ISO 噪声** | 边缘设备采图常有噪声，当前训练集过于"干净" | 中高 |
| **随机缩放 (Scale Jitter, 0.5x~2.0x)** | 当前固定 resize 到 640，缺少多尺度学习能力 | 中高 |
| **CLAHE / 对比度调整** | 工业场景光照差异大，CLAHE 能模拟不同曝光条件 | 中 |
| **随机模糊 (Gaussian / Motion Blur)** | 边缘设备拍照可能有运动模糊 | 中 |
| **CopyPaste / Mosaic** | 将一张图上的裂缝 mask 粘贴到另一张图的背景上，低成本倍增有效样本 | 高 |
| **GridDistortion / OpticalDistortion** | 模拟相机畸变 | 低中 |

**具体建议**：用 `albumentations` 库替换当前手写的 `torchvision.transforms.functional` 增强流水线，一次性获得上述所有能力，且原生支持 image + mask 联合变换，代码量更少、维护性更好。

### 1.2 缺乏多数据集联合训练

当前仅使用 CRACK500 的 traincrop。裂缝分割领域有多个公开数据集可以联合训练或预训练：

- **DeepCrack** (Liu et al.) — 含更精细的边缘标注
- **CrackTree200** — 树状裂缝结构
- **GAPs384** — 德国沥青路面裂缝
- **Crack Forest Dataset (CFD)** — 森林道路裂缝

即使不做正式的联合训练，也可以将这些数据集混合进来做**第一阶段预训练**，再在 CRACK500 上微调（两阶段迁移学习），能显著提升模型泛化能力。

### 1.3 缺乏 Hard Sample Mining 的数据侧手段

当前仅在损失函数层面做 OHEM。建议在 `diagnostic_results/` 中已有的错误分析基础上：

- 手动/半自动标注"困难样本"子集
- 使用 **过采样 (Oversampling)** 或 **WeightedRandomSampler** 让难样本被更频繁地训练到
- 考虑 **Curriculum Learning**：先用简单样本训练，逐步引入困难样本

---

## 二、模型架构（Architecture）— 收益中高

### 2.1 缺少多尺度特征融合模块

当前解码器是标准 U-Net skip connection 结构，对裂缝这种**跨尺度的细长结构**捕获不足。

**建议**：在 Bottleneck（1280ch, 20×20）处加入以下模块之一：

- **ASPP (Atrous Spatial Pyramid Pooling)**：用多个膨胀率（如 dilation=6, 12, 18）的空洞卷积并行提取多尺度上下文
- **PPM (Pyramid Pooling Module)**：多尺度池化后拼接

参数增量极小（几十 KB），但对裂缝走向的全局感知帮助很大。这是**架构层面投入产出比最高**的改动。

### 2.2 Backbone 偏老旧

MobileNetV2 (2018) 已不是最优的轻量级 backbone。在同等 FLOPs 约束下有更好的选择：

| Backbone | 参数量 | 优势 | 部署友好度 |
|---------|-------|------|-----------|
| **MobileNetV3-Large** | 5.4M | NAS 搜索的更优结构，h-swish 激活 | 高 |
| **EfficientNet-B0** | 5.3M | `model.py` 中已有实现，但未充分对比 | 高 |
| **MobileViT-S/XS** | 5.6M/2.3M | CNN+Transformer 混合，对全局裂缝走向感知强 | 中 |
| **EdgeNeXt-S** | 5.6M | 最新轻量级架构，SDTA 注意力 | 中 |

**建议**：优先测试 `MobileNetV3-Large`，这是投入产出比最高的替换，`timm` 库一行代码即可切换 backbone。

### 2.3 解码器跳跃连接偏简单

当前是 `concat(upsample(decoder_feat), attention(skip_feat))` 后双卷积。可以尝试：

- **Feature Refinement Module**：在 concat 之后加一个小的残差块（1×1 conv 降通道 → 3×3 conv → 残差连接），帮助融合不同层级的语义差异
- **Squeeze-and-Excitation 在 concat 之后**：重新标定融合后的通道权重

### 2.4 输出头过于简单

当前 head 仅是 `ConvTranspose2d → Conv2d → 1ch`。裂缝边界的精细度可以通过以下方式提升：

- 在输出前用 **PixelShuffle** 替代 ConvTranspose2d 上采样（更少棋盘格伪影）
- 或采用 **轻量级 Multi-Head 输出**：一个头输出语义 mask，一个头输出边界/距离场（multi-task），训练时互相约束

---

## 三、训练策略（Training Strategy）— 收益中

### 3.1 学习率调度可优化

当前使用 Cosine Annealing 在 epoch 3 warmup 后一路衰减到 `1e-6`。从训练曲线来看，模型在 epoch 80~120 之间达到最佳，之后过拟合/震荡。

**建议**：

- 尝试 **Cosine Annealing with Warm Restarts (SGDR)**，在 epoch 80 左右执行 "restart"，给优化器一次跳出局部最优的机会
- 或采用 **ReduceLROnPlateau**（监控 val IoU），在真正 plateau 时才降学习率，避免过早衰减
- 适当缩短总 epoch 数到 150，配合更激进的 early stopping（patience=40）

### 3.2 缺少正则化手段

当前仅有 `weight_decay=1e-4`（L2 正则），对于只有 5.5M 参数的模型和 CRACK500 这个中等规模的数据集，模型可能在中后期过拟合。

**建议**：

- 在解码器中加入 **Dropout2d (p=0.1~0.2)**，特别是在高分辨率特征图上
- 尝试 **Stochastic Depth / DropPath**（在跳跃连接处随机 drop）

### 3.3 加入 EMA（Exponential Moving Average）权重平滑

训练中维护一个模型参数的指数移动平均副本（decay=0.999），推理时用 EMA 权重。这对消除训练末期的权重震荡非常有效，通常能带来 0.5~1% 的稳定提升。实现成本极低（~20 行代码）。

```python
# EMA 伪代码
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.shadow[k] * self.decay + v * (1 - self.decay)
```

### 3.4 深度监督权重可优化

当前辅助输出权重写死为 `(1.0, 0.4, 0.2)`。建议：

- 让辅助损失权重也参与 cosine 衰减（训练初期辅助权重高，后期降低让主输出主导）
- 这能避免中间层过强的监督信号在后期干扰主输出的精细调整

---

## 四、损失函数（Loss Function）— 收益中

### 4.1 动态 Alpha 的 PID 控制存在震荡风险

当前公式：

```
α_new = clip(α_old + (recall - precision) * 0.1 + 0.02, 0.2, 0.8)
```

这里有一个常数偏移 `+0.02`，意味着 alpha 会持续上涨（偏向抑制误报）。这解释了为何 Precision 在早期很高（0.87）但后期下降 — alpha 被推得过高后又被 clip 回来，导致震荡。

**建议方案 A**（简单）：直接固定 `α=0.6, β=0.4`（实验已证明 0.7 偏高），去掉动态调整。

**建议方案 B**（完善 PID）：去掉固定偏移，加入积分项和微分项：

```python
error = recall - precision
integral += error
derivative = error - prev_error
adjustment = Kp * error + Ki * integral + Kd * derivative
alpha = clip(alpha + adjustment, 0.2, 0.8)
```

### 4.2 Boundary Loss 权重过低

当前 boundary loss 权重仅 0.1。从诊断结果来看掩码仍有粗糙的"香肠"效应，说明 boundary 约束不够。

**建议**：

- 逐步提高到 0.2~0.3
- 或用 **距离加权 boundary loss**：距离边界越远的错误像素，惩罚越重

### 4.3 缺少 Lovász-Softmax Loss

Lovász loss 是 IoU 指标的**直接可微近似**。你的目标指标是 IoU，但当前没有直接优化它。

**建议**：作为复合损失的第四个组件加入：

```
L_total = L_tversky_focal + L_ohem + 0.2 * L_boundary + 0.5 * L_lovasz
```

---

## 五、推理与后处理（Inference & Post-processing）— 收益低中

### 5.1 阈值 0.5 未经搜索（零成本提升）

当前硬编码 `THRESHOLD = 0.5`。不同的模型在不同阈值下表现差异巨大。

**建议**：在验证集上跑一次**阈值扫描 (Threshold Search)**，从 0.1 到 0.9 步长 0.05，找到 IoU 最优的阈值。裂缝分割中最优阈值通常在 **0.3~0.45** 之间（因为正样本极少，模型倾向于低置信度预测）。

### 5.2 TTA 可以更丰富

当前仅有水平翻转 TTA。可以低成本添加：

- 垂直翻转
- 90° / 180° / 270° 旋转
- 多尺度 TTA（0.75x, 1.0x, 1.25x 三种分辨率）

4~8 种 TTA 的集成通常能再提 1~3% IoU，代价是推理时间线性增长。

### 5.3 缺少 CRF 后处理

**DenseCRF** 能基于图像的颜色/位置信息对粗糙的分割结果进行精细化，特别擅长修正边界。`pydensecrf` 库可直接调用。

对于边缘设备部署，可以：
- 仅在离线评估/高精度场景下使用
- 或用轻量级的 Bilateral Solver 近似

### 5.4 小区域过滤阈值 150px 是否最优

当前硬编码过滤掉面积 < 150px 的连通域。建议：

- 在验证集上搜索最优阈值（100~500 范围）
- 或改用基于**面积+形状**的联合判断：裂缝是细长的（aspect ratio > 3 的保留），圆形的过滤

---

## 六、实验管理与评估（Evaluation）— 基础设施改进

### 6.1 缺少测试集评估

所有指标都是验证集上的。应在 CRACK500 的 `testcrop` 上做最终评估，避免在验证集上 overfit 超参数。

### 6.2 缺少边界质量专项指标

当前仅有 IoU / Dice / Precision / Recall。建议加入：

- **Boundary IoU (BIoU)**：只在边界附近（如 ±2px）计算 IoU，直接量化边缘精度
- **Hausdorff Distance (HD95)**：衡量预测边界与真实边界的最大偏差（取 95% 分位数）
- 这些指标能精准反映"香肠"问题的改善程度

### 6.3 缺少 K-Fold 交叉验证

单一 train/val split 的结果方差较大。在 CRACK500 上做 5-Fold CV 可以更可靠地评估每次改进的真实收益，避免"改了一个参数，指标波动 ±1% 但不知道是随机性还是真改进"的困境。

---

## 改进优先级总结

| 优先级 | 改进项 | 预期 IoU 提升 | 实施难度 | 备注 |
|--------|-------|-------------|---------|------|
| **P0** | Albumentations 增强（弹性变形+噪声+缩放+模糊） | +2~4% | 低 | 投入产出比最高 |
| **P0** | 阈值搜索（替代硬编码 0.5） | +1~2% | 极低 | 零训练成本 |
| **P1** | 加入 ASPP/PPM 多尺度模块 | +1~2% | 低 | 架构最优改动 |
| **P1** | EMA 权重平滑 | +0.5~1% | 极低 | ~20 行代码 |
| **P1** | 加入 Lovász Loss | +1~2% | 低 | 直接优化 IoU |
| **P1** | CopyPaste 数据增强 | +1~3% | 中 | 需要实现增强逻辑 |
| **P2** | 更换 MobileNetV3 backbone | +1~2% | 低 | timm 一行切换 |
| **P2** | 修复动态 alpha PID 控制 | +0.5~1% | 低 | 去掉 +0.02 偏移 |
| **P2** | Decoder Dropout2d | +0.5~1% | 极低 | 3 行代码 |
| **P2** | 更丰富的 TTA（多翻转+多尺度） | +1~2% | 低 | 推理时间换精度 |
| **P3** | DenseCRF 后处理 | +1~2% | 中 | 推理端优化 |
| **P3** | 多数据集联合预训练 | +2~4% | 高 | 需要数据收集与清洗 |
| **P3** | MobileViT 混合架构 | +1~3% | 中 | 更大的架构变动 |

---

## 推荐实施路线图

### 第一阶段：低成本高回报（1~2 天）
1. 验证集阈值搜索 → 确定最优 threshold
2. 替换为 `albumentations` 增强管线
3. 加入 EMA 权重平滑
4. 修复动态 alpha 控制（去掉 +0.02 偏移或固定 α=0.6）

### 第二阶段：架构微调（2~3 天）
5. Bottleneck 加入 ASPP 模块
6. 复合损失函数加入 Lovász Loss
7. Decoder 加入 Dropout2d
8. 提高 Boundary Loss 权重到 0.2

### 第三阶段：进一步突破（3~5 天）
9. 测试 MobileNetV3-Large backbone
10. 实现 CopyPaste 增强
11. 丰富 TTA 策略
12. 在 testcrop 上做最终评估

### 预期目标
- 第一阶段后：IoU **0.68~0.70**
- 第二阶段后：IoU **0.71~0.73**
- 第三阶段后：IoU **0.74+**

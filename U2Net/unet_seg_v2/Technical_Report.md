# 裂缝语义分割模型 (U-Net) 阶段性技术优化报告
## 1. 核心业务需求与痛点分析
在工业级裂缝检测任务中，我们面临的主要挑战是：

1. 极端类别不平衡 ：裂缝像素通常只占整张图片的不到 1%，背景噪声（水迹、接缝、树影）极具迷惑性。
2. 边缘精细度要求高 ：业务需求明确指出 “精度 (Precision) > 查全率 (Recall)” ，即宁可漏检部分模糊的微小裂缝，也绝不允许出现大面积的误报（把背景画成粗糙的“香肠”）。
3. 边缘设备部署限制 ：模型需要最终导出并部署在算力受限的终端设备上，对推理速度和内存占用有严格要求。
针对上述痛点，我们设计并实施了一套从“粗放式”走向“诊断式”的精准优化方案。

## 2. 模型架构优化 (Architecture Engineering)
### 2.1 轻量级主干网络：MobileNetV2
- 放弃了传统 U-Net 厚重的卷积层，采用 MobileNetV2 作为 Encoder。其深度可分离卷积机制大幅降低了 FLOPs 和参数量（仅约 5.5M），完美契合边缘部署的实时性要求，同时借助 ImageNet 预训练权重加速了收敛。
### 2.2 架构 A/B 测试：引入多种注意力机制 (Attention Mechanisms)
针对背景噪声干扰严重的问题，我们在解码器 (Decoder) 的跳跃连接处引入并测试了多种注意力模块：

- AG (Attention Gate) ：利用深层语义指导浅层特征过滤。
- CBAM (Convolutional Block Attention Module) ：结合通道和空间双重注意力，对复杂的裂缝纹理更敏感。
- ECA (Efficient Channel Attention) ：极轻量级的 1D 卷积通道注意力，几乎不增加参数量即可提升特征表达。 当前实验配置支持在 train.py 中通过 attention_type 字段进行热切换对比。
### 2.3 深度监督 (Deep Supervision)
- 借鉴 UNet++ 思想，在解码器的 d2 和 d3 层增加了辅助输出分支。
- 训练时，迫使网络中间层也学习裂缝特征（主输出权重 1.0，浅层辅助权重 0.4 和 0.2）。这极大加速了网络的收敛速度，并有效改善了边缘模糊问题。
## 3. 损失函数重构与自适应调优 (Loss Function Design)
这是解决“误报多”和“掩码粗糙”最核心的一环。我们彻底摒弃了传统的 BCE 或简单的 Dice Loss，设计了 三位一体的复合损失函数 ：

### 3.1 动态 Tversky Loss (Dynamic Tversky Loss)
- 原理 ：Tversky Loss 允许我们分别对假阳性 (FP, 误报) 和假阴性 (FN, 漏检) 施加不同的惩罚权重 ( [ o bj ec tO bj ec t ] α 和 [ o bj ec tO bj ec t ] β )。
- 自适应 PID 控制系统 ：为满足“精度 > 查全率”的需求，我们在训练循环中引入了一套基于 Epoch 反馈的动态权重调整机制。系统会实时监控验证集的 [ o bj ec tO bj ec t ] P rec i s i o n 和 [ o bj ec tO bj ec t ] R ec a ll 差值，并自动调高 [ o bj ec tO bj ec t ] α （加重对误报的惩罚）。这使得模型能够像“自动驾驶”一样，在训练中寻找既不严重漏检又保持极高精度的最佳平衡点。
### 3.2 在线难例挖掘 (OHEM + Focal Loss)
- 原理 ：不再对整张图的像素平均发力。计算 Focal BCE Loss 后，强制将其展平并排序， 只取 Loss 最大的前 20% 像素（即模型最拿不准的边缘和迷惑性背景）进行反向传播 。
- 效果 ：这本“错题本”机制逼迫模型死磕困难样本，显著消除了将水迹或接缝误判为裂缝的现象。
### 3.3 GPU 加速的边界损失 (Boundary Loss)
- 原理 ：为了把粗糙的预测掩码“削薄”，我们引入了边界损失。它惩罚任何溢出真实裂缝轮廓的预测像素。
- 工程优化 ：传统的距离变换（DTM）依赖 CPU 计算，导致训练速度暴降（30s -> 183s / Epoch）。我们巧妙地使用 torch.nn.functional.max_pool2d 实现了 基于 GPU 的形态学膨胀与腐蚀 ，在 CUDA 上极其高效地提取了近似边界，使训练速度恢复到秒级，同时实现了掩码的纤细化。
## 4. 训练策略与工程保障 (Training Strategy)
### 4.1 差分学习率与余弦退火
- 为预训练的 Encoder 设置较低的学习率（ [ o bj ec tO bj ec t ] 1 e − 4 ），为随机初始化的 Decoder 设置较高的学习率（ [ o bj ec tO bj ec t ] 1 e − 3 ）。
- 结合前 3 个 Epoch 的 Linear Warmup 和后续的 Cosine Annealing 调度，确保模型平稳起步并榨干最后的收敛性能。
### 4.2 梯度累加与断点续训 (Gradient Accumulation & Resume)
- 显存优化 ：受限于 Attention 和深层监督增加的显存占用，采用 batch_size=8 配合 grad_accumulation=2 ，等效实现 Batch Size 16 的平滑梯度下降。
- 工程鲁棒性 ：实现了包含模型权重、优化器状态和历史最佳指标的完整断点续训（Resume）逻辑，无惧意外中断。
## 5. 推理部署与后处理 (Inference & Deployment)
### 5.1 滑动窗口推理与 TTA (Test-Time Augmentation)
- 在 predict.py 中实现了基于 Padding 的无缝滑动窗口推理，支持任意尺寸的高清工业图片。
- 引入了 水平翻转 TTA ，通过双视角预测求均值，在不重新训练的前提下白嫖了 1-2% 的 Precision 提升。
### 5.2 形态学清洗 (Morphological Post-processing)
- 针对模型偶发的点状误报（如小石子），在推理后串联了基于连通域分析（ cv2.connectedComponentsWithStats ）的硬性过滤，直接抹除面积小于 150 像素的孤立噪点，确保输出画面的绝对干净。
### 5.3 工业级 ONNX 导出
- 编写了 export_onnx.py ，使用最新的 Opset 18 将模型转换为工业标准格式。
- 核心特性 ：开启了常量折叠优化（Constant Folding），且 支持动态的 Batch Size 和动态输入分辨率 (Dynamic Axes) ，为后续在海康威视等边缘设备上使用 C++ TensorRT/ONNXRuntime 极速推理扫清了障碍。
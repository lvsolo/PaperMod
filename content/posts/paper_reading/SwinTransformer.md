---
title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
author: "lvsolo"
date: "2026-06-11"
tags: ["paper reading", "Transformer", "backbone", "vision", "self-attention", "detection", "segmentation"]
ShowToc: true
TocOpen: true
---

# 论文信息

- **标题**: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- **作者**: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
- **机构**: Microsoft Research Asia (MSRA)
- **发表**: ICCV 2021 (**Best Paper Award**)
- **arXiv**: [2103.14030](https://arxiv.org/abs/2103.14030)
- **代码**: [github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

> **一句话总结**: 原版 ViT 用全局自注意力，复杂度 O(N²)，无法处理高分辨率密集预测任务（检测/分割）。Swin Transformer 引入 **移位窗口自注意力 (Shifted Window MSA)**——在窗口内算 attention（O(N) 线性复杂度），相邻层窗口移位以实现跨窗口信息流动，并构建**层级结构**（像 CNN backbone 一样输出多尺度特征），成为通用视觉 backbone。

---

# 1. Introduction (引言)

## 1.1 背景: Transformer 在 NLP 成功，但在视觉上不通用

```
NLP 领域:
  Transformer 已是主流 (GPT, BERT)
  处理 token 序列, 全局自注意力

视觉领域:
  ViT (Vision Transformer, 2020) 把 Transformer 引入图像分类
  但 ViT 有重大局限 → 不能作为通用 backbone
```

## 1.2 ViT 的两大问题

```
问题 ①: 复杂度 O(N²), 无法处理高分辨率
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ViT 的自注意力是全局的:
    每个 token 和所有 N 个 token 算 attention
    计算量: O(N² · d)

  图像分patch:
    分类: 224×224 图像, patch=16 → N=196 个 token (还行)
    检测/分割: 高分辨率特征图, 例如 H×W = 128×128 → N=16384
    → N² = 2.7 亿! 计算爆炸

  → ViT 无法直接用于密集预测任务 (检测/分割)

问题 ②: 单尺度特征, 无层级结构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CNN backbone (ResNet 等):
    产生多尺度特征 (C3, C4, C5, 不同分辨率)
    下游任务 (FPN, U-Net) 依赖多尺度

  ViT:
    始终保持单一分辨率 (patch 数不变)
    没有下采样的层级结构
    → 无法直接对接需要多尺量的下游检测/分割框架

  → ViT 不是 "通用 backbone", 只能做分类
```

## 1.3 Swin Transformer 的解决方案

```
Swin = Shifted WINdow (移位窗口)

两大核心设计:
━━━━━━━━━━━━━━━━

  ① 层级结构 (Hierarchical):
     像 ResNet 一样, 逐级下采样, 产生多尺度特征
     → 可作为通用 backbone

  ② 移位窗口自注意力 (Shifted Window MSA):
     Layer L:   窗口划分 W-MSA   (窗口内算 attention)
     Layer L+1: 窗口移位 SW-MSA  (移位后窗口内算 attention)
     → 复杂度 O(N) 线性!
     → 移位实现跨窗口信息流动
```

```
与 ViT 对比:
━━━━━━━━━━━━━━

  特性              ViT              Swin Transformer
  ─────────────────────────────────────────────────
  自注意力          全局             窗口内 (局部)
  复杂度            O(N²)            O(N) 线性
  特征尺度          单尺度           多尺度 (层级)
  位置编码          绝对             相对位置偏置
  适合分类          ✓                ✓
  适合检测/分割     ✗ (复杂度爆炸)   ✓ (通用 backbone)
```

---

# 2. Method (方法)

## 2.1 Overall Architecture (整体架构)

```
┌─────────────────────────────────────────────────────────────────┐
│                   Swin Transformer 整体架构                      │
│                                                                  │
│   输入图像 H×W×3                                                 │
│       │                                                          │
│       ▼ Patch Partition (4×4 patch)                              │
│   ┌─────────────────────────────────────────┐  Stage 1           │
│   │  Linear Embedding → C=96                │  分辨率: H/4×W/4   │
│   │  ┌─────────────┐  ┌─────────────┐      │  通道: C           │
│   │  │  Swin Block │→ │  Swin Block │ ×2   │                    │
│   │  │  (W-MSA)    │  │  (SW-MSA)   │      │                    │
│   │  └─────────────┘  └─────────────┘      │                    │
│   └───────────────────┬─────────────────────┘                    │
│                       │                                          │
│                       ▼ Patch Merging (2×2 合并)                  │
│   ┌─────────────────────────────────────────┐  Stage 2           │
│   │  ┌─────────────┐  ┌─────────────┐      │  分辨率: H/8×W/8   │
│   │  │  Swin Block │→ │  Swin Block │ ×2   │  通道: 2C          │
│   │  │  (W-MSA)    │  │  (SW-MSA)   │      │                    │
│   │  └─────────────┘  └─────────────┘      │                    │
│   └───────────────────┬─────────────────────┘                    │
│                       │                                          │
│                       ▼ Patch Merging                             │
│   ┌─────────────────────────────────────────┐  Stage 3           │
│   │  ┌─────────────┐  ┌─────────────┐      │  分辨率: H/16×W/16 │
│   │  │  Swin Block │→ │  Swin Block │ ×6   │  通道: 4C          │
│   │  │  (W-MSA)    │  │  (SW-MSA)   │      │  (Swin-T)          │
│   │  └─────────────┘  └─────────────┘      │                    │
│   └───────────────────┬─────────────────────┘                    │
│                       │                                          │
│                       ▼ Patch Merging                             │
│   ┌─────────────────────────────────────────┐  Stage 4           │
│   │  ┌─────────────┐  ┌─────────────┐      │  分辨率: H/32×W/32 │
│   │  │  Swin Block │→ │  Swin Block │ ×2   │  通道: 8C          │
│   │  │  (W-MSA)    │  │  (SW-MSA)   │      │                    │
│   │  └─────────────┘  └─────────────┘      │                    │
│   └─────────────────────────────────────────┘                    │
│                                                                  │
│   输出: 4 个尺度的特征图 (H/4, H/8, H/16, H/32)                  │
│         → 可直接对接 FPN / Mask R-CNN / UperNet 等下游框架        │
└─────────────────────────────────────────────────────────────────┘
```

> **关键**: 每个 Stage 包含多个 Swin Block，相邻 Block 一个用 W-MSA（常规窗口），下一个用 SW-MSA（移位窗口），交替进行。每个 Stage 结束后用 Patch Merging 下采样（分辨率减半，通道翻倍），模仿 CNN 的层级结构。

## 2.2 Patch Partition & Linear Embedding

```
图像 → Patch Partition (4×4):
  把 H×W×3 的图像切成 (H/4)×(W/4) 个 patch
  每个 patch 是 4×4×3 = 48 维
  → 特征图: (H/4)×(W/4)×48

Linear Embedding:
  把 48 维投影到 C 维 (Swin-T: C=96)
  → 特征图: (H/4)×(W/4)×C

  这相当于 ViT 的 Patch Embedding (用 4×4 卷积, stride 4)
```

## 2.3 Shifted Window based Self-Attention (核心!)

### 2.3.1 为什么需要窗口化?

```
全局自注意力 (ViT):
  对 (H/4)×(W/4) 个 token, 每个和所有 token 算 attention
  复杂度: O((HW)² · d)   ← 二次方!

窗口自注意力 (Swin):
  把特征图分成 ⌈H/M⌉×⌈W/M⌉ 个窗口 (窗口大小 M×M, 通常 M=7)
  每个窗口内独立算 self-attention
  复杂度: O((HW) · M² · d)   ← 对 HW 线性!

  复杂度对比 (M=7, C 固定):
    全局 MSA:   4hwC² + 2(hw)²C       ← (hw)² 项是瓶颈
    窗口 W-MSA: 4hwC² + 2M²hwC        ← 只有 hw 一次方
```

### 2.3.2 W-MSA: 常规窗口划分

```
窗口大小 M=4 的示例 (实际 M=7):

  特征图 8×8 (token 网格), 分成 2×2 = 4 个窗口:

  ┌────────┬────────┐
  │        │        │
  │  W1    │  W2    │   每个窗口 4×4 = 16 个 token
  │ (4×4)  │ (4×4)  │   窗口内独立做 self-attention
  │        │        │
  ├────────┼────────┤   W1 的 token 只和 W1 内的 token 交互
  │        │        │   W2 的 token 只和 W2 内的 token 交互
  │  W3    │  W4    │   ...
  │ (4×4)  │ (4×4)  │
  │        │        │
  └────────┴────────┘

  问题: 窗口之间没有信息交流!
    W1 和 W2 的 token 互不可见
    → 限制了建模能力 (感受野被限制在窗口内)
```

### 2.3.3 SW-MSA: 移位窗口划分 (关键创新!)

```
解决窗口隔离: 下一层把窗口移位 (⌊M/2⌋, ⌊M/2⌋)

  Layer L (W-MSA): 常规划分
  ┌────────┬────────┐
  │  W1    │  W2    │
  │ (4×4)  │ (4×4)  │
  ├────────┼────────┤
  │  W3    │  W4    │
  │ (4×4)  │ (4×4)  │
  └────────┴────────┘

  Layer L+1 (SW-MSA): 窗口移位 (向左上移 ⌊4/2⌋=2 格)
  ┌────┬───┬────┐
  │ Wc │  W2'  │  ← 移位后, 新窗口跨越了原来的窗口边界!
  ├────┤       │     W2' 包含原 W1 右半 + W2 左半
  │ W3'│───┤ W4'│     → 实现了跨窗口信息流动!
  │    │ Wd   │
  └────┴──────┘

  移位后的效果:
    原来 W1 的 token 现在和 W2 的 token 在同一个新窗口 W2'
    → 它们可以交互了! 跨窗口信息流动建立

  交替使用 W-MSA 和 SW-MSA:
    Block 2k:   W-MSA  (常规窗口)
    Block 2k+1: SW-MSA (移位窗口)
    → 信息的全局传播: 窗口移位让 attention 像 "滑动窗口" 一样
       逐渐覆盖整个特征图
```

### 2.3.4 移位窗口的高效实现 (cyclic-shift)

```
移位窗口的朴素实现:
  把整个特征图移位 → 边界出现 "残缺窗口" (size < M)
  → 这些小窗口也要算 attention (但大小不一, 难并行)

高效实现 (cyclic shift, 循环移位):
  ┌──────────┐        ┌────┬─────┐
  │ A  │  B  │        │ Cc │  A  │   把图 cyclic shift (循环移位)
  ├────┼────┤   →    │    │     │   上下左右循环挪 ⌊M/2⌋
  │ C  │  D  │        ├────┼─────┤
  └──────────┘        │ B  │ Dc  │
                      └────┴─────┘
  现在 A, B, C, D 都是规则的 M×M 窗口!
  → 可以并行计算

  但问题: Cc 和 Dc 实际上是原图中不相邻的两块拼在一起的
  → 它们不该互相做 attention!

  解决: Attention Mask (注意力掩码)
    给 Cc 内部的 token 和 "拼进来的" 部分之间加 mask (置 -∞)
    → softmax 后它们互不影响

  优势:
    - 不需要额外的 padding
    - 窗口大小统一, 高效并行
    - 通过 mask 处理边界, 计算量不增加
```

### 2.3.5 Swin Transformer Block 结构

```
一个 Swin Block (成对出现, 共享设计):

  ┌──────────────────────────────────────┐
  │            输入特征 X                 │
  │               │                      │
  │      ┌────────▼─────────┐            │
  │      │  LayerNorm       │            │
  │      └────────┬─────────┘            │
  │               │                      │
  │   奇偶 Block 不同:                   │
  │   ┌───────────┴───────────┐          │
  │   │ 偶数 Block: W-MSA     │          │
  │   │ 奇数 Block: SW-MSA    │          │
  │   └───────────┬───────────┘          │
  │               │                      │
  │      ┌────────▼─────────┐            │
  │      │  + X (残差连接)   │            │
  │      └────────┬─────────┘            │
  │               │                      │
  │      ┌────────▼─────────┐            │
  │      │  LayerNorm       │            │
  │      └────────┬─────────┘            │
  │               │                      │
  │      ┌────────▼─────────┐            │
  │      │  MLP (GELU)      │            │
  │      └────────┬─────────┘            │
  │               │                      │
  │      ┌────────▼─────────┐            │
  │      │  + (残差连接)     │            │
  │      └────────┬─────────┘            │
  │               │                      │
  │               ▼                      │
  │            输出特征                   │
  └──────────────────────────────────────┘

  即标准 Transformer Block, 唯一区别:
    Multi-Head Self-Attention → W-MSA / SW-MSA
```

## 2.4 Relative Position Bias (相对位置偏置)

```
Swin 不用绝对位置编码, 而用相对位置偏置 B:

  Attention(Q,K,V) = SoftMax(QK^T/√d + B) · V
                                       ↑
                                  相对位置偏置

  B_ij = 相对位置 (token i 相对 token j) 对应的可学习偏置

  在窗口内 (M×M 个 token):
    相对位置范围: x ∈ [-M+1, M-1], y ∈ [-M+1, M-1]
    → 共 (2M-1)×(2M-1) 个不同的相对位置
    → 用一个 (2M-1)×(2M-1) 的可学习表存储

  例 (M=7): 表大小 = 13×13 = 169 个偏置值

  优势:
    - 相对位置对平移不变 (物体移动, 相对关系不变)
    - 比绝对位置编码更符合视觉
    - 实验证明: 相对位置偏置对性能贡献很大 (>1% ImageNet)
```

## 2.5 Patch Merging (Patch 合并 / 下采样)

```
每个 Stage 结束, 用 Patch Merging 下采样 (类似 stride-2 卷积):

  输入: H×W×C 的特征图 (这里 H,W 是 patch 数)
       ┌──┬──┬──┬──┐
       │1 │2 │5 │6 │
       ├──┼──┼──┼──┤    取 2×2 邻域:
       │3 │4 │7 │8 │    {1,2,3,4}, {5,6,7,8}, ...
       ├──┼──┼──┼──┤
       │..│..│..│..│
       └──┴──┴──┴──┘

  Step 1: 2×2 邻域取出来, 分成 4 组
    group1 (左上): [1,5,..]
    group2 (右上): [2,6,..]
    group3 (左下): [3,7,..]
    group4 (右下): [4,8,..]

  Step 2: concat 4 组 → (H/2)×(W/2)×4C

  Step 3: Linear(4C → 2C) → (H/2)×(W/2)×2C

  效果:
    分辨率: H×W → (H/2)×(W/2)    ← 减半
    通道: C → 2C                   ← 翻倍

  类比 CNN: 这就是 stride-2 的下采样
  → 4 个 Stage 产生 H/4, H/8, H/16, H/32 四个尺度
```

## 2.6 Architecture Variants (模型变体)

```
通过调整 通道数C 和 每层block数 产生不同大小的模型:

  变体     通道C   每层 block 数 {S1,S2,S3,S4}   参数量
  ────────────────────────────────────────────────────
  Swin-T   96      {2, 2, 6,  2}                28M
  Swin-S   96      {2, 2, 18, 2}                50M
  Swin-B   128     {2, 2, 18, 2}                88M
  Swin-L   192     {2, 2, 18, 2}                197M

  窗口大小: M=7 (所有变体一致)
  头数: C/32 (Swin-T: 96/32=3 heads in stage1)

  Swin-T: 小模型, 适合对比 DeiT-Small / ResNet-50
  Swin-B: 对比 ViT-Base
  Swin-L: 大模型, 配合大规模数据 (ImageNet-22K 预训练)
```

---

# 3. Complexity Analysis (复杂度分析)

## 3.1 全局 MSA vs 窗口 MSA

```
设: 特征图 h×w, 通道 C, 窗口大小 M

① 全局 Multi-Head Self-Attention (ViT):
   Ω(MSA) = 4hwC² + 2(hw)²C
            ↑ QKV投影   ↑ attention 矩阵
            线性项       二次项 (hw)² ← 瓶颈!

② Window MSA (Swin):
   Ω(W-MSA) = 4hwC² + 2M²hwC
               ↑ QKV投影  ↑ 窗口内 attention
               线性项      对 hw 线性! (M 固定)

③ Shifted W-MSA (Swin):
   Ω(SW-MSA) = 4hwC² + 2M²hwC   (和 W-MSA 相同)
   cyclic-shift + mask 不增加计算量
```

## 3.2 数值对比 (论文 Table)

```
典型场景 (检测, h×w 较大):

  h×w        全局 MSA (hw)²项    窗口 MSA (M²hw项, M=7)
  ──────────────────────────────────────────────────
  56×56      9.8M                49×56² ≈ 0.15M     ← 窗口少 ~64倍
  112×112    157M                49×112² ≈ 0.6M     ← 少 ~260倍
  224×224    2.5G                49×224² ≈ 2.5M     ← 少 ~1000倍

  分辨率越高, 窗口化的优势越大!
  → 这就是 Swin 能用于高分辨率检测/分割的原因
```

---

# 4. Experiments (实验)

## 4.1 ImageNet-1K 分类

```
ImageNet-1K (224×224):
  方法              参数量    Top-1 acc
  ─────────────────────────────────────
  ResNet-50         25M       76.1%
  DeiT-Small        22M       79.8%
  ViT-Small/16      22M       77.7% (需大数据预训练)
  Swin-T            28M       81.3%   ← 显著超越
  ─────────────────────────────────────
  ResNet-152        60M       78.3%
  Swin-S            50M       83.0%
  ─────────────────────────────────────
  Swin-B (22K预训练) 88M      85.2%
  Swin-L (22K预训练) 197M     87.3%   ← 接近 SOTA

  结论: Swin 在同等参数量下全面超越 CNN (ResNet) 和 ViT/DeiT
```

## 4.2 COCO 目标检测

```
COCO 检测 (作为 backbone 替换 ResNet):

  Mask R-CNN 框架:
    Backbone          box AP   参数量
    ─────────────────────────────────
    ResNet-50         42.2     44M
    Swin-T (1K预训练)  43.7     48M    ← +1.5
    ─────────────────────────────────
    ResNet-101        44.1     60M
    Swin-S (1K)       45.7     69M

  更大框架:
    Cascade Mask R-CNN + Swin-L: 51.9 box AP
    HTC + Swin-L:               53.8 → 54.6 (加额外数据) → 58.7 box AP (SOTA)

  结论: Swin 作为检测 backbone 显著优于 CNN, 且能扩展到高精度
```

## 4.3 ADE20K 语义分割

```
ADE20K 语义分割 (UperNet 框架):

  Backbone          mIoU
  ────────────────────────
  ResNet-101        44.8
  Swin-T            41.5 (需调参)
  Swin-S            47.6
  Swin-B (22K)      48.1
  Swin-L (22K)      53.5 mIoU   ← SOTA

  结论: 分割任务同样受益于 Swin 的层级多尺度特征
```

## 4.4 消融实验

```
关键消融 (论文验证各设计的作用):

  ① 移位窗口 (SW-MSA) vs 不移位:
     不移位 (只有 W-MSA):  窗口隔离, 性能差
     + SW-MSA 移位:        +1.1% ImageNet, +2.0 box AP
     → 移位是跨窗口信息流动的关键

  ② 相对位置偏置 vs 绝对/无:
     无位置编码:     79.9% (ImageNet, Swin-T)
     绝对位置编码:   80.5%
     相对位置偏置:   81.3%   ← 最好
     → 相对位置最适合视觉任务

  ③ 不同移位策略:
     cyclic shift (循环移位) 最高效
```

---

# 5. Swin V2 (后续工作简介)

```
Swin Transformer V2: Scaling Up Capacity and Resolution
(Ze Liu et al., CVPR 2022, arXiv:2111.09883)

  解决 Swin V1 在超大模型/超高分辨率下的问题:

  ① Pre-normalization (预归一化):
     V1: post-norm (LN 在残差后)
     V2: pre-norm  (LN 在残差前)
     → 大模型训练更稳定

  ② Scaled Cosine Attention (缩放余弦注意力):
     V1: QK^T/√d
     V2: α · cos(Q,K)   ← 余弦相似度, 有可学习缩放 α
     → 防止注意力过于集中, 更鲁棒

  ③ Log-spaced CPB (对数连续位置偏置):
     V1: 相对位置偏置用固定大小表 (依赖窗口大小)
     V2: 连续函数 + 对数空间, 可迁移到任意窗口/分辨率
     → 支持预训练-微调时的分辨率变化

  规模: SwinV2-G (30亿参数), 在 1536×1536 分辨率上训练
  应用: 用于 COCO 等的下游任务刷新 SOTA
```

---

# 6. 核心要点总结

## 6.1 Swin 的三大核心设计

```
┌────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ① 移位窗口自注意力 (Shifted Window MSA):                        │
│     W-MSA (常规窗口) + SW-MSA (移位窗口) 交替                    │
│     → 复杂度从 O(N²) 降到 O(N) 线性                             │
│     → 移位实现跨窗口信息流动 (cyclic shift + mask 高效实现)      │
│                                                                 │
│  ② 层级结构 (Hierarchical):                                     │
│     4 个 Stage + Patch Merging 逐级下采样                       │
│     → 输出 H/4, H/8, H/16, H/32 多尺度特征                      │
│     → 可对接 FPN/UperNet 等下游框架 (通用 backbone)             │
│                                                                 │
│  ③ 相对位置偏置 (Relative Position Bias):                       │
│     Attention = SoftMax(QK^T/√d + B)V                          │
│     → 平移不变, 适合视觉                                        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## 6.2 为什么 Swin 是里程碑

```
1. 第一个真正通用的视觉 Transformer backbone:
   - 分类: 超越 ViT/DeiT
   - 检测/分割: 超越 ResNet 系列
   - 一个架构, 全任务通用

2. 让 Transformer 能处理高分辨率:
   - 窗口注意力把复杂度降到线性
   - 检测/分割需要的高分辨率成为可能

3. 移位窗口的思想影响深远:
   - Swin V2 (更大规模)
   - CSWin Transformer, Focal Transformer 等后续变体
   - 甚至 FlashAttention 也借鉴了 "分块" 思想 (不同层面的 tiling)

4. 证明了: CNN 的归纳偏置 (层级+局部) 仍然重要
   - 纯 ViT (无层级, 全局) 不如 Swin (层级+局部窗口)
   - 结合 Transformer 灵活性和 CNN 归纳偏置 → 最强
```

## 6.3 适用场景

```
适合 Swin:
  ✓ 通用视觉 backbone (分类/检测/分割全适用)
  ✓ 需要多尺度特征的任务 (配合 FPN)
  ✓ 高分辨率输入 (检测/分割)
  ✓ 追求高精度 (Swin-L/B)

不适合 / 注意:
  △ 极致推理速度: 窗口划分+mask 有额外开销, 移动端不如轻量CNN
  △ 纯分类小模型: DeiT 等可能更轻
  △ 需要全局长程依赖: 窗口注意力靠堆叠层才能扩大感受野
```

---

# 7. 参考资料

- **Swin Transformer 原论文**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021 (Best Paper), [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- **官方代码**: [github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- **Swin V2**: Liu et al., "Swin Transformer V2: Scaling Up Capacity and Resolution", CVPR 2022, [arXiv:2111.09883](https://arxiv.org/abs/2111.09883)
- **ViT (对比基础)**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021, [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **DeiT**: Touvron et al., "Training data-efficient image transformers & distillation through attention", ICML 2021
- **PVT (并行工作)**: Wang et al., "Pyramid Vision Transformer", [arXiv:2102.12122](https://arxiv.org/abs/2102.12122)
- **原版 Transformer**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- **ResNet (对比 backbone)**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016

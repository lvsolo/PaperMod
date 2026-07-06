---
title: "Traffic Light Base Model 设计文档"
author: "lvsolo"
date: "2026-06-09"
tags: ["model design", "autonomous driving", "traffic light", "3D detection", "tracking"]
ShowToc: true
TocOpen: true
---

# 1. 任务定义与需求

## 1.1 模型输出要求

| 输出任务 | 描述 | 形式 |
|---------|------|------|
| **2D 检测** | 多视角图像上的 2D 目标检测 | query-based, 输出 (cx, cy, w, h, class) |
| **3D 检测** | 3D 空间中的目标检测 | Sparse4D-based, 输出 (x,y,z,w,l,h,yaw, class) |
| **跟踪 (Tracking)** | 2D & 3D 的实例级跟踪 | instance_id, 帧间一致 |
| **交通灯信号识别** | 给定 ego 轨迹, 判断对应方向的信号状态 | {直行/左转/右转/掉头} → {红/黄/绿} |

## 1.2 核心设计约束

1. **2D → 3D 级联**: 2D 检测的 query / instance feature / 2D box 位置必须作为 3D instance anchor 的构造输入
2. **统一跟踪**: 2D 的跟踪信息需要传递给 3D，保证 ID 一致性
3. **交通灯为静态目标**: 无需速度/运动模型，但需要时序一致性来稳定信号状态
4. **One Model**: 所有任务在同一个模型中完成，共享 backbone 和大部分计算

## 1.3 参考方案

| 方案 | 核心思路 | 可借鉴之处 |
|------|---------|-----------|
| **UniAD** (CVPR 2023) | 统一 query 接口串联全栈任务 | 模块间 query 通信机制 |
| **Sparse4D v3** (Lin et al., 2023) | Sparse instance + 时序传播 + 跟踪 | 3D 检测与跟踪的基础框架 |
| **Deformable DETR** (Zhu et al., 2021) | Deformable attention + sparse sampling | 2D 检测的 query-based 方案 |
| **MUTR3D** (Zhang et al., 2022) | 多视角 3D tracking with queries | 2D+3D 统一 query 跟踪 |
| **Far3D** (Jiang et al., 2024) | 2D→3D query lifting for forward 3D detection | 2D query 提升到 3D 的具体方法 |
| **StreamPETR** (Wang et al., 2023) | 时序 streaming + PETR 位置编码 | 时序特征传播方案 |

### 1.3.1 Far3D 详解: 2D 辅助 3D 检测的代表性方案

Far3D (**Far**3D: Expanding the Horizon for Surround-view 3D Object Detection) 是我们 2D→3D Bridge 设计的直接灵感来源。它提出了一种**用 2D 检测结果来初始化 3D query** 的方法，专门解决远距离目标的 3D 检测问题。

**核心问题**: 纯 3D 的 sparse detection 方法（如 Sparse4D、PETR）对远距离小目标检测能力有限——远处车辆在图像上可能只有几个像素，3D 的可学习 anchor 很难覆盖到这些位置。但 2D 检测对小目标的检测能力通常更强（因为有更高分辨率的特征图）。

**Far3D 的解决方案**:

```
┌─────────────────────────────────────────────────────────┐
│                    Far3D Pipeline                        │
│                                                          │
│  ① 2D Detection:                                        │
│     Deformable DETR → 检测出 2D boxes                    │
│     每个 2D 检测有: instance_feature + 2D box + 深度预测   │
│                                                          │
│  ② 2D → 3D Query Initialization (核心创新):             │
│     对每个 2D 检测:                                      │
│     - 2D box center + predicted depth → 3D position      │
│     - instance_feature_2D → FC投影 → 3D query init      │
│     - 这些就是 "lifted 3D queries"                       │
│                                                          │
│  ③ 3D Decoder Refinement:                               │
│     lifted queries + learnable queries                   │
│     → Sparse4D 风格的 Decoder 精修                      │
│     → 输出最终 3D 检测结果                                │
└─────────────────────────────────────────────────────────┘
```

**Far3D 的关键设计细节**:

| 组件 | Far3D 做法 | 我们借鉴了什么 |
|------|----------|-------------|
| **2D 检测器** | Deformable DETR | ✅ 同样使用 Deformable DETR 作为 2D 分支 |
| **深度预测** | 从 2D feature 预测 depth distribution (LSS 风格) | ✅ 我们也用轻量 depth head，但更简单 (只预测中心深度) |
| **2D→3D 位置** | 2D center + depth → unproject 到 3D | ✅ 我们的 Anchor Lifting 完全借鉴此方法 |
| **特征传递** | 2D instance_feature 通过 FC 投影为 3D query | ✅ 我们的 Feature Transfer 用相同方式 |
| **3D Decoder** | deformable attention + iterative refine | ✅ 我们用 Sparse4D 的 Decoupled Attention |
| **时序处理** | 无（单帧方法） | ❌ 我们增加了 InstanceBank 时序传播 |
| **跟踪** | 无 | ❌ 我们增加了 Sparse4D 式跟踪 |
| **关注距离** | 远距离目标 (>50m) | 不同：我们关注交通灯 (远近都有) |

**我们与 Far3D 的关键区别**:

1. **Far3D 没有时序和跟踪**: Far3D 是单帧方法，没有 InstanceBank、没有帧间传播。我们结合了 Sparse4D v3 的时序机制
2. **Far3D 的目的是补强远距离检测**: 我们的目的更多是利用 2D 的高分辨率优势来引导 3D 初始化，同时实现 2D→3D 的 ID 传递
3. **Far3D 没有 2D→3D ID 传递**: Far3D 的 2D 和 3D 是独立的检测头，不做跟踪 ID 的级联传递。我们的 ID Transfer 是新增的设计

---

# 2. 整体架构

```
                Multi-view Images (N_cams)
                         │
                         ▼
              ┌─────────────────────┐
              │   Shared Backbone    │
              │  ResNet-50/VoVNet    │
              │     + FPN            │
              └──────────┬──────────┘
                         │
            Multi-scale Feature Maps (1/4, 1/8, 1/16, 1/32)
                         │
          ┌──────────────┴──────────────┐
          │                              │
          ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│  2D Detection    │          │  Camera Parameter     │
│  Branch          │          │  Encoding             │
│  (Deformable DETR│          │  (内参+外参 → embed)  │
│   Decoder, 6层)  │          └──────────┬───────────┘
│                  │                     │
│  输出:           │                     │
│  ├ 2D boxes      │                     │
│  ├ class labels  │                     │
│  ├ feat_2D (256d) │                     │
│  ├ track_emb(128d)│                    │
│  └ instance_id   │                     │
└────────┬─────────┘                     │
         │                               │
         ▼                               │
┌────────────────────────┐               │
│    2D → 3D Bridge      │               │
│                        │               │
│  ① Anchor Lifting      │               │
│    2D center + depth    │               │
│    → 3D anchor          │               │
│  ② Feature Transfer    │               │
│    feat_2D → feat_3D_init│               │
│  ③ ID Transfer         │               │
│    id_2D → id_3D         │               │
└────────┬───────────────┘               │
         │                               │
         ▼                               ▼
┌─────────────────────────────────────────────┐
│          3D Detection Branch                 │
│       (Sparse4D Decoder, 6层)                │
│                                              │
│  输入实例:                                    │
│   ① 从 2D 提升的实例 (≤N_lift)               │
│   ② 新初始化的可学习实例 (N_new)              │
│   ③ 时序传播实例 (来自 InstanceBank_3D)        │
│                                              │
│  每层 Decoder:                               │
│   temp_gnn → gnn → deformable → refine      │
│                                              │
│  输出:                                       │
│  ├ 3D anchor (x,y,z,w,l,h,yaw)              │
│  ├ class label (含 traffic_light 类)         │
│  ├ feat_3D (256d)                            │
│  ├ quality (centerness + yawness)            │
│  └ instance_id                              │
└──────────────────────┬───────────────────────┘
                       │
         ┌─────────────┼──────────────┐
         │             │              │
         ▼             ▼              ▼
  ┌─────────────┐ ┌──────────┐ ┌───────────────┐
  │ 3D 检测输出  │ │ TL Signal │ │ Tracking 输出  │
  │ (boxes,cls) │ │   Head    │ │ (instance_id) │
  └─────────────┘ └─────┬────┘ └───────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ 信号状态输出       │
               │ per turn-intent:  │
               │ {红/黄/绿}        │
               └──────────────────┘
```

---

# 3. 各模块详细设计

## 3.1 Shared Backbone + Feature Pyramid

**输入**: N_cams 个视角的图像，每张 H×W×3

**结构**: 标准的 ResNet-50 或 VoVNet-99 + FPN

**输出**: 4 个尺度的特征图 `{C3, C4, C5, C6}` (对应 1/4, 1/8, 1/16, 1/32 分辨率)

> 与 Sparse4D v3 共享相同的设计。FPN 输出统一通道数 256d，供 2D 和 3D 分支共同使用。

## 3.2 2D Detection Branch

### 3.2.1 结构

采用 **Deformable DETR** 的 decoder 结构：

```
N_2D 个可学习 queries
       │
       ▼
  ┌─────────────────────────────────┐
  │  Deformable DETR Decoder (6层)  │
  │                                 │
  │  每层:                          │
  │   cross-attn(query → 多尺度特征) │  ← deformable attention
  │   self-attn(query ↔ query)      │
  │   ffn                           │
  │   refine 2D anchor              │
  └─────────────┬───────────────────┘
                │
                ▼
  每个 query 输出:
   ├ anchor_2D: (cx, cy, w, h)       ← 2D 框位置
   ├ class_logits                    ← 分类得分
   ├ instance_feature_2D: (256d)      ← 语义特征
   ├ track_embedding: (128d)         ← 跟踪用 embedding
   └ confidence_score
```

### 3.2.2 2D 跟踪: InstanceBank_2D

直接沿用 Sparse4D 的跟踪机制：

```
帧 t-1 输出 → topk(K_2D) 按 confidence → InstanceBank_2D.cache()
                                                    │
帧 t: InstanceBank_2D.get() → K_2D 个时序 query        │
      + N_2D 个新可学习 query                          │
      → 拼接后送入 2D Decoder                        │
      → 前 K_2D 个输出继承 instance_id
```

> **关键**: 2D 跟踪使用和 Sparse4D v3 完全相同的机制 (topk + 置信度衰减 + 数组下标 ID 继承)，无需额外的跟踪头或跟踪 Loss。

### 3.2.3 track_embedding 的用途

`track_embedding` (128d) 是一个**可选的辅助特征**，不用于 2D 跟踪本身（2D 跟踪靠 Sparse4D 机制），而是作为附加信息传递给 3D 分支：

- **用途**: 当 2D→3D lifting 存在歧义时（如多个交通灯在同一视角重叠），track_embedding 提供额外的身份线索
- **训练**: 用对比学习 Loss 约束同一实例跨帧的 embedding 相似，不同实例的 embedding 远离（ReID-style）
- **是否必需**: 否。第一版可以不启用，仅用 ID 继承机制即可工作

## 3.3 2D → 3D Bridge (核心设计)

这是整个架构最关键的部分——如何将 2D 检测结果转化为 3D instance 的初始化输入。**核心思路借鉴自 Far3D**（详见 §1.3.1），但增加了时序传播和 ID 传递。

### 3.3.1 三个核心操作

```
2D 检测输出                         3D Instance 初始化
━━━━━━━━━━━                         ━━━━━━━━━━━━━━━━

① Anchor Lifting:
  anchor_2D (cx,cy,w,h)  ──→  anchor_3D (x,y,z,w,l,h,yaw)

② Feature Transfer:
  feat_2D (256d)          ──→  feat_3D_init (256d)

③ ID Transfer:
  instance_id_2D          ──→  instance_id_3D  (直接继承)
```

### 3.3.2 ① Anchor Lifting: 2D 框 → 3D anchor

#### 3.3.2.1 整体流程

```
                    2D box (cx, cy, w, h)
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
              Depth Prediction  2D Size
              (轻量级 depth     (w, h)
               head, 从 feat_2D    │
               预测中心深度 d)     ▼
                    │        Heuristic 3D Size:
                    │        w_3D ≈ f × w_2D / d  (近似投影反算)
                    │        h_3D ≈ f × h_2D / d
                    ▼
              3D Position (x, y, z):
              x = (cx - ppx) × d / fx
              y = (cy - ppy) × d / fy
              z = d
                    │
                    ▼
              anchor_3D = (x, y, z, w_3D, l_3D, h_3D, yaw=0)
              (yaw=0: 交通灯近似竖直, 后续 decoder 精修)
```

> **"中心深度"是什么意思?** 就是 2D 检测框的中心点 `(cx, cy)` 对应的 3D 点到相机的深度。隐含假设：3D 物体的几何中心投影到 2D box 的中心附近。对于交通灯这种近似对称的目标，这个假设基本成立。

#### 3.3.2.2 深度预测: Depth Bins 分类 vs 直接回归

深度预测有两种主流做法：

```
方法 1: Depth Bins 分类 (LSS / Far3D 风格)  ← 推荐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  将深度范围 [d_min, d_max] 离散化为 D 个 bin
  例如: [1m, 3m, 6m, 10m, 15m, 25m, 40m, ..., 200m]  (D=64 或 128)

  DepthHead 输出:
    logits: (B, N, D)  → softmax → depth_probs: (B, N, D)
    每个 bin 的中心深度: bin_centers: (D,)  ← 预定义的固定值

  最终深度 (取期望):
    d = Σ(depth_probs[i] × bin_centers[i])   ← 所有 bin 的概率加权平均

  或者 (取 argmax):
    d = bin_centers[argmax(depth_probs)]       ← 最高概率的 bin 对应的深度

  Loss:
    L_depth = CrossEntropy(depth_probs, gt_bin_index)
    或
    L_depth = L1(d_expected, gt_depth)

方法 2: 直接回归 (简单但不够稳定)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DepthHead 直接输出连续值:
    output = (depth_mean, depth_log_var)  ← 预测高斯分布的均值和方差

  Loss:
    L_depth = L1(depth_mean, gt_depth)
    或 L2 (Gaussian NLL)
```

**为什么 Depth Bins 更好？**

| 方面 | Depth Bins (分类) | 直接回归 |
|------|-------------------|---------|
| **训练稳定性** | ✅ 分类任务天然更稳定，softmax 提供归一化的概率 | ❌ 直接回归深度值（范围大，0~200m）梯度不稳定 |
| **多模态表达** | ✅ 概率分布天然支持多模态（如遮挡边界处的深度歧义） | ❌ 单峰高斯只能表达一个深度 |
| **不确定性** | ✅ 分布的熵天然表达不确定性 | ⚠️ 需要额外预测 variance |
| **远距离精度** | ✅ 可以用非线性 bin 间距（近处密、远处疏），自适应精度 | ❌ 近远同等精度要求，远距离回归困难 |
| **Far3D 做法** | ✅ 使用此方法 | — |

#### 3.3.2.3 DepthHead 实现 (推荐: Depth Bins)

```python
class DepthHead(nn.Module):
    """从 2D instance feature 预测中心点深度 (Depth Bins 分类)"""
    def __init__(self, feat_dim=256, num_bins=64, d_min=1.0, d_max=200.0):
        super().__init__()
        self.num_bins = num_bins

        # 预定义 bin 中心深度 (非线性间距: 近密远疏)
        # 使用指数间距: 近处 1m 精度, 远处 ~10m 精度
        bin_edges = torch.linspace(0, 1, num_bins + 1)
        bin_edges = d_min + (d_max - d_min) * (bin_edges ** 2)  # 二次间距
        self.register_buffer('bin_centers', (bin_edges[:-1] + bin_edges[1:]) / 2)

        # 预测 logits: 每个实例在 D 个 bin 上的得分
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, num_bins),
        )

    def forward(self, feat_2d):
        """
        feat_2d: (B, N, 256) — 2D 实例特征
        Returns:
          depth: (B, N) — 预测的中心深度 (概率加权和)
          depth_logits: (B, N, D) — 原始 logits (用于 Loss 计算)
        """
        logits = self.fc(feat_2d)                          # (B, N, D)
        probs = logits.softmax(dim=-1)                      # (B, N, D)
        depth = (probs * self.bin_centers).sum(dim=-1)      # (B, N) 取期望
        return depth, logits

    def loss(self, logits, gt_depths):
        """
        logits: (B, N, D)
        gt_depths: (B, N) — GT 深度值
        """
        # 将 GT 深度映射到 bin index
        gt_bins = torch.searchsorted(self.bin_centers, gt_depths.clamp(
            min=self.bin_centers[0], max=self.bin_centers[-1]))
        gt_bins = gt_bins.clamp(0, self.num_bins - 1)
        return F.cross_entropy(logits.flatten(0, 1), gt_bins.flatten(0, 1))
```

> **非线性 bin 间距的理由**: 交通灯近处 (5~20m) 需要高精度 (~0.5m)，远处 (50~100m) 精度要求低 (~2~5m)。二次/指数间距让近处 bin 更密，远处更疏，自适应地分配精度。

#### 3.3.2.4 深度 → 3D 位置: 反投影

拿到深度 `d` 后，利用相机内参 `(fx, fy, ppx, ppy)` 将 2D 中心反投影到 3D：

```
相机坐标系下:
  X_cam = (cx - ppx) × d / fx
  Y_cam = (cy - ppy) × d / fy
  Z_cam = d

然后通过相机外参 (R, t) 转换到 ego 车体坐标系:
  X_ego = R @ [X_cam, Y_cam, Z_cam]^T + t

最终 3D anchor = (X_ego, Y_ego, Z_ego, w_3D, h_3D, l_3D, yaw=0)
```

#### 3.3.2.5 3D 尺寸的启发式估算

```
已知: 2D box (w_2D, h_2D 像素), 深度 d, 焦距 f, 交通灯先验尺寸

方法 1: 投影反算 (简单但噪声大)
  w_3D ≈ w_2D × d / fx
  h_3D ≈ h_2D × d / fy
  问题: 深度误差会放大到尺寸误差

方法 2: 类别先验 (推荐, 交通灯尺寸变化范围小)
  交通灯的 3D 尺寸比较固定:
    - 标准横杆: ~2m × 0.5m × 0.3m
    - 标准竖杆: ~0.3m × 1.5m × 0.3m
  可以用固定的先验尺寸作为初始值:
    w_3D, h_3D, l_3D = class_prior_size["traffic_light"]
  让 3D Decoder 后续精修

方法 3: 混合 (远期)
  α × 投影反算 + (1-α) × 类别先验
  α 根据深度置信度动态调整
```

> **第一版推荐方法 2 (类别先验)**: 交通灯尺寸变化范围远小于车辆/行人，使用固定先验尺寸最简单可靠。3D Decoder 会在后续层精修这些尺寸。

> **与 Far3D 的区别**: Far3D 也做了 2D→3D lifting，但它主要用于前向长距检测。我们的方案更侧重于将 2D 检测作为 3D 的先验引导，而非替代 3D 检测。

### 3.3.3 ② Feature Transfer: 2D 特征 → 3D 特征初始化

#### 3.3.3.1 FC 投影的本质

```
"FC 投影"  =  nn.Linear(256, 256)  即一个全连接层 (Linear Layer)

  feat_2D:  (B, N, 256)  ──→  Linear(256, 256)  ──→  feat_3D_init:  (B, N, 256)

  维度不变 (256d → 256d), 不是升维也不是降维
  本质是: 把特征从 "2D 语义空间" 线性映射到 "3D 语义空间"

  类比: 就像翻译——同一个意思 (语义信息), 从 2D 的"语言"翻译成 3D 的"语言"
        2D feature 说 "我在图像左上角看到一个红色物体"
        3D feature 说 "我在 3D 空间 (x=10m, y=-2m, z=30m) 处有一个目标"
        线性变换就是那个"翻译器", 让 3D Decoder 能理解 2D 传来的信息
```

> **为什么不是升维/降维?** 2D 和 3D 的 feature 维度设计为相同 (256d)，这样 3D Decoder 不需要为 lifted 实例做特殊维度适配，可以直接和时序实例、新实例在同一个 Decoder 中处理。

#### 3.3.3.2 两种 Feature Transfer 方案

```python
# 方案 A: 简单 FC 投影 (推荐, 第一版使用)
self.feat_transfer = nn.Linear(256, 256)

feat_3d_init = self.feat_transfer(feat_2d)  # (B, N, 256) → (B, N, 256)

# 方案 B: Cross-Attention (更强大但更重)
# 用 3D anchor 的投影位置做 deformable attention
# 从多视角特征图中采样，结合 feat_2d 作为 query
feat_3d_init = CrossAttn(query=feat_2d, key_val=multi_view_features)
```

**第一版推荐方案 A**: 简单 FC 投影。理由：
- 2D feature 已经包含了丰富的语义信息
- 3D Decoder 后续的 deformable aggregation 会进一步从图像提取特征
- FC 投影零额外计算开销（仅一个矩阵乘法）

### 3.3.4 ③ ID Transfer: 2D ID 直接继承给 3D

```
2D 实例列表 (按 instance_id 排序):
  [id=0, id=1, id=2, ..., id=K-1]
       │
       ▼  Lifting
3D lifted 实例 (前 M 个):
  [id=0, id=1, id=2, ..., id=M-1]    ← 直接继承 2D ID
       │
       + N_new 个新初始化 3D 实例:
  [id=new_0, id=new_1, ...]           ← 分配新 ID
```

> **这是最简洁的跟踪传递方式**: 2D 的 InstanceBank 维护 instance_id，lifted 到 3D 后直接继承。3D 分支自己也有 InstanceBank_3D 来传播时序信息。两者的 ID 命名空间统一。

### 3.3.5 核心问题: 多视角中的同一物体如何处理?

这是整个 2D→3D Bridge 设计中最容易困惑的问题：同一个交通灯在多个相机视角中被检测到，怎么知道它们是"同一个物体"? 它们的 2D 特征怎么变成 3D instance feature?

#### 3.3.5.1 问题全景

```
6 个环视相机的 2D 检测结果:

  cam_front_left:           cam_front:              cam_front_right:
  ┌──────────┐             ┌──────────┐            ┌──────────┐
  │  TL_A ■  │             │  TL_A ■  │            │  TL_B ■  │
  │  TL_B ■  │             │  TL_C ■  │            └──────────┘
  └──────────┘             └──────────┘
        ↑                        ↑
        └──── 同一个交通灯 ───────┘

问题:
  ① 怎么知道 cam_front_left 的 TL_A 和 cam_front 的 TL_A 是同一个交通灯?
  ② 如果分别 lifting, 得到两个 3D 实例, 怎么合并?
  ③ 合并后的 3D feature 用哪个视角的 2D feature?
```

#### 3.3.5.2 关键洞察: 2D 阶段不知道跨视角关联，3D Decoder 解决一切

```
核心思路:
━━━━━━━━
  2D 阶段: 每个相机独立检测, 不知道其他视角的结果
           → 不做跨视角关联, 每个 2D 检测独立 lifting

  3D 阶段: 所有的 lifted 实例在同一个 3D 空间中
           → 同一物体的多个 lifted 实例会落在 3D 空间的相近位置
           → 3D Decoder 的注意力机制自然处理

  为什么可行:
    因为 lifting 的第一步 (深度预测 + 反投影) 已经把不同视角的
    2D 检测反投影到了统一的 3D ego 坐标系。同一个交通灯在两个
    视角下反投影得到的 3D 位置会很接近（精度取决于深度预测质量）。
```

#### 3.3.5.3 逐步展开: 多视角 lifting 的完整过程

```
Step 1: 2D 检测 (每个相机独立)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  cam_front_left 的 2D Decoder 输出:
    TL_A: (cx=150, cy=80, w=30, h=40), feat_front_left (256d), conf=0.88

  cam_front 的 2D Decoder 输出:
    TL_A: (cx=400, cy=100, w=35, h=45), feat_front (256d), conf=0.95
    TL_C: (cx=600, cy=90, w=25, h=35), feat_front_2 (256d), conf=0.91

  cam_front_right 的 2D Decoder 输出:
    TL_B: (cx=50, cy=120, w=28, h=38), feat_front_right (256d), conf=0.87

  → 2D 阶段不知道 TL_A 在两个视角中都出现了
  → 每个 2D 检测都是独立的, 没有跨视角通信

Step 2: 每个视角独立做 2D→3D Lifting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  cam_front_left 的 TL_A:
    depth = 35.2m
    3D pos = (x=12.1, y=-3.5, z=35.2)   ← 通过 cam_front_left 的外参转到 ego 系
    feat_3D_init = FC(feat_front_left)

  cam_front 的 TL_A:
    depth = 34.8m
    3D pos = (x=11.8, y=-3.3, z=34.8)   ← 通过 cam_front 的外参转到 ego 系
    feat_3D_init = FC(feat_front)

  → 两个 lifted 实例的 3D 位置非常接近! (差距 < 0.5m)
  → 这是因为它们确实是同一个交通灯, 反投影到同一个 3D 位置

  cam_front 的 TL_C:
    depth = 50.1m
    3D pos = (x=15.0, y=-8.2, z=50.1)   ← 与 TL_A 的位置明显不同

  cam_front_right 的 TL_B:
    depth = 28.5m
    3D pos = (x=8.5, y=5.2, z=28.5)

Step 3: 所有 lifted 实例汇总到 3D 空间
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3D 空间中的 lifted 实例:

      3D 位置                    来源视角              feature
  ┌─────────────────────┬──────────────────┬──────────────────────┐
  │ (12.1, -3.5, 35.2)  │ cam_front_left   │ FC(feat_front_left)  │  ← TL_A #1
  │ (11.8, -3.3, 34.8)  │ cam_front        │ FC(feat_front)       │  ← TL_A #2 (重复!)
  │ (15.0, -8.2, 50.1)  │ cam_front        │ FC(feat_front_2)     │  ← TL_C
  │ (8.5,  5.2, 28.5)   │ cam_front_right  │ FC(feat_front_right) │  ← TL_B
  └─────────────────────┴──────────────────┴──────────────────────┘

  TL_A #1 和 TL_A #2 在 3D 空间中距离 < 0.5m → 它们是同一个物体
  但 2D 阶段不知道, 它们作为两个独立的 lifted 实例送入 3D Decoder
```

#### 3.3.5.4 3D Decoder 如何自然地合并重复实例

```
3D Decoder 的三层处理:
━━━━━━━━━━━━━━━━━━━━━

① Self-Attention (gnn): 实例间自注意力
  ─────────────────────────────────────
  TL_A #1 和 TL_A #2:
    - 位置接近 → anchor_embedding 相似
    - 在 self-attention 中会互相"看到"
    - TL_A #1 (conf=0.88) 和 TL_A #2 (conf=0.95) 竞争
    - 高置信度的 TL_A #2 会抑制 TL_A #1
    - 或者两者融合: attention 输出是所有实例的加权组合

  结果: Decoder 输出时, 两个 TL_A 合并为一个 (或只剩一个)

② Deformable Aggregation: 多视角图像特征采样
  ─────────────────────────────────────────────
  这是最关键的一步! 即使 2D feature 来自不同视角,
  3D Decoder 的 deformable aggregation 会重新从所有视角采样:

  对于 TL_A 的 3D 实例:
    1. 将 3D anchor 投影回所有相机视角
    2. 在 cam_front 的特征图上采样一组特征  feat_front
    3. 在 cam_front_left 的特征图上采样一组特征  feat_front_left
    4. 学习权重, 加权融合: feat_3D = w₁·feat_front + w₂·feat_front_left

  → 3D 实例的最终 feature 不是来自"某一个视角的 2D feature"
  → 而是从所有视角重新采样的融合特征!

  这意味着: 初始的 FC(feat_2d) 只是一个起点 (warm start),
  3D Decoder 会覆盖它, 用多视角融合的更丰富特征替换。

③ Refinement: anchor 精修
  ──────────────────────────
  TL_A #1 和 TL_A #2 的位置会被精修到同一个点
  最终只剩一个高置信度输出
```

#### 3.3.5.5 图解: 完整的多视角处理流程

```
┌────────────────────────────────────────────────────────────────────────┐
│                  多视角 2D 检测 → 3D 实例 完整流程                       │
│                                                                        │
│  cam_front_left    cam_front      cam_front_right                      │
│  ┌──────────┐     ┌──────────┐    ┌──────────┐                        │
│  │ 2D Det   │     │ 2D Det   │    │ 2D Det   │                        │
│  │ TL_A(■)  │     │ TL_A(■)  │    │ TL_B(■)  │                        │
│  │ TL_B(■)  │     │ TL_C(■)  │    └────┬─────┘                        │
│  └────┬─────┘     └────┬─────┘         │                              │
│       │                │               │                              │
│  ┌────┴─────┐    ┌────┴─────┐    ┌────┴─────┐                        │
│  │ Depth +  │    │ Depth +  │    │ Depth +  │  ← 各视角独立做         │
│  │ Unproj   │    │ Unproj   │    │ Unproj   │    depth bins + 反投影  │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘                        │
│       │                │               │                              │
│       ▼                ▼               ▼                              │
│  (12.1,-3.5,35.2) (11.8,-3.3,34.8) (8.5,5.2,28.5)                   │
│  (10.2,-4.0,30.1) (15.0,-8.2,50.1)                                   │
│       │                │               │                              │
│       └────────────────┼───────────────┘                              │
│                        ▼                                              │
│          ┌─────────────────────────────┐                              │
│          │  3D 空间中的 lifted 实例     │                              │
│          │  (5个, 有重复)              │                              │
│          │                             │                              │
│          │  TL_A×2  TL_B×1  TL_C×1     │                              │
│          └────────────┬────────────────┘                              │
│                       ▼                                               │
│          ┌─────────────────────────────┐                              │
│          │  3D Decoder (6层)           │                              │
│          │                             │                              │
│          │  Self-Attention:            │                              │
│          │   TL_A×2 互相竞争/融合      │                              │
│          │                             │                              │
│          │  Deformable Aggregation:    │                              │
│          │   从所有视角重新采样特征      │                              │
│          │   (不再依赖初始 2D feature)  │                              │
│          │                             │                              │
│          │  Refine: 精修位置,去重       │                              │
│          └────────────┬────────────────┘                              │
│                       ▼                                               │
│          ┌─────────────────────────────┐                              │
│          │  最终 3D 输出 (去重后)       │                              │
│          │  TL_A(id=0)  ← 1个, 合并了  │                              │
│          │  TL_B(id=1)                 │                              │
│          │  TL_C(id=2)                 │                              │
│          └─────────────────────────────┘                              │
└────────────────────────────────────────────────────────────────────────┘
```

#### 3.3.5.6 回答核心问题的总结

| 问题 | 回答 |
|------|------|
| **2D 阶段知道跨视角关联吗?** | **不知道**。每个相机的 2D Decoder 独立检测，不跨视角通信 |
| **怎么知道不同视角的检测是同一物体?** | **2D 阶段不知道**。但 lifting 到 3D 后，同一物体的反投影位置自然接近 |
| **怎么合并?** | 3D Decoder 的 self-attention 让位置接近的实例竞争/融合，输出时只剩一个 |
| **合并后用谁的 2D feature?** | **两个都不用**。3D Decoder 的 deformable aggregation 从所有视角重新采样，生成融合的多视角 feature。初始的 FC(feat_2D) 只是 warm start |
| **如果深度预测不准，两个 3D 位置差很远怎么办?** | 这是 depth bins 的优势：概率分布让深度更稳定。但仍有风险，此时两个实例可能无法合并，靠 3D NMS 兜底 |

> **一句话总结**: 2D 阶段不做跨视角关联，每个 2D 检测独立 lifting 到 3D。重复在 3D 空间中自然暴露（位置接近），由 3D Decoder 的注意力机制 + 输出去重处理。3D 实例的最终特征不是来自某个视角的 2D feature，而是 3D Decoder 重新从所有视角采样融合的结果。

## 3.4 3D Detection Branch

### 3.4.1 输入实例的构成

每帧送入 3D Decoder 的实例由三部分组成：

```
实例构成 = ① 时序传播实例 (来自 InstanceBank_3D)
          + ② 2D 提升实例 (来自 2D→3D Bridge)
          + ③ 新初始化实例 (可学习)
```

| 来源 | 数量 | anchor | feature | instance_id |
|------|------|--------|---------|-------------|
| ① 时序传播 | K_3D (≤600) | 上一帧输出 + ego补偿 | 上一帧 decoder 输出 `.detach()` | 继承上一帧 |
| ② 2D 提升 | M (≤K_2D) | Bridge 的 anchor lifting 结果 | Bridge 的 feature transfer 结果 | 继承 2D ID |
| ③ 新初始化 | N_new (≈300) | 可学习 nn.Parameter | 全零 `new_zeros` | 本帧新分配 |

> **② 的特殊之处**: 这些实例的初始 feature **不是全零**（而是从 2D feature 投影来的），且 anchor **不是可学习参数**（而是从 2D box 计算来的）。这给了 3D Decoder 一个很好的起点。

### 3.4.2 Decoder 结构

沿用 Sparse4D v3 的 Decoupled Attention：

```
每层 Decoder:
  temp_gnn  → 时序注意力 (看历史帧传播实例)
  gnn       → 帧内自注意力 (看同伴实例)
  deformable → 图像交叉注意力 (从多视角特征提取)
  refine    → anchor 精修
```

**与原版 Sparse4D 的区别**:
- 输入中多了 "2D 提升实例"，它们的初始 feature 是有内容的（非零）
- Camera Parameter Encoding 注入权重预测的方式不变
- Quality Estimation (centerness + yawness) 保留

### 3.4.3 3D 跟踪: InstanceBank_3D

与 2D 跟踪完全相同的机制，独立维护：

```
3D Decoder 输出 (K_3D + M + N_new 个实例)
       │
       ▼
  confidence 排序 → topk(K_3D) → InstanceBank_3D.cache()
                                          │
下一帧: InstanceBank_3D.get() → K_3D 个时序实例
```

### 3.4.4 去重: 2D 提升实例 vs 时序传播实例

**潜在问题**: 同一个交通灯可能同时出现在 "② 2D 提升实例" 和 "① 时序传播实例" 中，造成重复。

**解决方案**: 在 3D Decoder 的 self-attention (gnn) 中，同一物体的多个实例会自然地"竞争"——特征相似的实例会在 attention 中互相抑制。同时，在输出时用 NMS (3D IoU) 去重。

## 3.5 Traffic Light Signal Head

### 3.5.1 任务定义

给定 ego 车辆的规划轨迹 (包含转弯意图)，判断对应方向的交通灯信号状态：

```
Ego 轨迹: {直行, 左转, 右转, 掉头} × {对应方向的交通灯}
                ↓
信号状态: {红, 黄, 绿}
```

### 3.5.2 设计方案

```
3D 检测输出 (过滤 traffic_light 类)
       │
       ▼
┌──────────────────────────────────┐
│   TL-Candidate Selector         │
│                                  │
│   输入:                          │
│   - 3D detected TLs (位置+feature)│
│   - Ego 轨迹 waypoints           │
│   - 转弯意图 (turn intent)        │
│                                  │
│   操作:                          │
│   1. 投影 TL 到 BEV              │
│   2. 按 turn intent 筛选相关 TL:  │
│      - 直行: 前方 ±θ₁ 范围内的 TL │
│      - 左转: 左前方 ±θ₂ 范围内    │
│      - 右转: 右前方 ±θ₂ 范围内    │
│      - 掉头: 后方/对面方向         │
│   3. 距离加权: 近处 TL 权重更高    │
│                                  │
│   输出: 每个意图方向关联的 TL 特征 │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   Signal State Classifier        │
│                                  │
│   输入:                          │
│   - 关联的 TL instance_feature   │
│   - 从原图裁剪的 TL 区域特征      │
│     (用 3D→2D 投影得到 2D 框,     │
│      从 backbone 特征图 ROI Align)│
│                                  │
│   结构:                          │
│   concat(feat_3D, feat_crop)      │
│     → MLP → 3-way softmax        │
│     → {red, yellow, green}       │
│                                  │
│   输出: 每个意图方向的信号状态     │
└──────────────────────────────────┘
```

### 3.5.3 交通灯的特殊处理

| 特性 | 影响 | 处理方式 |
|------|------|---------|
| **静态** | 无速度/运动模型 | anchor 中 velocity 维度固定为 0，InstanceBank 传播时不做 ego+velocity 补偿 |
| **小目标** | 特征提取困难 | 用 ROI Align 从高分辨率特征图 (1/4) 裁剪精细特征 |
| **状态时变** | 红黄绿切换 | 跟踪保持时序一致性，用多帧 feature 融合判断当前状态 |
| **多灯板** | 一个灯组有多个灯头 | 2D 检测可检测到单个灯头，3D 聚合为灯组，按位置判断激活灯头 |
| **远距离** | 像素极少 | 2D 分支用多尺度特征 (小目标检测能力) → lifting 给 3D 一个粗略位置 |

## 3.6 One Model 审计: 是否满足端到端单模型要求?

### 3.6.1 设计约束回顾

本文档 §1.2 的第 4 条约束: **One Model** — 所有任务在同一个模型中完成，共享 backbone 和大部分计算。

目标: 以下三个核心功能在同一个 forward pass 中输出，无需模型外的额外模型/模块：

| 功能 | 输入 | 输出 |
|------|------|------|
| **2D 交通灯检测** | 多视角图像 | 2D boxes + class + confidence + instance_id |
| **3D 交通灯检测** | 2D 检测结果 + 时序信息 | 3D boxes + class + confidence + instance_id |
| **信号状态识别** | 3D 检测结果 + ego 轨迹 | {直行/左转/右转/掉头} → {红/黄/绿} |

### 3.6.2 逐环节可微性审计

```
┌────────────────────────────────────────────────────────────────────┐
│                Forward Pass 逐环节审计                               │
│                                                                     │
│  ① Shared Backbone (ResNet + FPN)                                   │
│     模型内: ✅ 纯 nn.Module     可微: ✅     外部依赖: 无             │
│                                                                     │
│  ② 2D Detection Branch (Deformable DETR Decoder, 6层)              │
│     模型内: ✅                  可微: ✅     外部依赖: 无             │
│     输出: 2D boxes, instance features, depth_probs, track_emb       │
│                                                                     │
│  ③ 2D→3D Bridge (1:1 lifting)                                      │
│     depth = E[depth_probs]         可微: ✅ (加权求和)               │
│     pos = unproject(cx,cy,depth)   可微: ✅ (矩阵乘法)              │
│     feat = FC(feat_2D)             可微: ✅ (Linear)                │
│     track_emb 保留                 可微: ✅ (恒等映射)               │
│     外部依赖: 相机内参/外参 (传感器标定, 不是模型)                    │
│                                                                     │
│  ④ 3D Detection Branch (Sparse4D Decoder, 6层)                     │
│     模型内: ✅                  可微: ✅     外部依赖: 无             │
│     含: concat(feat, track_emb) + FC → self-attention → refine     │
│                                                                     │
│  ⑤ TL Signal Head                                                  │
│     TL candidate selector          可微: ✅ (基于 3D 位置的筛选)     │
│     MLP classifier                 可微: ✅                         │
│     外部依赖: ⚠️ ego 轨迹/转弯意图 (来自规划模块)                    │
│                                                                     │
│  ─── 以上为单次 forward pass, 全部可微 ───                           │
│                                                                     │
│  ⑥ InstanceBank (帧间状态管理)                                      │
│     topk 选择 (按置信度)           可微: ⚠️ (排序选择, 不可微)       │
│     ego motion 补偿               可微: ✅ (矩阵乘法)               │
│     置信度衰减                     可微: ✅ (标量乘法)               │
│     ID 继承 (数组下标)             可微: ⚠️ (整数操作)              │
│     外部依赖: ⚠️ ego pose (来自定位模块)                             │
│     → Sparse4D/StreamPETR/MUTR3D 都有相同机制, 业界公认"算模型内部" │
│                                                                     │
│  ⑦ NMS (输出去重)                                                   │
│     模型内: ❌ (纯后处理)    可微: ❌                                │
│     → 所有检测模型都有, 不算"额外模型"                               │
│                                                                     │
│  ⑧ Hungarian Matching                                               │
│     模型内: ❌ (仅训练时用, 推理不需要)                              │
└────────────────────────────────────────────────────────────────────┘
```

### 3.6.3 审计结论

```
是否满足 One Model 条件? ✅ 是
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ 单次 forward pass 输出全部结果:
     → 2D 检测、3D 检测、跟踪 ID、信号状态 全部在一次推理中产生

  ✅ 共享 backbone 和大部分计算:
     → 2D 和 3D 分支共享 ResNet+FPN 特征
     → 2D 的输出直接作为 3D 的输入 (无中间存储/序列化)

  ✅ 端到端可训练:
     → L_total = λ₁·L_2D + λ₂·L_3D + λ₃·L_signal + λ₄·L_depth + λ₅·L_track_emb
     → 所有 Loss 对 backbone 参数都有梯度

  ✅ 无额外模型/模块:
     → 没有任何环节需要跑一个单独的模型
     → 没有规则引擎、没有外部匹配算法、没有 CRF 等

  ⚠️ 需要外部输入 (不是模型, 是传感器/其他模块的数据):
     → 相机图像: 传感器输入 (所有感知模型都需要)
     → 相机内参/外参: 传感器标定 (所有多视角感知都需要)
     → ego pose: 定位模块输出 (所有时序感知都需要)
     → ego 轨迹/转弯意图: 规划模块输出 (TL 信号识别特有)
     → 这些是 "输入数据", 不是 "额外模型"
```

### 3.6.4 对"不可微操作"的澄清

之前审计中列出了 topk、ID 继承、NMS 三个"不可微操作"，但实际上它们的性质完全不同：

```
① topk (torch.topk): 模型内部的一行 tensor 操作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  操作: 按 confidence 排序取前 K 个实例缓存到下一帧
  不可微的只有 indices, 选出的 values (confidence) 仍有梯度
  Sparse4D/StreamPETR/MUTR3D 都这样做, 没人把它当成"额外模块"
  → 就是 InstanceBank.update() 里的一行代码
  → 不算"模型外"操作

② ID 继承: 纯模型外 bookkeeping, 只消费模型输出
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  操作: 维护一个整数计数器, 给模型输出的实例分配/继承 ID
  跟模型内部计算零交互:
    - 模型的参数、梯度、forward pass 完全感知不到 ID 的存在
    - ID 不参与 Loss 计算
    - ID 不影响任何网络层的计算
  相当于给数据库记录分配主键——跟数据库引擎的计算无关
  → 完全在模型外, 不应该出现在 One Model 审计里

③ NMS: DETR-like 架构设计上不需要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DETR / Deformable DETR 的设计初衷: 用 set prediction 替代 NMS
    → N 个 query 各自预测一个物体
    → Hungarian matching 做一对一匹配
    → 每个 query 天然只对应一个物体
    → 不需要 NMS!

  我们的方案:
    2D 分支 (Deformable DETR): 不需要 NMS ✅
    3D 分支 (Sparse4D Decoder): 不需要 NMS ✅

  跨视角重复怎么处理?
    → self-attention + track_embedding 让同一物体的多个 instance 互相融合
    → 3D Decoder 的 set prediction 会让冗余实例变成低置信度
    → 训练好的模型不会输出重复检测
    → 如果发现有多余输出 → 说明模型没训好, 不是架构缺陷

  → NMS 不是必需的后处理, 如果模型训好了就不需要
```

**修正后的结论：**

```
模型 forward pass 内部:     全部可微 ✅
模型内部 tensor 操作:       topk (一行代码, Sparse4D 同款) ✅
模型外 bookkeeping:        ID 继承 (只消费输出, 不影响计算) 📋
不需要的:                  NMS (DETR 设计上不需要) ❌

→ 模型本身是纯粹的端到端可微 forward pass
→ 没有任何不可微操作影响梯度计算
→ 没有任何额外模块需要单独运行
```

  ✅ 共享 backbone 和大部分计算:
     → 2D 和 3D 分支共享 ResNet+FPN 特征
     → 2D 的输出直接作为 3D 的输入 (无中间存储/序列化)

  ✅ 端到端可训练:
     → L_total = λ₁·L_2D + λ₂·L_3D + λ₃·L_signal + λ₄·L_depth + λ₅·L_track_emb
     → 所有 Loss 对 backbone 参数都有梯度

  ✅ 无额外模型/模块:
     → 没有任何环节需要跑一个单独的模型
     → 没有规则引擎、没有外部匹配算法、没有 CRF 等

  ⚠️ 需要外部输入 (不是模型, 是传感器/其他模块的数据):
     → 相机图像: 传感器输入 (所有感知模型都需要)
     → 相机内参/外参: 传感器标定 (所有多视角感知都需要)
     → ego pose: 定位模块输出 (所有时序感知都需要)
     → ego 轨迹/转弯意图: 规划模块输出 (TL 信号识别特有)
     → 这些是 "输入数据", 不是 "额外模型"

  ⚠️ 不可微但业界标准的操作:
     → InstanceBank 的 topk + ID 继承: Sparse4D/MUTR3D/StreamPETR 同款
     → NMS: 所有检测模型标配
     → Hungarian Matching: 仅训练时, 推理不需要
```

---

# 4. Tracking 方案对比与选型

## 4.1 三种方案对比

```
方案 A: 纯 Sparse4D 式 (ID 继承)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2D 跟踪: InstanceBank_2D, 数组下标继承 ID
  2D→3D:   直接继承 2D 的 instance_id
  3D 跟踪: InstanceBank_3D, 数组下标继承 ID

  优点: 零额外 Loss, 零额外参数, 最简洁
  缺点: 2D→3D lifting 失败时, ID 断链不可恢复

方案 B: Track Embedding 辅助
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2D 跟踪: 同方案 A
  2D→3D:   传递 track_embedding (128d) 给 3D feature
  3D 跟踪: 同方案 A, 但 3D 实例额外输入 track_embedding

  优点: track_embedding 提供身份线索, 对遮挡/重叠有鲁棒性
  缺点: 需要额外的对比学习 Loss 训练 embedding

方案 C: UniAD 式全 query 通信
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2D 跟踪: 同方案 A
  2D→3D:   2D query 直接作为 3D cross-attention 的 key/value
           (不是 lifting, 而是让 3D 去读取 2D 的信息)
  3D 跟踪: 同方案 A

  优点: 3D 可以灵活选择从哪些 2D query 获取信息
  缺点: 计算量大, 2D 和 3D 耦合更紧
```

## 4.2 推荐方案: A + B 渐进式

**第一版 (方案 A)**: 纯 Sparse4D 式 ID 继承，不引入 track_embedding

**第二版 (方案 A+B)**: 加入 track_embedding 作为辅助特征，用对比学习 Loss 训练

理由：
1. 方案 A 已经足够工作——Sparse4D v3 在动态目标上验证了 ID 继承的有效性
2. 交通灯是静态的，跟踪比动态目标更简单，ID 继承的成功率更高
3. 方案 B 作为增强，可以在方案 A 验证可行后再添加

## 4.3 ID 分配机制详解

### 4.3.1 统一 ID 命名空间

2D 和 3D 分支共享同一个 ID 命名空间，确保跨任务的实例一致性：

```
全局 ID 分配规则:
━━━━━━━━━━━━━━━━━━
  ID 是一个单调递增的整数计数器，由模型外的 InstanceBank 维护

  ┌──────────────────────────────────────────────────┐
  │  InstanceBank_2D          InstanceBank_3D         │
  │                                                    │
  │  cached_ids: [0,3,7]      cached_ids: [1,2,5,8]   │
  │                                                    │
  │  next_id: 9               next_id: 9               │
  │  ↑ 共享计数器，确保不冲突                           │
  └──────────────────────────────────────────────────┘

  注意: 两个 Bank 各自维护缓存，但 next_id 全局递增
  2D 分配了 id=0,3,7 → 3D 的下一个新 ID 从 9 开始
  3D 的 2D-lifted 实例继承 2D 的 ID (如 id=0,3)
```

> **为什么不用两个独立的 ID 空间？** 如果 2D 用 "2D-0, 2D-1..." 而 3D 用 "3D-0, 3D-1..."，下游任务（如信号识别）需要额外做 2D↔3D ID 映射，增加复杂度。统一命名空间让 2D 和 3D 对同一交通灯使用同一个 ID。

### 4.3.2 ID 生命周期

每个实例 ID 经历完整的生命周期：

```
ID 生命周期:
━━━━━━━━━━━

  ① 创建 (Birth)
     新的可学习 query / 新的 3D 实例首次被检测到
     → 分配新的全局唯一 ID (next_id++)
     
  ② 传播 (Propagation)
     实例被 InstanceBank 缓存 → 下一帧作为时序实例送入 Decoder
     → 继承上一帧的 ID (无需重新分配)
     
  ③ 活跃 (Active)
     Decoder 输出该实例的置信度 ≥ 阈值 → 保留在输出中
     
  ④ 衰减 (Decay)
     连续 N 帧置信度下降 → InstanceBank 中的 confidence *= decay_factor
     → 可能被新的高分实例挤出 topk
     
  ⑤ 死亡 (Death)
     实例未被选入 topk → 从 InstanceBank 中移除
     → ID 永久废弃，不会被复用
```

### 4.3.3 具体的 ID 分配流程 (逐帧)

```
帧 t=0 (冷启动):
━━━━━━━━━━━━━━━━━
  2D Decoder:
    输入: N_2D=300 个可学习 query (无时序实例)
    输出: 检测到 5 个交通灯
    InstanceBank_2D.cache(): topk(K_2D=100) → 存入
    分配 ID: [0, 1, 2, 3, 4]
    next_id = 5

  2D→3D Bridge:
    从 5 个 2D 检测中选 top-M=3 个做 lifting
    继承 ID: [0, 1, 2]

  3D Decoder:
    输入: 3 个 lifted 实例 + N_new=300 个可学习实例 (无时序实例)
    输出: 检测到 4 个交通灯 (3 个来自 lifted, 1 个来自新实例)
    InstanceBank_3D.cache(): topk(K_3D=600) → 存入
    ID 列表: [0, 1, 2, 5]    ← id=5 是新分配的
    next_id = 6

──────────────────────────────────────────────────────

帧 t=1:
━━━━━━
  2D Decoder:
    输入: 100 个时序实例 (id=[0,1,2,3,4,...]) + 300 个新 query
    输出: 检测到 6 个交通灯 (5 个旧 + 1 个新)
    InstanceBank_2D.cache(): topk → 存入
    ID 列表: [0, 1, 2, 3, 4, 6]    ← id=6 是新分配的
    next_id = 7

  2D→3D Bridge:
    选 top-M=3 做lifting, 继承 ID: [0, 1, 4]

  3D Decoder:
    输入: 600 个时序实例 (id=[0,1,2,5,...]) + 3 个 lifted (id=[0,1,4]) + 300 个新 query
    输出: 检测到 5 个交通灯
    ⚠️ 注意: id=0 和 id=1 同时出现在时序和 lifted 中 → 需要去重 (见 §4.5)
    InstanceBank_3D.cache(): 去重后 topk → 存入
    next_id = 7 (无新检测)
```

### 4.3.4 交通灯 vs 动态目标的 ID 管理差异

| 方面 | 动态目标 (车/行人) | 交通灯 (本方案) |
|------|-------------------|----------------|
| **ID 稳定性** | 低：遮挡、离开视野、高速运动导致频繁断链 | **高**：位置固定，只要在视野内就能持续跟踪 |
| **ego 补偿** | 需要 ego_motion + velocity 双重补偿 | 只需 ego_motion 补偿（velocity=0） |
| **死亡原因** | 遮挡、离开视野、置信度下降 | **仅离开视野**（不会被遮挡太长时间） |
| **新实例来源** | 进入视野的新目标 | 仅初始冷启动阶段大量出现 |
| **ID 复用风险** | 低（目标一旦离开很难再回来） | **极低**（交通灯位置固定） |

> **核心结论**: 交通灯的 **时序 ID 管理** 比动态目标简单，方案 A (纯 ID 继承) 在时序维度上够用。但跨视角的实例去重和特征聚合是另一个层面的问题，见 §4.3.5。

### 4.3.5 ⚠️ 纯 ID 继承够不够? 跨视角问题的再思考

前面说"方案 A (纯 ID 继承) 足够"——这个结论在**时序维度**上是对的。但有一个被忽略的问题：**同一帧内，跨视角的 2D 检测如何聚合?**

#### 问题 1: 同一物体被 lift 成多个 3D query，特征没有聚合

```
场景: TL_A 同时出现在 cam_front 和 cam_front_left

当前方案的流程:
  cam_front:     TL_A → depth=35.2m → 3D pos (12.1, -3.5, 35.2) → feat = FC(feat_front)
  cam_front_left: TL_A → depth=34.8m → 3D pos (11.8, -3.3, 34.8) → feat = FC(feat_front_left)

  两个 lifted 实例:
    实例 1: pos=(12.1,-3.5,35.2), feat 来自 cam_front 的 2D 特征
    实例 2: pos=(11.8,-3.3,34.8), feat 来自 cam_front_left 的 2D 特征

  ⚠️ 问题:
    - 两个实例各自只包含"一个视角"的信息
    - 它们之间没有特征层面的聚合
    - 送入 3D Decoder 后, 虽然 self-attention 能让它们互相"看到",
      但初始特征是割裂的——相当于让 3D Decoder 从零开始做跨视角融合
    - 而实际上我们手里有"两个视角的丰富 2D 特征"，却没利用上
```

#### 问题 2: 多个 3D query 浪费 Decoder 容量 + 可能误检

```
3D Decoder 的输入实例数量是有限的 (时序 + lifted + 新 ≈ 900~1200)

  正常情况: 10 个交通灯 → 10 个 lifted 实例
  跨视角重复: 10 个交通灯 → 18 个 lifted 实例 (8 个被重复 lift)

  影响:
    ① 浪费容量: 18 个实例中有 8 个是冗余的, 占用了本该给其他目标的 slot
    ② 误检风险: 两个重复实例可能在 Decoder 中互相抑制, 导致都不如单个实例好
       → 原本能 conf=0.95 的目标, 因为被拆成两个 conf=0.70 的, 可能被误过滤
    ③ 误检风险: 如果两个实例的 3D 位置差距较大 (深度预测不准),
       3D Decoder 可能认为它们是两个不同的目标, 输出两个检测框
```

#### 问题 3: 更深层——这不仅是去重问题，是特征融合问题

```
核心矛盾:
  ─────────
  2D 检测阶段已经有了"同一个交通灯在不同视角下的丰富特征"

    cam_front:     feat_front (256d)     ← 正面拍的, 清晰, 可能看到灯的颜色
    cam_front_left: feat_front_left (256d) ← 侧面拍的, 可能看到灯的轮廓

  如果能把这两个特征融合成一个:
    feat_merged = Aggregate(feat_front, feat_front_left)  ← 信息更丰富!

  但当前方案:
    → 两个特征各自独立 FC 投影 → 各自变成独立的 3D query
    → 3D Decoder 的 deformable aggregation 虽然会重新从多视角采样,
      但这个采样是基于 3D anchor 的投影位置, 而不是基于 2D 的原始检测特征
    → 等于丢弃了 2D 阶段已有的跨视角信息
```

#### 修正后的结论

```
时序跟踪 (帧→帧): ID 继承 ✅ 够用
  交通灯是静态的, InstanceBank 的 topk + ID 继承机制足够稳定

跨视角聚合 (同一帧, 不同相机): 需要额外机制 ⚠️
  纯靠 3D Decoder 的 self-attention 做去重/融合是次优的:
    - 特征没有提前聚合, 3D Decoder 要从零融合 → 效果差
    - 冗余实例浪费 Decoder 容量 → 效率低
    - 深度预测不准时可能导致误检 → 可靠性差
```


#### 推荐的解决方案: Track Embedding + 3D Decoder Self-Attention 自然融合

经过多轮讨论，最终推荐的方案比之前的设计**大幅简化**——不做显式的跨视角聚类和 Bridge 层聚合，而是利用 Track Embedding + 3D Decoder 的 Self-Attention 自然完成跨视角融合。

##### 核心思路

```
不再做:
  ❌ 显式跨视角聚类 (Union-Find)
  ❌ Bridge 层的 CrossViewAggregator 模块
  ❌ 显式深度融合 + 特征融合

改为:
  ✅ 有多少个 2D 检测框，就 lift 多少个 3D instance (1:1 映射)
  ✅ 每个 3D instance 的 anchor 附带 track_embedding 作为身份信息
  ✅ 3D Decoder 的 self-attention 自然完成跨视角特征融合
  ✅ Track embedding 让 self-attention 知道哪些 instance 是同一个物体
```

##### 为什么 Self-Attention 能做跨视角融合？

```
Self-Attention 的本质: 让不同的 query 之间交换信息
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  3D Decoder 的 self-attention 输入:
    Instance A (来自 cam_front 的 TL_A):     feat_A + track_emb_A
    Instance B (来自 cam_front_left 的 TL_A): feat_B + track_emb_B
    Instance C (来自 cam_front 的 TL_C):      feat_C + track_emb_C

  注意力计算:
    Q_A = W_q × combined_feat_A    ← combined_feat 包含 track_emb 信息
    K_B = W_k × combined_feat_B
    K_C = W_k × combined_feat_C

    score(A→B) = Q_A · K_B^T / √d    ← track_emb 相似 → score 高
    score(A→C) = Q_A · K_C^T / √d    ← track_emb 不同 → score 低

    softmax([score(A→A), score(A→B), score(A→C)]) ≈ [0.50, 0.40, 0.10]
                                                     ↑ 自身   ↑ 同物体  ↑ 无关

  输出:
    out_A = 0.50 × V_A + 0.40 × V_B + 0.10 × V_C
                            ↑ 吸收了 cam_front_left 视角的特征!

    out_B = 0.40 × V_A + 0.50 × V_B + 0.10 × V_C
                        ↑ 吸收了 cam_front 视角的特征!

  → 两个 TL_A instance 通过 self-attention 完成了跨视角特征交换!
  → 不需要显式聚类, 不需要额外的 CrossViewAggregator!
  → 这是 Transformer self-attention 的本职工作!
```

##### Track Embedding 怎么融入 Self-Attention？

```
方式 1: Concat + FC 投影 (推荐, 最简单)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 在 3D Decoder 每一层的 refine 步骤中:

  # Step 1: lift 时把 track_embedding 附带到 instance 上
  # 对从 2D lift 来的实例:
  combined = concat(instance_feature, track_embedding)  # 256+128 = 384d
  decoder_input = self.fc_proj(combined)                  # 384d → 256d

  # 对非 lift 的实例 (时序传播/新初始化):
  # track_embedding 为全零 (128d)
  combined = concat(instance_feature, zeros(128))
  decoder_input = self.fc_proj(combined)

  # Step 2: 送入 self-attention
  # 所有实例 (lifted + temporal + new) 的 decoder_input 维度一致 (256d)
  # Self-Attention 自动处理:
  #   - track_emb 相似的 → 高注意力 → 特征融合 (跨视角同一物体)
  #   - track_emb 为零的 → 正常注意力 (时序/新实例)
  output = self_attention(Q=decoder_input, K=decoder_input, V=decoder_input)

方式 2: 加性融合 (更轻量)
━━━━━━━━━━━━━━━━━━━━━━━━━
  track_emb_proj = Linear(128→256)(track_embedding)
  decoder_input = instance_feature + track_emb_proj

方式 3: 注意力偏置 (最精确, 可选)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 正常注意力分数
  scores = Q·K^T / √d

  # 额外的 track embedding 偏置
  track_bias = track_emb_i · track_emb_j^T   # 余弦相似度
  scores = scores + λ × track_bias            # λ 可学习

  → 显式地让 track_emb 相似的 pair 获得更高分数
```

> **第一版推荐方式 1 (Concat + FC)**: 简单、通用、不改变 Decoder 结构。`fc_proj` 就是一个 `nn.Linear(384, 256)`，让模型自己学习怎么利用 track_embedding 信息。

##### 完整的 Bridge + 3D Decoder 流程 (最终版)

```
2D 检测阶段 (各视角独立):
━━━━━━━━━━━━━━━━━━━━━━━━━
  cam_front:      [TL_A(box, feat, depth_probs, track_emb), TL_C(...)]
  cam_front_left:  [TL_A(box, feat, depth_probs, track_emb), TL_B(...)]
  cam_front_right: [TL_B(box, feat, depth_probs, track_emb)]

2D→3D Bridge (简单 1:1 lifting, 不做聚合):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  对每个 2D 检测:
    ① depth = E[depth_probs]    ← 概率分布取期望
    ② pos_3D = unproject(cx, cy, depth, cam_params)
    ③ feat_3D_init = FC(feat_2D) ← 256d → 256d
    ④ 把 track_embedding 保留在 instance 的属性里

  输出: N_lift 个 3D instance, 每个带有:
    ├ anchor_3D: (x,y,z,w,l,h,yaw)
    ├ instance_feature: (256d)
    ├ track_embedding: (128d)     ← 来自 2D, 保留不变
    ├ confidence: float
    └ instance_id: int

3D Decoder (6层 Decoupled Attention):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  输入实例:
    ① 时序传播实例 (InstanceBank_3D): K 个, track_emb = 缓存的上一帧值
    ② 2D lifted 实例: N_lift 个, track_emb = 来自 2D
    ③ 新初始化实例: N_new 个, track_emb = 全零

  每层 Decoder:
    # 准备输入: 把 track_emb 融入 instance_feature
    for instance in all_instances:
        combined = concat(instance.feature, instance.track_embedding)  # 384d
        instance.attn_input = fc_proj(combined)                        # 256d

    temp_gnn(attn_input)        # 时序注意力
    gnn(attn_input)             # 自注意力 ← 跨视角融合在这里发生!
    deformable(attn_input, img) # 图像交叉注意力
    refine(anchor, feature)     # 精修

  输出: 每个实例的精修后的 anchor + feature + confidence
    → 同一物体的多个实例经过 self-attention 后特征已经互相融合
    → 最终输出用 NMS 去重 (标准后处理)

输出后处理:
━━━━━━━━━━
  NMS (3D IoU > 0.3 的抑制)
    → 同一物体的多个 instance 位置已经收敛到相近位置 (通过 attention 精修)
    → NMS 去掉重复的, 保留最高置信度的
```

##### 为什么这个方案更好？

```
vs 之前的显式聚合方案:
━━━━━━━━━━━━━━━━━━━━━━

  | 方面         | 显式聚合 (CrossViewAggregator)     | Self-Attention 自然融合           |
  |--------------|-----------------------------------|----------------------------------|
  | 额外模块      | CrossViewAggregator (聚类+融合)    | 无 (复用已有 Decoder)             |
  | 聚类算法      | Union-Find + 距离/embedding 阈值   | 不需要显式聚类                    |
  | 可微性        | 聚类不可微 (硬决策)                 | ✅ 全程可微 (attention 是软加权)   |
  | 深度融合      | 需要手动设计加权方式                | attention 自动学习权重             |
  | 特征融合      | Cross-attention (额外参数)          | 复用 Decoder 的 self-attention    |
  | 输出去重      | 聚类后只剩一个 (可能丢失信息)        | NMS 后处理 (多个实例都经过了融合)  |
  | 实现复杂度    | 高 (200+ 行)                       | 低 (~10 行 concat + FC)           |
  | 灵活性        | 固定策略                            | 模型自己学最优策略                |

  核心优势: Self-Attention 是软加权 (softmax), 不是硬决策
    → 同一物体的多个实例会"部分融合"而非"完全合并"
    → 每个 instance 保留了独立的位置预测, 但吸收了其他视角的信息
    → 最后 NMS 选最好的那个
```

##### 对深度预测的处理

```
这个方案下, 深度预测怎么用?

  ① 每个 2D 检测独立预测 depth_probs
  ② 取期望深度, 独立反投影到 3D
  ③ 不做跨视角深度融合

  深度不准怎么办?
    - 同一物体的两个 lifted instance 的 3D 位置可能有差异
    - 但 3D Decoder 的 iterative refinement 会在 6 层 Decoder 中精修位置
    - Self-attention 让两个 instance 互相"拉"向更准确的共同位置
    - 最终 NMS 选高置信度的那个

  为什么不做深度融合?
    - 跨视角深度融合需要先做跨视角匹配 (先有鸡还是先有蛋)
    - 如果用 track_embedding 做匹配 → 等于又回到了显式聚类方案
    - 既然 self-attention 能处理, 就把深度融合也交给 Decoder
    - 更简单, 全程可微, 端到端学习
```

> **最终设计**: Bridge 层只做简单的 1:1 lifting (2D box → depth → 3D anchor + FC feature + track_embedding)，**不做显式聚类和融合**。所有跨视角的信息交互交给 3D Decoder 的 self-attention，通过 track_embedding 引导注意力权重。Track Embedding 在第一版就是必要的——不是用于时序 ReID，而是作为 self-attention 的**身份信号**，让 Decoder 知道哪些 instance 来自 2D、哪些是同一个物体。
#### 跨视角匹配: 主流论文怎么做的?

**关键发现**: 主流多视角 3D 检测方法**根本不做 per-view 2D 检测**，因此不存在跨视角匹配问题。

```
方案对比: 怎么处理多视角信息?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方案 A: 3D Query → 投影到所有相机 → 多视角采样 (主流做法)
  代表: DETR3D, PETR, Sparse4D, MUTR3D, BEVFormer

  ┌──────────┐
  │ 3D Query │ ← 一个 query 对应一个 3D 位置
  │  (1个)   │
  └────┬─────┘
       │ 投影到所有相机
       ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ cam_fl  │ │ cam_f   │ │ cam_fr  │
  │ 采样特征 │ │ 采样特征 │ │ 采样特征 │
  └────┬────┘ └────┬────┘ └────┬────┘
       └──────┬────┘──────────┘
              ▼
         加权融合特征 → 更新 query
         (天然多视角, 无需匹配)

  优点: 架构上消除了跨视角匹配问题
  缺点: 3D query 对小目标 (远处交通灯) 的初始位置可能不准

方案 B: 2D 检测 → Lifting → 3D Decoder 处理 (Far3D 做法)
  代表: Far3D

  每个 camera 独立 2D 检测 → 每个检测独立 lift 到 3D query
  → 所有 lifted queries + learnable queries → 3D Decoder
  → Hungarian matching 隐式去重

  优点: 2D 检测对小目标更友好 (高分辨率特征图)
  缺点: 跨视角重复需要 3D Decoder 隐式处理; Far3D 主要用于前向, 环视下问题更严重

方案 C: 2D 检测 → 跨视角聚合 → 3D Decoder (我们的方案)
  每个 camera 独立 2D 检测 → 跨视角聚类 + 深度/特征融合 → 3D Decoder

  优点: 结合方案 B 的 2D 小目标优势 + 显式跨视角聚合
  缺点: 需要跨视角聚类 (track embedding + 距离)
  与方案 B 的区别: 在送入 3D Decoder 前显式做聚合, 不依赖 Decoder 隐式去重

方案 D: 3D Query + 2D 辅助 Proposal (混合方案, 可考虑)
  3D 可学习 query (Sparse4D 式) 作为主体
  + 2D 检测结果作为额外的初始化 proposal (不是替代 3D query)
  → 3D query 天然多视角, 2D proposal 只用于补充远距离小目标

  优点: 不需要跨视角聚合 (3D query 天然处理), 同时享受 2D 小目标优势
  缺点: 2D proposal 如何融入 3D query 的设计需要仔细考虑
```

```
主流方案的多视角聚合方式对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| 方法        | 检测空间 | 多视角聚合方式                   | 跨视角匹配? |
|-------------|---------|---------------------------------|-----------|
| DETR3D      | 3D      | 3D ref point → 2D 采样 → concat | 不需要     |
| PETR        | 3D      | 3D 位置编码注入 2D feature      | 不需要     |
| StreamPETR  | 3D      | PETR + 时序 streaming            | 不需要     |
| MUTR3D      | 3D      | DETR3D 式 + 跨帧 query 传播     | 不需要     |
| Sparse4D    | 3D      | 4D keypoint → 多视角采样 → 融合 | 不需要     |
| BEVFormer   | BEV     | BEV grid → spatial cross-attention | 不需要  |
| Far3D       | 2D→3D   | 2D lift → 3D Decoder 隐式去重    | 隐式处理   |
| FCOS3D      | 2D      | 无聚合, 后处理 BEV NMS           | 需要 (NMS) |
| **本方案**  | **2D→3D**| **Track Emb 聚类 + 深度/特征融合** | **显式处理** |
```

> **重要反思**: 如果要彻底避免跨视角匹配问题，最干净的做法是采用**方案 D**——用 3D query (Sparse4D 式) 作为检测主体，2D 检测只作为辅助 proposal 补充远距离小目标。但当前方案 C 的设计也是可行的，前提是 track embedding 能提供可靠的跨视角聚类。

## 4.4 2D→3D 跟踪传递完整流程

### 4.4.1 正常流程 (无遮挡)

```
帧 t-1:
━━━━━━
  2D 检测: [TL_A(id=0, conf=0.95), TL_B(id=1, conf=0.88)]
    ↓ InstanceBank_2D 缓存

  2D→3D Bridge: lifting 两个实例
    ↓
  3D 检测: [TL_A(id=0, conf=0.92), TL_B(id=1, conf=0.85)]
    ↓ InstanceBank_3D 缓存

帧 t:
━━━━
  ① 2D 时序传播:
     InstanceBank_2D.get() → [TL_A(id=0), TL_B(id=1)] 作为时序 query
     + 300 个新可学习 query
     → 2D Decoder 输出: [TL_A(id=0), TL_B(id=1)]    ← ID 继承成功

  ② 2D→3D Bridge:
     从 2D 输出中 lifting → [TL_A(id=0), TL_B(id=1)]

  ③ 3D 时序传播:
     InstanceBank_3D.get() → [TL_A(id=0), TL_B(id=1)]

  ④ 3D Decoder 输入:
     时序 [id=0, id=1] + lifted [id=0, id=1] + 300 新 query
     → 去重后输出: [TL_A(id=0), TL_B(id=1)]    ← ID 传递成功 ✅
```

### 4.4.2 边缘情况 1: 2D 漏检但 3D 时序保持

```
帧 t-1:
  2D: [TL_A(id=0, conf=0.95)]     ← 正常检测到 TL_A
  3D: [TL_A(id=0, conf=0.92)]

帧 t: TL_A 被卡车短暂遮挡
  2D: []                            ← 2D 完全漏检 (遮挡)
      InstanceBank_2D 的时序实例 [id=0] 置信度衰减但仍存在
      → 2D Decoder 输出可能仍包含 [id=0] (低置信度)
      或者 2D 完全丢失 id=0

  2D→3D Bridge:
      没有 2D 检测结果可以 lifting → lifted 实例为空

  3D 时序传播:
      InstanceBank_3D 仍有 [id=0]   ← 3D 的时序记忆保留了 TL_A
      → 3D Decoder 输出 [TL_A(id=0, conf=0.70)]  ← 3D 独立维持了跟踪 ✅

  结果: 2D 断链, 但 3D 通过自己的 InstanceBank 保持了 ID
```

> **关键洞察**: 即使 2D→3D Bridge 断了，3D 的 InstanceBank 是独立的，可以自己维持跟踪。这就是为什么我们需要两套独立的 InstanceBank（而非只用一个）。

### 4.4.3 边缘情况 2: 3D 漏检但 2D 保持

```
帧 t: 3D Decoder 置信度不够，TL_A 被挤出 topk

  2D: [TL_A(id=0, conf=0.90)]      ← 2D 正常
  3D: InstanceBank_3D 丢失了 id=0  ← 3D 断链

帧 t+1:
  2D: [TL_A(id=0, conf=0.93)]      ← 2D 仍然跟踪
  2D→3D Bridge: lifting TL_A(id=0) ← 从 2D 重新传给 3D
  3D: 无 id=0 的时序实例, 但有 id=0 的 lifted 实例
      → 3D Decoder 输出 [TL_A(id=0, conf=0.85)]  ← 2D 帮 3D 恢复了 ✅
```

> **关键洞察**: 2D→3D Bridge 不仅是初始化，还是一个**恢复通道**。3D 断链后，2D 的持续检测可以通过 Bridge 把 ID 重新传给 3D。

### 4.4.4 边缘情况 3: 2D 和 3D 同时丢失

```
帧 t: TL_A 完全出视野 (车辆驶过路口)
  2D: 无 TL_A → InstanceBank_2D 中 id=0 衰减 → 被挤出 topk → 死亡
  3D: 无 TL_A → InstanceBank_3D 中 id=0 衰减 → 被挤出 topk → 死亡

  → id=0 永久废弃, 不会再被复用

帧 t+k: 车辆倒车回到原路口, TL_A 重新出现
  → 作为新检测, 分配新 ID (如 id=100)    ← 与旧的 id=0 无关
```

> **设计选择**: 不做 ID 复用。交通灯离场后再入场被视为新实例。这是合理的，因为自动驾驶场景中很少出现"同一个交通灯消失后又出现"的情况（除非掉头）。

### 4.4.5 完整的数据流总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                      跟踪信息流 (每帧)                                │
│                                                                      │
│  ┌──────────────┐    topk    ┌──────────────┐    get()    ┌───────┐ │
│  │ 2D Decoder   │──────────→│InstanceBank  │───────────→│2D 时序 │ │
│  │ 输出         │            │    _2D       │            │query  │ │
│  └──────┬───────┘            └──────────────┘            └───┬───┘ │
│         │                                                  │     │
│         │ Bridge (lifting)                    ┌────────────┘     │
│         │         │                          │                   │
│         │         ▼                          ▼                   │
│         │   ┌──────────┐              ┌──────────┐              │
│         │   │Lifted    │              │2D Decoder│              │
│         │   │Instances │              │(下一帧)  │              │
│         │   │(带2D ID) │              └──────────┘              │
│         │   └────┬─────┘                                        │
│         │        │                                              │
│         │        ▼                                              │
│  ┌──────┴────────────────┐    topk    ┌──────────────┐         │
│  │ 3D Decoder            │──────────→│InstanceBank  │         │
│  │ 输出 (含lifted+时序+新)│            │    _3D       │         │
│  └───────────────────────┘            └──────┬───────┘         │
│                                               │ get()           │
│                                               ▼                 │
│                                        ┌──────────┐            │
│                                        │3D 时序   │            │
│                                        │instances │            │
│                                        └──────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

## 4.5 去重与冲突处理

### 4.5.1 问题定义

同一个交通灯可能在 3D Decoder 的输入中出现多次：

```
重复来源:

  ① 时序传播实例 (来自 InstanceBank_3D):
     TL_A(id=0) — 上一帧 3D Decoder 检测到并缓存

  ② 2D 提升实例 (来自 2D→3D Bridge):
     TL_A(id=0) — 本帧 2D 检测到并 lifting

  ③ 多视角重复:
     TL_A 在 cam_front 和 cam_front_left 都被 2D 检测到
     → 被 lifting 成两个 3D 实例 (id 相同但位置略有差异)

  结果: 3D Decoder 输入中可能有 2-3 个实例对应同一个 TL_A
```

### 4.5.2 去重策略: 三层防线

```
第一层: Decoder 内部注意力竞争 (隐式去重)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  在 3D Decoder 的 self-attention (gnn) 中:
  - 两个 ID 相同的实例共享相似的 anchor 位置和特征
  - attention 机制会让它们互相"看到"
  - 高置信度的那个会抑制低置信度的那个
  - 结果: Decoder 输出时通常只剩一个高置信度实例

  优点: 零额外计算, 自然发生
  缺点: 不是100%可靠, 可能两个都存活下来

第二层: 基于 ID 的输出合并 (显式去重)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3D Decoder 输出后, 对相同 ID 的实例做合并:

  for each unique_id in output_ids:
      duplicates = instances_with_id(unique_id)
      if len(duplicates) > 1:
          # 保留置信度最高的
          winner = max(duplicates, key=lambda x: x.confidence)
          # 或者: 取置信度加权的 anchor 平均
          merged_anchor = weighted_avg(duplicates, weights=confidences)
          output.append(Instance(id=unique_id, anchor=merged_anchor, ...))
      else:
          output.append(duplicates[0])

  优点: 简单可靠, 保证输出无重复
  缺点: 后处理, 不参与梯度计算

第三层: 3D NMS (兜底)
━━━━━━━━━━━━━━━━━━━━
  对最终输出做 3D NMS:
  - 按 confidence 降序排列
  - 依次遍历, 与已保留的实例计算 3D IoU
  - IoU > threshold (如 0.3) 的被抑制

  优点: 处理所有类型的重复 (包括 ID 不同但位置重复的)
  缺点: 标准 NMS, 阈值敏感
```

### 4.5.3 推荐实现: 第二层为主，第三层兜底

```python
def dedup_3d_output(instances, id_merge=True, nms_thresh=0.3):
    """
    instances: List[Instance], 每个 Instance 有:
      - instance_id: int
      - anchor_3d: (x,y,z,w,l,h,yaw)
      - confidence: float
      - feature: (256,)
    """
    if not instances:
        return instances

    # 第一步: 相同 ID 合并
    if id_merge:
        id_groups = defaultdict(list)
        for inst in instances:
            id_groups[inst.instance_id].append(inst)

        merged = []
        for uid, group in id_groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # 置信度加权平均 anchor, 取最高置信度的 feature
                weights = torch.tensor([g.confidence for g in group])
                weights = weights / weights.sum()
                avg_anchor = sum(w * g.anchor_3d for w, g in zip(weights, group))
                best_feat = max(group, key=lambda x: x.confidence).feature
                merged.append(Instance(
                    instance_id=uid,
                    anchor_3d=avg_anchor,
                    confidence=max(g.confidence for g in group),
                    feature=best_feat,
                ))
        instances = merged

    # 第二步: 3D NMS 兜底
    instances.sort(key=lambda x: x.confidence, reverse=True)
    keep = []
    for inst in instances:
        should_keep = True
        for kept in keep:
            iou = compute_3d_iou(inst.anchor_3d, kept.anchor_3d)
            if iou > nms_thresh:
                should_keep = False
                break
        if should_keep:
            keep.append(inst)

    return keep
```

### 4.5.4 多视角 2D 检测的去重

```
问题: 同一个交通灯在 cam_front 和 cam_front_right 都被检测到
  cam_front:     TL_A(id=0, box=[200,100,30,40], conf=0.93)
  cam_front_right: TL_A(id=0, box=[50,120,25,35], conf=0.87)

  2D→3D Bridge:
  从两个视角分别 lifting → 得到两个 3D anchor (位置略有差异)

解决方案:
━━━━━━━━━━
  方案 1 (推荐): 在 2D 端去重 — 对多视角的 2D 检测做跨视角去重后再 lifting
    - 将 2D box 用已知深度(或前一帧深度)投影到 3D BEV
    - 在 BEV 空间做距离聚类 (阈值如 1.0m)
    - 同一聚类保留置信度最高的 2D 检测做 lifting

  方案 2: 不做 2D 去重, 让 3D 端处理
    - 两个 lifted 实例都送入 3D Decoder
    - 通过 §4.5.2 的三层防线去重
    - 简单但增加 3D Decoder 的计算量

  方案 3 (远期): 跨视角 deformable attention 自然融合
    - 不在 2D 端去重
    - 3D Decoder 的 deformable aggregation 已经会从多视角采样
    - 同一个 3D 位置的多个 lifted 实例在 attention 中融合
    - 最优雅但依赖 Decoder 能力
```

> **第一版推荐方案 1**: 在 2D 端做简单去重后再 lifting。理由：减少送入 3D Decoder 的冗余实例，降低去重负担；BEV 投影去重逻辑简单，计算量可忽略。

---

## 4.6 Track Embedding 详解

### 4.6.1 传统方法中的 Track Embedding 是什么？

Track embedding 本质上是一个**身份特征向量**（类似人脸识别里的 face embedding）：

```
核心性质:
  同一个物体在不同帧 → embedding 距离近 (余弦相似度高)
  不同物体在任意帧   → embedding 距离远 (余弦相似度低)
```

在传统的 query-based 跟踪方法（MOTR、TrackFormer 等）中，模型的输出只有 (box, class)，**不知道"这一帧的车 A 是不是上一帧的车 B"**。Track embedding 就是额外给每个检测结果一个"身份证"。

### 4.6.2 传统方法中怎么生成 Track Embedding？

在检测 Decoder 之后，额外加一个**轻量级 MLP 头**，从 instance_feature 中提取 embedding：

```python
class TrackEmbeddingHead(nn.Module):
    """从 instance_feature 提取身份 embedding"""
    def __init__(self, feat_dim=256, embed_dim=128):
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, embed_dim),
        )

    def forward(self, instance_feature):
        # instance_feature: (B, N, 256) — Decoder 输出的实例特征
        embedding = self.fc(instance_feature)       # (B, N, 128)
        embedding = F.normalize(embedding, dim=-1)  # L2 归一化到单位球面
        return embedding
```

> **为什么 L2 归一化?** 归一化后，两个 embedding 的内积 = 余弦相似度 ∈ [-1, 1]，便于用阈值判定"是否是同一物体"。

### 4.6.3 传统方法中怎么训练 Track Embedding？

用**对比学习 Loss** 训练——拉近同一物体的 embedding，推远不同物体的 embedding：

```python
def track_embedding_loss(embeddings_t, embeddings_t1, instance_ids_t, instance_ids_t1):
    """
    embeddings_t:  帧 t 的 embedding (B, N, 128)
    embeddings_t1: 帧 t-1 的 embedding (B, N, 128)
    instance_ids_t: 帧 t 的全局 ID (B, N)
    instance_ids_t1: 帧 t-1 的全局 ID (B, N)

    核心: 同一 ID 的 embedding 拉近，不同 ID 的推远
    """
    # 计算帧间相似度矩阵
    sim = torch.matmul(embeddings_t, embeddings_t1.T)  # (N, N)
    sim = sim / temperature  # 温度缩放, 通常 temperature=0.07

    # 构建正样本掩码: instance_id 相同 = 正样本
    # positive_mask[i,j] = 1 如果 ids_t[i] == ids_t1[j]
    positive_mask = (instance_ids_t.unsqueeze(1) == instance_ids_t1.unsqueeze(0))

    # InfoNCE Loss:
    # 对于每个 embedding_t[i], 它的正样本是 embedding_t1 中 ID 相同的
    # 负样本是 ID 不同的
    # loss = -log( exp(sim_pos) / sum(exp(sim_all)) )
    ...
```

**直觉理解**：这个 Loss 就是在训练网络学习"区分不同物体"的能力。训练好之后，同一辆车的 embedding 在不同帧中会非常接近（余弦相似度 > 0.9），而不同车的 embedding 会很远（余弦相似度 < 0.3）。

### 4.6.4 传统方法中推理时怎么用 Track Embedding？

推理时的流程：

```
帧 t-1 检测结果:
  [车A: box₁, emb₁]  [车B: box₂, emb₂]  [车C: box₃, emb₃]    ← 有 embedding
       │
       ▼ 存入 track pool
  track_pool = [(id=0, emb₁), (id=1, emb₂), (id=2, emb₃)]

帧 t 检测结果:
  [车?: box₄, emb₄]  [车?: box₅, emb₅]  [车?: box₆, emb₆]    ← 新的 embedding
       │
       ▼ 计算相似度矩阵
                    帧 t-1
              id=0(emb₁)  id=1(emb₂)  id=2(emb₃)
  帧 t  emb₄  [0.92]      [0.15]      [0.08]     ← emb₄ 和 id=0 最像 → 是车A
        emb₅  [0.10]      [0.88]      [0.12]     ← emb₅ 和 id=1 最像 → 是车B
        emb₆  [0.05]      [0.09]      [0.91]     ← emb₆ 和 id=2 最像 → 是车C
       │
       ▼ 匈牙利匹配 (基于相似度矩阵)
  结果: emb₄→id=0, emb₅→id=1, emb₆→id=2
```

> **关键**: 传统方法**必须**用 embedding 做跨帧匹配，因为它没有 Sparse4D 那种"实例自带身份传播"的机制。每一帧的检测是独立的，只能事后靠 embedding 来"辨认"。

### 4.6.5 在我们的方案中，Track Embedding 具体怎么用？

**核心区别**: 我们的方案中，**主跟踪机制是 Sparse4D 式的 ID 继承**（不需要 embedding 做匹配）。Track embedding 只是一个**辅助信号**，帮助 3D 分支更好地利用 2D 的身份信息。

#### 具体使用方式: 作为 3D 实例的辅助输入特征

```
2D 分支输出:
  每个 2D 实例有:
    ├ instance_feature_2D (256d)   ← 语义特征
    ├ track_embedding (128d)      ← 身份特征
    └ anchor_2D (cx,cy,w,h)       ← 位置

        ↓ 2D→3D Bridge

3D lifted 实例初始化:
    ├ feat_3D_init = FC(instance_feature_2D)      ← 语义初始化
    ├ anchor_3D = Lift(anchor_2D)                  ← 位置初始化
    └ aux_track_emb = track_embedding            ← 直接传递, 不做变换
```

在 3D Decoder 的每一层中，track_embedding 的使用方式：

```python
# 在 refine 步骤中, 将 track_embedding 作为辅助输入
def refine_layer(instance_feature, anchor_embed, track_embedding):
    """
    instance_feature: (B, N, 256) — 当前实例特征
    anchor_embed:     (B, N, 256) — anchor 编码
    track_embedding:  (B, N, 128) — 来自 2D 的身份 embedding (仅 lifted 实例有)
    """
    # 对没有 track_embedding 的实例 (时序传播/新初始化), 用全零填充
    # track_embedding: lifted 实例有值, 其他全零

    # 方式 1: 拼接后 FC (最简单)
    combined = torch.cat([instance_feature, track_embedding], dim=-1)  # (B, N, 384)
    output = self.refine_fc(combined)  # (B, N, 256)

    # 方式 2: 加性融合
    # track_emb_proj = self.embed_fc(track_embedding)  # 128d → 256d
    # output = instance_feature + track_emb_proj

    return output, refined_anchor
```

#### Track Embedding 解决的具体问题

| 问题场景 | 不用 embedding (纯 ID 继承) | 用 embedding (辅助) |
|---------|--------------------------|-------------------|
| **2D→3D lifting 歧义**: 两个交通灯在同一视角很近，2D 分开了但 3D 初始位置几乎重叠 | 3D Decoder 可能混淆两个实例 | embedding 提供"它们是不同物体"的额外信号 |
| **遮挡后恢复**: 交通灯被卡车短暂遮挡，2D 漏检一帧，下一帧重新检测到 | ID 断链，分配了新 ID | 如果用 track embedding 做二次匹配，可以恢复旧 ID |
| **多视角融合**: 同一个交通灯在两个相机中都被检测到，lifted 成了两个 3D 实例 | 需要 NMS 后处理去重 | embedding 相似度可以辅助判断"这两个是同一物体" |

#### 在 InstanceBank 中的存储

```python
class InstanceBank3D(nn.Module):
    def __init__(self, ...):
        self.cached_feature = None      # (B, K, 256) — 实例特征
        self.cached_anchor = None        # (B, K, 8)   — 3D anchor
        self.cached_confidence = None    # (B, K)      — 置信度
        self.cached_track_emb = None     # (B, K, 128) — 来自 2D 的 track embedding
        # ↑ 新增字段: 缓存 track embedding, 传播到下一帧
```

### 4.6.6 训练时的 Loss

```python
# 整体 Loss 中新增一项
L_track_emb = InfoNCE(embeddings_frame_t, embeddings_frame_t1, instance_ids)

# 权重通常较小, 因为这是辅助 Loss
# λ₅ ≈ 0.1 ~ 0.5 (远小于检测 Loss 的权重)
```

### 4.6.7 为什么推荐第一版不用 Track Embedding?

| | 不用 (方案 A) | 用 (方案 A+B) |
|---|---|---|
| **额外参数** | 0 | ~33K (TrackEmbeddingHead) |
| **额外 Loss** | 0 | InfoNCE, 需要调参 |
| **数据需求** | 不需要跟踪 ID 标注训练 embedding | 需要跨帧全局 ID 标注 |
| **开发复杂度** | 低 | 中 |
| **跟踪效果** | 对静态目标 (交通灯) 足够 | 对遮挡/密集场景更好 |

> **建议路径**: 先用方案 A 跑通整个 pipeline → 验证 2D→3D 级联的可行性 → 如果发现跟踪在遮挡/密集场景有问题 → 再加入方案 B 的 track embedding。

---

# 5. 训练策略

## 5.1 Loss 设计

```
L_total = λ₁·L_2D_det + λ₂·L_3D_det + λ₃·L_TL_signal + λ₄·L_depth + λ₅·L_track_emb

其中:
  L_2D_det     = Focal(cls) + L1(bbox)                    ← 标准 DETR loss
  L_3D_det     = Focal(cls) + L1(anchor) + BCE(quality)   ← Sparse4D loss
  L_TL_signal = CE(signal_state)                          ← 3 分类交叉熵
  L_depth     = L1(predicted_depth, gt_depth)             ← 2D→3D bridge 深度监督
  L_track_emb = InfoNCE / TripletLoss (仅方案 B 启用)     ← 对比学习
```

## 5.2 为什么不能同时训练所有任务?

### 5.2.1 多任务同时训练的核心困难

```
如果一开始就把所有 Loss 加在一起从头训练:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  L_total = λ₁·L_2D + λ₂·L_3D + λ₃·L_signal + λ₄·L_depth + λ₅·L_track_emb

  问题 1: Loss 尺度冲突
    L_2D (Focal Loss) ≈ 1~5
    L_3D (L1 Loss)    ≈ 0.5~3
    L_signal (CE)     ≈ 0.1~1
    L_depth (L1)      ≈ 10~50  ← 深度的数值范围大, Loss 天然更大

    → 如果不调权重, L_depth 会主导梯度, 其他任务学不动
    → 但调权重本身是个玄学, 不同阶段最优权重不同

  问题 2: 梯度冲突 (Gradient Conflict)
    2D 检测的梯度: "让 backbone 关注图像中的物体区域"
    3D 检测的梯度: "让 backbone 关注深度/距离信息"
    信号识别的梯度: "让 backbone 关注颜色/灯光状态"

    → 同一个 backbone 参数, 不同任务要求它学习不同特征
    → 梯度方向可能相反, 互相抵消
    → 结果: 每个任务都学不好

  问题 3: 训练信号质量
    Stage 1 (随机初始化): 2D 检测输出是垃圾 → lifting 到 3D 也是垃圾
      → 3D Decoder 拿到垃圾输入 → 3D Loss 反传的梯度也是噪声
      → 没有任何任务能收敛

    正确做法: 先让 2D 检测收敛 → lifting 输出有意义的 3D 初始化
      → 3D Decoder 在好的起点上训练 → 级联地逐步收敛
```

### 5.2.2 行业标准做法: 渐进式训练 (Progressive Training)

```
核心思路: 像盖楼一样, 先打地基, 再一层层加
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  UniAD 的训练:
    Stage 1: 训检测 → Stage 2: 加跟踪 → Stage 3: 加预测 → Stage 4: 加规划

  Sparse4D v3 的训练:
    Stage 1: 训 3D 检测 → Stage 2: 加时序传播 → Stage 3: 加跟踪

  Far3D 的训练:
    Stage 1: 训 2D 检测 → Stage 2: 训 3D 检测 → Stage 3: 联合微调

  共同规律:
    ① 先训最基础的感知任务 (检测)
    ② 再加依赖基础任务的高级任务 (跟踪/信号识别)
    ③ 最后联合微调 (全部解冻, 用小学习率)
```

## 5.3 具体的训练阶段设计

### Stage 1: 2D 检测 (最基础)

```
训练内容:
  Shared Backbone + FPN + 2D Deformable DETR Decoder + DepthHead

Loss:
  L = L_2D_det + L_depth
    L_2D_det = Focal(cls) + L1(bbox)
    L_depth  = CE(depth_probs, gt_bin)

冻结: 无 (全部可学习)
数据: 2D 检测标注 + 深度标注

训练目标:
  - Backbone 学会提取多视角图像的通用特征
  - 2D Decoder 学会在多视角图像上检测交通灯
  - DepthHead 学会从 2D feature 预测深度

这个阶段结束后:
  ✅ 2D 检测能工作 (有意义的 2D boxes + features)
  ✅ 深度预测能工作 (有意义的 depth_probs)
  ✅ Backbone 已经学到了有用的特征 (边缘/颜色/形状/深度线索)
```

### Stage 2: 3D 检测 (在 2D 基础上)

```
训练内容:
  在 Stage 1 基础上, 加入 3D Decoder

冻结策略:
  ❄️ Freeze: Backbone + 2D Decoder + DepthHead (Stage 1 学好的不动)
  🔥 Train: 3D Decoder + Feature Transfer (FC) + Anchor Lifting

Loss:
  L = L_2D_det + L_depth + L_3D_det
                          L_3D_det = Focal(cls) + L1(anchor) + BCE(quality)

  注意: L_2D_det 和 L_depth 仍然计算 (保持不退化的信号),
        但因为 2D 分支冻结, 梯度只更新 3D 分支

数据: 3D 检测标注 (需要 2D-3D 关联, 即每个 3D GT 对应哪个 2D 检测)

训练目标:
  - 3D Decoder 学会从 lifted 3D instance 做 3D 检测
  - Feature Transfer (FC) 学会把 2D feature 翻译成 3D feature
  - 在 2D 提供的良好起点上做 3D 精修

这个阶段结束后:
  ✅ 3D 检测能工作
  ✅ 2D→3D Bridge 能工作
  ⚠️ 但 Backbone 还没学 3D 相关的特征 → 后续需要解冻微调
```

### Stage 3: 联合微调 (解冻 Backbone)

```
训练内容:
  解冻所有模块, 联合训练

冻结策略:
  🔥 全部解冻, 但用小学习率 (1/10 of Stage 1)
  🔥 Backbone 学习率更小 (1/100 of Stage 1) — 防止灾难性遗忘

Loss:
  L = L_2D_det + L_depth + L_3D_det + λ₅·L_track_emb
                                       ↑ 加入 track embedding 对比学习 Loss

数据: 全部标注 (2D + 3D + 深度 + 跟踪 ID)

训练目标:
  - Backbone 学会兼顾 2D 和 3D 的特征需求
  - 2D 和 3D 分支互相适应 (2D 可能微调输出以更适合 3D lifting)
  - Track embedding 开始学习身份区分能力

关键技巧 — 学习率差异化:
  Backbone:      lr = 1e-5   ← 最小, 已经学好了, 只微调
  2D Decoder:    lr = 5e-5   ← 小, 保持 2D 性能
  3D Decoder:    lr = 1e-4   ← 中, 继续精修
  Bridge (FC):   lr = 2e-4   ← 较大, 让 2D→3D 映射更准
  TrackEmb Head: lr = 2e-4   ← 较大, 新加入的模块
```

### Stage 4: 交通灯信号识别

```
训练内容:
  在 Stage 3 基础上, 加入 TL Signal Head

冻结策略:
  ❄️ Freeze: Backbone (已经足够好)
  🔥 Train: TL Signal Head (新模块) + 3D Decoder (微调)

Loss:
  L = L_2D_det + L_depth + L_3D_det + L_track_emb + L_TL_signal

数据: 交通灯信号状态标注 + ego 轨迹

训练目标:
  - TL Signal Head 学会从 3D TL feature + 裁剪特征 判断信号状态
  - 3D Decoder 可能微调 feature 以更适合信号识别

为什么信号识别放最后:
  - 信号识别依赖高质量的 3D 检测 (需要准确的 TL 位置和 feature)
  - 如果 3D 检测本身都不准, 信号识别无从学起
  - 所以先确保 3D 检测可靠, 再训信号识别
```

### Stage 5 (可选): 全量联合微调

```
训练内容:
  全部解冻, 小学习率, 长训练

Loss:
  L_total = λ₁·L_2D_det + λ₂·L_3D_det + λ₃·L_TL_signal + λ₄·L_depth + λ₅·L_track_emb

  权重调整:
    λ₁ (2D): 1.0    ← 基础任务, 权重正常
    λ₂ (3D): 2.0    ← 核心任务, 权重较大
    λ₃ (signal): 0.5 ← 辅助任务, 权重较小
    λ₄ (depth): 0.5  ← 辅助任务
    λ₅ (track_emb): 0.1~0.5 ← 正则化, 权重最小

训练目标:
  - 所有任务达到最佳平衡
  - Loss 权重需要 grid search 或用 GradNorm 等自适应方法
```

### 5.3.1 训练阶段总结图

```
时间线:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→

Stage 1         Stage 2          Stage 3          Stage 4        Stage 5
2D 检测         + 3D 检测         联合微调          + 信号识别      全量微调
(基础)          (冻结2D)         (解冻Backbone)    (冻结Backbone)  (全部解冻)

┌────────┐   ┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────┐
│Backbone│   │Backbone  │   │Backbone      │   │Backbone   │   │Backbone  │
│  🔥    │   │  ❄️      │   │  🔥(小lr)    │   │  ❄️       │   │  🔥(极小)│
├────────┤   ├──────────┤   ├──────────────┤   ├───────────┤   ├──────────┤
│2D Det  │   │2D Det    │   │2D Det        │   │2D Det     │   │2D Det    │
│  🔥    │   │  ❄️      │   │  🔥(小lr)    │   │  🔥(小lr) │   │  🔥(小lr)│
├────────┤   ├──────────┤   ├──────────────┤   ├───────────┤   ├──────────┤
│Bridge  │   │Bridge    │   │Bridge        │   │Bridge     │   │Bridge    │
│  🔥    │   │  🔥      │   │  🔥          │   │  🔥       │   │  🔥      │
├────────┤   ├──────────┤   ├──────────────┤   ├───────────┤   ├──────────┤
│3D Det  │   │3D Det    │   │3D Det        │   │3D Det     │   │3D Det    │
│  -     │   │  🔥      │   │  🔥          │   │  🔥       │   │  🔥      │
├────────┤   ├──────────┤   ├──────────────┤   ├───────────┤   ├──────────┤
│Signal  │   │Signal    │   │Signal        │   │Signal     │   │Signal    │
│  -     │   │  -       │   │  -           │   │  🔥       │   │  🔥      │
├────────┤   ├──────────┤   ├──────────────┤   ├───────────┤   ├──────────┤
│TrackEmb│   │TrackEmb  │   │TrackEmb      │   │TrackEmb   │   │TrackEmb  │
│  -     │   │  -       │   │  🔥          │   │  🔥       │   │  🔥      │
└────────┘   └──────────┘   └──────────────┘   └───────────┘   └──────────┘

🔥 = 训练   ❄️ = 冻结   - = 不存在   (小lr) = 小学习率
```

> **核心原则**: 先训基础任务（检测），再加高级任务（跟踪、信号识别）。每个阶段在上一个阶段的良好输出上构建，避免"垃圾进垃圾出"。最后阶段全部解冻联合微调，让所有任务互相适应。

## 5.3 数据需求

| 数据 | 来源 | 标注格式 |
|------|------|---------|
| 2D 检测 | nuScenes / ONCE / 自采集 | 多视角 2D box (cx,cy,w,h,cls) |
| 3D 检测 | nuScenes / 自采集 | 3D box (x,y,z,w,l,h,yaw,cls) |
| 2D-3D 关联 | nuScenes (sample_annotation 有 2D/3D 映射) | 每个 3D box 对应的 2D box |
| 交通灯状态 | nuScenes (traffic_lights) | per-instance: {red,yellow,green} |
| 跟踪 ID | nuScenes (instance_name 跨帧) | 全局唯一 ID |
| Ego 轨迹 | 自车规划模块输出 / 录制 | waypoints + turn_intent |

---

# 6. 关键设计决策总结

## 6.1 为什么 2D 先于 3D (而非并行)?

```
并行方案: 2D 和 3D 同时从 backbone 特征中检测
         问题: 2D 信息无法帮助 3D; 3D 对小目标 (远处交通灯) 检测能力弱

级联方案: 2D 先检测 → lifting → 3D 在 2D 基础上精修
         优势: 2D 对小目标更友好 (更高分辨率); 3D 初始位置更准;
               跟踪 ID 自然从 2D 流向 3D
```

## 6.2 为什么交通灯是静态目标反而更简单?

| 方面 | 动态目标 (车/行人) | 静态目标 (交通灯) |
|------|-------------------|-------------------|
| Anchor 维度 | 11D (含 velocity) | 8D (无 velocity) |
| InstanceBank 传播 | ego补偿 + velocity补偿 | 仅 ego 补偿 |
| 跟踪难度 | 高 (运动+遮挡) | 低 (位置固定) |
| 独有挑战 | 无 | 信号状态时变 + 极小目标 |

## 6.3 与 UniAD 的区别

| | UniAD | 本方案 |
|---|---|---|
| **任务范围** | 检测+跟踪+建图+预测+规划 | 检测+跟踪+信号识别 |
| **2D/3D 关系** | 独立 BEV 感知, 2D 不辅助 3D | **级联: 2D 辅助 3D** |
| **跟踪方式** | 跟踪头 + 匹配 | **Sparse4D 式 ID 继承** |
| **交通灯** | 不处理 | **核心任务** |
| **模块间通信** | 全 query 交互 | 轻量: 仅 2D→3D bridge |

---

# 7. 开放问题

1. **2D→3D 去重策略**: 同一目标同时出现在时序传播和 2D 提升实例中时，除了 attention 竞争外，是否需要显式的去重模块？
2. **交通灯关联策略**: 如何精确地将 3D 交通灯与 ego 轨迹的转弯意图关联？是否需要 HD Map 辅助？
3. **深度预测精度**: 仅从 2D feature 预测深度的精度是否足够？是否需要 dense depth 辅助监督？
4. **多帧信号融合**: 交通灯状态判断是否需要时序融合？如何处理状态切换时的延迟？
5. **2D→3D 失败恢复**: 当 2D 检测漏检时，3D 的新初始化实例能否独立发现该目标？

---

# 8. 参考资料

- **Sparse4D v3**: Lin et al., "Sparse4D v3: Advancing End-to-End 3D Detection and Tracking", arXiv:2311.11722
- **UniAD**: Hu et al., "Planning-oriented Autonomous Driving", CVPR 2023, arXiv:2212.10156
- **Deformable DETR**: Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021
- **MUTR3D**: Zhang et al., "Mutr3d: A multi-camera tracking framework via 3d-to-2d queries", CVPR 2023
- **Far3D**: Jiang et al., "Far3D: Expanding the Horizon for Surround-view 3D Object Detection", AAAI 2024, arXiv:2308.09616, [GitHub](https://github.com/megvii-research/Far3D)
- **StreamPETR**: Wang et al., "Exploring Object-Centric Temporal Modeling for High-Performance Spatio-Temporal Perception", ICLR 2024
- **BEVFormer**: Li et al., "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers", ECCV 2022

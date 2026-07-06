---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
author: "lvsolo"
date: "2026-06-11"
tags: ["paper reading", "attention", "GPU", "memory hierarchy", "tiling", "efficient computing"]
ShowToc: true
TocOpen: true
---

# 论文信息

- **标题**: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **作者**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- **机构**: Stanford University; University at Buffalo, SUNY
- **发表**: NeurIPS 2022
- **arXiv**: [2205.14135](https://arxiv.org/abs/2205.14135)
- **代码**: [github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)

> **一句话总结**: Transformer 的 attention 之所以慢，不是因为计算量 (FLOP) 大，而是因为**内存读写 (HBM access) 多**。FlashAttention 通过 **tiling (分块) + recomputation (重计算)** 两个经典技巧，让 attention 不再读写巨大的 N×N 矩阵到 HBM，从而在**不牺牲精度 (exact attention)** 的前提下获得 2-4 倍加速和更少的显存占用。

---

# 1. Introduction (引言)

## 1.1 问题: Transformer 在长序列上又慢又费内存

Transformer 的核心是 self-attention，其时间和内存复杂度都是 **序列长度 N 的二次方** O(N²)：

```
序列长度 N 翻倍 → 计算量 ×4, 内存 ×4

  N = 1024:  attention 矩阵 = 1024 × 1024 ≈ 1M 元素
  N = 8192:  attention 矩阵 = 8192 × 8192 ≈ 67M 元素
  N = 65536: attention 矩阵 = 65536 × 65536 ≈ 4B 元素 (4 billion!)
```

当序列变长，attention 矩阵 (N×N) 占用的内存爆炸式增长。

## 1.2 已有近似方法的缺陷

很多 **approximate attention** (近似注意力) 方法试图降低复杂度：

| 方法 | 类型 | 计算复杂度 | 问题 |
|------|------|-----------|------|
| **Reformer** | 稀疏 (hash) | O(N log N) | 理论快，实际没 wall-clock 加速 |
| **Linformer** | 低秩 | O(N) | 理论快，实际加速有限 |
| **Performer** | 低秩 (kernel) | O(N) | 精度损失大 |
| **Longformer/BigBird** | 稀疏 | O(N) | 实现复杂 |

**核心问题**: 这些方法都盯着 **FLOP (浮点运算量) 减少**，但 FLOP 减少不等于 wall-clock (实际运行时间) 加速。它们**忽略了内存访问 (IO) 的开销**。

## 1.3 关键洞察: IO-aware (关注内存读写)

```
现代 GPU 的计算速度 vs 内存速度:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  计算速度 (FLOP/s):  爆炸式增长 (摩尔定律, tensor core)
  内存带宽 (HBM):     增长缓慢

  → 结果: 大多数 Transformer 操作是 "memory-bound" (受限于内存带宽)
  → 即: GPU 大部分时间在等数据从 HBM 搬进搬出, 而不是在计算!

FlashAttention 的核心论点:
  不要只盯着减少 FLOP, 要减少 HBM 访问次数!
```

## 1.4 FlashAttention 的两大技巧

```
① Tiling (分块):
   把 Q, K, V 切成小块, 逐块加载到快速 SRAM 中计算
   → 避免把巨大的 N×N attention 矩阵写到 HBM

② Recomputation (重计算):
   反向传播时不存储中间的 N×N 矩阵, 而是从 Q, K, V 重新计算
   → 用增加 FLOP 的代价换取更少的 HBM 访问 (反而更快!)

结果 (Figure 1 右):
  GPT-2 (seq=1K): FlashAttention 比 PyTorch 实现 快 7.6 倍
```

> **看似矛盾**: 重计算增加了 FLOP，但反而更快——因为减少的 HBM 访问省下的时间远多于多算 FLOP 的时间。这就是 IO-aware 的精髓。

---

# 2. Background (背景)

## 2.1 Hardware Performance (硬件性能)

### 2.1.1 GPU 内存层级 (Memory Hierarchy)

GPU 有多种内存，**越小越快**。以 A100 为例：

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU 内存层级 (A100)                            │
│                                                                  │
│  ┌──────────────────────┐  最快                                 │
│  │  On-chip SRAM        │  带宽: ~19 TB/s                       │
│  │  (片上, 寄存器/L1)   │  容量: ~192 KB / SM × 108 SM          │
│  │                      │         (非常小!)                      │
│  └──────────┬───────────┘                                       │
│             │  数据搬运 (IO 的关键!)                              │
│  ┌──────────▼───────────┐                                       │
│  │  HBM (高带宽内存)     │  带宽: 1.5-2.0 TB/s                  │
│  │  (GPU 主显存)        │  容量: 40-80 GB                       │
│  │                      │  (PyTorch tensor 都在这)               │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│  ┌──────────▼───────────┐  最慢                                 │
│  │  DRAM (CPU 主存)     │  带宽: ~12.8 GB/s                     │
│  │                      │  容量: >1 TB                          │
│  └──────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘

  关键数据:
    SRAM 带宽 (19 TB/s) ≈ 10 × HBM 带宽 (1.5 TB/s)
    SRAM 容量 (192KB) ≪ HBM 容量 (40GB)    ← 小了 20 万倍!

  → SRAM 快但太小, HBM 大但慢
  → 算法的目标: 尽量在 SRAM 里算, 减少 HBM 来回搬运
```

> **论文 Figure 1 左半**: 展示了这个内存层级，SRAM 19TB/s、HBM 1.5TB/s、DRAM 12.8GB/s。

### 2.1.2 计算类型: Compute-bound vs Memory-bound

操作分为两类，用 **arithmetic intensity (算术强度)** 衡量 = 计算次数 / 内存访问字节数。

```
① Compute-bound (计算密集):
   时间由计算量决定, HBM 访问时间可忽略
   例子: 大矩阵乘法 (matmul), 大通道卷积
   特征: 算术强度高 (每搬 1 字节数据, 做很多次计算)

② Memory-bound (内存密集):
   时间由内存访问量决定, 计算时间可忽略
   例子: elementwise (激活函数, dropout), reduction (sum, softmax, layernorm)
   特征: 算术强度低 (每搬 1 字节数据, 只做很少计算)

  ⚠️ attention 里的 softmax, mask, dropout 都是 memory-bound!
  → 标准实现里它们反复读写 HBM, 成为瓶颈
```

### 2.1.3 Kernel Fusion (算子融合)

```
naive kernel fusion (朴素的算子融合):
  如果多个操作作用于同一输入, 可以只从 HBM 读一次输入, 在 SRAM 里做完所有操作

  标准做法 (无融合):                融合后:
  ┌────────┐                       ┌──────────────┐
  │ HBM    │                       │ HBM          │
  │  ↓ 读  │                       │   ↓ 读一次   │
  │ softmax│  ← 读+写 HBM          │ ┌──────────┐ │
  │  ↓     │                       │ │ softmax  │ │
  │ dropout│  ← 读+写 HBM          │ │ dropout  │ │  全在 SRAM
  │  ↓     │                       │ │ matmul   │ │
  │ matmul │  ← 读+写 HBM          │ └──────────┘ │
  └────────┘                       │   ↑ 写一次   │
                                   └──────────────┘

  问题: 训练时, 中间值还是要存到 HBM 给反向传播用
  → 朴素融合对训练帮助有限
  → FlashAttention 的解决方案见 §3 (用重计算绕过)
```

## 2.2 Standard Attention Implementation (标准 Attention 实现)

### 2.2.1 标准公式

输入 Q, K, V ∈ ℝ^(N×d) (N=序列长度, d=head 维度)，计算输出 O ∈ ℝ^(N×d)：

```
S = QK^T          ∈ ℝ^(N×N)     ← attention score 矩阵
P = softmax(S)    ∈ ℝ^(N×N)     ← softmax 按行做
O = PV            ∈ ℝ^(N×d)     ← 加权求和
```

### 2.2.2 标准实现的流程图 (Algorithm 0)

```
┌──────────────────────────────────────────────────────────────────┐
│           标准 Attention 实现的数据流 (Algorithm 0)                │
│                                                                   │
│   HBM                    SRAM                HBM                   │
│  ┌───────┐              ┌──────┐           ┌───────┐              │
│  │ Q, K  │──读(分块)──→ │ 计算 │──写 S ──→ │  S    │  ← N×N 矩阵  │
│  └───────┘              │QK^T  │           │ (N×N) │    物化到HBM  │
│                         └──────┘           └───┬───┘              │
│                                                │ 读               │
│                                                ▼                  │
│                         ┌──────┐           ┌───────┐              │
│                         │ 计算 │←──读 S─── │  P    │  ← N×N 矩阵  │
│                         │softmax│──写 P─→ │ (N×N) │    物化到HBM  │
│                         └──────┘           └───┬───┘              │
│                                                │ 读               │
│                                                ▼                  │
│  ┌───────┐              ┌──────┐           ┌───────┐              │
│  │ V     │──读(分块)──→ │ 计算 │←──读 P─── │  O    │              │
│  └───────┘              │ PV   │──写 O ──→ │ (N×d) │              │
│                         └──────┘           └───────┘              │
│                                                                   │
│  问题: S 和 P 这两个 N×N 矩阵被反复读写 HBM!                       │
│       - 写 S 到 HBM: 1 次                                         │
│       - 读 S 算 softmax: 1 次                                     │
│       - 写 P 到 HBM: 1 次                                         │
│       - 读 P 算 PV: 1 次                                          │
│       - (反向传播时还要再读写)                                     │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2.3 标准实现的内存与 IO 分析

```
内存占用:
  S, P 各占 O(N²) 内存    ← 随序列长度二次方增长!
  通常 N ≫ d (如 GPT-2: N=1024, d=64)

HBM 访问次数 (IO 复杂度):
  读 Q,K: O(Nd), 写 S: O(N²)
  读 S: O(N²), 写 P: O(N²)
  读 P: O(N²), 读 V: O(Nd), 写 O: O(Nd)
  ─────────────────────
  总计: Θ(Nd + N²) ≈ Θ(N²)    ← 二次方级 HBM 访问!

  ⚠️ 这就是标准 attention 慢的根源: N×N 矩阵在 HBM 来回搬运
```

> **论文 §2.2 关键结论**: 标准 attention 把 S, P 这两个 N×N 矩阵物化 (materialize) 到 HBM，导致 O(N²) 的内存占用和 O(N²) 的 HBM 访问。FlashAttention 的目标就是**避免物化这个大矩阵**。

---

# 3. FlashAttention: Algorithm, Analysis, and Extensions

## 3.1 An Efficient Attention Algorithm With Tiling and Recomputation

### 3.1.1 目标

**目标**: 计算 exact (精确) attention，但只用 sub-quadratic (次二次方) 的 HBM 访问，且不存储 N×N 中间矩阵用于反向传播。

### 3.1.2 技巧一: Tiling (分块) + 在线 Softmax

#### 核心挑战: Softmax 耦合了 K 的列

```
为什么 attention 难以分块计算?

  softmax(S) 第 i 行 = softmax(S_{i,1}, S_{i,2}, ..., S_{i,N})

  ↑ 算第 i 行的 softmax, 需要看到整行 (所有 N 个元素)
  ↑ 这就是 "softmax 耦合了列" — 必须有完整的行才能算 softmax

  如果把 K 分成两块 K=[K₁, K₂]:
    S = [QK₁^T | QK₂^T]   ← S 也分成两块 [S₁ | S₂]
    softmax([S₁ | S₂])    ← 算 S₁ 的 softmax 需要知道 S₂! (耦合)

  → 无法直接 "算完 S₁ 的 softmax 就丢掉"
  → 怎么办? → "在线 softmax" (online softmax) 技巧
```

#### 在线 Softmax 的数学原理

为了数值稳定性，softmax 用 max-shifting：

```
对向量 x ∈ ℝ^B:
  m(x) := max_i x_i                          ← 最大值 (用于数值稳定)
  f(x) := [e^{x_1 - m(x)}, ..., e^{x_B - m(x)}]   ← 减去 max 再 exp
  ℓ(x) := Σ_i f(x)_i                          ← 归一化常数 (分母)
  softmax(x) := f(x) / ℓ(x)
```

**关键引理**: 如果把向量 x 拼成两段 x = [x⁽¹⁾, x⁽²⁾] ∈ ℝ^(2B)，softmax 可以**分块增量计算**：

```
m(x) = m([x⁽¹⁾, x⁽²⁾]) = max(m(x⁽¹⁾), m(x⁽²⁾))     ← 全局 max = 两段 max 的 max

f(x) = [e^{m(x⁽¹⁾) - m(x)} · f(x⁽¹⁾) ,                ← 第一段: 用新旧 max 修正
        e^{m(x⁽²⁾) - m(x)} · f(x⁽²⁾)]                  ← 第二段: 用新旧 max 修正

ℓ(x) = e^{m(x⁽¹⁾) - m(x)} · ℓ(x⁽¹⁾) + e^{m(x⁽²⁾) - m(x)} · ℓ(x⁽²⁾)

softmax(x) = f(x) / ℓ(x)
```

**直觉**: 只要保留每段的两个统计量 **m (当前 max)** 和 **ℓ (当前归一化常数)**，就能一块一块地累加 softmax，最后得到正确结果。这就是 **online softmax / algebraic aggregation**。

```
在线 softmax 累加示意:
━━━━━━━━━━━━━━━━━━━━━

  初始: m = -∞, ℓ = 0, O = 0

  处理块 1 (K₁, V₁):
    S₁ = QK₁^T
    m_new = max(m, rowmax(S₁))           ← 更新全局 max
    ℓ_new = e^{m-m_new}·ℓ + e^{rowmax(S₁)-m_new}·rowsum(exp(S₁-m_new))
    O = (e^{m-m_new}·ℓ·O + exp(S₁-m_new)·V₁) / ℓ_new   ← 累加输出

  处理块 2 (K₂, V₂):
    S₂ = QK₂^T
    m_new = max(m, rowmax(S₂))
    ℓ_new = ...                          ← 同样的更新
    O = ...                              ← 继续累加

  ...
  处理完所有块 → O = softmax(QK^T)V   ← 正确的最终输出!
```

#### Tiling 的整体流程

```
┌──────────────────────────────────────────────────────────────────┐
│              FlashAttention Tiling 流程 (Figure 1 左半)            │
│                                                                   │
│  Q ∈ ℝ^(N×d),  K, V ∈ ℝ^(N×d)  全部在 HBM                         │
│                                                                   │
│  Outer Loop (外循环, 红色箭头): 遍历 K, V 的块                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  for j = 1 to T_c:    (K, V 分成 T_c 块)                  │    │
│  │                                                           │    │
│  │    ① 把 K_j, V_j 从 HBM 加载到 SRAM                       │    │
│  │       ┌─────────┐                                        │    │
│  │       │ HBM: K_j│ ─── copy block ──→ ┌─────────┐         │    │
│  │       │      V_j│                    │ SRAM    │         │    │
│  │       └─────────┘                    │ K_j, V_j│         │    │
│  │                                       └─────────┘         │    │
│  │    ② Inner Loop (内循环, 蓝色箭头): 遍历 Q 的块            │    │
│  │       for i = 1 to T_r:    (Q 分成 T_r 块)                │    │
│  │                                                           │    │
│  │         把 Q_i, O_i, ℓ_i, m_i 从 HBM 加载到 SRAM          │    │
│  │         ┌─────────┐                                       │    │
│  │         │ HBM: Q_i│ ─── copy block ──→ ┌─────────┐        │    │
│  │         │  O_i... │                    │ SRAM    │        │    │
│  │         └─────────┘                    │ Q_i,O_i │        │    │
│  │                                         │  K_j,V_j│        │    │
│  │         在 SRAM 内计算:                  └────┬────┘        │    │
│  │           S_ij = Q_i · K_j^T               │              │    │
│  │           在线 softmax 更新 m, ℓ           │              │    │
│  │           O_i += 更新                      │              │    │
│  │                                            │              │    │
│  │         把更新后的 O_i, ℓ_i, m_i 写回 HBM  │ copy block   │    │
│  │                                            ▼              │    │
│  │                                       ┌─────────┐         │    │
│  │                                       │ HBM: O_i│         │    │
│  │                                       └─────────┘         │    │
│  │       end for                                             │    │
│  │  end for                                                  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  关键: N×N 的 S, P 矩阵 NEVER 写到 HBM! (图中虚线框被阻止)         │
│       只在 SRAM 里临时存在, 用完即弃                                │
└──────────────────────────────────────────────────────────────────┘
```

#### Block Size (块大小) 的选择

```
SRAM 容量为 M, 需要让以下块都能放进 SRAM:

  K_j, V_j 块:  B_c × d 个元素  →  需要 B_c · d ≤ M
  Q_i, O_i 块:  B_r × d 个元素  →  需要 B_r · d ≤ M
  S_ij 块:      B_r × B_c 个元素 →  需要 B_r · B_c ≤ M

论文设置 (Algorithm 1 line 1):
  B_c = ⌊M / (4d)⌋                  ← K, V 的列块大小
  B_r = min(⌊M / (4d)⌋, d)          ← Q 的行块大小 (上界为 d)

  例: A100 SRAM M ≈ 100KB, d = 64
      B_c = 100KB / (4×64) ≈ 400 → 设为 128 或 256 (留余量)
      B_r = min(400, 64) = 64
```

### 3.1.3 技巧二: Recomputation (重计算)

#### 问题: 反向传播需要 S, P

```
标准反向传播需要:
  - S = QK^T ∈ ℝ^(N×N)   ← 算 dQ, dK 要用
  - P = softmax(S) ∈ ℝ^(N×N)  ← 算 dQ, dK, dV 要用

  → 如果不存这些, 反向传播没法算
  → 如果存这些, 又回到 O(N²) 内存 (违背初衷)
```

#### FlashAttention 的解决方案: 不存, 重算!

```
FlashAttention 的做法:
  前向传播时, 只存: O (输出), ℓ (归一化常数), m (max 统计量)
                  ↑ 这些都是 O(N) 大小 (线性!)

  反向传播时, 从 Q, K, V + (ℓ, m) 重新计算 S, P:
    S_ij = Q_i · K_j^T       ← 重算, 在 SRAM 里分块做
    P_ij = diag(ℓ_i)^{-1} · exp(S_ij - m_i)   ← 用存的 ℓ, m 还原 P

  → 不需要存 N×N 矩阵!
  → 多了重算的 FLOP, 但因为都在 SRAM 里做, HBM 访问少, 反而更快
```

```
重计算 vs 标准反向传播 的对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  标准反向传播:
    前向: 存 S, P 到 HBM      [O(N²) 内存]
    反向: 从 HBM 读 S, P      [O(N²) HBM 读]
    FLOP: 少
    问题: 内存大 + HBM 访问多

  FlashAttention (重计算):
    前向: 只存 O, ℓ, m 到 HBM  [O(N) 内存]
    反向: 重算 S, P 在 SRAM    [O(N²/M) HBM 访问]
    FLOP: 多 (重算了一遍 matmul)
    优势: 内存小 + HBM 访问少 → 实际更快!

  ⚠️ 这是一种 selective gradient checkpointing (选择性梯度检查点)
     不同于传统 checkpointing (用速度换内存),
     FlashAttention 的重计算反而更快 (因为减少了 HBM 访问)
```

> **论文 Figure 2 左**验证了这点: FlashAttention 的 GFLOP (75.2) 比标准 attention (66.6) 高 (因为重计算)，但 HBM 读写 (4.4GB) 远少于标准 (40.3GB)，所以 runtime (7.3ms) 远快于标准 (41.7ms)。

### 3.1.4 实现细节: Kernel Fusion (算子融合)

```
Tiling 使得 FlashAttention 可以用一个 CUDA kernel 完成:

  ┌───────────────────────────────────────────────────┐
  │           单个 Fused CUDA Kernel                  │
  │                                                   │
  │  从 HBM 读 Q, K, V 块                             │
  │         ↓                                         │
  │  ┌─────────────────────────────────────┐          │
  │  │  全在 SRAM 内, 不写回 HBM:           │          │
  │  │   1. matmul:  S = QK^T              │          │
  │  │   2. (可选) mask                    │          │
  │  │   3. softmax (在线, 分块)           │          │
  │  │   4. (可选) dropout                 │          │
  │  │   5. matmul:  O = PV               │          │
  │  └─────────────────────────────────────┘          │
  │         ↓                                         │
  │  只把最终的 O 写回 HBM                             │
  └───────────────────────────────────────────────────┘

  → 中间结果 (S, P) 从不写 HBM
  → 避免了反复读写, 这就是 "fused kernel"
```

### 3.1.5 Algorithm 1: FlashAttention (简化版, 论文正文)

```
Algorithm 1: FlashAttention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: Q, K, V ∈ ℝ^(N×d) (在 HBM), SRAM 大小 M

 1: 设块大小  B_c = ⌊M / (4d)⌋,  B_r = min(⌊M / (4d)⌋, d)

 2: 初始化 O = 0 ∈ ℝ^(N×d),  ℓ = 0 ∈ ℝ^N,  m = -∞ ∈ ℝ^N   (在 HBM)

 3: Q 分成 T_r = ⌈N/B_r⌉ 块 (每块 B_r×d)
    K, V 各分成 T_c = ⌈N/B_c⌉ 块 (每块 B_c×d)

 4: O, ℓ, m 也对应分块

 5: for j = 1 to T_c:                          ← 外循环: 遍历 K, V 块
 6:    把 K_j, V_j 从 HBM 加载到 SRAM

 7:    for i = 1 to T_r:                       ← 内循环: 遍历 Q 块
 8:       把 Q_i, O_i, ℓ_i, m_i 从 HBM 加载到 SRAM

 9:       在 SRAM 内: S_ij = Q_i K_j^T ∈ ℝ^(B_r × B_c)

10:       在 SRAM 内:                          ← 在线 softmax
            m̃_ij = rowmax(S_ij) ∈ ℝ^(B_r)
            P̃_ij = exp(S_ij - m̃_ij) ∈ ℝ^(B_r × B_c)   (逐元素)
            ℓ̃_ij = rowsum(P̃_ij) ∈ ℝ^(B_r)

11:       在 SRAM 内: 更新全局统计量
            m_new_i = max(m_i, m̃_ij)
            ℓ_new_i = e^{m_i - m_new_i}·ℓ_i + e^{m̃_ij - m_new_i}·ℓ̃_ij

12:       把更新后的输出写回 HBM:
            O_i = diag(ℓ_new_i)^{-1} · [diag(ℓ_i)·e^{m_i - m_new_i}·O_i
                                        + e^{m̃_ij - m_new_i}·P̃_ij V_j]
                    ↑ 把旧 O 按 max 修正系数 rescale, 再加上新块的贡献

13:       把 ℓ_i ← ℓ_new_i, m_i ← m_new_i 写回 HBM

14:    end for
15: end for
16: 返回 O
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  理解 line 12 (最关键的一步):
    旧 O_i 是基于旧 max m_i 算的, 现在来了新块, max 变成 m_new_i (可能更大)
    → 旧 O_i 要乘 e^{m_i - m_new_i} 修正 (因为 exp 的基准变了)
    → 再加上新块的贡献 e^{m̃_ij - m_new_i}·P̃_ij·V_j
    → 最后除以新的归一化常数 ℓ_new_i
```

### 3.1.6 Theorem 1: 正确性与复杂度

> **Theorem 1**: Algorithm 1 返回 O = softmax(QK^T)V，需要 O(N²d) FLOP，额外内存仅 O(N)（除输入输出外）。
>
> - **FLOP**: O(N²d) — 和标准 attention 一样 (没有减少计算量!)
> - **额外内存**: O(N) — 只存 ℓ, m 统计量 (线性!)

---

## 3.2 Analysis: IO Complexity of FlashAttention (IO 复杂度分析)

### 3.2.1 Theorem 2: HBM 访问次数

> **Theorem 2**: 设 N=序列长度, d=head 维度, M=SRAM 大小, 且 d ≤ M ≤ Nd。
>
> - 标准 attention (Algorithm 0): **Θ(Nd + N²)** 次 HBM 访问
> - FlashAttention (Algorithm 1): **Θ(N²d²/M)** 次 HBM 访问

```
为什么 FlashAttention 更少?
━━━━━━━━━━━━━━━━━━━━━━━━━

  外循环遍历 K, V 的 T_c 块:
    T_c = N / B_c = N / (M/d) = Nd/M      ← 外循环次数

  每次外循环, 把整个 Q (N×d) 从 HBM 读一遍到 SRAM:
    每次读 Θ(Nd) 元素

  总 HBM 访问 = T_c × Θ(Nd) = (Nd/M) × Nd = Θ(N²d²/M)

  对比标准: Θ(Nd + N²) ≈ Θ(N²)

  当 d² < M (即 d < √M, 通常成立):
    N²d²/M < N²   →  FlashAttention 更少!

  例 (A100, M≈100KB, d=64):
    标准: Θ(N²) = Θ(N²)
    Flash: Θ(N²·64²/100000) ≈ Θ(N²·0.04) = Θ(0.04·N²)
    → HBM 访问减少约 25 倍 (理论), 实测约 9 倍 (Figure 2)
```

### 3.2.2 证明直觉

```
证明 FlashAttention IO 复杂度的核心思路:

  ① SRAM 大小 M, 所以每个 K, V 块大小 = Θ(M) (即 B_c·d = Θ(M))
  ② K, V 共有 N 行 → 分成 N/B_c = N·d/M 块
  ③ 对每个 K, V 块 (外循环一次), 要遍历所有 Q 块 → 读完整个 Q (Θ(Nd))
  ④ 总访问 = (N·d/M) 次外循环 × Θ(Nd) = Θ(N²d²/M) ∎

  类似地, 反向传播 (Algorithm 4):
    标准: Θ(Nd + N²)
    Flash: Θ(N²d²/M)
```

### 3.2.3 Proposition 3: 下界 (无法再改进)

> **Proposition 3**: 对所有 M ∈ [d, Nd]，不存在能用 o(N²d²/M) 次 HBM 访问计算 exact attention 的算法。
>
> **即**: FlashAttention 的 IO 复杂度是**渐近最优**的，没有 exact attention 算法能在所有 SRAM 大小下做得更好。

```
证明 (反证法):
  假设存在算法, 对所有 M 用 o(N²d²/M) 访问

  取 M = Θ(Nd):
    则访问次数 = o(N²d²/(Nd)) = o(Nd)

  但 Q, K, V (输入) 和 O (输出) 大小都是 Nd, 它们一开始在 HBM
  → 任何正确算法至少要读输入写输出 → Ω(Nd) 访问

  o(Nd) < Ω(Nd)  ← 矛盾! ∎
```

### 3.2.4 IO 复杂度对比总结

```
┌────────────────────────────────────────────────────────────┐
│              IO 复杂度对比 (Figure 2 左的数据)              │
│                                                             │
│  (GPT-2 medium: N=1024, d=64, 16 heads, batch=64, A100)    │
│                                                             │
│  指标              标准 attention    FlashAttention         │
│  ──────────────────────────────────────────────             │
│  GFLOPs            66.6              75.2   (Flash 多,重算) │
│  HBM R/W (GB)      40.3              4.4    (Flash 少 9×)  │
│  Runtime (ms)      41.7              7.3    (Flash 快 5.7×) │
│                                                             │
│  结论: HBM 访问量是 runtime 的决定因素!                      │
│       Flash 虽然 FLOP 多, 但 HBM 访问少, 所以更快            │
└────────────────────────────────────────────────────────────┘
```

### 3.2.5 Block Size 对性能的影响 (Figure 2 中)

```
块大小 B_c 越大 → 外循环次数 T_c 越少 → HBM 访问越少 → 越快

  但是:
    - 块大小受限于 SRAM 容量 (不能超过 M)
    - 超过一定大小后, 被 compute-bound 主导, 加速不再明显
    - 太大放不进 SRAM

  Figure 2 中间图:
    B_c = 64 →  慢 (外循环多, HBM 访问多)
    B_c = 128 → 较快
    B_c = 256 → 快 (HBM 访问少)
    B_c = 512 → 没明显提升 (已被其他因素主导, 且接近 SRAM 上限)
```

---

## 3.3 Extension: Block-Sparse FlashAttention (块稀疏)

### 3.3.1 动机

很多场景用稀疏 attention (只算部分 query-key 对)，如 Longformer/BigBird。FlashAttention 可以作为基础，把稀疏 attention 也加速。

### 3.3.2 块稀疏的定义

```
给定块稀疏 mask M̃ ∈ {0,1}^(N×N) (块形式):
  M̃_{kl} = 1 → 计算 S_{kl}
  M̃_{kl} = 0 → 跳过 (置 -∞)

要求 M̃ 是块结构的:
  对块大小 B_r × B_c, 存在 M ∈ {0,1}^(⌈N/B_r⌉ × ⌈N/B_c⌉)
  使得 M̃_{kl} = M_{ij}, 其中 i = ⌊k/B_r⌋, j = ⌊l/B_c⌋

  即: 稀疏以块为单位, 整块保留或整块丢弃

  示意 (B_r = B_c = 2):
    M̃ (N×N):              M (⌈N/B⌉ × ⌈N/B⌉):
    ┌────────────┐        ┌─────┐
    │■ ■ □ □ ■ ■│        │1 0 1│
    │■ ■ □ □ ■ ■│   ←→   │1 0 1│
    │□ □ ■ ■ □ □│        │0 1 0│
    │□ □ ■ ■ □ □│        │0 1 0│
    │■ ■ □ □ ■ ■│        └─────┘
    │■ ■ □ □ ■ ■│
    └────────────┘
    ■ = 计算,  □ = 跳过
```

### 3.3.3 Algorithm 5: Block-Sparse FlashAttention

```
和 Algorithm 1 几乎一样, 唯一区别:
  在内循环开始处加一个判断:
    if M_{ij} ≠ 0:   ← 只算非零块
       执行原来的计算
    else:
       跳过该块

  → 零块完全不读不写 HBM, 不计算
```

### 3.3.4 Proposition 4: 块稀疏的 IO 复杂度

> **Proposition 4**: 设 s = 非零块比例。Block-sparse FlashAttention 需要 **Θ(Nd + N²d²s/M)** 次 HBM 访问。

```
对比:
  Dense FlashAttention: Θ(N²d²/M)
  Block-sparse:         Θ(N²d²s/M)     ← 乘以稀疏比例 s

  s 越小 (越稀疏), HBM 访问越少

  常见稀疏设置:
    s = N^{-1/2}        → Θ(N^{3/2})     (如 Sparse Transformer)
    s = 1/log(N)        → Θ(N²/log N)    (如 BigBird)

  Figure 2 右图验证:
    sparsity 越高 (非零块越少), 运行时间线性下降
    block-sparse FlashAttention 比 dense FlashAttention 快 2-4 倍
```

> **关键**: block-sparse 把稀疏的好处直接体现到 IO 复杂度上——这是其他近似 attention 方法做不到的 (它们稀疏了但实现上有 overhead，加速不显著)。

### 3.3.5 Butterfly Sparsity (蝴蝶稀疏)

论文实验中用 **fixed butterfly sparsity pattern** [Dao et al.]:
- 蝴蝶矩阵结构，已被证明能近似任意稀疏模式
- 固定稀疏模式贯穿训练
- 可看作 "fixed lottery ticket" (固定彩票)

---

# 4. Experiments (实验)

## 4.1 Faster Models with FlashAttention (训练加速)

### 4.1.1 BERT (Table 1)

```
BERT-large (seq=512), 达到 72.0% MLM 精度的训练时间 (8×A100, 平均10次):

  实现                     训练时间 (分钟)
  ─────────────────────────────────────
  Nvidia MLPerf 1.1        20.0 ± 1.5
  FlashAttention (ours)    17.4 ± 1.4    ← 快 15%!

  → 比 MLPerf 训练速度记录还快 15%
```

### 4.1.2 GPT-2 (Table 2)

```
GPT-2 在 OpenWebText 上的训练时间 (8×A100):

  模型                实现              PPL    训练时间 (加速比)
  ─────────────────────────────────────────────────────────────
  GPT-2 small         HuggingFace      18.2   9.5 天 (1.0×)
  GPT-2 small         Megatron-LM      18.2   4.7 天 (2.0×)
  GPT-2 small         FlashAttention   18.2   2.7 天 (3.5×)   ← 最快
  ─────────────────────────────────────────────────────────────
  GPT-2 medium        HuggingFace      14.2   21.0 天 (1.0×)
  GPT-2 medium        Megatron-LM      14.3   11.5 天 (1.8×)
  GPT-2 medium        FlashAttention   14.3   6.9 天 (3.0×)   ← 最快

  关键: FlashAttention 达到相同的 perplexity (精度无损, exact attention)
        但训练快 3 倍
```

### 4.1.3 Long-Range Arena (Table 3)

```
LRA benchmark (序列长度 1K-4K), 各方法的平均加速比:

  方法                    Avg Score    Speedup
  ───────────────────────────────────────────
  Transformer (标准)      59.3         1.0×
  FlashAttention          59.8         2.4×       ← 精度持平, 快 2.4×
  Block-sparse FA         59.6         2.8×       ← 快 2.8×
  Linformer               54.9         2.5×       ← 精度掉!
  Linear Attention        59.6         2.3×
  Performer               58.9         1.8×
  Local Attention         56.0         1.7×
  Reformer                57.6         1.3×

  结论: Block-sparse FlashAttention 是最快的, 且精度不损失
```

## 4.2 Better Models with Longer Sequences (更长序列 → 更好模型)

### 4.2.1 GPT-2 长上下文 (Table 4)

```
FlashAttention 让 GPT-2 用更长 context 还更快:

  实现              Context   PPL    训练时间 (相对加速)
  ──────────────────────────────────────────────────
  Megatron-LM       1K        18.2   4.7 天 (1.0×)
  FlashAttention    1K        18.2   2.7 天 (1.7×)
  FlashAttention    2K        17.6   3.0 天 (1.6×)   ← 序列×2, PPL 更好
  FlashAttention    4K        17.5   3.6 天 (1.3×)   ← 序列×4, 还比 Megatron 1K 快!

  惊人结论: FlashAttention 用 4× 长度 (4K), 仍比 Megatron 1K 快 30%, 且 PPL 好 0.7!
```

### 4.2.2 长文档分类 (Table 5)

```
MIMIC-III / ECtHR 数据集, 不同序列长度的 micro-F1:

  长度     512    1024   2048   4096   8192   16384
  ─────────────────────────────────────────────────
  MIMIC    52.8   50.7   51.7   54.6   56.4   57.1    ← 16K 比 512 好 4.3 分
  ECtHR    72.2   74.3   77.1   78.6   80.7   79.2    ← 8K 比 512 好 8.5 分

  → 用更长序列建模, 显著提升长文档分类质量
```

### 4.2.3 Path-X / Path-256 (Table 6) — 突破性能力

```
Path-X (seq=16K) 和 Path-256 (seq=64K) 是极长序列任务,
此前所有 Transformer 都只能达到随机水平 (50%):

  方法                    Path-X    Path-256
  ──────────────────────────────────────────
  所有现有方法            随机(50%) 随机(50%)
  FlashAttention          61.4%     —          ← 首次超过随机!
  Block-sparse FA         56.0%     63.1%      ← 首次在 64K 上工作!

  → FlashAttention 让 Transformer 第一次能解决 Path-X/Path-256
```

## 4.3 Benchmarking Attention (运行时与内存基准)

### 4.3.1 Runtime (Figure 3 左)

```
不同序列长度下 attention 的 forward+backward 时间 (A100):

  观察:
  ① FlashAttention 比标准 PyTorch attention 快最多 3 倍
  ② 短序列 (<512) 时, FlashAttention 比所有近似方法都快
  ③ 序列 512-1024 之间, 近似方法 (Linformer) 开始反超
  ④ Block-sparse FlashAttention 在所有长度都是最快的!

  交叉点 (crossover):
    seq ≈ 512-1024: Linformer 等 O(N) 方法开始比 dense FlashAttention 快
    但 block-sparse FA 仍然最快
```

### 4.3.2 Memory Footprint (Figure 3 右)

```
显存占用随序列长度变化:

  ① FlashAttention 显存 = O(N) 线性增长
  ② 标准 attention = O(N²) 二次方增长
  ③ FlashAttention 比标准 attention 节省最多 20 倍显存!
  ④ 即使 Linformer (也 O(N) 内存), FlashAttention 仍省 2 倍
  ⑤ 其他方法在 64K 之前就 OOM, FlashAttention 能跑到 128K

  → FlashAttention 既快又省内存, 还能跑超长序列
```

---

# 5. Limitations and Future Directions (局限与未来方向)

## 5.1 需要写 CUDA (Compiling to CUDA)

```
局限:
  - 每个 attention 变体都要手写新的 CUDA kernel
  - 比 PyTorch 低级得多, 工程量大
  - 难以跨 GPU 架构迁移

未来方向:
  → 希望有方法能: 用高级语言 (PyTorch) 写 attention, 编译成 IO-aware CUDA
  → 类似图像处理领域的 Halide
  (这个方向后来催生了 Triton, FlashAttention-2/3 等)
```

## 5.2 IO-Aware Deep Learning (IO 感知的深度学习)

```
FlashAttention 只是 Transformer 里最耗内存的计算
但深度网络每一层都触碰 GPU HBM

希望: IO-aware 的思想扩展到更多模块
  - 稀疏 MLP (sparse MLP layers)
  - 核方法 (kernel machine learning)
```

## 5.3 Multi-GPU IO-Aware Methods (多 GPU)

```
单 GPU 上 FlashAttention 已最优
但多 GPU (模型并行) 引入新的内存层级:
  - GPU SRAM
  - GPU HBM
  - 其他 GPU 的 HBM (通过 NVLink/PCIe)

未来: 考虑 GPU 间数据传输的 IO-aware 算法
```

---

# 6. 算法细节 (Appendix B 详解)

## 6.1 内存高效的前向传播推导 (B.1)

```
核心思路: 把 softmax 归一化常数单独算出来, 解耦 K 的列

  S_ij = q_i^T k_j
  归一化常数: L_i = Σ_j e^{q_i^T k_j}

  输出第 i 行: o_i = Σ_j P_ij v_j = Σ_j (e^{q_i^T k_j} / L_i) v_j

  → 一旦算出 L_i, 就可以逐块累加 o_i, 不需要存整个 P 矩阵
  → 额外内存: O(N) (存 L_i) + O(d) (累加 o_i)
```

## 6.2 内存高效的反向传播推导 (B.2)

给定输出梯度 dO ∈ ℝ^(N×d)，求 dQ, dK, dV：

```
① dV 容易: dV = P^T dO
   dv_j = Σ_i P_ij do_i = Σ_i (e^{q_i^T k_j}/L_i) do_i
   → 用 L_i, 逐块累加, 不需额外内存

② 定义 D_i = P_{i:}^T dP_{i:} = do_i^T o_i    ← 关键中间量!
   dS_ij = P_ij (dP_ij - D_i)

③ dQ, dK:
   dq_i = Σ_j dS_ij k_j = Σ_j P_ij (dP_ij - D_i) k_j
   dk_j = Σ_i dS_ij q_i = Σ_i P_ij (dP_ij - D_i) q_i

  → 全部可以用 L_i (前向存的) + 逐块累加完成
  → 额外内存: O(N) (存 D_i)
```

## 6.3 Algorithm 2: 完整前向传播 (含 mask + dropout)

```
相比 Algorithm 1, 增加了:
  - softmax scaling τ (通常 1/√d)
  - masking (key padding mask / causal mask)
  - dropout (保存随机数状态 R 供反向用)

  forward: S = τ·QK^T → S_masked = mask(S) → P = softmax(S_masked)
           → P_dropped = dropout(P, p) → O = P_dropped V

  保存: O, ℓ, m, R (随机数生成器状态)
  → 注意: 不保存 dropout mask, 而是保存 RNG 状态, 反向时重新生成
  → 节省 O(N²) 的 mask 存储
```

## 6.4 Algorithm 4: 完整反向传播 (重计算)

```
FlashAttention Backward Pass:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  输入: Q, K, V, O, dO ∈ ℝ^(N×d),  ℓ, m ∈ ℝ^N,  RNG 状态 R

  关键观察:
  ① 不存 dropout mask, 用 R 重新生成 → 省 O(N²) 内存
  ② 计算 D_i = rowsum(dO_i ⊙ O_i)   ← 用 o_i 而非整个 P 行

  流程 (同样是 tiling):
    for j = 1 to T_c:
      加载 K_j, V_j 到 SRAM, 初始化 ~dK_j = 0, ~dV_j = 0
      for i = 1 to T_r:
        加载 Q_i, O_i, dO_i, dQ_i, ℓ_i, m_i 到 SRAM

        重算: S_ij = τ·Q_i K_j^T          ← 重算! 不从 HBM 读
                P_ij = diag(ℓ_i)^{-1} exp(S_ij - m_i)
        重生成 dropout mask Z_ij
                P_dropped_ij = P_ij ⊙ Z_ij

        ~dV_j += P_dropped_ij^T dO_i
        dP_dropped_ij = dO_i V_j^T
        dP_ij = dP_dropped_ij ⊙ Z_ij
        D_i = rowsum(dO_i ⊙ O_i)
        dS_ij = P_ij ⊙ (dP_ij - D_i)

        写回 dQ_i += τ·dS_ij K_j
        ~dK_j += τ·dS_ij^T Q_i
      写回 dK_j, dV_j

  → 同样避免物化 N×N 矩阵
  → Theorem 5: 反向 IO 复杂度 = Θ(N²d²/M) (和前向一样)
```

---

# 7. 核心要点总结

## 7.1 FlashAttention 的三个核心思想

```
┌────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ① IO-aware:                                                    │
│     attention 慢不是因为 FLOP 多, 而是因为 HBM 访问多            │
│     → 减少 HBM 读写才是关键                                     │
│                                                                 │
│  ② Tiling + Online Softmax:                                     │
│     分块加载到 SRAM, 用在线 softmax 增量计算                     │
│     → 避免物化 N×N 矩阵到 HBM                                   │
│                                                                 │
│  ③ Recomputation:                                               │
│     反向传播不存中间矩阵, 从 Q,K,V 重算                          │
│     → 用 FLOP 换 HBM 访问, 反而更快                              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## 7.2 性能对比一览

```
指标                  标准 attention    FlashAttention    提升
─────────────────────────────────────────────────────────────────
FLOP                  O(N²d)           O(N²d)            相同 (重算略多)
HBM 访问              Θ(Nd + N²)       Θ(N²d²/M)         ~9× 更少
显存                  O(N²)            O(N)              ~20× 更省
Runtime (GPT-2)       baseline         7.6× faster       7.6×
精度                  exact            exact             完全相同!
最大序列长度          ~8K (OOM)        64K-128K          8-16×
```

## 7.3 为什么这个工作重要

```
1. 改变了 attention 加速的思路:
   从 "减少 FLOP" (近似方法) → "减少 IO" (exact 方法)

2. 让长序列 Transformer 成为可能:
   - GPT-4, LLaMA 等都用 FlashAttention
   - 支持 32K, 128K 甚至更长 context

3. 开启了 IO-aware 深度学习的方向:
   - FlashAttention-2, FlashAttention-3 相继出现
   - Triton 等工具让写 IO-aware kernel 更容易

4. 证明了: 有时候 "多算" 反而 "更快"
   (只要能减少内存访问)
```

---

# 8. 参考资料

- **FlashAttention 原论文**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **官方代码**: [github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
- **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- **FlashAttention-3**: Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- **Online Softmax (前置工作)**: Milakov & Gimelshein, "Online normalizer calculation for softmax", [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- **Rabe & Staats (相关工作)**: "Self-attention does not need O(n²) memory", [arXiv:2112.05682](https://arxiv.org/abs/2112.05682)
- **原版 Transformer**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- **Tri Dao 讲解视频**: [YouTube - FlashAttention](https://www.youtube.com/watch?v=qfb3LPYmZ5M)

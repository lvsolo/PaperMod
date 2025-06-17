title: "人工智能commonsense-QAT-PTQ-Slimming-Pruning"
author: "lvsolo"
date: "2025-05-18"
tags: ["interview",  "common sense"]





## ShuffleNet


## FaseVit


## RegNet

RegNet 论文全名为  **"Designing Network Design Spaces"** （CVPR 2020），作者来自 Facebook AI Research（FAIR）。这篇论文并不只是提出了 RegNet 模型，更重要的是系统性地总结和验证了 **神经网络设计的经验技巧（design heuristics）** ，并引入了 **"网络设计空间"（design space）**的概念。

以下是论文中提到并深入分析的一些 **关键经验技巧和设计原则** ，我会逐一介绍这些技巧的背景、作用和RegNet中的实际应用：

---

### 🔷 1. **Wider networks are better than deeper ones**

* **背景** ：以往网络发展常强调“越深越好”（如 ResNet-152 > ResNet-50），但训练很深的网络存在梯度消失、收敛慢、计算低效等问题。
* **经验技巧** ：
* 在计算预算固定的情况下， **加宽比加深更有效** ，也更容易训练。
* **RegNet 中做法** ：
* 用 `w_0`（初始宽度）、`w_a`（增长率）等参数控制宽度的变化；
* 网络宽度随着层数逐渐增加，而不是保持固定。

---

### 🔷 2. **Avoiding stage-wise bottlenecks**

* **背景** ：ResNet 等结构将网络划分为多个阶段，每阶段通道数翻倍。但这种突然的跳跃会导致信息流中断或损失。
* **经验技巧** ：
* 使用 **平滑的通道数增长（smooth channel scaling）** ，避免每一 stage 中出现突然的瓶颈或突变。
* **RegNet 中做法** ：
* 使用一种叫做 **"quantized linear function"** 的方法来平滑控制每一层的通道数，通道数呈现近似线性增长。

---

### 🔷 3. **Simple linear parameterization of widths is sufficient**

* **背景** ：复杂的设计空间（如随机搜索、NAS）带来高成本且难以解释。
* **经验技巧** ：
* 通道数的设计只需要简单的线性增长公式即可达到极好的效果。
* **RegNet 中做法** ：
* 通过一个线性函数（`w_i = w_0 + i * Δ`）生成所有 block 的宽度，再量化（quantization）使之可落到合适的值。

---

### 🔷 4. **Deeper networks require careful balancing of parameters**

* **背景** ：网络深度一旦加大，很容易不稳定或过拟合，必须合理分配参数预算。
* **经验技巧** ：
* 在设计网络时应保持“ **参数均衡分布** ”，而不是让某些 block 太大，某些太小。
* **RegNet 中做法** ：
* 使用 **`group width`** 和 **`bottleneck ratio`** 等参数控制特征维度结构，确保整体架构参数分布合理。

---

### 🔷 5. **Use of group convolutions for better efficiency**

* **背景** ：标准卷积计算量大，深层网络容易出现冗余。
* **经验技巧** ：
* 使用 **group convolution** 提高效率，同时保持表达能力。
* **RegNet 中做法** ：
* 使用 **group width（g）** 控制每个卷积 block 中的组数，通过调节 group 数和通道数来维持平衡。

---

### 🔷 6. **Avoid overly large or small bottleneck ratios**

* **背景** ：ResNet 的 bottleneck block 将输入维度先降再升，但过度压缩或扩张都可能带来损失。
* **经验技巧** ：
* bottleneck 比例应适度（如 1:1, 1:2, 1:4 常见），避免使用极端的比例。
* **RegNet 中做法** ：
* 通过引入超参数 **`b` (bottleneck ratio)** 控制维度压缩程度，发现 b=1~4 时最优。

---

### 🔷 7. **Regular design spaces outperform randomly searched ones**

* **背景** ：神经架构搜索（NAS）固然流行，但代价高昂且结果不稳定。
* **经验技巧** ：
* 一个结构良好、参数简单的手工设计空间，在实际中往往超过自动搜索的结构。
* **RegNet 中做法** ：
* 直接使用参数化设计空间 + 贪心搜索（simple grid search），就能找到非常高效的模型。

---

### 🔷 8. **Model design spaces should be explainable and scalable**

* **背景** ：模型可解释性越来越重要，黑盒设计不利于维护和迁移。
* **经验技巧** ：
* 构建可以被人类理解和调节的设计空间，例如通过 4~5 个参数就能定义整个网络结构。
* **RegNet 中做法** ：
* 设计空间只依赖于 4 个核心参数：
  * 初始宽度 `w_0`
  * 宽度增长率 `w_a`
  * 网络深度 `d`
  * 组卷积宽度 `g`

---

### 🔷 9. **Downsampling should be done at proper intervals**

* **背景** ：下采样过于频繁会导致信息流失，不足则会导致计算负担过重。
* **经验技巧** ：
* 均匀或策略性地下采样（如在 block 间插入 stride=2 的卷积）；
* **RegNet 中做法** ：
* 下采样仅发生在 stage 变化时，每个 stage 内保持特征图大小不变。

---

### 🔷 10. **Maintain balance between depth, width, and complexity**

* **背景** ：某些模型过于强调深度（如 ResNeXt），有些过于强调宽度（如 WideResNet），都可能出现参数浪费或性能瓶颈。
* **经验技巧** ：
* 构建模型应考虑 depth、width、group size、bottleneck ratio 等因素的平衡。
* **RegNet 中做法** ：
* 通过 grid search 在设计空间中优化这些参数组合，找到性能最优的结构。

---

### 总结：RegNet 提供的设计原则可以概括为：

| 类别     | 技巧                   | RegNet 应对方法                    |
| -------- | ---------------------- | ---------------------------------- |
| 架构设计 | 深度 vs 宽度           | 更倾向宽度增长，减少过深网络       |
| 通道分布 | 平滑变化               | 用线性增长控制每层通道数           |
| 参数效率 | group conv, bottleneck | 合理设置 group 和压缩比            |
| 可解释性 | 简洁空间               | 4-5 参数定义整个网络               |
| 搜索策略 | 网格搜索               | 避免黑盒 NAS，依赖简单 grid search |

---

如果你需要，我还可以提供 RegNet 设计空间参数的具体公式、RegNetY 与 RegNetX 的区别、以及 PyTorch 实现代码。是否需要进一步深入？

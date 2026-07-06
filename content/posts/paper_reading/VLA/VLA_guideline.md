# VLA（Vision-Language-Action）系统学习路径（工程导向）

---

## 🧭 总体目标

完成本路线后，你应具备：

* 理解 VLM（视觉-语言对齐）的核心机制
* 理解 Action / Policy 的序列建模方式
* 能读懂 RT-2 / OpenVLA 等论文
* 能进行基础 VLA 微调或实验

---

## 🧱 总体结构

```text
Vision（视觉表征）
    ↓
VLM（多模态对齐）
    ↓
Action（序列决策）
    ↓
VLA（统一建模）
```

---

# 📍 阶段 1：视觉表征（Vision Foundation）

## 🎯 目标

理解：

```text
image → embedding/token
```

---

## 📚 必看论文

### 1. Vision Transformer（ViT）

🔗 **精读笔记**: [ViT.md](ViT.md)

https://arxiv.org/abs/2010.11929

**重点：**

* patch embedding
* Transformer 如何处理图像

---

### 2. DINO（Self-Supervised Learning）

🔗 **精读笔记**: [DINO.md](DINO/DINO.md)  ·  **复现代码**: [DINO/repro/](DINO/repro/)

https://arxiv.org/abs/2104.14294

**重点：**

* 自监督学习
* teacher-student 结构

---

### 3. DINOv2（强烈推荐）

🔗 **精读笔记**: [DINOv2.md](DINOv2.md)  ·  **复现代码**: [DINOv2/repro/](DINOv2/repro/)

https://arxiv.org/abs/2304.07193

**重点：**

* 高质量视觉特征
* 泛化能力来源

---

### 4. SigLIP（OpenVLA使用）

🔗 **精读笔记**: [SigLIP.md](SigLIP.md)

https://arxiv.org/abs/2303.15343

**重点：**

* sigmoid loss vs softmax
* 更稳定的多模态对齐

---

## ⏱ 时间建议

**3–5 天（可快速过）**

---

# 📍 阶段 2：VLM（核心阶段）

## 🎯 目标

理解：

```text
image token ↔ text token 对齐
```

---

## 📚 核心三件套

---

### 1. CLIP

🔗 **精读笔记**: [CLIP.md](CLIP.md)

https://arxiv.org/abs/2103.00020

**重点：**

* 对比学习
* embedding 对齐空间

---

### 2. BLIP-2

🔗 **精读笔记**: [BLIP-2.md](BLIP-2.md)

https://arxiv.org/abs/2301.12597

**重点：**

* Q-Former（关键模块）
* frozen encoder + LLM

---

### 3. LLaVA

🔗 **精读笔记**: [LLaVA.md](LLaVA.md)

https://arxiv.org/abs/2304.08485

**重点：**

* image token 输入 LLM
* instruction tuning

---

## 📚 进阶论文（建议选看）

---

### 4. Flamingo

🔗 **精读笔记**: [Flamingo.md](Flamingo.md)

https://arxiv.org/abs/2204.14198

**重点：**

* cross-attention
* few-shot能力来源

---

### 5. PaLI

🔗 **精读笔记**: [PaLI.md](PaLI.md)

https://arxiv.org/abs/2209.06794

---

## 🧠 本阶段必须掌握

---

### 1. 两种范式

```text
CLIP → 对比学习
BLIP / LLaVA → 生成式建模
```

---

### 2. 核心模块

* projection layer
* cross-attention
* Q-Former

---

## ⏱ 时间建议

**7–10 天（重点）**

---

# 📍 阶段 3：序列决策（Action Modeling）

## 🎯 目标

理解：

```text
action = sequence modeling
```

---

## 📚 必看论文

---

### 1. Decision Transformer

🔗 **精读笔记**: [DecisionTransformer.md](DecisionTransformer.md)

https://arxiv.org/abs/2106.01345

**重点：**

* RL → Transformer
* trajectory = sequence

---

### 2. Diffusion Policy

🔗 **精读笔记**: [DiffusionPolicy.md](DiffusionPolicy.md)

https://arxiv.org/abs/2303.04137

**重点：**

* 连续控制生成
* diffusion在控制中的作用

---

### 3. Behavior Cloning（基础）

🔗 **精读笔记**: [BehaviorCloning.md](BehaviorCloning.md)

https://arxiv.org/abs/2005.07648

---

## 🧠 核心理解

```text
state + instruction → action sequence
```

---

## ⏱ 时间建议

**4–6 天**

---

# 📍 阶段 4：VLA（核心）

## 🎯 目标

理解：

```text
VLM + Action → VLA
```

---

## 📚 必看论文

---

### 1. RT-1

🔗 **精读笔记**: [RT-1.md](RT-1.md)

https://arxiv.org/abs/2212.06817

---

### 2. RT-2（关键）

🔗 **精读笔记**: [RT-2.md](RT-2.md)

https://arxiv.org/abs/2307.15818

**重点：**

* VLM → robot control
* reasoning + action

---

### 3. Open-X Embodiment（数据）

🔗 **精读笔记**: [OpenXEmbodiment.md](OpenXEmbodiment.md)

https://arxiv.org/abs/2310.08864

---

### 4. OpenVLA（重点）

🔗 **精读笔记**: [OpenVLA.md](OpenVLA.md)

https://arxiv.org/abs/2406.09246

**重点：**

* LLaMA-based VLA
* action tokenization
* LoRA 微调

---

## 🧠 核心理解

---

### 1. 本质

```text
视觉 + 语言 → 序列 → 动作
```

---

### 2. 关键技术

* action token（离散化）
* autoregressive policy
* 多模态输入

---

## ⏱ 时间建议

**5–7 天**

---

# 📍 阶段 5：工程实践（强烈建议）

## 🎯 目标

真正掌握（不是只看论文）

---

## 推荐路径

```text
1. 跑 LLaVA inference
2. 跑 OpenVLA inference（量化）
3. LoRA 微调小数据
```

---

# 🧩 总时间规划

| 阶段     | 时间     |
| ------ | ------ |
| Vision | 3–5 天  |
| VLM    | 7–10 天 |
| Action | 4–6 天  |
| VLA    | 5–7 天  |

👉 总计：约 3 周（高强度）

---

# 🧠 核心总结

```text
VLA = VLM（多模态对齐） + 序列决策（Transformer）
```

---

# 🚀 学习建议

1. 不要跳过 VLM（最关键）
2. 每篇论文关注：

   * 输入输出
   * token设计
   * loss函数
3. 尽早看代码（OpenVLA / LLaVA）

---

# 📌 精简最优路径（推荐）

```text
CLIP → BLIP-2 → LLaVA → Decision Transformer → RT-2 → OpenVLA
```

---

# tiny_dino —— 用最小代码复现 DINO, 自己填算法

> 目标：**通过亲手实现 DINO 论文里的算法逻辑来吃透它**。我把训练/数据/评估/可视化的脚手架都写好了，把**论文的核心算法留成 TODO 给你填**。填完即可在 RTX 2070 (8GB) 上跑通完整 DINO。

## 0. 环境与选型（已为你定好）

- 硬件：RTX 2070 / 8GB 显存（你的机器）。代码已按这个约束调好。
- 模型：**ViT-Tiny**（embed 192, 12 层, 3 头, ~5.4M 参数）。
- 数据集（`--dataset` 切换）：
  - **`stl10`（默认）**：96×96，10 万无标注图 + 5 千标注图。专门为自监督设计，自带标注 split 方便 k-NN 验证，attention 能看出物体轮廓。下载 ~2.5GB，训练约几小时。
  - **`cifar10`（冒烟用）**：32×32，170MB，下载几分钟、训练 <1 小时。验证防坍塌/EMA/loss 最快，但 attention 涌现偏弱。
- 装依赖：`pip install torch torchvision numpy matplotlib`

## 1. 文件总览

| 文件 | 内容 | 你要填 |
|---|---|---|
| `model.py` | ViT-Tiny 骨干 + DINO projection head + MultiCropWrapper | `prepare_tokens`、`get_last_selfattention` |
| `loss.py` | DINO 自蒸馏损失 | `DINOLoss.forward`、`DINOLoss.update_center` |
| `data.py` | 数据集 + multi-crop 增强 | `DataAugmentationDINO.__init__` 组装 global/local transform |
| `train.py` | 训练循环/优化器/调度/k-NN/可视化/main | `ema_update_teacher` |

## 2. TODO 清单（按建议顺序）

> 📌 **卡住了？** `_solution/` 目录里有我填好的**可运行参考实现**（model/loss/data/train 四个文件），
> 跑通过的。实在写不出来可以打开对照；但建议先自己想，对照官方代码（`../DINO.md` 里有逐行注释）。

每个 TODO 在代码里都有详细的步骤注释 + 论文章节引用。下面是总览。

### TODO ① `model.py :: VisionTransformer.prepare_tokens`  ——  [ViT.md §2.2-2.3]
把图像变成 token 序列：patch_embed → 前面拼 `[class]` token → 加位置编码。
**验证**：填完后 `bb(torch.randn(2,3,96,96))` 能返回 `(2, 192)` 的 cls 特征。

### TODO ② `model.py :: get_last_selfattention`  ——  [DINO.md §3.1.1]
返回最后一个 block 的 attention 权重 `(B, heads, 1+N, 1+N)`，供"涌现分割"可视化用。
**验证**：填完后能跑通 `train.py` 里的 `visualize_attention`。

### TODO ③ `data.py :: DataAugmentationDINO.__init__`  ——  [DINO.md §2.2]
用 `RandomResizedCrop` 组装出 2 个 global crop（scale 0.4–1.0，含模糊/solarization）+ N 个 local crop（scale 0.05–0.4）。helper 增强（flip+color jitter+normalize+GaussianBlur+Solarization）已写好。
**验证**：`aug(pil_img)` 返回长度 = 2+local_crops 的 Tensor 列表，global 比 local 大。

### TODO ④ `loss.py :: DINOLoss.forward` + `update_center`  ——  [DINO.md §2.4 / §2.4.1 / §2.5]（**最核心**）
- `forward`：student 除温度→chunk；teacher **减 center→softmax(温度)→detach→chunk(2)**；双重循环算交叉熵，`if v==iq: continue`；调 `update_center`。
- `update_center`：center 的 EMA。
**验证**：`python loss.py` 自带小测试，填完会打印一个有限正数 loss + 非零 center。

### TODO ⑤ `train.py :: ema_update_teacher`  ——  [DINO.md §2.6]
θ_t ← momentum·θ_t + (1−momentum)·θ_s。
**验证**：跑训练能开始，teacher 随 student 缓慢变化。

## 3. 怎么跑

> ⚠️ **8GB 显存(RTX 2070)注意**：multi-crop 会让有效 batch 翻 10 倍(2 global + 8 local)，
> `--batch-size` 不要超过 **64**，否则 OOM。建议再加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少碎片。

```bash
cd repro    # 即 content/posts/paper_reading/VLA/DINO/repro

# 1) 先单独验证 loss 填对了没
python loss.py

# 2) 冒烟测试(CIFAR, ~10 分钟看到不坍塌 + k-NN 高于随机)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py --dataset cifar10 --epochs 3 --batch-size 64 --eval-every 3

# 3) 正式跑 STL-10(几小时), 周期性 k-NN 评估 + 存 attention 热力图
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py --dataset stl10 --epochs 100 --batch-size 64 --eval-every 10
```
输出：`./out/` 下有 `attention.png`（[CLS]→patch 热力图）和 `ckpt.pt`。

### ✅ 已验证（参考实现冒烟记录）

用 `_solution/` 里的参考实现跑过 cifar10 2 epoch：**k-NN top-1 = 35.96%**（随机才 10%），
确认整条管线 + DINO 机制（防坍塌、自蒸馏、attention 提取）都正常工作。
（loss 在 warmup 阶段会贴在 ≈ln(65536)=11.09 附近，warmup 过后 lr 升上来才会明显下降——属正常现象。）

## 4. 你该看到什么（判断有没有复现成功）

| 现象 | 说明 |
|---|---|
| loss 平稳下降（不掉到 0、不 NaN） | 自蒸馏训练正常 ✅ |
| k-NN top-1 随训练稳步上升（STL 上能到 40–50%+） | 特征有含金量 ✅ |
| `attention.png` 里高响应区大致贴在物体上 | 涌现的"物体定位"（STL 上较温和，CIFAR 更弱）✅ |

## 5. 杀手级演示：复现"防坍塌"消融（DINO.md §2.5）

填完 ④ 之后，临时改一下做对比实验，亲眼看 collapse：
- **把 `update_center` 注释掉**（不中心化）→ 几个 epoch 内 loss 会诡异掉到 ~0、k-NN 跌到随机水平（≈10%）→ 这就是坍塌。
- **把 teacher 温度调大**（如 1.0，分布太平坦）→ 同样会坍塌。
- 恢复 centering + 小温度 → 稳定训练。

这一步能让你真正理解论文为什么强调"centering 和 sharpening 缺一不可"。

## 6. 老实说：什么复现不了

8GB + STL/CIFAR 规模下，论文里 **77% linear probe 这种绝对分数**复现不了——那需要完整 ImageNet-1k + 多卡 + 长训。但 DINO 的**机制**（防坍塌、EMA 自蒸馏、跨视角对齐、attention 涌现）你都能亲手验证，这才是理解论文的关键。

## 7. 对照阅读

边填边对照论文精读笔记（本工程位于 `VLA/DINO/repro/`）：
- 算法原理 & 官方逐行代码：`../DINO.md`（DINO 精读笔记，就在上一级目录）
- ViT 骨干细节：`../../ViT.md`（VLA 目录下）

---

填的过程中卡住，随时把报错或"这段不懂"发给我，我给你讲原理或提示（不会直接给答案，除非你要）。

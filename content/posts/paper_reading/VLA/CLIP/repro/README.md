# CLIP repro —— 最小复现 (CIFAR-10 zero-shot 主打, Flickr8k 诚实尝试)

> 目标同 DINO/DINOv2 repro：用最小数据 + 算力复现论文的**机制**（不追 SOTA），靠**手填关键算法**加深理解。
> CLIP 的机制 = **双塔对比学习（对称 InfoNCE）+ 跨模态检索/zero-shot**。
>
> **两条数据路径都实现了**，实测结论（见 §1）：CIFAR-10+类名 prompt 的 zero-shot 分类效果极强（90%），是主打；
> Flickr8k 真实 caption 的图文检索在小规模太难点（test R@1 卡随机），作为诚实尝试保留。

## 为什么用 Flickr8k 而不是 CIFAR+类名（关键）

CLIP 的 InfoNCE 假设：**每张图有独一无二的匹配文本，batch 内其余文本全是负样本**。

- **CIFAR+类名 prompt**：只有 100 句重复 prompt（500 张猫图共用 "a photo of a cat"）→ ① **假负样本**（同类的相同文本被当负样本推开）；② **文本塔学不到语言**（只学 100 个点）。本质是"用文本塔伪装的分类器"，不是真 CLIP。
- **Flickr8k**：8091 图 × 5 句**各不相同**的真实自然语言 caption → ① 无假负样本，对比损失按设计 work；② 文本塔真学语言；③ 用**标准图文检索 R@1/5/10**评估（CLIP/COCO 标准指标）。**这才是真 CLIP。**

## 0. 你要填的两个 TODO

| TODO | 文件 | 是什么 | 对照 |
|---|---|---|---|
| ① `clip_contrastive_loss` | loss.py | 对称 InfoNCE：batch 内 N 个图文对，对角线正样本、其余负样本，图→文 + 文→图交叉熵取平均 | CLIP.md §2.2 |
| ② `retrieve_topk` | train.py | 按余弦相似度取 top-k：`query @ candidate.T` 后 topk。**图文检索（Flickr8k）和 zero-shot（CIFAR）共用** | CLIP.md §3.1 |

> 两个 TODO 各 1-3 行。模型（小 ResNet 图像塔 + Transformer 文本塔 + 投影 + 可学习 logit_scale）、
> Flickr8k 加载/词表/检索评估、CIFAR zero-shot、训练循环**全预填**，调用你这两个函数。

## 1. 小数据能训出 CLIP 效果吗？（实测结论）

**分数据集，结论很不一样（实测）：**

- **CIFAR-10 + 类名 prompt → ✅ 效果非常显著**：zero-shot top-1 从随机 10% 爬到 **90.4%**（20 epoch，曲线 70→81→83→87→89.8→90.4%）。CIFAR-10 类别清晰(10类) + batch 256 对比信号足，CLIP zero-shot 分类学得飞快。**这是最可靠的"看出效果"路径。**（CIFAR-10 有 100 类稀释，假负样本问题很轻：batch 256 里平均每类 ~2.5 张，绝大多数负样本是真负样本。）
- **Flickr8k 真实 caption 检索 → ⚠️ 这个规模太难**：batch 内 retrieval acc 涨到 2%（随机 0.4%），但 **test 检索 R@1 卡在随机（0.1~0.5%）**——典型泛化鸿沟。原因：6k 图从头训 + 8GB 卡 batch 上限 ~64-224（CLIP 真靠大 batch 的海量负样本，真 CLIP 用 32k）+ 小模型。**Flickr8k 检索要看出效果，需要更大 batch / 更多数据 / 更大模型，超出"最小复现"。**
- 复现不了 CLIP 的"海量数据带来的强泛化"（要 WIT/LAION 4 亿对）。

> **所以本 repro 的主打路径是 CIFAR-10 zero-shot（90% 强效果）**；Flickr8k 作为"真实 caption 的诚实尝试"保留，结果 modest 是 CLIP 在小规模的真实表现，本身也是一课。

## 2. 怎么跑

```bash
cd content/posts/paper_reading/VLA/CLIP/repro

# CIFAR-10(主打, 零下载, ~25 min, RTX 2070) —— 复现出 90% zero-shot 的那条:
python3 train.py --dataset cifar10 --epochs 20 --batch-size 256 \
  --data /path/to/DINO/repro/data --out out_cifar10
# (CIFAR/STL 自动用原生分辨率 32/96px, 不用传 --img-size; --img-size 只对 flickr8k 生效)

# Flickr8k(真实 caption 的诚实尝试, 首次自动下 ~1GB):
python3 train.py --dataset flickr8k --epochs 30 --batch-size 64 --img-size 96 --out out_flickr8k
```
> CLIP 无 EMA/自蒸馏，就是标准监督 loop。CIFAR 32px→batch 256；Flickr 图大→batch 64、img 96。
> Flickr8k 首次跑会下 ~1GB（HF `jxie/flickr8k`）。CIFAR-10/STL 用 v1 已下的数据（`--data` 指过去那份数据目录）。

## 3. 实测结果

**CIFAR-10 zero-shot（主打，20 epoch）**：top-1 从随机 10% → **90.36%**

| epoch | 2 | 5 | 8 | 11 | 14 | 17 | 19 |
|---|---|---|---|---|---|---|---|
| zero-shot top-1 | 69.8 | 80.7 | 83.3 | 87.2 | 89.8 | 90.4 | **90.4** |

- loss 从 ln(256)=5.54 平稳降到 3.46；batch 内 i2t/t2i retrieval acc 涨到 80%+；scale 从 14.29 缓降到 ~13（CLIP 学到的温度）。
- **机制完整 work，效果显著。** 老实话：这本质是"用文本塔伪装的 10 类分类 + zero-shot 推理"，不是 CLIP 的语言泛化（那要 WIT 级数据），但对比对齐 + zero-shot 这条 CLIP 核心链路验证得很干净。

**Flickr8k 检索（诚实尝试）**：test R@1 卡随机（0.1~0.5%），batch 内涨但泛化不开。原因见 §1（batch/数据/模型规模不足）。要看 Flickr 检索效果需加大 batch（≥256 负样本）+ 更多数据。

## 4. 对照阅读

- 算法原理 + 官方代码：`../CLIP.md`（§2.2 InfoNCE、§3 zero-shot/检索）
- 同系列复现：`../../DINO/repro/`（自蒸馏）、`../../DINOv2/repro/`（+iBOT+KoLeo）
- 框架/约定：仓库根 `精读文档写作规范.md` §11

## 5. 想法卡住时

两个 TODO 的思路都在各自 docstring 里（loss.py / train.py）：
- ① 对称 InfoNCE：相似度矩阵 `image_emb @ text_emb.t()` 的**对角线是正样本**，`arange(N)` 当 label 做两次交叉熵（图→文 + 文→图）取平均。对照 `../CLIP.md` §2.2。
- ② retrieve_topk：`sim = query_emb @ candidate_emb.t()`，`sim.topk(k, dim=1).indices`。对照 `../CLIP.md` §3.1。
- 真卡住再看本目录 `_solution/`（参考实现，填好后放进去）。

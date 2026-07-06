# CLIP repro —— 最小复现 (Flickr8k 真实 caption + 图文检索)

> 目标同 DINO/DINOv2 repro：用最小数据 + 算力复现论文的**机制**（不追 SOTA），靠**手填关键算法**加深理解。
> CLIP 的机制 = **双塔对比学习（对称 InfoNCE）+ 图文跨模态检索**。
>
> **数据集选 Flickr8k（真实 caption）而非 CIFAR+类名**，原因见下。CIFAR/STL 作为零下载备选保留。

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

## 1. 小数据能训出 CLIP 效果吗？

**能。** Flickr8k（6000 训练图 × 5 caption = 3 万对，~1GB）：
- **图文检索 R@1** 会从随机 ~0.2% 爬到 **10~30%**（小模型，~30 epoch）；R@5/R@10 更高。batch 内 retrieval acc（正样本排第一）爬到 80%+。**机制完整 work，是标准 CLIP 评估。**
- 复现不了 CLIP 的"海量数据带来的强泛化"（要 WIT/LAION 4 亿对），但"对比对齐 + 跨模态检索"这条 CLIP 核心链路能完整验证。

## 2. 怎么跑

```bash
cd content/posts/paper_reading/VLA/CLIP/repro

# Flickr8k(推荐, 首次自动下 ~1GB, 之后缓存):
python3 train.py --dataset flickr8k --epochs 30 --batch-size 64 --img-size 96 --out out_flickr8k

# CIFAR-100(备选, 零下载, zero-shot 分类评估):
python3 train.py --dataset cifar100 --epochs 20 --batch-size 256 --out out_cifar100
```
> CLIP 无 EMA/自蒸馏，就是标准监督 loop。Flickr 图大→batch 64、img 96；CIFAR 32px→batch 256。
> Flickr8k 首次跑会下 ~1GB（HF `jxie/flickr8k`），下完缓存，后续秒载。

## 3. 预期（Flickr8k, ~30 epoch）

- **i2t / t2i R@1 高于随机 0.2%**、对比 loss 平稳下降 → 机制 work。
- log 里 **i2t_acc / t2i_acc**（batch 内正样本 retrieval）升到 80%+ → 对齐在学。
- **scale**（= exp(logit_scale)，初始 ≈14.3 = 1/0.07）随训练变化（CLIP 学到的温度）。
- 老实话：比不过真 CLIP（4 亿对 + ViT-L），但"对比对齐 + 跨模态检索"都能验证。

## 4. 对照阅读

- 算法原理 + 官方代码：`../CLIP.md`（§2.2 InfoNCE、§3 zero-shot/检索）
- 同系列复现：`../../DINO/repro/`（自蒸馏）、`../../DINOv2/repro/`（+iBOT+KoLeo）
- 框架/约定：仓库根 `精读文档写作规范.md` §11

## 5. 想法卡住时

两个 TODO 的思路都在各自 docstring 里（loss.py / train.py）：
- ① 对称 InfoNCE：相似度矩阵 `image_emb @ text_emb.t()` 的**对角线是正样本**，`arange(N)` 当 label 做两次交叉熵（图→文 + 文→图）取平均。对照 `../CLIP.md` §2.2。
- ② retrieve_topk：`sim = query_emb @ candidate_emb.t()`，`sim.topk(k, dim=1).indices`。对照 `../CLIP.md` §3.1。
- 真卡住再看本目录 `_solution/`（参考实现，填好后放进去）。

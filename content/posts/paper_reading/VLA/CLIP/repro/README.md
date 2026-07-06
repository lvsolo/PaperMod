# CLIP repro —— 最小复现 (CIFAR-100 + class-prompt + zero-shot)

> 目标同 DINO/DINOv2 repro：用最小数据 + 算力复现论文的**机制**（不追 SOTA），靠**手填关键算法**加深理解。
> CLIP 的机制 = **双塔对比学习（对称 InfoNCE）+ zero-shot 推理**。
> 数据用分类集（CIFAR-100/10、STL-10）+ 类名套 prompt 模板造图文对 —— 零额外下载，能完整跑通
> "对比训练 → zero-shot 分类"这条 CLIP 招牌链路。

## 0. 你要填的两个 TODO

| TODO | 文件 | 是什么 | 对照 |
|---|---|---|---|
| ① `clip_contrastive_loss` | loss.py | 对称 InfoNCE：batch 内 N 个图文对，对角线正样本、其余负样本，图→文 + 文→图两向交叉熵取平均 | CLIP.md §2.2 |
| ② `zero_shot_classify` | train.py | zero-shot 分类：测试图嵌入 vs 各类 prompt 文本嵌入比余弦相似度，argmax | CLIP.md §3.1 |

> 两个 TODO 各 3-5 行。模型（CNN 图像塔 + Transformer 文本塔 + 投影头 + logit_scale）、
> 数据（CIFAR + prompt 模板 + 词级 tokenizer）、训练循环、ensemble zero-shot 评估**全预填**，调用你这两个函数。

### 为什么是这两个 TODO？
- **对比损失**是 CLIP 的核心：一切魔法都在"把匹配图文拉近、不匹配推远"里。填完你就理解了 InfoNCE。
- **zero-shot**是 CLIP 的招牌能力：不训分类头，靠图文跨模态相似度直接分类。填完你就理解了 CLIP 为什么"开箱即用"。

## 1. 小数据能训出 CLIP 效果吗？（你最关心的）

**能，但要管理预期。** 本复现用 CIFAR-100（50k 图 × 100 类）+ class-prompt 当文本：
- **会有效果**：zero-shot top-1 会从随机 1% 爬到 30~50%（小 ResNet + 小 Transformer，~20 epoch）。i2t/t2i retrieval acc（batch 内正样本能否排第一）会从随机爬到 80%+。**机制完整 work**。
- **复现不了的部分**：CLIP 的"语言泛化"（任意自然语言指令）。因为文本端只见到 100 个类名级 prompt，不是真实 caption。要复现那部分得 WIT/LAION 级（4 亿图文对）数据 + 大算力，超出"最小复现"范围。
- 和 DINO repro 同理：复现**机制 + 涌现**，不追绝对分数。

## 2. 怎么跑

```bash
cd content/posts/paper_reading/VLA/CLIP/repro

# 复用 v1 已下数据目录(cifar100 会自动下 ~170MB; cifar10/stl10 已在):
DATA=/home/lvsolo/work/git/PaperMod/content/posts/paper_reading/VLA/DINO/repro/data

# 填完两个 TODO 后, 跑 CIFAR-100(~30-60 min, RTX 2070):
python3 train.py --dataset cifar100 --epochs 20 --batch-size 256 \
  --data $DATA --out out_cifar100

# 或用已下好的 CIFAR-10/STL-10(10 类, zero-shot 更粗):
python3 train.py --dataset cifar10  --epochs 20 --batch-size 256 --data $DATA --out out_cifar10
python3 train.py --dataset stl10    --epochs 20 --batch-size 128 --data $DATA --out out_stl10
```
> CLIP 没有 EMA/自蒸馏，就是个标准监督 loop；batch 大（256）对对比学习很重要（负样本多）。
> STL-10 图大（96px），batch 调小到 128；CIFAR（32px）batch 256 轻松。

## 3. 预期（CIFAR-100, ~20 epoch）

- **zero-shot top-1 高于随机 1%**、对比 loss 平稳下降 → 机制 work。
- log 里 **i2t_acc / t2i_acc**（batch 内正样本 retrieval）持续升到 80%+ → 对齐在学。
- **scale**（= exp(logit_scale)，相似度乘子，初始 ≈14.3 = 1/0.07，等价 softmax 温度 0.07）随训练变化（CLIP 学到的温度）。
- 老实话：比不过真 CLIP（400M 对 + ViT-L），但"对比对齐 + zero-shot"两条链路都能验证。

## 4. 对照阅读

- 算法原理 + 官方代码：`../CLIP.md`（§2.2 InfoNCE、§3 zero-shot）
- 同系列复现：`../../DINO/repro/`（自蒸馏）、`../../DINOv2/repro/`（+iBOT+KoLeo）
- 框架/约定：仓库根 `精读文档写作规范.md` §11（最小复现流程）

## 5. 想法卡住时

两个 TODO 的实现思路都写在各自函数的 docstring 里（loss.py / train.py），照着"实现思路"四步写即可。
- ① 对称 InfoNCE 的关键是：相似度矩阵 `image_emb @ text_emb.t()` 的**对角线是正样本**，用 `arange(N)` 当 label 做两次交叉熵（图→文 + 文→图）取平均。原理对照 `../CLIP.md` §2.2。
- ② zero-shot 的关键是：`image_emb @ class_text_emb.t()` 相似度最大的类就是预测，`.argmax(dim=1)`。原理对照 `../CLIP.md` §3.1。
- 真卡住再看本目录 `_solution/`（参考实现，填好后我会放进去）。

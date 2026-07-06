# DINOv2 repro —— 最小复现 (基于 DINO v1 + iBOT + KoLeo)

> 目标和 v1 一样：用最小数据 + 算力复现论文的**机制**（不追 SOTA），靠**手填关键算法**加深理解。
> DINOv2 = DINO v1 + **iBOT（patch 级自蒸馏）** + **KoLeo 正则** + 规模化。
> v1 你已掌握的部分（multi-crop、DINO 图像级损失、EMA、centering、prepare_tokens）**全部预填**，你只需填 v2 的**两个新东西**。

## 0. 你要填的两个 TODO

| TODO | 文件 | 是什么 | 对照 |
|---|---|---|---|
| ① `ibot_patch_loss` | loss.py | 在被 mask 的 patch 位置上 student↔teacher 交叉熵（patch 级自蒸馏） | DINOv2.md §2.2 |
| ② `koleo_loss` | loss.py | 让 cls 特征在超球面均匀分布（提升 retrieval/泛化） | DINOv2.md §2.3 |

> loss.py 的 `forward`（三损失组合 + 两套 center 更新 + teacher 目标准备）**已写好**，调用你这两个方法。
> 想法卡住时看 `_solution/loss.py`（参考实现，冒烟测试跑通过的）。

### iBOT 是什么（一句话）
DINO 只对齐 `[cls]` token（图像级）。iBOT 额外**随机 mask 掉 student 输入里一部分 patch**（换成 `mask_token`），让 student 从"残缺的图"去**预测被 mask 位置的 patch 表示**（teacher 看完整图作目标）。→ 学到**密集的 patch 级特征**（对分割/检测友好）。mask 的生成 + 注入都由 `data.random_patch_mask` 和 `model.prepare_tokens(mask=...)` 处理好了，你只写"在被 mask 位置算交叉熵"。

### KoLeo 是什么（一句话）
一个正则：让一个 batch 里的特征点在超球面上**尽量互相撑开**（每个点离最近邻越远越好）。物理意义 = 防止特征挤在某个子空间，提升 retrieval/泛化。

## 1. 怎么跑

```bash
cd content/posts/paper_reading/VLA/DINOv2/repro

# 1) 填完两个 TODO 后, 先冒烟(cifar, ~10 分钟): k-NN 应高于随机 10%、不坍塌
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 train.py --dataset cifar10 --epochs 3 --batch-size 64 --eval-every 3

# 2) STL-10(几小时), 周期性 k-NN + attention
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 train.py --dataset stl10 --epochs 30 --batch-size 32 --eval-every 5
```
> ⚠️ RTX 2070 8GB：batch 别超 32（DINOv2 比 v1 多 patch head + mask，显存更紧）。建议加 `expandable_segments`。

## 2. 预期（STL-10, ~20-30 epoch）

- **k-NN top-1 高于随机**、loss（total = cls + ibot + koleo）平稳下降 → 没坍塌，机制 work。
- log 里 **`ibot` 和 `koleo` 都是非零有限**（说明你填对了、且它们在起作用）。
- attention 图逐步聚焦（比 v1 更密，因 iBOT 学 patch 级）。
- 老实话：STL/CIFAR 规模复现不了"beat CLIP"那种分数（要 LVD-142M + 多卡），但**三个损失 + 防坍塌 + 涌现**都能验证。

## 3. 防坍塌消融（理解为什么需要每个 trick）

填完后试这几组对比，亲眼看机制：
- **关 centering**（注释掉 `_update_center`）→ 几个 epoch 坍塌、k-NN 跌到随机。
- **关 iBOT**（`ibot_patch_loss` 返回 0）→ k-NN 还行，但 patch 级/密集任务变差（看 attention 是否更糊）。
- **关 KoLeo**（`koleo_loss` 返回 0）→ retrieval 一般会降（k-NN 略降）。

## 4. 对照阅读

- 算法原理 + 官方代码：`../DINOv2.md`（§2.2 iBOT、§2.3 KoLeo）
- v1 的 DINO 损失 / multi-crop / EMA：`../../DINO/repro/` 和 `../../DINO.md`
- 框架/约定：仓库根 `精读文档写作规范.md` §11（最小复现流程）

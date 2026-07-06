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
> 想对答案/卡住时：`loss.py` 里的 `ibot_patch_loss` / `koleo_loss` 现已是**填好并冒烟通过**的参考实现。两个真实踩坑点（值得记住）：
> - **iBOT**：要遍历 `n_global`(=2) 个 global crop 再取平均，别只写一个 crop 的 body。
> - **KoLeo**：置对角线排除自己时，**别用 in-place `fill_diagonal_`**——它原地改了 `cdist` 的输出，破坏反向图（autograd 版本校验报错）。用非原地 `dist + torch.diag(inf)`。详见 `../../../../interview/Python_PyTorch_Notes.md` §3.4。

### iBOT 是什么（一句话）
DINO 只对齐 `[cls]` token（图像级）。iBOT 额外**随机 mask 掉 student 输入里一部分 patch**（换成 `mask_token`），让 student 从"残缺的图"去**预测被 mask 位置的 patch 表示**（teacher 看完整图作目标）。→ 学到**密集的 patch 级特征**（对分割/检测友好）。mask 的生成 + 注入都由 `data.random_patch_mask` 和 `model.prepare_tokens(mask=...)` 处理好了，你只写"在被 mask 位置算交叉熵"。

### KoLeo 是什么（一句话）
一个正则：让一个 batch 里的特征点在超球面上**尽量互相撑开**（每个点离最近邻越远越好）。物理意义 = 防止特征挤在某个子空间，提升 retrieval/泛化。

## 1. 怎么跑

```bash
cd content/posts/paper_reading/VLA/DINOv2/repro

# 复用 v1 已下好的数据(--data 指向 DINO/repro/data, 内含 stl10_binary / cifar-10-batches-py):
DATA=/home/lvsolo/work/git/PaperMod/content/posts/paper_reading/VLA/DINO/repro/data

# 1) 填完两个 TODO 后, 先冒烟(cifar, ~15 分钟): k-NN 应高于随机 10%、不坍塌
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 train.py --dataset cifar10 --epochs 2 --batch-size 32 --eval-every 2 \
    --data $DATA --out out_smoke

# 2) STL-10(几小时), 周期性 k-NN + attention
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 train.py --dataset stl10 --epochs 10 --batch-size 32 --eval-every 5 \
    --out-dim 8192 --lr 5e-4 --data $DATA --out out_stl10
```
> ⚠️ RTX 2070 8GB 显存关键参数（**实测，否则必 OOM**）：
> - **`--out-dim` 别用 65536**（那是 DINO 满配值）。patch 级 logits 形状 = `(B·n_global, N_patches, out_dim)` 极占显存，是 OOM 主因。默认已改 **16384**；STL（96px、144 patch）需再降到 **8192**，cifar（32px、64 patch）16384 够。
> - **batch ≤ 32**（DINOv2 比 v1 多 patch head + mask，显存更紧；batch 64 必 OOM）。
> - 务必加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减碎片。

## 2. 预期（STL-10, ~10-30 epoch）

- **k-NN top-1 高于随机**、loss（total = cls + ibot + koleo）平稳下降 → 没坍塌，机制 work。
- log 里 **`ibot` 和 `koleo` 都是非零有限**（说明你填对了、且它们在起作用）。
- `koleo` 可能降到 **负数**——这是好事：KoLeo = `-mean(log d_i)`，特征在超球面撑开后最近邻距离 `d_i>1` → `log>0` → loss<0。**变负 = 特征分布变均匀**（cifar 2 epoch 实测 koleo 从 1.57 降到 -0.21）。
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

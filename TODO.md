# TODO — 待办清单（长耗时 / 需空闲 GPU）

> **给未来的 Claude session 或自己**：这里只列【还没做、需要较长时间或需要空闲 GPU】的任务。
> 所有短任务已当场做完。新 session 起来后，先读本文件 + `MEMORY.md` 即可对齐全貌、知道还有哪些坑要填。
> 路径都以仓库根 `/home/lvsolo/work/git/PaperMod/` 为基准。

---

## 0. 项目快速定位

| 是什么 | 在哪 |
|---|---|
| DINO 复现工程（用户独立填的 6 个 TODO + 脚手架） | `content/posts/paper_reading/VLA/DINO/repro/` |
| DINO 论文精读笔记（含官方代码 + k-NN 解释） | `content/posts/paper_reading/VLA/DINO/DINO.md` |
| ViT 精读（含 pos_embed 要点 §2.4.1） | `content/posts/paper_reading/VLA/ViT.md` |
| 论文精读写作规范 | `精读文档写作规范.md` |
| 复现说明（TODO 清单 / 怎么跑 / 预期） | `content/posts/paper_reading/VLA/DINO/repro/README.md` |

**已确认能跑**：用户自己填的 6 个 TODO 全对；cifar10 冒烟 k-NN=36.6%，STL-10 20 epoch k-NN=66.1%。

---

## 1. ⏳ 长任务：STL-10 续训到 ~50 epoch（验证 attention 收敛）

**为什么**：20 epoch 时 attention 熵（最聚焦单头，中位）= 4.15，远未收敛（论文级 DINO ~1–2，均匀=4.97）。k-NN 还在涨（38.8→50.9→62.6→66.1）。继续训能让 attention overlay 明显聚焦到物体。

### 环境 / 前置（已就绪，确认一下）
- GPU：RTX 2070 8GB。Python 3.13 + torch 2.11 (CUDA)。依赖：`pip install torch torchvision numpy matplotlib aria2c`(aria2c 是系统命令，下数据用)。
- **STL-10 已下载并缓存**在 `content/posts/paper_reading/VLA/DINO/repro/data/stl10_binary/`（不用再下）。
- **续训起点 ckpt**：`content/posts/paper_reading/VLA/DINO/repro/out_stl10/ckpt.pt`（epoch 19，k-NN 66.1%）。
- 训练脚本已加好 `--resume`、每次 eval 记录 attention 熵、存 `ckpt_ep{N}.pt`。

### 一条命令启动（续训 30 epoch，到总 50；约 8 小时）
```bash
cd /home/lvsolo/work/git/PaperMod/content/posts/paper_reading/VLA/DINO/repro
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -u train.py \
  --dataset stl10 --epochs 50 --batch-size 32 --eval-every 5 --lr 5e-4 \
  --resume out_stl10/ckpt.pt --out out_stl10 --data ./data \
  > out_stl10/train_resume.log 2>&1 & disown
```
> `--epochs 50` 是【总目标 epoch】；脚本会从 ckpt 的 epoch 19 续到 49（共训 30 个）。
> `nohup ... & disown`：关掉终端/会话也不中断。

### 怎么看结果
```bash
cd content/posts/paper_reading/VLA/DINO/repro
grep "k-NN" out_stl10/train_resume.log          # 每个 eval 的 k-NN + attention 熵
ls out_stl10/attention_ep*.png                  # 各阶段 attention 快照(看聚焦演化)
python3 viz_overlay.py --ckpt out_stl10/ckpt.pt --out out_stl10/attention_overlay.png --n 10
                                                # 重新生成【叠到原图】的 overlay(看物体轮廓)
```

### 预期（50 epoch 时）
- k-NN top-1 → **75%+**
- attention 熵（最聚焦单头中位）→ **3 以下**（越接近 1~2 越锐利）
- `attention_overlay.png` 第三列（最聚焦单头）能看到高响应区贴到物体上。

### 判读标准
- **没 collapse**：k-NN 持续高于随机 10%、熵持续低于均匀 4.97 → 健康。
- **收敛信号**：连续两次 eval 的 k-NN 增量 < 1% 且熵基本不动 → 可停。
- 如果 OOM：把 `--batch-size` 降到 24 或 16（8GB 下 32 已实测峰值 4.7GB，一般不会 OOM）。

---

## 2. 可选小任务（不急，GPU 闲几分钟就能做）

- **attention rollout**（更锐利的可视化）：当前只取最后一层 attention；可改成把各层 attention 相乘（rollout），轮廓更清晰。改 `viz_overlay.py`，加一个 `--rollout` 选项即可。
- **cifar 长跑**（更快，~1.7h 跑 100 epoch）：`python3 train.py --dataset cifar10 --epochs 100 --batch-size 64 ...`，用来对比 STL、或快速验证改动。GPU 短暂空闲就能跑。

---

## 3. 已完成（不用再动，备查）

- 16 篇 VLA 精读文档：mermaid 图 + LaTeX 公式 + 官方/权威代码片段（含中文注释）。
- `精读文档写作规范.md`：含"概念首现必释""图用 mermaid""公式用 LaTeX""原理配代码"等规则。
- DINO 复现：用户独立填完 6 个 TODO，cifar 冒烟 + STL 20 epoch 验证通过。
- 权限：`.claude/settings.json` 开了 `Bash(*)` + `acceptEdits`（全项目免打扰）。

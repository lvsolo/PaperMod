"""
tiny_dino / train.py
====================
训练主脚本: 构建模型/优化器, 训练循环, k-NN 评估, attention 可视化。

【需要你填的只有 1 处】: ema_update_teacher (DINO 的 EMA 更新, DINO.md §2.6)。
其余训练逻辑、优化器、调度、k-NN、可视化都是脚手架, 已写好。

运行(填完所有 TODO 后):
    python train.py --dataset stl10  --epochs 100 --batch-size 64
    python train.py --dataset cifar10 --epochs 50  --batch-size 128   # 更快的冒烟测试
"""
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import VisionTransformer, DINOHead, MultiCropWrapper
from loss import DINOLoss
from data import make_augmentation, get_loaders, DATASET_CFG


# =====================================================================
# TODO (唯一一处)  —— EMA 更新 teacher  (DINO.md §2.6)
# =====================================================================
@torch.no_grad()
def ema_update_teacher(teacher, student, momentum):
    """
    teacher 参数 = student 参数的指数滑动平均:
        θ_t ← momentum * θ_t + (1 - momentum) * θ_s

    实现: 遍历 zip(teacher.parameters(), student.parameters()),
          对每一对做 in-place EMA (用 .data)。写完删掉 raise。
    """
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1 - momentum)


# =====================================================================
# 下面是脚手架, 一般不用改
# =====================================================================
def build_models(cfg, out_dim, device):
    def make():
        bb = VisionTransformer(img_size=cfg["img"], patch_size=cfg["patch"])
        head = DINOHead(in_dim=bb.embed_dim, out_dim=out_dim)
        return MultiCropWrapper(bb, head)
    student = make().to(device)
    teacher = make().to(device)
    teacher.load_state_dict(student.state_dict())        # teacher 初始 = student
    for p in teacher.parameters():
        p.requires_grad = False                          # teacher 不接收梯度
    return student, teacher


def cosine_scheduler(base_lr, warmup_lr, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr = warmup_lr + (base_lr - warmup_lr) * step / max(1, warmup_steps)
        else:
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = warmup_lr + 0.5 * (base_lr - warmup_lr) * (1 + math.cos(math.pi * prog))
        lrs.append(lr)
    return lrs


def momentum_schedule(base_momentum, total_epochs):
    """teacher EMA 动量从 base_momentum(0.996) 余弦升到 1.0。"""
    return [base_momentum + 0.5 * (1 - base_momentum) * (1 + math.cos(math.pi * e / total_epochs))
            for e in range(total_epochs)]


def extract_features(backbone, dataset, device, batch_size=256):
    """用 backbone(取 [class] token) 抽取整个数据集的特征 + 标签。"""
    backbone.eval()
    plain = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in plain:
            f = backbone(x.to(device))                 # (B, D)
            feats.append(f.cpu()); labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def knn_eval(backbone, ltrain, ltest, device, k=20):
    """k-NN 评估特征质量: 用 train 特征做最近邻投票, 报告 test top-1。
    这是自监督表征最直接的"含金量"检验。"""
    ftr, ytr = extract_features(backbone, ltrain, device)
    fte, yte = extract_features(backbone, ltest, device)
    ftr = F.normalize(ftr); fte = F.normalize(fte)
    sims = fte @ ftr.t()                               # (Ntest, Ntrain)
    topk = sims.topk(k, dim=1).indices
    pred = ytr[topk].mode(dim=1).values
    acc = (pred == yte).float().mean().item()
    return acc


@torch.no_grad()
def visualize_attention(backbone, dataset, device, out_dir, n=8):
    """把 [CLS]->patch 的 attention 画成热力图 (DINO.md §3.1.1)。
    依赖你实现的 backbone.get_last_selfattention。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    backbone.eval()
    cfg_img = backbone.patch_embed  # 取 patch 大小用
    patch_size = cfg_img.proj.kernel_size if isinstance(cfg_img.proj.kernel_size, int) \
        else cfg_img.proj.kernel_size[0]
    rows = min(n, len(dataset))
    fig, axes = plt.subplots(2, rows, figsize=(2*rows, 4))
    for i in range(rows):
        img, _ = dataset[i]
        x = img.unsqueeze(0).to(device)
        attn = backbone.get_last_selfattention(x)      # (1, heads, 1+N, 1+N)
        nh = attn.shape[1]
        feat = attn[0, :, 0, 1:]                        # (heads, N)  [CLS]->patch
        gs = int(math.sqrt(feat.shape[1]))
        # 取所有头平均, 还原成 2D 网格并上采样到原图
        amap = feat.mean(0).reshape(gs, gs).cpu().numpy()
        axes[0, i].imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1)); axes[0, i].axis("off")
        axes[1, i].imshow(amap, cmap="viridis"); axes[1, i].axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, "attention.png")
    plt.savefig(path, dpi=80); plt.close()
    print(f"attention 热力图已存到 {path}")


def train_one_epoch(student, teacher, dino_loss, loader, optimizer, lrs, epoch,
                    momentum, device, log_every=50):
    student.train()
    total, n = 0.0, 0
    t0 = time.time()
    for it, (crops, _labels) in enumerate(loader):
        crops = [c.to(device, non_blocking=True) for c in crops]   # list[(B,C,H,W)]
        # 路由: teacher 只看 2 个 global crop; student 看全部
        teacher_out = teacher(crops[:2])            # (2B, out_dim)
        student_out = student(crops)                # (ncrops*B, out_dim)

        loss = dino_loss(student_out, teacher_out.detach(), epoch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
        lr = lrs.pop(0)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()
        ema_update_teacher(teacher, student, momentum)

        total += loss.item(); n += 1
        if it % log_every == 0:
            print(f"[ep {epoch} it {it}/{len(loader)}] loss={loss.item():.4f} "
                  f"lr={lr:.2e} mom={momentum:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["stl10", "cifar10"], default="stl10")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--out-dim", type=int, default=65536)
    p.add_argument("--local-crops", type=int, default=8)
    p.add_argument("--momentum", type=float, default=0.996)
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out", type=str, default="./tiny_dino/out")
    p.add_argument("--eval-every", type=int, default=10)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    aug, cfg = make_augmentation(args.dataset, local_crops_number=args.local_crops)
    ncrops = 2 + args.local_crops
    loader, ltrain, ltest = get_loaders(args.dataset, args.data, aug,
                                        args.batch_size, collate=collate_crops)
    print(f"dataset={args.dataset}  img={cfg['img']} patch={cfg['patch']}  "
          f"steps/epoch={len(loader)}  device={device}")

    student, teacher = build_models(cfg, args.out_dim, device)
    dino_loss = DINOLoss(args.out_dim, ncrops, warmup_teacher_temp=0.04, teacher_temp=0.07,
                         warmup_teacher_temp_epochs=10, nepochs=args.epochs).to(device)

    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    lrs = cosine_scheduler(args.lr, 1e-6, 10, args.epochs, len(loader))
    moms = momentum_schedule(args.momentum, args.epochs)

    for epoch in range(args.epochs):
        loss = train_one_epoch(student, teacher, dino_loss, loader, optimizer, lrs,
                               epoch, moms[epoch], device)
        print(f"== epoch {epoch} done, mean loss={loss:.4f} ==")
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            acc = knn_eval(student.backbone, ltrain, ltest, device)
            print(f"   [k-NN top-1] {acc*100:.2f}%")
            visualize_attention(student.backbone, ltest, device, args.out)
            torch.save({"student": student.state_dict(), "epoch": epoch},
                       os.path.join(args.out, "ckpt.pt"))


# ---- 自定义 collate: 把一批 "crop 列表" 整理成 "按 crop 维堆叠的 tensor 列表" ----
def collate_crops(batch):
    """batch: list of (crop_list, label)。返回 (list[Tensor], labels)。"""
    crop_lists = [item[0] for item in batch]
    ncrops = len(crop_lists[0])
    out = [torch.stack([c[j] for c in crop_lists], dim=0) for j in range(ncrops)]
    labels = torch.tensor([item[1] for item in batch])
    return out, labels


import os
if __name__ == "__main__":
    main()

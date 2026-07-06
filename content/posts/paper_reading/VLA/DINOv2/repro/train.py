"""
DINOv2 repro / train.py
=======================
训练循环 / 优化器 / k-NN / 可视化, 全部【预填】。
DINOv2 相对 v1 在训练步里多了一步: 给 student 的 2 个 global crop 生成 iBOT 随机 mask。
你要填的 TODO 在 loss.py (ibot_patch_loss / koleo_loss)。
"""
import argparse, math, time, os
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import VisionTransformer, DINOv2Wrapper
from loss import DINOv2Loss
from data import make_augmentation, get_loaders, DATASET_CFG, random_patch_mask


# -------------------- EMA: teacher = student 的滑动平均 (沿用 v1, 预填) --------------------
@torch.no_grad()
def ema_update_teacher(teacher, student, momentum):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def build_models(cfg, out_dim, device):
    def make():
        bb = VisionTransformer(cfg["img"], cfg["patch"])
        return DINOv2Wrapper(bb, out_dim)
    student, teacher = make().to(device), make().to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    return student, teacher


def cosine_scheduler(base_lr, warmup_lr, warmup_ep, total_ep, steps):
    ws, ts = warmup_ep * steps, total_ep * steps
    out = []
    for s in range(ts):
        if s < ws:
            out.append(warmup_lr + (base_lr - warmup_lr) * s / max(1, ws))
        else:
            p = (s - ws) / max(1, ts - ws)
            out.append(warmup_lr + 0.5 * (base_lr - warmup_lr) * (1 + math.cos(math.pi * p)))
    return out


def momentum_schedule(base, total_ep):
    return [base + 0.5 * (1 - base) * (1 + math.cos(math.pi * e / total_ep)) for e in range(total_ep)]


# -------------------- 评估 / 可视化 (预填) --------------------
@torch.no_grad()
def extract_features(backbone, dataset, device, batch_size=256):
    backbone.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    feats, labels = [], []
    for x, y in loader:
        cls_tok, _ = backbone.forward_features(x.to(device))     # 用 cls token 作特征
        feats.append(cls_tok.cpu()); labels.append(y)
    backbone.train()
    return torch.cat(feats), torch.cat(labels)


def knn_eval(backbone, ltrain, ltest, device, k=20):
    ftr, ytr = extract_features(backbone, ltrain, device)
    fte, yte = extract_features(backbone, ltest, device)
    ftr, fte = F.normalize(ftr), F.normalize(fte)
    pred = ytr[fte @ ftr.t()].mode(dim=1).values
    return (pred == yte).float().mean().item()


@torch.no_grad()
def visualize_attention(backbone, dataset, device, out_dir, n=8, epoch=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True); backbone.eval()
    rows = min(n, len(dataset))
    fig, axes = plt.subplots(2, rows, figsize=(2 * rows, 4))
    for i in range(rows):
        img, _ = dataset[i]; x = img.unsqueeze(0).to(device)
        attn = backbone.get_last_selfattention(x)[0, :, 0, 1:]
        gs = int(round(math.sqrt(attn.shape[1])))
        amap = (attn / attn.sum(-1, keepdim=True)).mean(0).reshape(gs, gs).cpu().numpy()
        axes[0, i].imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1)); axes[0, i].axis("off")
        axes[1, i].imshow(amap, cmap="viridis"); axes[1, i].axis("off")
    plt.tight_layout()
    p = os.path.join(out_dir, "attention.png"); plt.savefig(p, dpi=80)
    if epoch is not None:
        plt.savefig(os.path.join(out_dir, f"attention_ep{epoch:03d}.png"), dpi=80)
    plt.close(); backbone.train()
    print(f"attention -> {p}")


# -------------------- 一个训练 step (预填; iBOT 的 mask 注入在这里) --------------------
def train_one_epoch(student, teacher, dino_loss, loader, optimizer, lrs, epoch,
                    momentum, device, n_global, n_patch_global, mask_ratio, log_every=50):
    student.train(); total, n = 0.0, 0; t0 = time.time()
    for it, (crops, _) in enumerate(loader):
        crops = [c.to(device, non_blocking=True) for c in crops]
        B = crops[0].size(0)
        # 🆕 给 student 的 global crop 生成 iBOT 随机 mask (local crop 不 mask)
        masks = [random_patch_mask(B, n_patch_global, mask_ratio).to(device) for _ in range(n_global)]
        masks = masks + [None] * (len(crops) - n_global)
        # student: 全部 crop(global 带 mask) ; teacher: 只 global、不带 mask
        s_out = student(crops, masks, n_patch_crops=n_global)
        t_out = teacher(crops[:n_global], [None] * n_global, n_patch_crops=n_global)
        loss, log = dino_loss(s_out, t_out, masks, epoch)

        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
        lr = lrs.pop(0)
        for pg in optimizer.param_groups: pg["lr"] = lr
        optimizer.step()
        ema_update_teacher(teacher, student, momentum)
        total += log["total"]; n += 1
        if it % log_every == 0:
            print(f"[ep{epoch} it{it}/{len(loader)}] total={log['total']:.3f} "
                  f"cls={log['cls']:.3f} ibot={log['ibot']:.3f} koleo={log['koleo']:.4f} "
                  f"lr={lr:.2e} ({time.time()-t0:.0f}s)", flush=True)
    return total / max(1, n)


def collate_crops(batch):
    cl = [b[0] for b in batch]; nc = len(cl[0])
    return [torch.stack([c[j] for c in cl], 0) for j in range(nc)], torch.tensor([b[1] for b in batch])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["stl10", "cifar10"], default="stl10")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--out-dim", type=int, default=65536)
    p.add_argument("--local-crops", type=int, default=8)
    p.add_argument("--momentum", type=float, default=0.996)
    p.add_argument("--mask-ratio", type=float, default=0.3, help="iBOT 每 global crop mask 的 patch 比例")
    p.add_argument("--koleo-weight", type=float, default=1.0)
    p.add_argument("--data", default="./data"); p.add_argument("--out", default="./out_dinov2")
    p.add_argument("--eval-every", type=int, default=5)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    aug, cfg = make_augmentation(args.dataset, local_crops_number=args.local_crops)
    ncrops = 2 + args.local_crops
    n_global = 2
    n_patch_global = (cfg["img"] // cfg["patch"]) ** 2          # 一个 global crop 的 patch 数
    loader, ltrain, ltest = get_loaders(args.dataset, args.data, aug, args.batch_size, collate=collate_crops)
    print(f"dataset={args.dataset} img={cfg['img']} patch={cfg['patch']} "
          f"n_patch_global={n_patch_global} steps/epoch={len(loader)} device={device}")

    student, teacher = build_models(cfg, args.out_dim, device)
    dino_loss = DINOv2Loss(args.out_dim, ncrops, n_global, nepochs=args.epochs,
                           koleo_weight=args.koleo_weight).to(device)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    lrs = cosine_scheduler(args.lr, 1e-6, 10, args.epochs, len(loader))
    moms = momentum_schedule(args.momentum, args.epochs)

    for ep in range(args.epochs):
        mean_loss = train_one_epoch(student, teacher, dino_loss, loader, opt, lrs, ep,
                                    moms[ep], device, n_global, n_patch_global, args.mask_ratio)
        print(f"== epoch {ep} done, mean loss={mean_loss:.4f} ==")
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            acc = knn_eval(student.backbone, ltrain, ltest, device)
            print(f"   [ep{ep}] k-NN top-1 = {acc*100:.2f}%")
            visualize_attention(student.backbone, ltest, device, args.out, epoch=ep)
            torch.save({"student": student.state_dict(), "epoch": ep}, os.path.join(args.out, "ckpt.pt"))


if __name__ == "__main__":
    main()

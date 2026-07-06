"""
CLIP repro / train.py  (Flickr8k 主, CIFAR 备)
=============================================
训练循环【通用】(图文对 → 双塔 → 对比损失), 与 DINO 不同: CLIP 无自蒸馏/EMA, 就是标准监督 loop。
评估按数据集分支:
  - Flickr8k: 图文检索 R@1/5/10 (image→text, text→image) —— CLIP/COCO 标准指标
  - CIFAR/STL: zero-shot 分类 (类名 prompt 比相似度)

两个 TODO:
  ① loss.py 的 clip_contrastive_loss (对称 InfoNCE)
  ② 本文件 retrieve_topk (按余弦相似度取 top-k, 检索/zero-shot 共用)
"""
import argparse, math, time, os, io
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import CLIPWrapper
from loss import CLIPLoss
from data import (get_flickr8k, get_cifar, Tokenizer, PROMPT_TEMPLATES, build_transforms)


def cosine_scheduler(base_lr, warmup_ep, total_ep, steps):
    ws, ts = warmup_ep * steps, total_ep * steps
    out = []
    for s in range(ts):
        out.append(base_lr * s / max(1, ws) if s < ws
                   else base_lr * 0.5 * (1 + math.cos(math.pi * (s - ws) / max(1, ts - ws))))
    return out


# ====================================================================
# TODO ② —— 通用 top-k 检索 (图文检索 + zero-shot 共用)
# ====================================================================
def retrieve_topk(query_emb, candidate_emb, k):
    """
    输入(都已 L2 归一化):
        query_emb     : (M, D)
        candidate_emb : (N, D)
        k             : int
    应返回: (M, k) long —— 每个 query 最相似的 k 个 candidate 的下标(按相似度从高到低)。

    实现思路:
        sim = query_emb @ candidate_emb.t()      # (M, N) 余弦相似度矩阵
        return sim.topk(k, dim=1).indices        # (M, k)
    提示: 相似度最大的 k 个 = 最匹配的 k 个候选。检索 R@K 和 zero-shot 都基于它。
    """
    # ===== TODO: 把下面这行替换成你的实现 =====
    raise NotImplementedError("填我: top-k 检索 (见上方注释)")


# -------------------- Flickr8k 检索评估 (预填, 调你的 retrieve_topk) --------------------
@torch.no_grad()
def retrieval_eval_flickr(model, test_ds, eval_tf, device, ks=(1, 5, 10), batch_size=64):
    """1000 test 图 × 5 caption。image→text / text→image 的 R@K。
    约定: 第 i 张图的 5 句 caption 展平后位于 [5i .. 5i+4]。"""
    model.eval()
    from PIL import Image
    # 编码全部测试图
    img_embs = []
    for i in range(0, len(test_ds), batch_size):
        imgs = torch.stack([eval_tf(Image.open(io.BytesIO(test_ds[j]["image"]["bytes"])).convert("RGB"))
                            for j in range(i, min(i + batch_size, len(test_ds)))], 0).to(device)
        img_embs.append(model.encode_image(imgs))
    img_embs = torch.cat(img_embs, 0)                                   # (1000, D)
    # 编码全部测试 caption (5 × 1000 = 5000), 需要一个 tokenizer —— 复用 model 自带的(挂一份)
    tok = retrieval_eval_flickr._tokenizer
    cap_list = [test_ds[i][f"caption_{k}"] for i in range(len(test_ds)) for k in range(5)]  # (5000,)
    cap_embs = []
    for i in range(0, len(cap_list), batch_size * 5):
        ids = tok.encode_batch(cap_list[i:i + batch_size * 5]).to(device)
        cap_embs.append(model.encode_text(ids))
    cap_embs = torch.cat(cap_embs, 0)                                  # (5000, D)

    n_img = img_embs.shape[0]
    gt = torch.arange(n_img, device=device).repeat_interleave(5)        # (5000,) caption j → 图 j//5
    kmax = max(ks)
    # image → text: 每张图取 top-k caption, 看是否含它的 5 句之一
    i2t_topk = retrieve_topk(img_embs, cap_embs, kmax)                 # (1000, kmax)  ★ 你的 TODO
    i2t = {}
    for kk in ks:
        # 第 i 张图的正样本 caption 下标 = 5i..5i+4
        hit = torch.zeros(n_img, dtype=torch.bool, device=device)
        for r in range(n_img):
            lo = 5 * r
            hit[r] = torch.isin(i2t_topk[r, :kk], torch.arange(lo, lo + 5, device=device)).any()
        i2t[kk] = hit.float().mean().item()
    # text → image: 每句 caption 取 top-k 图, 看是否含它的图 (j//5)
    t2i_topk = retrieve_topk(cap_embs, img_embs, kmax)                 # (5000, kmax)
    t2i = {}
    for kk in ks:
        hit = torch.zeros(len(cap_list), dtype=torch.bool, device=device)
        target = torch.arange(len(cap_list), device=device) // 5
        for kkidx in range(kk):
            hit |= (t2i_topk[:, kkidx] == target)
        t2i[kk] = hit.float().mean().item()
    model.train()
    return i2t, t2i


# -------------------- CIFAR zero-shot 评估 (预填, 调你的 retrieve_topk) --------------------
@torch.no_grad()
def encode_class_texts(model, tokenizer, classes, templates, device):
    model.eval()
    caps = [t.format(c) for c in classes for t in templates]            # (n_classes * n_templates,)
    ids = tokenizer.encode_batch(caps).to(device)
    emb = model.encode_text(ids).view(len(classes), len(templates), -1)
    return F.normalize(emb.mean(dim=1), dim=-1)                        # 模板 ensemble → (n_classes, D)


@torch.no_grad()
def zero_shot_eval_cifar(model, tokenizer, test_set, classes, templates, device, batch_size=256):
    model.eval()
    class_emb = encode_class_texts(model, tokenizer, classes, templates, device)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    correct, total = 0, 0
    for x, y in loader:
        img_emb = model.encode_image(x.to(device))
        pred = retrieve_topk(img_emb, class_emb, 1)[:, 0]              # (M,) top-1 类  ★ 你的 TODO
        correct += (pred.cpu() == y).sum().item(); total += y.numel()
    model.train()
    return correct / max(1, total)


# -------------------- 训练一个 epoch (通用, 预填) --------------------
def train_one_epoch(model, loss_fn, loader, optimizer, lrs, device, log_every=50):
    model.train(); total, n = 0.0, 0; t0 = time.time()
    for it, batch in enumerate(loader):
        imgs, token_ids = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        img_emb, txt_emb = model(imgs, token_ids)
        logit_scale = model.logit_scale.clamp(0, math.log(100.0))      # τ ∈ [1,100]
        loss, log = loss_fn(img_emb, txt_emb, logit_scale)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        lr = lrs.pop(0)
        for pg in optimizer.param_groups: pg["lr"] = lr
        total += log["loss"]; n += 1
        if it % log_every == 0:
            print(f"[it{it}/{len(loader)}] loss={log['loss']:.3f} i2t={log['i2t_acc']*100:.1f}% "
                  f"t2i={log['t2i_acc']*100:.1f}% scale={logit_scale.exp().item():.2f} "
                  f"lr={lr:.2e} ({time.time()-t0:.0f}s)", flush=True)
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["flickr8k", "cifar100", "cifar10", "stl10"], default="flickr8k")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./out_clip")
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    if args.dataset == "flickr8k":
        loader, test_ds, eval_tf, tokenizer = get_flickr8k(
            args.img_size, args.batch_size, tokenizer_max_len=32, num_workers=args.workers, root=args.data)
        retrieval_eval_flickr._tokenizer = tokenizer                   # 挂给评估函数用
        eval_fn = lambda m: retrieval_eval_flickr(m, test_ds, eval_tf, device)
        print(f"dataset=flickr8k train={len(loader.dataset)} test=1000 vocab={tokenizer.vocab_size}")
    else:
        loader, test_set, classes, tokenizer = get_cifar(
            args.dataset, args.data, args.batch_size, args.workers, img_size=args.img_size)
        ncls = len(classes)
        eval_fn = lambda m: (zero_shot_eval_cifar(m, tokenizer, test_set, classes, PROMPT_TEMPLATES, device), ncls)
        print(f"dataset={args.dataset} classes={ncls} vocab={tokenizer.vocab_size}")

    model = CLIPWrapper(tokenizer.vocab_size, embed_dim=args.embed_dim).to(device)
    loss_fn = CLIPLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lrs = cosine_scheduler(args.lr, 1, args.epochs, len(loader))
    print(f"steps/epoch={len(loader)} device={device}")

    for ep in range(args.epochs):
        mean_loss = train_one_epoch(model, loss_fn, loader, opt, lrs, device)
        print(f"== epoch {ep} done, mean loss={mean_loss:.4f} ==")
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            res = eval_fn(model)
            if args.dataset == "flickr8k":
                i2t, t2i = res
                print(f"   [ep{ep}] i2t R@1/5/10 = {i2t[1]*100:.1f}/{i2t[5]*100:.1f}/{i2t[10]*100:.1f}%  "
                      f"t2i R@1/5/10 = {t2i[1]*100:.1f}/{t2i[5]*100:.1f}/{t2i[10]*100:.1f}%  (随机 R@1≈0.2%)")
            else:
                acc, ncls = res
                print(f"   [ep{ep}] zero-shot top-1 = {acc*100:.2f}%  (随机 = {100/ncls:.1f}%)")
            torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(args.out, "ckpt.pt"))


if __name__ == "__main__":
    main()

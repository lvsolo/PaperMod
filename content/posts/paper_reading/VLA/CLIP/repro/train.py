"""
CLIP repro / train.py
=====================
训练循环 + zero-shot 评估, 绝大部分预填。CLIP 的训练很简单(不像 DINO 要自蒸馏/EMA):
  就是一个标准的监督式 loop: batch 里 N 个图文对 → 双塔前向 → 对比损失 → backward。
唯一要你填的是 zero-shot 推理函数 zero_shot_classify(train.py 里)。

zero-shot 是 CLIP 的招牌: 不训任何分类头, 直接拿【类名 prompt 的文本嵌入】和【测试图嵌入】
比余弦相似度, 取最大的类作为预测。和 DINO 的 k-NN 评估是两套思路:
  - DINO: 特征空间 k-NN(需要 labeled train 集做参考)。
  - CLIP: 图文跨模态相似度(不需要 labeled train, 只需类名文本)。
"""
import argparse, math, time, os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import CLIPWrapper
from loss import CLIPLoss
from data import get_loaders, Tokenizer, PROMPT_TEMPLATES, DATASET_CFG


def cosine_scheduler(base_lr, warmup_ep, total_ep, steps):
    ws, ts = warmup_ep * steps, total_ep * steps
    out = []
    for s in range(ts):
        if s < ws:
            out.append(base_lr * s / max(1, ws))
        else:
            p = (s - ws) / max(1, ts - ws)
            out.append(base_lr * 0.5 * (1 + math.cos(math.pi * p)))
    return out


# -------------------- zero-shot 推理 (你要填 TODO) --------------------
@torch.no_grad()
def encode_class_texts(model, tokenizer, classes, templates, device):
    """把每个类的所有 prompt 模板编码后 ensemble(取平均再归一化) → (n_classes, embed_dim)。"""
    model.eval()
    ids = tokenizer.class_text_ids(classes, templates).to(device)      # (C, T, L)
    C, T, L = ids.shape
    emb = model.encode_text(ids.view(C * T, L)).view(C, T, -1)        # (C, T, D)
    emb = F.normalize(emb.mean(dim=1), dim=-1)                         # 模板 ensemble → (C, D)
    return emb


def zero_shot_classify(image_emb, class_text_emb, logit_scale):
    """
    TODO ② —— zero-shot 分类(CLIP 招牌)。预填了 encode_class_texts, 你填这个。
    输入:
        image_emb      : (M, D)  M 张测试图的归一化嵌入
        class_text_emb : (C, D)  C 个类的 ensemble 文本嵌入(已归一化)
        logit_scale    : 标量, τ = exp(logit_scale)
    应返回: (M,) long —— 每张图预测的类下标。

    实现思路:
        sim = image_emb @ class_text_emb.t() * exp(logit_scale)   # (M, C) 每张图对每个类的相似度
        return sim.argmax(dim=1)
    提示: 相似度最大 = 余弦最近 = 该图最像那个类的 prompt。
    """
    # ===== TODO: 把下面这行替换成你的实现 =====
    raise NotImplementedError("填我: zero-shot 分类 (见上方注释)")


# -------------------- zero-shot 评估 (预填, 调用你的 TODO) --------------------
@torch.no_grad()
def zero_shot_eval(model, tokenizer, loss_fn, test_set, classes, templates, device, batch_size=256):
    model.eval()
    class_emb = encode_class_texts(model, tokenizer, classes, templates, device)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    correct, total = 0, 0
    for x, y in loader:
        img_emb = model.encode_image(x.to(device))
        pred = zero_shot_classify(img_emb, class_emb, model.logit_scale)   # ★ 调你的 TODO
        correct += (pred.cpu() == y).sum().item(); total += y.numel()
    model.train()
    return correct / max(1, total)


# -------------------- 训练一个 epoch (预填) --------------------
def train_one_epoch(model, loss_fn, loader, optimizer, lrs, device, log_every=50):
    model.train(); total, n = 0.0, 0; t0 = time.time()
    for it, (imgs, token_ids, _labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True); token_ids = token_ids.to(device, non_blocking=True)
        img_emb, txt_emb = model(imgs, token_ids)
        # ★ CLIP 训练时把 logit_scale clamp 到 [0, ln(100)≈4.6], 即 τ ∈ [1, 100], 防温度爆炸
        logit_scale = model.logit_scale.clamp(0, math.log(100.0))
        loss, log = loss_fn(img_emb, txt_emb, logit_scale)

        optimizer.zero_grad(); loss.backward()
        optimizer.step()
        lr = lrs.pop(0)
        for pg in optimizer.param_groups: pg["lr"] = lr
        total += log["loss"]; n += 1
        if it % log_every == 0:
            print(f"[it{it}/{len(loader)}] loss={log['loss']:.3f} "
                  f"i2t_acc={log['i2t_acc']*100:.1f}% t2i_acc={log['t2i_acc']*100:.1f}% "
                  f"scale={logit_scale.exp().item():.2f} lr={lr:.2e} ({time.time()-t0:.0f}s)", flush=True)
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar100", "cifar10", "stl10"], default="cifar100")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./out_clip")
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    loader, test_set, cfg = get_loaders(args.dataset, args.data, args.batch_size, args.workers)
    classes = cfg["classes"]
    tokenizer = Tokenizer(classes, PROMPT_TEMPLATES, max_len=16)
    # 把 tokenizer 注入 collate(随机套模板)
    loader.collate_fn = tokenizer.make_collate(classes, PROMPT_TEMPLATES)
    print(f"dataset={args.dataset} classes={len(classes)} vocab={tokenizer.vocab_size} "
          f"steps/epoch={len(loader)} device={device}")

    model = CLIPWrapper(tokenizer.vocab_size, embed_dim=args.embed_dim).to(device)
    loss_fn = CLIPLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lrs = cosine_scheduler(args.lr, 1, args.epochs, len(loader))

    for ep in range(args.epochs):
        mean_loss = train_one_epoch(model, loss_fn, loader, opt, lrs, device)
        print(f"== epoch {ep} done, mean loss={mean_loss:.4f} ==")
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            acc = zero_shot_eval(model, tokenizer, loss_fn, test_set, classes, PROMPT_TEMPLATES, device)
            print(f"   [ep{ep}] zero-shot top-1 = {acc*100:.2f}%  (随机基线 = {100/len(classes):.1f}%)")
            torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(args.out, "ckpt.pt"))


if __name__ == "__main__":
    main()

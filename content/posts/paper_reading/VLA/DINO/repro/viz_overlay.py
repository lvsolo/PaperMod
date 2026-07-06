"""
把 [CLS]->patch 的 attention 作为热力图层, 叠加到原图上(类似深度估计图叠原图)。
同时画出【平均头】和【最聚焦的单头】, 因为单头通常比平均聚焦得多。
用法: python viz_overlay.py [--ckpt out_stl10/ckpt.pt] [--out out_stl10/attention_overlay.png]
"""
import argparse, math, os, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from model import VisionTransformer, DINOHead, MultiCropWrapper


def load_backbone(ckpt_path, device):
    mcw = MultiCropWrapper(VisionTransformer(96, 8), DINOHead(192, 65536)).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcw.load_state_dict(ck["student"])
    bb = mcw.backbone
    bb.eval()
    return bb


def attn_maps(bb, img_tensor, device):
    """返回 (mean_attn, best_attn) 都是 [0,1] 归一化的 gs×gs numpy 数组。"""
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        attn = bb.get_last_selfattention(x)[0, :, 0, 1:]   # (heads, N)
        attn = attn / attn.sum(-1, keepdim=True)            # 每头归一化
        gs = int(round(math.sqrt(attn.shape[1])))
        # 平均头
        a_mean = attn.mean(0); a_mean = a_mean / a_mean.sum()
        # 最聚焦的单头(熵最小)
        ent = -(attn * (attn + 1e-12).log()).sum(-1)
        a_best = attn[ent.argmin()]; a_best = a_best / a_best.sum()
        return a_mean.reshape(gs, gs).cpu().numpy(), a_best.reshape(gs, gs).cpu().numpy(), ent.min().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_stl10/ckpt.pt")
    ap.add_argument("--out", default="out_stl10/attention_overlay.png")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb = load_backbone(args.ckpt, device)
    plain = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # 用原始(未 normalize)的图来显示, 用 normalized 的喂模型
    raw = datasets.STL10("./data", split="test", download=False,
                         transform=transforms.ToTensor())
    norm = datasets.STL10("./data", split="test", download=False, transform=plain)

    random.seed(args.seed)
    idxs = random.sample(range(len(raw)), args.n)

    cols = 3  # 原图 | 平均头叠加 | 最聚焦单头叠加
    fig, axes = plt.subplots(args.n, cols, figsize=(2.6 * cols, 2.6 * args.n))
    for r, i in enumerate(idxs):
        img_show = raw[i][0].permute(1, 2, 0).numpy()          # HWC, [0,1], 用于显示
        img_norm = norm[i][0]                                   # normalized, 喂模型
        a_mean, a_best, e_best = attn_maps(bb, img_norm, device)
        # 上采样 attention 到原图大小 (96x96)
        up = lambda a: F.interpolate(torch.tensor(a)[None, None], size=(96, 96),
                                     mode="bilinear", align_corners=False)[0, 0].numpy()
        am_up, ab_up = up(a_mean), up(a_best)

        axes[r, 0].imshow(img_show); axes[r, 0].set_title("original", fontsize=9)
        for c, (a, title) in enumerate([(am_up, "mean-head overlay"),
                                        (ab_up, f"best single head\nentropy={e_best:.2f}")], start=1):
            axes[r, c].imshow(img_show)
            axes[r, c].imshow(a, cmap="jet", alpha=0.5)         # 半透明热力图叠上去
            axes[r, c].set_title(title, fontsize=9)
        for c in range(cols):
            axes[r, c].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=110); plt.close()
    print(f"已存: {args.out}  ({args.n} 张图 x 3 列)")


if __name__ == "__main__":
    main()

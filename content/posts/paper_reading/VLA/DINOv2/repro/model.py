"""
DINOv2 repro / model.py
=======================
ViT-Tiny 骨干, 在 DINO(v1) 基础上加了两样【为 iBOT 服务】的东西:
  ① mask_token: iBOT 会随机 mask 掉 student 部分 patch, 被 mask 的位置用 mask_token 替代输入。
  ② 同时输出 cls token (图像级) 和 patch tokens (patch 级), 分别配 DINOHead / PatchHead。

v1 已掌握的部分 (PatchEmbed/Attention/Block/prepare_tokens 的 cls+pos) 沿用、已预填;
新增的 mask 应用、patch 输出也由脚手架写好。你要填的都在 loss.py (iBOT/KoLeo)。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=192):
        super().__init__()
        ih, iw = pair(img_size)
        self.num_patches = (ih // pair(patch_size)[0]) * (iw // pair(patch_size)[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)     # (B, N, D)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return (out, attn) if return_attention else out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=True)
            x = x + y
            return x + self.mlp(self.norm2(x)), attn
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """ViT-Tiny。比 v1 多了 mask_token, forward_features 同时返回 cls 和 patch。"""
    def __init__(self, img_size=96, patch_size=8, embed_dim=192, depth=12, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))   # 🆕 iBOT 用
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _interp_pos(self, n_cur):
        """multi-crop 分辨率不同时, 把 pos_embed 的 patch 部分双三次插值到 n_cur 个 patch。(同 v1)"""
        pe = self.pos_embed
        if pe.shape[1] - 1 == n_cur:
            return pe
        D = pe.shape[-1]
        gs_old = int(round((pe.shape[1] - 1) ** 0.5))
        gs_new = int(round(n_cur ** 0.5))
        patch_pe = pe[:, 1:].reshape(1, gs_old, gs_old, D).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=(gs_new, gs_new), mode="bicubic", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, D)
        return torch.cat([pe[:, :1], patch_pe], dim=1)

    def prepare_tokens(self, x, mask=None):
        """
        x: (B,C,H,W); mask: (B,N) bool, True=该 patch 被 iBOT mask 掉。
        返回 token 序列 (B, 1+N, D): cls + patch(被mask处替换为 mask_token) + pos_embed。
        """
        x = self.patch_embed(x)                                  # (B, N, D)
        B, N, D = x.shape
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(B, N, D), x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)                           # (B, 1+N, D)
        x = x + self._interp_pos(N)
        return x

    def forward_features(self, x, mask=None):
        """返回 (cls_token, patch_tokens), 都过最终 LayerNorm。"""
        x = self.prepare_tokens(x, mask)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]                                 # cls, patch

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks[:-1]:
            x = blk(x)
        _, attn = self.blocks[-1](x, return_attention=True)
        return attn


class DINOHead(nn.Module):
    """3 层 MLP + L2 归一化 + 最后一层映射到 out_dim。返回 (bottleneck_feat, logits)。"""
    def __init__(self, in_dim=192, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, bottleneck_dim))
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x):
        f = nn.functional.normalize(self.mlp(x), dim=-1)          # bottleneck 特征 (KoLeo 用)
        return f, self.last_layer(f)


class DINOv2Wrapper(nn.Module):
    """student/teacher 的前向: 吃 crops(+可选 masks), 吐 cls_feat / cls_logits / patch_logits。"""
    def __init__(self, backbone, out_dim=65536):
        super().__init__()
        self.backbone = backbone
        self.dino_head = DINOHead(backbone.embed_dim, out_dim)    # 图像级 (cls)
        self.patch_head = DINOHead(backbone.embed_dim, out_dim)   # patch 级 (iBOT)

    def forward(self, crops, masks=None, n_patch_crops=None):
        """
        crops: list[(B,C,H,W)]。masks: 与 crops 等长的 list, 每个 (B,N) bool 或 None。
        n_patch_crops: 前 n_patch_crops 个 crop 收集 patch logits(给 iBOT); None=全部收集。
            - student: masks=[m,m,None,...], n_patch_crops=n_global  (global 带 mask, 收 patch)
            - teacher: masks=[None,None],          n_patch_crops=None (不带 mask, 也收 patch 作目标)
        返回 dict: cls_feat / cls_logits(所有 crop) + patch_logits(前 n_patch_crops 个, 形状 (sumB, N, out))。
        """
        if masks is None:
            masks = [None] * len(crops)
        if n_patch_crops is None:
            n_patch_crops = len(crops)
        cls_feats, cls_logits, patch_logits = [], [], []
        for i, (c, m) in enumerate(zip(crops, masks)):
            cls_tok, patch_tok = self.backbone.forward_features(c, m)
            f, lg = self.dino_head(cls_tok)
            cls_feats.append(f); cls_logits.append(lg)
            if i < n_patch_crops:                                # 前 n_patch_crops 个收 patch logits
                _, plg = self.patch_head(patch_tok)
                patch_logits.append(plg)
        return {"cls_feat": torch.cat(cls_feats),
                "cls_logits": torch.cat(cls_logits),
                "patch_logits": torch.cat(patch_logits) if patch_logits else None}

"""
tiny_dino / model.py
====================
ViT-Tiny 视觉骨干 + DINO projection head + MultiCropWrapper。

这部分的代码【不是 DINO 论文的核心创新】，而是标准 ViT，所以我（脚手架）已经帮你写好。
你只需要填两个带 `TODO` 的小函数（prepare_tokens / get_last_selfattention）——
它们对应论文里两个关键概念：[class] token 拼接、以及"取最后一层 attention 做分割可视化"。

对照阅读：VLA/ViT.md（ViT 架构）与 VLA/DINO.md §3.1.1（attention 可视化）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbed(nn.Module):
    """图像 -> patch token。关键: 一个 kernel=stride=patch 的 Conv2d 就等价于
    '切 patch + 展平 + 线性投影'（见 ViT.md §2.2）。"""
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=192):
        super().__init__()
        ih, iw = pair(img_size)
        ph, pw = pair(patch_size)
        self.num_patches = (ih // ph) * (iw // pw)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)                       # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)       # (B, N, D)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # 各 (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        if return_attention:
            return out, attn                              # 返回注意力权重供可视化
        return out


class Block(nn.Module):
    """Pre-LN + MHSA(残差) + MLP(残差)，标准 ViT block。"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=True)
            x = x + y
            x = x + self.mlp(self.norm2(x))
            return x, attn
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT-Tiny: embed_dim=192, depth=12, heads=3。"""
    def __init__(self, img_size=96, patch_size=8, in_chans=3, embed_dim=192,
                 depth=12, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    # ------------------------------------------------------------------
    # TODO 1/2  (DINO.md §2.3, ViT.md §2.3)
    # ------------------------------------------------------------------
    def prepare_tokens(self, x):
        """
        把图像变成送入 Transformer 的 token 序列。

        输入:
            x: (B, C, H, W) 原始图像
        应返回:
            tokens: (B, 1 + N, D)  —— 在 patch token 前面拼上 [class] token,
                                     再加上位置编码 pos_embed。

        步骤:
            1) x = self.patch_embed(x)                 # (B, N, D)
            2) 把 self.cls_token 扩到 batch 维并拼到 x 前面  -> (B, 1+N, D)
            3) x = x + self.pos_embed                  # 加位置编码
        写完后, 把下面的 `raise` 替换成 `return x`。
        """
        x = self.patch_embed(x)                              # (B, N, D)
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        x = torch.cat((cls, x), dim=1)                       # (B, 1+N, D)
        # multi-crop 时 global/local 分辨率不同 → patch 数不同, 需要把 pos_embed
        # 双三次插值到当前 patch 网格 (timm/官方 ViT 的标准做法)
        pe = self.pos_embed
        if pe.shape[1] != x.shape[1]:
            cls_pe, patch_pe = pe[:, :1], pe[:, 1:]
            gs_old = int(round(patch_pe.shape[1] ** 0.5))
            gs_new = int(round(N ** 0.5))
            patch_pe = patch_pe.reshape(1, gs_old, gs_old, D).permute(0, 3, 1, 2)
            patch_pe = F.interpolate(patch_pe, size=(gs_new, gs_new),
                                     mode="bicubic", align_corners=False)
            patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, D)
            pe = torch.cat([cls_pe, patch_pe], dim=1)
        x = x + pe                                           # 加位置编码
        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                  # 返回 [class] token 作为图表征

    # ------------------------------------------------------------------
    # TODO 2/2  (DINO.md §3.1.1)
    # ------------------------------------------------------------------
    def get_last_selfattention(self, x):
        """
        返回【最后一个 block】的 self-attention 权重, 用于"涌现分割"可视化。

        输入: x: (B, C, H, W)
        应返回: attn: (B, heads, 1+N, 1+N)

        步骤:
            1) x = self.prepare_tokens(x)
            2) 前 (depth-1) 层正常前向:  for blk in self.blocks[:-1]: x = blk(x)
            3) 最后一层带 return_attention=True 调用, 返回它的 attn:
                 _, attn = self.blocks[-1](x, return_attention=True)
                 return attn
        """
        x = self.prepare_tokens(x)
        for blk in self.blocks[:-1]:                        # 前 (depth-1) 层正常前向
            x = blk(x)
        _, attn = self.blocks[-1](x, return_attention=True)  # 最后一层返回 attention
        return attn                                         # (B, heads, 1+N, 1+N)


class DINOHead(nn.Module):
    """3 层 MLP projection head + L2 归一化。训练时才用; 下游评估丢弃它。
    (对应 DINO.md §2.3 网络结构图里的 Projection Head)"""
    def __init__(self, in_dim=192, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1)          # L2 归一化
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """把一组不同分辨率的 crop 分组过 backbone(同分辨率拼一批), 再统一过 head。
    (对应 DINO.md §2.1 整体架构图里 student/teacher 的前向)"""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        """x: list of Tensor, 每个 (B, C, H, W), 可能有多种分辨率(=global/local)。"""
        if not isinstance(x, list):
            x = [x]
        # 按分辨率分组, 同分辨率拼成一个大 batch 一次前向(提速)
        idx_crops = torch.cumsum(
            torch.unique_consecutive(torch.tensor([t.shape[-1] for t in x]), return_counts=True)[1], 0)
        start, output = 0, torch.empty(0).to(x[0].device)
        for end in idx_crops:
            _out = self.backbone(torch.cat(x[start:end]))
            output = torch.cat((output, _out))
            start = end
        # 统一过 projection head
        return self.head(output)

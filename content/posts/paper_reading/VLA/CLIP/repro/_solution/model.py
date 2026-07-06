"""
CLIP repro / model.py
=====================
CLIP = 双塔(dual-encoder): 图像塔 + 文本塔, 各自把输入映到【共享 embedding 空间】,
用对比损失拉近"匹配的图文对"、推开"不匹配的"。本文件把两个编码器 + 投影头 + 温度参数全预填好。
你要填的对比损失在 loss.py, zero-shot 推理在 train.py。

架构选型(为什么这样选, 见 CLIP.md §2.3):
  - 图像塔: 小 ResNet(CNN)。CLIP 官方有 RN50 / ViT 两种; CIFAR/STL 这种低分辨率小数据,
    CNN 比从头训的 ViT 收敛快得多、zero-shot 效果更明显(本复现优先"看出效果")。
  - 文本塔: 小 Transformer。CLIP 文本端就是 Transformer(取 <EOS> 位置作句向量)。
  - 投影头: 两塔各一个 Linear → 共享 embed_dim, 输出 L2 归一化(CLIP 比的是余弦相似度)。
  - logit_scale: 可学习温度 τ = exp(logit_scale), CLIP 初始化为 log(1/0.07)≈2.66。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------- 图像塔: 小 ResNet (预填) --------------------
class BasicBlock(nn.Module):
    """ResNet 基本块: 两层 3x3 conv + 跳连。"""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ImageEncoder(nn.Module):
    """小 ResNet: stem + 4 个 stage(每 stage 下采样一次) → 自适应池化 → 512 维。"""
    def __init__(self, feat_dim=512):
        super().__init__()
        self.stem = nn.Sequential(                       # 3 → 64, stride 1
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.stages = nn.Sequential(
            BasicBlock(64, 64), BasicBlock(64, 64),       # stage1
            BasicBlock(64, 128, stride=2), BasicBlock(128, 128),   # stage2 (下采样)
            BasicBlock(128, 256, stride=2), BasicBlock(256, 256),  # stage3
            BasicBlock(256, feat_dim, stride=2), BasicBlock(feat_dim, feat_dim),  # stage4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)               # 任意分辨率 → 1x1, 再 flatten → (B, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.pool(x).flatten(1)                    # (B, feat_dim)


# -------------------- 文本塔: 小 Transformer (预填) --------------------
class TextEncoder(nn.Module):
    """Transformer encoder: token embedding + 位置编码 → N 层 → 取 <EOS> 位置作句向量 →投影前特征。"""
    def __init__(self, vocab_size, embed_dim=256, depth=4, num_heads=4, max_len=16, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.ln = nn.LayerNorm(embed_dim)
        self.feat_dim = embed_dim

    def forward(self, token_ids):
        """
        token_ids: (B, L) long, 已 pad 到 max_len。返回 (B, feat_dim) —— 每句取 <EOS> 位置特征。
        """
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)            # (B, L, D)
        # key_padding_mask: True = 该位是 pad, attention 忽略它(PyTorch 约定)
        pad_mask = (token_ids == self.pad_id)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.ln(x)
        # 取每句最后一个非 pad 位置(= <EOS>) 的特征作句向量(同 CLIP)
        eos_idx = (~pad_mask).sum(dim=1) - 1                       # (B,) 每句真实长度-1
        return x[torch.arange(B), eos_idx]                         # (B, D)


# -------------------- CLIP 双塔封装 (预填) --------------------
class CLIPWrapper(nn.Module):
    """图像塔 + 文本塔 + 各自 projection → 共享 embed_dim, 输出 L2 归一化。"""
    def __init__(self, vocab_size, embed_dim=256, img_feat=512, text_feat=256, max_len=16):
        super().__init__()
        self.image_encoder = ImageEncoder(img_feat)
        self.text_encoder = TextEncoder(vocab_size, text_feat, max_len=max_len)   # max_len 必须和 tokenizer 一致, 否则 pos_emb 越界
        # 投影头: 把两塔特征映到同一 embed_dim 空间
        self.image_proj = nn.Linear(img_feat, embed_dim, bias=False)
        self.text_proj = nn.Linear(text_feat, embed_dim, bias=False)
        # ★ 可学习温度: logits *= exp(logit_scale)。CLIP 初始化 logit_scale=log(1/0.07)≈2.66,
        #   即初始乘子 exp(logit_scale)=1/0.07≈14.29(把相似度放大约 14 倍, 等价 softmax 温度 τ=0.07)。
        #   train.py 训练时把 logit_scale clamp 到 [0, ln100], 即乘子 ∈ [1, 100], 防温度爆炸。
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
        self.embed_dim = embed_dim

    def encode_image(self, x):
        h = self.image_proj(self.image_encoder(x))                 # (B, embed_dim)
        return F.normalize(h, dim=-1)                              # L2 归一化 → 余弦相似度

    def encode_text(self, token_ids):
        h = self.text_proj(self.text_encoder(token_ids))           # (B, embed_dim)
        return F.normalize(h, dim=-1)

    def forward(self, images, token_ids):
        return self.encode_image(images), self.encode_text(token_ids)

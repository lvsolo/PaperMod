"""
CLIP repro / loss.py
====================
CLIP 的核心就是【对称 InfoNCE 对比损失】。这是你要填的唯一 TODO。

直觉(对照 CLIP.md §2.2):
  一个 batch 里 N 个图文对 (image_i, text_i) 是【正样本】(匹配), 其余 N-1 个文本对 image_i
  来说都是【负样本】。目标: 让 image_i 和它匹配的 text_i 相似度最高。
  - 图→文方向: 对每个 image, 在 N 个 text 里挑出它的匹配(对角线)。
  - 文→图方向: 对称地, 对每个 text, 在 N 个 image 里挑匹配。
  两个方向的交叉熵取平均 → 对称损失。

相似度 = (归一化的 image_emb) @ (归一化的 text_emb).T * τ,   τ = exp(logit_scale)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # ====================================================================
    # TODO —— 对称 InfoNCE 对比损失  (CLIP.md §2.2)
    # ====================================================================
    def clip_contrastive_loss(self, image_emb, text_emb, logit_scale):
        """
        输入(都已是 L2 归一化, 维度 = embed_dim):
            image_emb  : (N, D)   一个 batch 的 N 张图嵌入
            text_emb   : (N, D)   对应的 N 句文本嵌入(image_i 配 text_i, 即对角线是正样本)
            logit_scale: 标量 Parameter, 温度 τ = exp(logit_scale)
        应返回: 标量 loss

        实现思路(对称 InfoNCE):
            1) logits = image_emb @ text_emb.t() * exp(logit_scale)      # (N, N) 图→文相似度矩阵
            2) labels = arange(N)                                        # 第 i 张图的正样本是第 i 句文本
            3) loss_i2t = cross_entropy(logits, labels)                  # 图→文
               loss_t2i = cross_entropy(logits.t(), labels)             # 文→图(转置)
            4) return (loss_i2t + loss_t2i) / 2
        提示:
            - 相似度矩阵对角线 = 正样本对, 其余 = 负样本对(同一 batch 内其它图文)。
            - 用 F.cross_entropy(logits, labels); labels 是 long。
            - exp(logit_scale) 用 .exp(); 训练时 train.py 会把 logit_scale clamp 到 [0, 4.6](≈τ∈[1,100])。
        """
        # ===== TODO: 把下面这行替换成你的实现 =====
        raise NotImplementedError("填我: 对称 InfoNCE 对比损失 (见上方注释)")

    def forward(self, image_emb, text_emb, logit_scale):
        loss = self.clip_contrastive_loss(image_emb, text_emb, logit_scale)
        # 记录图→文/文→图 top-1 准确率(正样本是否排在相似度最高位), 用来观察对齐质量
        with torch.no_grad():
            logits = image_emb @ text_emb.t() * logit_scale.exp()
            labels = torch.arange(logits.shape[0], device=logits.device)
            i2t_acc = (logits.argmax(dim=1) == labels).float().mean().item()
            t2i_acc = (logits.argmax(dim=0) == labels).float().mean().item()
        return loss, {"loss": loss.item(), "i2t_acc": i2t_acc, "t2i_acc": t2i_acc}

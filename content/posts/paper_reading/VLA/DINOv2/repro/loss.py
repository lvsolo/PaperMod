"""
DINOv2 repro / loss.py
======================
DINOv2 的损失 = DINO 图像级损失(v1, 已预填) + 【iBOT patch 级损失】(TODO) + 【KoLeo 正则】(TODO)。

你要填的就是下面两个 TODO:
  ① ibot_patch_loss  —— 在被 mask 的 patch 位置上, student↔teacher 的交叉熵(patch 级自蒸馏)
  ② koleo_loss       —— 让 cls 特征在超球面上均匀分布(提升 retrieval/泛化)
forward 的组合/center 更新/teacher 目标准备都由脚手架写好, 调用你这俩方法。
对照: DINOv2.md §2.2 (iBOT)、§2.3 (KoLeo); 官方代码 dinov2/train/ssl_meta_arch.py。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DINOv2Loss(nn.Module):
    def __init__(self, out_dim, ncrops, n_global=2,
                 warmup_teacher_temp=0.04, teacher_temp=0.07,
                 warmup_teacher_temp_epochs=10, nepochs=30,
                 student_temp=0.1, center_momentum=0.9, koleo_weight=1.0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.n_global = n_global
        self.koleo_weight = koleo_weight
        # teacher 温度调度 (clamp, 同 v1)
        we = min(warmup_teacher_temp_epochs, max(0, nepochs - 1))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, we) if we > 0 else np.array([]),
            np.ones(max(1, nepochs - we)) * teacher_temp,
        ))
        # 两套 centering: 图像级(cls) 和 patch 级(iBOT) 各自维护
        self.register_buffer("center_cls", torch.zeros(1, out_dim))
        self.register_buffer("center_patch", torch.zeros(1, out_dim))

    # -------------------- 脚手架: teacher 目标 + cls 损失 (沿用 v1) --------------------
    def softmax_center_teacher(self, logits, center, temp):
        """teacher 输出 → 减 center → softmax(温度) → detach。作为不回传梯度的软标签。"""
        return F.softmax((logits - center) / temp, dim=-1).detach()

    def dino_cls_loss(self, student_cls_chunks, teacher_cls_chunks):
        """DINO v1 的图像级跨视角交叉熵(跳过同一视角)。student_cls 已除 student_temp。"""
        total, n = 0.0, 0
        for iq, q in enumerate(teacher_cls_chunks):
            for v in range(len(student_cls_chunks)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_cls_chunks[v], dim=-1), dim=-1)
                total = total + loss.mean(); n += 1
        return total / n

    # ====================================================================
    # TODO 1/2  —— iBOT patch 级损失  (DINOv2.md §2.2)
    # ====================================================================
    def ibot_patch_loss(self, student_patch_chunks, teacher_patch_chunks, mask_chunks):
        """
        在【被 mask 的 patch 位置】上做 student↔teacher 的交叉熵(patch 级自蒸馏)。

        输入(都已按 global crop 切好, 列表长度 = n_global):
            student_patch_chunks : list of (B, N, out_dim)  student 的 patch logits (输入是被 mask 过的)
            teacher_patch_chunks : list of (B, N, out_dim)  teacher 的 patch【目标】(已 softmax_center+detach)
            mask_chunks          : list of (B, N) bool      True = 该 patch 被 mask
        应返回: 标量 loss

        实现思路:
            对每个 global crop:
              logp = log_softmax(student_patch / self.student_temp, dim=-1)   # (B,N,out)
              per  = -sum(teacher_patch * logp, dim=-1)                        # (B,N) 每个 patch 的损失
              只保留 mask=True 的位置求平均 (用 mask 做加权, 除以 mask 数)
            再对 n_global 个 crop 取平均。
        提示: mask 是 bool, 转 float 后和 per 相乘; 分母 clamp_min(1) 防止全 False 除零。
        """
        raise NotImplementedError("TODO: 实现 ibot_patch_loss (masked patch 交叉熵)")

    # ====================================================================
    # TODO 2/2  —— KoLeo 正则  (DINOv2.md §2.3)
    # ====================================================================
    def koleo_loss(self, x):
        """
        KoLeo: 鼓励特征在超球面上均匀分布(每个点离最近邻越远越好)。

        输入:
            x : (M, d) 已经 L2 归一化的 cls bottleneck 特征(M = 本 batch 所有 crop)
        应返回: 标量 loss

        实现思路:
            1) 算两两点间欧氏距离矩阵 dist (M,M); 对角线填 +inf 排除自己
            2) d_i = 每个点到【最近邻】的距离 = dist.min(dim=1)
            3) loss = -mean( log(d_i + eps) )     # 越大(分布越开) loss 越小
        提示: dist = sqrt(((x[:,None,:]-x[None,:,:])**2).sum(-1) + 1e-12); 用 .fill_diagonal_(inf)。
        """
        raise NotImplementedError("TODO: 实现 koleo_loss (球面均匀分布正则)")

    # -------------------- 脚手架: 组合 + center 更新 (已写好, 调用上面两个 TODO) --------------------
    def forward(self, student_out, teacher_out, masks, epoch):
        """student_out/teacher_out: DINOv2Wrapper 的输出 dict; masks: student 的 mask 列表(global 才有)。"""
        temp = self.teacher_temp_schedule[epoch]
        # ---- ① DINO 图像级损失 (cls) ----
        s_cls = (student_out["cls_logits"] / self.student_temp).chunk(self.ncrops)
        t_cls_tgt = self.softmax_center_teacher(teacher_out["cls_logits"], self.center_cls, temp).chunk(self.n_global)
        loss_cls = self.dino_cls_loss(s_cls, t_cls_tgt)
        # ---- ② iBOT patch 级损失 (仅 global crop 的被 mask 位置) ----
        s_patch = (student_out["patch_logits"] / self.student_temp).chunk(self.n_global)
        t_patch_tgt = self.softmax_center_teacher(teacher_out["patch_logits"], self.center_patch, temp).chunk(self.n_global)
        loss_ibot = self.ibot_patch_loss(s_patch, t_patch_tgt, [m for m in masks if m is not None])
        # ---- ③ KoLeo (student cls bottleneck 特征) ----
        loss_koleo = self.koleo_loss(student_out["cls_feat"])
        # ---- 组合 ----
        total = loss_cls + loss_ibot + self.koleo_weight * loss_koleo
        # ---- 更新两套 center (EMA, 同 v1) ----
        self._update_center(self.center_cls, teacher_out["cls_logits"])
        self._update_center(self.center_patch, teacher_out["patch_logits"])
        return total, {"cls": loss_cls.item(), "ibot": loss_ibot.item(),
                       "koleo": loss_koleo.item(), "total": total.item()}

    @torch.no_grad()
    def _update_center(self, center, teacher_logits):
        bc = teacher_logits.mean(dim=0, keepdim=True)
        center.mul_(self.center_momentum).add_(bc, alpha=1 - self.center_momentum)

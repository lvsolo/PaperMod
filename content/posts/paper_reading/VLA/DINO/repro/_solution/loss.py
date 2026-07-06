"""
tiny_dino / loss.py
===================
DINO 的自蒸馏损失 —— 【这是论文的核心创新, 全部留给你填】。

你要实现两类东西:
  1) DINOLoss.forward         —— 跨视角蒸馏损失 (centering + sharpening + cross-entropy)
  2) DINOLoss.update_center   —— centering 向量的 EMA 更新

强烈建议先读 VLA/DINO.md 的 §2.4 / §2.4.1 / §2.5, 那里有逐行中文注释的官方实现可对照。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops                                 # 总 crop 数 = 2 global + N local
        self.n_global_crops = 2                              # teacher 只看 2 个 global crop
        # teacher 温度调度(线性 warmup 到 teacher_temp) —— 这只是调度, 不是算法, 已写好
        # 注意 clamp: warmup epochs 不能超过 nepochs-1, 否则短训练会负维度
        we = min(warmup_teacher_temp_epochs, max(0, nepochs - 1))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, we) if we > 0 else np.array([]),
            np.ones(max(1, nepochs - we)) * teacher_temp,
        ))
        # centering 向量, 形状 (1, out_dim), 初始为 0
        self.register_buffer("center", torch.zeros(1, out_dim))

    # ------------------------------------------------------------------
    # TODO 1/2  —— 跨视角蒸馏损失  (DINO.md §2.4 + §2.5)
    # ------------------------------------------------------------------
    def forward(self, student_output, teacher_output, epoch):
        """
        输入:
            student_output : (B*ncrops, out_dim)  student 在【全部 ncrops 个 crop】上的输出
            teacher_output : (B*2,      out_dim)  teacher 在【2 个 global crop】上的输出
            epoch          : int                  当前 epoch (用来取 teacher 温度)
        应返回:
            total_loss : 标量

        实现步骤(逐行对照 DINO.md §2.4.1 的官方代码):
            ① student:  student_out = (student_output / self.student_temp)
                        然后 student_out = student_out.chunk(self.ncrops)   # 切成 ncrops 份
            ② teacher 温度:  temp = self.teacher_temp_schedule[epoch]
            ③ teacher:  teacher_out = softmax((teacher_output - self.center) / temp, dim=-1)
                        teacher_out = teacher_out.detach().chunk(2)         # detach! 切成 2 份(global)
            ④ 双重循环求交叉熵:
                        total = 0; n = 0
                        for iq, q in enumerate(teacher_out):                # q: 某个 global 视角
                            for v in range(len(student_out)):               # v: 某个 student 视角
                                if v == iq:                                  # ⭐ 跳过同一视角!
                                    continue
                                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                                total = total + loss.mean()
                                n += 1
                        total_loss = total / n
            ⑤ 调用 self.update_center(teacher_output) 更新 centering
            ⑥ return total_loss

        检验(写完后自己跑一下):
            - loss 应该是有限正数;
            - 把 centering 关掉(见下面 update_center)会观察到 collapse。
        """
        # ① student: 除温度 + 切成 ncrops 份
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        # ②③ teacher: 减 center -> softmax(温度) -> detach -> 切成 2 份(global)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        # ④ 双重循环求交叉熵, 跳过 "同一视角" 的配对
        total_loss, n_loss_terms = 0.0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:                                  # ⭐ 跳过 student 与 teacher 同视角
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss = total_loss + loss.mean()
                n_loss_terms += 1
        total_loss = total_loss / n_loss_terms
        # ⑤ 更新 centering
        self.update_center(teacher_output)
        return total_loss

    # ------------------------------------------------------------------
    # TODO 2/2  —— centering 向量 EMA 更新  (DINO.md §2.5)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        用本 batch teacher 输出的均值, EMA 更新 self.center。

        实现步骤:
            batch_center = teacher_output.mean(dim=0, keepdim=True)         # (1, out_dim)
            self.center = self.center * self.center_momentum
                        + batch_center * (1 - self.center_momentum)

        物理意义: center 追踪 teacher 输出分布的中心, 让 teacher 减去它后"居中",
                  防止某一维独大导致 collapse。 (DINO.md §2.5 的 ① Centering)
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)        # (1, out_dim)
        # in-place EMA: center = momentum*center + (1-momentum)*batch_center
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


# === 一个用于自查的小测试: 填完 forward 后跑这个能跑通即说明接口对 ===
if __name__ == "__main__":
    B, ncrops, out_dim = 4, 10, 65536
    loss_fn = DINOLoss(out_dim, ncrops, warmup_teacher_temp=0.04, teacher_temp=0.04,
                       warmup_teacher_temp_epochs=0, nepochs=10)
    try:
        s = torch.randn(B * ncrops, out_dim)
        t = torch.randn(B * 2, out_dim)
        l = loss_fn(s, t, epoch=3)
        print("forward OK, loss =", float(l))
        print("center after update:", loss_fn.center.mean().item(), "(应非零)")
    except NotImplementedError as e:
        print("还没填:", e)

"""HaltHead — ACT 软停止 + 温度退火二值化

训练: p_halt = sigmoid(halt_logit / τ_halt), 所有轮都跑, 加权和输出
推理: p_halt > 0.5 硬停止
τ_halt 从 1.0 退火到 0.01 → p_halt 从平滑变二值化 → 训练/推理一致
"""

import torch
import torch.nn as nn


class HaltHead(nn.Module):
    """停止决策头

    halt_logit = Linear(mean_pool(h))
    p_halt = sigmoid(halt_logit / τ_halt)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        # 初始化 bias 使得初期倾向于继续循环
        nn.init.constant_(self.linear.bias, -1.0)

    def forward(
        self, h: torch.Tensor, tau_halt: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, d_model] — 当前循环隐状态
            tau_halt: 退火温度 (1.0→0.01)

        Returns:
            p_halt: [batch] — 每个样本的停止概率
        """
        # mean pool over sequence, match linear weight dtype (CPU fp16 may promote)
        target_dtype = self.linear.weight.dtype
        h_pooled = h.mean(dim=1).to(target_dtype)  # [batch, d_model]
        halt_logit = self.linear(h_pooled).squeeze(-1)  # [batch]
        p_halt = torch.sigmoid(halt_logit / tau_halt)
        return p_halt

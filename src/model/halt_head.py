"""HaltHead — Per-token ACT 软停止 + 温度退火二值化

每个 token 独立决定是否停止循环:
- easy token (高预测准确率) → 早停
- hard token (低预测准确率) → 多迭代

p_halt = sigmoid(Linear(h) / τ_halt)  — [batch, seq_len]
τ_halt 从 1.0 退火到 0.01 → p_halt 从平滑变二值化
"""

import torch
import torch.nn as nn


class HaltHead(nn.Module):
    """Per-token 停止决策头

    halt_logit = Linear(h)  → [batch, seq_len, 1]
    p_halt = sigmoid(halt_logit / τ_halt)  → [batch, seq_len]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        nn.init.constant_(self.linear.bias, -1.0)

    def forward(
        self, h: torch.Tensor, tau_halt: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, d_model] — 当前循环隐状态
            tau_halt: 退火温度 (1.0→0.01)

        Returns:
            p_halt: [batch, seq_len] — 每个 token 的停止概率
        """
        halt_logit = self.linear(h.float()).squeeze(-1)  # fp32 防 sigmoid 饱和
        p_halt = torch.sigmoid(halt_logit / tau_halt)
        return p_halt

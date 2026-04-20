"""EntropyProbe — 轻量 entropy 预测器

从隐状态 h 直接预测 H_norm ∈ [0, 1]，绕过 back → norm → lm_head (128K) 管线。
用于中间迭代的 L_mono 梯度信号，同时以 MSE 与真实 entropy 对齐。

关键设计：
- L_mono 梯度不流经 probe 权重（使用 functional_call + detached params）
- 只有 MSE loss 训练 probe 参数
- 避免 probe "假装"单调递减
"""

import torch
import torch.nn as nn


class EntropyProbe(nn.Module):
    """Lightweight entropy predictor: h [B,T,D] → H_norm [B,T]"""

    def __init__(self, d_model: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, D] → [B, T]"""
        return self.net(h).squeeze(-1)

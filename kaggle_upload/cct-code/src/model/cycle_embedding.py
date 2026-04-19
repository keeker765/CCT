"""RotaryCycleEmbedding — 黄金比例旋转循环嵌入

在 Q/K 投影前施加，编码循环轮次信息。
angle_i(k) = k × φ × base_freq^(-2i/d_model)
范数不变: ||R·x|| = ||x||，0 可学习参数。
"""

import torch
import torch.nn as nn
import math


class RotaryCycleEmbedding(nn.Module):
    """旋转循环嵌入 (Rotary Cycle Embedding)

    对 hidden_states 的每个维度对 (2i, 2i+1) 施加旋转，
    角度由循环轮次 k 和黄金比例 φ 决定。

    必须在 W_Q/W_K 投影前施加：
    - 投影前: Q = W_Q(R_k · h)，R_k 信息保留在 Q·K^T 中
    - 投影后: Q = R_k · W_Q(h)，Q·K^T = h^T W_Q^T R_k^T R_k W_Q h = h^T W_Q^T W_Q h
      → R_k^T R_k = I，循环信息被正交性消除
    """

    def __init__(self, d_model: int, phi: float = 1.618, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.phi = phi

        # base_freq^(-2i/d_model) for i = 0, 1, ..., d_model/2 - 1
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, hidden_states: torch.Tensor, cycle_k: int
    ) -> torch.Tensor:
        """对 hidden_states 施加循环旋转

        Args:
            hidden_states: [batch, seq_len, d_model]
            cycle_k: 当前循环轮次 (0-indexed)

        Returns:
            rotated: [batch, seq_len, d_model]，范数不变
        """
        if cycle_k == 0:
            return hidden_states

        # angle_i = k × φ × inv_freq_i
        angles = cycle_k * self.phi * self.inv_freq  # [d_model/2]

        cos_vals = angles.cos().to(hidden_states.dtype)  # [d_model/2]
        sin_vals = angles.sin().to(hidden_states.dtype)  # [d_model/2]

        # 分割为偶数/奇数维度
        x_even = hidden_states[..., 0::2]  # [..., d_model/2]
        x_odd = hidden_states[..., 1::2]   # [..., d_model/2]

        # 旋转: (cos·x_even - sin·x_odd, sin·x_even + cos·x_odd)
        rotated_even = cos_vals * x_even - sin_vals * x_odd
        rotated_odd = sin_vals * x_even + cos_vals * x_odd

        # 交错合并
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        rotated = rotated.reshape_as(hidden_states)

        return rotated

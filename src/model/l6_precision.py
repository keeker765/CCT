"""L6Precision — 精度加权注意力调制

precision = 1 - sigmoid(score / τ_p)
attention_bias = λ · precision

纯调制: 只影响 attention logits, 不融合输入, 不做残差。
τ_p 是固定超参数 (推荐 0.5), 控制 easy/hard token 区分度。
"""

import torch
import torch.nn as nn


class L6Precision(nn.Module):
    """L6 精度加权模块

    将 prediction score 转换为 attention bias:
    1. precision = 1 - sigmoid(score / τ_p)
       - score 高 (预测准) → precision 低 → 少关注
       - score 低 (预测差) → precision 高 → 多关注
    2. attention_bias = λ · precision
       - λ 为可学习标量, 控制调制强度
    """

    def __init__(
        self,
        lambda_init: float = 1.0,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.lambda_precision = nn.Parameter(torch.tensor(lambda_init))
        self.temperature = temperature  # 固定超参数

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score: [batch, seq_len] — prediction score (已 detach)

        Returns:
            attention_bias: [batch, seq_len] — 用于调制 attention logits
        """
        precision = 1.0 - torch.sigmoid(score / self.temperature)
        attention_bias = self.lambda_precision * precision
        return attention_bias

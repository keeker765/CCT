"""L6Precision — 皮层柱 L6 统一模块: 注意力调制 + 停止决策

生物学依据:
  - L6CT → TRN → Thalamus: prediction error 驱动注意力增益控制
    (Sherman & Guillery 2006; Kahn et al. 2010)
  - Free Energy 收敛: error → 0 时处理终止, L5 输出当前信念
    (Friston 2005; Bastos et al. 2012)
  - L6CT → L5a 强兴奋: L6 激活同时抑制输入层(L4)、兴奋输出层(L5)
    (Kim et al. 2014, J Neurosci)

两个输出, 同一个信号 (prediction score):
  attention_bias = λ · (1 - σ(score / τ_p))    — 注意力增益
  p_halt = σ((gain · score + bias) / τ_halt)    — 停止概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L6Precision(nn.Module):
    """L6 皮层柱核心: prediction error → attention + halt

    注意力调制 (L6→Thalamus):
      precision = 1 - σ(score / τ_p)
      attention_bias = λ · precision
      高 score (好预测) → precision 低 → 少关注
      低 score (surprise) → precision 高 → 多关注

    停止决策 (L6→L5, Free Energy 收敛):
      p_halt = σ((softplus(gain) · score + bias) / τ_halt)
      高 score → 收敛 → halt
      低 score → 继续处理
    """

    def __init__(
        self,
        lambda_init: float = 1.0,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.lambda_precision = nn.Parameter(torch.tensor(lambda_init))
        self.temperature = temperature  # 固定超参数

        # Halt 参数: score → 停止决策
        self._halt_gain = nn.Parameter(torch.tensor(1.0))   # softplus 保证 > 0
        self.halt_bias = nn.Parameter(torch.tensor(-2.0))    # 初始偏低, 鼓励多迭代

    def compute_attention_bias(self, score: torch.Tensor) -> torch.Tensor:
        """Score → attention 增益调制 (用于 column layers)

        Args:
            score: [batch, seq_len] — prediction score (detached)
        Returns:
            attention_bias: [batch, seq_len]
        """
        precision = 1.0 - torch.sigmoid(score / self.temperature)
        return self.lambda_precision * precision

    def compute_halt(
        self, score: torch.Tensor, tau_halt: float = 1.0
    ) -> torch.Tensor:
        """Score → 停止概率 (用于 ACT)

        p_halt = σ((softplus(gain) · score + bias) / τ_halt)

        Args:
            score: [batch, seq_len] — prediction score (detached)
            tau_halt: 退火温度 (1.0 → 0.01)
        Returns:
            p_halt: [batch, seq_len]
        """
        gain = F.softplus(self._halt_gain)
        halt_logit = gain * score + self.halt_bias
        return torch.sigmoid(halt_logit / tau_halt)

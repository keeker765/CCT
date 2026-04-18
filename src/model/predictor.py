"""CCT Predictor — 前向预测 (predict column output BEFORE seeing it)

设计 (与 PPG GroupPredictor 完全对齐):
  PPG:  predict(info_proj(h_before_anchor))  vs  info_proj(h_after_anchor)
  CCT:  predict(info_proj(h_before_column))  vs  info_proj(h_after_column)

为什么旧设计 (predict x_column from h) 会崩塌:
  当 eff_iters≈1 时, h ≈ x_column → info_proj(h) ≈ info_proj(x_column) → cos_sim≈1
  无论加多少层 predictor head, 任务本身就太简单了

为什么新设计 (predict h_k from h_{k-1}) 不会:
  h_{k-1} 和 h_k 经过 3 层 column 变换, 有 6 次残差 + self-attn + FFN
  即使 column 接近恒等, 变换 Δ 也是 token-dependent → 有意义的 per-token 方差
  predictor 必须学会预测 column 的变换行为, 而非简单恒等映射

梯度流向 (与 PPG 一致):
  预测侧: h_prev.detach() → info_proj → predictor → z_pred
           ✓ info_proj + predictor 获得梯度
  目标侧: h_curr.detach() → info_proj → z_anchor.detach()
           ✗ 双重 detach 防止共享投影坍缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CCTPredictor(nn.Module):
    """前向预测器: predict column output from pre-column state

    info_proj: Linear(d_model → info_dim, bias=False) — 共享投影
    predictor: Linear(info_dim → info_dim) — 预测 head
    """

    def __init__(self, d_model: int, info_dim: int = 256):
        super().__init__()
        self.info_dim = info_dim
        self.info_proj = nn.Linear(d_model, info_dim, bias=False)
        self.predictor = nn.Linear(info_dim, info_dim)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """投影到 info 空间 (用于 anchor/target)"""
        return self.info_proj(x.to(self.info_proj.weight.dtype))

    def predict(self, h_prev: torch.Tensor) -> torch.Tensor:
        """预测下一轮 column 输出的 info 表征"""
        z = self.project(h_prev)
        return self.predictor(z)

    def compute_pred_loss(
        self, h_prev: torch.Tensor, h_curr: torch.Tensor
    ) -> torch.Tensor:
        """L_pred: 预测列变换的准确度

        prediction = predictor(info_proj(h_prev.detach()))   — 从上一轮状态预测
        z_anchor = info_proj(h_curr.detach()).detach()        — 实际列输出

        Args:
            h_prev: [B, T, D] — 列运算前的状态 (iteration k-1 的输出, 或 x_column at k=0)
            h_curr: [B, T, D] — 列运算后的状态 (iteration k 的输出)
        Returns:
            loss: 标量
        """
        z_pred = self.predict(h_prev.detach())
        z_anchor = self.project(h_curr.detach()).detach()
        cos_sim = F.cosine_similarity(
            z_pred.float(), z_anchor.float(), dim=-1
        )
        return (1.0 - cos_sim).mean()

    def compute_score(
        self, h_prev: torch.Tensor, h_curr: torch.Tensor
    ) -> torch.Tensor:
        """Per-token prediction score: column 输出的可预测性

        使用 cosine similarity (有界 [-1, 1])，确保下游 L6Precision
        的 sigmoid(score / tau_p) 不会因 score 过大而饱和。

        高 score → 列变换可预测 → 低 precision → 正常 attention
        低 score → 列变换出乎意料 → 高 precision → 增强 attention

        Args:
            h_prev: [B, T, D] — 列运算前
            h_curr: [B, T, D] — 列运算后
        Returns:
            score: [B, T] — cosine similarity, 范围 [-1, 1]
        """
        z_pred = self.predict(h_prev).detach()
        z_anchor = self.project(h_curr.detach()).detach()
        score = F.cosine_similarity(z_pred.float(), z_anchor.float(), dim=-1)
        return score

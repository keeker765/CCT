"""CCT Predictor — 前向预测 (predict column residual Δ)

设计:
  predictor 预测的是 column 变换的 **残差** delta = h_curr - h_prev,
  而不是 h_curr 本身。

为什么预测残差:
  column 层含 residual connection → h_curr ≈ h_prev + δ
  直接比较 h_prev 和 h_curr 的 cosine similarity 天然接近 1
  → score 饱和 → L6 Precision 无法区分 token
  预测 δ 后, 预测难度取决于 column 变换的非线性部分,
  token-dependent 方差更大, score 分布更展开。

梯度流向:
  预测侧: h_prev.detach() → info_proj → predictor → z_pred
           ✓ info_proj + predictor 获得梯度
  目标侧: delta.detach() → info_proj → z_anchor.detach()
           ✗ 双重 detach 防止共享投影坍缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CCTPredictor(nn.Module):
    """前向预测器: predict column residual from pre-column state

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
        """预测 column 残差的 info 表征"""
        z = self.project(h_prev)
        return self.predictor(z)

    def compute_pred_loss(
        self, h_prev: torch.Tensor, h_curr: torch.Tensor
    ) -> torch.Tensor:
        """L_pred: 预测列变换残差的准确度

        prediction = predictor(info_proj(h_prev.detach()))
        z_anchor = info_proj((h_curr - h_prev).detach()).detach()

        Args:
            h_prev: [B, T, D] — 列运算前
            h_curr: [B, T, D] — 列运算后
        Returns:
            loss: 标量
        """
        delta = (h_curr - h_prev).detach()
        z_pred = self.predict(h_prev.detach())
        z_anchor = self.project(delta).detach()
        cos_sim = F.cosine_similarity(
            z_pred.float(), z_anchor.float(), dim=-1
        )
        return (1.0 - cos_sim).mean()

    def compute_score(
        self, h_prev: torch.Tensor, h_curr: torch.Tensor
    ) -> torch.Tensor:
        """Per-token prediction score: column 残差的可预测性

        高 score → 残差可预测 → 低 precision → 正常 attention
        低 score → 残差出乎意料 → 高 precision → 增强 attention

        Args:
            h_prev: [B, T, D] — 列运算前
            h_curr: [B, T, D] — 列运算后
        Returns:
            score: [B, T] — cosine similarity, 范围 [-1, 1]
        """
        delta = (h_curr - h_prev).detach()
        z_pred = self.predict(h_prev).detach()
        z_anchor = self.project(delta).detach()
        score = F.cosine_similarity(z_pred.float(), z_anchor.float(), dim=-1)
        return score

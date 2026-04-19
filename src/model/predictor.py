"""CCT Predictor — 前向预测 (predict column residual Δ, 低容量版)

设计:
  predictor 直接从 d_model 空间映射到 info_dim (单层),
  去掉了 info_proj 共享投影, 降低 predictor 容量防止塌缩。

  目标侧使用冻结随机投影 (target_proj), 不参与优化。
  predictor 无法通过联合优化投影空间来降低任务难度。

梯度流向:
  预测侧: h_prev.detach() → predictor(d_model→info_dim) → z_pred
           ✓ predictor 获得梯度
  目标侧: delta.detach() → target_proj(冻结) → z_anchor.detach()
           ✗ 完全无梯度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CCTPredictor(nn.Module):
    """前向预测器: predict column residual from pre-column state

    predictor:   Linear(d_model → info_dim) — 直接映射, 单层
    target_proj: Linear(d_model → info_dim) — 冻结随机投影
    """

    def __init__(self, d_model: int, info_dim: int = 256):
        super().__init__()
        self.info_dim = info_dim
        self.predictor = nn.Linear(d_model, info_dim)
        # 冻结随机投影: delta → info_dim (不可训练)
        self.target_proj = nn.Linear(d_model, info_dim, bias=False)
        self.target_proj.weight.requires_grad_(False)

    def predict(self, h_prev: torch.Tensor) -> torch.Tensor:
        """从 h_prev 直接预测 delta 的投影"""
        return self.predictor(h_prev.to(self.predictor.weight.dtype))

    def compute_pred_loss(
        self, h_prev: torch.Tensor, h_curr: torch.Tensor
    ) -> torch.Tensor:
        """L_pred: 预测列变换残差的准确度

        Args:
            h_prev: [B, T, D] — 列运算前
            h_curr: [B, T, D] — 列运算后
        Returns:
            loss: 标量
        """
        delta = (h_curr - h_prev).detach()
        z_pred = self.predict(h_prev.detach())
        z_anchor = self.target_proj(
            delta.to(self.target_proj.weight.dtype)
        ).detach()
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
        z_anchor = self.target_proj(
            delta.to(self.target_proj.weight.dtype)
        ).detach()
        score = F.cosine_similarity(z_pred.float(), z_anchor.float(), dim=-1)
        return score

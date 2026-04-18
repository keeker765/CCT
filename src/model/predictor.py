"""CCT Predictor — info_proj + predictor head (与 PPG 一致)

设计:
1. info_proj: Linear(d_model → info_dim, bias=False) — 投影到比较空间
2. predictor: Linear(info_dim → info_dim) — 在比较空间中预测 anchor

为什么需要两层:
- 只有 info_proj 时, info_proj(h) ≈ info_proj(x_column) → 平凡解 (cos_sim≈1)
- 加 predictor 后: predictor(info_proj(h)) ≠ info_proj(x_column) → 必须学习真正的预测
- 这与 PPG (GroupPredictor) 的设计完全一致

梯度流向:
  预测侧: h.detach() → info_proj → predictor → z_pred
           ✓ info_proj 和 predictor 都获得梯度

  目标侧: x_column.detach() → info_proj → z_anchor.detach()
           ✗ info_proj 不从此侧获得梯度 (第2次 detach)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CCTPredictor(nn.Module):
    """info_proj + predictor 双层预测器

    info_proj: Linear(d_model → info_dim, bias=False)
    predictor: Linear(info_dim → info_dim) — 预测 head
    """

    def __init__(self, d_model: int, info_dim: int = 256):
        super().__init__()
        self.info_dim = info_dim
        self.info_proj = nn.Linear(d_model, info_dim, bias=False)
        self.predictor = nn.Linear(info_dim, info_dim)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """投影到 info 空间 (仅 info_proj, 用于 anchor)"""
        return self.info_proj(x.to(self.info_proj.weight.dtype))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测: info_proj → predictor (用于 prediction 侧)"""
        z = self.project(x)
        return self.predictor(z)

    def compute_pred_loss(
        self, h: torch.Tensor, x_column: torch.Tensor
    ) -> torch.Tensor:
        """计算预测损失 L_pred

        prediction = predictor(info_proj(h.detach()))    — 2 层
        z_anchor = info_proj(x_column.detach()).detach() — 1 层 + detach

        Args:
            h: [batch, seq_len, d_model] — Column 循环输出
            x_column: [batch, seq_len, d_model] — Column 原始输入
        Returns:
            loss: 标量
        """
        z_pred = self.predict(h.detach())
        z_anchor = self.project(x_column.detach()).detach()
        cos_sim = F.cosine_similarity(
            z_pred.float(), z_anchor.float(), dim=-1
        )
        return (1.0 - cos_sim).mean()

    def compute_score(
        self, h: torch.Tensor, x_column: torch.Tensor
    ) -> torch.Tensor:
        """计算 per-token prediction score (双重 detach)

        score_t = dot(z_pred_t, z_anchor_t) / √info_dim

        Args:
            h: [batch, seq_len, d_model]
            x_column: [batch, seq_len, d_model]
        Returns:
            score: [batch, seq_len]
        """
        z_pred = self.predict(h).detach()
        z_anchor = self.project(x_column.detach()).detach()
        score = (z_pred * z_anchor).sum(dim=-1) / math.sqrt(self.info_dim)
        return score

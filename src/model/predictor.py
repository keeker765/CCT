"""CCT Predictor — 共享 info_proj + 双重 detach 防崩塌

设计:
1. info_proj: 共享投影 d_model → info_dim, 将 h 和 x_column 映射到同一比较空间
2. 双重 detach: 目标侧 .detach() 防止崩塌

梯度流向:
  预测侧: h.detach() → info_proj → z_pred
           ✓ info_proj 获得梯度

  目标侧: x_column.detach() → info_proj → z_anchor.detach()
           ✗ info_proj 不从此侧获得梯度 (第2次 detach)

为什么需要 info_proj:
- h 经过旋转循环编码 + Column 层处理, 与 x_column 分布不同
- info_proj 将两者投影到共同的低维比较空间

为什么用余弦距离而非 MSE:
- MSE 可被 info_proj 缩小输出 norm 来作弊
- 余弦距离只比较方向, 必须学到真正的方向预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CCTPredictor(nn.Module):
    """共享 info_proj 预测器

    info_proj: Linear(d_model → info_dim, bias=False) — 一层投影
    """

    def __init__(self, d_model: int, info_dim: int = 256):
        super().__init__()
        self.info_dim = info_dim
        self.info_proj = nn.Linear(d_model, info_dim, bias=False)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """投影到 info 空间"""
        return self.info_proj(x.to(self.info_proj.weight.dtype))

    def compute_pred_loss(
        self, h: torch.Tensor, x_column: torch.Tensor
    ) -> torch.Tensor:
        """计算预测损失 L_pred (双重 detach 防崩塌)

        L_pred = 1 - cos_sim(info_proj(h.detach()),
                              info_proj(x_column.detach()).detach())

        第1次 detach (h.detach, x_column.detach): 隔离基座模型梯度
        第2次 detach (z_anchor.detach): 阻止 L_pred 从目标侧更新 info_proj

        Args:
            h: [batch, seq_len, d_model] — Column 循环输出
            x_column: [batch, seq_len, d_model] — Column 原始输入
        Returns:
            loss: 标量
        """
        z_pred = self.project(h.detach())
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
        z_pred = self.project(h).detach()
        z_anchor = self.project(x_column.detach()).detach()
        score = (z_pred * z_anchor).sum(dim=-1) / math.sqrt(self.info_dim)
        return score

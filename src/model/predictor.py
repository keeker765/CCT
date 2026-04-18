"""CCT Predictor — 共享 info_proj + 双重 detach 防崩塌

设计来自 PPG (layerPickerLLM), 核心思想:
1. info_proj: 共享投影, 将 d_model → info_dim (降维)
2. predictor: 在 info_dim 空间中做预测
3. 双重 detach: 防止表征崩塌 (representation collapse)

梯度流向:
  预测侧: h.detach() → info_proj → predictor → prediction
           ✓ info_proj 获得梯度    ✓ predictor 获得梯度

  目标侧: x_column.detach() → info_proj → z_anchor.detach()
           ✗ info_proj 不从此侧获得梯度 (第2次 detach)

为什么用余弦距离而非 MSE?
- MSE 可被 info_proj 缩小输出 norm 来作弊 (error→0 但没学到真正预测)
- 余弦距离只比较方向, 不受 norm 影响 → 必须学到真正的方向预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CCTPredictor(nn.Module):
    """共享 info_proj 预测器 (PPG 风格)

    info_proj: Linear(d_model → info_dim, bias=False) — 共享
    predictor: Linear(info_dim → info_dim) — 仅预测侧
    """

    def __init__(self, d_model: int, info_dim: int = 256):
        super().__init__()
        self.info_dim = info_dim
        self.info_proj = nn.Linear(d_model, info_dim, bias=False)
        self.predictor = nn.Linear(info_dim, info_dim)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """预测侧: info_proj → predictor

        Args:
            h: [batch, seq_len, d_model] — 当前循环输出
        Returns:
            prediction: [batch, seq_len, info_dim]
        """
        z = self.info_proj(h.to(self.info_proj.weight.dtype))
        return self.predictor(z)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """投影到 info 空间 (用于目标侧和 score)

        Args:
            x: [batch, seq_len, d_model]
        Returns:
            z: [batch, seq_len, info_dim]
        """
        return self.info_proj(x.to(self.info_proj.weight.dtype))

    def compute_pred_loss(
        self, h: torch.Tensor, x_column: torch.Tensor
    ) -> torch.Tensor:
        """计算预测损失 L_pred (双重 detach 防崩塌)

        L_pred = 1 - cos_sim(predictor(info_proj(h.detach())),
                              info_proj(x_column.detach()).detach())

        第1次 detach (h.detach, x_column.detach): 隔离基座模型梯度
        第2次 detach (z_anchor.detach): 阻止 L_pred 从目标侧更新 info_proj
          → info_proj 仅从预测侧获得梯度, 防止共享投影坍缩

        Args:
            h: [batch, seq_len, d_model] — Column 循环输出
            x_column: [batch, seq_len, d_model] — Column 原始输入
        Returns:
            loss: 标量
        """
        prediction = self.predict(h.detach())
        z_anchor = self.project(x_column.detach()).detach()  # 第2次 detach!
        cos_sim = F.cosine_similarity(
            prediction.float(), z_anchor.float(), dim=-1
        )
        return (1.0 - cos_sim).mean()

    def compute_score(
        self, h: torch.Tensor, x_column: torch.Tensor
    ) -> torch.Tensor:
        """计算 per-token prediction score (双重 detach)

        score_t = dot(z_pred_t, z_anchor_t) / √info_dim
        用于生成 precision 信号, 不创建梯度回路。

        Args:
            h: [batch, seq_len, d_model] — Column 循环输出
            x_column: [batch, seq_len, d_model] — Column 原始输入
        Returns:
            score: [batch, seq_len]
        """
        z_pred = self.project(h).detach()
        z_anchor = self.project(x_column.detach()).detach()
        score = (z_pred * z_anchor).sum(dim=-1) / math.sqrt(self.info_dim)
        return score

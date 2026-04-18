"""Predictor + AnchorMLP — 预测编码核心组件

CCT 与 PPG 的关键区别:
- PPG: info_proj 共享 → 需要双重 detach 防止共享投影坍缩
- CCT: Predictor 和 AnchorMLP 完全独立 → 不需要双重 detach

Detach 策略:
- L_pred: cos_sim(Predictor(h.detach()), AnchorMLP(x_column.detach()))
  → h/x_column detach 隔离基座梯度; AnchorMLP 输出不 detach (需要梯度学习)
- Score: dot(pred.detach(), anchor.detach()) / √d
  → 双重 detach, precision 是只读信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Predictor(nn.Module):
    """L2/3 → L5 投射: 线性预测器

    从当前循环的 hidden state 预测下一输入的锚定表征。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, d_model] — 当前循环输出
        Returns:
            prediction: [batch, seq_len, d_model]
        """
        return self.proj(h.to(self.proj.weight.dtype))


class AnchorMLP(nn.Module):
    """丘脑 → L5 直达通路: 2层 MLP

    将 Column 的原始输入 x_column 映射到锚定空间。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] — Column 原始输入
        Returns:
            anchor: [batch, seq_len, d_model]
        """
        return self.fc2(self.act(self.fc1(x.to(self.fc1.weight.dtype))))


def compute_score(
    pred: torch.Tensor, anchor: torch.Tensor, d_model: int
) -> torch.Tensor:
    """计算 per-token prediction score (双重 detach)

    score_t = dot(pred_t, anchor_t) / √d_model
    用于生成 precision 信号, 不创建梯度回路。

    Args:
        pred: [batch, seq_len, d_model] — Predictor 输出
        anchor: [batch, seq_len, d_model] — AnchorMLP 输出
        d_model: 模型维度

    Returns:
        score: [batch, seq_len] — 每个 token 的预测质量分数
    """
    # 双重 detach: precision 是只读信号
    pred_d = pred.detach()
    anchor_d = anchor.detach()
    score = (pred_d * anchor_d).sum(dim=-1) / math.sqrt(d_model)
    return score


def compute_pred_loss(
    predictor: Predictor,
    anchor_mlp: AnchorMLP,
    h: torch.Tensor,
    x_column: torch.Tensor,
) -> torch.Tensor:
    """计算预测损失 L_pred

    L_pred = 1 - cos_sim(Predictor(h.detach()), AnchorMLP(x_column.detach()))
    - h.detach(): 阻止梯度回传到基座模型
    - x_column.detach(): 同上
    - AnchorMLP 输出不 detach: 需要从 L_pred 获得梯度来学习

    Args:
        predictor: Predictor 模块
        anchor_mlp: AnchorMLP 模块
        h: [batch, seq_len, d_model] — Column 循环输出
        x_column: [batch, seq_len, d_model] — Column 原始输入

    Returns:
        loss: 标量, mean over all tokens
    """
    pred = predictor(h.detach())
    anchor = anchor_mlp(x_column.detach())
    cos_sim = F.cosine_similarity(pred.float(), anchor.float(), dim=-1)
    return (1.0 - cos_sim).mean()

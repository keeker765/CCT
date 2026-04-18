"""CCT 损失函数

L_total = L_LM + λ_pred · L_pred + λ_flops · L_flops

- L_LM: CrossEntropy (next-token), 驱动整个模型
- L_pred: 预测损失, 仅训练 Predictor/AnchorMLP (h.detach)
- L_flops: ponder_cost 线性惩罚 (标准 ACT)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple


def compute_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """标准 next-token CrossEntropy 损失"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def compute_flops_loss(
    p_halts: List[torch.Tensor],
    remainders: List[torch.Tensor],
) -> torch.Tensor:
    """计算 ACT ponder cost (线性)

    ponder_cost = Σ_k (remainder_k · p_halt_k · k)
    近似期望迭代次数, L_flops 梯度通过 halt_head(h) 回传到 Column。

    Args:
        p_halts: list of [batch] — 每轮的 p_halt
        remainders: list of [batch] — 每轮开始时的 remainder

    Returns:
        loss: 标量
    """
    ponder_cost = torch.zeros(1, device=p_halts[0].device)
    for k, (p_halt, remainder) in enumerate(zip(p_halts, remainders)):
        ponder_cost = ponder_cost + (remainder * p_halt * k).mean()

    max_iter = len(p_halts)
    # 归一化到 [0, 1] 范围
    ponder_cost = ponder_cost / max_iter
    return ponder_cost


def compute_total_loss(
    lm_loss: torch.Tensor,
    pred_losses: List[torch.Tensor],
    p_halts: List[torch.Tensor],
    remainders: List[torch.Tensor],
    lambda_pred: float = 0.1,
    lambda_flops: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算总损失

    L_total = L_LM + λ_pred · mean(L_pred_k) + λ_flops · L_flops

    Args:
        lm_loss: 语言模型损失
        pred_losses: list of scalar — 每轮的 L_pred
        p_halts: list of [batch] — 每轮的 p_halt
        remainders: list of [batch] — 每轮的 remainder
        lambda_pred: L_pred 权重
        lambda_flops: L_flops 权重

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失详情
    """
    # L_pred: 所有迭代的平均
    if pred_losses:
        l_pred = torch.stack(pred_losses).mean()
    else:
        l_pred = torch.tensor(0.0, device=lm_loss.device)

    # L_flops
    if p_halts:
        l_flops = compute_flops_loss(p_halts, remainders)
    else:
        l_flops = torch.tensor(0.0, device=lm_loss.device)

    total = lm_loss + lambda_pred * l_pred + lambda_flops * l_flops

    loss_dict = {
        "loss_total": total.item(),
        "loss_lm": lm_loss.item(),
        "loss_pred": l_pred.item(),
        "loss_flops": l_flops.item(),
    }

    return total, loss_dict

"""CCT 损失函数

L_total = L_LM + λ_pred · L_pred + λ_entropy · L_entropy

- L_LM: CrossEntropy (next-token), 驱动整个模型
- L_pred: 预测损失, 仅训练 info_proj (h.detach)
- L_entropy: per-token halting 分布熵 (最小化 → 锐利决策)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


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



def compute_halt_entropy(
    p_halts: List[torch.Tensor],
    remainders: List[torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-token halting 分布熵

    对每个 token, halting 分布为:
    [w_0, w_1, ..., w_{K-1}, remainder_final]

    H_t = -Σ_k m_{k,t} · log(m_{k,t} + ε)

    最小化 H → 每个 token 果断选择在某一轮停止

    Args:
        p_halts: list of [batch, seq_len]
        remainders: list of [batch, seq_len]
        valid_mask: [batch, seq_len] — 1=有效, 0=padding

    Returns:
        entropy: 标量
    """
    eps = 1e-8
    all_weights = []
    for p_halt, remainder in zip(p_halts, remainders):
        w = remainder * p_halt  # [batch, seq_len]
        all_weights.append(w)

    # 最后一轮的剩余 remainder (分布的最后一个桶)
    if remainders:
        final_remainder = remainders[-1] * (1.0 - p_halts[-1])
        all_weights.append(final_remainder)

    # Stack: [K+1, batch, seq_len]
    weights = torch.stack(all_weights, dim=0)

    # Per-token entropy: H_t = -Σ_k w_k * log(w_k + eps)
    entropy = -(weights * torch.log(weights + eps)).sum(dim=0)  # [batch, seq_len]

    if valid_mask is not None:
        entropy = (entropy * valid_mask).sum() / (valid_mask.sum() + eps)
    else:
        entropy = entropy.mean()

    return entropy


def compute_total_loss(
    lm_loss: torch.Tensor,
    pred_losses: List[torch.Tensor],
    p_halts: List[torch.Tensor],
    remainders: List[torch.Tensor],
    lambda_pred: float = 0.1,
    lambda_entropy: float = 0.01,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算总损失

    L_total = L_LM + λ_pred · L_pred + λ_entropy · L_entropy

    Args:
        lm_loss: 语言模型损失
        pred_losses: list of scalar — 每轮的 L_pred
        p_halts: list of [batch, seq_len] — 每轮的 p_halt
        remainders: list of [batch, seq_len] — 每轮的 remainder
        lambda_pred: L_pred 权重
        lambda_entropy: L_entropy 权重 (最小化熵)
        valid_mask: [batch, seq_len] — padding mask

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失详情
    """
    # L_pred: 所有迭代的平均
    if pred_losses:
        l_pred = torch.stack(pred_losses).mean()
    else:
        l_pred = torch.tensor(0.0, device=lm_loss.device)

    # L_entropy
    if p_halts:
        l_entropy = compute_halt_entropy(p_halts, remainders, valid_mask)
    else:
        l_entropy = torch.tensor(0.0, device=lm_loss.device)

    total = lm_loss + lambda_pred * l_pred + lambda_entropy * l_entropy

    loss_dict = {
        "loss_total": total.item(),
        "loss_lm": lm_loss.item(),
        "loss_pred": l_pred.item(),
        "loss_entropy": l_entropy.item(),
    }

    return total, loss_dict

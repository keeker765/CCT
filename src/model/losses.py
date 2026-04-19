"""CCT 损失函数 (v2: entropy-driven)

L_total = mean(L_LM_k) + λ_mono · L_mono

- L_LM_k: 每次迭代通过 back+norm+lm_head 的 CrossEntropy
- L_mono: 单调 entropy loss — 惩罚 entropy 在相邻迭代间上升
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def compute_monotonic_entropy_loss(
    entropies: List[torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_mono: 惩罚 entropy 在相邻迭代间上升

    L_mono = Σ_{k=0}^{K-2} mean(ReLU(H_{k+1} - H_k)) / (K-1)

    Args:
        entropies: [H_0, H_1, ..., H_{K-1}], 每个 [B, T]
        valid_mask: [B, T] — 1=有效, 0=padding

    Returns:
        l_mono: 标量
    """
    if len(entropies) < 2:
        return torch.tensor(0.0, device=entropies[0].device)

    eps = 1e-8
    loss = torch.tensor(0.0, device=entropies[0].device)
    for k in range(len(entropies) - 1):
        diff = entropies[k + 1] - entropies[k]  # [B, T]
        penalty = F.relu(diff)
        if valid_mask is not None:
            penalty = (penalty * valid_mask).sum() / (valid_mask.sum() + eps)
        else:
            penalty = penalty.mean()
        loss = loss + penalty

    return loss / (len(entropies) - 1)


def compute_total_loss(
    lm_losses: List[torch.Tensor],
    entropies: List[torch.Tensor],
    lambda_mono: float = 0.1,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算总损失

    L_total = mean(lm_losses) + λ_mono · L_mono

    Args:
        lm_losses: 每次迭代的 LM loss (标量 list)
        entropies: 每次迭代的 per-token entropy [B, T] list
        lambda_mono: L_mono 权重
        valid_mask: [B, T] — padding mask

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失详情
    """
    # 平均所有迭代的 LM loss (确保中间迭代也是有效停止点)
    lm_loss = torch.stack(lm_losses).mean()

    # L_mono
    l_mono = compute_monotonic_entropy_loss(entropies, valid_mask)

    total = lm_loss + lambda_mono * l_mono

    loss_dict = {
        "loss_total": total.item(),
        "loss_lm": lm_loss.item(),
        "loss_mono": l_mono.item(),
    }

    return total, loss_dict

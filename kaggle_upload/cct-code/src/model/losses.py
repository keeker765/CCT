"""CCT 损失函数 (v2: entropy-driven)

L_total = mean(L_LM_k) + λ_mono · scale · L_mono

- L_LM_k: 每次迭代通过 back+norm+lm_head 的 CrossEntropy
- L_mono: entropy 方向损失 — 奖励递减，惩罚递增 (无 ReLU，双向)
- scale = |L_LM|/|L_mono| (自适应, detached) — 使 mono 梯度自动匹配 LM 量级
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
    entropy_floor: float = 0.0,
) -> torch.Tensor:
    """L_mono: 鼓励 entropy 在相邻迭代间递减

    L_mono = Σ_{k=0}^{K-2} masked_mean(H_{k+1} - H_k) / (K-1)

    正值 = entropy 上升 → 惩罚 (增大 total loss)
    负值 = entropy 下降 → 奖励 (减小 total loss)

    entropy_floor: H_norm 低于此值后不再奖励继续降低 (防止 entropy 崩溃)
                   实现: clamp(H, min=floor) 再算 diff

    Args:
        entropies: [H_0, H_1, ..., H_{K-1}], 每个 [B, T]
        valid_mask: [B, T] — 1=有效, 0=padding
        entropy_floor: H_norm 下限 (默认 0.0 = 无下限)

    Returns:
        l_mono: 标量 (可为负)
    """
    if len(entropies) < 2:
        return torch.tensor(0.0, device=entropies[0].device)

    # clamp entropy — 低于 floor 的部分不参与梯度
    if entropy_floor > 0:
        entropies = [e.clamp(min=entropy_floor) for e in entropies]

    eps = 1e-8
    loss = torch.tensor(0.0, device=entropies[0].device)
    for k in range(len(entropies) - 1):
        diff = entropies[k + 1] - entropies[k]  # [B, T]
        if valid_mask is not None:
            step_loss = (diff * valid_mask).sum() / (valid_mask.sum() + eps)
        else:
            step_loss = diff.mean()
        loss = loss + step_loss

    return loss / (len(entropies) - 1)


def compute_total_loss(
    lm_losses: List[torch.Tensor],
    entropies: List[torch.Tensor],
    lambda_mono: float = 0.1,
    valid_mask: Optional[torch.Tensor] = None,
    entropy_floor: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算总损失 (自适应缩放)

    L_total = mean(lm_losses) + λ_mono · (|lm|/|mono|) · L_mono

    scale = |lm_loss.detach()| / |mono.detach()| 使 mono 梯度自动匹配 LM 量级。
    lambda_mono=0.1 → mono 贡献约 10% of LM loss。

    Args:
        lm_losses: 每次迭代的 LM loss (标量 list)
        entropies: 每次迭代的 per-token entropy [B, T] list
        lambda_mono: L_mono 相对权重 (0.1 = 10% of LM)
        valid_mask: [B, T] — padding mask
        entropy_floor: H_norm 下限 (低于此值不再奖励 entropy 降低)

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失详情
    """
    # 平均所有迭代的 LM loss (确保中间迭代也是有效停止点)
    lm_loss = torch.stack(lm_losses).mean()

    # L_mono
    l_mono = compute_monotonic_entropy_loss(entropies, valid_mask, entropy_floor)

    # 自适应缩放: mono 梯度 ≈ lambda_mono × LM 梯度
    mono_abs = l_mono.detach().abs().clamp(min=1e-6)
    scale = lm_loss.detach().abs() / mono_abs

    total = lm_loss + lambda_mono * scale * l_mono

    loss_dict = {
        "loss_total": total.item(),
        "loss_lm": lm_loss.item(),
        "loss_mono": l_mono.item(),
    }

    return total, loss_dict

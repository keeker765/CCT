"""CCT 损失函数 (v2: entropy-driven, pace-constrained)

L_total = mean(L_LM_k, weighted by active mask) + λ_mono · L_mono

- L_LM_k: 每次迭代通过 back+norm+lm_head 的 CrossEntropy (per-sample)
- L_mono (Pace-Constrained): 双向惩罚
  - 惩罚 entropy 上升 (increase_penalty)
  - 惩罚 entropy 下降过快 (rush_penalty, 超过 delta_max)
  - 允许 entropy 在 [0, -delta_max] 范围内下降 → 无惩罚
- per-sample halt: 不同样本在不同迭代停止，只计入 active 迭代的损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


def compute_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """标准 next-token CrossEntropy 损失 (batch 平均)"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def compute_lm_loss_per_sample(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Per-sample next-token CrossEntropy 损失

    Returns:
        per_sample_loss: [B] — 每个样本的平均 CE loss
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    B, T, V = shift_logits.shape

    per_token = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        reduction='none',
    ).reshape(B, T)

    valid = (shift_labels != -100).float()
    per_sample = (per_token * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1)
    return per_sample  # [B]


def compute_monotonic_entropy_loss(
    entropies: List[torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None,
    entropy_floor: float = 0.0,
    iter_active: Optional[List[torch.Tensor]] = None,
    max_iter: int = 10,
    delta_max: float = 0.08,
) -> torch.Tensor:
    """L_mono (Pace-Constrained + Bounded Reward): 控制 entropy 下降节奏

    对每对相邻迭代:
      diff = H[k+1] - H[k]
      increase_penalty = max(0, diff)               # 惩罚 entropy 上升
      rush_penalty     = max(0, -diff - delta_max)   # 惩罚下降过快
      decrease_reward  = clamp(-diff, 0, delta_max)  # 奖励适度下降 (有上界)

    L_mono = mean(increase_penalty + rush_penalty - decrease_reward) / (K-1)

    - 崩塌到 0 (diff=-0.35): rush_penalty=0.27 > reward=0.08 → 净惩罚 +0.19
    - 适度下降 (diff=-0.05): reward=0.05, no penalty → 净奖励 -0.05
    - 不下降 (diff=0): 全部为 0 → 无信号

    Returns:
        l_mono: 标量 (可为负 = 奖励)
    """
    if len(entropies) < 2:
        return torch.tensor(0.0, device=entropies[0].device)

    if entropy_floor > 0:
        entropies = [e.clamp(min=entropy_floor) for e in entropies]

    eps = 1e-8
    loss = torch.tensor(0.0, device=entropies[0].device)
    n_diffs = 0

    for k in range(len(entropies) - 1):
        diff = entropies[k + 1] - entropies[k]  # [B, T]

        increase_penalty = diff.clamp(min=0)              # 上升 → 惩罚
        rush_penalty = (-diff - delta_max).clamp(min=0)   # 下降超过 delta_max → 惩罚
        decrease_reward = (-diff).clamp(min=0, max=delta_max)  # 适度下降 → 奖励 (有上界)

        step_val = increase_penalty + rush_penalty - decrease_reward  # [B, T]

        if iter_active is not None and k + 1 < len(iter_active):
            both = (iter_active[k] & iter_active[k + 1]).float()[:, None]
        else:
            both = 1.0

        if valid_mask is not None:
            mask = valid_mask * both
            denom = mask.sum().clamp(min=eps)
            step_loss = (step_val * mask).sum() / denom
        else:
            step_loss = (step_val * both).sum() / (both.sum() * step_val.size(1)).clamp(min=eps)

        loss = loss + step_loss
        n_diffs += 1

    return loss / max(n_diffs, 1)


def compute_total_loss(
    per_sample_lm_losses: List[torch.Tensor],
    entropies: List[torch.Tensor],
    lambda_mono: float = 0.1,
    valid_mask: Optional[torch.Tensor] = None,
    entropy_floor: float = 0.0,
    iter_active: Optional[List[torch.Tensor]] = None,
    max_iter: int = 10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算总损失 (直接相加, per-sample halt)

    L_total = mean(lm_losses, weighted) + λ_mono · L_mono

    Args:
        per_sample_lm_losses: 每次迭代的 per-sample LM loss, [B] tensors
        entropies: 每次迭代的 per-token entropy [B, T] list
        lambda_mono: L_mono 相对权重 (0.3 = 30% of LM)
        valid_mask: [B, T] — padding mask
        entropy_floor: H_norm 下限
        iter_active: per-iteration per-sample active mask [B] bools
        max_iter: 最大迭代数 (用于 diff clamp 下限)

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失详情
    """
    # Per-sample weighted LM loss: 每个样本只 average 其 active 迭代
    stacked = torch.stack(per_sample_lm_losses)  # [K, B]

    if iter_active is not None:
        active = torch.stack(iter_active[:len(per_sample_lm_losses)]).float()  # [K, B]
        weighted = (stacked * active).sum(dim=0)  # [B]
        counts = active.sum(dim=0).clamp(min=1)   # [B]
        lm_loss = (weighted / counts).mean()       # scalar
    else:
        lm_loss = stacked.mean()

    # L_mono (with per-sample active masking and diff clamp)
    l_mono = compute_monotonic_entropy_loss(
        entropies, valid_mask, entropy_floor, iter_active, max_iter
    )

    # 直接相加 (无自适应缩放)
    total = lm_loss + lambda_mono * l_mono

    loss_dict = {
        "loss_total": total.item(),
        "loss_lm": lm_loss.item(),
        "loss_mono": l_mono.item(),
    }

    return total, loss_dict

"""学习率调度器 — 余弦退火 + 线性预热 + τ_halt 退火"""

import math


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """余弦退火 + 线性预热"""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def compute_halt_tau(
    current_step: int,
    total_steps: int,
    tau_start: float = 1.0,
    tau_end: float = 0.01,
) -> float:
    """线性退火 τ_halt: 从 tau_start → tau_end"""
    progress = min(1.0, current_step / max(1, total_steps))
    return tau_start + (tau_end - tau_start) * progress

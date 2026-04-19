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
    steepness: float = 10.0,
) -> float:
    """DEPRECATED: v1 ACT τ 退火 (保留向后兼容)"""
    progress = min(1.0, current_step / max(1, total_steps))
    raw_start = 1.0 / (1.0 + math.exp(steepness * 0.5))
    raw_end = 1.0 / (1.0 + math.exp(-steepness * 0.5))
    raw = 1.0 / (1.0 + math.exp(-steepness * (progress - 0.5)))
    sigmoid_progress = (raw - raw_start) / (raw_end - raw_start)
    return tau_start + (tau_end - tau_start) * sigmoid_progress


def compute_halt_threshold(
    current_step: int,
    total_steps: int,
    threshold_start: float = 0.8,
    threshold_end: float = 0.3,
) -> float:
    """线性退火 entropy halt 阈值: threshold_start → threshold_end

    训练初期阈值高 (不停), 随训练推进降至目标值 (逐渐允许提前停止)。
    仅影响 eval 模式的推理停止行为, 不影响训练梯度。
    """
    progress = min(1.0, current_step / max(1, total_steps))
    return threshold_start + (threshold_end - threshold_start) * progress

"""Net2WiderNet — MLP 加宽 (保持函数恒等)

两种模式:
1. self: Net2WiderNet — 复制自身神经元 (Chen & Goodfellow, ICLR 2016)
2. cross: Cross-Layer Fusion — 融合其他预训练层的 FFN 知识 (ours)

Cross-Layer Fusion 原理:
  output_new = down_A @ act_A(x) + scale * down_B @ act_B(x)
  其中 A = 原始层, B = donor 层, scale 初始很小 → 训练分化
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


def widen_mlp(
    mlp: nn.Module,
    new_d_ff: int,
    noise_std: float = 0.01,
) -> nn.Module:
    """Net2WiderNet (self mode): 复制自身神经元来加宽

    Args:
        mlp: LlamaMLP 实例 (含 gate_proj, up_proj, down_proj)
        new_d_ff: 目标 intermediate 维度
        noise_std: 噪声强度 (相对于权重 std 的比例)

    Returns:
        mlp: 原地修改后的 MLP (权重加宽)
    """
    old_d_ff = mlp.gate_proj.out_features
    d_model = mlp.gate_proj.in_features

    if new_d_ff <= old_d_ff:
        return mlp

    extra = new_d_ff - old_d_ff

    dup_indices = torch.randint(0, old_d_ff, (extra,))
    mapping = torch.cat([torch.arange(old_d_ff), dup_indices])

    count = torch.zeros(old_d_ff)
    count.scatter_add_(0, mapping, torch.ones(new_d_ff))

    with torch.no_grad():
        orig_dtype = mlp.gate_proj.weight.dtype
        gate_w = mlp.gate_proj.weight.data.float()
        up_w = mlp.up_proj.weight.data.float()
        down_w = mlp.down_proj.weight.data.float()

        new_gate_w = gate_w[mapping]
        new_up_w = up_w[mapping]

        down_divided = down_w / count.unsqueeze(0)
        new_down_w = down_divided[:, mapping]

        dup_mask = torch.zeros(new_d_ff, dtype=torch.bool)
        dup_mask[old_d_ff:] = True

        gate_noise_scale = noise_std * gate_w.std()
        up_noise_scale = noise_std * up_w.std()
        down_noise_scale = noise_std * down_w.std()

        new_gate_w[dup_mask] += gate_noise_scale * torch.randn(extra, d_model)
        new_up_w[dup_mask] += up_noise_scale * torch.randn(extra, d_model)
        new_down_w[:, dup_mask] += down_noise_scale * torch.randn(d_model, extra)

        mlp.gate_proj = nn.Linear(d_model, new_d_ff, bias=False)
        mlp.up_proj = nn.Linear(d_model, new_d_ff, bias=False)
        mlp.down_proj = nn.Linear(new_d_ff, d_model, bias=False)

        mlp.gate_proj.weight.data = new_gate_w.to(orig_dtype)
        mlp.up_proj.weight.data = new_up_w.to(orig_dtype)
        mlp.down_proj.weight.data = new_down_w.to(orig_dtype)

    return mlp


def widen_mlp_cross_layer(
    target_mlp: nn.Module,
    donor_mlp: nn.Module,
    donor_init_scale: float = 0.1,
) -> nn.Module:
    """Cross-Layer Fusion: 用 donor 层的 FFN 知识来加宽 target 层

    concat target and donor neurons:
      gate_new = [gate_target; gate_donor]
      up_new   = [up_target;   up_donor]
      down_new = [down_target;  scale * down_donor]

    scale < 1 保证初始行为接近原始 (donor 贡献小),
    训练中 donor 逐渐增强贡献。

    Args:
        target_mlp: 目标 column 层的 MLP (将被原地修改)
        donor_mlp: donor 层的 MLP (只读取权重)
        donor_init_scale: donor 的 down_proj 初始缩放 (推荐 0.01-0.1)

    Returns:
        target_mlp: 原地修改后的 MLP (d_ff 翻倍)
    """
    d_model = target_mlp.gate_proj.in_features
    old_d_ff = target_mlp.gate_proj.out_features
    donor_d_ff = donor_mlp.gate_proj.out_features

    new_d_ff = old_d_ff + donor_d_ff

    with torch.no_grad():
        orig_dtype = target_mlp.gate_proj.weight.dtype

        # 取出权重 (fp32)
        t_gate = target_mlp.gate_proj.weight.data.float()
        t_up = target_mlp.up_proj.weight.data.float()
        t_down = target_mlp.down_proj.weight.data.float()

        d_gate = donor_mlp.gate_proj.weight.data.float()
        d_up = donor_mlp.up_proj.weight.data.float()
        d_down = donor_mlp.down_proj.weight.data.float()

        # 拼接: [target neurons; donor neurons]
        new_gate_w = torch.cat([t_gate, d_gate], dim=0)  # [new_d_ff, d_model]
        new_up_w = torch.cat([t_up, d_up], dim=0)
        new_down_w = torch.cat(
            [t_down, donor_init_scale * d_down], dim=1  # [d_model, new_d_ff]
        )

        # 替换 Linear 层
        target_mlp.gate_proj = nn.Linear(d_model, new_d_ff, bias=False)
        target_mlp.up_proj = nn.Linear(d_model, new_d_ff, bias=False)
        target_mlp.down_proj = nn.Linear(new_d_ff, d_model, bias=False)

        target_mlp.gate_proj.weight.data = new_gate_w.to(orig_dtype)
        target_mlp.up_proj.weight.data = new_up_w.to(orig_dtype)
        target_mlp.down_proj.weight.data = new_down_w.to(orig_dtype)

    return target_mlp


def auto_donor_mapping(
    column_layers: list,
    num_base_layers: int,
    front_layers: list,
    back_layers: list,
) -> Dict[int, int]:
    """自动计算 column → donor 映射 (就近原则)

    每个 column 层选最近的未使用层作为 donor。

    Returns:
        mapping: {column_layer_idx: donor_layer_idx}
    """
    used = set(front_layers + column_layers + back_layers)
    unused = sorted(set(range(num_base_layers)) - used)

    mapping = {}
    remaining = list(unused)
    for col_idx in column_layers:
        if not remaining:
            break
        # 选最近的未使用层
        nearest = min(remaining, key=lambda u: abs(u - col_idx))
        mapping[col_idx] = nearest
        remaining.remove(nearest)

    return mapping

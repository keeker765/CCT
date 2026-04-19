"""Net2WiderNet — MLP 加宽 (保持函数恒等)

将 LlamaMLP 的 intermediate_size (d_ff) 扩展到 new_d_ff:
1. gate_proj / up_proj: 复制随机选择的行 (神经元)
2. down_proj: 复制对应列, 除以 count (保持输出不变)
3. 对复制品加小噪声 → 训练时分化

参考: Chen & Goodfellow, "Net2Net: Accelerating Learning via
Knowledge Transfer", ICLR 2016
"""

import torch
import torch.nn as nn


def widen_mlp(
    mlp: nn.Module,
    new_d_ff: int,
    noise_std: float = 0.01,
) -> nn.Module:
    """Net2WiderNet: 将 MLP 的 intermediate 维度从 old_d_ff 扩展到 new_d_ff

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

    # 建立映射: 前 old_d_ff 保持原位, 后 extra 随机复制
    dup_indices = torch.randint(0, old_d_ff, (extra,))
    mapping = torch.cat([torch.arange(old_d_ff), dup_indices])  # [new_d_ff]

    # 计算每个原始神经元的出现次数 (用于 down_proj 除 count)
    count = torch.zeros(old_d_ff)
    count.scatter_add_(0, mapping, torch.ones(new_d_ff))

    with torch.no_grad():
        orig_dtype = mlp.gate_proj.weight.dtype
        # 取出原始权重 (fp32 计算)
        gate_w = mlp.gate_proj.weight.data.float()  # [old_d_ff, d_model]
        up_w = mlp.up_proj.weight.data.float()
        down_w = mlp.down_proj.weight.data.float()  # [d_model, old_d_ff]

        # 扩展 gate / up: 按 mapping 取行
        new_gate_w = gate_w[mapping]  # [new_d_ff, d_model]
        new_up_w = up_w[mapping]

        # 扩展 down: 按 mapping 取列, 除以 count
        down_divided = down_w / count.unsqueeze(0)  # [d_model, old_d_ff]
        new_down_w = down_divided[:, mapping]  # [d_model, new_d_ff]

        # 对复制的神经元加噪声 (仅 index >= old_d_ff)
        dup_mask = torch.zeros(new_d_ff, dtype=torch.bool)
        dup_mask[old_d_ff:] = True

        gate_noise_scale = noise_std * gate_w.std()
        up_noise_scale = noise_std * up_w.std()
        down_noise_scale = noise_std * down_w.std()

        new_gate_w[dup_mask] += gate_noise_scale * torch.randn(extra, d_model)
        new_up_w[dup_mask] += up_noise_scale * torch.randn(extra, d_model)
        new_down_w[:, dup_mask] += down_noise_scale * torch.randn(d_model, extra)

        # 替换 Linear 层
        mlp.gate_proj = nn.Linear(d_model, new_d_ff, bias=False)
        mlp.up_proj = nn.Linear(d_model, new_d_ff, bias=False)
        mlp.down_proj = nn.Linear(new_d_ff, d_model, bias=False)

        mlp.gate_proj.weight.data = new_gate_w.to(orig_dtype)
        mlp.up_proj.weight.data = new_up_w.to(orig_dtype)
        mlp.down_proj.weight.data = new_down_w.to(orig_dtype)

    return mlp


def verify_widen(
    mlp_old: nn.Module,
    mlp_new: nn.Module,
    d_model: int,
    atol: float = 1e-4,
) -> bool:
    """验证加宽前后的输出是否一致 (用于测试)"""
    x = torch.randn(1, 4, d_model)
    with torch.no_grad():
        y_old = mlp_old(x)
        y_new = mlp_new(x)
    diff = (y_old - y_new).abs().max().item()
    return diff < atol

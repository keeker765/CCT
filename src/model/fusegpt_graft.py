"""FuseGPT-style 可学习低秩融合 — 吸收被删层知识到 Column 层

核心公式: W_effective = W_col + (A @ B^T) ⊙ W_pruned
  - A, B: 可训练低秩矩阵 (rank r << min(d_out, d_in))
  - W_pruned: 被删层的权重 (frozen, non-persistent buffer)
  - A 初始化为 0 → 初始融合贡献为零, 逐步学习吸收

训练完成后调用 fold_all_fusions() 将融合折叠进基础权重 → 零推理开销

Checkpoint 约定:
  - 训练中保存: A, B 在 state_dict 中, W_pruned 不在 (persistent=False)
  - 加载训练中 checkpoint: 需从 base_model 重建 W_pruned (相同构造流程)
  - 训练后部署: 先 fold, 再保存 → 普通 Linear, 无需 base_model

Reference: FuseGPT (arXiv:2411.14507, ICML)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class FusionLinear(nn.Module):
    """Linear + FuseGPT 可学习低秩融合

    Forward: base_out + ((A @ B^T) ⊙ W_pruned) @ x
    Fold:    W_base += (A @ B^T) * W_pruned → 恢复普通 Linear
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        pruned_weight: torch.Tensor,
        rank: int,
    ):
        super().__init__()
        self.linear = base_linear
        d_out, d_in = pruned_weight.shape

        # Frozen pruned weight (不保存到 state_dict → reload 时从 base_model 重建)
        self.register_buffer("W_pruned", pruned_weight, persistent=False)

        # Learnable fusion matrices (与模型同 dtype)
        dtype = pruned_weight.dtype
        inv_sqrt_r = 1.0 / math.sqrt(rank)
        self.A = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.B = nn.Parameter(
            torch.empty(d_in, rank, dtype=dtype).uniform_(-inv_sqrt_r, inv_sqrt_r)
        )
        self._folded = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        if self._folded:
            return base_out
        C = (self.A @ self.B.T) * self.W_pruned  # (d_out, d_in)
        return base_out + F.linear(x, C)

    def fold(self):
        """折叠融合到基础权重, 之后等价于普通 Linear"""
        if self._folded:
            return
        with torch.no_grad():
            C = (self.A @ self.B.T) * self.W_pruned
            self.linear.weight.data += C.to(self.linear.weight.dtype)
        del self.A, self.B, self.W_pruned
        self._folded = True

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


# ── 吸收映射 ──────────────────────────────────────────────

def build_multi_absorb_map(
    column_layers: List[int],
    num_base_layers: int,
    front_layers: List[int],
    back_layers: List[int],
) -> Dict[int, List[int]]:
    """计算多 donor 吸收映射

    每个 column 层吸收其与下一个 used 层之间的所有未使用层.
    Example:
      used = [0,1, 2,7,12, 14,15]
      column[2]  → [3,4,5,6]
      column[7]  → [8,9,10,11]
      column[12] → [13]
    """
    used = set(front_layers + column_layers + back_layers)
    all_used_sorted = sorted(used)

    absorb_map: Dict[int, List[int]] = {}
    for col_idx in column_layers:
        next_used = num_base_layers
        for u in all_used_sorted:
            if u > col_idx:
                next_used = u
                break
        donors = [i for i in range(col_idx + 1, next_used) if i not in used]
        if donors:
            absorb_map[col_idx] = donors

    return absorb_map


# ── 投影层路径 ────────────────────────────────────────────

_ATTN_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")
_ALL_PROJ_NAMES = _ATTN_PROJ_NAMES + _MLP_PROJ_NAMES


def _get_linear(layer: nn.Module, proj_name: str) -> nn.Linear:
    if proj_name in _ATTN_PROJ_NAMES:
        return getattr(layer.self_attn, proj_name)
    return getattr(layer.mlp, proj_name)


def _set_linear(layer: nn.Module, proj_name: str, module: nn.Module):
    if proj_name in _ATTN_PROJ_NAMES:
        setattr(layer.self_attn, proj_name, module)
    else:
        setattr(layer.mlp, proj_name, module)


# ── 核心 API ─────────────────────────────────────────────

def attach_fusion_grafts(
    cct_layer: nn.Module,
    donor_layers: List[nn.Module],
    rank: int,
    pool_donors: bool = True,
    freeze_base: bool = False,
) -> None:
    """为 CCTDecoderLayer 的每个 Linear 添加 FusionLinear 包装

    Args:
        cct_layer: 目标 CCTDecoderLayer
        donor_layers: 被删除的 LlamaDecoderLayer 列表
        rank: 低秩融合的秩
        pool_donors: True=先平均多个 donor 权重
        freeze_base: True=冻结基础权重, 仅训练 A/B
    """
    for proj_name in _ALL_PROJ_NAMES:
        base_linear = _get_linear(cct_layer, proj_name)

        donor_weights = []
        for donor in donor_layers:
            w = _get_linear(donor, proj_name).weight.data.clone()
            donor_weights.append(w)

        pooled = torch.stack(donor_weights).mean(dim=0)
        fusion = FusionLinear(base_linear, pooled, rank)

        if freeze_base:
            fusion.linear.weight.requires_grad = False

        _set_linear(cct_layer, proj_name, fusion)


def fold_all_fusions(model: nn.Module) -> None:
    """折叠所有 FusionLinear, 恢复为普通 Linear (零推理开销)"""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, FusionLinear):
                child.fold()
                setattr(module, name, child.linear)


def get_fusion_params(model: nn.Module) -> List[nn.Parameter]:
    """收集所有融合参数 (A, B), 用于 optimizer param group"""
    params = []
    for module in model.modules():
        if isinstance(module, FusionLinear) and not module._folded:
            params.extend([module.A, module.B])
    return params


def get_fusion_param_count(model: nn.Module) -> int:
    """统计融合参数总量"""
    count = 0
    for module in model.modules():
        if isinstance(module, FusionLinear) and not module._folded:
            count += module.A.numel() + module.B.numel()
    return count


def get_fusion_buffer_count(model: nn.Module) -> int:
    """统计融合 buffer (W_pruned) 总量"""
    count = 0
    for module in model.modules():
        if isinstance(module, FusionLinear) and not module._folded:
            count += module.W_pruned.numel()
    return count

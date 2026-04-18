"""CCTAttention — 改造自 LlamaAttention, 内置循环嵌入 + precision bias

复制 LlamaAttention 源码, 新增:
1. 旋转循环嵌入 (CycleEmbed): Q/K 投影前施加 → 保留循环信息
2. Precision bias: attention logits 加入 precision 调制

每层双重旋转操作:
  CycleEmbed(φ=1.618, 投影前) + 标准 RoPE(投影后)
"""

import math
from typing import Optional, Tuple
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
    LlamaConfig,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .cycle_embedding import RotaryCycleEmbedding


class CCTAttention(nn.Module):
    """CCT Attention — 在 LlamaAttention 基础上新增循环嵌入和 precision bias"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        cycle_embedding: RotaryCycleEmbedding,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.cycle_embedding = cycle_embedding

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cycle_k: int = 0,
        precision_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # === 修改点 1: Q/K 投影前施加旋转循环嵌入 ===
        if cycle_k > 0:
            hidden_states = self.cycle_embedding(hidden_states, cycle_k)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 标准 RoPE (位置编码) — 在 Q/K 投影后施加
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # KV cache
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # GQA: repeat KV heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) * self.scaling

        # === 修改点 2: precision 增益调制 (query 侧乘性) ===
        if precision_bias is not None:
            # precision_bias: [batch, seq_len] → gain: [batch, 1, seq_len, 1]
            # Query 侧乘性增益: 预测差的 token 注意力更锐利 (per-token temperature)
            # 数学等价: cross-attention(Q'=W_Q·diag(g)·H, K=W_K·H, V=W_V·H)
            # 其中 g_i = 1 + λ·precision_i, 有效温度 τ_i = √d / g_i
            gain = 1.0 + precision_bias.unsqueeze(1).unsqueeze(-1)  # [B, 1, S_q, 1]
            attn_weights = attn_weights * gain

        # Causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax + dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(
                attn_weights, p=self.attention_dropout, training=True
            )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

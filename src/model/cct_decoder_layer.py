"""CCTDecoderLayer — 改造自 LlamaDecoderLayer, 使用 CCTAttention

复制 LlamaDecoderLayer 源码, 替换 LlamaAttention → CCTAttention。
FFN / RMSNorm / 残差连接保持不变。
支持 entropy_temperature 透传给 CCTAttention。
"""

from typing import Optional, Tuple

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaMLP,
    LlamaRMSNorm,
)

from .cct_attention import CCTAttention
from .cycle_embedding import RotaryCycleEmbedding


class CCTDecoderLayer(nn.Module):
    """CCT Decoder Layer — 与 LlamaDecoderLayer 结构相同, 使用 CCTAttention"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        cycle_embedding: RotaryCycleEmbedding,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CCTAttention(
            config=config,
            layer_idx=layer_idx,
            cycle_embedding=cycle_embedding,
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cycle_k: int = 0,
        precision_bias: Optional[torch.Tensor] = None,  # DEPRECATED
        entropy_temperature: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self Attention (pre-norm)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cycle_k=cycle_k,
            precision_bias=precision_bias,
            entropy_temperature=entropy_temperature,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # FFN (pre-norm)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

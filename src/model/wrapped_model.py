"""CCTLlamaModel — 皮层柱 Transformer 主模型

架构: Fixed Front (2层) → Column (3层 × K循环) → Fixed Back (2层)
预训练层映射: Layer 0,1 → Front; Layer 2,3,4 → Column; Layer 14,15 → Back

Per-token ACT + 温度退火:
- 训练: 所有轮都跑, 输出 = per-token 加权和 (remainder × p_halt × h)
- 推理: 同机制, remainder < threshold 时早停; inference_temperature 控制推理强度
- τ_halt 从 1.0 退火到 0.01 → 训练/推理一致
"""

import inspect
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple

from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
)

try:
    from transformers.models.llama.modeling_llama import create_causal_mask
    _MASK_PARAMS = set(inspect.signature(create_causal_mask).parameters.keys())
except ImportError:
    create_causal_mask = None
    _MASK_PARAMS = set()

from .column_config import CCTConfig
from .cct_decoder_layer import CCTDecoderLayer
from .cycle_embedding import RotaryCycleEmbedding
from .predictor import CCTPredictor
from .l6_precision import L6Precision
from .halt_head import HaltHead
from .losses import compute_lm_loss, compute_total_loss


class CCTLlamaModel(nn.Module):
    """CCT 包装的 Llama 模型"""

    def __init__(
        self,
        base_model: LlamaForCausalLM,
        config: CCTConfig,
    ):
        super().__init__()
        self.config = config
        self.base_config = base_model.config

        # === 提取基座组件 ===
        self.embed_tokens = base_model.model.embed_tokens
        self.final_norm = base_model.model.norm
        self.lm_head = base_model.lm_head
        self.rotary_emb = base_model.model.rotary_emb

        # === 构建 Fixed Front 层 (标准 LlamaDecoderLayer) ===
        self.front_layers = nn.ModuleList()
        for src_idx in config.pretrained_front_layers:
            self.front_layers.append(base_model.model.layers[src_idx])

        # === 构建 Column 层 (CCTDecoderLayer, 共享权重循环) ===
        self.cycle_embedding = RotaryCycleEmbedding(
            d_model=config.d_model, phi=config.phi
        )

        self.column_layers = nn.ModuleList()
        for i, src_idx in enumerate(config.pretrained_column_layers):
            src_layer = base_model.model.layers[src_idx]
            cct_layer = CCTDecoderLayer(
                config=self.base_config,
                layer_idx=config.num_front_layers + i,
                cycle_embedding=self.cycle_embedding,
            )
            # 复制权重
            cct_layer.self_attn.q_proj.load_state_dict(
                src_layer.self_attn.q_proj.state_dict()
            )
            cct_layer.self_attn.k_proj.load_state_dict(
                src_layer.self_attn.k_proj.state_dict()
            )
            cct_layer.self_attn.v_proj.load_state_dict(
                src_layer.self_attn.v_proj.state_dict()
            )
            cct_layer.self_attn.o_proj.load_state_dict(
                src_layer.self_attn.o_proj.state_dict()
            )
            cct_layer.mlp.load_state_dict(src_layer.mlp.state_dict())
            cct_layer.input_layernorm.load_state_dict(
                src_layer.input_layernorm.state_dict()
            )
            cct_layer.post_attention_layernorm.load_state_dict(
                src_layer.post_attention_layernorm.state_dict()
            )
            self.column_layers.append(cct_layer)

        # === 构建 Fixed Back 层 (标准 LlamaDecoderLayer) ===
        self.back_layers = nn.ModuleList()
        for src_idx in config.pretrained_back_layers:
            self.back_layers.append(base_model.model.layers[src_idx])

        # === CCT 新增模块 ===
        self.cct_predictor = CCTPredictor(config.d_model, config.info_dim)
        self.l6_precision = L6Precision(
            lambda_init=config.lambda_precision_init,
            temperature=config.precision_temperature,
        )
        self.halt_head = HaltHead(config.d_model)

        # 迭代间 RMSNorm (防发散)
        self.inter_iter_norm = LlamaRMSNorm(
            config.d_model, eps=config.rms_norm_eps
        )

        # 退火温度 (由外部 scheduler 更新)
        self.register_buffer(
            "halt_tau", torch.tensor(config.halt_tau_start)
        )

        self._gradient_checkpointing = False

        # 记录模型 dtype (用于 forward 中的 dtype 统一)
        self._model_dtype = self.embed_tokens.weight.dtype

        # 将新模块转换为与基座相同的 dtype
        base_dtype = self._model_dtype
        self.cct_predictor.to(base_dtype)
        self.l6_precision.to(base_dtype)
        self.halt_head.to(base_dtype)
        self.inter_iter_norm.to(base_dtype)

        # 释放原模型的未使用层以节省显存
        del base_model

    def _build_causal_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values=None,
        cache_position=None,
    ) -> Optional[torch.Tensor]:
        """构建因果注意力掩码"""
        if attention_mask is None or create_causal_mask is None:
            return None

        mask_kwargs = {}
        if "config" in _MASK_PARAMS:
            mask_kwargs["config"] = self.base_config
        if "inputs_embeds" in _MASK_PARAMS:
            mask_kwargs["inputs_embeds"] = hidden_states
        if "input_tensor" in _MASK_PARAMS:
            mask_kwargs["input_tensor"] = hidden_states
        if "attention_mask" in _MASK_PARAMS:
            mask_kwargs["attention_mask"] = attention_mask
        if "past_key_values" in _MASK_PARAMS:
            mask_kwargs["past_key_values"] = past_key_values
        if "position_ids" in _MASK_PARAMS:
            mask_kwargs["position_ids"] = position_ids
        if "cache_position" in _MASK_PARAMS:
            mask_kwargs["cache_position"] = cache_position

        try:
            return create_causal_mask(**mask_kwargs)
        except Exception:
            return None

    def _run_standard_layer(
        self, layer: nn.Module, hidden_states: torch.Tensor, layer_kwargs: dict
    ) -> torch.Tensor:
        """运行标准 LlamaDecoderLayer"""
        hidden_states = hidden_states.to(self._model_dtype)
        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                layer, hidden_states, **layer_kwargs, use_reentrant=False
            )
        return layer(hidden_states, **layer_kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        # === 1. Embedding ===
        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = torch.arange(
                    seq_len, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        causal_mask = self._build_causal_mask(
            hidden_states, attention_mask, position_ids
        )

        layer_kwargs = dict(
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        if causal_mask is not None:
            layer_kwargs["attention_mask"] = causal_mask

        # === 2. Fixed Front ===
        for layer in self.front_layers:
            hidden_states = self._run_standard_layer(layer, hidden_states, layer_kwargs)

        # === 3. Column 循环 (ACT 软停止) ===
        x_column = hidden_states  # 保存 Column 输入
        h = hidden_states

        tau_halt = self.halt_tau.item()
        pred_losses: List[torch.Tensor] = []
        p_halts: List[torch.Tensor] = []
        remainders_list: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        valid_mask = None  # per-token halting 的 padding mask

        if self.training:
            # 训练: per-token ACT 加权和输出
            seq_len = h.size(1)
            remainder = torch.ones(batch_size, seq_len, device=h.device)
            output = torch.zeros_like(h)

            # Padding token 不参与 halting
            if attention_mask is not None:
                valid_mask = attention_mask[:, :seq_len].float()
                remainder = remainder * valid_mask
            else:
                valid_mask = None

            for k in range(self.config.max_iter):
                # a. 3个 CCTDecoderLayer forward (共享权重)
                precision_bias = None
                if k > 0 and all_scores:
                    precision_bias = self.l6_precision(all_scores[-1])

                for col_layer in self.column_layers:
                    if self._gradient_checkpointing:
                        h = torch.utils.checkpoint.checkpoint(
                            col_layer, h.to(self._model_dtype),
                            layer_kwargs.get("attention_mask"),
                            position_ids,
                            None,  # past_key_values
                            position_embeddings,
                            k,  # cycle_k
                            precision_bias,
                            use_reentrant=False,
                        )
                    else:
                        h = col_layer(
                            h.to(self._model_dtype),
                            attention_mask=layer_kwargs.get("attention_mask"),
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            cycle_k=k,
                            precision_bias=precision_bias,
                        )

                # b. 迭代间 RMSNorm (防发散)
                h = self.inter_iter_norm(h)

                # c. HaltHead
                p_halt = self.halt_head(h, tau_halt)
                p_halts.append(p_halt)
                remainders_list.append(remainder.clone())

                # d. ACT per-token 加权累积
                weight = (remainder * p_halt).unsqueeze(-1)  # [batch, seq_len, 1]
                output = output + weight * h

                # e. 更新 per-token remainder
                remainder = remainder * (1.0 - p_halt)

                # f. Score (双重 detach, 共享 info_proj)
                score = self.cct_predictor.compute_score(h, x_column)
                all_scores.append(score)

                # g. L_pred (双重 detach 防崩塌)
                l_pred_k = self.cct_predictor.compute_pred_loss(h, x_column)
                pred_losses.append(l_pred_k)

                # h. 提前退出优化 (remainder 极小时无意义继续)
                if remainder.max().item() < 1e-4:
                    break

            # 分配剩余 remainder 给最后一轮
            output = output + remainder.unsqueeze(-1) * h
            hidden_states = output

        else:
            # 推理: per-token ACT + inference_temperature 控制推理强度
            seq_len = h.size(1)
            remainder = torch.ones(batch_size, seq_len, device=h.device)
            output = torch.zeros_like(h)

            if attention_mask is not None:
                valid_mask = attention_mask[:, :seq_len].float()
                remainder = remainder * valid_mask
            else:
                valid_mask = None

            # 推理温度: >1 → 更多迭代 (尤其 hard token); <1 → 更少迭代
            inf_tau = tau_halt * self.config.inference_temperature

            for k in range(self.config.max_iter):
                precision_bias = None
                if k > 0 and all_scores:
                    precision_bias = self.l6_precision(all_scores[-1])

                for col_layer in self.column_layers:
                    h = col_layer(
                        h.to(self._model_dtype),
                        attention_mask=layer_kwargs.get("attention_mask"),
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        cycle_k=k,
                        precision_bias=precision_bias,
                    )

                h = self.inter_iter_norm(h)

                p_halt = self.halt_head(h, inf_tau)

                # Per-token ACT 累积
                weight = (remainder * p_halt).unsqueeze(-1)
                output = output + weight * h
                remainder = remainder * (1.0 - p_halt)

                # Score for precision
                score = self.cct_predictor.compute_score(h, x_column)
                all_scores.append(score)

                # 所有有效 token 的 remainder 都足够小时停止
                if k >= self.config.min_iter - 1 and remainder.max().item() < 1e-3:
                    break

            # 分配剩余 remainder
            output = output + remainder.unsqueeze(-1) * h
            hidden_states = output

        # === 4. Fixed Back ===
        for layer in self.back_layers:
            hidden_states = self._run_standard_layer(layer, hidden_states, layer_kwargs)

        # === 5. Final Norm + LM Head ===
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # === 6. 计算损失 ===
        loss = None
        loss_dict = {}

        if labels is not None:
            lm_loss = compute_lm_loss(logits, labels)
            loss, loss_dict = compute_total_loss(
                lm_loss=lm_loss,
                pred_losses=pred_losses,
                p_halts=p_halts,
                remainders=remainders_list,
                lambda_pred=self.config.lambda_pred,
                lambda_entropy=self.config.lambda_entropy,
                valid_mask=valid_mask,
            )

        return {
            "loss": loss,
            "logits": logits,
            "loss_dict": loss_dict,
            "p_halts": p_halts,
            "scores": all_scores,
            "num_iterations": len(all_scores),
        }

    def set_halt_tau(self, tau: float):
        """更新 HaltHead 退火温度"""
        self.halt_tau.fill_(tau)

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def get_param_groups(self) -> List[dict]:
        """返回分层学习率参数组"""
        new_modules = [
            self.cct_predictor, self.l6_precision,
            self.halt_head, self.inter_iter_norm,
        ]
        new_params = set()
        for m in new_modules:
            new_params.update(id(p) for p in m.parameters())

        base_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in new_params
        ]
        new_params_list = [
            p for p in self.parameters()
            if p.requires_grad and id(p) in new_params
        ]

        return [
            {"params": base_params, "lr": self.config.learning_rate},
            {"params": new_params_list, "lr": self.config.new_module_lr},
        ]

    def get_trainable_params_info(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cct_modules = [
            self.cct_predictor, self.l6_precision,
            self.halt_head, self.inter_iter_norm,
        ]
        cct_params = sum(
            p.numel() for m in cct_modules for p in m.parameters()
        )
        return (
            f"Total params: {total:,}\n"
            f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)\n"
            f"CCT new params: {cct_params:,} ({100*cct_params/total:.2f}%)"
        )

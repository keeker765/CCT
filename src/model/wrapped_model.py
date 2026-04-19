"""CCTLlamaModel — 皮层柱 Transformer 主模型

架构: Fixed Front (2层) → Column (3层 × K循环) → Fixed Back (2层)
预训练层映射: Layer 0,1 → Front; Layer 2,3,4 → Column; Layer 14,15 → Back

Per-token ACT + L6 驱动停止:
- 训练: 所有轮都跑, 输出 = per-token 加权和 (remainder × p_halt × h)
- 推理: 同机制, remainder < threshold 时早停; inference_temperature 控制推理强度
- L6 同时输出 attention_bias (注意力调制) 和 p_halt (停止决策)
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
from .losses import compute_lm_loss, compute_total_loss
from .net2wider import widen_mlp, widen_mlp_cross_layer, auto_donor_mapping
from .fusegpt_graft import (
    FusionLinear,
    build_multi_absorb_map,
    attach_fusion_grafts,
    fold_all_fusions,
    get_fusion_params,
    get_fusion_param_count,
    get_fusion_buffer_count,
)


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

        # === MLP 加宽 (在 column 层构建完成后) ===
        if config.use_ffn_expansion and config.column_d_ff > config.d_ff:
            if config.widen_mode == "cross":
                donor_map = auto_donor_mapping(
                    config.pretrained_column_layers,
                    config.num_base_layers,
                    config.pretrained_front_layers,
                    config.pretrained_back_layers,
                )
                for i, src_idx in enumerate(config.pretrained_column_layers):
                    if src_idx in donor_map:
                        donor_layer = base_model.model.layers[donor_map[src_idx]]
                        widen_mlp_cross_layer(
                            self.column_layers[i].mlp,
                            donor_layer.mlp,
                            donor_init_scale=config.donor_init_scale,
                        )
                        print(f"  Column[{src_idx}] ← Donor[{donor_map[src_idx]}] "
                              f"(cross-layer, scale={config.donor_init_scale})")
            else:
                for i, cct_layer in enumerate(self.column_layers):
                    widen_mlp(cct_layer.mlp, config.column_d_ff,
                              noise_std=config.widen_noise_std)
                    print(f"  Column[{config.pretrained_column_layers[i]}] "
                          f"widened to d_ff={config.column_d_ff} (self)")

        # === FuseGPT-style 在线融合 (吸收被删层知识) ===
        if config.use_fusion_graft:
            absorb_map = build_multi_absorb_map(
                config.pretrained_column_layers,
                config.num_base_layers,
                config.pretrained_front_layers,
                config.pretrained_back_layers,
            )
            for i, src_idx in enumerate(config.pretrained_column_layers):
                if src_idx in absorb_map:
                    donor_layers = [
                        base_model.model.layers[j]
                        for j in absorb_map[src_idx]
                    ]
                    attach_fusion_grafts(
                        self.column_layers[i],
                        donor_layers,
                        rank=config.fusion_rank,
                        pool_donors=config.fusion_pool_donors,
                        freeze_base=config.fusion_freeze_base,
                    )
                    print(
                        f"  Column[{src_idx}] ← Fusion{absorb_map[src_idx]} "
                        f"(rank={config.fusion_rank}, "
                        f"pool={config.fusion_pool_donors})"
                    )

        # === 构建 Fixed Back 层 (标准 LlamaDecoderLayer) ===
        self.back_layers = nn.ModuleList()
        for src_idx in config.pretrained_back_layers:
            self.back_layers.append(base_model.model.layers[src_idx])

        # === CCT 新增模块 ===
        self.cct_predictor = CCTPredictor(
            config.d_model, config.info_dim, config.delta_noise_scale
        )
        self.l6_precision = L6Precision(
            lambda_init=config.lambda_precision_init,
            temperature=config.precision_temperature,
        )

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
        """构建因果注意力掩码

        优先使用 transformers 内置 create_causal_mask；
        若返回 None（transformers 5.x 在 _attn_implementation 未设置时会跳过），
        则手动构建标准上三角因果掩码，确保 CCTAttention 等手动注意力模块
        不会泄露未来 token 信息。
        """
        causal_mask = None

        # 尝试 transformers 内置方法
        if attention_mask is not None and create_causal_mask is not None:
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
                causal_mask = create_causal_mask(**mask_kwargs)
            except Exception:
                causal_mask = None

        # 回退：手动构建因果掩码（上三角 -inf）
        if causal_mask is None:
            seq_len = hidden_states.shape[1]
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    torch.finfo(hidden_states.dtype).min,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                diagonal=1,
            )
            # 扩展到 4D: [1, 1, seq_len, seq_len]，兼容 multi-head attention
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            # 如果有 2D padding mask，将 padding 位置也屏蔽
            if attention_mask is not None and attention_mask.ndim == 2:
                # attention_mask: [batch, seq_len], 1=有效, 0=padding
                padding_mask = attention_mask[:, None, None, :]  # [B, 1, 1, S]
                padding_mask = (1.0 - padding_mask.to(causal_mask.dtype)) * torch.finfo(hidden_states.dtype).min
                causal_mask = causal_mask + padding_mask

        return causal_mask

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

        # Column 层 SDPA 优化: 无 padding 时传 None mask → is_causal=True (Flash Attention)
        has_padding = attention_mask is not None and (attention_mask == 0).any()
        column_mask = layer_kwargs.get("attention_mask") if has_padding else None

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
                # a. 保存列运算前的状态 (用于前向预测)
                h_before = h

                # b. L6 注意力调制 (用 score[k-1], 仅 k>0)
                precision_bias = None
                if k > 0 and all_scores:
                    precision_bias = self.l6_precision.compute_attention_bias(
                        all_scores[-1]
                    )

                # c. 3个 CCTDecoderLayer forward (共享权重)
                for ci, col_layer in enumerate(self.column_layers):
                    col_cycle_k = k * len(self.column_layers) + ci
                    col_precision = precision_bias if ci == 0 else None
                    if self._gradient_checkpointing:
                        h = torch.utils.checkpoint.checkpoint(
                            col_layer, h.to(self._model_dtype),
                            column_mask,
                            position_ids,
                            None,  # past_key_values
                            position_embeddings,
                            col_cycle_k,  # cycle_k
                            col_precision,
                            use_reentrant=False,
                        )
                    else:
                        h = col_layer(
                            h.to(self._model_dtype),
                            attention_mask=column_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            cycle_k=col_cycle_k,
                            precision_bias=col_precision,
                        )

                # d. 迭代间 RMSNorm (防发散)
                h = self.inter_iter_norm(h)

                # e. Score: 当前迭代的预测误差
                score = self.cct_predictor.compute_score(h_before, h)
                all_scores.append(score)

                # f. L6 停止决策 (用 score[k], 当前迭代)
                p_halt = self.l6_precision.compute_halt(score, tau_halt)
                p_halts.append(p_halt)
                remainders_list.append(remainder.clone())

                # g. ACT per-token 加权累积
                weight = (remainder * p_halt).unsqueeze(-1)  # [batch, seq_len, 1]
                output = output + weight * h

                # h. 更新 per-token remainder
                remainder = remainder * (1.0 - p_halt)

                # i. 前向预测 L_pred: 列变换可预测性
                l_pred_k = self.cct_predictor.compute_pred_loss(h_before, h)
                pred_losses.append(l_pred_k)

                # j. 提前退出优化 (remainder 极小时无意义继续)
                # 注: 训练时 soft ACT 几乎不触发, 但避免 .item() 以兼容 torch.compile
                if not torch.compiler.is_compiling():
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
                h_before = h

                # L6 注意力调制 (用 score[k-1], 仅 k>0)
                precision_bias = None
                if k > 0 and all_scores:
                    precision_bias = self.l6_precision.compute_attention_bias(
                        all_scores[-1]
                    )

                for ci, col_layer in enumerate(self.column_layers):
                    col_cycle_k = k * len(self.column_layers) + ci
                    col_precision = precision_bias if ci == 0 else None
                    h = col_layer(
                        h.to(self._model_dtype),
                        attention_mask=column_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        cycle_k=col_cycle_k,
                        precision_bias=col_precision,
                    )

                h = self.inter_iter_norm(h)

                # Score: 当前迭代的预测误差
                score = self.cct_predictor.compute_score(h_before, h)
                all_scores.append(score)

                # L6 停止决策 (用 score[k], 当前迭代)
                p_halt = self.l6_precision.compute_halt(score, inf_tau)
                p_halts.append(p_halt)
                remainders_list.append(remainder.clone())

                # Per-token ACT 累积
                weight = (remainder * p_halt).unsqueeze(-1)
                output = output + weight * h
                remainder = remainder * (1.0 - p_halt)

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
                lambda_ponder=self.config.lambda_ponder,
                use_ponder_cost=self.config.use_ponder_cost,
                valid_mask=valid_mask,
            )

        # 计算 effective_iters (期望迭代数, 用于日志)
        effective_iters = 0.0
        eff_iters_std = 0.0
        if p_halts:
            with torch.no_grad():
                eff = torch.zeros_like(p_halts[0])
                for k, (ph, rm) in enumerate(zip(p_halts, remainders_list)):
                    eff = eff + (k + 1) * rm * ph
                if remainders_list:
                    eff = eff + len(p_halts) * remainders_list[-1] * (1.0 - p_halts[-1])
                if valid_mask is not None:
                    effective_iters = (eff * valid_mask).sum().item() / max(valid_mask.sum().item(), 1)
                    eff_masked = eff[valid_mask.bool()]
                    eff_iters_std = eff_masked.std().item() if eff_masked.numel() > 1 else 0.0
                else:
                    effective_iters = eff.mean().item()
                    eff_iters_std = eff.std().item()

        # score 在 token 维度的 std (L6 Precision 健康指标)
        score_std_mean = 0.0
        if all_scores:
            with torch.no_grad():
                score_stds = [s.std(dim=-1).mean().item() for s in all_scores]
                score_std_mean = sum(score_stds) / len(score_stds)

        return {
            "loss": loss,
            "logits": logits,
            "loss_dict": loss_dict,
            "p_halts": p_halts,
            "scores": all_scores,
            "num_iterations": len(all_scores),
            "effective_iters": effective_iters,
            "eff_iters_std": eff_iters_std,
            "score_std": score_std_mean,
        }

    def set_halt_tau(self, tau: float):
        """更新 L6 停止退火温度"""
        self.halt_tau.fill_(tau)

    def fold_fusions(self):
        """折叠所有 FusionLinear → 普通 Linear (零推理开销)

        调用后 use_fusion_graft 的效果永久生效于权重中,
        模型行为不变, 但不再有额外参数/buffer.
        部署前或保存最终 checkpoint 前调用.
        """
        fold_all_fusions(self)

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def get_param_groups(self) -> List[dict]:
        """返回分层学习率参数组

        Group 0: 基座参数 (learning_rate)
        Group 1: CCT 新模块 (new_module_lr)
        Group 2: 融合参数 A/B (fusion_lr) — 仅 use_fusion_graft=True 时存在
        """
        new_modules = [
            self.cct_predictor, self.l6_precision,
            self.inter_iter_norm,
        ]
        new_params = set()
        for m in new_modules:
            new_params.update(id(p) for p in m.parameters())

        # Fusion params (from FusionLinear.A, FusionLinear.B)
        fusion_param_ids = set()
        fusion_params_list = []
        if self.config.use_fusion_graft:
            fp = get_fusion_params(self)
            fusion_param_ids = set(id(p) for p in fp)
            fusion_params_list = [p for p in fp if p.requires_grad]

        # Disjointness check
        assert not (new_params & fusion_param_ids), \
            "融合参数与新模块参数重叠!"

        base_params = [
            p for p in self.parameters()
            if p.requires_grad
            and id(p) not in new_params
            and id(p) not in fusion_param_ids
        ]
        new_params_list = [
            p for p in self.parameters()
            if p.requires_grad and id(p) in new_params
        ]

        groups = [
            {"params": base_params, "lr": self.config.learning_rate},
            {"params": new_params_list, "lr": self.config.new_module_lr},
        ]
        if fusion_params_list:
            groups.append(
                {"params": fusion_params_list, "lr": self.config.fusion_lr}
            )
        return groups

    def get_trainable_params_info(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cct_modules = [
            self.cct_predictor, self.l6_precision,
            self.inter_iter_norm,
        ]
        cct_params = sum(
            p.numel() for m in cct_modules for p in m.parameters()
        )
        col_mlp_params = sum(
            p.numel() for layer in self.column_layers for p in layer.mlp.parameters()
        )
        info = (
            f"Total params: {total:,}\n"
            f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)\n"
            f"CCT new params: {cct_params:,} ({100*cct_params/total:.2f}%)\n"
            f"Column MLP params: {col_mlp_params:,}"
        )
        if self.config.use_ffn_expansion and self.config.column_d_ff > self.config.d_ff:
            info += f" (widened d_ff={self.config.column_d_ff})"
        if self.config.use_fusion_graft:
            fp_count = get_fusion_param_count(self)
            fb_count = get_fusion_buffer_count(self)
            info += (
                f"\nFusion params (A,B): {fp_count:,} "
                f"(rank={self.config.fusion_rank})\n"
                f"Fusion buffers (W_pruned): {fb_count:,} "
                f"(non-persistent, 不计入 state_dict)"
            )
        return info

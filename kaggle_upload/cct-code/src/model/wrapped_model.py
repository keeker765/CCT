"""CCTLlamaModel — 皮层柱 Transformer 主模型 (v2: entropy-driven)

架构: Fixed Front (1层) → Column (3层 × K循环) → Fixed Back (1层)
预训练层映射: Layer [0] → Front; Layer [3,8,12] → Column; Layer [15] → Back

Entropy-driven halt:
- 每次迭代后通过 Back → Norm → LM_head 计算输出分布 entropy
- 训练/推理统一: mean(H_norm) < threshold → 硬停止 (train/infer 对齐)
- Per-query attention temperature: temp = 1 - scale × H_norm
- L_mono: 监督 entropy 逐迭代递减 (entropy_floor 防崩溃)
"""

import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """CCT 包装的 Llama 模型 (v2: entropy-driven)"""

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
            # Column 层融合 (标准: 每个 column 吸收到下一个 used 层之间的 donor)
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

            # Front 层融合 (v2 新增: front 吸收被删的中间层)
            used = set(config.pretrained_front_layers
                       + config.pretrained_column_layers
                       + config.pretrained_back_layers)
            all_used_sorted = sorted(used)
            for fi, front_idx in enumerate(config.pretrained_front_layers):
                next_used = config.num_base_layers
                for u in all_used_sorted:
                    if u > front_idx:
                        next_used = u
                        break
                front_donors = [
                    i for i in range(front_idx + 1, next_used) if i not in used
                ]
                if front_donors:
                    donor_layers = [
                        base_model.model.layers[j] for j in front_donors
                    ]
                    attach_fusion_grafts(
                        self.front_layers[fi],
                        donor_layers,
                        rank=config.fusion_rank,
                        pool_donors=config.fusion_pool_donors,
                        freeze_base=config.fusion_freeze_base,
                    )
                    print(
                        f"  Front[{front_idx}] ← Fusion{front_donors} "
                        f"(rank={config.fusion_rank})"
                    )

        # === 构建 Fixed Back 层 (标准 LlamaDecoderLayer) ===
        self.back_layers = nn.ModuleList()
        for src_idx in config.pretrained_back_layers:
            self.back_layers.append(base_model.model.layers[src_idx])

        # === CCT 新增模块 (v2: entropy-driven) ===
        # 迭代间 RMSNorm (防发散)
        self.inter_iter_norm = LlamaRMSNorm(
            config.d_model, eps=config.rms_norm_eps
        )

        # Entropy 归一化常数: log(vocab_size)
        vocab_size = base_model.config.vocab_size
        self.register_buffer(
            "log_vocab_size",
            torch.tensor(math.log(vocab_size), dtype=torch.float32),
        )

        self._gradient_checkpointing = False
        self._model_dtype = self.embed_tokens.weight.dtype

        # 将新模块转换为与基座相同的 dtype
        self.inter_iter_norm.to(self._model_dtype)

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

    def _compute_entropy_and_lm_loss(
        self,
        h: torch.Tensor,
        labels: Optional[torch.LongTensor],
        layer_kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """通过 Back → Norm → LM_head 计算 entropy 和 LM loss

        Returns:
            entropy: [B, T] — 原始 entropy (nats)
            h_norm_entropy: [B, T] — 归一化 entropy [0, 1]
            lm_loss: 标量 (仅 labels 不为 None 时)
        """
        # Back layers
        h_back = h
        for layer in self.back_layers:
            h_back = self._run_standard_layer(layer, h_back, layer_kwargs)

        # Final norm + LM head
        h_back = self.final_norm(h_back)
        logits = self.lm_head(h_back)

        # Entropy: -Σ p log p (用 log_softmax 提高数值稳定性)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, T]
        h_norm_entropy = entropy / self.log_vocab_size.float()  # [0, 1]

        # LM loss (如果有 labels)
        lm_loss = None
        if labels is not None:
            lm_loss = compute_lm_loss(logits, labels)

        return entropy, h_norm_entropy, lm_loss

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

        # === 3. Column 循环 (entropy-driven) ===
        h = hidden_states

        # Column 层 SDPA 优化: 无 padding 时传 None mask → is_causal=True
        has_padding = attention_mask is not None and (attention_mask == 0).any()
        column_mask = layer_kwargs.get("attention_mask") if has_padding else None

        entropy_temperature = None  # 第一次迭代不调制
        all_entropies: List[torch.Tensor] = []
        all_lm_losses: List[torch.Tensor] = []
        num_iters_executed = 0

        for k in range(self.config.max_iter):
            # a. 3个 CCTDecoderLayer forward (共享权重)
            for ci, col_layer in enumerate(self.column_layers):
                col_cycle_k = k * len(self.column_layers) + ci
                col_temp = entropy_temperature if ci == 0 else None
                if self._gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(
                        col_layer, h.to(self._model_dtype),
                        column_mask,
                        position_ids,
                        None,  # past_key_values
                        position_embeddings,
                        col_cycle_k,
                        None,  # precision_bias (deprecated)
                        col_temp,
                        use_reentrant=False,
                    )
                else:
                    h = col_layer(
                        h.to(self._model_dtype),
                        attention_mask=column_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        cycle_k=col_cycle_k,
                        entropy_temperature=col_temp,
                    )

            # b. 迭代间 RMSNorm (防发散)
            h = self.inter_iter_norm(h)

            # c. Entropy + LM loss (通过 back → norm → lm_head)
            if self._gradient_checkpointing and self.training:
                entropy, h_norm, lm_loss_k = torch.utils.checkpoint.checkpoint(
                    self._compute_entropy_and_lm_loss,
                    h, labels, layer_kwargs,
                    use_reentrant=False,
                )
            else:
                entropy, h_norm, lm_loss_k = self._compute_entropy_and_lm_loss(
                    h, labels, layer_kwargs,
                )

            all_entropies.append(h_norm)
            if lm_loss_k is not None:
                all_lm_losses.append(lm_loss_k)

            num_iters_executed = k + 1

            # d. Per-query temperature for next iteration (detached)
            entropy_temperature = (
                1.0 - self.config.entropy_temp_scale * h_norm.detach()
            ).clamp(min=0.5, max=1.0)  # [B, T]

            # e. 硬停止 (训练和推理都执行, 保持 train/infer 对齐)
            if k >= self.config.min_iter - 1:
                if attention_mask is not None:
                    valid = attention_mask[:, :seq_len].bool()
                    mean_h_norm = h_norm[valid].mean().item() if valid.any() else h_norm.mean().item()
                else:
                    mean_h_norm = h_norm.mean().item()
                if mean_h_norm < self.config.halt_entropy_threshold:
                    break

        # === 4. 最终输出 ===
        # 最后一次迭代已经跑过 back+norm+lm_head 算 entropy/loss，
        # 但 h 仍是 column output (未经 back)，需要重新跑 back
        hidden_states = h
        for layer in self.back_layers:
            hidden_states = self._run_standard_layer(layer, hidden_states, layer_kwargs)

        # === 5. Final Norm + LM Head (最终 logits) ===
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # === 6. 计算损失 ===
        loss = None
        loss_dict = {}

        if labels is not None and all_lm_losses:
            # L_mono 只在有监督的 token 上施加压力 (labels != -100)
            valid_mask = (labels != -100).float()
            if attention_mask is not None:
                valid_mask = valid_mask * attention_mask[:, :seq_len].float()

            loss, loss_dict = compute_total_loss(
                lm_losses=all_lm_losses,
                entropies=all_entropies,
                lambda_mono=self.config.lambda_mono,
                valid_mask=valid_mask,
                entropy_floor=self.config.entropy_floor,
            )

        # 报告每次迭代的 mean±std entropy (仅有效 token)
        mean_entropy = 0.0
        std_entropy = 0.0
        per_iter_entropy = []  # list of (mean, std) tuples
        if all_entropies:
            with torch.no_grad():
                valid = None
                if attention_mask is not None:
                    valid = attention_mask[:, :seq_len].bool()

                for h_k in all_entropies:
                    if valid is not None and valid.any():
                        vh = h_k[valid]
                        per_iter_entropy.append((vh.mean().item(), vh.std().item() if vh.numel() > 1 else 0.0))
                    else:
                        per_iter_entropy.append((h_k.mean().item(), h_k.std().item()))

                # 最后一次迭代的统计
                last_h = all_entropies[-1]
                if valid is not None and valid.any():
                    valid_h = last_h[valid]
                    mean_entropy = valid_h.mean().item()
                    std_entropy = valid_h.std().item() if valid_h.numel() > 1 else 0.0
                else:
                    mean_entropy = last_h.mean().item()
                    std_entropy = last_h.std().item()

        return {
            "loss": loss,
            "logits": logits,
            "loss_dict": loss_dict,
            "num_iterations": num_iters_executed,
            "mean_entropy": mean_entropy,
            "std_entropy": std_entropy,
            "per_iter_entropy": per_iter_entropy,
            "entropies": [e.detach() for e in all_entropies],
        }

    def fold_fusions(self):
        """折叠所有 FusionLinear → 普通 Linear (零推理开销)"""
        fold_all_fusions(self)

    def get_fusion_magnitudes(self) -> Dict[str, float]:
        """返回各层 FusionLinear 的融合量 ||A @ B^T||_F / ||W_pruned||_F

        Returns:
            dict: {层名: 融合比例}, 例如 {"col.0.q_proj": 0.032, ...}
        """
        result = {}
        for name, module in self.named_modules():
            if isinstance(module, FusionLinear) and not module._folded:
                with torch.no_grad():
                    ab = module.A @ module.B.T  # (d_out, d_in)
                    fusion_norm = ab.norm().item()
                    pruned_norm = module.W_pruned.norm().item()
                    ratio = fusion_norm / max(pruned_norm, 1e-8)
                    short_name = name.replace("column_layers.", "col.").replace(
                        "front_layers.", "front.").replace(
                        "back_layers.", "back.").replace(
                        "self_attn.", "").replace("mlp.", "")
                    result[short_name] = ratio
        return result

    def set_halt_threshold(self, threshold: float):
        """动态设置推理 halt 阈值 (用于退火)"""
        self.config.halt_entropy_threshold = threshold

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def get_param_groups(self) -> List[dict]:
        """返回分层学习率参数组

        Group 0: 基座参数 (learning_rate)
        Group 1: CCT 新模块 (new_module_lr)
        Group 2: 融合参数 A/B (fusion_lr) — 仅 use_fusion_graft=True 时存在
        """
        new_modules = [self.inter_iter_norm]
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
        cct_modules = [self.inter_iter_norm]
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

"""CCTConfig — 皮层柱 Transformer 配置 (v2: entropy-driven)"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CCTConfig:
    """CCT 整体配置

    架构: Fixed Front (1层) → Column (3层 × K循环) → Fixed Back (1层)
    预训练层映射: Llama 3.2 1B (16层) → 5层迁移
    v2: output entropy 驱动停止 + per-query attention temperature
    """
    # 基座模型
    model_name: str = "unsloth/Llama-3.2-1B"
    num_base_layers: int = 16
    d_model: int = 2048
    head_dim: int = 64
    num_q_heads: int = 32
    num_kv_heads: int = 8
    d_ff: int = 8192
    rms_norm_eps: float = 1e-5

    # 预训练层映射
    pretrained_front_layers: List[int] = field(default_factory=lambda: [0])
    pretrained_column_layers: List[int] = field(default_factory=lambda: [3, 8, 12])
    pretrained_back_layers: List[int] = field(default_factory=lambda: [15])

    # 循环参数
    min_iter: int = 1
    max_iter: int = 10
    phi: float = 1.618  # 黄金比例

    # Column MLP 加宽 (Net2WiderNet) — 与 fusion_graft 互斥
    use_ffn_expansion: bool = False  # 总开关: False=关闭加宽, True=启用加宽
    column_d_ff: int = 8192  # 默认=原始, 设 12288(1.5x) 或 16384(2x) 激活加宽
    widen_noise_std: float = 0.01  # 复制神经元的噪声强度
    widen_mode: str = "cross"  # "self"=Net2WiderNet复制自身, "cross"=融合donor层FFN
    donor_init_scale: float = 0.1  # cross模式: donor层down_proj初始缩放

    # FuseGPT-style 在线可学习融合 — 与 ffn_expansion 互斥
    use_fusion_graft: bool = False   # 总开关
    fusion_rank: int = 64            # 低秩融合的秩 (64 适合 1B 模型, 7B 用 128)
    fusion_lr: float = 1e-3          # 融合参数 (A, B) 的独立学习率
    fusion_pool_donors: bool = True  # True=先平均多个 donor 权重, False=未来扩展
    fusion_freeze_base: bool = False # True=冻结被包装的基础权重, 仅训练 A/B

    # Entropy-based halt (v2)
    lambda_mono: float = 0.1         # L_mono 相对权重 (0.1 = 10% of LM, 自适应缩放)
    entropy_temp_scale: float = 0.5  # per-query temperature: temp = 1 - scale * H_norm
    halt_entropy_threshold: float = 0.3  # 推理硬停止最终阈值
    halt_threshold_start: float = 0.8    # 退火起始阈值 (训练初期 eval 几乎不停)
    halt_threshold_end: float = 0.3      # 退火结束阈值 (= halt_entropy_threshold)

    # 训练
    learning_rate: float = 2e-5
    new_module_lr: float = 1e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 512
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    max_steps: int = 10000
    warmup_steps: int = 500

    @property
    def num_front_layers(self) -> int:
        return len(self.pretrained_front_layers)

    @property
    def num_column_layers(self) -> int:
        return len(self.pretrained_column_layers)

    @property
    def num_back_layers(self) -> int:
        return len(self.pretrained_back_layers)

    def __post_init__(self):
        """验证层映射合法性"""
        all_layers = (
            self.pretrained_front_layers
            + self.pretrained_column_layers
            + self.pretrained_back_layers
        )
        if len(all_layers) != len(set(all_layers)):
            from collections import Counter
            counts = Counter(all_layers)
            dups = [l for l, c in counts.items() if c > 1]
            raise ValueError(f"层映射重叠: {dups}")

        for l in all_layers:
            if l < 0 or l >= self.num_base_layers:
                raise ValueError(
                    f"层索引 {l} 超出范围 [0, {self.num_base_layers})"
                )

        if self.max_iter < self.min_iter:
            raise ValueError(
                f"max_iter ({self.max_iter}) < min_iter ({self.min_iter})"
            )

        if self.use_fusion_graft and self.use_ffn_expansion:
            raise ValueError(
                "use_fusion_graft 和 use_ffn_expansion 不可同时启用。"
                "融合 (fusion_graft) 通过低秩矩阵吸收被删层知识；"
                "加宽 (ffn_expansion) 通过拼接神经元扩展 MLP。"
            )

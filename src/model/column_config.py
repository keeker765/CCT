"""CCTConfig — 皮层柱 Transformer 配置"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CCTConfig:
    """CCT 整体配置

    架构: Fixed Front → Column (循环) → Fixed Back
    层映射: Llama 3.2 1B (16层) → 7层迁移
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
    pretrained_front_layers: List[int] = field(default_factory=lambda: [0, 1])
    pretrained_column_layers: List[int] = field(default_factory=lambda: [2, 7, 12])
    pretrained_back_layers: List[int] = field(default_factory=lambda: [14, 15])

    # 循环参数
    min_iter: int = 1
    max_iter: int = 10
    phi: float = 1.618  # 黄金比例

    # L6 Precision
    lambda_precision_init: float = 1.0
    precision_temperature: float = 0.5  # 固定超参数

    # Predictor
    info_dim: int = 256  # info_proj 降维维度 (与 PPG 一致)

    # HaltHead 退火温度
    halt_tau_start: float = 1.0
    halt_tau_end: float = 0.01

    # 损失权重
    lambda_pred: float = 0.1
    lambda_entropy: float = 0.01  # halting 熵正则 (最小化 → 锐利停止决策)
    lambda_ponder: float = 0.0  # ponder cost (已关闭: eff_iters→1 过快)
    use_ponder_cost: bool = False  # 开关: 关闭 L_ponder 以避免压制迭代

    # 推理温度 (控制推理强度)
    inference_temperature: float = 1.0  # >1 → 更多迭代 (尤其 hard token); <1 → 更少迭代

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

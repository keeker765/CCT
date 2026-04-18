"""CCTTrainer — 单阶段训练器"""

import os
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict
import logging

from ..model.wrapped_model import CCTLlamaModel
from ..model.column_config import CCTConfig
from .scheduler import get_cosine_schedule_with_warmup, compute_halt_tau

logger = logging.getLogger(__name__)


class CCTTrainer:
    """CCT 单阶段训练器

    - 全参数解冻, 分层 LR
    - ACT 软停止, τ_halt 线性退火
    - AMP (bf16/fp16) + gradient checkpointing + gradient accumulation
    """

    def __init__(
        self,
        model: CCTLlamaModel,
        config: CCTConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        output_dir: str = "output/cct",
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.output_dir = output_dir

        # 确定 AMP 精度模式
        self.use_amp = config.fp16 or config.bf16
        self.amp_dtype = torch.bfloat16 if config.bf16 else torch.float16

        # 分层学习率优化器
        param_groups = model.get_param_groups()
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=getattr(config, "weight_decay", 0.01),
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        # bf16 不需要 GradScaler; fp16 需要但要求 fp32 master weights
        use_scaler = config.fp16 and not config.bf16
        self.scaler = GradScaler("cuda", enabled=use_scaler)
        self.global_step = 0

        if config.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        os.makedirs(output_dir, exist_ok=True)

    def train(self):
        self.model.train()
        device = next(self.model.parameters()).device
        accum_steps = self.config.gradient_accumulation_steps
        max_grad_norm = getattr(self.config, "max_grad_norm", 1.0)

        pbar = tqdm(total=self.config.max_steps, desc="Training")
        data_iter = iter(self.train_dataloader)

        self.optimizer.zero_grad()

        while self.global_step < self.config.max_steps:
            # 获取 batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            batch = {k: v.to(device) for k, v in batch.items()}

            # 更新 τ_halt
            tau_halt = compute_halt_tau(
                self.global_step,
                self.config.max_steps,
                self.config.halt_tau_start,
                self.config.halt_tau_end,
            )
            self.model.set_halt_tau(tau_halt)

            # Forward + backward
            with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = outputs["loss"] / accum_steps

            self.scaler.scale(loss).backward()

            # 梯度累积
            if (self.global_step + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # 日志
            if self.global_step % getattr(self.config, "logging_steps", 10) == 0:
                loss_dict = outputs.get("loss_dict", {})
                num_iter = outputs.get("num_iterations", 0)
                eff_iters = outputs.get("effective_iters", 0)
                logger.info(
                    f"Step {self.global_step} | "
                    f"loss={loss_dict.get('loss_total', 0):.4f} | "
                    f"lm={loss_dict.get('loss_lm', 0):.4f} | "
                    f"pred={loss_dict.get('loss_pred', 0):.4f} | "
                    f"entropy={loss_dict.get('loss_entropy', 0):.4f} | "
                    f"ponder={loss_dict.get('loss_ponder', 0):.4f} | "
                    f"eff_iters={eff_iters:.2f} | τ_halt={tau_halt:.4f}"
                )

            # 保存
            save_steps = getattr(self.config, "save_steps", 500)
            if (self.global_step + 1) % save_steps == 0:
                self.save_checkpoint()

            self.global_step += 1
            pbar.update(1)

        pbar.close()
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        torch.save(
            {"global_step": self.global_step},
            os.path.join(path, "trainer_state.pt"),
        )
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        )
        self.optimizer.load_state_dict(
            torch.load(os.path.join(path, "optimizer.pt"), map_location="cpu")
        )
        state = torch.load(os.path.join(path, "trainer_state.pt"))
        self.global_step = state["global_step"]
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")

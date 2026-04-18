"""CCT 训练入口"""

import argparse
import logging
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM

from ..model.column_config import CCTConfig
from ..model.wrapped_model import CCTLlamaModel
from ..data.dataset import TextDataset
from ..data.collator import CCTCollator
from .trainer import CCTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> CCTConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return CCTConfig(**{
        k: v for k, v in cfg.items()
        if k in CCTConfig.__dataclass_fields__
    })


def main():
    parser = argparse.ArgumentParser(description="CCT Training")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    logger.info(f"Config: {config}")

    # 加载基座模型
    logger.info(f"Loading base model: {config.model_name}")
    if config.bf16:
        load_dtype = torch.bfloat16
    elif config.fp16:
        load_dtype = torch.float16
    else:
        load_dtype = torch.float32

    base_model = LlamaForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=load_dtype,
        device_map="cpu",
    )

    # 构建 CCT 模型
    logger.info("Building CCT model...")
    model = CCTLlamaModel(base_model, config)
    model = model.to(args.device)
    logger.info(model.get_trainable_params_info())

    # 数据集
    logger.info("Loading dataset...")
    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    dataset = TextDataset(
        dataset_name=raw_cfg.get("dataset_name", "HuggingFaceFW/fineweb-edu"),
        dataset_config=raw_cfg.get("dataset_config", "sample-10BT"),
        dataset_split=raw_cfg.get("dataset_split", "train"),
        max_seq_len=config.max_seq_len,
        tokenizer_name=config.model_name,
        max_samples=raw_cfg.get("max_samples", None),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        collate_fn=CCTCollator(),
        num_workers=0,
        pin_memory=True,
    )

    # 训练
    trainer = CCTTrainer(
        model=model,
        config=config,
        train_dataloader=dataloader,
        output_dir=raw_cfg.get("output_dir", "output/cct_base"),
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

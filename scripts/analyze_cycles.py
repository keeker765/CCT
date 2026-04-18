# -*- coding: utf-8 -*-
"""分析 CCT 模型的循环迭代模式

功能:
1. 从已保存的模型输出中提取迭代次数
2. 绘制迭代次数分布直方图
3. 与 Simple Repeat 基线对比
4. 按 token 复杂度分析迭代分配

使用方法:
    python scripts/analyze_cycles.py --model_path output/cct_base/checkpoint-10000
    python scripts/analyze_cycles.py --model_path output/cct_base/checkpoint-10000 --num_samples 500
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_data(
    model_path: str,
    config_path: str = "configs/base_config.yaml",
    device: str = "cuda",
    max_samples: int = 200,
):
    """加载模型和评测数据

    Args:
        model_path: 检查点路径 (含 model.pt)
        config_path: 配置文件路径
        device: 计算设备
        max_samples: 最大评测样本数

    Returns:
        model, dataloader, config 元组
    """
    from src.model.column_config import CCTConfig
    from src.model.wrapped_model import CCTLlamaModel
    from src.data.dataset import TextDataset
    from src.data.collator import CCTCollator
    from transformers import LlamaForCausalLM
    from torch.utils.data import DataLoader

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    config = CCTConfig(**{
        k: v for k, v in raw_cfg.items()
        if k in CCTConfig.__dataclass_fields__
    })

    logger.info(f"加载基座模型: {config.model_name}")
    base_model = LlamaForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="cpu",
    )

    model = CCTLlamaModel(base_model, config)

    # 加载检查点权重
    ckpt_file = os.path.join(model_path, "model.pt")
    if os.path.exists(ckpt_file):
        state_dict = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(f"已加载检查点: {ckpt_file}")
    else:
        logger.warning(f"未找到检查点文件: {ckpt_file}，使用初始化权重")

    model = model.to(device)
    model.eval()

    # 加载数据
    dataset = TextDataset(
        dataset_name=raw_cfg.get("dataset_name", "HuggingFaceFW/fineweb-edu"),
        dataset_config=raw_cfg.get("dataset_config", "sample-10BT"),
        dataset_split=raw_cfg.get("dataset_split", "train"),
        max_seq_len=config.max_seq_len,
        tokenizer_name=config.model_name,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=CCTCollator(),
        num_workers=0,
    )

    return model, dataloader, config


def collect_cycle_data(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda",
) -> Dict[str, list]:
    """收集每个样本的循环迭代数据

    Args:
        model: CCTLlamaModel 实例
        dataloader: 数据加载器
        device: 计算设备

    Returns:
        data: {
            'iteration_counts': 每个样本的迭代次数,
            'p_halt_means': 每个样本最终轮的平均 p_halt,
            'scores': 每个样本各轮的 score 列表,
        }
    """
    from tqdm import tqdm

    iteration_counts: List[int] = []
    p_halt_means: List[float] = []
    all_scores: List[List[float]] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="收集循环数据"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )

            iteration_counts.append(outputs["num_iterations"])

            if outputs.get("p_halts"):
                last_p = outputs["p_halts"][-1]
                p_halt_means.append(last_p.mean().item())

            if outputs.get("scores"):
                sample_scores = [s.mean().item() for s in outputs["scores"]]
                all_scores.append(sample_scores)

    return {
        "iteration_counts": iteration_counts,
        "p_halt_means": p_halt_means,
        "scores": all_scores,
    }


def simple_repeat_baseline(
    fixed_iter: int,
    num_samples: int,
) -> List[int]:
    """Simple Repeat 基线：固定迭代次数

    Args:
        fixed_iter: 固定迭代次数
        num_samples: 样本数

    Returns:
        counts: 长度为 num_samples 的固定迭代次数列表
    """
    return [fixed_iter] * num_samples


def plot_comparison(
    cct_counts: List[int],
    baseline_counts: List[int],
    max_iter: int = 10,
    save_dir: str = "results",
):
    """绘制 CCT vs Simple Repeat 迭代分布对比图

    Args:
        cct_counts: CCT 模型的迭代次数列表
        baseline_counts: 基线的迭代次数列表
        max_iter: 最大迭代次数
        save_dir: 图片保存目录
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.arange(0.5, max_iter + 1.5, 1)

    # 左: CCT 迭代分布
    axes[0].hist(
        cct_counts, bins=bins, color="steelblue",
        edgecolor="white", alpha=0.8, density=True,
    )
    mean_cct = np.mean(cct_counts)
    axes[0].axvline(mean_cct, color="red", linestyle="--", label=f"均值 = {mean_cct:.2f}")
    axes[0].set_xlabel("迭代次数")
    axes[0].set_ylabel("频率密度")
    axes[0].set_title("CCT 自适应循环分布")
    axes[0].set_xticks(range(1, max_iter + 1))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # 右: 对比箱线图
    data_to_plot = [cct_counts, baseline_counts]
    bp = axes[1].boxplot(
        data_to_plot, labels=["CCT (自适应)", "Simple Repeat"],
        patch_artist=True, widths=0.5,
    )
    colors = ["steelblue", "coral"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_ylabel("迭代次数")
    axes[1].set_title("CCT vs Simple Repeat 对比")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "cycle_analysis.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"图表已保存: {save_path}")
    plt.close(fig)

    # 统计摘要
    logger.info(f"\n{'='*50}")
    logger.info("迭代统计摘要:")
    logger.info(f"  CCT: mean={mean_cct:.2f}, median={np.median(cct_counts):.1f}, "
                f"std={np.std(cct_counts):.2f}, range=[{min(cct_counts)}, {max(cct_counts)}]")
    base_mean = np.mean(baseline_counts)
    logger.info(f"  Baseline: mean={base_mean:.2f} (固定)")
    logger.info(f"  计算节省: {(1 - mean_cct / base_mean) * 100:.1f}% vs 固定{int(base_mean)}轮")


def main():
    parser = argparse.ArgumentParser(description="分析 CCT 循环迭代模式")
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=200, help="评测样本数")
    parser.add_argument("--baseline_iter", type=int, default=5, help="Simple Repeat 基线迭代次数")
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    logger.info("加载模型和数据...")
    model, dataloader, config = load_model_and_data(
        model_path=args.model_path,
        config_path=args.config,
        device=args.device,
        max_samples=args.num_samples,
    )

    logger.info("收集循环数据...")
    cycle_data = collect_cycle_data(model, dataloader, device=args.device)

    cct_counts = cycle_data["iteration_counts"]
    baseline_counts = simple_repeat_baseline(args.baseline_iter, len(cct_counts))

    logger.info("绘制对比图表...")
    plot_comparison(
        cct_counts=cct_counts,
        baseline_counts=baseline_counts,
        max_iter=config.max_iter,
        save_dir=args.save_dir,
    )

    # 保存原始数据
    data_path = os.path.join(args.save_dir, "cycle_data.pt")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(cycle_data, data_path)
    logger.info(f"原始数据已保存: {data_path}")


if __name__ == "__main__":
    main()

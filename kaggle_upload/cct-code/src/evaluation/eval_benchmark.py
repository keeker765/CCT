# -*- coding: utf-8 -*-
"""CCT 基准评测 — WikiText-103 PPL 与下游任务

使用 lm-eval 库进行标准化评测:
- WikiText-103 困惑度 (PPL)
- HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande
"""

import math
import logging
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate_ppl(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """在给定数据集上计算困惑度 (PPL)

    Args:
        model: CCTLlamaModel 实例，需实现 forward(input_ids, attention_mask, labels)
        eval_dataloader: 评测数据加载器，每个 batch 含 input_ids, attention_mask, labels
        device: 计算设备

    Returns:
        ppl: 困惑度值
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="评估 PPL"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )

            loss = outputs["loss"]
            # 统计有效 token 数 (labels != -100)
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)

    logger.info(f"评估完成: avg_loss={avg_loss:.4f}, PPL={ppl:.2f} ({total_tokens} tokens)")
    return ppl


def run_benchmarks(
    model_path: str,
    tasks: Optional[List[str]] = None,
    num_fewshot: int = 0,
    batch_size: int = 4,
    device: str = "cuda",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """使用 lm-eval 库运行标准下游任务评测

    Args:
        model_path: 模型检查点路径 (含 model.pt 和配置)
        tasks: 评测任务列表，默认为 HellaSwag/ARC/PIQA/WinoGrande
        num_fewshot: few-shot 示例数
        batch_size: 推理批大小
        device: 计算设备
        limit: 每个任务最多评测样本数 (调试用)

    Returns:
        results: 包含各任务指标的字典
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    if tasks is None:
        tasks = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]

    logger.info(f"加载模型: {model_path}")
    logger.info(f"评测任务: {tasks}, few-shot={num_fewshot}")

    # 使用 lm-eval 的 HuggingFace 包装器
    # 对于自定义模型，需通过 pretrained 参数传入路径
    lm = HFLM(
        pretrained=model_path,
        device=device,
        batch_size=batch_size,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
    )

    # 提取并格式化结果
    summary: Dict[str, Any] = {}
    for task_name, task_result in results.get("results", {}).items():
        summary[task_name] = {
            k: v for k, v in task_result.items()
            if isinstance(v, (int, float))
        }
        logger.info(f"  {task_name}: {summary[task_name]}")

    return summary


def main():
    """命令行评测入口"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="CCT 基准评测")
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--tasks", nargs="+", default=None, help="评测任务列表")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="每任务最大样本数 (调试)")
    args = parser.parse_args()

    results = run_benchmarks(
        model_path=args.model_path,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    print("\n=== 评测结果汇总 ===")
    for task, metrics in results.items():
        print(f"  {task}: {metrics}")


if __name__ == "__main__":
    main()

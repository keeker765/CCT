# -*- coding: utf-8 -*-
"""CCT 效率评测 — PPL/FLOP 曲线、推理延迟、循环迭代统计

测量模型在不同计算预算下的表现:
1. PPL/FLOP 曲线: 不同 max_iter 下的 PPL 与 FLOPs 关系
2. 推理延迟: 端到端推理时间
3. 循环迭代统计: 平均/中位/分布
"""

import math
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyResult:
    """效率评测结果"""
    ppl: float = 0.0
    avg_flops: float = 0.0
    avg_iterations: float = 0.0
    median_iterations: float = 0.0
    latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    iteration_counts: List[int] = field(default_factory=list)


def estimate_column_flops(
    config,
    num_iterations: int,
    seq_len: int,
    batch_size: int = 1,
) -> float:
    """估算 Column 部分的 FLOPs

    每次迭代 = num_column_layers × (Attention + FFN) FLOPs
    Attention ≈ 4·d·d·seq + 2·d·seq²
    FFN ≈ 3·d·d_ff·seq (gate + up + down)

    Args:
        config: CCTConfig 实例
        num_iterations: 循环迭代次数
        seq_len: 序列长度
        batch_size: 批大小

    Returns:
        estimated_flops: 估算的浮点运算量
    """
    d = config.d_model
    d_ff = config.d_ff
    n_col = config.num_column_layers

    # 单层 FLOPs (粗估, 乘2 for multiply-add)
    attn_flops = 2 * (4 * d * d * seq_len + 2 * d * seq_len * seq_len)
    ffn_flops = 2 * 3 * d * d_ff * seq_len
    layer_flops = attn_flops + ffn_flops

    total = batch_size * num_iterations * n_col * layer_flops
    return total


def evaluate_ppl_flops_curve(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    max_iter_values: Optional[List[int]] = None,
    device: str = "cuda",
) -> List[Dict[str, float]]:
    """在不同 max_iter 设置下评估 PPL 与 FLOPs，生成 PPL/FLOP 曲线数据

    Args:
        model: CCTLlamaModel 实例
        eval_dataloader: 评测数据加载器
        max_iter_values: 要测试的 max_iter 列表，默认 [1, 2, 3, 5, 7, 10]
        device: 计算设备

    Returns:
        results: 每个 max_iter 对应的 {max_iter, ppl, avg_flops, avg_iterations} 列表
    """
    if max_iter_values is None:
        max_iter_values = [1, 2, 3, 5, 7, 10]

    model.eval()
    original_max_iter = model.config.max_iter
    results: List[Dict[str, float]] = []

    for max_iter in max_iter_values:
        model.config.max_iter = max_iter
        logger.info(f"评估 max_iter={max_iter}...")

        total_loss = 0.0
        total_tokens = 0
        total_iters = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc=f"max_iter={max_iter}", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )

                num_tokens = (batch["labels"] != -100).sum().item()
                total_loss += outputs["loss"].item() * num_tokens
                total_tokens += num_tokens
                total_iters += outputs["num_iterations"]
                num_batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 100.0))  # 限制防溢出
        avg_iters = total_iters / max(num_batches, 1)

        seq_len = next(iter(eval_dataloader))["input_ids"].shape[1]
        avg_flops = estimate_column_flops(model.config, avg_iters, seq_len)

        result = {
            "max_iter": max_iter,
            "ppl": ppl,
            "avg_flops": avg_flops,
            "avg_iterations": avg_iters,
        }
        results.append(result)
        logger.info(f"  max_iter={max_iter}: PPL={ppl:.2f}, avg_iters={avg_iters:.2f}")

    # 恢复原始设置
    model.config.max_iter = original_max_iter
    return results


def measure_latency(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Dict[str, float]:
    """测量单次推理延迟

    Args:
        model: CCTLlamaModel 实例
        input_ids: [1, seq_len] 输入 token IDs
        attention_mask: [1, seq_len] 注意力掩码
        num_warmup: GPU 预热次数
        num_runs: 正式测量次数

    Returns:
        stats: {mean_ms, std_ms, p50_ms, p95_ms, tokens_per_sec}
    """
    model.eval()
    device = input_ids.device

    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_ids=input_ids, attention_mask=attention_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 正式测量
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            model(input_ids=input_ids, attention_mask=attention_mask)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    latencies_t = torch.tensor(latencies)
    seq_len = input_ids.shape[1]

    stats = {
        "mean_ms": latencies_t.mean().item(),
        "std_ms": latencies_t.std().item(),
        "p50_ms": latencies_t.median().item(),
        "p95_ms": latencies_t.quantile(0.95).item(),
        "tokens_per_sec": seq_len / (latencies_t.mean().item() / 1000),
    }

    logger.info(
        f"延迟: mean={stats['mean_ms']:.1f}ms, "
        f"p50={stats['p50_ms']:.1f}ms, p95={stats['p95_ms']:.1f}ms, "
        f"throughput={stats['tokens_per_sec']:.0f} tok/s"
    )
    return stats


def collect_iteration_stats(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    device: str = "cuda",
) -> EfficiencyResult:
    """收集模型在评测集上的循环迭代统计

    Args:
        model: CCTLlamaModel 实例
        eval_dataloader: 评测数据加载器
        device: 计算设备

    Returns:
        result: EfficiencyResult 包含迭代计数分布、均值、中位数等
    """
    model.eval()
    iteration_counts: List[int] = []
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="收集迭代统计"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )

            iteration_counts.append(outputs["num_iterations"])
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs["loss"].item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    iters_t = torch.tensor(iteration_counts, dtype=torch.float)

    result = EfficiencyResult(
        ppl=math.exp(min(avg_loss, 100.0)),
        avg_iterations=iters_t.mean().item(),
        median_iterations=iters_t.median().item(),
        iteration_counts=iteration_counts,
    )

    logger.info(
        f"迭代统计: avg={result.avg_iterations:.2f}, "
        f"median={result.median_iterations:.1f}, "
        f"range=[{min(iteration_counts)}, {max(iteration_counts)}], "
        f"PPL={result.ppl:.2f}"
    )
    return result


def main():
    """命令行效率评测入口"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="CCT 效率评测")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mode", choices=["ppl_flops", "latency", "iter_stats"], default="iter_stats")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    logger.info(f"效率评测模式: {args.mode}")
    logger.info(f"注意: 此脚本需要预先加载模型和数据，当前仅展示框架。")
    logger.info(f"请参考 evaluate_ppl_flops_curve / measure_latency / collect_iteration_stats 函数。")


if __name__ == "__main__":
    main()

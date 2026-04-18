# -*- coding: utf-8 -*-
"""CCT 可视化工具 — 误差收敛曲线、精度分布、Token 循环热力图

所有绘图函数返回 matplotlib Figure 对象，便于保存或嵌入 notebook。
"""

import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 支持中文显示 (如系统无中文字体则回退)
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_error_convergence(
    scores_per_iter: List[List[float]],
    title: str = "预测误差收敛曲线",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """绘制各循环轮次的预测误差 (1 - score) 收敛曲线

    每条线代表一个样本，x 轴为迭代轮次，y 轴为 1 - score (误差)。

    Args:
        scores_per_iter: 形状 [num_samples][num_iterations]，每个样本各轮的 score
        title: 图表标题
        save_path: 保存路径，None 则不保存

    Returns:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    num_samples = len(scores_per_iter)
    max_iters = max(len(s) for s in scores_per_iter)

    # 转为数组，短序列用 NaN 填充
    data = np.full((num_samples, max_iters), np.nan)
    for i, scores in enumerate(scores_per_iter):
        data[i, :len(scores)] = [1.0 - s for s in scores]

    # 均值 + 标准差带
    mean_err = np.nanmean(data, axis=0)
    std_err = np.nanstd(data, axis=0)
    iters = np.arange(1, max_iters + 1)

    ax.plot(iters, mean_err, color="steelblue", linewidth=2, label="平均误差")
    ax.fill_between(
        iters, mean_err - std_err, mean_err + std_err,
        alpha=0.2, color="steelblue", label="±1 标准差",
    )

    # 绘制部分个体轨迹 (最多20条)
    sample_idx = np.random.choice(num_samples, size=min(20, num_samples), replace=False)
    for idx in sample_idx:
        valid = ~np.isnan(data[idx])
        ax.plot(
            iters[valid], data[idx][valid],
            alpha=0.15, color="gray", linewidth=0.8,
        )

    ax.set_xlabel("循环迭代轮次")
    ax.set_ylabel("预测误差 (1 - score)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(1, max_iters)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_precision_distribution(
    scores: List[float],
    temperature: float = 0.5,
    title: str = "L6 精度分布",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """绘制 L6 Precision 分布直方图

    precision = 1 - sigmoid(score / τ)

    Args:
        scores: 所有 token 的 prediction score 列表
        temperature: L6 precision 温度参数 τ_p
        title: 图表标题
        save_path: 保存路径

    Returns:
        fig: matplotlib Figure
    """
    scores_arr = np.array(scores)
    precision = 1.0 - 1.0 / (1.0 + np.exp(-scores_arr / temperature))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: score 分布
    sns.histplot(scores_arr, bins=50, kde=True, ax=axes[0], color="coral")
    axes[0].set_xlabel("Prediction Score")
    axes[0].set_ylabel("频数")
    axes[0].set_title("Score 分布")

    # 右: precision 分布
    sns.histplot(precision, bins=50, kde=True, ax=axes[1], color="steelblue")
    axes[1].set_xlabel("Precision (注意力加权)")
    axes[1].set_ylabel("频数")
    axes[1].set_title(f"Precision 分布 (τ={temperature})")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_token_cycle_heatmap(
    iteration_map: np.ndarray,
    tokens: Optional[List[str]] = None,
    title: str = "Token 循环迭代热力图",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """绘制 Token 级循环迭代次数热力图

    Args:
        iteration_map: [num_samples, seq_len] 每个 token 的迭代次数
            (若模型按 batch 输出，则为单个 batch 的迭代次数)
        tokens: 可选的 token 文本列表（用于 x 轴标注）
        title: 图表标题
        save_path: 保存路径

    Returns:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, max(3, iteration_map.shape[0] * 0.5)))

    # 绘制热力图
    im = sns.heatmap(
        iteration_map,
        ax=ax,
        cmap="YlOrRd",
        vmin=1,
        annot=iteration_map.shape[1] <= 30,  # 短序列才标注数值
        fmt=".0f" if iteration_map.shape[1] <= 30 else "",
        cbar_kws={"label": "迭代次数"},
        linewidths=0.5 if iteration_map.shape[1] <= 50 else 0,
    )

    if tokens and len(tokens) <= 50:
        ax.set_xticks(np.arange(len(tokens)) + 0.5)
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xlabel("Token 位置")

    ax.set_ylabel("样本")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ppl_flops_curve(
    results: List[Dict[str, float]],
    baseline_ppl: Optional[float] = None,
    title: str = "PPL / FLOPs 曲线",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """绘制 PPL vs FLOPs 曲线

    Args:
        results: evaluate_ppl_flops_curve 的返回值列表
            每项含 {max_iter, ppl, avg_flops, avg_iterations}
        baseline_ppl: 基线模型 PPL（用于参考线）
        title: 图表标题
        save_path: 保存路径

    Returns:
        fig: matplotlib Figure
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    flops = [r["avg_flops"] for r in results]
    ppls = [r["ppl"] for r in results]
    iters = [r["avg_iterations"] for r in results]

    # 归一化 FLOPs 到 GFLOPs
    flops_g = [f / 1e9 for f in flops]

    ax1.plot(flops_g, ppls, "o-", color="steelblue", linewidth=2, markersize=8, label="CCT")

    # 标注 max_iter 值
    for r, x, y in zip(results, flops_g, ppls):
        ax1.annotate(
            f"K={r['max_iter']}\n(avg {r['avg_iterations']:.1f})",
            xy=(x, y), xytext=(5, 10),
            textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )

    if baseline_ppl is not None:
        ax1.axhline(y=baseline_ppl, color="red", linestyle="--", alpha=0.7, label="基线 PPL")

    ax1.set_xlabel("GFLOPs (Column 部分)")
    ax1.set_ylabel("困惑度 (PPL)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_iteration_distribution(
    iteration_counts: List[int],
    max_iter: int = 10,
    title: str = "循环迭代次数分布",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """绘制循环迭代次数分布柱状图

    Args:
        iteration_counts: 每个样本的迭代次数列表
        max_iter: 最大允许迭代次数
        title: 图表标题
        save_path: 保存路径

    Returns:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    counts = np.array(iteration_counts)
    bins = np.arange(0.5, max_iter + 1.5, 1)

    ax.hist(counts, bins=bins, color="steelblue", edgecolor="white", alpha=0.8, density=True)

    mean_val = counts.mean()
    median_val = np.median(counts)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"均值 = {mean_val:.2f}")
    ax.axvline(median_val, color="orange", linestyle=":", linewidth=1.5, label=f"中位数 = {median_val:.1f}")

    ax.set_xlabel("迭代次数")
    ax.set_ylabel("频率密度")
    ax.set_title(title)
    ax.set_xticks(range(1, max_iter + 1))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

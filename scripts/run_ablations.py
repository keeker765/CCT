# -*- coding: utf-8 -*-
"""CCT 消融实验批量运行脚本

定义消融配置并依次运行:
- A1: 无 L6 Precision (lambda_precision_init=0, 禁用精度调制)
- A3: 无 RotaryCycleEmbedding (phi=0, 禁用循环嵌入)
- A9: Rotation after Q/K (标记模式，用于与默认 before Q/K 对比)

使用方法:
    python scripts/run_ablations.py
    python scripts/run_ablations.py --device cuda --max_steps 5000
"""

import os
import sys
import copy
import subprocess
import argparse
import yaml
import logging
from typing import Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 消融实验定义
ABLATION_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "A1_no_precision",
        "description": "消融 A1: 禁用 L6 Precision (λ_precision=0)",
        "overrides": {
            "lambda_precision_init": 0.0,
            "output_dir": "output/ablation_A1_no_precision",
            "run_name": "cct-ablation-A1-no-precision",
        },
    },
    {
        "name": "A3_no_cycle_embed",
        "description": "消融 A3: 禁用 RotaryCycleEmbedding (φ=0)",
        "overrides": {
            "phi": 0.0,
            "output_dir": "output/ablation_A3_no_cycle_embed",
            "run_name": "cct-ablation-A3-no-cycle-embed",
        },
    },
    {
        "name": "A9_rotation_after_qk",
        "description": "消融 A9: Rotation after Q/K (标记模式，需模型层支持)",
        "overrides": {
            "rotation_mode": "after_qk",
            "output_dir": "output/ablation_A9_rotation_after_qk",
            "run_name": "cct-ablation-A9-rotation-after-qk",
        },
    },
]


def load_base_config(config_path: str) -> Dict[str, Any]:
    """加载基础配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_ablation_config(
    base_cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    output_path: str,
) -> str:
    """生成消融配置文件

    Args:
        base_cfg: 基础配置字典
        overrides: 覆盖参数
        output_path: 输出配置文件路径

    Returns:
        output_path: 写入的文件路径
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.update(overrides)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return output_path


def run_single_ablation(
    config_path: str,
    device: str,
    project_root: str,
) -> int:
    """运行单个消融实验

    Args:
        config_path: 消融配置文件路径
        device: 计算设备
        project_root: 项目根目录

    Returns:
        returncode: 子进程返回码
    """
    cmd = [
        sys.executable, "-m", "src.training.train",
        "--config", config_path,
        "--device", device,
    ]
    logger.info(f"  命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="CCT 消融实验批量运行")
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/base_config.yaml",
        help="基础配置文件路径",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="覆盖 max_steps (用于快速验证)",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        help="指定运行的消融实验名称 (默认全部运行)",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    base_cfg = load_base_config(args.base_config)

    # 筛选消融实验
    ablations = ABLATION_CONFIGS
    if args.ablations:
        ablations = [a for a in ablations if a["name"] in args.ablations]
        if not ablations:
            logger.error(f"未找到匹配的消融实验: {args.ablations}")
            sys.exit(1)

    logger.info(f"将运行 {len(ablations)} 个消融实验")
    results: Dict[str, int] = {}

    for ablation in ablations:
        logger.info(f"\n{'='*60}")
        logger.info(f"开始: {ablation['description']}")
        logger.info(f"{'='*60}")

        overrides = copy.deepcopy(ablation["overrides"])
        if args.max_steps is not None:
            overrides["max_steps"] = args.max_steps

        config_path = os.path.join(
            "configs", "ablation", f"{ablation['name']}.yaml"
        )
        write_ablation_config(base_cfg, overrides, config_path)
        logger.info(f"  配置已写入: {config_path}")

        returncode = run_single_ablation(config_path, args.device, project_root)
        results[ablation["name"]] = returncode

        if returncode != 0:
            logger.warning(f"  {ablation['name']} 失败 (returncode={returncode})")
        else:
            logger.info(f"  {ablation['name']} 完成")

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info("消融实验汇总:")
    for name, code in results.items():
        status = "✓ 成功" if code == 0 else f"✗ 失败 (code={code})"
        logger.info(f"  {name}: {status}")


if __name__ == "__main__":
    main()

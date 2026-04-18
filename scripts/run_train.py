# -*- coding: utf-8 -*-
"""CCT 训练启动脚本

使用方法:
    python scripts/run_train.py
    python scripts/run_train.py --config configs/base_config.yaml
    python scripts/run_train.py --config configs/ablation/A1_no_precision.yaml
"""

import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="CCT 训练启动器")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备 (cuda / cpu)",
    )
    args = parser.parse_args()

    # 确保从项目根目录运行
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "src.training.train",
        "--config", args.config,
        "--device", args.device,
    ]

    print(f"启动训练: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

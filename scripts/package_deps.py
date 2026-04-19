"""依赖打包脚本 — 下载 pip wheels 供 Kaggle 离线安装

下载最新版本的 transformers, accelerate, datasets 等关键依赖的 wheel 文件,
打包为 Kaggle Dataset 供无互联网环境使用.

用法:
    python scripts/package_deps.py              # 下载到 kaggle_wheels/
    python scripts/package_deps.py --upload     # 下载并上传到 Kaggle
    python scripts/package_deps.py --update     # 更新已有 Kaggle dataset
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

KAGGLE_USER = "wukeneth"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHEELS_DIR = PROJECT_ROOT / "kaggle_wheels"

# CCT 所需的关键依赖 (Kaggle 预装 torch, 只需补充)
PACKAGES = [
    "transformers>=4.48.0",
    "accelerate>=1.0.0",
    "datasets>=3.0.0",
    "sentencepiece>=0.2.0",
    "protobuf>=5.0.0",
    "safetensors>=0.4.0",
    "huggingface_hub>=0.27.0",
    "pyyaml>=6.0",
]


def download_wheels(target_platform: bool = False):
    """下载 pip wheels 到本地目录"""
    if WHEELS_DIR.exists():
        for f in WHEELS_DIR.iterdir():
            if f.suffix in ('.whl', '.gz', '.zip'):
                f.unlink()
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)

    print("下载 wheels 到 %s ..." % WHEELS_DIR)
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(WHEELS_DIR),
        "--only-binary", ":all:",
    ]

    if target_platform:
        # Kaggle 环境: Linux x86_64, Python 3.10
        cmd.extend([
            "--platform", "manylinux2014_x86_64",
            "--python-version", "310",
            "--implementation", "cp",
            "--abi", "cp310",
        ])
        print("  (目标平台: manylinux2014_x86_64, cp310)")

    cmd.extend(PACKAGES)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("pip download 失败:")
        print(result.stderr)
        # 某些包可能没有纯 wheel, 尝试不带 --only-binary
        cmd2 = [
            sys.executable, "-m", "pip", "download",
            "--dest", str(WHEELS_DIR),
        ] + PACKAGES
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            print("第二次尝试也失败:")
            print(result2.stderr)
            return False

    wheels = list(WHEELS_DIR.glob("*.whl")) + list(WHEELS_DIR.glob("*.tar.gz"))
    print("已下载 %d 个包:" % len(wheels))
    for w in sorted(wheels):
        size_mb = w.stat().st_size / 1e6
        print("  %s (%.1f MB)" % (w.name, size_mb))
    return True


def upload_to_kaggle(update: bool = False):
    """上传 wheels 到 Kaggle dataset"""
    slug = "cct-wheels"
    upload_dir = WHEELS_DIR

    # dataset metadata
    meta = {
        "title": "CCT Dependencies (pip wheels)",
        "id": f"{KAGGLE_USER}/{slug}",
        "licenses": [{"name": "MIT"}],
    }
    with open(upload_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 上传
    if update:
        cmd = ["kaggle", "datasets", "version", "-p", str(upload_dir),
               "-m", "update", "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(upload_dir),
               "--dir-mode", "zip"]

    print("上传: %s" % " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("上传成功!")
        print("Kaggle 路径: /kaggle/input/%s-%s/" % (KAGGLE_USER, slug))
    else:
        print("上传失败:")
        print(result.stdout)
        print(result.stderr)


def main():
    parser = argparse.ArgumentParser(description="打包 CCT 依赖为离线 wheels")
    parser.add_argument("--upload", action="store_true", help="下载并上传到 Kaggle")
    parser.add_argument("--update", action="store_true", help="更新已有 Kaggle dataset")
    parser.add_argument("--kaggle-platform", action="store_true",
                        help="下载 Kaggle 目标平台 (manylinux2014_x86_64, cp310) 的 wheels")
    args = parser.parse_args()

    success = download_wheels(target_platform=args.kaggle_platform)
    if not success:
        print("下载失败, 中止")
        return

    if args.upload or args.update:
        upload_to_kaggle(args.update)


if __name__ == "__main__":
    main()

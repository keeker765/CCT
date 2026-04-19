"""Kaggle 数据集上传脚本

上传两份数据集到 Kaggle (离线笔记本用):
1. cct-code: 项目源代码 (src/ 目录)
2. cct-data: OpenHermes 2.5 训练数据 (40k 子集)

用法:
    python scripts/upload_kaggle_datasets.py --code       # 仅上传代码
    python scripts/upload_kaggle_datasets.py --data       # 仅上传训练数据
    python scripts/upload_kaggle_datasets.py --all        # 上传全部
    python scripts/upload_kaggle_datasets.py --update     # 更新已有数据集
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
CODE_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "kaggle_upload"
MAX_SAMPLES = 40000


def upload_code(update: bool = False):
    """上传项目代码到 Kaggle dataset"""
    slug = "cct-code"
    upload_dir = DATA_DIR / slug
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)

    # dataset metadata
    meta = {
        "title": "CCT Source Code",
        "id": f"{KAGGLE_USER}/{slug}",
        "licenses": [{"name": "MIT"}],
    }
    with open(upload_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 复制 src/ 目录
    dst_src = upload_dir / "src"
    shutil.copytree(CODE_DIR, dst_src)
    print(f"[code] 已复制 src/ -> {dst_src}")

    # 上传
    _kaggle_upload(upload_dir, slug, update)


def upload_data(update: bool = False):
    """下载 OpenHermes 2.5 子集并上传到 Kaggle dataset"""
    slug = "cct-data"
    upload_dir = DATA_DIR / slug
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)

    # dataset metadata
    meta = {
        "title": "CCT Training Data (OpenHermes 2.5 40k)",
        "id": f"{KAGGLE_USER}/{slug}",
        "licenses": [{"name": "CC-BY-4.0"}],
    }
    with open(upload_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 下载数据
    print(f"[data] 下载 OpenHermes 2.5 (取前 {MAX_SAMPLES} 条)...")
    from datasets import load_dataset

    raw = load_dataset("teknium/OpenHermes-2.5", split="train")
    subset = raw.select(range(min(MAX_SAMPLES, len(raw))))

    out_file = upload_dir / "openhermes_40k.json"
    print(f"[data] 保存到 {out_file} ...")
    subset.to_json(str(out_file))
    print(f"[data] 已保存 {len(subset)} 条记录")

    # 上传
    _kaggle_upload(upload_dir, slug, update)


def _kaggle_upload(upload_dir: Path, slug: str, update: bool):
    """执行 kaggle datasets create / version"""
    cmd = ["kaggle", "datasets", "create", "-p", str(upload_dir), "--dir-mode", "zip"]
    if update:
        cmd = ["kaggle", "datasets", "version", "-p", str(upload_dir),
               "-m", "update", "--dir-mode", "zip"]

    print(f"[%s] 执行: %s" % (slug, " ".join(cmd)))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[{slug}] 上传成功!")
    else:
        print(f"[{slug}] 上传失败:")
        print(result.stdout)
        print(result.stderr)


def main():
    parser = argparse.ArgumentParser(description="上传数据集到 Kaggle")
    parser.add_argument("--code", action="store_true", help="仅上传代码")
    parser.add_argument("--data", action="store_true", help="仅上传训练数据")
    parser.add_argument("--all", action="store_true", help="上传全部")
    parser.add_argument("--update", action="store_true", help="更新已有数据集")
    args = parser.parse_args()

    if not args.code and not args.data and not args.all:
        parser.print_help()
        return

    if args.all or args.code:
        upload_code(args.update)
    if args.all or args.data:
        upload_data(args.update)

    print("\n完成! Kaggle 数据集路径:")
    print(f"  代码: /kaggle/input/{KAGGLE_USER}-cct-code/src/")
    print(f"  数据: /kaggle/input/{KAGGLE_USER}-cct-data/openhermes_40k.json")


if __name__ == "__main__":
    main()

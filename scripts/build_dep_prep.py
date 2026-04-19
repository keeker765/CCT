"""CCT 依赖打包 Notebook 构建器

在 Kaggle (Internet ON) 上下载所有需要的 pip 包,
打包成 tar.gz 供离线使用。

用法:
    python -m scripts.build_dep_prep
"""

import json
import argparse
from pathlib import Path
from typing import List

DEFAULT_OUTPUT = "notebooks/dep_prep_kaggle.ipynb"

# 主训练所需的包 (Kaggle 自带 torch, 只需升级 transformers 等)
PACKAGES = [
    "transformers>=4.48.0",
    "accelerate>=1.0.0",
    "datasets>=3.0.0",
    "sentencepiece",
    "safetensors",
]


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": _lines(source),
        "outputs": [],
        "execution_count": None,
    }


def _lines(text: str) -> List[str]:
    raw = text.strip("\n")
    parts = raw.split("\n")
    result = [line + "\n" for line in parts[:-1]]
    result.append(parts[-1])
    return result


def build_notebook() -> dict:
    pkg_list = " ".join(f"'{p}'" for p in PACKAGES)

    cells = [
        md("""\
# CCT 依赖打包

> **Internet ON**, 无需 GPU (用 CPU 即可)

下载所有 CCT 训练所需的 pip 包并打包为 tar.gz，之后上传为 Kaggle Dataset 供离线训练使用。"""),

        # ── 下载 wheels ──
        code(f"""\
import subprocess, sys, os, shutil

WORK = '/tmp/cct_wheels'
OUT = '/kaggle/working'

# 清理旧数据
if os.path.exists(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK, exist_ok=True)

PACKAGES = {PACKAGES}

print('下载依赖 wheels...')
for pkg in PACKAGES:
    print('  下载: %s' % pkg)
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'download',
         '--dest', WORK,
         '--only-binary=:all:',
         pkg],
        capture_output=True, text=True)
    if r.returncode != 0:
        print('    ⚠ binary 失败, 尝试允许 source...')
        subprocess.run(
            [sys.executable, '-m', 'pip', 'download',
             '--dest', WORK, pkg],
            capture_output=True, text=True)

wheels = os.listdir(WORK)
print('\\n下载完成: %d 个文件' % len(wheels))
for w in sorted(wheels):
    sz = os.path.getsize(os.path.join(WORK, w)) / 1024 / 1024
    print('  %s (%.1f MB)' % (w, sz))"""),

        # ── 打包 tar.gz ──
        code("""\
import tarfile

TAR_NAME = 'cct_wheels.tar.gz'
tar_path = os.path.join(OUT, TAR_NAME)

print('打包 %s ...' % tar_path)
with tarfile.open(tar_path, 'w:gz') as tar:
    for fname in sorted(os.listdir(WORK)):
        fpath = os.path.join(WORK, fname)
        tar.add(fpath, arcname=fname)

sz_mb = os.path.getsize(tar_path) / 1024 / 1024
print('完成: %s (%.1f MB)' % (tar_path, sz_mb))
print()

# 同时保留散装 wheels 目录供直接使用
wheels_dir = os.path.join(OUT, 'wheels')
if os.path.exists(wheels_dir):
    shutil.rmtree(wheels_dir)
shutil.copytree(WORK, wheels_dir)
print('散装 wheels: %s/' % wheels_dir)"""),

        # ── 验证 ──
        code("""\
# 验证: 用离线 wheels 做一次 dry-run 安装
import subprocess, sys

r = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--dry-run',
     '--no-index', '--find-links', wheels_dir] + PACKAGES,
    capture_output=True, text=True)

if r.returncode == 0:
    print('✅ 离线安装验证通过!')
    print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)
else:
    print('⚠ 验证失败:')
    print(r.stderr[-500:] if len(r.stderr) > 500 else r.stderr)

print()
print('=== 下一步 ===')
print('验证通过后运行下一个 cell 自动上传到 Kaggle')"""),

        # ── 上传到 Kaggle ──
        code("""\
# === 上传 wheels 到 Kaggle Dataset ===
import json, os, subprocess, sys

KAGGLE_USER = 'wukeneth'
DS_SLUG = 'cct-wheels'
UPLOAD_DIR = wheels_dir  # from previous cell

# 写 dataset-metadata.json
meta = {
    'title': 'CCT Wheels',
    'id': '%s/%s' % (KAGGLE_USER, DS_SLUG),
    'licenses': [{'name': 'other'}],
}
meta_path = os.path.join(UPLOAD_DIR, 'dataset-metadata.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print('metadata 已写入: %s' % meta_path)

# 检查 dataset 是否已存在
r = subprocess.run(
    ['kaggle', 'datasets', 'list', '-m', '--search', DS_SLUG],
    capture_output=True, text=True)
exists = DS_SLUG in r.stdout

if exists:
    print('Dataset 已存在, 创建新版本...')
    cmd = ['kaggle', 'datasets', 'version', '-p', UPLOAD_DIR,
           '-m', 'auto update wheels', '--dir-mode', 'zip']
else:
    print('创建新 Dataset...')
    cmd = ['kaggle', 'datasets', 'create', '-p', UPLOAD_DIR, '--dir-mode', 'zip']

r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print('ERROR:', r.stderr)
else:
    print('✅ 上传完成! Dataset: https://www.kaggle.com/datasets/%s/%s' % (KAGGLE_USER, DS_SLUG))"""),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description="CCT 依赖打包 Notebook")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    nb = build_notebook()
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("依赖打包 notebook 已生成: %s (%d cells)" % (p.resolve(), len(nb["cells"])))


if __name__ == "__main__":
    main()

"""Kaggle 数据集上传 Colab Notebook 构建器

在 Colab 上运行:
  1. 上传 kaggle.json 认证
  2. 下载 OpenHermes 2.5 40k 子集
  3. 打包 CCT 源代码
  4. 通过 Kaggle API 上传 cct-code + cct-data

用法:
    python -m src.build_notebook_upload
    python -m src.build_notebook_upload -o notebooks/upload_kaggle.ipynb
"""

import json
import argparse
from pathlib import Path
from typing import List

GH_REPO = "keeker765/CCT"
KAGGLE_USER = "wukeneth"
DEFAULT_OUTPUT = "notebooks/upload_kaggle.ipynb"
VERSION = "upload-v1"


# ── Cell 构造器 ───────────────────────────────────────
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


# ── Notebook Cells ────────────────────────────────────
def cells_header() -> List[dict]:
    return [
        md(f"""\
# CCT — 上传 Kaggle 数据集 ({VERSION})

**目的**: 在 Colab 上准备并上传 CCT 项目所需的 Kaggle Datasets

上传内容:
1. **`{KAGGLE_USER}/cct-code`** — 项目源代码 (`src/` 目录)
2. **`{KAGGLE_USER}/cct-data`** — OpenHermes 2.5 训练数据 (40k 子集)

**前置**: 准备好 `kaggle.json` (从 kaggle.com → Account → Create API Token 下载)"""),
    ]


def cells_kaggle_auth() -> List[dict]:
    return [
        md("## 0. Kaggle API 认证"),
        code(f"""\
# === 0. 硬编码 kaggle.json 认证 ===
import os, json

KAGGLE_CREDS = {{"username": "{KAGGLE_USER}", "key": "51c30f09c0d8ec8e5884a4c16b49a0a5"}}

kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_path = os.path.join(kaggle_dir, 'kaggle.json')
with open(kaggle_path, 'w') as f:
    json.dump(KAGGLE_CREDS, f)
os.chmod(kaggle_path, 0o600)
print('kaggle.json 已写入 %s' % kaggle_path)"""),
        code("""\
# === 安装 kaggle CLI ===
!pip install -q kaggle datasets transformers

import kaggle
print('Kaggle API 就绪, 用户: %s' % kaggle.api.get_config_value('username'))"""),
    ]


def cells_clone_code() -> List[dict]:
    return [
        md("## 1. 准备代码数据集 (cct-code)"),
        code(f"""\
# === 1. 克隆 CCT 仓库 ===
import os, subprocess, shutil, json

WORK_DIR = '/content/CCT'
KAGGLE_USER = '{KAGGLE_USER}'

if os.path.exists(WORK_DIR):
    subprocess.run(['git', '-C', WORK_DIR, 'pull'], check=True)
else:
    subprocess.run(['git', 'clone', 'https://github.com/{GH_REPO}.git', WORK_DIR], check=True)

print('代码目录: %s' % WORK_DIR)
print('src/ 文件:')
for root, dirs, fs in os.walk(os.path.join(WORK_DIR, 'src')):
    level = root.replace(os.path.join(WORK_DIR, 'src'), '').count(os.sep)
    indent = '  ' * level
    print('%s%s/' % (indent, os.path.basename(root)))
    for f in sorted(fs):
        if f.endswith('.py'):
            print('%s  %s' % (indent, f))"""),
        code(f"""\
# === 打包 cct-code 数据集 ===
slug_code = 'cct-code'
upload_dir_code = '/content/upload_code'

if os.path.exists(upload_dir_code):
    shutil.rmtree(upload_dir_code)
os.makedirs(upload_dir_code, exist_ok=True)

# dataset-metadata.json
meta_code = {{
    'title': 'CCT Source Code',
    'id': '%s/%s' % (KAGGLE_USER, slug_code),
    'licenses': [{{'name': 'MIT'}}],
}}
with open(os.path.join(upload_dir_code, 'dataset-metadata.json'), 'w') as f:
    json.dump(meta_code, f, indent=2)

# 复制 src/
shutil.copytree(os.path.join(WORK_DIR, 'src'),
                os.path.join(upload_dir_code, 'src'))
print('[cct-code] 已准备: %s' % upload_dir_code)
!ls -la {{upload_dir_code}}"""),
    ]


def cells_prepare_data() -> List[dict]:
    return [
        md("## 2. 准备训练数据集 (cct-data)"),
        code(f"""\
# === 2. 下载 OpenHermes 2.5 并取 40k 子集 ===
from datasets import load_dataset

MAX_SAMPLES = 40000
slug_data = 'cct-data'
upload_dir_data = '/content/upload_data'

print('下载 OpenHermes 2.5 (取前 %d 条)...' % MAX_SAMPLES)
raw = load_dataset('teknium/OpenHermes-2.5', split='train')
subset = raw.select(range(min(MAX_SAMPLES, len(raw))))

# 只保留 conversations 列, 避免 null 字段导致 load 时 schema 冲突
keep_cols = ['conversations']
drop_cols = [c for c in subset.column_names if c not in keep_cols]
if drop_cols:
    subset = subset.remove_columns(drop_cols)
    print('已移除多余列: %s' % drop_cols)

if os.path.exists(upload_dir_data):
    shutil.rmtree(upload_dir_data)
os.makedirs(upload_dir_data, exist_ok=True)

# dataset-metadata.json
meta_data = {{
    'title': 'CCT Training Data (OpenHermes 2.5 40k)',
    'id': '%s/%s' % (KAGGLE_USER, slug_data),
    'licenses': [{{'name': 'CC-BY-4.0'}}],
}}
with open(os.path.join(upload_dir_data, 'dataset-metadata.json'), 'w') as f:
    json.dump(meta_data, f, indent=2)

out_file = os.path.join(upload_dir_data, 'openhermes_40k.json')
subset.to_json(out_file)
fsize = os.path.getsize(out_file) / 1024 / 1024
print('[cct-data] 已保存 %d 条 -> %s (%.1f MB)' % (len(subset), out_file, fsize))"""),
    ]


def cells_upload() -> List[dict]:
    return [
        md("## 3. 上传到 Kaggle"),
        code("""\
# === 3a. 上传 cct-code ===
import subprocess

def kaggle_upload(upload_dir, slug, update=False):
    if update:
        cmd = ['kaggle', 'datasets', 'version', '-p', upload_dir,
               '-m', 'auto update', '--dir-mode', 'zip']
    else:
        cmd = ['kaggle', 'datasets', 'create', '-p', upload_dir, '--dir-mode', 'zip']
    print('[%s] %s' % (slug, ' '.join(cmd)))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        if 'already exists' in r.stderr.lower() or 'already exists' in r.stdout.lower():
            print('[%s] 数据集已存在, 改用 version 更新...' % slug)
            return kaggle_upload(upload_dir, slug, update=True)
        print('[%s] 错误: %s' % (slug, r.stderr))
        return False
    return True

print('=== 上传 cct-code ===')
kaggle_upload(upload_dir_code, slug_code)"""),
        code("""\
# === 3b. 上传 cct-data ===
print('=== 上传 cct-data ===')
kaggle_upload(upload_dir_data, slug_data)"""),
    ]


def cells_verify() -> List[dict]:
    return [
        md("## 4. 验证"),
        code(f"""\
# === 4. 验证上传结果 ===
print('=== Kaggle Datasets 列表 ===')
!kaggle datasets list -m --user {KAGGLE_USER}

print()
print('上传完成! 在 Kaggle Notebook 中添加:')
print('  Dataset: {KAGGLE_USER}/cct-code')
print('  Dataset: {KAGGLE_USER}/cct-data')
print('  Model: meta-llama/Llama-3.2-1B')"""),
    ]


# ── 组装 Notebook ─────────────────────────────────────
def build_notebook() -> dict:
    cells = []
    for fn in [
        cells_header,
        cells_kaggle_auth,
        cells_clone_code,
        cells_prepare_data,
        cells_upload,
        cells_verify,
    ]:
        cells.extend(fn())

    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def save_notebook(output_path: str) -> str:
    nb = build_notebook()
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return str(p.resolve())


def main():
    parser = argparse.ArgumentParser(description="构建 Kaggle 上传 Notebook")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="输出 .ipynb 路径")
    args = parser.parse_args()

    path = save_notebook(args.output)
    print("Upload notebook 已生成: %s" % path)
    print("   %d cells, 版本 %s" % (len(build_notebook()["cells"]), VERSION))


if __name__ == "__main__":
    main()

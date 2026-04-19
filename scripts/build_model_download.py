"""构建 Kaggle Notebook: 下载 Llama 3.2 1B Base 并上传到 Kaggle Dataset

用法:
    python -m scripts.build_model_download
"""

import json
import argparse
from pathlib import Path
from typing import List

DEFAULT_OUTPUT = "notebooks/model_download_kaggle.ipynb"
KAGGLE_USER = "wukeneth"
DATASET_NAME = "llama-3-2-1b-base"
HF_MODEL_ID = "unsloth/Llama-3.2-1B"


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
    cells = [
        md(f"""\
# 下载 Llama 3.2 1B Base (Unsloth) → 上传 Kaggle Dataset

> ⚠️ **需要**: Internet ON

**流程**: HuggingFace `{HF_MODEL_ID}` → `/tmp/llama-base` → `{KAGGLE_USER}/{DATASET_NAME}`

Unsloth 版本**无需 Meta 许可**，可直接下载。"""),

        code("""\
!pip install -q huggingface_hub kaggle"""),

        code(f"""\
import os

MODEL_ID = '{HF_MODEL_ID}'
SAVE_DIR = '/tmp/llama-base'
os.makedirs(SAVE_DIR, exist_ok=True)
print('Model: %s' % MODEL_ID)
print('Save:  %s' % SAVE_DIR)

# Unsloth 版本无需 token, 但如果有可以加速
hf_token = os.environ.get('HF_TOKEN', None)
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    hf_token = secrets.get_secret("HF_TOKEN")
except Exception:
    pass
print('HF Token: %s' % ('available' if hf_token else 'not set (OK for unsloth)'))"""),

        code("""\
# === 下载模型 ===
from huggingface_hub import snapshot_download
import time

print('开始下载 (约 2-5 分钟)...')
t0 = time.time()

path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=SAVE_DIR,
    token=hf_token,
    ignore_patterns=['*.gguf', '*.bin', 'original/*', '.gitattributes'],
)

elapsed = time.time() - t0
print('下载完成! %.1f 秒' % elapsed)
print('路径: %s' % path)

# 显示文件
total_size = 0
for root, dirs, files in os.walk(SAVE_DIR):
    for f in files:
        fp = os.path.join(root, f)
        sz = os.path.getsize(fp)
        total_size += sz
        rel = os.path.relpath(fp, SAVE_DIR)
        print('  %s (%.1f MB)' % (rel, sz / 1e6))
print('\\n总大小: %.1f GB' % (total_size / 1e9))"""),

        code("""\
# === 验证模型可加载 ===
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

config = AutoConfig.from_pretrained(SAVE_DIR)
print('Model config:')
print('  hidden_size: %d' % config.hidden_size)
print('  num_layers:  %d' % config.num_hidden_layers)
print('  vocab_size:  %d' % config.vocab_size)

tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
print('\\nTokenizer vocab: %d' % len(tokenizer))

# 快速加载验证 (不加载全部权重到 GPU)
model = AutoModelForCausalLM.from_pretrained(
    SAVE_DIR, torch_dtype=torch.float16, device_map='cpu'
)
n_params = sum(p.numel() for p in model.parameters())
print('Parameters: %.2fB' % (n_params / 1e9))
del model
import gc; gc.collect()
print('\\n模型验证 ✓')"""),

        code(f"""\
# === 上传到 Kaggle Dataset ===
import json, subprocess

KAGGLE_USER = '{KAGGLE_USER}'
DATASET_NAME = '{DATASET_NAME}'
DATASET_ID = '%s/%s' % (KAGGLE_USER, DATASET_NAME)

# 创建 metadata
metadata = {{
    'title': DATASET_NAME,
    'id': DATASET_ID,
    'licenses': [{{'name': 'other'}}],
}}
meta_path = os.path.join(SAVE_DIR, 'dataset-metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print('Metadata written: %s' % meta_path)

# 检查 dataset 是否已存在
result = subprocess.run(
    ['kaggle', 'datasets', 'list', '--mine', '--search', DATASET_NAME],
    capture_output=True, text=True
)
exists = DATASET_NAME in result.stdout

if exists:
    print('Dataset 已存在, 创建新版本...')
    cmd = ['kaggle', 'datasets', 'version', '-p', SAVE_DIR,
           '-m', 'Llama 3.2 1B Base weights', '--dir-mode', 'zip']
else:
    print('Dataset 不存在, 首次创建...')
    cmd = ['kaggle', 'datasets', 'create', '-p', SAVE_DIR, '--dir-mode', 'zip']

print('Running: %s' % ' '.join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print('STDERR:', result.stderr)
    # 如果 create 失败 (已存在), 尝试 version
    if 'already exists' in result.stderr.lower() or 'already exists' in result.stdout.lower():
        print('Retrying as version update...')
        cmd = ['kaggle', 'datasets', 'version', '-p', SAVE_DIR,
               '-m', 'Llama 3.2 1B Base weights', '--dir-mode', 'zip']
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr: print('STDERR:', result.stderr)

print('\\n完成! 检查: https://www.kaggle.com/datasets/%s' % DATASET_ID)"""),
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
    parser = argparse.ArgumentParser(description="构建模型下载 Notebook")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    nb = build_notebook()
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Notebook 已生成: %s (%d cells)" % (p.resolve(), len(nb["cells"])))


if __name__ == "__main__":
    main()

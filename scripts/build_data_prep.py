"""数据准备 Kaggle Notebook 构建器

在 Kaggle 上运行 (需开启 Internet):
  1. 流式下载 HuggingFaceFW 50/30/20 混合数据 (已 shuffle)
  2. Tokenize + Pack 成固定 seq_len chunks
  3. 上传到 Kaggle dataset: wukeneth/cct-pretrain-data

数据源:
  HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled
  = 50% FinePDFs + 30% DCLM + 20% FineWeb-Edu (全局 shuffle, seed=42)
  来源: codelion "The 1 Billion Token Challenge" (2025)

输出:
  - train_packed/  (N 个 .pt 文件, 每个 = 1 optimizer step 的数据)
  - eval_packed/   (固定 eval chunks)
  - metadata.json  (seq_len, batch_size, num_chunks 等)

用法:
    python -m scripts.build_data_prep
    python -m scripts.build_data_prep -o notebooks/data_prep_kaggle.ipynb
"""

import json
import argparse
from pathlib import Path
from typing import List

KAGGLE_USER = "wukeneth"
DEFAULT_OUTPUT = "notebooks/data_prep_kaggle.ipynb"


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


def cells_header() -> List[dict]:
    return [
        md(f"""\
# CCT 数据准备 — 流式取 1B tokens → Pack → 上传 Kaggle

> ⚠️ **必须开启 Internet**: Settings → Internet → On

**目的**: 从 HuggingFace 流式下载 50/30/20 混合数据 → Pack → 上传 Kaggle Dataset

**数据源**: `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled`
- 50% FinePDFs (学术 PDF) + 30% DCLM (web) + 20% FineWeb-Edu (教育)
- 已全局 shuffle (seed=42), 100B tokens, 我们只取 ~1B
- 来源: codelion "The 1 Billion Token Challenge" (2025)

**输出**: 上传到 `{KAGGLE_USER}/cct-pretrain-data`"""),
    ]


def cells_install() -> List[dict]:
    return [
        md("## 0. 安装依赖 + Kaggle 认证"),
        code("""\
!pip install -q datasets transformers accelerate"""),
        code(f"""\
# === Kaggle API 认证 ===
import os, json

# 方式 1: 环境变量 (优先)
os.environ['KAGGLE_USERNAME'] = '{KAGGLE_USER}'
os.environ['KAGGLE_KEY'] = '51c30f09c0d8ec8e5884a4c16b49a0a5'

# 方式 2: 同时写入文件 (兜底)
for kaggle_dir in [os.path.expanduser('~/.kaggle'), '/root/.kaggle', '/home/jupyter/.kaggle']:
    try:
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_path = os.path.join(kaggle_dir, 'kaggle.json')
        with open(kaggle_path, 'w') as f:
            json.dump({{'username': '{KAGGLE_USER}', 'key': '51c30f09c0d8ec8e5884a4c16b49a0a5'}}, f)
        os.chmod(kaggle_path, 0o600)
        print('kaggle.json → %s' % kaggle_path)
    except Exception as e:
        print('Skip %s: %s' % (kaggle_dir, e))

# 验证 API
import subprocess
r = subprocess.run(['kaggle', 'config', 'view'],
                   capture_output=True, text=True)
if r.returncode == 0 and 'username' in r.stdout:
    print('API 验证: ✅ OK')
    print(r.stdout.strip())
else:
    print('API 验证: ❌ FAIL')
    print('stdout:', r.stdout.strip())
    print('stderr:', r.stderr.strip())"""),
    ]


def cells_config() -> List[dict]:
    return [
        md("## 1. 配置"),
        code("""\
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import IterableDataset

MODEL_NAME = "unsloth/Llama-3.2-1B"

# === Packing 参数 (与训练 notebook 对齐) ===
SEQ_LEN = 2048
BATCH_SIZE = 32
GRAD_ACCUM = 4
MAX_STEPS = 3800
EVAL_CHUNKS = 100

# 数据源: HuggingFaceFW 官方 50/30/20 混合 (已 shuffle)
DS_NAME = 'HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled'

# 计算需要的总 chunk 数
total_chunks_needed = MAX_STEPS * GRAD_ACCUM * BATCH_SIZE
total_tokens_needed = total_chunks_needed * SEQ_LEN
print('Seq len: %d' % SEQ_LEN)
print('Batch size: %d, Grad accum: %d' % (BATCH_SIZE, GRAD_ACCUM))
print('Max steps: %d' % MAX_STEPS)
print('Total chunks needed: %d (%.1fM tokens ≈ %.2fB tokens)' % (
    total_chunks_needed, total_tokens_needed / 1e6, total_tokens_needed / 1e9))
print()
print('Data source: %s' % DS_NAME)"""),
    ]


def cells_download_pack() -> List[dict]:
    return [
        md("## 2. 流式下载 + Packing (批量 tokenize 加速)"),
        code("""\
# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print('Vocab: %d, EOS: %d' % (tokenizer.vocab_size, tokenizer.eos_token_id))

# === Packing Dataset (批量 tokenize, ~2-3x 加速) ===
class PackedDataset(IterableDataset):
    \"\"\"流式 tokenize + pack: 批量 tokenize (512 docs) → 拼接 EOS → 切 seq_len 块\"\"\"
    def __init__(self, hf_dataset, tokenizer, seq_len=2048, batch_docs=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_docs = batch_docs

    def __iter__(self):
        buffer = []
        batch_texts = []
        eos = self.tokenizer.eos_token_id

        def flush():
            if not batch_texts:
                return
            enc = self.tokenizer(batch_texts, add_special_tokens=False,
                                 return_attention_mask=False)
            for ids in enc['input_ids']:
                buffer.extend(ids)
                buffer.append(eos)
            batch_texts.clear()

        def pop_chunks():
            while len(buffer) >= self.seq_len:
                t = torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                del buffer[:self.seq_len]
                yield {
                    'input_ids': t,
                    'attention_mask': torch.ones_like(t),
                    'labels': t.clone(),
                }

        for example in self.dataset:
            text = example.get('text', '')
            if not text or not text.strip():
                continue
            batch_texts.append(text)
            if len(batch_texts) >= self.batch_docs:
                flush()
                yield from pop_chunks()

        flush()
        yield from pop_chunks()

# === 流式数据 (单源, 已混合+shuffle) ===
print('Loading streaming dataset: %s' % DS_NAME)
ds_train = load_dataset(DS_NAME, split='train', streaming=True)
ds_eval = load_dataset(DS_NAME, split='train', streaming=True)

# Eval 跳过前 N 条避免与 train 重叠
EVAL_SKIP = 500000
ds_eval = ds_eval.skip(EVAL_SKIP)

# Eval: 收集固定 chunks
print('Packing eval chunks (%d) (skip first %d docs)...' % (EVAL_CHUNKS, EVAL_SKIP))
eval_packed = PackedDataset(ds_eval, tokenizer, seq_len=SEQ_LEN)
eval_data = []
for item in eval_packed:
    eval_data.append(item)
    if len(eval_data) >= EVAL_CHUNKS:
        break
print('Eval: %d chunks collected' % len(eval_data))"""),
    ]


def cells_save() -> List[dict]:
    return [
        md("## 3. 保存 Pre-packed Data"),
        code("""\
# === 保存到 /tmp (working 空间不足) ===
import os, json, time, threading, queue

WORK = '/tmp/cct_data'
train_dir = os.path.join(WORK, 'train_packed')
eval_dir = os.path.join(WORK, 'eval_packed')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# 保存 eval
torch.save(eval_data, os.path.join(eval_dir, 'eval_chunks.pt'))
print('Eval saved: %d chunks' % len(eval_data))

# === 异步写入线程 (tokenize 和 I/O 并行) ===
save_queue = queue.Queue(maxsize=8)
save_errors = []

def _writer():
    while True:
        item = save_queue.get()
        if item is None:
            break
        path, data = item
        try:
            torch.save(data, path)
        except Exception as e:
            save_errors.append(str(e))
        save_queue.task_done()

writer_thread = threading.Thread(target=_writer, daemon=True)
writer_thread.start()

# 保存 train: 按 optimizer step 打包
print('Packing and saving train data (async I/O)...')
train_packed = PackedDataset(ds_train, tokenizer, seq_len=SEQ_LEN)
train_iter = iter(train_packed)

file_idx = 0
total_saved = 0
chunks_per_file = BATCH_SIZE  # 每文件 = 1 micro-batch, 训练时 100% 利用
t0 = time.time()

while total_saved < total_chunks_needed:
    batch_data = []
    for _ in range(chunks_per_file):
        try:
            item = next(train_iter)
            batch_data.append(item)
        except StopIteration:
            break
    if not batch_data:
        break
    save_queue.put((os.path.join(train_dir, 'step_%05d.pt' % file_idx), batch_data))
    total_saved += len(batch_data)
    file_idx += 1
    if file_idx % 100 == 0:
        elapsed = time.time() - t0
        tokens_done = total_saved * SEQ_LEN
        speed = tokens_done / elapsed if elapsed > 0 else 0
        print('  [%d files] %d chunks (%.1fM tokens) | %.0f tok/s | %.1f min' % (
            file_idx, total_saved, tokens_done / 1e6, speed, elapsed / 60))

# 等待写入完成
save_queue.put(None)
writer_thread.join()

if save_errors:
    print('WARNING: %d save errors: %s' % (len(save_errors), save_errors[:3]))

elapsed = time.time() - t0
print('Train saved: %d files, %d chunks (%.2fB tokens) in %.1f min' % (
    file_idx, total_saved, total_saved * SEQ_LEN / 1e9, elapsed / 60))

# 保存 metadata
metadata = {
    'seq_len': SEQ_LEN,
    'batch_size': BATCH_SIZE,
    'grad_accum': GRAD_ACCUM,
    'max_steps': MAX_STEPS,
    'eval_chunks': EVAL_CHUNKS,
    'total_train_chunks': total_saved,
    'total_train_tokens': total_saved * SEQ_LEN,
    'num_train_files': file_idx,
    'data_source': DS_NAME,
}
with open(os.path.join(WORK, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print('Metadata saved')

# 统计磁盘使用
total_size = 0
for dirpath, dirnames, filenames in os.walk(WORK):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        if os.path.isfile(fp):
            total_size += os.path.getsize(fp)
print('Total output: %.2f GB' % (total_size / 1e9))"""),
    ]


def cells_upload_kaggle() -> List[dict]:
    return [
        md("## 4. 上传到 Kaggle Dataset"),
        code(f"""\
# === 准备上传目录 ===
import subprocess, shutil

SLUG = 'cct-pretrain-data'
KAGGLE_USER = '{KAGGLE_USER}'
upload_dir = '/tmp/upload_pretrain'

if os.path.exists(upload_dir):
    shutil.rmtree(upload_dir)
os.makedirs(upload_dir, exist_ok=True)

# dataset-metadata.json (Kaggle 要求)
meta = {{
    'title': 'CCT Pretrain Data (50/30/20 packed)',
    'id': '%s/%s' % (KAGGLE_USER, SLUG),
    'licenses': [{{'name': 'Apache 2.0'}}],
}}
with open(os.path.join(upload_dir, 'dataset-metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)

# 移动 (非复制, 节省磁盘) packed 数据到上传目录
shutil.move(train_dir, os.path.join(upload_dir, 'train_packed'))
shutil.move(eval_dir, os.path.join(upload_dir, 'eval_packed'))
shutil.move(os.path.join(WORK, 'metadata.json'), os.path.join(upload_dir, 'metadata.json'))

# 统计大小
total_size = 0
for dirpath, dirnames, filenames in os.walk(upload_dir):
    for f in filenames:
        total_size += os.path.getsize(os.path.join(dirpath, f))
print('Upload dir: %s (%.2f GB)' % (upload_dir, total_size / 1e9))"""),
        code("""\
# === 上传到 Kaggle ===
import subprocess

def kaggle_upload(upload_dir, slug, update=False):
    if update:
        cmd = ['kaggle', 'datasets', 'version', '-p', upload_dir,
               '-m', 'auto update', '--dir-mode', 'zip']
    else:
        cmd = ['kaggle', 'datasets', 'create', '-p', upload_dir, '--dir-mode', 'zip']
    print('[%s] %s' % (slug, ' '.join(cmd)))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    print(r.stdout)
    if r.returncode != 0:
        if 'already exists' in r.stderr.lower() or 'already exists' in r.stdout.lower():
            print('[%s] 数据集已存在, 改用 version 更新...' % slug)
            return kaggle_upload(upload_dir, slug, update=True)
        print('[%s] 错误: %s' % (slug, r.stderr))
        return False
    return True

print('=== 上传 %s ===' % SLUG)
kaggle_upload(upload_dir, SLUG)"""),
    ]


def cells_verify() -> List[dict]:
    return [
        md("## 5. 验证"),
        code(f"""\
# === 验证上传 ===
print('=== Kaggle Datasets ===')
!kaggle datasets list -m --user {KAGGLE_USER}

print()
print('✅ 上传完成!')
print()
print('在训练 Kaggle Notebook 中添加 Dataset:')
print('  {KAGGLE_USER}/cct-pretrain-data')
print()
print('数据格式:')
print('  train_packed/step_NNNNN.pt  — 每个含 %d chunks' % BATCH_SIZE)
print('  eval_packed/eval_chunks.pt  — %d eval chunks' % EVAL_CHUNKS)
print('  metadata.json               — 配置信息')"""),
    ]


def build_notebook() -> dict:
    cells = []
    for fn in [cells_header, cells_install, cells_config,
               cells_download_pack, cells_save, cells_upload_kaggle,
               cells_verify]:
        cells.extend(fn())

    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "kaggle": {
                "accelerator": "none",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
            },
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
    parser = argparse.ArgumentParser(description="构建数据准备 Colab Notebook")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    path = save_notebook(args.output)
    print("Data prep notebook 已生成: %s" % path)
    print("   %d cells (Kaggle 版)" % len(build_notebook()["cells"]))


if __name__ == "__main__":
    main()

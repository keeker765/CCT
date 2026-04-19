"""CCT 消融实验 Baseline 2 — Kaggle Notebook 构建器 (离线版)

Kaggle 环境无互联网, 所有依赖通过 Kaggle Datasets / Models 挂载:
  - 代码: /kaggle/input/cct-code/src/
  - 数据: /kaggle/input/cct-data/openhermes_40k.json
  - 模型: /kaggle/input/ (用户在 Kaggle UI 中添加 meta-llama/Llama-3.2-1B)

前置步骤:
  1. python scripts/upload_kaggle_datasets.py --all
  2. 在 Kaggle Notebook 设置中添加:
     - Dataset: wukeneth/cct-code
     - Dataset: wukeneth/cct-data
     - Model: meta-llama/Llama-3.2-1B (从 Kaggle Models 添加)
  3. 开启 GPU (P100 或 T4)

用法:
    python -m src.build_notebook_kaggle
    python -m src.build_notebook_kaggle -o notebooks/ablation_kaggle.ipynb
"""

import json
import argparse
import glob as glob_mod
from pathlib import Path
from typing import List

# ── 常量 ──────────────────────────────────────────────
KAGGLE_USER = "wukeneth"
DEFAULT_OUTPUT = "notebooks/ablation_kaggle.ipynb"
VERSION = "ablation-b2-kaggle"

# Kaggle 挂载路径 (运行时自动探测)
KAGGLE_INPUT = "/kaggle/input"

SELECTED_LAYERS = [0, 1, 2, 7, 12, 14, 15]
NUM_LAYERS = len(SELECTED_LAYERS)


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
        md("""\
# CCT 消融实验 — Baseline 2: 无循环 7 层顺序前向 (Kaggle 离线版)

**Kaggle 依赖 (在右侧 Add Data 中添加)**:
1. **Dataset**: `wukeneth/cct-code` — 项目源代码
2. **Dataset**: `wukeneth/cct-data` — OpenHermes 2.5 40k 子集
3. **Model**: `meta-llama/Llama-3.2-1B` — 基座模型 (从 Kaggle Models 添加)

**架构**: 从 Llama-3.2-1B 中取 7 层 [0,1,2,7,12,14,15], 顺序拼接, 无循环
**目的**: 与 CCT 对比, 验证 Column 循环机制的增益

**Settings**: GPU P100 or T4, Internet OFF"""),
    ]


def cells_setup() -> List[dict]:
    return [
        md("## 0. 环境检查 + 路径自动探测"),
        code(f"""\
# === 0. 自动探测 Kaggle 挂载路径 ===
import os, sys

def find_path(keyword, file_hint=None):
    \"\"\"在 /kaggle/input 下递归搜索含 keyword 的目录\"\"\"
    for root, dirs, files in os.walk('/kaggle/input'):
        dirname = os.path.basename(root).lower()
        if keyword in dirname:
            if file_hint is None or file_hint in files:
                return root
    return None

# 1. 探测代码路径
CODE_DIR = find_path('cct-code')
if CODE_DIR is None:
    # 备选: 直接搜索含 wrapped_model.py 的目录
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'wrapped_model.py' in files:
            CODE_DIR = root
            break
assert CODE_DIR is not None, (
    '代码数据集未挂载! 请在 Kaggle 添加 wukeneth/cct-code dataset')
# CODE_DIR 指向 src/ 的父目录或 src/ 本身
if os.path.exists(os.path.join(CODE_DIR, 'src')):
    CODE_DIR = os.path.join(CODE_DIR, 'src')
print('代码: %s' % CODE_DIR)

# 2. 探测训练数据路径
DATA_FILE = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'openhermes_40k.json' in files:
        DATA_FILE = os.path.join(root, 'openhermes_40k.json')
        break
assert DATA_FILE is not None, (
    '训练数据未挂载! 请在 Kaggle 添加 wukeneth/cct-data dataset')
print('数据: %s' % DATA_FILE)

# 3. 探测模型路径 (Llama-3.2-1B)
MODEL_DIR = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'config.json' in files and 'llama' in root.lower():
        MODEL_DIR = root
        break
assert MODEL_DIR is not None, (
    'Llama 模型未挂载! 请在 Kaggle 添加 meta-llama/Llama-3.2-1B')
print('模型: %s' % MODEL_DIR)

# 环境信息
import torch, transformers
print('\\nPyTorch: %s' % torch.__version__)
print('Transformers: %s' % transformers.__version__)
print('GPU: %s' % torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"""),
    ]


_CHAT_TEMPLATE = (
    '{% for message in messages %}'
    '{% if message["role"] == "user" %}'
    '<|start_header_id|>user<|end_header_id|>\\n\\n'
    '{{ message["content"] }}<|eot_id|>'
    '{% elif message["role"] == "assistant" %}'
    '<|start_header_id|>assistant<|end_header_id|>\\n\\n'
    '{{ message["content"] }}<|eot_id|>'
    '{% endif %}'
    '{% endfor %}'
)


def cells_data_model() -> List[dict]:
    layers_str = str(SELECTED_LAYERS)
    return [
        md("## 1. 数据加载 + 模型构建"),
        code(f"""\
# === 1. 数据 + 模型 ===
import sys, math, time, gc
import torch
import torch.nn as nn
from datasets import load_dataset, Features, Value, Sequence
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig

# 添加代码路径
sys.path.insert(0, CODE_DIR)

# === 超参数 (与 CCT Run 3 完全对齐) ===
CFG = {{
    'num_epochs': 1,
    'batch_size': 32,
    'grad_accum': 1,
    'max_seq_len': 512,
    'lr': 2e-5,
    'max_grad_norm': 1.0,
    'log_interval': 20,
    'eval_interval': 100,
}}

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
print('padding_side:', tokenizer.padding_side)

LLAMA3_CHAT_TEMPLATE = '{_CHAT_TEMPLATE}'
if tokenizer.chat_template is None:
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    print('已手动设置 Llama 3 chat template')

# === 数据集 (从本地 JSON 加载, 无需联网) ===
class SFTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        for ex in dataset:
            convs = ex.get('conversations', [])
            if not convs: continue
            user_parts, gpt_parts = [], []
            for turn in convs:
                role = turn.get('from', '')
                val = turn.get('value', '').strip()
                if not val: continue
                if role == 'human':
                    user_parts.append(val)
                elif role == 'gpt':
                    gpt_parts.append(val)
            if not user_parts or not gpt_parts: continue
            user_msg = '\\n\\n'.join(user_parts)
            gpt_msg = '\\n\\n'.join(gpt_parts)
            prompt_msgs = [{{'role': 'user', 'content': user_msg}}]
            full_msgs = prompt_msgs + [{{'role': 'assistant', 'content': gpt_msg}}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False)
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)['input_ids'])
            self.data.append((full_text, prompt_len))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        text, prompt_len = self.data[idx]
        enc = self.tokenizer(text, truncation=True,
                             max_length=self.max_length,
                             padding='max_length', return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attn = enc['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attn == 0] = -100
        if prompt_len > 0:
            labels[:prompt_len] = -100
        return {{'input_ids': input_ids, 'attention_mask': attn, 'labels': labels}}

print('加载训练数据...')
_features = Features({{'conversations': [{{
    'from': Value('string'), 'value': Value('string')}}]}})
raw = load_dataset('json', data_files=DATA_FILE, split='train',
                   features=_features)
print('数据量: %d' % len(raw))

_pv = SFTDataset([raw[0]], tokenizer, max_length=128)
_s = _pv[0]
_valid = (_s['labels'] != -100).sum().item()
_attn = _s['attention_mask'].sum().item()
print('Labels check: valid=%d, attn=%d' % (_valid, _attn))
assert _valid < _attn

full_ds = SFTDataset(raw, tokenizer, max_length=512)
eval_sz = int(len(full_ds) * 0.05)
train_ds, eval_ds = random_split(full_ds, [len(full_ds) - eval_sz, eval_sz])
print('Train: %d, Eval: %d' % (len(train_ds), len(eval_ds)))

# === 模型: 从 Llama-3.2-1B 选取 7 层 ===
DTYPE = torch.bfloat16

SELECTED_LAYERS = {layers_str}
NUM_SELECTED = {NUM_LAYERS}

base = LlamaForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=DTYPE, trust_remote_code=True)

selected = nn.ModuleList([base.model.layers[i] for i in SELECTED_LAYERS])
for new_idx, layer in enumerate(selected):
    layer.layer_idx = new_idx
    layer.self_attn.layer_idx = new_idx

base.model.layers = selected
base.config.num_hidden_layers = NUM_SELECTED
base.config.use_cache = False

gc.collect(); torch.cuda.empty_cache()

device = torch.device('cuda')
model = base.to(device)
model.train()

total_p = sum(p.numel() for p in model.parameters())
print('Layers: %s (%d)' % (SELECTED_LAYERS, NUM_SELECTED))
print('Total params: %d (%.1fM)' % (total_p, total_p / 1e6))"""),
    ]


def cells_train_config() -> List[dict]:
    return [
        md("## 2. 训练 (标准 LM, 无 CCT 机制)"),
        code("""\
# === 2. 训练配置 ===
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01)

train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                          shuffle=True, num_workers=2, pin_memory=True)
eval_loader = DataLoader(eval_ds, batch_size=CFG['batch_size'], num_workers=2)
total_steps = len(train_loader) * CFG['num_epochs'] // CFG['grad_accum']
lr_sched = CosineAnnealingLR(optimizer, T_max=total_steps)

os.makedirs('/kaggle/working/output', exist_ok=True)
gs, best_eval = 0, float('inf')

print('Total steps: %d (batch=%d, effective=%d)' % (
    total_steps, CFG['batch_size'], CFG['batch_size'] * CFG['grad_accum']))"""),
    ]


def cells_train_loop() -> List[dict]:
    return [
        code("""\
# === 训练循环 (标准 LM loss only) ===
import time as _time
model.train()
_t0 = _time.time()
max_steps = total_steps  # 跑完整 epoch
print('Training for %d steps' % max_steps)

_stop = False
for epoch in range(CFG['num_epochs']):
    if _stop: break
    avg_loss, avg_n = 0, 0

    for bi, batch in enumerate(train_loader):
        if gs >= max_steps:
            _stop = True; break
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        loss = out.loss / CFG['grad_accum']
        loss.backward()

        if (bi + 1) % CFG['grad_accum'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['max_grad_norm'])
            optimizer.step(); optimizer.zero_grad(); lr_sched.step()
            gs += 1

        avg_loss += out.loss.item()
        avg_n += 1

        if gs > 0 and gs % CFG['log_interval'] == 0 and (bi + 1) % CFG['grad_accum'] == 0:
            n = max(avg_n, 1)
            elapsed = _time.time() - _t0
            eta_m = (elapsed / gs) * (max_steps - gs) / 60 if gs > 0 else 0
            print('[Step %d/%d] loss=%.4f | lr=%.2e | ETA %.0fm' % (
                gs, max_steps, avg_loss/n, optimizer.param_groups[0]['lr'], eta_m))
            avg_loss, avg_n = 0, 0

        if gs > 0 and gs % CFG['eval_interval'] == 0 and (bi + 1) % CFG['grad_accum'] == 0:
            model.eval()
            ev_loss, ev_n = 0, 0
            with torch.no_grad():
                for eb in eval_loader:
                    eb = {k: v.to(device) for k, v in eb.items()}
                    with torch.amp.autocast('cuda', dtype=DTYPE):
                        eo = model(input_ids=eb['input_ids'],
                                  attention_mask=eb['attention_mask'],
                                  labels=eb['labels'])
                    ev_loss += eo.loss.item(); ev_n += 1
            avg_ev = ev_loss / max(ev_n, 1)
            ppl = math.exp(min(avg_ev, 20))
            print('  [Eval] loss=%.4f PPL=%.2f' % (avg_ev, ppl))
            if avg_ev < best_eval:
                best_eval = avg_ev
                torch.save(model.state_dict(), '/kaggle/working/output/best_model.pt')
                print('  New best!')
            model.train()

    print('Epoch %d done (%.1f min elapsed)' % (epoch + 1, (_time.time() - _t0) / 60))

torch.save(model.state_dict(), '/kaggle/working/output/final_model.pt')
print('Training complete! Best eval: %.4f (%.1f min total)' % (
    best_eval, (_time.time() - _t0) / 60))"""),
    ]


def cells_eval() -> List[dict]:
    return [
        md("## 3. 评估"),
        code("""\
# === 3. 评估 ===
model.eval()

ev_loss, ev_n = 0, 0
with torch.no_grad():
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        ev_loss += out.loss.item(); ev_n += 1

avg_ev = ev_loss / max(ev_n, 1)
ppl = math.exp(min(avg_ev, 20))

print()
print('=' * 50)
print('| Baseline 2 — Ablation (No Column Recycling) |')
print('=' * 50)
print('| LM Loss  | PPL    | Layers |')
print('|----------|--------|--------|')
print('| %.4f  | %.2f | %d      |' % (avg_ev, ppl, NUM_SELECTED))
print('=' * 50)
print()
print('对比 CCT Run 3: Best eval loss=1.8659 PPL=6.46')"""),
    ]


def cells_inference() -> List[dict]:
    return [
        md("## 3.5 推理测试"),
        code("""\
# === 3.5 推理测试 ===
model.eval()

_base_ds = eval_ds.dataset if hasattr(eval_ds, 'dataset') else eval_ds
import random
random.seed(42)
sample_indices = random.sample(range(len(eval_ds)), min(5, len(eval_ds)))

print('=' * 80)
for idx_i, si in enumerate(sample_indices):
    real_idx = eval_ds.indices[si] if hasattr(eval_ds, 'indices') else si
    text, prompt_len = _base_ds.data[real_idx]
    prompt_ids = tokenizer(text, truncation=True, max_length=CFG['max_seq_len'],
                          add_special_tokens=False)['input_ids'][:prompt_len]
    full_ids = tokenizer(text, truncation=True, max_length=CFG['max_seq_len'],
                        add_special_tokens=False)['input_ids']
    gt_ids = full_ids[prompt_len:]
    gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)

    input_tensor = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        gen_ids = model.generate(
            input_tensor,
            max_new_tokens=min(256, len(gt_ids) + 50),
            do_sample=False,
            use_cache=True,
        )
    gen_text = tokenizer.decode(gen_ids[0][len(prompt_ids):], skip_special_tokens=True)
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

    print('\\n--- Sample %d ---' % (idx_i + 1))
    print('[PROMPT] %s' % prompt_text[:200])
    print('[GT]     %s' % gt_text[:300])
    print('[MODEL]  %s' % gen_text[:300])
    print('[MATCH]  %s' % ('YES' if gt_text.strip()[:100] == gen_text.strip()[:100] else 'NO'))

print('=' * 80)"""),
    ]


def cells_comparison() -> List[dict]:
    return [
        md("## 4. CCT vs Baseline 2 对比"),
        code("""\
# === 4. 对比汇总 ===
print()
print('=' * 70)
print('|              | CCT (Run 3)  | Baseline 2 (No Recycling) |')
print('|--------------|-------------|---------------------------|')
print('| Architecture | 7L + recycle| 7L sequential             |')
print('| CCT modules  | Yes         | No                        |')
print('| Loss         | LM+pred+ent | LM only                   |')
print('|--------------|-------------|---------------------------|')
print('| Best PPL     | 6.46        | ? (填入实际值)             |')
print('| Final Loss   | 1.8659      | ? (填入实际值)             |')
print('=' * 70)"""),
    ]


# ── 组装 Notebook ─────────────────────────────────────
def build_notebook() -> dict:
    cells = []
    for section_fn in [
        cells_header,
        cells_setup,
        cells_data_model,
        cells_train_config,
        cells_train_loop,
        cells_eval,
        cells_inference,
        cells_comparison,
    ]:
        cells.extend(section_fn())

    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
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


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="构建 CCT 消融实验 Kaggle Notebook")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="输出 .ipynb 路径")
    args = parser.parse_args()

    path = save_notebook(args.output)
    print("Kaggle Notebook 已生成: %s" % path)
    print("   %d cells, 版本 %s" % (len(build_notebook()["cells"]), VERSION))
    print()
    print("使用步骤:")
    print("  1. python scripts/upload_kaggle_datasets.py --all")
    print("  2. 在 Kaggle 创建新 Notebook, 上传此 .ipynb")
    print("  3. 添加 Dataset: wukeneth/cct-code, wukeneth/cct-data")
    print("  4. 添加 Model: meta-llama/Llama-3.2-1B")
    print("  5. Settings: GPU P100/T4, Internet OFF")


if __name__ == "__main__":
    main()

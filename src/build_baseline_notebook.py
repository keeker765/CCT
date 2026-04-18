"""CCT Baseline Colab Notebook 构建器

生成与 CCT 实验完全对齐的 baseline notebook（标准 Llama fine-tuning）。
除模型架构外所有超参数一致，用于公平对比。

用法:
    python -m src.build_baseline_notebook
    python -m src.build_baseline_notebook -o my_baseline.ipynb
"""

import json
import argparse
from pathlib import Path
from typing import List

MODEL_NAME = "unsloth/Llama-3.2-1B"
DEFAULT_OUTPUT = "notebooks/baseline_colab.ipynb"
VERSION = "v1.0-baseline"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(source),
    }


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
# Baseline {VERSION}: 标准 Llama Fine-tuning

**目的**: 与 CCT 实验公平对比
- 相同 base model: `{MODEL_NAME}`
- 相同数据集: Alpaca (40k samples, 512 max_length)
- 相同学习率: 2e-5
- 相同训练配置: 1 epoch, batch_size=32, AdamW, cosine LR
- **无 CCT Column 循环** — 直接 full fine-tuning all 16 layers"""),
    ]


def cells_install() -> List[dict]:
    return [
        md("## 0. 安装依赖"),
        code("""\
# === 0. 安装依赖 ===
!pip install -q torch transformers datasets accelerate sentencepiece protobuf"""),
    ]


# Llama 3 chat template (与 CCT notebook 完全一致)
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
    return [
        md("## 1. 数据加载 + 模型初始化"),
        code(f"""\
# === 1. 数据 + 模型 ===
import os, sys, math, time, gc
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, LlamaForCausalLM

MODEL_NAME = "{MODEL_NAME}"

CFG = {{
    'num_epochs': 1,
    'batch_size': 32,
    'grad_accum': 1,
    'max_seq_len': 512,
    'lr': 2e-5,
    'max_grad_norm': 1.0,
    'log_interval': 20,
    'eval_interval': 200,
    'max_samples': 40000,
}}

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

LLAMA3_CHAT_TEMPLATE = '{_CHAT_TEMPLATE}'
if tokenizer.chat_template is None:
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    print('已手动设置 Llama 3 chat template')

# === 数据集 (Alpaca SFT) — 与 CCT notebook 完全一致 ===
class SFTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        for i, ex in enumerate(dataset):
            if max_samples and i >= max_samples: break
            if 'instruction' in ex:
                user_msg = ex['instruction']
                inp = ex.get('input', '').strip()
                if inp:
                    user_msg += '\\n\\n' + inp
                prompt_msgs = [{{'role': 'user', 'content': user_msg}}]
                full_msgs = prompt_msgs + [{{'role': 'assistant', 'content': ex.get('output', '')}}]
                prompt_text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True)
                full_text = tokenizer.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False)
                prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)['input_ids'])
                self.data.append((full_text, prompt_len))
            elif 'text' in ex:
                self.data.append((ex['text'], 0))
            else:
                continue
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

raw = load_dataset('tatsu-lab/alpaca', split='train')
full_ds = SFTDataset(raw, tokenizer, max_length=512, max_samples=CFG['max_samples'])
eval_sz = int(len(full_ds) * 0.05)
train_ds, eval_ds = random_split(full_ds, [len(full_ds) - eval_sz, eval_sz])
print('Train: %d, Eval: %d' % (len(train_ds), len(eval_ds)))

# === 模型 (标准 Llama, 全 16 层) ===
DTYPE = torch.bfloat16
model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True)

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params: %s' % format(total_params, ','))
print('Trainable: %s (%.2f%%)' % (format(trainable, ','), 100*trainable/total_params))

device = torch.device('cuda')
model = model.to(device)"""),
    ]


def cells_train() -> List[dict]:
    return [
        md("## 2. 训练"),
        code("""\
# === 2. 训练 ===
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time as _time

optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01)
train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                          shuffle=True, num_workers=0, pin_memory=True)
eval_loader = DataLoader(eval_ds, batch_size=CFG['batch_size'], num_workers=0)
total_steps = len(train_loader) * CFG['num_epochs'] // CFG['grad_accum']
lr_sched = CosineAnnealingLR(optimizer, T_max=total_steps)

os.makedirs('output/baseline', exist_ok=True)
gs, best_eval = 0, float('inf')

print('Total steps: %d (batch=%d x accum=%d)' % (
    total_steps, CFG['batch_size'], CFG['grad_accum']))

model.train()
_t0 = _time.time()

for epoch in range(CFG['num_epochs']):
    avg_loss, avg_n = 0, 0
    for bi, batch in enumerate(train_loader):
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
            eta_m = (elapsed / gs) * (total_steps - gs) / 60 if gs > 0 else 0
            print('[Step %d/%d] loss=%.4f | ETA %.0fm' % (
                gs, total_steps, avg_loss/n, eta_m))
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
                torch.save(model.state_dict(), 'output/baseline/best_model.pt')
                print('  New best!')
            model.train()

    print('Epoch %d done (%.1f min elapsed)' % (epoch + 1, (_time.time() - _t0) / 60))

torch.save(model.state_dict(), 'output/baseline/final_model.pt')
print('Training complete! Best eval: %.4f (%.1f min total)' % (
    best_eval, (_time.time() - _t0) / 60))"""),
    ]


def cells_eval() -> List[dict]:
    return [
        md("## 3. 评估"),
        code("""\
# === 3. 最终评估 ===
model.eval()
total_loss, count = 0, 0
with torch.no_grad():
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        total_loss += out.loss.item(); count += 1

avg_loss = total_loss / max(count, 1)
ppl = math.exp(min(avg_loss, 20))
print('=== Baseline: Llama-3.2-1B Full Fine-tuning ===')
print('Eval Loss = %.4f' % avg_loss)
print('Eval PPL  = %.2f' % ppl)
print()
print('| Model | Eval Loss | PPL | Layers | Params |')
print('|-------|-----------|-----|--------|--------|')
print('| Baseline (16L) | %.4f | %.2f | 16 | 1.24B |' % (avg_loss, ppl))
print('| CCT (7L循环)   | ???   | ??? | 7  | ~0.70B |')
print()
print('对比要点:')
print('  - Baseline 使用全部 16 层, CCT 只用 7 层 (约 56% 参数)')
print('  - CCT 通过 Column 循环复用 + Predictor/Precision 弥补层数差距')
print('  - 如果 CCT 接近 Baseline, 说明循环复用是有效的')"""),
    ]


def cells_backup() -> List[dict]:
    return [
        md("## 4. 备份到 Google Drive"),
        code("""\
# === 4. Backup ===
try:
    from google.colab import drive
    drive.mount('/content/drive')
    import shutil
    bk = '/content/drive/MyDrive/CCT/results'
    os.makedirs(bk, exist_ok=True)
    src = 'output/baseline'
    if os.path.exists(src):
        dst = os.path.join(bk, 'baseline')
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print('Backed up %s -> %s' % (src, dst))
    print('Done!')
except Exception as e:
    print('Backup skipped: %s' % e)"""),
    ]


def build_notebook() -> dict:
    cells = []
    for fn in [
        cells_header,
        cells_install,
        cells_data_model,
        cells_train,
        cells_eval,
        cells_backup,
    ]:
        cells.extend(fn())

    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "A100"},
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


def main():
    parser = argparse.ArgumentParser(description="构建 CCT Baseline Colab Notebook")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="输出 .ipynb 路径")
    args = parser.parse_args()

    path = save_notebook(args.output)
    print("✅ Baseline Notebook 已生成: %s" % path)
    cell_count = len(build_notebook()["cells"])
    print("   %d cells, 版本 %s" % (cell_count, VERSION))


if __name__ == "__main__":
    main()

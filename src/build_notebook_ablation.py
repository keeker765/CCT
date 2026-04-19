"""CCT 消融实验 Baseline 2 — Colab Notebook 构建器

消融设计:
  使用与 CCT 相同的 7 层 (front [0,1] + column [2,7,12] + back [14,15]),
  但不做循环复用, 直接顺序前向 → 标准 Transformer forward pass。

对比目标:
  验证 Column 循环机制带来的增益 (CCT vs 7-layer sequential baseline)

用法:
    python -m src.build_notebook_ablation
    python -m src.build_notebook_ablation -o notebooks/ablation_baseline2.ipynb
"""

import json
import argparse
from pathlib import Path
from typing import List

# ── 常量 ──────────────────────────────────────────────
GH_REPO = "keeker765/CCT"
MODEL_NAME = "unsloth/Llama-3.2-1B"
DEFAULT_OUTPUT = "notebooks/ablation_baseline2.ipynb"
VERSION = "ablation-b2"

# CCT 使用的层索引
SELECTED_LAYERS = [0, 1, 2, 7, 12, 14, 15]


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


# ── Notebook 各段 Cell 定义 ───────────────────────────
def cells_header() -> List[dict]:
    return [
        md(f"""\
# CCT 消融实验 — Baseline 2: 无循环 7 层顺序前向

**目的**: 与 CCT 对比, 验证 Column 循环机制的增益

**架构**: 从 Llama-3.2-1B (16层) 中选取 CCT 使用的 7 层, 顺序拼接:
- L0, L1 (CCT 的 Front)
- L2, L7, L12 (CCT 的 Column, 但**不复用**, 仅前向一次)
- L14, L15 (CCT 的 Back)

**与 CCT 的区别**:
- 无 Column 循环 (max_iter=1, 直接顺序)
- 无 Predictor / L6 Precision / HaltHead / RotaryCycleEmbed
- 仅 L_LM 损失
- 全参数统一 lr (无分层 lr)

**公平对比**:
- 相同数据 (40k OpenHermes 2.5), 相同 batch/lr/epochs
- 相同层数, 相同初始权重"""),
    ]


def cells_install() -> List[dict]:
    return [
        md("## 0. 安装依赖"),
        code("""\
# === 0. 安装依赖 ===
!pip install -q torch transformers datasets accelerate pyyaml sentencepiece protobuf huggingface_hub"""),
    ]


def cells_download() -> List[dict]:
    return [
        md("## 1. 下载项目代码 (仅用于数据加载工具)"),
        code(f"""\
# === 1. 克隆代码 (复用数据集类) ===
import os, subprocess

WORK_DIR = "/content/CCT"
MODEL_NAME = "{MODEL_NAME}"

if os.path.exists(WORK_DIR):
    print("目录已存在, 执行 git pull...")
    subprocess.run(["git", "-C", WORK_DIR, "pull"], check=True)
else:
    print("克隆仓库...")
    subprocess.run(["git", "clone", "https://github.com/{GH_REPO}.git", WORK_DIR], check=True)

os.chdir(WORK_DIR)
print("CWD: %s" % os.getcwd())"""),
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
    num_layers = len(SELECTED_LAYERS)
    return [
        md("## 2. 数据加载 + 模型构建"),
        code(f"""\
# === 2. 数据 + 模型 ===
import sys, math, time, gc
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, LlamaForCausalLM

# === 超参数 (与 CCT Run 3 完全对齐) ===
CFG = {{
    'num_epochs': 1,
    'batch_size': 32,
    'grad_accum': 1,
    'max_seq_len': 512,
    'lr': 2e-5,            # 统一 lr (无分层, 公平对比)
    'max_grad_norm': 1.0,
    'log_interval': 20,
    'eval_interval': 100,
    'max_samples': 40000,
}}

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
print('padding_side:', tokenizer.padding_side)

LLAMA3_CHAT_TEMPLATE = '{_CHAT_TEMPLATE}'
if tokenizer.chat_template is None:
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    print('已手动设置 Llama 3 chat template')
else:
    print('Tokenizer 已有 chat_template')

# === 数据集 (与 CCT 完全相同) ===
class SFTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        for i, ex in enumerate(dataset):
            if max_samples and i >= max_samples: break
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

raw = load_dataset('teknium/OpenHermes-2.5', split='train')
print('OpenHermes 2.5 总量: %d' % len(raw))
_pv = SFTDataset([raw[0]], tokenizer, max_length=128, max_samples=1)
_s = _pv[0]
_valid_labels = (_s['labels'] != -100).sum().item()
_total_attn = _s['attention_mask'].sum().item()
print('Labels check: valid=%d, attn_ones=%d' % (_valid_labels, _total_attn))
assert _valid_labels < _total_attn, 'BUG: prompt tokens not masked in labels!'

full_ds = SFTDataset(raw, tokenizer, max_length=512, max_samples=CFG['max_samples'])
eval_sz = int(len(full_ds) * 0.05)
train_ds, eval_ds = random_split(full_ds, [len(full_ds) - eval_sz, eval_sz])
print('Train: %d, Eval: %d' % (len(train_ds), len(eval_ds)))

# === 模型: 从 Llama-3.2-1B 选取 7 层, 顺序拼接 ===
DTYPE = torch.bfloat16

SELECTED_LAYERS = {layers_str}  # 与 CCT 使用的层完全相同
NUM_SELECTED = {num_layers}

base = LlamaForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True)

# 提取选中的层, 顺序拼接
selected = nn.ModuleList([base.model.layers[i] for i in SELECTED_LAYERS])

# 关键: 重新编号 layer_idx, 否则 KV cache 越界
# (原始 layer_idx=7,12,14,15 会超出 7 层 cache 范围)
for new_idx, layer in enumerate(selected):
    layer.layer_idx = new_idx
    layer.self_attn.layer_idx = new_idx
    if hasattr(layer.self_attn, 'config'):
        pass  # config 是共享的, 不需要改

base.model.layers = selected
base.config.num_hidden_layers = NUM_SELECTED
base.config.use_cache = False  # 训练时不需要 KV cache

# 释放不需要的引用
gc.collect(); torch.cuda.empty_cache()

device = torch.device('cuda')
model = base.to(device)
model.train()

# 打印参数量
total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Selected layers: %s (%d layers)' % (SELECTED_LAYERS, NUM_SELECTED))
print('Total params: %d' % total_p)
print('Trainable params: %d (%.2f%%)' % (train_p, 100 * train_p / total_p))"""),
    ]


def cells_train_config() -> List[dict]:
    return [
        md("## 3. 训练 (标准 LM, 无 CCT 机制)"),
        code("""\
# === 3. 训练配置 ===
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01)

train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                          shuffle=True, num_workers=0, pin_memory=True)
eval_loader = DataLoader(eval_ds, batch_size=CFG['batch_size'], num_workers=0)
total_steps = len(train_loader) * CFG['num_epochs'] // CFG['grad_accum']
lr_sched = CosineAnnealingLR(optimizer, T_max=total_steps)

os.makedirs('output/ablation_b2', exist_ok=True)
gs, best_eval = 0, float('inf')

print('Total steps: %d (batch=%d x accum=%d, effective=%d)' % (
    total_steps, CFG['batch_size'], CFG['grad_accum'],
    CFG['batch_size'] * CFG['grad_accum']))"""),
    ]


def cells_train_loop() -> List[dict]:
    return [
        code("""\
# === 训练循环 (标准 LM loss only) ===
import time as _time
model.train()
_t0 = _time.time()
max_steps = CFG.get('max_steps', total_steps)
print('Training for %d steps (max_steps=%d, total_steps=%d)' % (
    min(max_steps, total_steps), max_steps, total_steps))

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
                torch.save(model.state_dict(), 'output/ablation_b2/best_model.pt')
                print('  New best!')
            model.train()

    print('Epoch %d done (%.1f min elapsed)' % (epoch + 1, (_time.time() - _t0) / 60))

torch.save(model.state_dict(), 'output/ablation_b2/final_model.pt')
print('Training complete! Best eval: %.4f (%.1f min total)' % (
    best_eval, (_time.time() - _t0) / 60))"""),
    ]


def cells_eval() -> List[dict]:
    return [
        md("## 4. 评估"),
        code("""\
# === 4. 评估 ===
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
        md("## 4.5 推理测试"),
        code("""\
# === 4.5 推理测试 (与 CCT 相同的 5 个样本) ===
model.eval()

_base_ds = eval_ds.dataset if hasattr(eval_ds, 'dataset') else eval_ds
import random
random.seed(42)
sample_indices = random.sample(range(len(eval_ds)), min(5, len(eval_ds)))

print('=' * 80)
print('推理测试: eval 集中抽取 %d 个问题' % len(sample_indices))
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
    # 使用 HuggingFace .generate() (标准模型直接支持)
    with torch.no_grad():
        gen_ids = model.generate(
            input_tensor,
            max_new_tokens=min(256, len(gt_ids) + 50),
            do_sample=False,  # 贪心解码, 与 CCT 对比一致
            use_cache=True,
        )
    gen_text = tokenizer.decode(gen_ids[0][len(prompt_ids):], skip_special_tokens=True)
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

    print('\\n--- Sample %d (eval idx=%d, real=%d) ---' % (idx_i + 1, si, real_idx))
    print('[PROMPT] %s' % prompt_text[:300])
    print('[GROUND TRUTH] %s' % gt_text[:500])
    print('[MODEL OUTPUT] %s' % gen_text[:500])
    match = gt_text.strip()[:100] == gen_text.strip()[:100]
    print('[MATCH first 100 chars] %s' % ('YES' if match else 'NO'))
    print()

print('=' * 80)
print('对比 CCT Run 3 的推理输出, 观察 7 层无循环 vs 7 层循环的生成质量差异')
print('=' * 80)"""),
    ]


def cells_viz() -> List[dict]:
    return [
        md("## 5. 训练曲线可视化"),
        code("""\
# === 5. Loss 曲线 (从日志解析) ===
import matplotlib.pyplot as plt
import re

# 从 Colab 输出中手动记录的 eval 数据点 (训练完成后填入)
# 格式: [(step, loss), ...]
eval_history = [
    # 示例: (100, 2.6827),
    # 训练完成后替换为实际数据
]

if eval_history:
    steps, losses = zip(*eval_history)
    ppls = [math.exp(min(l, 20)) for l in losses]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, losses, 'o-', color='steelblue')
    ax1.set_title('Baseline 2 — Eval Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('LM Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, ppls, 'o-', color='coral')
    ax2.set_title('Baseline 2 — PPL')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('PPL')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/ablation_b2/loss_curve.png', dpi=150)
    plt.show()
    print('Saved to output/ablation_b2/loss_curve.png')
else:
    print('训练完成后, 将 eval 结果填入 eval_history 列表并重新运行此 cell')"""),
    ]


def cells_comparison() -> List[dict]:
    return [
        md("## 6. CCT vs Baseline 2 对比表"),
        code("""\
# === 6. 对比汇总 ===
print()
print('=' * 70)
print('|              | CCT (Run 3)  | Baseline 2 (No Recycling) |')
print('|--------------|-------------|---------------------------|')
print('| Architecture | 7L + recycle| 7L sequential             |')
print('| Layers       | 7 (×K iter) | 7 (×1 pass)               |')
print('| CCT modules  | Yes         | No                        |')
print('| Loss         | LM+pred+ent | LM only                   |')
print('|--------------|-------------|---------------------------|')
print('| Best PPL     | 6.46        | ? (填入实际值)             |')
print('| Final Loss   | 1.8659      | ? (填入实际值)             |')
print('=' * 70)
print()
print('如果 CCT PPL < Baseline 2 PPL → Column 循环机制有正面增益')
print('如果 CCT PPL ≈ Baseline 2 PPL → 循环机制无显著效果, 需要调整')"""),
    ]


def cells_backup() -> List[dict]:
    return [
        md("## 7. 备份到 Google Drive"),
        code("""\
# === 7. Backup ===
try:
    from google.colab import drive
    drive.mount('/content/drive')
    import shutil
    bk = '/content/drive/MyDrive/CCT/results'
    os.makedirs(bk, exist_ok=True)
    src = 'output/ablation_b2'
    if os.path.exists(src):
        dst = os.path.join(bk, 'ablation_b2')
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print('Backed up %s -> %s' % (src, dst))
    print('Done!')
except Exception as e:
    print('Backup skipped: %s' % e)"""),
    ]


# ── 组装 Notebook ─────────────────────────────────────
def build_notebook() -> dict:
    cells = []
    for section_fn in [
        cells_header,
        cells_install,
        cells_download,
        cells_data_model,
        cells_train_config,
        cells_train_loop,
        cells_eval,
        cells_inference,
        cells_viz,
        cells_comparison,
        cells_backup,
    ]:
        cells.extend(section_fn())

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


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="构建 CCT 消融实验 Baseline 2 Notebook")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="输出 .ipynb 路径")
    args = parser.parse_args()

    path = save_notebook(args.output)
    print("Notebook 已生成: %s" % path)

    cell_count = len(build_notebook()["cells"])
    print("   %d cells, 版本 %s" % (cell_count, VERSION))


if __name__ == "__main__":
    main()

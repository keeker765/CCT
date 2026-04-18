"""CCT Colab Notebook 构建器

用法:
    python -m src.build_notebook                 # 默认输出到 notebooks/cct_colab.ipynb
    python -m src.build_notebook -o my_nb.ipynb   # 自定义输出路径
"""

import json
import argparse
from pathlib import Path
from typing import List

# ── 常量 ──────────────────────────────────────────────
GH_REPO = "keeker765/CCT"
MODEL_NAME = "unsloth/Llama-3.2-1B"
DEFAULT_OUTPUT = "notebooks/cct_colab.ipynb"
VERSION = "v1.0"


# ── Cell 构造器 ───────────────────────────────────────
def md(source: str) -> dict:
    """Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(source),
    }


def code(source: str) -> dict:
    """Code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": _lines(source),
        "outputs": [],
        "execution_count": None,
    }


def _lines(text: str) -> List[str]:
    """将多行字符串拆成 notebook 的 source list 格式（每行带 \\n，末行不带）。"""
    raw = text.strip("\n")
    parts = raw.split("\n")
    result = [line + "\n" for line in parts[:-1]]
    result.append(parts[-1])
    return result


# ── Notebook 各段 Cell 定义 ───────────────────────────
def cells_header() -> List[dict]:
    return [
        md(f"""\
# CCT {VERSION} (Cortical Column Transformer) — Colab 训练

**架构**: Fixed Front (L0-L1) → Column (L2-L4, 循环复用 K 次) → Fixed Back (L14-L15)
- **Predictor + AnchorMLP**: 预测编码误差信号
- **L6 Precision**: query 侧乘性增益调制 (error-driven temperature scaling)
- **HaltHead**: ACT 软停止 + τ 退火二值化
- **RotaryCycleEmbed**: φ 黄金比例旋转循环嵌入

**损失**: L_LM + λ_pred · L_pred + λ_flops · L_flops

**流程**: 安装 → 下载代码 → 数据+模型 → 训练 → 评估 → 可视化 → 备份"""),
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
        md("## 1. 下载项目代码"),
        code(f"""\
# === 1. 从 GitHub 克隆代码 ===
import os, subprocess
from getpass import getpass

WORK_DIR = "/content/CCT"
MODEL_NAME = "{MODEL_NAME}"

# 仓库是 private, 需要 GitHub PAT
# GitHub Settings -> Developer settings -> Personal access tokens -> Generate
GH_TOKEN = getpass("GitHub PAT (repo scope): ")
REPO_URL = f"https://{{GH_TOKEN}}@github.com/{GH_REPO}.git"

if os.path.exists(WORK_DIR):
    print("目录已存在, 执行 git pull...")
    subprocess.run(["git", "-C", WORK_DIR, "pull"], check=True)
else:
    print("克隆私有仓库...")
    subprocess.run(["git", "clone", REPO_URL, WORK_DIR], check=True)

# 安全: 清除 remote URL 中的 PAT, 防止泄露
subprocess.run(["git", "-C", WORK_DIR, "remote", "set-url", "origin",
                "https://github.com/{GH_REPO}.git"], check=False)
del GH_TOKEN, REPO_URL

os.chdir(WORK_DIR)
print("CWD: %s" % os.getcwd())

# 验证关键文件
key_files = [
    "src/model/wrapped_model.py", "src/model/cct_attention.py",
    "src/model/predictor.py", "src/model/l6_precision.py",
    "src/model/halt_head.py", "src/model/cycle_embedding.py",
    "src/model/losses.py", "src/model/column_config.py",
]
for f in key_files:
    assert os.path.exists(f), "缺失: %s" % f
    print("  ✅ %s" % f)"""),
    ]


# Llama 3 chat template — base 模型没有内置 chat_template，必须手动设置
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
        md("## 2. 数据加载 + 模型初始化"),
        code(f"""\
# === 2. 数据 + 模型 ===
import sys, math, time, gc
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, LlamaForCausalLM

sys.path.insert(0, '.')
from src.model.wrapped_model import CCTLlamaModel
from src.model.column_config import CCTConfig
from src.training.scheduler import compute_halt_tau

# === 超参数 (与 PPG 对齐) ===
CFG = {{
    'num_epochs': 1,
    'batch_size': 32,
    'grad_accum': 1,
    'max_seq_len': 512,
    'lr': 2e-5,
    'new_lr': 1e-4,
    'max_grad_norm': 1.0,
    'log_interval': 20,
    'eval_interval': 200,
    'max_samples': 40000,
}}

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# unsloth/Llama-3.2-1B 是 base 模型, 没有 chat_template, 手动设置
LLAMA3_CHAT_TEMPLATE = '{_CHAT_TEMPLATE}'
if tokenizer.chat_template is None:
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    print('已手动设置 Llama 3 chat template')
else:
    print('Tokenizer 已有 chat_template')

# === 数据集 (Alpaca SFT) ===
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
                messages = [
                    {{'role': 'user', 'content': user_msg}},
                    {{'role': 'assistant', 'content': ex.get('output', '')}},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
            elif 'text' in ex:
                text = ex['text']
            else:
                continue
            self.data.append(text)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.data[idx], truncation=True,
                             max_length=self.max_length,
                             padding='max_length', return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attn = enc['attention_mask'].squeeze(0)
        labels = input_ids.clone(); labels[attn == 0] = -100
        return {{'input_ids': input_ids, 'attention_mask': attn, 'labels': labels}}

# 预览
raw = load_dataset('tatsu-lab/alpaca', split='train')
_pv = SFTDataset([raw[0]], tokenizer, max_length=128, max_samples=1)
print('=== Chat Template 预览 ===')
print(tokenizer.decode(_pv[0]['input_ids'][:80]))
print('...\\n')

full_ds = SFTDataset(raw, tokenizer, max_length=512, max_samples=CFG['max_samples'])
eval_sz = int(len(full_ds) * 0.05)
train_ds, eval_ds = random_split(full_ds, [len(full_ds) - eval_sz, eval_sz])
print('Train: %d, Eval: %d' % (len(train_ds), len(eval_ds)))

# === GPU 检测 + dtype 选择 ===
assert torch.cuda.is_available(), 'Need GPU!'
USE_BF16 = torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print('GPU: %s, VRAM: %.1f GB, dtype: %s' % (
    torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_mem / 1e9,
    'bf16' if USE_BF16 else 'fp16'))

# 小显存自动降 batch_size
vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
if vram_gb < 20 and CFG['batch_size'] > 8:
    CFG['batch_size'] = 8
    print('VRAM < 20GB, batch_size 降至 %d' % CFG['batch_size'])

# === 模型 ===
cct_config = CCTConfig(
    max_iter=5,
    lambda_pred=0.1,
    lambda_flops=0.01,
    bf16=USE_BF16,
    gradient_checkpointing=True,
)

base = LlamaForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True)
model = CCTLlamaModel(base, cct_config)

# 释放 base 模型, 节省 ~2GB GPU 内存
del base; gc.collect(); torch.cuda.empty_cache()

model.enable_gradient_checkpointing()
device = torch.device('cuda')
model = model.to(device)
print(model.get_trainable_params_info())
print('Column layers: %d, Max iter: %d' % (
    len(cct_config.pretrained_column_layers), cct_config.max_iter))"""),
    ]


def cells_train_config() -> List[dict]:
    return [
        md("## 3. 训练 (单阶段, 全参数)"),
        code("""\
# === 3. 训练配置 ===
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 分层 LR
param_groups = model.get_param_groups()
param_groups[0]['lr'] = CFG['lr']
param_groups[1]['lr'] = CFG['new_lr']
optimizer = AdamW(param_groups, weight_decay=0.01)

train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                          shuffle=True, num_workers=0, pin_memory=True)
eval_loader = DataLoader(eval_ds, batch_size=CFG['batch_size'], num_workers=0)
total_steps = len(train_loader) * CFG['num_epochs'] // CFG['grad_accum']
lr_sched = CosineAnnealingLR(optimizer, T_max=total_steps)

os.makedirs('output/cct', exist_ok=True)
gs, best_eval = 0, float('inf')

print('Total steps: %d (batch=%d x accum=%d, effective=%d)' % (
    total_steps, CFG['batch_size'], CFG['grad_accum'],
    CFG['batch_size'] * CFG['grad_accum']))"""),
    ]


def cells_train_loop() -> List[dict]:
    return [
        code("""\
# === 训练循环 ===
import time as _time
model.train()
_t0 = _time.time()

for epoch in range(CFG['num_epochs']):
    avg = {'total': 0, 'lm': 0, 'pred': 0, 'flops': 0, 'iters': 0}
    avg_n = 0

    for bi, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # tau_halt 线性退火
        tau_halt = compute_halt_tau(gs, total_steps,
                                   cct_config.halt_tau_start, cct_config.halt_tau_end)
        model.set_halt_tau(tau_halt)

        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        loss = out['loss'] / CFG['grad_accum']
        loss.backward()

        if (bi + 1) % CFG['grad_accum'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['max_grad_norm'])
            optimizer.step(); optimizer.zero_grad(); lr_sched.step()
            gs += 1

        ld = out['loss_dict']
        avg['total'] += ld.get('loss_total', 0)
        avg['lm'] += ld.get('loss_lm', 0)
        avg['pred'] += ld.get('loss_pred', 0)
        avg['flops'] += ld.get('loss_flops', 0)
        avg['iters'] += out.get('num_iterations', 0)
        avg_n += 1

        if gs > 0 and gs % CFG['log_interval'] == 0 and (bi + 1) % CFG['grad_accum'] == 0:
            n = max(avg_n, 1)
            elapsed = _time.time() - _t0
            eta_m = (elapsed / gs) * (total_steps - gs) / 60 if gs > 0 else 0
            print('[Step %d/%d] loss=%.4f | lm=%.4f pred=%.4f flops=%.4f | '
                  'iters=%.1f tau=%.3f | ETA %.0fm' % (
                gs, total_steps, avg['total']/n, avg['lm']/n, avg['pred']/n,
                avg['flops']/n, avg['iters']/n, tau_halt, eta_m))
            avg = {'total': 0, 'lm': 0, 'pred': 0, 'flops': 0, 'iters': 0}
            avg_n = 0

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
                    ev_loss += eo['loss_dict'].get('loss_lm', 0); ev_n += 1
            avg_ev = ev_loss / max(ev_n, 1)
            ppl = math.exp(min(avg_ev, 20))
            print('  [Eval] loss=%.4f PPL=%.2f' % (avg_ev, ppl))
            if avg_ev < best_eval:
                best_eval = avg_ev
                torch.save(model.state_dict(), 'output/cct/best_model.pt')
                print('  New best!')
            model.train()

    print('Epoch %d done (%.1f min elapsed)' % (epoch + 1, (_time.time() - _t0) / 60))

torch.save(model.state_dict(), 'output/cct/final_model.pt')
print('Training complete! Best eval: %.4f (%.1f min total)' % (
    best_eval, (_time.time() - _t0) / 60))"""),
    ]


def cells_eval() -> List[dict]:
    return [
        md("## 4. 评估 (训练态 vs 推理态)"),
        code("""\
# === 4. 评估: ACT 加权和 vs 硬停止 ===
model.eval()

def evaluate(model, loader):
    total_loss, count = 0, 0
    all_iters = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=DTYPE):
                out = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])
            total_loss += out['loss_dict'].get('loss_lm', 0)
            all_iters.append(out.get('num_iterations', 0))
            count += 1
    avg = total_loss / max(count, 1)
    avg_iter = sum(all_iters) / max(len(all_iters), 1)
    return avg, math.exp(min(avg, 20)), avg_iter

# 训练态 (低 tau, ACT 加权和)
model.set_halt_tau(cct_config.halt_tau_end)
model.train()
tl, tp, ti = evaluate(model, eval_loader)

# 推理态 (硬停止)
model.eval()
il, ip, ii = evaluate(model, eval_loader)

print()
print('=' * 60)
print('| Mode     | LM Loss | PPL    | Avg Iters | Gap    |')
print('|----------|---------|--------|-----------|--------|')
print('| Train    | %.4f  | %.2f | %.1f       | -      |' % (tl, tp, ti))
print('| Infer    | %.4f  | %.2f | %.1f       | %+.4f |' % (il, ip, ii, il - tl))
print('=' * 60)"""),
    ]


def cells_viz() -> List[dict]:
    return [
        md("## 5. 可视化: Precision + 循环次数分布"),
        code("""\
# === 5. 可视化 ===
import matplotlib.pyplot as plt
import numpy as np

model.eval()
all_scores_viz = []
all_iters_viz = []

with torch.no_grad():
    for bi, batch in enumerate(eval_loader):
        if bi >= 20: break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        if out['scores']:
            all_scores_viz.append(out['scores'][-1].float().cpu())
        all_iters_viz.append(out['num_iterations'])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) Score 分布
if all_scores_viz:
    scores_flat = torch.cat(all_scores_viz).flatten().numpy()
    axes[0].hist(scores_flat, bins=50, alpha=0.7, color='steelblue')
    axes[0].set_title('Prediction Score Distribution')
    axes[0].set_xlabel('score (pred . anchor / sqrt(d))')
    axes[0].axvline(x=np.mean(scores_flat), color='red', ls='--',
                    label='mean=%.3f' % np.mean(scores_flat))
    axes[0].legend()

# (b) Precision 分布
if all_scores_viz:
    tau_p = cct_config.precision_temperature
    precision = 1.0 - 1.0 / (1.0 + np.exp(-scores_flat / tau_p))
    axes[1].hist(precision, bins=50, alpha=0.7, color='coral')
    axes[1].set_title('Precision Distribution (tau_p=%.2f)' % tau_p)
    axes[1].set_xlabel('precision = 1 - sigma(score/tau)')
    axes[1].axvline(x=np.mean(precision), color='red', ls='--',
                    label='mean=%.3f' % np.mean(precision))
    axes[1].legend()

# (c) 循环次数分布
axes[2].hist(all_iters_viz, bins=range(1, cct_config.max_iter + 2),
             alpha=0.7, color='seagreen', align='left')
axes[2].set_title('Iterations per Batch')
axes[2].set_xlabel('num iterations')
axes[2].set_xticks(range(1, cct_config.max_iter + 1))
mean_iter = np.mean(all_iters_viz)
axes[2].axvline(x=mean_iter, color='red', ls='--',
                label='mean=%.1f' % mean_iter)
axes[2].legend()

plt.tight_layout()
plt.savefig('output/cct/viz_distributions.png', dpi=150)
plt.show()
print('Saved to output/cct/viz_distributions.png')"""),
    ]


def cells_backup() -> List[dict]:
    return [
        md("## 6. 备份到 Google Drive"),
        code("""\
# === 6. Backup ===
try:
    from google.colab import drive
    drive.mount('/content/drive')
    import shutil
    bk = '/content/drive/MyDrive/CCT/results'
    os.makedirs(bk, exist_ok=True)
    src = 'output/cct'
    if os.path.exists(src):
        dst = os.path.join(bk, 'cct_run')
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print('Backed up %s -> %s' % (src, dst))
    print('Done!')
except Exception as e:
    print('Backup skipped: %s' % e)"""),
    ]


# ── 组装 Notebook ─────────────────────────────────────
def build_notebook() -> dict:
    """组装完整的 Colab notebook。"""
    cells = []
    for section_fn in [
        cells_header,
        cells_install,
        cells_download,
        cells_data_model,
        cells_train_config,
        cells_train_loop,
        cells_eval,
        cells_viz,
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
    """构建并保存 notebook，返回绝对路径。"""
    nb = build_notebook()
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return str(p.resolve())


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="构建 CCT Colab Notebook")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="输出 .ipynb 路径")
    args = parser.parse_args()

    path = save_notebook(args.output)
    print("✅ Notebook 已生成: %s" % path)

    cell_count = len(build_notebook()["cells"])
    print("   %d cells, 版本 %s" % (cell_count, VERSION))


if __name__ == "__main__":
    main()

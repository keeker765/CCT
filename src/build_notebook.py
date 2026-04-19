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
VERSION = "v2.0"


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
# CCT {VERSION} — Cortical Column Transformer — Continued Pretraining

**架构**: Fixed Front (L0-L1) → Column (L2,L7,L12 × K 循环) → Fixed Back (L14-L15)
- **Cross-Layer Fusion**: Column MLP 2x 加宽 (d_ff=16384), 融合 Donor 层 FFN
- **Predictor + L6 Precision**: 预测编码误差 → 注意力调制 + ACT 停止
- **RotaryCycleEmbed**: φ 黄金比例旋转循环嵌入

**数据**: 50% FinePDFs + 30% DCLM + 20% FineWeb-Edu (1B tokens, Packing)
**训练**: Continued Pretraining (CLM loss on all tokens), seq=2048, effective batch=128

**流程**: 安装 → 下载 → 数据+模型 → 训练 → 评估 → 可视化 → 备份"""),
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

WORK_DIR = "/content/CCT"
MODEL_NAME = "{MODEL_NAME}"

if os.path.exists(WORK_DIR):
    print("目录已存在, 执行 git pull...")
    subprocess.run(["git", "-C", WORK_DIR, "pull"], check=True)
else:
    print("克隆仓库...")
    subprocess.run(["git", "clone", "https://github.com/{GH_REPO}.git", WORK_DIR], check=True)

os.chdir(WORK_DIR)
print("CWD: %s" % os.getcwd())

# 验证关键文件
key_files = [
    "src/model/wrapped_model.py", "src/model/cct_attention.py",
    "src/model/predictor.py", "src/model/l6_precision.py",
    "src/model/cycle_embedding.py",
    "src/model/losses.py", "src/model/column_config.py",
]
for f in key_files:
    assert os.path.exists(f), "缺失: %s" % f
    print("  ✅ %s" % f)"""),
    ]


def cells_data_model() -> List[dict]:
    return [
        md("## 2. 数据加载 + 模型初始化"),
        code("""\
# === 2a. Imports + 超参数 ===
import sys, math, time, gc
import torch
import torch.nn as nn
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer, LlamaForCausalLM

sys.path.insert(0, '.')
from src.model.wrapped_model import CCTLlamaModel
from src.model.column_config import CCTConfig
from src.training.scheduler import compute_halt_tau, get_cosine_schedule_with_warmup

CFG = {
    'max_steps': 3800,           # 1B tokens / 262K tokens_per_step
    'batch_size': 32,            # micro batch (96GB VRAM)
    'grad_accum': 4,             # effective batch = 128 seqs = 262K tokens/step
    'max_seq_len': 2048,         # match MoR
    'lr': 1e-4,                  # pretrained backbone (CPT)
    'new_lr': 5e-4,              # CCT new modules (predictor, L6, cycle_emb)
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'warmup_steps': 200,         # ~5% of total steps
    'log_interval': 20,          # log every N optimizer steps
    'eval_interval': 200,        # eval every N optimizer steps
    'eval_chunks': 100,          # eval set: 100 chunks x 2048 = 205K tokens
}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print('Vocab size: %d, EOS id: %d' % (tokenizer.vocab_size, tokenizer.eos_token_id))"""),
        code("""\
# === 2b. 流式数据集: 50% FinePDFs + 30% DCLM + 20% FineWeb-Edu ===
# codelion's 1B Token Challenge (2025): 各源预采样 1B tokens

DS_NAMES = [
    'codelion/finepdfs-1B',      # PDF 学术文本 (最高质量单源)
    'codelion/dclm-baseline-1B', # DataComp-LM 筛选 web 文本
    'codelion/fineweb-edu-1B',   # 教育类 web 页面 (MoR 同源)
]
# 文档级采样概率. 由于 FinePDFs 文档较长 (~3K tok/doc vs ~1.3K),
# 实际 token 比例会偏向 FinePDFs (~70/18/12).
# 如需精确 token 级 50/30/20, 可调整为 [0.29, 0.40, 0.31]
DS_PROBS = [0.5, 0.3, 0.2]

print('Loading 3 datasets (streaming)...')
ds_list_train = [load_dataset(n, split='train', streaming=True) for n in DS_NAMES]
mixed_train = interleave_datasets(ds_list_train, probabilities=DS_PROBS,
                                   seed=42, stopping_strategy='all_exhausted')

# Eval: 独立流 (不同 seed -> 不同文档顺序)
ds_list_eval = [load_dataset(n, split='train', streaming=True) for n in DS_NAMES]
mixed_eval = interleave_datasets(ds_list_eval, probabilities=DS_PROBS,
                                  seed=123, stopping_strategy='all_exhausted')

print('Interleaved: 50% FinePDFs + 30% DCLM + 20% FineWeb-Edu')"""),
        code("""\
# === 2c. Packing: 文档拼接 + EOS -> 固定 seq_len 切块 ===
class PackedDataset(IterableDataset):
    \"\"\"Packing: concatenate docs with EOS separator, chunk into seq_len.
    零 padding 浪费, 工业标准预训练数据策略.\"\"\"
    def __init__(self, hf_dataset, tokenizer, seq_len=2048):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = example.get('text', '')
            if not text or not text.strip():
                continue
            ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
            buffer.extend(ids)
            buffer.append(self.tokenizer.eos_token_id)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                t = torch.tensor(chunk, dtype=torch.long)
                yield {
                    'input_ids': t,
                    'attention_mask': torch.ones_like(t),
                    'labels': t.clone(),
                }

class ListDataset(Dataset):
    \"\"\"Wrap a list as a map-style Dataset for DataLoader.\"\"\"
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# Eval set: 收集固定数量的 packed chunks 到内存
print('Building eval set (%d chunks)...' % CFG['eval_chunks'])
eval_packed = PackedDataset(mixed_eval, tokenizer, seq_len=CFG['max_seq_len'])
eval_data = []
for item in eval_packed:
    eval_data.append(item)
    if len(eval_data) >= CFG['eval_chunks']:
        break
eval_loader = DataLoader(ListDataset(eval_data), batch_size=CFG['batch_size'], num_workers=0)
print('Eval: %d chunks (%d tokens)' % (len(eval_data), len(eval_data) * CFG['max_seq_len']))

# Train DataLoader (streaming Packing)
train_packed = PackedDataset(mixed_train, tokenizer, seq_len=CFG['max_seq_len'])
train_loader = DataLoader(train_packed, batch_size=CFG['batch_size'],
                          num_workers=0, pin_memory=True)
print('Train loader ready (streaming Packing, seq_len=%d)' % CFG['max_seq_len'])"""),
        code("""\
# === 2d. 模型初始化 ===
DTYPE = torch.bfloat16

cct_config = CCTConfig(
    max_iter=5,
    lambda_pred=0.1,
    lambda_entropy=0.0,
    lambda_ponder=0.0,
    use_ponder_cost=False,
    column_d_ff=16384,       # 2x FFN 加宽 (Cross-Layer Fusion)
    widen_mode='cross',
    donor_init_scale=0.1,
    bf16=True,
    gradient_checkpointing=True,
    max_seq_len=2048,
    per_device_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    new_module_lr=5e-4,
    max_steps=3800,
    warmup_steps=200,
)

base = LlamaForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True)
model = CCTLlamaModel(base, cct_config)
del base; gc.collect(); torch.cuda.empty_cache()

model.enable_gradient_checkpointing()
device = torch.device('cuda')
model = model.to(device)

# 加速设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

print(model.get_trainable_params_info())
print('Column layers: %d, Max iter: %d, seq_len: %d' % (
    len(cct_config.pretrained_column_layers), cct_config.max_iter, CFG['max_seq_len']))
print('SDPA enabled, TF32 matmul enabled')"""),
    ]


def cells_train_config() -> List[dict]:
    return [
        md("## 3. 训练 (Continued Pretraining)"),
        code("""\
# === 3. 训练配置 ===
from torch.optim import AdamW

# 分层 LR: pretrained backbone vs CCT new modules
param_groups = model.get_param_groups()
param_groups[0]['lr'] = CFG['lr']      # backbone: 1e-4
param_groups[1]['lr'] = CFG['new_lr']  # new modules: 5e-4
optimizer = AdamW(param_groups, weight_decay=CFG['weight_decay'])

# Cosine schedule with linear warmup
lr_sched = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CFG['warmup_steps'],
    num_training_steps=CFG['max_steps'],
    min_lr_ratio=0.1,
)

os.makedirs('output/cct', exist_ok=True)
best_eval = float('inf')

eff_batch = CFG['batch_size'] * CFG['grad_accum']
tokens_per_step = eff_batch * CFG['max_seq_len']
print('Effective batch: %d seqs = %dK tokens/step' % (eff_batch, tokens_per_step // 1000))
print('Total: %d steps = %.2fB tokens' % (CFG['max_steps'], CFG['max_steps'] * tokens_per_step / 1e9))"""),
    ]


def cells_train_loop() -> List[dict]:
    return [
        code("""\
# === 训练循环 (step-based, streaming) ===
import time as _time
model.train()
_t0 = _time.time()
max_steps = CFG['max_steps']
print('Training for %d optimizer steps (grad_accum=%d)...' % (max_steps, CFG['grad_accum']))

avg = {'total': 0, 'lm': 0, 'pred': 0, 'eff_iters': 0, 'eff_std': 0, 'score_std': 0}
avg_n = 0
train_iter = iter(train_loader)

for gs in range(max_steps):
    # tau_halt sigmoid 退火
    tau_halt = compute_halt_tau(gs, max_steps,
                                cct_config.halt_tau_start, cct_config.halt_tau_end)
    model.set_halt_tau(tau_halt)

    # Gradient accumulation: grad_accum 个 micro-batch
    for _micro in range(CFG['grad_accum']):
        try:
            batch = next(train_iter)
        except StopIteration:
            print('[Step %d] Stream exhausted, restarting...' % gs)
            train_packed_new = PackedDataset(
                interleave_datasets(
                    [load_dataset(n, split='train', streaming=True) for n in DS_NAMES],
                    probabilities=DS_PROBS, seed=42 + gs,
                    stopping_strategy='all_exhausted'),
                tokenizer, seq_len=CFG['max_seq_len'])
            train_loader_new = DataLoader(train_packed_new, batch_size=CFG['batch_size'],
                                          num_workers=0, pin_memory=True)
            train_iter = iter(train_loader_new)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        loss = out['loss'] / CFG['grad_accum']
        loss.backward()

        ld = out['loss_dict']
        avg['total'] += ld.get('loss_total', 0)
        avg['lm'] += ld.get('loss_lm', 0)
        avg['pred'] += ld.get('loss_pred', 0)
        avg['eff_iters'] += out.get('effective_iters', 0)
        avg['eff_std'] += out.get('eff_iters_std', 0)
        avg['score_std'] += out.get('score_std', 0)
        avg_n += 1

    # Optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['max_grad_norm'])
    optimizer.step(); optimizer.zero_grad(); lr_sched.step()

    # Logging
    if (gs + 1) % CFG['log_interval'] == 0:
        n = max(avg_n, 1)
        elapsed = _time.time() - _t0
        eta_m = (elapsed / (gs + 1)) * (max_steps - gs - 1) / 60
        tokens_done = (gs + 1) * CFG['batch_size'] * CFG['grad_accum'] * CFG['max_seq_len']
        print('[Step %d/%d] loss=%.4f | lm=%.4f pred=%.4f | '
              'eff_iters=%.2f+/-%.2f score_std=%.4f tau=%.3f | '
              'lr=%.2e | %.1fM tok | ETA %.0fm' % (
            gs + 1, max_steps, avg['total']/n, avg['lm']/n, avg['pred']/n,
            avg['eff_iters']/n, avg['eff_std']/n, avg['score_std']/n, tau_halt,
            optimizer.param_groups[0]['lr'], tokens_done / 1e6, eta_m))
        avg = {'total': 0, 'lm': 0, 'pred': 0, 'eff_iters': 0, 'eff_std': 0, 'score_std': 0}
        avg_n = 0

    # Eval
    if (gs + 1) % CFG['eval_interval'] == 0:
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
        print('  [Eval step %d] loss=%.4f PPL=%.2f' % (gs + 1, avg_ev, ppl))
        if avg_ev < best_eval:
            best_eval = avg_ev
            torch.save(model.state_dict(), 'output/cct/best_model.pt')
            print('  New best! Saved.')
        model.train()

torch.save(model.state_dict(), 'output/cct/final_model.pt')
print('\\nTraining complete! Best eval loss: %.4f (%.1f min total)' % (
    best_eval, (_time.time() - _t0) / 60))"""),
    ]


def cells_eval() -> List[dict]:
    return [
        md("## 4. 评估 (训练态 vs 推理态)"),
        code("""\
# === 4. 评估: 训练态 (ACT 加权和) vs 推理态 (硬停止) ===
# effective_iters = per-token 期望迭代数 (加权平均, 可为小数, 如 2.3)
# num_iterations = column 循环实际运行次数 (整数, 训练态=max_iter, 推理态<=max_iter)
model.eval()

def evaluate(model, data_list, batch_size=32):
    loader = DataLoader(ListDataset(data_list), batch_size=batch_size, num_workers=0)
    total_loss, count = 0, 0
    all_eff_iters, all_num_iters = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=DTYPE):
                out = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])
            total_loss += out['loss_dict'].get('loss_lm', 0)
            all_eff_iters.append(out.get('effective_iters', 0))
            all_num_iters.append(out.get('num_iterations', 0))
            count += 1
    avg = total_loss / max(count, 1)
    avg_eff = sum(all_eff_iters) / max(len(all_eff_iters), 1)
    avg_num = sum(all_num_iters) / max(len(all_num_iters), 1)
    return avg, math.exp(min(avg, 20)), avg_eff, avg_num

# 训练态 (低 tau, ACT 加权和 — 所有 max_iter 轮都跑)
model.set_halt_tau(cct_config.halt_tau_end)
model.train()
tl, tp, tei, tni = evaluate(model, eval_data, CFG['batch_size'])

# 推理态 (硬停止 — remainder < threshold 时提前退出)
model.eval()
il, ip, iei, ini = evaluate(model, eval_data, CFG['batch_size'])

print()
print('=' * 70)
print('| Mode  | LM Loss | PPL    | Eff Iters | Num Iters | Loss Gap |')
print('|-------|---------|--------|-----------|-----------|----------|')
print('| Train | %.4f  | %6.2f | %.2f      | %.1f       | -        |' % (tl, tp, tei, tni))
print('| Infer | %.4f  | %6.2f | %.2f      | %.1f       | %+.4f   |' % (il, ip, iei, ini, il - tl))
print('=' * 70)
print()
print('Eff Iters = per-token 期望迭代数 (加权平均)')
print('Num Iters = column 循环实际运行次数 (训练态=max_iter, 推理态可提前退出)')"""),
    ]


def cells_inference() -> List[dict]:
    return [
        md("## 4.5 推理测试: 文本续写"),
        code("""\
# === 4.5 推理: 从 eval 数据截取 prompt -> 续写并对比 ===
model.eval()

@torch.no_grad()
def greedy_generate(model, input_ids, max_new_tokens=128):
    \"\"\"手动贪心解码 (CCTLlamaModel 无 .generate())\"\"\"
    ids = input_ids.clone()
    for _ in range(max_new_tokens):
        if ids.size(1) > CFG['max_seq_len']:
            ids = ids[:, -CFG['max_seq_len']:]
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = model(input_ids=ids)
        next_id = out['logits'][:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    return ids

import random
random.seed(42)
sample_indices = random.sample(range(len(eval_data)), min(5, len(eval_data)))

print('=' * 80)
print('推理测试: 从 eval packed chunks 截取 prompt (前 256 tokens) -> 续写')
print('=' * 80)

for idx_i, si in enumerate(sample_indices):
    input_ids = eval_data[si]['input_ids']
    prompt_len = 256
    prompt_ids = input_ids[:prompt_len].unsqueeze(0).to(device)
    gt_ids = input_ids[prompt_len:prompt_len + 128]
    gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)

    gen_ids = greedy_generate(model, prompt_ids, max_new_tokens=128)
    gen_text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
    prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)

    print('\\n--- Sample %d (eval idx=%d) ---' % (idx_i + 1, si))
    print('[PROMPT last 200 chars] ...%s' % prompt_text[-200:])
    print('[GROUND TRUTH] %s' % gt_text[:300])
    print('[MODEL OUTPUT] %s' % gen_text[:300])
    print()

print('=' * 80)
print('对比续写质量: 模型是否学到了文本分布')
print('=' * 80)"""),
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
all_eff_iters_viz = []

viz_loader = DataLoader(ListDataset(eval_data), batch_size=CFG['batch_size'], num_workers=0)
with torch.no_grad():
    for bi, batch in enumerate(viz_loader):
        if bi >= 20: break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        if out['scores']:
            all_scores_viz.append(out['scores'][-1].float().cpu())
        if out['p_halts']:
            p_halts_cpu = [ph.float().cpu() for ph in out['p_halts']]
            remainder = torch.ones_like(p_halts_cpu[0])
            eff = torch.zeros_like(p_halts_cpu[0])
            for k, ph in enumerate(p_halts_cpu):
                eff += (k + 1) * remainder * ph
                remainder = remainder * (1.0 - ph)
            eff += len(p_halts_cpu) * remainder
            mask = batch['attention_mask'][:, :eff.size(1)].cpu()
            eff_tokens = eff[mask.bool()].numpy()
            all_eff_iters_viz.append(eff_tokens)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

if all_scores_viz:
    scores_flat = torch.cat(all_scores_viz).flatten().numpy()
    axes[0].hist(scores_flat, bins=50, alpha=0.7, color='steelblue')
    axes[0].set_title('Prediction Score Distribution')
    axes[0].set_xlabel('cosine similarity')
    axes[0].axvline(x=np.mean(scores_flat), color='red', ls='--',
                    label='mean=%.3f' % np.mean(scores_flat))
    axes[0].legend()

if all_scores_viz:
    tau_p = cct_config.precision_temperature
    precision = 1.0 - 1.0 / (1.0 + np.exp(-scores_flat / tau_p))
    axes[1].hist(precision, bins=50, alpha=0.7, color='coral')
    axes[1].set_title('Precision Distribution (tau_p=%.2f)' % tau_p)
    axes[1].set_xlabel('precision = 1 - sigma(score/tau)')
    axes[1].axvline(x=np.mean(precision), color='red', ls='--',
                    label='mean=%.3f' % np.mean(precision))
    axes[1].legend()

if all_eff_iters_viz:
    eff_flat = np.concatenate(all_eff_iters_viz)
    axes[2].hist(eff_flat, bins=50, alpha=0.7, color='seagreen')
    axes[2].set_title('Per-Token Effective Iterations (N=%d)' % len(eff_flat))
    axes[2].set_xlabel('effective iterations')
    mean_eff = np.mean(eff_flat)
    std_eff = np.std(eff_flat)
    axes[2].axvline(x=mean_eff, color='red', ls='--',
                    label='mean=%.2f+/-%.2f' % (mean_eff, std_eff))
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
        cells_inference,
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

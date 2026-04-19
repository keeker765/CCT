"""CCT Kaggle 测试 Notebook 构建器

精简版, 用于快速验证训练 pipeline:
  - Internet ON, pip 安装依赖
  - 仅挂载 3 项: Model + Data + Code
  - 20 步训练 + eval
  - 无离线 wheels

用法:
    python -m src.build_notebook_kaggle_test
    python -m src.build_notebook_kaggle_test --steps 50
"""

import json
import argparse
from pathlib import Path
from typing import List

DEFAULT_OUTPUT = "notebooks/cct_kaggle_test.ipynb"

DEFAULT_MODEL = "/kaggle/input/datasets/wukeneth/llama-3-2-1b-base"
DEFAULT_DATA = "/kaggle/input/datasets/wukeneth/cct-pretrain-data"
DEFAULT_CODE = "/kaggle/input/datasets/wukeneth/cct-code"


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


def build_notebook(model_path: str = DEFAULT_MODEL,
                   data_path: str = DEFAULT_DATA,
                   code_path: str = DEFAULT_CODE,
                   max_steps: int = 20) -> dict:
    cells = [
        # ── Header ──
        md(f"""\
# CCT 快速测试 — {max_steps} 步验证 Pipeline

> ⚠️ **需要**: GPU T4/P100, **Internet ON**

**挂载 3 项**:
1. Model: `{model_path}`
2. Data:  `{data_path}` (pre-packed .pt 或 OpenHermes JSON)
3. Code:  `{code_path}` (CCT src/)"""),

        # ── 安装依赖 (在线) ──
        code("""\
# !pip install -q transformers>=4.48.0 accelerate>=1.0.0 datasets>=3.0.0 sentencepiece safetensors"""),

        # ── 挂载 + 代码路径 ──
        code(f"""\
import os, sys, gc, math, time, glob

MOUNT_MODEL = '{model_path}'
MOUNT_DATA  = '{data_path}'
MOUNT_CODE  = '{code_path}'

# 验证挂载
for name, path in [('Model', MOUNT_MODEL), ('Data', MOUNT_DATA), ('Code', MOUNT_CODE)]:
    status = '✓' if os.path.exists(path) else '✗'
    print('%s %s: %s' % (status, name, path))

# 模型目录 (找 config.json)
MODEL_DIR = MOUNT_MODEL
for root, dirs, files in os.walk(MOUNT_MODEL):
    if 'config.json' in files:
        MODEL_DIR = root
        break

# 代码目录
CODE_DIR = MOUNT_CODE
if os.path.exists(os.path.join(MOUNT_CODE, 'src', 'model')):
    CODE_DIR = MOUNT_CODE
elif os.path.basename(MOUNT_CODE) == 'src':
    CODE_DIR = os.path.dirname(MOUNT_CODE)
else:
    # cct-code 里没有 src/ 层 → 创建 symlink 使 from src.xxx 能工作
    for root, dirs, files in os.walk(MOUNT_CODE):
        if 'wrapped_model.py' in files:
            # root = .../model → parent = code root or src
            parent = os.path.dirname(root)
            if os.path.basename(parent) == 'src':
                CODE_DIR = os.path.dirname(parent)
            else:
                # parent 就是代码根目录 (没有 src 层)
                import tempfile
                _tmpdir = tempfile.mkdtemp()
                os.symlink(parent, os.path.join(_tmpdir, 'src'))
                CODE_DIR = _tmpdir
                print('创建 src symlink: %s → %s' % (os.path.join(_tmpdir, 'src'), parent))
            break
sys.path.insert(0, CODE_DIR)
print('模型: %s' % MODEL_DIR)
print('代码: %s' % CODE_DIR)

# 验证
from src.model.wrapped_model import CCTLlamaModel
print('src 导入 ✓')"""),

        # ── 数据检测 + 加载 ──
        code(f"""\
import torch
from torch.utils.data import DataLoader, Dataset, random_split

class ListDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# === 检测数据类型 ===
import glob as _glob
DATA_DIR = None
DATA_FILE = None

if os.path.isdir(MOUNT_DATA):
    pt_files = _glob.glob(os.path.join(MOUNT_DATA, '**', 'step_*.pt'), recursive=True)
    if pt_files:
        DATA_DIR = os.path.dirname(pt_files[0])
        parent = os.path.dirname(DATA_DIR) if os.path.basename(DATA_DIR) == 'train_packed' else DATA_DIR
        DATA_DIR = parent
    else:
        json_files = _glob.glob(os.path.join(MOUNT_DATA, '**', '*.json'), recursive=True)
        json_files = [f for f in json_files if 'metadata' not in f.lower()]
        if json_files:
            DATA_FILE = json_files[0]
elif os.path.isfile(MOUNT_DATA):
    DATA_FILE = MOUNT_DATA

assert DATA_DIR or DATA_FILE, '数据路径无效: %s' % MOUNT_DATA

MAX_STEPS = {max_steps}
CFG = {{
    'max_steps': MAX_STEPS,
    'batch_size': 16,
    'grad_accum': 2,
    'max_seq_len': 2048,
    'lr': 1e-4,
    'new_lr': 5e-4,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'warmup_steps': min(10, MAX_STEPS // 5),
    'log_interval': 5,
    'eval_interval': 100,         # eval every 100 steps
    'eval_chunks': 50,
}}

if DATA_DIR:
    print('Pre-packed 数据: %s' % DATA_DIR)
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, 'train_packed', 'step_*.pt')))
    if not train_files:
        train_files = sorted(glob.glob(os.path.join(DATA_DIR, 'step_*.pt')))
    eval_file = os.path.join(DATA_DIR, 'eval_packed', 'eval_chunks.pt')
    if not os.path.exists(eval_file):
        eval_file = os.path.join(DATA_DIR, 'eval_chunks.pt')
    eval_data = torch.load(eval_file, weights_only=False)
    print('Train files: %d, Eval: %d chunks' % (len(train_files), len(eval_data)))
    sample = torch.load(train_files[0], weights_only=False)
    print('Chunks/file: %d' % len(sample))
    del sample
else:
    print('OpenHermes JSON: %s' % DATA_FILE)
    import json as _json
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    LLAMA3_CHAT_TEMPLATE = (
        '{{% for message in messages %}}'
        '{{% if message["role"] == "user" %}}'
        '<|start_header_id|>user<|end_header_id|>\\n\\n{{{{ message["content"] }}}}<|eot_id|>'
        '{{% elif message["role"] == "assistant" %}}'
        '<|start_header_id|>assistant<|end_header_id|>\\n\\n{{{{ message["content"] }}}}<|eot_id|>'
        '{{% endif %}}{{% endfor %}}'
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
        print('已手动设置 chat template')

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
                    if role == 'human': user_parts.append(val)
                    elif role == 'gpt': gpt_parts.append(val)
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
            if prompt_len > 0: labels[:prompt_len] = -100
            return {{'input_ids': input_ids, 'attention_mask': attn, 'labels': labels}}

    print('加载数据...')
    raw = []
    with open(DATA_FILE, 'r', encoding='utf-8') as _f:
        first = _f.read(2).strip()
        _f.seek(0)
        if first.startswith('['):
            raw = _json.load(_f)
        else:
            for line in _f:
                line = line.strip()
                if line:
                    raw.append(_json.loads(line))
    print('数据量: %d' % len(raw))
    full_ds = SFTDataset(raw, tokenizer, max_length=512)
    eval_sz = int(len(full_ds) * 0.05)
    train_ds, eval_ds = random_split(full_ds, [len(full_ds) - eval_sz, eval_sz])
    eval_data = [eval_ds[i] for i in range(min(len(eval_ds), CFG['eval_chunks']))]
    print('Train: %d, Eval: %d' % (len(train_ds), len(eval_data)))
    train_files = None

eval_loader = DataLoader(ListDataset(eval_data), batch_size=CFG['batch_size'], num_workers=0)
print('\\n数据就绪 ✓')"""),

        # ── 模型初始化 ──
        code("""\
# === 模型初始化 ===
from transformers import LlamaForCausalLM
from src.model.wrapped_model import CCTLlamaModel
from src.model.column_config import CCTConfig

DTYPE = torch.bfloat16

cct_config = CCTConfig(
    max_iter=10,
    lambda_mono=0.1,
    entropy_temp_scale=0.5,
    halt_entropy_threshold=0.3,
    use_ffn_expansion=False,
    use_fusion_graft=True,
    fusion_rank=64,
    bf16=True,
    gradient_checkpointing=True,
    max_seq_len=CFG['max_seq_len'],
    per_device_batch_size=CFG['batch_size'],
    gradient_accumulation_steps=CFG['grad_accum'],
    learning_rate=CFG['lr'],
    new_module_lr=CFG['new_lr'],
    max_steps=CFG['max_steps'],
    warmup_steps=CFG['warmup_steps'],
)

base = LlamaForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=DTYPE, trust_remote_code=True)
model = CCTLlamaModel(base, cct_config)
del base; gc.collect(); torch.cuda.empty_cache()

model.enable_gradient_checkpointing()
device = torch.device('cuda')
model = model.to(device)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(model.get_trainable_params_info())
print('模型就绪 ✓')"""),

        # ── 训练 ──
        code(f"""\
# === 训练 ({max_steps} 步测试) ===
from torch.optim import AdamW
from src.training.scheduler import get_cosine_schedule_with_warmup, compute_halt_threshold

param_groups = model.get_param_groups()
param_groups[0]['lr'] = CFG['lr']
param_groups[1]['lr'] = CFG['new_lr']
optimizer = AdamW(param_groups, weight_decay=CFG['weight_decay'])

lr_sched = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=CFG['warmup_steps'],
    num_training_steps=CFG['max_steps'], min_lr_ratio=0.1)

model.train()
t0 = time.time()
max_steps = CFG['max_steps']
eff_batch = CFG['batch_size'] * CFG['grad_accum']
print('Training %d steps (eff_batch=%d)...' % (max_steps, eff_batch))

avg = {{'total': 0, 'lm': 0, 'mono': 0, 'iters': 0}}
avg_h_per_iter = []
avg_n = 0

if train_files is not None:
    file_idx = 0
    for gs in range(max_steps):

        for _micro in range(CFG['grad_accum']):
            if file_idx >= len(train_files):
                file_idx = 0
            batch_data = torch.load(train_files[file_idx], weights_only=False)
            file_idx += 1
            loader_tmp = DataLoader(ListDataset(batch_data),
                                    batch_size=CFG['batch_size'], shuffle=True)
            batch = next(iter(loader_tmp))
            batch = {{k: v.to(device) for k, v in batch.items()}}

            with torch.amp.autocast('cuda', dtype=DTYPE):
                out = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])
            loss = out['loss'] / CFG['grad_accum']
            loss.backward()

            ld = out['loss_dict']
            avg['total'] += ld.get('loss_total', 0)
            avg['lm'] += ld.get('loss_lm', 0)
            avg['mono'] += ld.get('loss_mono', 0)
            avg['iters'] += out.get('num_iterations', 0)
            avg_h_per_iter.append(out.get('per_iter_entropy', []))
            avg_n += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['max_grad_norm'])
        optimizer.step(); optimizer.zero_grad(); lr_sched.step()

        if (gs + 1) % CFG['log_interval'] == 0:
            n = max(avg_n, 1)
            elapsed = time.time() - t0
            eta_m = (elapsed / (gs + 1)) * (max_steps - gs - 1) / 60
            tokens_done = (gs + 1) * eff_batch * CFG['max_seq_len']
            max_k = max((len(h) for h in avg_h_per_iter), default=0)
            h_avg = []
            for ki in range(max_k):
                vals = [h[ki] for h in avg_h_per_iter if ki < len(h)]
                h_avg.append(sum(vals) / len(vals) if vals else 0)
            h_str = '[' + ','.join(['%.3f' % v for v in h_avg]) + ']'
            print('[Step %d/%d] loss=%.4f | lm=%.4f mono=%.4f | '
                  'H=%s iters=%d | '
                  'lr=%.2e | %.1fM tok | ETA %.0fm' % (
                gs + 1, max_steps, avg['total']/n, avg['lm']/n, avg['mono']/n,
                h_str, avg['iters']/n,
                optimizer.param_groups[0]['lr'], tokens_done / 1e6, eta_m))
            avg = {{'total': 0, 'lm': 0, 'mono': 0, 'iters': 0}}
            avg_h_per_iter = []
            avg_n = 0

        # === Eval ===
        if (gs + 1) % CFG['eval_interval'] == 0:
            model.eval()
            ev_loss, ev_n = 0, 0
            with torch.no_grad():
                for eb in eval_loader:
                    eb = {{k: v.to(device) for k, v in eb.items()}}
                    with torch.amp.autocast('cuda', dtype=DTYPE):
                        eo = model(input_ids=eb['input_ids'],
                                  attention_mask=eb['attention_mask'],
                                  labels=eb['labels'])
                    ev_loss += eo['loss_dict'].get('loss_lm', 0); ev_n += 1
            avg_ev = ev_loss / max(ev_n, 1)
            ppl = math.exp(min(avg_ev, 20))
            print('  [Eval step %d] loss=%.4f PPL=%.2f' % (gs + 1, avg_ev, ppl))
            model.train()

else:
    # OpenHermes DataLoader path
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True, num_workers=0, drop_last=True)
    gs_count = 0
    for bi, batch in enumerate(train_loader):
        if gs_count >= max_steps: break
        batch = {{k: v.to(device) for k, v in batch.items()}}
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        loss = out['loss'] / CFG['grad_accum']
        loss.backward()

        if (bi + 1) % CFG['grad_accum'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['max_grad_norm'])
            optimizer.step(); optimizer.zero_grad(); lr_sched.step()
            gs_count += 1
            ld = out.get('loss_dict', {{}})
            avg['total'] += ld.get('loss_total', 0)
            avg['lm'] += ld.get('loss_lm', 0)
            avg['mono'] += ld.get('loss_mono', 0)
            avg['iters'] += out.get('num_iterations', 0)
            avg_h_per_iter.append(out.get('per_iter_entropy', []))
            avg_n += 1
            if gs_count % CFG['log_interval'] == 0:
                n = max(avg_n, 1)
                elapsed = time.time() - t0
                eta_m = (elapsed / gs_count) * (max_steps - gs_count) / 60 if gs_count > 0 else 0
                tokens_done = gs_count * eff_batch * CFG['max_seq_len']
                max_k = max((len(h) for h in avg_h_per_iter), default=0)
                h_avg = []
                for ki in range(max_k):
                    vals = [h[ki] for h in avg_h_per_iter if ki < len(h)]
                    h_avg.append(sum(vals) / len(vals) if vals else 0)
                h_str = '[' + ','.join(['%.3f' % v for v in h_avg]) + ']'
                print('[Step %d/%d] loss=%.4f | lm=%.4f mono=%.4f | '
                      'H=%s iters=%d | '
                      'lr=%.2e | %.1fM tok | ETA %.0fm' % (
                    gs_count, max_steps, avg['total']/n, avg['lm']/n, avg['mono']/n,
                    h_str, avg['iters']/n,
                    optimizer.param_groups[0]['lr'], tokens_done / 1e6, eta_m))
                avg = {{'total': 0, 'lm': 0, 'mono': 0, 'iters': 0}}
                avg_h_per_iter = []
                avg_n = 0
            if gs_count % CFG['eval_interval'] == 0:
                model.eval()
                ev_loss, ev_n = 0, 0
                with torch.no_grad():
                    for eb in eval_loader:
                        eb = {{k: v.to(device) for k, v in eb.items()}}
                        with torch.amp.autocast('cuda', dtype=DTYPE):
                            eo = model(input_ids=eb['input_ids'],
                                      attention_mask=eb['attention_mask'],
                                      labels=eb['labels'])
                        ev_loss += eo['loss_dict'].get('loss_lm', 0); ev_n += 1
                avg_ev = ev_loss / max(ev_n, 1)
                ppl = math.exp(min(avg_ev, 20))
                print('  [Eval step %d] loss=%.4f PPL=%.2f' % (gs_count, avg_ev, ppl))
                model.train()

elapsed = time.time() - t0
print('\\n训练完成! %d 步, %.1f 秒' % (max_steps, elapsed))"""),

        # ── 评估 ──
        code("""\
# === 评估 ===
model.eval()

def evaluate(model, data_list, batch_size=32):
    loader = DataLoader(ListDataset(data_list), batch_size=batch_size, num_workers=0)
    total_loss, count = 0, 0
    all_iters, all_entropy = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=DTYPE):
                out = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])
            total_loss += out['loss_dict'].get('loss_lm', 0)
            all_iters.append(out.get('num_iterations', 0))
            all_entropy.append(out.get('mean_entropy', 0))
            count += 1
    avg = total_loss / max(count, 1)
    avg_iters = sum(all_iters) / max(len(all_iters), 1)
    avg_ent = sum(all_entropy) / max(len(all_entropy), 1)
    return avg, math.exp(min(avg, 20)), avg_iters, avg_ent

model.train()
tl, tp, tni, teh = evaluate(model, eval_data, CFG['batch_size'])
model.eval()
il, ip, ini, ieh = evaluate(model, eval_data, CFG['batch_size'])

print('=' * 55)
print('| Mode  | LM Loss | PPL    | Iters | H_norm | Gap     |')
print('|-------|---------|--------|-------|--------|---------|')
print('| Train | %.4f  | %6.2f | %d     | %.3f  | -       |' % (tl, tp, tni, teh))
print('| Infer | %.4f  | %6.2f | %d     | %.3f  | %+.4f  |' % (il, ip, ini, ieh, il - tl))
print('=' * 55)
print('\\n✅ Pipeline 验证完成!')"""),

        # ── 推理测试 ──
        code("""\
# === 推理测试 ===
model.eval()

@torch.no_grad()
def greedy_generate(model, input_ids, max_new_tokens=128):
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

# tokenizer 可能在 OpenHermes 路径已初始化, 否则加载
try:
    tokenizer
except NameError:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

import random
random.seed(42)
n_samples = min(3, len(eval_data))
sample_indices = random.sample(range(len(eval_data)), n_samples)

print('=' * 70)
print('推理测试: eval prompt → 续写 128 tokens')
print('=' * 70)

for idx_i, si in enumerate(sample_indices):
    input_ids = eval_data[si]['input_ids']
    prompt_len = min(256, input_ids.size(0) // 2)
    prompt_ids = input_ids[:prompt_len].unsqueeze(0).to(device)
    gt_ids = input_ids[prompt_len:prompt_len + 128]
    gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)

    gen_ids = greedy_generate(model, prompt_ids, max_new_tokens=128)
    gen_text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
    prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)

    print('\\n--- Sample %d ---' % (idx_i + 1))
    print('[PROMPT] ...%s' % prompt_text[-200:])
    print('[TRUTH]  %s' % gt_text[:200])
    print('[MODEL]  %s' % gen_text[:200])"""),

        # ── 可视化 ──
        code("""\
# === 可视化: Score + Precision + Eff Iters 分布 ===
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

model.eval()
all_scores_viz, all_eff_iters_viz = [], []
viz_loader = DataLoader(ListDataset(eval_data), batch_size=CFG['batch_size'], num_workers=0)

with torch.no_grad():
    for bi, batch in enumerate(viz_loader):
        if bi >= 10: break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=DTYPE):
            out = model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'])
        if out.get('scores'):
            all_scores_viz.append(out['scores'][-1].float().cpu())
        if out.get('p_halts'):
            p_halts_cpu = [ph.float().cpu() for ph in out['p_halts']]
            remainder = torch.ones_like(p_halts_cpu[0])
            eff = torch.zeros_like(p_halts_cpu[0])
            for k, ph in enumerate(p_halts_cpu):
                eff += (k + 1) * remainder * ph
                remainder = remainder * (1.0 - ph)
            eff += len(p_halts_cpu) * remainder
            mask = batch['attention_mask'][:, :eff.size(1)].cpu()
            all_eff_iters_viz.append(eff[mask.bool()].numpy())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

if all_scores_viz:
    scores_flat = torch.cat(all_scores_viz).flatten().numpy()
    axes[0].hist(scores_flat, bins=50, alpha=0.7, color='steelblue')
    axes[0].set_title('Score Distribution')
    axes[0].set_xlabel('cosine similarity')
    axes[0].axvline(x=np.mean(scores_flat), color='red', ls='--',
                    label='mean=%.3f' % np.mean(scores_flat))
    axes[0].legend()

    tau_p = cct_config.precision_temperature
    precision = 1.0 - 1.0 / (1.0 + np.exp(-scores_flat / tau_p))
    axes[1].hist(precision, bins=50, alpha=0.7, color='coral')
    axes[1].set_title('Precision (tau=%.2f)' % tau_p)
    axes[1].set_xlabel('precision')
    axes[1].axvline(x=np.mean(precision), color='red', ls='--',
                    label='mean=%.3f' % np.mean(precision))
    axes[1].legend()

if all_eff_iters_viz:
    eff_flat = np.concatenate(all_eff_iters_viz)
    axes[2].hist(eff_flat, bins=50, alpha=0.7, color='seagreen')
    axes[2].set_title('Eff Iters (N=%d)' % len(eff_flat))
    axes[2].set_xlabel('effective iterations')
    m, s = np.mean(eff_flat), np.std(eff_flat)
    axes[2].axvline(x=m, color='red', ls='--', label='%.2f±%.2f' % (m, s))
    axes[2].legend()

plt.tight_layout()
plt.savefig('/kaggle/working/viz_test.png', dpi=100)
plt.show()
print('Saved viz_test.png')"""),
    ]

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


def main():
    parser = argparse.ArgumentParser(description="CCT Kaggle 测试 Notebook")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--code", default=DEFAULT_CODE)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    nb = build_notebook(args.model, args.data, args.code, args.steps)
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("测试 notebook 已生成: %s (%d cells, %d steps)" % (
        p.resolve(), len(nb["cells"]), args.steps))


if __name__ == "__main__":
    main()

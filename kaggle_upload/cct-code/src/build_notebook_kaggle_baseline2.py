"""CCT Kaggle Baseline 2 Notebook 构建器

消融实验: 无循环 5 层顺序前向 (与 CCT v2 对比)
  - 从 Llama-3.2-1B 选取 CCT v2 使用的 5 层, 顺序拼接
  - L0 (Front), L3, L8, L12 (Column), L15 (Back)
  - 无 Column 循环 / Entropy halt / RotaryCycleEmbed / L_mono
  - 仅 L_LM 损失, 全参数统一 lr
  - 支持 pre-packed .pt 和 OpenHermes JSON 数据

用法:
    python -m src.build_notebook_kaggle_baseline2
    python -m src.build_notebook_kaggle_baseline2 --lr 2e-5
"""

import json
import argparse
from pathlib import Path
from typing import List

DEFAULT_OUTPUT = "notebooks/cct_kaggle_baseline2_test.ipynb"

DEFAULT_MODEL = "/kaggle/input/datasets/wukeneth/llama-3-2-1b-base"
DEFAULT_DATA = "/kaggle/input/datasets/wukeneth/cct-data"
DEFAULT_CODE = "/kaggle/input/datasets/wukeneth/cct-code"
DEFAULT_WHEELS = "/kaggle/input/datasets/wukeneth/cct-wheels"


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
                   wheels_path: str = DEFAULT_WHEELS,
                   lr: float = 2e-5) -> dict:
    cells = [
        # ── Header ──
        md(f"""\
# CCT 消融实验 — Baseline 2: 无循环 5 层顺序前向 (Kaggle)

**目的**: 与 CCT v2 对比, 验证 Column 循环机制的增益

**架构**: 从 Llama-3.2-1B (16层) 选取 CCT v2 使用的 5 层, 顺序拼接:
- L0 (CCT 的 Front)
- L3, L8, L12 (CCT 的 Column, 但**不循环**, 仅前向一次)
- L15 (CCT 的 Back)

**与 CCT v2 的区别**:
- 无 Column 循环 (仅顺序前向一次)
- 无 Entropy-based halt / RotaryCycleEmbed / L_mono / per-query temperature
- 仅 L_LM 损失
- 全参数统一 lr={lr} (无分层)

**公平对比**:
- 相同数据, 相同 batch/lr/epochs
- 相同 5 层, 相同初始权重"""),

        # ── 安装 wheels ──
        code(f"""\
# === 0. 安装离线 wheels ===
import subprocess, glob, os

WHEELS_DIR = '{wheels_path}'
wheels = []
for root, dirs, files in os.walk(WHEELS_DIR):
    for f in files:
        if f.endswith('.whl'):
            wheels.append(os.path.join(root, f))

print('从 %s 安装 %d 个 wheel...' % (WHEELS_DIR, len(wheels)))
result = subprocess.run(
    ['pip', 'install', '-q', '--no-deps'] + wheels,
    capture_output=True, text=True)
if result.returncode == 0:
    print('安装成功!')
else:
    print('安装输出:', result.stdout[-500:] if result.stdout else '')
    print('安装错误:', result.stderr[-500:] if result.stderr else '')

import torch, transformers
print('PyTorch:', torch.__version__)
print('Transformers:', transformers.__version__)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0),
          '(%.0f GB)' % (torch.cuda.get_device_properties(0).total_memory / 1e9))"""),

        # ── 挂载 + 路径 ──
        code(f"""\
import os, sys, gc, math, time, glob
import torch
import torch.nn as nn

MOUNT_MODEL = '{model_path}'
MOUNT_DATA  = '{data_path}'
MOUNT_CODE  = '{code_path}'

for name, path in [('Model', MOUNT_MODEL), ('Data', MOUNT_DATA), ('Code', MOUNT_CODE)]:
    status = '✓' if os.path.exists(path) else '✗'
    print('%s %s: %s' % (status, name, path))

MODEL_DIR = MOUNT_MODEL
for root, dirs, files in os.walk(MOUNT_MODEL):
    if 'config.json' in files:
        MODEL_DIR = root
        break

# 代码路径 (仅用于 data utils, baseline2 不需要 CCT 模块)
CODE_DIR = MOUNT_CODE
if os.path.exists(os.path.join(MOUNT_CODE, 'src', 'model')):
    CODE_DIR = MOUNT_CODE
elif os.path.basename(MOUNT_CODE) == 'src':
    CODE_DIR = os.path.dirname(MOUNT_CODE)
else:
    for root, dirs, files in os.walk(MOUNT_CODE):
        if 'wrapped_model.py' in files:
            parent = os.path.dirname(root)
            if os.path.basename(parent) == 'src':
                CODE_DIR = os.path.dirname(parent)
            else:
                import tempfile
                _tmpdir = tempfile.mkdtemp()
                os.symlink(parent, os.path.join(_tmpdir, 'src'))
                CODE_DIR = _tmpdir
            break
sys.path.insert(0, CODE_DIR)
print('模型: %s' % MODEL_DIR)
print('代码: %s' % CODE_DIR)"""),

        # ── 数据加载 ──
        code(f"""\
# === 数据加载 ===
from torch.utils.data import DataLoader, Dataset, random_split

class ListDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

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

CFG = {{
    'max_steps': None,   # auto-compute from data
    'batch_size': 32,
    'grad_accum': 1,
    'max_seq_len': 512,
    'lr': {lr},
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'warmup_steps': 50,
    'log_interval': 20,
    'eval_interval': 100,
    'eval_chunks': 100,
    'save_interval': 500,
    'max_train_hours': 10.5,
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
    train_ds = None
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

    raw = []
    with open(DATA_FILE, 'r', encoding='utf-8') as _f:
        first = _f.read(2).strip()
        _f.seek(0)
        if first.startswith('['):
            raw = _json.load(_f)
        else:
            for line in _f:
                line = line.strip()
                if line: raw.append(_json.loads(line))
    print('数据量: %d' % len(raw))
    full_ds = SFTDataset(raw, tokenizer, max_length=CFG['max_seq_len'])
    eval_sz = int(len(full_ds) * 0.05)
    train_ds, eval_ds = random_split(full_ds, [len(full_ds) - eval_sz, eval_sz])
    eval_data = [eval_ds[i] for i in range(min(len(eval_ds), CFG['eval_chunks']))]
    print('Train: %d, Eval: %d' % (len(train_ds), len(eval_data)))
    train_files = None

eval_loader = DataLoader(ListDataset(eval_data), batch_size=CFG['batch_size'], num_workers=0)

# === 自动计算 max_steps ===
eff_batch = CFG['batch_size'] * CFG['grad_accum']
if train_files is not None:
    sample0 = torch.load(train_files[0], weights_only=False)
    chunks_per_file = len(sample0)
    del sample0
    total_chunks = len(train_files) * chunks_per_file
    CFG['max_steps'] = total_chunks // eff_batch
    total_tokens = CFG['max_steps'] * eff_batch * CFG['max_seq_len']
    print('Auto max_steps: %d files × %d chunks = %d chunks → %d steps (%.2fB tokens, 1 epoch)' % (
        len(train_files), chunks_per_file, total_chunks, CFG['max_steps'], total_tokens / 1e9))
elif train_files is None and CFG['max_steps'] is None:
    total_steps_ds = len(train_ds) // eff_batch
    CFG['max_steps'] = total_steps_ds
    print('Auto max_steps: %d samples / %d = %d steps' % (len(train_ds), eff_batch, CFG['max_steps']))

if CFG['warmup_steps'] > CFG['max_steps'] // 5:
    CFG['warmup_steps'] = max(CFG['max_steps'] // 10, 10)
    print('Adjusted warmup to %d' % CFG['warmup_steps'])

print('\\n数据就绪 ✓')"""),

        # ── 模型构建 ──
        code("""\
# === 模型: 从 Llama-3.2-1B 选取 5 层, 顺序拼接 ===
from transformers import LlamaForCausalLM

DTYPE = torch.bfloat16

SELECTED_LAYERS = [0, 3, 8, 12, 15]  # 与 CCT v2 使用的层完全相同
NUM_SELECTED = len(SELECTED_LAYERS)

base = LlamaForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=DTYPE, trust_remote_code=True)

selected = nn.ModuleList([base.model.layers[i] for i in SELECTED_LAYERS])

# 重新编号 layer_idx, 否则 KV cache 越界
for new_idx, layer in enumerate(selected):
    layer.layer_idx = new_idx
    layer.self_attn.layer_idx = new_idx

base.model.layers = selected
base.config.num_hidden_layers = NUM_SELECTED
base.config.use_cache = False

gc.collect(); torch.cuda.empty_cache()

device = torch.device('cuda')
model = base.to(device)
del base

# gradient checkpointing
model.gradient_checkpointing_enable()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Selected layers: %s (%d layers)' % (SELECTED_LAYERS, NUM_SELECTED))
print('Total params: %.2fM' % (total_p / 1e6))
print('Trainable params: %.2fM (%.2f%%)' % (train_p / 1e6, 100 * train_p / total_p))
print('模型就绪 ✓')"""),

        # ── 训练配置 ──
        code("""\
# === 训练配置 ===
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
lr_sched = CosineAnnealingLR(optimizer, T_max=CFG['max_steps'], eta_min=CFG['lr'] * 0.1)

# warmup wrapper
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_scheduler):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self._step = 0
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            scale = self._step / self.warmup_steps
            for g, blr in zip(self.optimizer.param_groups, self.base_lrs):
                g['lr'] = blr * scale
        else:
            self.base_scheduler.step()

lr_sched = WarmupScheduler(optimizer, CFG['warmup_steps'], lr_sched)

os.makedirs('/kaggle/working/output', exist_ok=True)
best_eval = float('inf')

eff_batch = CFG['batch_size'] * CFG['grad_accum']
tokens_per_step = eff_batch * CFG['max_seq_len']
print('Effective batch: %d seqs = %dK tokens/step' % (eff_batch, tokens_per_step // 1000))
print('Total: %d steps = %.2fB tokens' % (CFG['max_steps'], CFG['max_steps'] * tokens_per_step / 1e9))"""),

        # ── 训练循环 ──
        code(f"""\
# === 训练循环 (标准 LM loss, 无 CCT 机制) ===
import time as _time, threading

_save_thread = None
_save_error = None

def save_checkpoint(model, optimizer, step, best_eval, last_loss, path, block=False):
    global _save_thread, _save_error
    if _save_thread is not None and _save_thread.is_alive():
        _save_thread.join()
    if _save_error is not None:
        print('  ⚠ 上次 checkpoint 保存失败: %s' % _save_error)
        _save_error = None
    _t_snap = _time.time()
    ckpt = {{
        'step': step,
        'best_eval': best_eval,
        'last_train_loss': last_loss,
        'model_state': {{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}},
        'optimizer_state': {{k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
                           for k, v in optimizer.state_dict().items()}},
    }}
    snap_s = _time.time() - _t_snap
    def _write(ckpt, path, step, snap_s):
        global _save_error
        try:
            _t_w = _time.time()
            tmp_path = path + '.tmp'
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, path)
            size_mb = os.path.getsize(path) / 1e6
            print('  💾 Checkpoint saved: step %d (snap %.1fs + write %.1fs, %.0f MB)' % (
                step, snap_s, _time.time() - _t_w, size_mb))
        except Exception as e:
            _save_error = str(e)
    if block:
        _write(ckpt, path, step, snap_s)
    else:
        _save_thread = threading.Thread(target=_write, args=(ckpt, path, step, snap_s), daemon=True)
        _save_thread.start()

model.train()
_t0 = _time.time()
max_steps = CFG['max_steps']
max_hours = CFG['max_train_hours']
ckpt_path = '/kaggle/working/output/baseline2_checkpoint.pt'

print('Training for %d steps (Baseline 2: 5-layer sequential, lr={lr})...' % max_steps)

avg_loss, avg_n = 0, 0
_last_loss = 0.0
_timeout_exit = False

if train_files is not None:
    # === Pre-packed 数据路径 ===
    file_idx = 0
    for gs in range(max_steps):
        for _micro in range(CFG['grad_accum']):
            if file_idx >= len(train_files):
                file_idx = 0
                print('[Step %d] Train data exhausted, restarting from file 0' % gs)
            batch_data = torch.load(train_files[file_idx], weights_only=False)
            file_idx += 1
            batch_loader = DataLoader(ListDataset(batch_data),
                                     batch_size=CFG['batch_size'], shuffle=True)
            batch = next(iter(batch_loader))
            batch = {{k: v.to(device) for k, v in batch.items()}}

            with torch.amp.autocast('cuda', dtype=DTYPE):
                out = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])
            loss = out.loss / CFG['grad_accum']
            loss.backward()

            avg_loss += out.loss.item()
            avg_n += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['max_grad_norm'])
        optimizer.step(); optimizer.zero_grad(); lr_sched.step()

        if (gs + 1) % CFG['log_interval'] == 0:
            n = max(avg_n, 1)
            elapsed = _time.time() - _t0
            eta_m = (elapsed / (gs + 1)) * (max_steps - gs - 1) / 60
            tokens_done = (gs + 1) * eff_batch * CFG['max_seq_len']
            _last_loss = avg_loss / n
            print('[Step %d/%d] loss=%.4f | lr=%.2e | %.1fM tok | ETA %.0fm' % (
                gs + 1, max_steps, avg_loss/n,
                optimizer.optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'optimizer') else optimizer.param_groups[0]['lr'],
                tokens_done / 1e6, eta_m))
            avg_loss, avg_n = 0, 0

        # === 超时检查 ===
        _elapsed_h = (_time.time() - _t0) / 3600
        if _elapsed_h >= max_hours:
            print('\\n⏰ 超时 (%.1f h >= %.1f h), 保存 checkpoint 并退出...' % (_elapsed_h, max_hours))
            save_checkpoint(model, optimizer, gs + 1, best_eval, _last_loss, ckpt_path, block=True)
            _timeout_exit = True
            break

        # === 定期 Eval ===
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
                    ev_loss += eo.loss.item(); ev_n += 1
            avg_ev = ev_loss / max(ev_n, 1)
            ppl = math.exp(min(avg_ev, 20))
            print('  [Eval step %d] loss=%.4f PPL=%.2f' % (gs + 1, avg_ev, ppl))
            if avg_ev < best_eval:
                best_eval = avg_ev
                torch.save(model.state_dict(), '/kaggle/working/output/baseline2_best.pt')
                print('  New best!')
            model.train()

        # === 定期 Checkpoint ===
        if (gs + 1) % CFG['save_interval'] == 0:
            save_checkpoint(model, optimizer, gs + 1, best_eval, _last_loss, ckpt_path)

else:
    # === OpenHermes DataLoader 路径 ===
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=True)
    gs_count = 0
    _stop = False
    for epoch in range(10):
        if _stop: break
        for bi, batch in enumerate(train_loader):
            if gs_count >= max_steps:
                _stop = True; break
            batch = {{k: v.to(device) for k, v in batch.items()}}
            with torch.amp.autocast('cuda', dtype=DTYPE):
                out = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])
            loss = out.loss / CFG['grad_accum']
            loss.backward()
            if (bi + 1) % CFG['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['max_grad_norm'])
                optimizer.step(); optimizer.zero_grad(); lr_sched.step()
                gs_count += 1
                avg_loss += out.loss.item()
                avg_n += 1
                if gs_count % CFG['log_interval'] == 0:
                    n = max(avg_n, 1)
                    elapsed = _time.time() - _t0
                    eta_m = (elapsed / gs_count) * (max_steps - gs_count) / 60 if gs_count > 0 else 0
                    tokens_done = gs_count * eff_batch * CFG['max_seq_len']
                    _last_loss = avg_loss / n
                    print('[Step %d/%d] loss=%.4f | lr=%.2e | %.1fM tok | ETA %.0fm' % (
                        gs_count, max_steps, avg_loss/n,
                        optimizer.optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'optimizer') else optimizer.param_groups[0]['lr'],
                        tokens_done / 1e6, eta_m))
                    avg_loss, avg_n = 0, 0
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
                            ev_loss += eo.loss.item(); ev_n += 1
                    avg_ev = ev_loss / max(ev_n, 1)
                    ppl = math.exp(min(avg_ev, 20))
                    print('  [Eval step %d] loss=%.4f PPL=%.2f' % (gs_count, avg_ev, ppl))
                    if avg_ev < best_eval:
                        best_eval = avg_ev
                        torch.save(model.state_dict(), '/kaggle/working/output/baseline2_best.pt')
                        print('  New best!')
                    model.train()
                # 超时检查
                _elapsed_h = (_time.time() - _t0) / 3600
                if _elapsed_h >= max_hours:
                    print('\\n⏰ 超时')
                    _stop = True; break

elapsed = _time.time() - _t0
if not _timeout_exit:
    torch.save(model.state_dict(), '/kaggle/working/output/baseline2_final.pt')
print('\\n训练完成! %d 步, %.1f 分钟, best_eval=%.4f' % (max_steps, elapsed / 60, best_eval))"""),

        # ── 评估 ──
        code("""\
# === 评估 ===
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
print('=' * 60)
print('| Baseline 2 — Ablation (No Column Recycling)           |')
print('=' * 60)
print('| LM Loss   | PPL     | Layers | Params        |')
print('|-----------|---------|--------|---------------|')
total_p = sum(p.numel() for p in model.parameters())
print('| %.4f   | %.2f  | %d      | %.2fM        |' % (avg_ev, ppl, NUM_SELECTED, total_p / 1e6))
print('=' * 60)"""),

        # ── 保存输出 ──
        code("""\
# === 保存到 Kaggle Output ===
import shutil, json

output_dir = '/kaggle/working/output'

summary = {
    'experiment': 'baseline2_no_recycling',
    'layers': SELECTED_LAYERS,
    'num_layers': NUM_SELECTED,
    'best_eval_loss': best_eval,
    'best_ppl': math.exp(min(best_eval, 20)),
    'total_params': sum(p.numel() for p in model.parameters()),
    'config': {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in CFG.items()},
}
with open(os.path.join(output_dir, 'baseline2_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('输出目录:')
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    size = os.path.getsize(fpath) / 1e6
    print('  %s (%.1f MB)' % (f, size))"""),
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
    parser = argparse.ArgumentParser(description="Build Kaggle Baseline2 notebook")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--code", default=DEFAULT_CODE)
    parser.add_argument("--wheels", default=DEFAULT_WHEELS)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    nb = build_notebook(args.model, args.data, args.code, args.wheels, args.lr)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Baseline2 notebook 已生成: %s (%d cells, lr=%s)" % (
        out, len(nb["cells"]), args.lr))


if __name__ == "__main__":
    main()

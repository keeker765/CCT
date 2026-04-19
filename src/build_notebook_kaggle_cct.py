"""CCT 正式版 Kaggle Notebook 构建器 (离线版)

基于最新 build_notebook.py 改写, 适配 Kaggle 离线环境:
  - 挂载 4 个 Kaggle 资源 (均可通过 CLI 参数配置):
    1. Model:   预训练模型 (meta-llama/Llama-3.2-1B)
    2. Data:    训练数据 (pre-packed .pt 或 OpenHermes JSON)
    3. Code:    CCT 源代码 (src/ 目录)
    4. Wheels:  离线 pip 依赖包

前置步骤:
  1. python scripts/package_deps.py --upload     # 打包+上传依赖
  2. python scripts/upload_kaggle_datasets.py --code --update  # 上传代码
  3. Colab 运行 data_prep notebook → 上传 prepacked data 到 Kaggle
  4. 在 Kaggle Notebook 添加上述 3 个 Dataset + 1 个 Model

用法:
    python -m src.build_notebook_kaggle_cct
    python -m src.build_notebook_kaggle_cct --model /kaggle/input/llama-3-2-1b
    python -m src.build_notebook_kaggle_cct --data /kaggle/input/cct-prepacked-data
"""

import json
import argparse
from pathlib import Path
from typing import List

DEFAULT_OUTPUT = "notebooks/cct_kaggle.ipynb"
VERSION = "cct-kaggle-v2"

# 默认 Kaggle 挂载路径
DEFAULT_MODEL = "/kaggle/input/meta-llama-llama-3.2-1b"
DEFAULT_DATA = "/kaggle/input/cct-prepacked-data"
DEFAULT_CODE = "/kaggle/input/cct-code"
DEFAULT_WHEELS = "/kaggle/input/cct-wheels"


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


# ── Cells ─────────────────────────────────────────────
def cells_header() -> List[dict]:
    return [
        md("""\
# CCT (Cortical Column Transformer) — Kaggle 离线训练

**架构**: Fixed Front (L0-L1) → Column (L2,L7,L12 循环 K=5 次) → Fixed Back (L14-L15)
- **Predictor + AnchorMLP**: 预测编码误差信号
- **L6 Precision**: 误差驱动的注意力增益调制
- **HaltHead**: ACT 软停止 + tau 退火二值化
- **RotaryCycleEmbed**: phi 黄金比例旋转循环嵌入
- **Cross-Layer Fusion**: 2x FFN 加宽 (donor 层知识融合)

**Kaggle 挂载 (4 项)**:
1. Model: 预训练模型 (Llama-3.2-1B)
2. Dataset: 训练数据 (pre-packed 或 OpenHermes)
3. Dataset: CCT 源代码
4. Dataset: 离线 pip 依赖包

**Settings**: GPU T4/P100, Internet OFF"""),
    ]


def cells_mount_config(model_path: str, data_path: str,
                       code_path: str, wheels_path: str) -> List[dict]:
    return [
        md("## 0. 挂载配置"),
        code(f"""\
# ╔══════════════════════════════════════════════════╗
# ║  挂载路径配置 — 修改下方路径匹配你的 Kaggle 资源  ║
# ╚══════════════════════════════════════════════════╝
import os, sys

MOUNT_MODEL  = '{model_path}'   # 预训练模型
MOUNT_DATA   = '{data_path}'    # 训练数据
MOUNT_CODE   = '{code_path}'    # CCT 源代码
MOUNT_WHEELS = '{wheels_path}'  # 离线 pip 依赖

# 验证挂载
for name, path in [('Model', MOUNT_MODEL), ('Data', MOUNT_DATA),
                    ('Code', MOUNT_CODE), ('Wheels', MOUNT_WHEELS)]:
    if os.path.exists(path):
        print('✓ %s: %s' % (name, path))
    else:
        print('✗ %s: %s (未找到!)' % (name, path))

# 解析模型目录 (查找 config.json 所在层级)
MODEL_DIR = MOUNT_MODEL
for root, dirs, files in os.walk(MOUNT_MODEL):
    if 'config.json' in files:
        MODEL_DIR = root
        break
print('\\n模型目录: %s' % MODEL_DIR)

# 解析代码目录
CODE_DIR = MOUNT_CODE
if os.path.exists(os.path.join(MOUNT_CODE, 'src')):
    CODE_DIR = MOUNT_CODE
elif os.path.basename(MOUNT_CODE) == 'src':
    CODE_DIR = os.path.dirname(MOUNT_CODE)
sys.path.insert(0, CODE_DIR)
print('代码目录: %s' % CODE_DIR)

# 解析数据目录/文件
DATA_DIR = None
DATA_FILE = None
if os.path.isdir(MOUNT_DATA):
    # 检查是否有 pre-packed .pt 文件
    import glob as _glob
    pt_files = _glob.glob(os.path.join(MOUNT_DATA, '**', 'step_*.pt'), recursive=True)
    if pt_files:
        DATA_DIR = os.path.dirname(pt_files[0])
        # 如果在子目录 train_packed/ 下
        parent = os.path.dirname(DATA_DIR) if os.path.basename(DATA_DIR) == 'train_packed' else DATA_DIR
        DATA_DIR = parent
    else:
        # 检查 JSON
        json_files = _glob.glob(os.path.join(MOUNT_DATA, '**', '*.json'), recursive=True)
        json_files = [f for f in json_files if 'metadata' not in f.lower()]
        if json_files:
            DATA_FILE = json_files[0]
elif os.path.isfile(MOUNT_DATA):
    DATA_FILE = MOUNT_DATA

assert DATA_DIR or DATA_FILE, '数据路径无效: %s' % MOUNT_DATA
if DATA_DIR:
    print('数据 (pre-packed): %s' % DATA_DIR)
else:
    print('数据 (JSON): %s' % DATA_FILE)"""),
    ]


def cells_install_deps() -> List[dict]:
    return [
        md("## 0.5 离线依赖安装"),
        code("""\
# === 离线安装依赖 (在 import 其他库之前) ===
import subprocess, os
from pathlib import Path

if os.path.exists(MOUNT_WHEELS):
    wheels = sorted(Path(MOUNT_WHEELS).glob('*.whl'))
    if wheels:
        print('从 %s 安装 %d 个 wheel...' % (MOUNT_WHEELS, len(wheels)))
        # 使用 --no-index 确保纯离线, --find-links 解析依赖
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--quiet',
             '--no-index', '--find-links', MOUNT_WHEELS,
             'transformers>=4.48.0', 'accelerate>=1.0.0',
             'datasets>=3.0.0', 'sentencepiece', 'safetensors'],
            capture_output=True, text=True)
        if result.returncode == 0:
            print('安装成功!')
        else:
            print('安装失败 (可能已满足):')
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    else:
        print('WARNING: %s 下无 .whl 文件' % MOUNT_WHEELS)
else:
    print('跳过离线安装 (wheels 路径不存在)')

# 验证关键库版本
import torch, transformers
print('\\nPyTorch: %s' % torch.__version__)
print('Transformers: %s' % transformers.__version__)
if torch.cuda.is_available():
    print('GPU: %s (%.0f GB)' % (torch.cuda.get_device_name(0),
                                   torch.cuda.get_device_properties(0).total_memory / 1e9))"""),
    ]


def cells_data_model() -> List[dict]:
    return [
        md("## 1. 数据加载 + 模型初始化"),
        code("""\
# === 1. 数据 + 模型 ===
import sys, math, time, gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.model.wrapped_model import CCTLlamaModel
from src.model.column_config import CCTConfig
from src.training.scheduler import compute_halt_tau, get_cosine_schedule_with_warmup

# === 超参数 ===
CFG = {
    'max_steps': 3800,
    'batch_size': 32,
    'grad_accum': 4,
    'max_seq_len': 2048,
    'lr': 1e-4,
    'new_lr': 5e-4,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'log_interval': 20,
    'eval_interval': 200,
    'eval_chunks': 100,
}

# === 数据加载 ===
class ListDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

if DATA_DIR:
    # Pre-packed 数据
    print('Loading pre-packed data from: %s' % DATA_DIR)
    import glob
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, 'train_packed', 'step_*.pt')))
    eval_file = os.path.join(DATA_DIR, 'eval_packed', 'eval_chunks.pt')

    if not train_files:
        train_files = sorted(glob.glob(os.path.join(DATA_DIR, 'step_*.pt')))
        eval_file = os.path.join(DATA_DIR, 'eval_chunks.pt')

    eval_data = torch.load(eval_file, weights_only=False)
    print('Eval: %d chunks' % len(eval_data))

    meta_file = os.path.join(DATA_DIR, 'metadata.json')
    if os.path.exists(meta_file):
        import json
        with open(meta_file) as f:
            meta = json.load(f)
        print('Metadata: %d train chunks, %d steps' % (
            meta.get('total_train_chunks', '?'), meta.get('max_steps', '?')))

    print('Train files: %d' % len(train_files))
    sample_batch = torch.load(train_files[0], weights_only=False)
    print('Sample: %d chunks per file' % len(sample_batch))
    del sample_batch

else:
    # OpenHermes JSON fallback
    print('Loading OpenHermes JSON: %s' % DATA_FILE)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    LLAMA3_CHAT_TEMPLATE = (
        '{% for message in messages %}'
        '{% if message["role"] == "user" %}'
        '<|start_header_id|>user<|end_header_id|>\\n\\n{{ message["content"] }}<|eot_id|>'
        '{% elif message["role"] == "assistant" %}'
        '<|start_header_id|>assistant<|end_header_id|>\\n\\n{{ message["content"] }}<|eot_id|>'
        '{% endif %}{% endfor %}'
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
                prompt_msgs = [{'role': 'user', 'content': user_msg}]
                full_msgs = prompt_msgs + [{'role': 'assistant', 'content': gpt_msg}]
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
            return {'input_ids': input_ids, 'attention_mask': attn, 'labels': labels}

    raw = load_dataset('json', data_files=DATA_FILE, split='train')
    full_ds = SFTDataset(raw, tokenizer, max_length=512)
    eval_sz = int(len(full_ds) * 0.05)
    train_ds, eval_ds = random_split(full_ds, [len(full_ds) - eval_sz, eval_sz])
    eval_data = [eval_ds[i] for i in range(min(len(eval_ds), CFG['eval_chunks']))]
    print('Train: %d, Eval: %d' % (len(train_ds), len(eval_data)))
    train_files = None

eval_loader = DataLoader(ListDataset(eval_data), batch_size=CFG['batch_size'], num_workers=0)"""),
    ]


def cells_model_init() -> List[dict]:
    return [
        md("## 1.5 模型初始化"),
        code("""\
# === 1.5 模型 ===
from transformers import LlamaForCausalLM

DTYPE = torch.bfloat16

cct_config = CCTConfig(
    max_iter=5,
    lambda_pred=0.1,
    lambda_entropy=0.0,
    lambda_ponder=0.0,
    use_ponder_cost=False,
    use_ffn_expansion=True,   # 启用 Cross-Layer Fusion
    column_d_ff=16384,        # 2x FFN 加宽
    widen_mode='cross',
    donor_init_scale=0.1,
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

base = LlamaForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=DTYPE, trust_remote_code=True)
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
        md("## 2. 训练配置"),
        code("""\
# === 2. 训练配置 ===
from torch.optim import AdamW

param_groups = model.get_param_groups()
param_groups[0]['lr'] = CFG['lr']
param_groups[1]['lr'] = CFG['new_lr']
optimizer = AdamW(param_groups, weight_decay=CFG['weight_decay'])

lr_sched = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CFG['warmup_steps'],
    num_training_steps=CFG['max_steps'],
    min_lr_ratio=0.1,
)

os.makedirs('/kaggle/working/output', exist_ok=True)
best_eval = float('inf')

eff_batch = CFG['batch_size'] * CFG['grad_accum']
tokens_per_step = eff_batch * CFG['max_seq_len']
print('Effective batch: %d seqs = %dK tokens/step' % (eff_batch, tokens_per_step // 1000))
print('Total: %d steps = %.2fB tokens' % (CFG['max_steps'], CFG['max_steps'] * tokens_per_step / 1e9))"""),
    ]


def cells_train_loop() -> List[dict]:
    return [
        code("""\
# === 训练循环 ===
import time as _time, glob

model.train()
_t0 = _time.time()
max_steps = CFG['max_steps']
print('Training for %d optimizer steps (grad_accum=%d)...' % (max_steps, CFG['grad_accum']))

avg = {'total': 0, 'lm': 0, 'pred': 0, 'eff_iters': 0, 'eff_std': 0, 'score_std': 0}
avg_n = 0

if train_files is not None:
    # === Pre-packed 数据路径 ===
    file_idx = 0
    for gs in range(max_steps):
        tau_halt = compute_halt_tau(gs, max_steps,
                                    cct_config.halt_tau_start, cct_config.halt_tau_end)
        model.set_halt_tau(tau_halt)

        for _micro in range(CFG['grad_accum']):
            if file_idx >= len(train_files):
                file_idx = 0  # 循环
                print('[Step %d] Train data exhausted, restarting from file 0' % gs)

            batch_data = torch.load(train_files[file_idx], weights_only=False)
            file_idx += 1

            # 从 file 中取 batch_size 个 chunks
            loader_tmp = DataLoader(ListDataset(batch_data),
                                    batch_size=CFG['batch_size'], shuffle=True)
            batch = next(iter(loader_tmp))
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['max_grad_norm'])
        optimizer.step(); optimizer.zero_grad(); lr_sched.step()

        if (gs + 1) % CFG['log_interval'] == 0:
            n = max(avg_n, 1)
            elapsed = _time.time() - _t0
            eta_m = (elapsed / (gs + 1)) * (max_steps - gs - 1) / 60
            tokens_done = (gs + 1) * eff_batch * CFG['max_seq_len']
            print('[Step %d/%d] loss=%.4f | lm=%.4f pred=%.4f | '
                  'eff=%.2f+/-%.2f score_std=%.4f tau=%.3f | '
                  'lr=%.2e | %.1fM tok | ETA %.0fm' % (
                gs + 1, max_steps, avg['total']/n, avg['lm']/n, avg['pred']/n,
                avg['eff_iters']/n, avg['eff_std']/n, avg['score_std']/n, tau_halt,
                optimizer.param_groups[0]['lr'], tokens_done / 1e6, eta_m))
            avg = {'total': 0, 'lm': 0, 'pred': 0, 'eff_iters': 0, 'eff_std': 0, 'score_std': 0}
            avg_n = 0

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
                torch.save(model.state_dict(), '/kaggle/working/output/best_model.pt')
                print('  New best!')
            model.train()

else:
    # === OpenHermes DataLoader 路径 ===
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=True)
    total_steps_ds = len(train_loader) * 1 // CFG['grad_accum']
    max_steps = min(max_steps, total_steps_ds)
    print('OpenHermes mode: %d steps' % max_steps)

    gs_count = 0
    for epoch in range(10):
        for bi, batch in enumerate(train_loader):
            if gs_count >= max_steps: break
            batch = {k: v.to(device) for k, v in batch.items()}
            tau_halt = compute_halt_tau(gs_count, max_steps,
                                        cct_config.halt_tau_start, cct_config.halt_tau_end)
            model.set_halt_tau(tau_halt)
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
                ld = out['loss_dict']
                avg['total'] += ld.get('loss_total', 0)
                avg['lm'] += ld.get('loss_lm', 0)
                avg['pred'] += ld.get('loss_pred', 0)
                avg['eff_iters'] += out.get('effective_iters', 0)
                avg_n += 1
                if gs_count % CFG['log_interval'] == 0:
                    n = max(avg_n, 1)
                    print('[Step %d/%d] loss=%.4f lm=%.4f pred=%.4f eff=%.2f tau=%.3f' % (
                        gs_count, max_steps, avg['total']/n, avg['lm']/n,
                        avg['pred']/n, avg['eff_iters']/n, tau_halt))
                    avg = {'total': 0, 'lm': 0, 'pred': 0, 'eff_iters': 0, 'eff_std': 0, 'score_std': 0}
                    avg_n = 0
                if gs_count % CFG['eval_interval'] == 0:
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

torch.save(model.state_dict(), '/kaggle/working/output/final_model.pt')
print('\\nTraining complete! Best eval loss: %.4f (%.1f min total)' % (
    best_eval, (_time.time() - _t0) / 60))"""),
    ]


def cells_eval() -> List[dict]:
    return [
        md("## 3. 评估"),
        code("""\
# === 3. 评估 ===
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

model.set_halt_tau(cct_config.halt_tau_end)
model.train()
tl, tp, tei, tni = evaluate(model, eval_data, CFG['batch_size'])

model.eval()
il, ip, iei, ini = evaluate(model, eval_data, CFG['batch_size'])

print()
print('=' * 70)
print('| Mode  | LM Loss | PPL    | Eff Iters | Num Iters | Loss Gap |')
print('|-------|---------|--------|-----------|-----------|----------|')
print('| Train | %.4f  | %6.2f | %.2f      | %.1f       | -        |' % (tl, tp, tei, tni))
print('| Infer | %.4f  | %6.2f | %.2f      | %.1f       | %+.4f   |' % (il, ip, iei, ini, il - tl))
print('=' * 70)"""),
    ]


def cells_output() -> List[dict]:
    return [
        md("## 4. 输出保存"),
        code("""\
# === 4. 保存输出 ===
import shutil

output_dir = '/kaggle/working/output'
files = os.listdir(output_dir) if os.path.exists(output_dir) else []
print('Output files:')
for f in files:
    size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1e6
    print('  %s (%.1f MB)' % (f, size_mb))

print('\\nDone! 在 Kaggle Output 中下载结果文件。')"""),
    ]


# ── 组装 Notebook ─────────────────────────────────────
def build_notebook(model_path: str = DEFAULT_MODEL,
                   data_path: str = DEFAULT_DATA,
                   code_path: str = DEFAULT_CODE,
                   wheels_path: str = DEFAULT_WHEELS) -> dict:
    cells = []
    for fn in [
        cells_header,
        lambda: cells_mount_config(model_path, data_path, code_path, wheels_path),
        cells_install_deps,
        cells_data_model,
        cells_model_init,
        cells_train_config,
        cells_train_loop,
        cells_eval,
        cells_output,
    ]:
        cells.extend(fn())

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


def save_notebook(output_path: str, **kwargs) -> str:
    nb = build_notebook(**kwargs)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return str(p.resolve())


def main():
    parser = argparse.ArgumentParser(description="构建 CCT Kaggle 正式版 Notebook")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="模型挂载路径 (默认: %s)" % DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="数据挂载路径 (默认: %s)" % DEFAULT_DATA)
    parser.add_argument("--code", default=DEFAULT_CODE,
                        help="代码挂载路径 (默认: %s)" % DEFAULT_CODE)
    parser.add_argument("--wheels", default=DEFAULT_WHEELS,
                        help="依赖挂载路径 (默认: %s)" % DEFAULT_WHEELS)
    args = parser.parse_args()

    path = save_notebook(args.output,
                         model_path=args.model,
                         data_path=args.data,
                         code_path=args.code,
                         wheels_path=args.wheels)
    nb = build_notebook(args.model, args.data, args.code, args.wheels)
    print("CCT Kaggle notebook 已生成: %s" % path)
    print("   %d cells, 版本 %s" % (len(nb["cells"]), VERSION))
    print()
    print("挂载配置:")
    print("  Model:  %s" % args.model)
    print("  Data:   %s" % args.data)
    print("  Code:   %s" % args.code)
    print("  Wheels: %s" % args.wheels)
    print()
    print("Kaggle 使用步骤:")
    print("  1. python scripts/package_deps.py --upload")
    print("  2. python scripts/upload_kaggle_datasets.py --code --update")
    print("  3. Kaggle 创建新 Notebook → 上传此 .ipynb")
    print("  4. Add Data: cct-code, cct-prepacked-data, cct-wheels")
    print("  5. Add Model: meta-llama/Llama-3.2-1B")
    print("  6. Settings: GPU T4/P100, Internet OFF")


if __name__ == "__main__":
    main()

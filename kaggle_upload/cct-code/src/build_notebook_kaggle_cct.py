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
DEFAULT_MODEL = "/kaggle/input/datasets/wukeneth/llama-3-2-1b-base"
DEFAULT_DATA = "/kaggle/input/datasets/wukeneth/cct-pretrain-data"
DEFAULT_CODE = "/kaggle/input/datasets/wukeneth/cct-code"
DEFAULT_WHEELS = "/kaggle/input/datasets/wukeneth/cct-wheels"
DEFAULT_CHECKPOINT = ""  # 恢复训练时设置, 例如 "/kaggle/input/datasets/wukeneth/cct-checkpoint"


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

**Kaggle 挂载 (4+1 项)**:
1. Model: 预训练模型 (Llama-3.2-1B)
2. Dataset: 训练数据 (pre-packed 或 OpenHermes)
3. Dataset: CCT 源代码
4. Dataset: 离线 pip 依赖包
5. (可选) Dataset: Checkpoint (恢复训练)

**Checkpoint/Resume**: 每 500 步自动保存, 超时 (8h) 自动保存并退出
**Settings**: GPU T4/P100, Internet OFF"""),
    ]


def cells_mount_config(model_path: str, data_path: str,
                       code_path: str, wheels_path: str,
                       checkpoint_path: str = "") -> List[dict]:
    return [
        md("## 0. 挂载配置"),
        code(f"""\
# ╔══════════════════════════════════════════════════╗
# ║  挂载路径配置 — 修改下方路径匹配你的 Kaggle 资源  ║
# ╚══════════════════════════════════════════════════╝
import os, sys

# 减少 CUDA 内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

MOUNT_MODEL  = '{model_path}'   # 预训练模型
MOUNT_DATA   = '{data_path}'    # 训练数据
MOUNT_CODE   = '{code_path}'    # CCT 源代码
MOUNT_WHEELS = '{wheels_path}'  # 离线 pip 依赖
MOUNT_CHECKPOINT = '{checkpoint_path}'  # 恢复训练 (留空=从头训练)

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
                print('创建 src symlink: %s → %s' % (os.path.join(_tmpdir, 'src'), parent))
            break
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
    print('数据 (JSON): %s' % DATA_FILE)

# Checkpoint resume
CHECKPOINT_FILE = None
if MOUNT_CHECKPOINT and os.path.exists(MOUNT_CHECKPOINT):
    import glob as _glob
    ckpt_files = _glob.glob(os.path.join(MOUNT_CHECKPOINT, '**', 'latest_checkpoint.pt'), recursive=True)
    if ckpt_files:
        CHECKPOINT_FILE = ckpt_files[0]
        print('✓ Checkpoint: %s' % CHECKPOINT_FILE)
    elif os.path.isfile(MOUNT_CHECKPOINT):
        CHECKPOINT_FILE = MOUNT_CHECKPOINT
        print('✓ Checkpoint: %s' % CHECKPOINT_FILE)
    else:
        print('⚠ Checkpoint 路径存在但无 latest_checkpoint.pt: %s' % MOUNT_CHECKPOINT)
elif MOUNT_CHECKPOINT:
    print('✗ Checkpoint 路径不存在: %s (从头训练)' % MOUNT_CHECKPOINT)
else:
    print('ℹ 从头训练 (无 checkpoint)')"""),
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
from src.training.scheduler import get_cosine_schedule_with_warmup, compute_halt_threshold

# === 超参数 ===
CFG = {
    'max_steps': None,    # None = 自动按数据量算 (1 epoch)
    'batch_size': 16,
    'grad_accum': 4,
    'max_seq_len': 2048,
    'lr': 2e-5,
    'new_lr': 5e-4,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'log_interval': 5,
    'eval_interval': 200,
    'eval_chunks': 100,
    'save_interval': 500,     # checkpoint 保存间隔 (步数)
    'max_train_hours': 8.0,   # 超时自动保存并停止
    # --- CCT 架构超参 (可在 Kaggle 直接调) ---
    'max_iter': 6,
    'lambda_mono': 1.0,     # L_mono 权重 (直接乘 l_mono)
    'entropy_temp_scale': 0.5,
    'entropy_floor': 0.15,
    'halt_threshold_start': 0.5,
    'halt_threshold_end': 0.2,
    'halt_warmup_ratio': 0.1,  # halt threshold warmup 占总步数比例
    'use_fusion_graft': True,
    'fusion_rank': 64,
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
    import json as _json
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

    import json as _json
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
    print('Adjusted warmup to %d' % CFG['warmup_steps'])"""),
    ]


def cells_model_init() -> List[dict]:
    return [
        md("## 1.5 模型初始化"),
        code("""\
# === 1.5 模型 ===
from transformers import LlamaForCausalLM

DTYPE = torch.bfloat16

cct_config = CCTConfig(
    max_iter=CFG['max_iter'],
    lambda_mono=CFG['lambda_mono'],
    entropy_temp_scale=CFG['entropy_temp_scale'],
    entropy_floor=CFG['entropy_floor'],
    halt_threshold_start=CFG['halt_threshold_start'],
    halt_threshold_end=CFG['halt_threshold_end'],
    halt_entropy_threshold=CFG['halt_threshold_end'],
    use_ffn_expansion=False,
    use_fusion_graft=CFG['use_fusion_graft'],
    fusion_rank=CFG['fusion_rank'],
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

# torch.compile (训练时 column 循环固定 max_iter 次, 可以 compile)
USE_COMPILE = False
if USE_COMPILE:
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print('torch.compile: ON (reduce-overhead)')
    except Exception as e:
        print('torch.compile 失败, 跳过: %s' % e)
        USE_COMPILE = False

print(model.get_trainable_params_info() if not USE_COMPILE else
      'Params: compiled model (run forward to see stats)')
print('Column layers: %d, Max iter: %d, seq_len: %d' % (
    len(cct_config.pretrained_column_layers), cct_config.max_iter, CFG['max_seq_len']))
print('SDPA + TF32 + compile=%s' % USE_COMPILE)

# Flash Attention 诊断
_fa_ok = torch.backends.cuda.flash_sdp_enabled()
_mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
_math = torch.backends.cuda.math_sdp_enabled()
print('\\n=== SDPA Backends ===')
print('  Flash SDP:        %s' % ('✓' if _fa_ok else '✗'))
print('  Mem-efficient SDP: %s' % ('✓' if _mem_eff else '✗'))
print('  Math SDP:         %s' % ('✓' if _math else '✗'))
# Column attention 走 is_causal=True → Flash SDP (如果 GPU 支持)
if _fa_ok:
    print('Column layers 将使用 Flash Attention (is_causal=True)')
elif _mem_eff:
    print('Column layers 将使用 Memory-Efficient Attention')
else:
    print('⚠ 仅 Math backend 可用, 性能较低')"""),
    ]


def cells_vram_profile() -> List[dict]:
    return [
        md("## 1.8 VRAM 诊断 (可选)"),
        code("""\
# === VRAM 诊断 — 设 RUN_VRAM_PROFILE=True 开启 ===
RUN_VRAM_PROFILE = False

if RUN_VRAM_PROFILE:
    import torch, gc

    def _mb(b): return b / 1024**2
    def _gb(b): return b / 1024**3
    def _vram():
        return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()

    a0, _ = _vram()
    print('=== VRAM 诊断 ===')
    print('[0] Model on GPU: %.1f MB' % _mb(a0))

    # --- 模拟一次 forward ---
    bs_test = CFG['batch_size']
    sl = CFG['max_seq_len']
    print('\\n[1] Forward pass (bs=%d, seq=%d)...' % (bs_test, sl))

    dummy_ids = torch.randint(0, 32000, (bs_test, sl), device='cuda')
    dummy_mask = torch.ones(bs_test, sl, dtype=torch.long, device='cuda')
    dummy_labels = dummy_ids.clone()

    model.train()
    torch.cuda.reset_peak_memory_stats()
    a_pre, _ = _vram()

    with torch.amp.autocast('cuda', dtype=DTYPE):
        out = model(input_ids=dummy_ids, attention_mask=dummy_mask, labels=dummy_labels)
    loss = out['loss']

    a_fwd, peak_fwd = _vram()
    print('  Allocated after fwd: %.1f MB (delta: +%.1f MB)' % (_mb(a_fwd), _mb(a_fwd - a_pre)))
    print('  Peak during fwd:     %.1f MB' % _mb(peak_fwd))

    # --- backward ---
    print('\\n[2] Backward pass...')
    torch.cuda.reset_peak_memory_stats()
    a_pre_bwd, _ = _vram()
    loss.backward()
    a_bwd, peak_bwd = _vram()
    print('  Allocated after bwd: %.1f MB (delta: +%.1f MB)' % (_mb(a_bwd), _mb(a_bwd - a_pre_bwd)))
    print('  Peak during bwd:     %.1f MB' % _mb(peak_bwd))

    # --- 清理 ---
    del out, loss, dummy_ids, dummy_mask, dummy_labels
    model.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache()

    a_clean, _ = _vram()
    print('\\n[3] After cleanup: %.1f MB' % _mb(a_clean))

    # --- 总结 ---
    print('\\n=== 总结 ===')
    print('Model static:          %.2f GB' % _gb(a0))
    print('Forward peak:          %.2f GB' % _gb(peak_fwd))
    print('Backward peak:         %.2f GB' % _gb(peak_bwd))
    print('Gradient checkpointing: %s' % ('ON' if model._gradient_checkpointing else 'OFF'))
    print('Config: bs=%d, seq=%d, max_iter=%d, grad_ckpt=%s' % (
        bs_test, sl, cct_config.max_iter, model._gradient_checkpointing))

    # --- 预估 optimizer 开销 ---
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optim_est = n_params * 2 * 4  # AdamW: m + v in fp32
    print('\\nOptimizer states est:  %.2f GB (AdamW fp32 m+v for %dM params)' % (
        _gb(optim_est), n_params // 1_000_000))
    print('Predicted total peak:  %.2f GB' % _gb(peak_bwd + optim_est))
    print()
    if peak_bwd + optim_est > 90e9:
        print('⚠ 预计超 90 GB! 建议减小 batch_size 或 max_iter')
    else:
        print('✓ 预计 %.1f GB, 在 %.0f GB GPU 内' % (_gb(peak_bwd + optim_est),
              torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    print('VRAM 诊断已跳过 (设 RUN_VRAM_PROFILE=True 开启)')"""),
    ]


def cells_vram_cleanup() -> List[dict]:
    return [
        md("### 🧹 显存清理"),
        code("""\
# === 显存清理 (训练前/切换任务时运行) ===
import gc, torch

# 清理所有临时变量
for _name in list(globals()):
    _obj = globals()[_name]
    if isinstance(_obj, torch.Tensor) and _name not in ('model',):
        if _obj.is_cuda and _name.startswith(('dummy_', 'out', 'loss')):
            del globals()[_name]

# 清零梯度
if 'model' in dir():
    model.zero_grad(set_to_none=True)

# 清理 optimizer 状态 (如果需要重建 optimizer)
if 'optimizer' in dir():
    optimizer.zero_grad(set_to_none=True)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

_alloc = torch.cuda.memory_allocated() / 1024**3
_resv  = torch.cuda.memory_reserved() / 1024**3
_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'Allocated: {_alloc:.2f} GB | Reserved: {_resv:.2f} GB | Free: {_total - _resv:.2f} GB | Total: {_total:.1f} GB')"""),
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
print('Total: %d steps = %.2fB tokens' % (CFG['max_steps'], CFG['max_steps'] * tokens_per_step / 1e9))

# === Checkpoint Resume ===
import random, numpy as np
start_step = 0
resume_file_idx = 0

if CHECKPOINT_FILE and os.path.exists(CHECKPOINT_FILE):
    print('\\nLoading checkpoint: %s' % CHECKPOINT_FILE)
    _ckpt = torch.load(CHECKPOINT_FILE, weights_only=False, map_location='cpu')
    # 验证配置兼容性
    for _key in ['max_steps', 'warmup_steps', 'grad_accum']:
        _ckpt_v = _ckpt.get(_key)
        _curr_v = CFG.get(_key)
        if _ckpt_v and _ckpt_v != _curr_v:
            print('⚠ %s 不匹配 (ckpt=%s, current=%s)' % (_key, _ckpt_v, _curr_v))
    model.load_state_dict(_ckpt['model_state'])
    optimizer.load_state_dict(_ckpt['optimizer_state'])
    lr_sched.load_state_dict(_ckpt['lr_sched_state'])
    start_step = _ckpt['step']
    resume_file_idx = _ckpt.get('file_idx', start_step * CFG['grad_accum'])
    best_eval = _ckpt.get('best_eval', float('inf'))
    # RNG 状态恢复
    if 'rng_torch' in _ckpt:
        torch.set_rng_state(_ckpt['rng_torch'])
    if 'rng_cuda' in _ckpt:
        torch.cuda.set_rng_state(_ckpt['rng_cuda'])
    if 'rng_python' in _ckpt:
        random.setstate(_ckpt['rng_python'])
    if 'rng_numpy' in _ckpt:
        np.random.set_state(_ckpt['rng_numpy'])
    print('✓ 从 step %d 恢复 (best_eval=%.4f, file_idx=%d)' % (start_step, best_eval, resume_file_idx))
    _prev_loss = _ckpt.get('last_train_loss', '?')
    print('  上次训练 loss: %s' % _prev_loss)
    del _ckpt; gc.collect(); torch.cuda.empty_cache()
else:
    print('\\n从头开始训练 (step 0)')"""),
    ]


def cells_train_loop() -> List[dict]:
    return [
        code("""\
# === 训练循环 ===
import time as _time, glob, random, numpy as np
import threading

_save_thread = None
_save_error = None

def _deep_copy_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    elif isinstance(obj, dict):
        return {k: _deep_copy_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_deep_copy_to_cpu(v) for v in obj)
    return obj

def save_checkpoint(model, optimizer, lr_sched, step, file_idx, best_eval, last_loss, path, block=False):
    global _save_thread, _save_error
    # 检查上一次异步保存
    if _save_thread is not None and _save_thread.is_alive():
        _save_thread.join()
    if _save_error is not None:
        print('  ⚠ 上次 checkpoint 保存失败: %s' % _save_error)
        _save_error = None
    # 快速拷贝所有状态到 CPU
    _t_snap = _time.time()
    ckpt = {
        'step': step,
        'file_idx': file_idx,
        'best_eval': best_eval,
        'last_train_loss': last_loss,
        'max_steps': CFG['max_steps'],
        'warmup_steps': CFG['warmup_steps'],
        'grad_accum': CFG['grad_accum'],
        'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state': _deep_copy_to_cpu(optimizer.state_dict()),
        'lr_sched_state': lr_sched.state_dict(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state(),
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
    }
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
            print('  ❌ Checkpoint 保存失败: %s' % e)
    if block:
        _write(ckpt, path, step, snap_s)
    else:
        _save_thread = threading.Thread(target=_write, args=(ckpt, path, step, snap_s), daemon=True)
        _save_thread.start()

model.train()
_t0 = _time.time()
max_steps = CFG['max_steps']
save_interval = CFG['save_interval']
max_hours = CFG['max_train_hours']
ckpt_path = '/kaggle/working/output/latest_checkpoint.pt'

if start_step > 0:
    print('Resuming from step %d/%d (grad_accum=%d)...' % (start_step, max_steps, CFG['grad_accum']))
else:
    print('Training for %d optimizer steps (grad_accum=%d)...' % (max_steps, CFG['grad_accum']))

avg = {'total': 0, 'lm': 0, 'mono': 0, 'iters': 0, 'iters_std': 0}
avg_h_per_iter = []  # list of lists: each inner list = per-iter entropy for one step
avg_n = 0
_last_loss = 0.0
_timeout_exit = False

if train_files is not None:
    # === Pre-packed 数据路径 ===
    file_idx = resume_file_idx % len(train_files) if resume_file_idx else 0
    for gs in range(start_step, max_steps):

        for _micro in range(CFG['grad_accum']):
            if file_idx >= len(train_files):
                file_idx = 0
                print('[Step %d] Train data exhausted, restarting from file 0' % gs)

            batch_data = torch.load(train_files[file_idx], weights_only=False)
            file_idx += 1

            batch_loader = DataLoader(ListDataset(batch_data),
                                    batch_size=CFG['batch_size'], shuffle=True)
            batch = next(iter(batch_loader))
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
            avg['mono'] += ld.get('loss_mono', 0)
            avg['iters'] += out.get('num_iterations', 0)
            avg['iters_std'] += out.get('halt_iter_std', 0)
            avg_h_per_iter.append(out.get('per_iter_entropy', []))
            avg_n += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['max_grad_norm'])
        optimizer.step(); optimizer.zero_grad(); lr_sched.step()

        if (gs + 1) % CFG['log_interval'] == 0:
            n = max(avg_n, 1)
            elapsed = _time.time() - _t0
            eta_m = (elapsed / (gs + 1 - start_step)) * (max_steps - gs - 1) / 60
            tokens_done = (gs + 1) * eff_batch * CFG['max_seq_len']
            _last_loss = avg['total'] / n
            # 计算每次迭代的平均 entropy (mean±std)
            max_k = max((len(h) for h in avg_h_per_iter), default=0)
            h_parts = []
            for ki in range(max_k):
                means = [h[ki][0] for h in avg_h_per_iter if ki < len(h)]
                stds = [h[ki][1] for h in avg_h_per_iter if ki < len(h)]
                m = sum(means) / len(means) if means else 0
                s = sum(stds) / len(stds) if stds else 0
                h_parts.append('%.3f±%.3f' % (m, s))
            h_str = '[' + ', '.join(h_parts) + ']'
            th = compute_halt_threshold(gs + 1, max_steps, cct_config.halt_threshold_start, cct_config.halt_threshold_end, warmup_steps=int(max_steps * CFG['halt_warmup_ratio']))
            log_msg = ('[Step %d/%d] loss=%.4f | lm=%.4f Δh=%+.4f | '
                  'H=%s iters=%.1f±%.1f th=%.3f | '
                  'lr=%.2e | %.1fM tok | ETA %.0fm' % (
                gs + 1, max_steps, avg['total']/n, avg['lm']/n, avg['mono']/n,
                h_str, avg['iters']/n, avg['iters_std']/n, th,
                optimizer.param_groups[0]['lr'], tokens_done / 1e6, eta_m))
            # 融合量 (每 50 个 log interval 报告一次)
            if cct_config.use_fusion_graft and (gs + 1) % (CFG['log_interval'] * 50) == 0:
                fmag = model.get_fusion_magnitudes()
                if fmag:
                    fstr = ' '.join(['%s=%.4f' % (k, v) for k, v in sorted(fmag.items())[:8]])
                    log_msg += '\\n  [Fusion] ' + fstr
            print(log_msg)
            avg = {'total': 0, 'lm': 0, 'mono': 0, 'iters': 0, 'iters_std': 0}
            avg_h_per_iter = []
            avg_n = 0

        # === 超时检查 (eval 前) ===
        _elapsed_h = (_time.time() - _t0) / 3600
        if _elapsed_h >= max_hours:
            print('\\n⏰ 超时 (%.1f h >= %.1f h), 保存 checkpoint 并退出...' % (_elapsed_h, max_hours))
            save_checkpoint(model, optimizer, lr_sched, gs + 1, file_idx, best_eval, _last_loss, ckpt_path, block=True)
            _timeout_exit = True
            break

        # === 定期 Eval ===
        if (gs + 1) % CFG['eval_interval'] == 0:
            # 退火 halt 阈值
            halt_th = compute_halt_threshold(gs + 1, max_steps,
                                             cct_config.halt_threshold_start,
                                             cct_config.halt_threshold_end,
                                             warmup_steps=int(max_steps * CFG['halt_warmup_ratio']))
            model.set_halt_threshold(halt_th)
            model.eval()
            ev_loss, ev_n, ev_ent, ev_iters = 0, 0, 0, 0
            with torch.no_grad():
                for eb in eval_loader:
                    eb = {k: v.to(device) for k, v in eb.items()}
                    with torch.amp.autocast('cuda', dtype=DTYPE):
                        eo = model(input_ids=eb['input_ids'],
                                  attention_mask=eb['attention_mask'],
                                  labels=eb['labels'])
                    ev_loss += eo['loss_dict'].get('loss_lm', 0)
                    ev_ent += eo.get('mean_entropy', 0)
                    ev_iters += eo.get('num_iterations', 0)
                    ev_n += 1
            avg_ev = ev_loss / max(ev_n, 1)
            ppl = math.exp(min(avg_ev, 20))
            avg_ev_ent = ev_ent / max(ev_n, 1)
            avg_ev_iters = ev_iters / max(ev_n, 1)
            print('  [Eval step %d] loss=%.4f PPL=%.2f H=%.3f iters=%.1f th=%.3f' % (
                gs + 1, avg_ev, ppl, avg_ev_ent, avg_ev_iters, halt_th))
            if avg_ev < best_eval:
                best_eval = avg_ev
                torch.save(model.state_dict(), '/kaggle/working/output/best_model.pt')
                print('  New best!')
            model.train()

        # === 定期 Checkpoint ===
        if (gs + 1) == start_step + 10 or (gs + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, lr_sched, gs + 1, file_idx, best_eval, _last_loss, ckpt_path)

else:
    # === OpenHermes DataLoader 路径 ===
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=True)
    total_steps_ds = len(train_loader) * 1 // CFG['grad_accum']
    max_steps = min(max_steps, total_steps_ds)
    print('OpenHermes mode: %d steps' % max_steps)

    gs_count = start_step
    for epoch in range(10):
        for bi, batch in enumerate(train_loader):
            if gs_count >= max_steps: break
            batch = {k: v.to(device) for k, v in batch.items()}
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
                ld = out['loss_dict']
                avg['total'] += ld.get('loss_total', 0)
                avg['lm'] += ld.get('loss_lm', 0)
                avg['mono'] += ld.get('loss_mono', 0)
                avg['iters'] += out.get('num_iterations', 0)
                avg['iters_std'] += out.get('halt_iter_std', 0)
                avg_h_per_iter.append(out.get('per_iter_entropy', []))
                avg_n += 1
                if gs_count % CFG['log_interval'] == 0:
                    n = max(avg_n, 1)
                    elapsed = _time.time() - _t0
                    eta_m = (elapsed / (gs_count - start_step)) * (max_steps - gs_count) / 60 if gs_count > start_step else 0
                    tokens_done = gs_count * eff_batch * CFG['max_seq_len']
                    _last_loss = avg['total'] / n
                    max_k = max((len(h) for h in avg_h_per_iter), default=0)
                    h_parts = []
                    for ki in range(max_k):
                        means = [h[ki][0] for h in avg_h_per_iter if ki < len(h)]
                        stds = [h[ki][1] for h in avg_h_per_iter if ki < len(h)]
                        m = sum(means) / len(means) if means else 0
                        s = sum(stds) / len(stds) if stds else 0
                        h_parts.append('%.3f±%.3f' % (m, s))
                    h_str = '[' + ', '.join(h_parts) + ']'
                    th = compute_halt_threshold(gs_count, max_steps, cct_config.halt_threshold_start, cct_config.halt_threshold_end, warmup_steps=int(max_steps * CFG['halt_warmup_ratio']))
                    log_msg = ('[Step %d/%d] loss=%.4f | lm=%.4f Δh=%+.4f | '
                          'H=%s iters=%.1f±%.1f th=%.3f | '
                          'lr=%.2e | %.1fM tok | ETA %.0fm' % (
                        gs_count, max_steps, avg['total']/n, avg['lm']/n,
                        avg['mono']/n, h_str, avg['iters']/n, avg['iters_std']/n, th,
                        optimizer.param_groups[0]['lr'], tokens_done / 1e6, eta_m))
                    if cct_config.use_fusion_graft and gs_count % (CFG['log_interval'] * 50) == 0:
                        fmag = model.get_fusion_magnitudes()
                        if fmag:
                            fstr = ' '.join(['%s=%.4f' % (k, v) for k, v in sorted(fmag.items())[:8]])
                            log_msg += '\\n  [Fusion] ' + fstr
                    print(log_msg)
                    avg = {'total': 0, 'lm': 0, 'mono': 0, 'iters': 0, 'iters_std': 0}
                    avg_h_per_iter = []
                    avg_n = 0
                # 超时检查
                _elapsed_h = (_time.time() - _t0) / 3600
                if _elapsed_h >= max_hours:
                    print('\\n⏰ 超时 (%.1f h >= %.1f h), 保存 checkpoint 并退出...' % (_elapsed_h, max_hours))
                    save_checkpoint(model, optimizer, lr_sched, gs_count, 0, best_eval, _last_loss, ckpt_path, block=True)
                    _timeout_exit = True
                    break
                if gs_count % CFG['eval_interval'] == 0:
                    halt_th = compute_halt_threshold(gs_count, max_steps,
                                                     cct_config.halt_threshold_start,
                                                     cct_config.halt_threshold_end,
                                                     warmup_steps=int(max_steps * CFG['halt_warmup_ratio']))
                    model.set_halt_threshold(halt_th)
                    model.eval()
                    ev_loss, ev_n, ev_ent, ev_iters = 0, 0, 0, 0
                    with torch.no_grad():
                        for eb in eval_loader:
                            eb = {k: v.to(device) for k, v in eb.items()}
                            with torch.amp.autocast('cuda', dtype=DTYPE):
                                eo = model(input_ids=eb['input_ids'],
                                          attention_mask=eb['attention_mask'],
                                          labels=eb['labels'])
                            ev_loss += eo['loss_dict'].get('loss_lm', 0)
                            ev_ent += eo.get('mean_entropy', 0)
                            ev_iters += eo.get('num_iterations', 0)
                            ev_n += 1
                    avg_ev = ev_loss / max(ev_n, 1)
                    ppl = math.exp(min(avg_ev, 20))
                    avg_ev_ent = ev_ent / max(ev_n, 1)
                    avg_ev_iters = ev_iters / max(ev_n, 1)
                    print('  [Eval] loss=%.4f PPL=%.2f H=%.3f iters=%.1f th=%.3f' % (
                        avg_ev, ppl, avg_ev_ent, avg_ev_iters, halt_th))
                    if avg_ev < best_eval:
                        best_eval = avg_ev
                        torch.save(model.state_dict(), '/kaggle/working/output/best_model.pt')
                        print('  New best!')
                    model.train()
                if gs_count == start_step + 10 or gs_count % save_interval == 0:
                    save_checkpoint(model, optimizer, lr_sched, gs_count, 0, best_eval, _last_loss, ckpt_path)
        if gs_count >= max_steps or _timeout_exit: break

# 等待异步保存完成
if _save_thread is not None and _save_thread.is_alive():
    print('等待 checkpoint 写入完成...')
    _save_thread.join()
if _save_error is not None:
    print('⚠ 最后一次 checkpoint 保存失败: %s' % _save_error)
    print('  尝试同步保存...')
    save_checkpoint(model, optimizer, lr_sched,
                    gs + 1 if train_files else gs_count,
                    file_idx if train_files else 0,
                    best_eval, _last_loss, ckpt_path, block=True)

if not _timeout_exit:
    torch.save(model.state_dict(), '/kaggle/working/output/final_model.pt')
    print('\\nTraining complete! Best eval loss: %.4f (%.1f min total)' % (
        best_eval, (_time.time() - _t0) / 60))
else:
    print('\\n训练因超时中断 (step %d/%d). 下次运行时:' % (gs + 1 if train_files else gs_count, max_steps))
    print('1. 将 /kaggle/working/output/ 上传为 Kaggle Dataset')
    print('2. 设置 MOUNT_CHECKPOINT 指向该 dataset 路径')
    print('3. 重新运行 notebook 即可自动恢复')"""),
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
    all_num_iters, all_entropy = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=DTYPE):
                out = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])
            total_loss += out['loss_dict'].get('loss_lm', 0)
            all_num_iters.append(out.get('num_iterations', 0))
            all_entropy.append(out.get('mean_entropy', 0))
            count += 1
    avg = total_loss / max(count, 1)
    avg_iters = sum(all_num_iters) / max(len(all_num_iters), 1)
    avg_ent = sum(all_entropy) / max(len(all_entropy), 1)
    return avg, math.exp(min(avg, 20)), avg_iters, avg_ent

model.train()
tl, tp, tni, teh = evaluate(model, eval_data, CFG['batch_size'])

model.eval()
il, ip, ini, ieh = evaluate(model, eval_data, CFG['batch_size'])

print()
print('=' * 70)
print('| Mode  | LM Loss | PPL    | Iters | H_norm | Loss Gap |')
print('|-------|---------|--------|-------|--------|----------|')
print('| Train | %.4f  | %6.2f | %d     | %.3f  | -        |' % (tl, tp, tni, teh))
print('| Infer | %.4f  | %6.2f | %d     | %.3f  | %+.4f   |' % (il, ip, ini, ieh, il - tl))
print('=' * 70)"""),
    ]


def cells_output() -> List[dict]:
    return [
        md("## 4. 推理测试 + 可视化"),
        code("""\
# === 4a. 推理测试 ===
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

import random
random.seed(42)
n_samples = min(5, len(eval_data))
sample_indices = random.sample(range(len(eval_data)), n_samples)

print('=' * 80)
print('推理测试: eval 数据前 256 tokens 做 prompt, 续写 128 tokens')
print('=' * 80)

for idx_i, si in enumerate(sample_indices):
    input_ids = eval_data[si]['input_ids']
    prompt_len = min(256, input_ids.size(0) // 2)
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
    print()"""),

        code("""\
# === 4b. 可视化: Score + Precision + Eff Iters 分布 ===
import matplotlib
matplotlib.use('Agg')
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
plt.savefig('/kaggle/working/output/viz_distributions.png', dpi=150)
plt.show()
print('Saved to /kaggle/working/output/viz_distributions.png')"""),

        md("## 5. 输出保存"),
        code("""\
# === 5. 保存输出 ===
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
                   wheels_path: str = DEFAULT_WHEELS,
                   checkpoint_path: str = DEFAULT_CHECKPOINT) -> dict:
    cells = []
    for fn in [
        cells_header,
        lambda: cells_mount_config(model_path, data_path, code_path, wheels_path, checkpoint_path),
        cells_install_deps,
        cells_data_model,
        cells_model_init,
        cells_vram_profile,
        cells_vram_cleanup,
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
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Checkpoint 路径 (默认: 空=从头训练)")
    args = parser.parse_args()

    path = save_notebook(args.output,
                         model_path=args.model,
                         data_path=args.data,
                         code_path=args.code,
                         wheels_path=args.wheels,
                         checkpoint_path=args.checkpoint)
    nb = build_notebook(args.model, args.data, args.code, args.wheels, args.checkpoint)
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

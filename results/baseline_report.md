# Baseline 实验报告: Llama-3.2-1B 标准 Fine-tuning

## 1. 实验目的

为 CCT (Cortical Column Transformer) 提供**公平的对照基线**。Baseline 使用与 CCT 完全相同的数据集、base model、学习率和训练配置，但**不使用 Column 循环复用**——直接对全部 16 层进行标准 fine-tuning。

## 2. 实验配置

### 2.1 数据集

| 项目 | 详情 |
|------|------|
| **数据集** | [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |
| **格式** | ShareGPT 多轮对话 (human/gpt turns) |
| **总量** | 1,001,551 条 |
| **采样量** | 40,000 条 (max_samples) |
| **训练集** | 38,000 条 (95%) |
| **验证集** | 2,000 条 (5%) |
| **数据处理** | Llama 3 Chat Template 格式化，tokenize + padding |
| **最大序列长度** | 512 tokens |
| **数据类型** | 复杂 SFT 混合 (推理、代码、数学、通用对话) |

### 2.2 模型

| 项目 | 详情 |
|------|------|
| **Base Model** | `unsloth/Llama-3.2-1B` |
| **架构** | LlamaForCausalLM (标准 16 层 Transformer) |
| **总参数** | 1,235,814,400 (1.24B) |
| **可训练参数** | 1,235,814,400 (100% — 全量 fine-tuning) |
| **精度** | bfloat16 (混合精度训练) |

### 2.3 训练超参数

| 超参数 | 值 |
|--------|------|
| **Epochs** | 1 |
| **Batch Size** | 32 |
| **Gradient Accumulation** | 1 |
| **Total Steps** | 1,188 |
| **Learning Rate** | 2e-5 |
| **Optimizer** | AdamW (weight_decay=0.01) |
| **LR Scheduler** | CosineAnnealingLR (T_max=1188) |
| **Max Grad Norm** | 1.0 |
| **Eval Interval** | 每 200 步 |
| **Log Interval** | 每 20 步 |

### 2.4 硬件环境

| 项目 | 详情 |
|------|------|
| **平台** | Google Colab |
| **GPU** | NVIDIA L4 (G4 instance, high-memory) |
| **训练时间** | 14.1 分钟 |

## 3. 训练曲线

### 3.1 训练 Loss

| Step | Loss |
|------|------|
| 20 | 1.1263 |
| 100 | 1.0378 |
| 200 | 0.9673 |
| 400 | 0.8792 |
| 600 | 0.8812 |
| 800 | 0.8697 |
| 1000 | 0.8189 |
| 1180 | 0.8697 |

### 3.2 验证 Loss & PPL

| Step | Eval Loss | PPL |
|------|-----------|-----|
| 200 | 0.9618 | 2.62 |
| 400 | 0.9300 | 2.53 |
| 600 | 0.9154 | 2.50 |
| 800 | 0.9097 | 2.48 |
| 1000 | 0.9085 | 2.48 |

### 3.3 最终评估

| 指标 | 值 |
|------|------|
| **Final Eval Loss** | 0.9085 |
| **Final Eval PPL** | 2.48 |

## 4. 与 CCT 对比框架

| 维度 | Baseline | CCT |
|------|----------|-----|
| **Base Model** | Llama-3.2-1B | Llama-3.2-1B |
| **层数** | 16 层 (全部) | 7 层 (2 Front + 3 Column + 2 Back) |
| **参数量** | 1.24B (100%) | ~0.70B (~56%) |
| **训练方式** | 全量 fine-tuning | Front/Back 冻结 + Column 循环复用 |
| **数据集** | OpenHermes 2.5 (40k) | OpenHermes 2.5 (40k) |
| **学习率** | 2e-5 | 2e-5 (base) / 1e-4 (new modules) |
| **Batch Size** | 32 | 32 |
| **Max Seq Len** | 512 | 512 |
| **Eval Loss** | **0.9085** | (待测) |
| **Eval PPL** | **2.48** | (待测) |

## 5. 关键观察

1. **Loss 下降平稳**: 从 1.13 (step 20) 稳步下降到 0.82 (step 1000)，无异常波动
2. **PPL 趋势**: 2.62 → 2.48，在 step 800 后基本收敛
3. **训练效率**: 14.1 分钟完成 1188 步，约 0.71 秒/步
4. **过拟合风险低**: 训练 loss (0.82-0.87) 与验证 loss (0.91) 差距较小

## 6. 对比要点

> **CCT 的目标不是超越 Baseline，而是用更少的参数 (~56%) 逼近 Baseline 性能。**

- 如果 CCT 的 PPL 接近 2.48（差距 <20%），说明 Column 循环复用机制有效
- CCT 的额外价值在于**自适应计算深度** — 简单 token 少迭代，复杂 token 多迭代
- 即使 PPL 略高，CCT 的推理效率优势（可变深度）也是重要贡献

---

*报告生成日期: 2025-07*
*实验 notebook: baseline_colab.ipynb*

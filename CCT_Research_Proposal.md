# Cortical Column Transformer (CCT): 皮层柱预测编码启发的循环 Transformer

## 一、摘要

我们提出 **Cortical Column Transformer (CCT)**，将大脑皮层柱内的**预测编码环路**引入 Transformer。核心机制：将若干 Transformer 层组成一个"柱"，柱内实现**预测→比较→精度加权反馈→再预测**的迭代循环。与简单层重复不同，CCT 每次循环都由预测误差驱动——误差大的 token 在下一轮获得更高的 attention 权重（Precision Weighting），误差小的 token 快速通过。这直接对应大脑 L6 的精度加权机制（Friston 2005, Feldman & Friston 2010）。

---

## 二、动机与背景

### 2.1 皮层柱微电路

大脑每个皮层柱内存在有方向的信息处理环路（Douglas & Martin 1989, 2004）：

```
丘脑输入
  │
  ├──(慢通路)──→ L4(输入门) ──→ L2/3(内容处理 + 预测生成)
  │                                      │
  └──(快通路)──→ L5(直接接收真实信号)     │
                    │                    │
                    │ ←── L2/3 的预测 ───┘
                    │
                    ↓
              L5 计算误差 = 预测 - 真实
                    │
                    ↓
                  L6(增益控制器)
                    │
                    ├──→ L4(调节下一圈的输入增益)
                    └──→ 丘脑(反馈调控输入源)
```

**关键文献**：
- **Costa et al. 2025 (Nature Communications)**：L2/3 整合过去输入 + top-down 上下文，生成对当前感觉输入的预测。L5 同时接收丘脑直达信号作为"真实值"。误差 = L2/3 预测 - L5 真实输入。这是大脑版的 self-supervised predictive coding。
- **Constantinople & Bruno 2013 (Science)**：L4 和 L5 同时被丘脑激活，L2/3 有延迟——证实双通路。
- **Potjans & Diesmann 柱模型**：L5 活动 ≈ L4 输入 - L2/3 输入，L5 天然执行**减法**。
- **Bastos et al. 2012 (Neuron)**：预测编码的标准微电路模型（见 2.3）。

### 2.2 L6 的角色：增益控制与精度加权

L6 是皮层最深层，它**不处理内容**，而是**控制信号的增益和可信度**。

**三大功能**：

| 功能 | 具体机制 | 核心文献 |
|------|----------|---------|
| **增益控制** | L6 双向调制上层神经元的反应幅度，不改变其调谐特性。通过同时兴奋丘脑（直接）和抑制丘脑（经 TRN 间接），实现对输入信号的"音量旋钮"控制 | Olsen et al. 2012 Nature |
| **状态调制** | L6 根据上下文（运动信号、觉醒状态等）动态增强或抑制丘脑信号。弱激活→抑制，中等激活→增强（甜蜜点），强激活→再抑制 | Dimwamwa et al. 2024 Nat Comm |
| **注意力/精度加权** | L6b 是 top-down 反馈和神经调质的交汇处，控制高阶丘脑皮层回路。在预测编码中，注意力 = 选择性增加预测误差的增益（精度加权），L6 正是执行这一功能的物理基质 | Zolnik et al. 2026 Neuron; Bastos 2012 |

**关键数据**：L6→丘脑的轴突数量是丘脑→皮层的 **10 倍**；L6 提供丘脑感觉核团中 **30-50%** 的突触（Van Horn et al. 2000）。

**一句话总结**：L6 = 精度调制器 — 决定"听多大声"和"信不信这个误差"。

### 2.3 预测编码的核心机制：误差到底怎么反馈？

这是 CCT 的理论核心。根据 Bastos et al. 2012 (Neuron)，预测编码在皮层中的实现如下：

**层级间（区域→区域）的预测编码**：

```
高阶皮层区域                     低阶皮层区域
┌─────────────┐                ┌─────────────┐
│             │                │             │
│  L2/3 误差  │ ←── 前馈(γ) ── │  L2/3 误差  │ ← 误差 = 输入 - 预测
│             │                │             │
│  L5/6 预测  │ ── 反馈(α/β) → │  L5/6 预测  │ ← 来自高阶的预测
│             │                │             │
└─────────────┘                └─────────────┘
```

- **浅层 L2/3** 编码**预测误差**，通过前馈连接（gamma 频率）向上传递
- **深层 L5/6** 编码**预测本身**（条件期望），通过反馈连接（alpha/beta 频率）向下传递
- **反馈的目的**：用预测去**抑制**（解释掉）低阶区域的输入信号，减少预测误差

**柱内（单个区域内部）的预测编码**（Costa 2025 模型）：

```
第 1 步：输入到达
    丘脑 → L4（慢通路，经过输入门控）
    丘脑 → L5（快通路，直达，作为"真实值"参考）

第 2 步：预测生成
    L4 → L2/3：L2/3 整合当前输入 + 过去上下文 → 生成预测

第 3 步：误差计算
    L5 比较：误差 = L2/3→L5 的预测 − L5 的真实输入（丘脑直达）

第 4 步：误差反馈 ← 这是最关键的一步
    L5 → L6：误差信号传给 L6
    L6 执行精度加权（Precision Weighting）→ 调制"哪些位置需要更认真地处理"
    ※ 反馈是调制性的（multiplicative），不是加法性的（Friston 2010; Bastos 2012）

第 5 步：循环
    用更新后的精度权重 + 新的循环相位嵌入重新处理 → 回到第 2 步
    直到误差足够小 → 输出稳定表示
```

**误差反馈的本质**：不是简单地"把误差加回去"，而是通过**精度加权**（Precision Weighting, Friston 2005）改变下一轮处理的方式——prediction error 大的位置获得更高的精度权重 → self-attention 增强 → 更努力地利用上下文信息。

### 2.4 现有工作与空白

| 工作 | 贡献 | 缺失 |
|------|------|------|
| König & Negrello 2026 (arXiv) | 完整的 Transformer↔皮层柱理论映射 | ❌ 没有实现柱内循环 |
| TRC² 2025 (arXiv) | 皮层柱架构 + 丘脑路由 + 海马记忆 | ❌ 柱内无循环 |
| LoopFormer (ICLR 2026) | 共享权重循环 + 时间条件化 | ❌ 无功能分化，非误差驱动 |
| ITT (ACL 2025) | 层内循环 + token 选择 | ❌ 范围太窄，无增益调制 |
| PPG (本项目) | 组级预测误差路由 | ❌ 无柱内循环 |

**CCT 填补的空白**：首次在 Transformer 中实现完整的**预测→比较→精度加权反馈→再预测**循环。与 FiLM/GAIN 的纯增益调制不同，CCT 的精度加权是 **error-driven** 且直接调制 **self-attention**。

---

## 三、模型架构

### 3.1 总体结构

参考维度：**d_model=2048, head_dim=64, Q=32h, KV=8h, d_ff=8192**（参考 Llama 维度设计，不绑定具体基座模型）

```
┌────────────────────────────────────────────────────────────┐
│  Fixed Front：2 层 Transformer                              │ ← 不循环（知识存储/特征提取）
├────────────────────────────────────────────────────────────┤
│  Column：2-3 层 Transformer（共享权重，循环 1-10 次）        │ ← 核心循环单元
│  ※ 2层方案：有效深度 = 2+2×K+2，max_iter=10 时最大 24 层   │
│  ※ 3层方案：有效深度 = 2+3×K+2，max_iter=10 时最大 34 层（≈2× baseline 16 层）│
├────────────────────────────────────────────────────────────┤
│  Fixed Back：2 层 Transformer → LM Head                     │ ← 不循环（输出对齐/决策映射）
└────────────────────────────────────────────────────────────┘
```

**Column 两种方案对比**：

| | **2层方案** | **3层方案** |
|---|-----------|-----------|
| Column 独立权重 | 2 层 | 3 层 |
| 总独立权重 | 6 层 (2+2+2) | 7 层 (2+3+2) |
| 循环 2 次有效深度 | 8 层 | 10 层 |
| 循环 3 次有效深度 | 10 层 | 13 层 |
| max_iter=10 有效深度 | 24 层 | 34 层（≈2× baseline 16 层） |
| 参数效率 | 更高（用更少参数换深度） | 中等 |
| 表达能力 | 中等 | 更强（每轮更多层处理） |

**为什么固定前后层？**（层剪枝研究证据）
- **浅层是最关键的**：移除导致断崖下降，负责知识存储和信息检索（Gromov et al. 2025 ICLR; Sun et al. 2025）
- **最后层绝不能移除**：与 LM Head 的对齐关系特殊（所有剪枝论文一致）
- **中间层负责推理**：是循环处理应该发生的位置

**大脑对应**：
- Fixed Front ≈ 感觉皮层初级区（V1, A1）：固定特征提取
- Column ≈ 联合皮层：灵活的信息整合与推理
- Fixed Back ≈ 运动皮层/输出层：最终决策映射

Column 内部实现预测编码循环机制（见 3.2）。**一个 Column，参数共享**——只有 2-3 层独立权重，通过循环复用达到更深的有效深度。

### 3.1.1 预训练层映射（迁移学习）

基座模型：Llama 3.2 1B（16 层，d_model=2048）

| CCT 组件 | 预训练层来源 | 说明 |
|----------|-------------|------|
| Fixed Front (2层) | **Layer 0, 1** | 低层 token 特征提取 |
| Column (3层, 循环复用) | **Layer 2, 3, 4** | 皮层柱循环处理 |
| Fixed Back (2层) | **Layer 14, 15** | 高层语义 → LM Head |

中间层 5-13 被丢弃。选择依据：
- 浅层 (0-4) 负责知识存储和特征提取，最关键
- 最后两层 (14-15) 与 LM Head 对齐关系特殊，不可移除
- 中间层 (5-13) 功能冗余度高（Gromov et al. 2025），可安全丢弃

### 3.2 预测编码反馈流程：一步一步

这是 CCT 的核心。以唯一的 Column（2-3 层共享权重 Transformer）为例，展示一次完整的预测编码循环：

```
           柱输入 x [batch, seq, 2048]
             │
    ┌────────┼──────────────────────────────────────────┐
    │        │                                          │
    │   ①  R(k) · x  旋转循环嵌入（before W_Q/W_K）     │
    │   然后经过内容层（2-3 层 Transformer，共享权重）    │
    │   ※ 第2轮起：attention 受 precision 调制           │
    │   ※ RoPE 照常在 W_Q/W_K 之后施加（编码 token 位置）│
    │        │                                          │
    │        ↓                                          │
    │     内容层输出 h [batch, seq, 2048]                 │
    │        │                                          │
    │   ② 预测与锚点（全维度，不压缩）                   │
    │   pred   = Predictor(h)  → [batch,seq,2048]        │
    │   anchor = Anchor_MLP(x) → [batch,seq,2048]        │
    │        │                                          │
    │   ③ 误差评分（对角线点积）                         │
    │   score_t = dot(pred_t, anchor_t) / √d             │
    │     → 每个 token 一个标量                           │
    │        │                                          │
    │   ④ 停止判断（ACT 软停止 + 温度退火二值化）           │
    │     HaltHead: halt_logit = Linear(mean_pool(h))     │
    │     p_halt = sigmoid(halt_logit / τ_halt)            │
    │     训练: τ_halt 从 1.0 → 0.01 线性退火              │
    │       → 初期 p_halt 平滑 (梯度友好)                  │
    │       → 后期 p_halt 二值化 (0.01/0.99, 匹配推理)     │
    │     推理: p_halt > 0.5 → 硬停止                      │
    │     ACT 加权和: output += remainder · p_halt · h      │
    │        │                                          │
    │   ⑤ L6 Precision Weighting（纯调制）               │
    │   precision_t = 1 - σ(score_t)                     │
    │     → error大的token获得高precision                 │
    │                                                    │
    │   ⑥ 下一轮循环：回到 ①                            │
    │     R(k+1) 旋转注入新的循环相位                     │
    │     内容层每一层 self-attention 中加入：             │
    │     attn_logits[t, :] += λ · precision_t            │
    │     → error大的token"更努力地注意"上下文             │
    │     ※ 输入 x 不变（无加法误差注入）                 │
    └────────────────────────────────────────────────────┘

双重旋转应用点：
  ┌─────────────────────────────────────────────────┐
  │  x → [旋转循环嵌入 R(k)] → x' → W_Q/W_K →      │
  │      → [RoPE R(pos)] → Q'/K' → Attention        │
  │                                                  │
  │  ● R(k): 编码循环轮次 k（before 投影）           │
  │  ● RoPE: 编码 token 位置 pos（after 投影）       │
  └─────────────────────────────────────────────────┘
```

**逐步解读**：

**① 内容处理（对应大脑 L2/3）**
- 柱输入 x 先经过**旋转循环嵌入** R(k)·x（范数不变的旋转操作），再经过 Column 内 2-3 个 Transformer 层（共享权重，每轮复用）
- 旋转循环嵌入：对隐藏状态每个维度对 (2i, 2i+1) 施加旋转，角度 = k × φ × base_freq^(-2i/d_model)，其中 φ=1.618（黄金比例）
- **旋转在 W_Q/W_K 投影之前施加**（编码循环轮次），RoPE 在投影之后施加（编码 token 位置）
- 这些层做 attention + FFN，生成内容表示 h
- **第2轮起**：每层的 self-attention 受 precision 偏置调制，error 大的 token "更认真地看"上下文

**② 预测与锚点（对应大脑 L2/3→L5 + 丘脑→L5）**
- 将内容层输出 h 和原始柱输入 x 映射到同一空间
- **Predictor**：单层线性投射 `W_pred · h`（L2/3 的处理结果）
- **Anchor**：2 层 MLP `Linear(ReLU(Linear(x)))` + stop-gradient（丘脑→L5 的直达信号）
- **不压缩**：在全 d_model=2048 维度上比较，保留完整信息
- Anchor 用 stop-gradient：它是固定参考，只通过 LM loss 间接优化

**③ 误差评分（对应大脑 L5 的比较运算）**
- score_t = dot(pred_t, anchor_t) / √d_model — 每个 token 的预测-锚点相似度
- score 高 → pred 和 anchor 方向一致 → 内容层"解释"了输入 → 预测成功
- score 低 → 方向不一致 → 内容层的表示还不够好

**④ 停止判断（ACT 软停止 + 温度退火，对应大脑 SC 阈值机制）**
- HaltHead：`halt_logit = Linear(mean_pool(h))`，`p_halt = sigmoid(halt_logit / τ_halt)`
- 训练时：τ_halt 从 1.0 线性退火到 0.01 → 初期 p_halt 平滑（梯度友好），后期 p_halt 二值化（匹配推理）
- 推理时：p_halt > 0.5 → 硬停止（break）
- ACT 加权和：`output += remainder · p_halt · h`
- 对应 Stine et al. (2023) 发现的 SC 阈值爆发终止机制

**⑤ Precision Weighting——核心创新（对应大脑 L6 精度加权）**
- precision_t = 1 - sigmoid(score_t / τ_p)，τ_p = 0.5（固定超参数）
- τ_p 越小 → sigmoid 越陡 → easy/hard token 区分度越大
- score 低（预测差）→ precision 高 → 下一轮该 token 的 attention 增强
- **这正是 Friston 精度加权的直接实现**：L6 不处理内容，只控制"每个位置的信号处理强度"
- 数学实现：在下一轮每层 self-attention 的 logits 上加 query 侧偏置
  `attn_scores[t, :] += λ · precision_t`
- 效果：prediction error 大的 token "更努力地注意"上下文，利用更多信息改善自身表示
- **误差信息仅通过调制传递**——不对输入做加法注入。这符合大脑证据：L6 的反馈是调制性的（modulatory），而非加法性的（Friston 2010; Bastos 2012）

**⑥ 循环（对应大脑柱内 ~15-25ms 的 oscillation cycle）**
- 输入 x 不变，但 R(k+1) 旋转改变相位 + precision 调制 attention → 生成更好的表示 → score 更高
- 通常 2 次就够了（大脑也是 ~2-3 个 gamma 周期）
- 旋转循环嵌入使模型区分不同轮次（Universal Transformer 证明这是共享权重架构的必要条件，Dehghani et al. 2019）
- 旋转操作保持范数不变（||R·x|| = ||x||），不会影响训练稳定性

### 3.2.1 残差设计（大脑启发）

标准 Transformer 每层有 `h = layer(x) + x` 残差。**大脑皮层柱内没有标准的 additive skip connection**（König 2026, Suzuki 2023 NRN）。CCT 采用大脑启发的三层残差策略：

| 级别 | 策略 | 理由 |
|------|------|------|
| **Transformer 层内** | 保留 `h = layer(x) + x` | 属于层的内部实现；对应大脑层内局部循环连接 |
| **Cycle→Cycle** | ❌ **无残差** | 只有 precision 调制传递到下一轮；输入 x 保持不变；大脑柱内无 cycle-level skip |
| **Column 整体** | ❌ **无门控残差**：`out = h_final` | 大脑皮层柱无输入旁路融合；L6 纯调制不融合输入 |

Column 被视为一个整体计算单元——内部 Transformer 层是它的"内部实现"，无门控残差意味着 Column 输出纯粹依赖迭代处理结果。

### 3.2.2 双频编码架构：RoPE ≈ Gamma，旋转循环嵌入 ≈ Theta

CCT 的位置编码和循环编码设计精确复现了大脑的 **Theta-Gamma 双频架构**（Lisman & Buzsáki 2013）。

**大脑的双频机制**：
- **Gamma 振荡**（~40Hz，~25ms/cycle）：编码空间/项目信息。不同神经元在不同 gamma 相位放电 → **成对、相对**的编码方式
- **Theta 振荡**（~7Hz，~140ms/cycle）：编码时间/迭代信息。Theta 相位调制所有神经元的**全局兴奋性** → **全局、绝对**的编码方式（Singer 2019）
- 内嗅皮层用**速度控制振荡器（VCO）**编码位置：位置 = 振荡器相位，地图布局 = 连接权重（Orchard et al. 2013）。数学上，VCO 的相位编码与 RoPE 的旋转编码**近乎同构**

**RoPE 与 VCO 的数学对应**：

| | 大脑 VCO | RoPE |
|---|---------|------|
| 编码对象 | 空间位置 | token 序列位置 |
| 编码方法 | 相位 = ∫ω dt ≈ ω·t | 旋转角 = θ_i · pos |
| 多频率 | 不同 VCO 频率不同 | 不同维度对（dim pair）频率不同 |
| 信息提取 | 相位差 → 干涉 → 网格细胞激活 | 相对旋转 → Q·K 点积调制 |

**为什么旋转循环嵌入必须在 W_Q/W_K 之前施加**：

标准 RoPE 在 W_Q/W_K **之后**施加旋转，编码 token 位置：
```
Q = R(pos) · W_Q · x,   K = R(pos) · W_K · x
→ Q·K^T 包含 R(pos_i - pos_j)，位置信息保留
```

如果循环嵌入也像 RoPE 一样在投影**之后**施加，循环信息会被消除：
```
Q = R(k) · W_Q · x,   K = R(k) · W_K · x
Q·K^T = (R(k)·W_Q·x)^T · (R(k)·W_K·x)
      = x^T · W_Q^T · R(k)^T · R(k) · W_K · x
      = x^T · W_Q^T · W_K · x    ← k 消失！（因为 R^T·R = I）
```
原因：同一次 forward pass 中所有 token 共享同一个 k，R(k)^T·R(k) = I。

旋转循环嵌入在投影**之前**施加时，循环信息保留：
```
Q = W_Q · R(k) · x,   K = W_K · R(k) · x
Q·K^T = x^T · R(k)^T · W_Q^T · W_K · R(k) · x
      = x^T · R(k)^T · M · R(k) · x    （其中 M = W_Q^T · W_K）
```
由于 R(k)^T · M · R(k) ≠ M（除非 M 恰好与 R 对易），循环信息被保留在注意力计算中。

**CCT 的双频设计**：

| 频段 | CCT 对应 | 应用位置 | 编码对象 |
|------|---------|---------|---------|
| Gamma (~40Hz) | **RoPE** 旋转编码 | W_Q/W_K **之后** | token 间**相对位置** |
| Theta (~7Hz) | **旋转循环嵌入** | W_Q/W_K **之前** | **循环轮次**（全局相位） |

两者都是旋转操作，但应用位置不同、编码对象不同——正如大脑用不同频段编码不同信息。

**黄金比例 φ 的理论基础**：

旋转循环嵌入使用 φ = 1.618（黄金比例）作为频率偏移因子：cycle_freq = RoPE_freq × φ。选择 φ 的神经科学依据：

| 文献 | 发现 | 设计含义 |
|------|------|---------|
| **Pletzer et al. 2010** (Brain Research) | φ 是"**最不理性数**"（最难被有理数逼近）→ 两个频率之比为 φ 时，**虚假相位同步最少** | 位置通道和循环通道最大独立，互不干扰 |
| **Roopun et al. 2008** | 皮层 gamma/beta 频率比 ≈ φ → 自然演化出的**最大干扰免疫**方案 | CCT 的双频比例遵循生物学最优 |
| **Kramer et al. 2022** (Neuron) | φ 同时支持**分离**（anti-commensurability：频率永不整除）和**整合**（Fibonacci additivity：F(n)+F(n+1)=F(n+2)） | 循环编码既独立于位置编码，又能在需要时协同 |

公式：`angle_i(k) = k × φ × base_freq^(-2i/d_model)`，其中 k 为循环轮次，i 为维度索引。对隐藏状态 h 的每个维度对 (2i, 2i+1) 做旋转。**0 可学习参数**（固定公式），范数不变（||R·x|| = ||x||）。

### 3.3 核心问题解答：与"简单重复"的区别

为什么不直接把这几层跑两遍（Simple Repeat）？区别在于：

| 机制 | Simple Repeat | CCT |
|------|--------------|-----|
| 第二遍的输入 | 和第一遍完全一样 | 旋转循环嵌入注入新的相位 |
| 有方向性吗 | ❌ 盲目重复 | ✅ precision 告诉"哪些 token 需要更多处理" |
| 知道该不该继续吗 | ❌ 固定次数 | ✅ score 高就停 |
| 知道第几轮吗 | ❌ 无区分 | ✅ 旋转循环嵌入（theta 相位） |
| Attention 调制 | ❌ 没有 | ✅ error 大的 token 更努力看上下文 |
| Column 残差 | 标准加法 | ❌ 无残差（纯调制） |
| 大脑对应 | 无 | 皮层柱内预测编码环路 + 精度加权 |

### 3.4 新增模块清单

CCT 在预训练 Transformer 之上只新增以下轻量模块（每个 Column 独立）：

| 模块 | 维度 | 参数量 | 大脑对应 |
|------|------|--------|---------|
| Anchor MLP | 2048→2048→2048 (2层MLP) | ~8.39M | 丘脑→L5 直达通路（非线性） |
| Predictor | 2048→2048 (线性) | ~4.20M | L2/3→L5 投射 |
| L6 Precision | λ 可学习标量 (τ_p=0.5 固定) | 1 | L6 精度加权控制 |
| HaltHead | Linear(2048→1) + bias | ~2K | 停止决策（ACT 软停止） |
| 迭代间 RMSNorm | 2048 | ~2K | 防止迭代间 hidden state 发散 |
| 旋转循环嵌入 | 固定旋转矩阵（φ 黄金比例频率） | 0 (固定) | theta 振荡相位编码 |
| **单柱合计** | | **~12.59M** | 轻量（占基座 ~1.8%） |

注：只有一个 Column，参数共享。旋转循环嵌入无可学习参数，纯调制设计轻量，也更符合大脑 L6 "只调制不处理"的生物学原型。HaltHead 实现 ACT 软停止，迭代间 RMSNorm 防止循环累积发散。设计变更依据：Friston (2010) 和 Bastos (2012) 的证据表明皮层反馈是调制性的，而非加法性的。

**参数量总览（d_model=2048, Column=3层方案）**：

| 组件 | 参数量 |
|------|--------|
| Token Embedding (128256×2048, tied) | 262.67M |
| Fixed Front (2层 × 60.82M) | 121.64M |
| Column (3层 × 60.82M, 循环复用) | 182.46M |
| Fixed Back (2层 × 60.82M) | 121.64M |
| CCT 新增模块 | 12.59M |
| **总计** | **~701M (0.70B)** |
| CCT 开销占比 | 1.83% |

---

## 四、训练策略

### 4.1 单阶段训练

- 全参数解冻，从头开始联合训练
- 循环次数由 HaltHead 自适应决定（ACT 软停止）：
  - 训练时跑所有 max_iter 轮，每轮计算 p_halt，输出 = 加权和
  - τ_halt 从 1.0 退火到 0.01 → p_halt 从平滑变为二值化
  - 推理时：p_halt > 0.5 → break
- **迭代间 RMSNorm**：每次 Column 循环后施加，防止 hidden state 累积发散
- 分层学习率：基座 LR=2e-5，新模块 LR=1e-4
- 梯度裁剪、AMP、gradient checkpointing

### 4.2 损失函数

```
L_total = L_LM + λ_pred · mean(L_pred_k) + λ_flops · ponder_cost
```

- **L_LM** = 标准 next-token prediction (cross-entropy)
- **L_pred** = 1 - cos_sim(Predictor(h.detach()), AnchorMLP(x_column.detach()))
  - h.detach()：阻止 L_pred 梯度回传到基座模型
  - x_column.detach()：同上
  - Predictor 和 AnchorMLP 都从 L_pred 获得梯度学习
  - **与 PPG 的区别**：PPG info_proj 共享 → 需双重 detach；CCT Predictor/AnchorMLP 完全独立 → 不需要
- **L_flops** = λ_flops · ponder_cost（线性，标准 ACT）
  - ponder_cost = Σ_k (remainder_k · p_halt_k · k)，归一化 by max_iter
  - L_flops 梯度通过 halt_head(h) → Column 回传，**有意影响基座**
  - 目的：鼓励 Column 产生"果断"的表征，使 halt 更早触发

**超参数推荐**：
- λ_pred = 0.1
- λ_flops = 0.01
- max_iter = 10（≈2× baseline 有效深度）
- τ_halt: 1.0 → 0.01 线性退火
- τ_p = 0.5（固定超参数，precision sigmoid 温度）

---

## 五、实验设计

### 5.1 数据

- 训练数据：与基座模型官方 SFT 配置对齐（确保公平对比）
- 具体使用：RedPajama 子集或 FineWeb-Edu
- 数据量：单阶段训练，约 **5-10B tokens**
- 序列长度：4096

### 5.2 基线模型

| 模型 | 参数量 | 描述 | 对比目的 |
|------|--------|------|---------|
| **Vanilla Baseline** | ~700M | 同参数量标准预训练模型（7层 d=2048） | 主基线 |
| **LoopFormer** (复现) | ~700M | 共享权重循环×2 | 循环 vs CCT 循环 |
| **ITT-style** (复现) | ~700M | 层内循环 + token 选择 | 局部循环 vs 柱级循环 |
| **PPG** (本项目) | ~700M | 预测误差路由（skip/repeat） | 路由 vs 循环 |
| **Simple Repeat** | ~700M | 简单层组重复 2 次（无误差/增益） | 消融基线 |
| **Larger Model** | ~1.2B | 更大模型（Llama 3.2 全16层） | 循环是否能弥补参数差距 |

### 5.3 评估方法

**主要指标**：
1. **困惑度 (Perplexity)**：在 WikiText-103 / C4 验证集上的 PPL
2. **下游任务**：HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande（0-shot/5-shot）
3. **FLOPs 效率**：PPL per FLOP 曲线（不同循环次数下的 PPL-FLOPs 权衡）

**效率指标**：
4. **推理延迟**：每 token 生成时间（ms）
5. **平均循环次数**：不同 token 类型（内容词/功能词/标点）的自适应深度
6. **循环分布**：柱级和 token 级的循环次数直方图

**分析指标**：
7. **误差收敛曲线**：每次循环后 score（pred·anchor 相似度）的上升率
8. **Precision 分布**：不同 token 类型（内容词/功能词/标点）的 precision 值分布
9. **与大脑数据对比**：循环次数分布是否类似皮层处理时间

### 5.4 公平对比策略

由于 CCT 的循环增加了计算量，我们采用**多维度公平对比**：

| 对比维度 | 方法 |
|----------|------|
| **参数匹配** | CCT vs Vanilla（相同参数量，CCT 额外模块很轻量） |
| **训练 FLOPs 匹配** | CCT(5-10B tokens) vs Vanilla(5-10B tokens) |
| **推理 FLOPs 匹配** | CCT(avg 1.5 loops) vs Vanilla(对应深度) |
| **质量匹配** | CCT 达到 Vanilla PPL 需要多少 FLOPs vs Vanilla 训练相同 FLOPs 的 PPL |

---

## 六、消融实验

### 6.1 组件消融

| 实验 | 移除组件 | 预期效果 | 验证假设 |
|------|----------|----------|---------|
| A1 | 移除 Precision Weighting | PPL 上升 | 精度加权是误差传递的唯一通道 |
| A2 | 加回加法误差注入（x'=x+δ） | PPL 接近/下降 | 纯调制 vs 调制+加法对比 |
| A3 | 移除旋转循环嵌入 | PPL 上升 | 循环感知对共享权重必要（Universal Transformer 验证） |
| A4 | 移除 HaltHead（固定循环） | PPL 接近/相同 | 自适应停止的价值 |
| A5 | Anchor 用单层线性 → 2层MLP | PPL 上升 | MLP Anchor 更好 |
| A6 | 用减法 error 替代 dot-product score | PPL 接近 | 点积 vs 减法对比 |
| A8 | 用加法正弦嵌入替代旋转嵌入 | PPL 接近 | 旋转 vs 加法对比 |
| A9 | 旋转嵌入在 W_Q/W_K **之后**而非之前 | PPL 上升（循环信息被消除） | 旋转位置的必要性（见 §3.2.2 数学证明） |
| A10 | 循环频率不使用黄金比例偏移 | PPL 上升 | φ 的去同步化价值（Pletzer 2010） |

### 6.2 循环次数消融

| 实验 | max_iter | 预期 |
|------|----------|------|
| L1 | 1（无循环） | 基线 PPL |
| L2 | 2 | PPL 显著下降 |
| L3 | 3 | PPL 进一步下降，但边际递减 |
| L4 | 4 | PPL 几乎不变（饱和） |
| L_adaptive | 自适应 | PPL ≈ L3，但 FLOPs ≈ L2 |

### 6.3 固定层范围消融

| 实验 | 结构 | 对比 |
|------|------|------|
| F1 | Fixed 2层 + Column 2层×K + Fixed 2层 | 默认 2层方案 |
| F2 | Fixed 2层 + Column 3层×K + Fixed 2层 | 默认 3层方案 |
| F3 | Fixed 3层 + Column 2层×K + Fixed 2层 | 更多固定前端 |
| F4 | Fixed 1层 + Column 3层×K + Fixed 1层 | 最少固定层 |

### 6.4 关键验证实验

**V1：误差-收益相关性验证**
- 在训练早期，测量每个 token 的 prediction error 大小
- 测量该 token 额外循环一次带来的 ΔL_LM（LM loss 变化）
- 计算两者的相关系数
- 如果相关性很低（<0.3），则误差驱动路由的假设可能有问题
- 此实验应在训练早期就做，指导后续设计

**V2：与简单重复的对比**
- Simple Repeat（无误差/无增益/无控制器）vs CCT
- 如果 Simple Repeat 就已经很好，则 CCT 额外机制价值有限
- 这个对比是论文的核心论点支撑

---

## 七、风险分析

| 风险 | 可能性 | 影响 | 缓解策略 |
|------|--------|------|---------|
| 误差信号与 LM 收益低相关 | 中 | 高 | V1 验证实验；备选方案：用 logit 熵作为路由信号 |
| 循环导致训练不稳定 | 中 | 高 | sigmoid 有界精度；纯调制无加法注入；迭代间 RMSNorm 防发散 |
| 自适应停止坍缩 | 中 | 中 | τ_halt 温度退火确保二值化收敛；L_flops 正则化 |
| 循环增加的 FLOPs 不值得 | 中 | 高 | 与 Simple Repeat 对比；FLOPs 匹配评估 |
| KV-cache 不兼容 | 低 | 高 | 仅缓存最终循环；测试推理正确性 |
| Precision Weighting 实际效果微弱 | 中 | 低 | 消融 A1 验证；如无效则简化设计 |

---

## 八、预期贡献

1. **架构创新**：首个在 LLM 中实现完整皮层柱内环路（误差驱动循环 + Precision Weighting + ACT 软停止 + 温度退火二值化）
2. **神经科学-AI 桥梁**：为 König 2026 的理论映射提供首个可训练的实现，首次将 Friston 精度加权映射到 Transformer attention
3. **实证发现**：量化误差驱动循环 vs 简单重复 vs 自适应深度的收益
4. **分析工具**：提供循环行为分析框架（误差收敛、precision 分布、token 级深度分布）

---

## 九、参考文献

### 核心神经科学
1. Douglas, R. J., Martin, K. A. C., & Whitteridge, D. (1989). A canonical microcircuit for neocortex. Neural Computation.
2. Douglas, R. J., & Martin, K. A. C. (2004). Neuronal circuits of the neocortex. Annual Review of Neuroscience.
3. Costa, R. P., et al. (2025). Self-supervised predictive learning accounts for cortical layer-specificity. Nature Communications.
4. Constantinople, C. M., & Bruno, R. M. (2013). Deep cortical layers are activated directly by the thalamus. Science.
5. Larkum, M. (2013). A cellular mechanism for cortical associations. Trends in Neurosciences.

### L6 增益控制与精度加权
6. Olsen, S. R., Bortone, D. S., Adesnik, H., & Scanziani, M. (2012). Gain control by layer six in cortical circuits of vision. Nature, 483, 47–52.
7. Dimwamwa, E. D., et al. (2024). Dynamic corticothalamic modulation of the somatosensory thalamocortical circuit during wakefulness. Nature Communications.
8. Zolnik, T. A., Eickholt, B. J., Molnár, Z., & Larkum, M. E. (2026). The layer 6b theory of attention. Neuron.
9. Lam, Y.-W., & Sherman, S. M. (2010). Functional organization of the somatosensory cortical layer 6 feedback to the thalamus. Cerebral Cortex, 20, 13–24.
10. Katzner, S., Rose, T., Tchumatchenko, T., & Busse, L. (2025). The role of layer 6 corticothalamic circuits in vision. Annual Review of Vision Science.

### Transformer-皮层映射
11. König, P., & Negrello, M. (2026). The neuroscience of transformers. arXiv:2603.15339.
12. TRC² (2025). Thalamically routed cortical columns. arXiv:2602.22479.

### 预测编码
13. Bastos, A. M., et al. (2012). Canonical microcircuits for predictive coding. Neuron, 76(4), 695–711.
14. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex. Nature Neuroscience.
15. Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society.
16. Rao, R. P. (2022). A sensory-motor theory of the neocortex based on active predictive coding. bioRxiv.

### 循环/深度自适应 Transformer
17. Chen, L., et al. (2025). Inner thinking transformer. ACL 2025.
18. LoopFormer (2026). ICLR 2026.
19. Dehghani, M., et al. (2019). Universal transformers. ICLR 2019.

### 增益调制与门控
20. Perez, E., et al. (2018). FiLM: Visual reasoning with a general conditioning layer. AAAI 2018.（Feature-wise Linear Modulation）
21. GAIN (2025). Gain modulation for domain adaptation of LLMs.（乘性增益调制防止灾难性遗忘）
22. Feldman, H., & Friston, K. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience.（精度加权理论）

### 循环终止与决策机制
26. Stine, G. M., et al. (2023). A neural mechanism for terminating decisions. Neuron.（SC 阈值爆发终止机制）
27. Stine, G. M., et al. (2022). LIP activity during accumulation evidence in a decision task. Nature Neuroscience.

### 循环计数与相位编码
28. Lisman, J., & Buzsáki, G. (2013). A neural coding scheme formed by the combined function of gamma and theta oscillations. Neuron.（Theta-Gamma Code）
29. Pletzer, B., et al. (2010). When frequencies never synchronize: the golden mean and the resting EEG. Brain Research.（黄金比例去同步化）
30. Kramer, M. A. (2022). Golden rhythms as a theoretical framework for neural oscillation hierarchy. Neuron.（φ 频率框架）
31. Klimesch, W. (2013). An algorithm for the EEG frequency architecture of consciousness and brain body coupling. Frontiers in Human Neuroscience.（EEG 频段架构算法）
32a. Roopun, A. K., et al. (2008). Period concatenation underlies interactions between gamma and beta rhythms in neocortex. Frontiers in Cellular Neuroscience.（皮层 gamma/beta 频率比 ≈ φ）

### 振荡编码与 RoPE 类比
40. Orchard, J., Yang, H., & Ji, X. (2013). Does the entorhinal cortex use the Fourier transform? Frontiers in Computational Neuroscience.（VCO 相位编码 ≈ 旋转位置编码）
41. Singer, W. (2019). Neuronal oscillations: unavoidable and useful? European Journal of Neuroscience.（振荡中神经元经历周期性兴奋性变化）
42. Nature 2025. Flexible perceptual encoding by discrete gamma events. Nature.（gamma 事件是离散网络事件）
43. PMC (2024). Theta phase precession supports memory formation and retrieval of naturalistic experience in humans. Nature Human Behaviour.（相位进动作为通用编码原则）

### 反馈调制
32. Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience.（反馈是调制性的）
33. Nature Communications (2022). Two cortico-cortical feedback pathways in V1: temporal sharpening, not simple addition.
34. PLOS Computational Biology (2025). SST neurons carry top-down predictions as gain modulation.

### 共享权重循环
35. Dehghani, M., et al. (2019). Universal Transformers. ICLR 2019.（step embedding 是共享权重循环的必要条件）
36. HRM-LM (2026). Hierarchical recurrent memory language model.（flat iteration 停滞，需要层次/步骤感知）

### LLM 层分析与剪枝
37. Gromov, A., et al. (2025). The unreasonable ineffectiveness of the deeper layers. ICLR 2025.（浅层关键，深层可剪枝）
38. Sun, Y., et al. (2025). Curse of Depth: deeper layers become ineffective due to variance explosion.（深层失效分析）
39. arXiv 2510.02091 (2026). Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning.（层级功能分化）

---

## 十、时间线与硬件需求

### 硬件
- 调试：RTX 5070 8GB（本地）
- 训练 + 消融：NVIDIA RTX PRO 6000 96GB（云端）

### 最小可行实验（MVP）

**先做最简版本验证核心假设**：
1. 单 Column = 2-3 层 Transformer，参数共享，最多循环 10 次
2. Fixed Front 2 层，Fixed Back 2 层
3. Anchor = stop-grad 2层MLP
4. Predictor = 线性投射（2048→2048）
5. score = dot(pred, anchor) / √d → precision = 1 - sigmoid(score / τ_p)，τ_p = 0.5
6. Precision Weighting = per-token attention 偏置（query 侧，纯调制）
7. 旋转循环嵌入（0 可学习参数，φ 黄金比例频率，W_Q/W_K 之前施加）
8. ACT 软停止 + 温度退火 → 推理时硬停止
9. 迭代间 RMSNorm 防 hidden state 发散
10. 仅缓存最终 KV
11. **无加法误差注入**（误差信息仅通过 precision 调制传递）
12. **立即与 Simple Repeat 基线对比**

如果 MVP 显示皮层环路优于简单重复 → 继续完整版
如果 MVP 与简单重复无显著差异 → 重新评估设计

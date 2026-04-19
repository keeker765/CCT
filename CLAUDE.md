# CCT Project Notes

## 实验记录

### Run 2: 因果掩码修复后 (2025-04-19)
- 数据量: 40k samples (OpenHermes 2.5)
- 训练步数: 200 steps (debug mode, total_steps=1188)
- Batch: 32, lr_base=2e-5, lr_new=1e-4

```
[Step  20/200] loss=5.7023 | lm=5.6337 pred=0.6861 ent=1.3719 ponder=0.0000 | eff_iters=2.10 tau=0.984
[Step  40/200] loss=3.6936 | lm=3.6576 pred=0.3599 ent=1.4117 ponder=0.0000 | eff_iters=2.20 tau=0.968
[Step  60/200] loss=3.2108 | lm=3.1929 pred=0.1795 ent=1.4611 ponder=0.0000 | eff_iters=2.30 tau=0.951
[Step  80/200] loss=2.9259 | lm=2.9158 pred=0.1016 ent=1.4713 ponder=0.0000 | eff_iters=2.32 tau=0.934
[Step 100/200] loss=2.7627 | lm=2.7555 pred=0.0722 ent=1.4744 ponder=0.0000 | eff_iters=2.34 tau=0.917
  [Eval] loss=2.7350 PPL=15.41 (New best!)
[Step 120/200] loss=2.6442 | lm=2.6381 pred=0.0610 ent=1.4926 ponder=0.0000 | eff_iters=2.37 tau=0.901
[Step 140/200] loss=2.5852 | lm=2.5801 pred=0.0509 ent=1.4897 ponder=0.0000 | eff_iters=2.39 tau=0.884
[Step 160/200] loss=2.4845 | lm=2.4801 pred=0.0433 ent=1.4803 ponder=0.0000 | eff_iters=2.35 tau=0.867
[Step 180/200] loss=2.3791 | lm=2.3753 pred=0.0377 ent=1.4748 ponder=0.0000 | eff_iters=2.37 tau=0.851
[Step 200/200] loss=2.4317 | lm=2.4283 pred=0.0343 ent=1.4782 ponder=0.0000 | eff_iters=2.36 tau=0.834
```

### Run 3: 全量训练 (tau 退火完整) (2025-04-19)
- 数据量: 40k samples (OpenHermes 2.5), Train: 38000, Eval: 2000
- 训练步数: 1188 steps (完整 1 epoch, max_steps=total_steps)
- Batch: 32, grad_accum: 1, lr_base=2e-5, lr_new=1e-4, CosineAnnealingLR
- 关键变化: tau 正确从 0.984 退火至 0.017 (Run 2 因 bug 只降到 0.834)
- 总训练时间: 29.7 min (Colab T4)

```
[Step  100/1188] loss=2.7054 | lm=2.6990 pred=0.0634 ent=1.4766 | eff_iters=2.29 tau=0.917 | lr_new=9.83e-05
  [Eval] loss=2.6827 PPL=14.62 (New best!)
[Step  200/1188] loss=2.3208 | lm=2.3176 pred=0.0322 ent=1.4735 | eff_iters=2.29 tau=0.834 | lr_new=9.32e-05
  [Eval] loss=2.3468 PPL=10.45 (New best!)
[Step  300/1188] loss=2.2567 | lm=2.2544 pred=0.0231 ent=1.4842 | eff_iters=2.34 tau=0.751 | lr_new=8.51e-05
  [Eval] loss=2.1838 PPL=8.88 (New best!)
[Step  400/1188] loss=2.0878 | lm=2.0857 pred=0.0215 ent=1.4497 | eff_iters=2.31 tau=0.667 | lr_new=7.45e-05
  [Eval] loss=2.0777 PPL=7.99 (New best!)
[Step  500/1188] loss=1.9229 | lm=1.9210 pred=0.0195 ent=1.4000 | eff_iters=2.28 tau=0.584 | lr_new=6.23e-05
  [Eval] loss=2.0030 PPL=7.41 (New best!)
[Step  600/1188] loss=2.0468 | lm=2.0449 pred=0.0194 ent=1.3698 | eff_iters=2.32 tau=0.501 | lr_new=4.92e-05
  [Eval] loss=1.9520 PPL=7.04 (New best!)
[Step  700/1188] loss=1.9559 | lm=1.9541 pred=0.0181 ent=1.3241 | eff_iters=2.38 tau=0.418 | lr_new=3.62e-05
  [Eval] loss=1.9141 PPL=6.78 (New best!)
[Step  800/1188] loss=1.9229 | lm=1.9209 pred=0.0191 ent=1.2268 | eff_iters=2.37 tau=0.334 | lr_new=2.41e-05
  [Eval] loss=1.8879 PPL=6.61 (New best!)
[Step  900/1188] loss=1.8531 | lm=1.8513 pred=0.0178 ent=1.0579 | eff_iters=2.31 tau=0.251 | lr_new=1.38e-05
  [Eval] loss=1.8719 PPL=6.50 (New best!)
[Step 1000/1188] loss=1.8000 | lm=1.7982 pred=0.0182 ent=0.8131 | eff_iters=2.28 tau=0.167 | lr_new=6.05e-06
  [Eval] loss=1.8659 PPL=6.46 (New best!)
[Step 1100/1188] loss=1.8764 | lm=1.8746 pred=0.0177 ent=0.4724 | eff_iters=2.19 tau=0.084 | lr_new=1.35e-06
  [Eval] loss=1.8722 PPL=6.50
[Step 1180/1188] loss=1.9618 | lm=1.9600 pred=0.0178 ent=0.1422 | eff_iters=2.02 tau=0.017 | lr_new=1.12e-08
```

**评估 (Train vs Infer)**:
| Mode | LM Loss | PPL | Avg Iters | Gap |
|------|---------|-----|-----------|-----|
| Train | 1.9023 | 6.70 | 5.0 | - |
| Infer | 1.9023 | 6.70 | 5.0 | +0.0000 |

**推理测试**: 5 个 eval 集样本，贪心解码 128-256 tokens。模型输出全部退化（重复循环），无有效生成。

**关键发现**:
1. loss 持续下降 (5.77→1.87), PPL 从 14.62 降至 6.46，学习信号正常
2. pred loss 收敛到 ~0.018，说明 Predictor 学到了稳定的误差信号
3. eff_iters 始终 ~2.3，tau 退火未实际影响迭代次数 → Halt 机制未学会提前停止
4. Train/Infer gap = 0.0, 两者均使用 max_iter=5 → 硬停止与软加权结果一致
5. **生成质量差**: 贪心解码产生重复退化文本，loss 低但不代表生成能力好
6. entropy 从 1.5 单调降至 0.14，halt 分布过度集中

## Kaggle 备用账号

- **用户名**: vhmbhjfhgdgfx
- **API Key**: eb633cf87424f56a940e7b72c1dedefc

## 已修复的 Bug

1. **因果掩码缺失** (wrapped_model.py): transformers 5.x 的 create_causal_mask 返回 None，导致 Column 层无 causal mask → 添加手动上三角掩码回退
2. **Score 尺度溢出** (predictor.py): compute_score 用原始点积(~57) → 改为 cosine_similarity([-1,1])，使 L6 Precision 正常工作
3. **tau schedule 用 total_steps 而非 max_steps**: 训练循环 compute_halt_tau 传入 total_steps=1188 而非 max_steps=200，导致 tau 只降到 0.834 而非 0.01（待修复）

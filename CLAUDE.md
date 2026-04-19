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

## 已修复的 Bug

1. **因果掩码缺失** (wrapped_model.py): transformers 5.x 的 create_causal_mask 返回 None，导致 Column 层无 causal mask → 添加手动上三角掩码回退
2. **Score 尺度溢出** (predictor.py): compute_score 用原始点积(~57) → 改为 cosine_similarity([-1,1])，使 L6 Precision 正常工作
3. **tau schedule 用 total_steps 而非 max_steps**: 训练循环 compute_halt_tau 传入 total_steps=1188 而非 max_steps=200，导致 tau 只降到 0.834 而非 0.01（待修复）

# P4：恢复效果表征、技能选择与经验复用

状态：**CPU 初始实现完成；真实技能效果标签待采集**（2026-09-20）。

## 计划

冻结恢复技能、基础策略、触觉前端和交还规则；用实际重复分支结果生成带版本、成本、违规和有效性掩码的效果标签；训练多头效果预测器和带距离约束的表示，并按版本兼容性检索旧经验。

## 已实现

- `learning/effect_model.py` 实现成功／成本／违规多头、缺失标签 mask、Huber metric loss 和归一化效果距离。
- `EffectHistoryEncoder` 提供独立于控制 GRU 的 8 帧历史分支和 64D effect embedding；`EffectTrainer` 可从历史窗口端到端训练该分支。metric loss 经过可训练 projection，开启 `metric_weight` 时不会变成无梯度的常数项。
- `learning/effect_training.py` 提供带缺失分支 mask 的多头训练器、可选效果距离监督和带 schema 的 checkpoint；未试验技能不会被当作负样本。
- `collate_effect_labels` 将稀疏技能分支打包成带有效性 mask 的训练 batch，不把未试验分支伪造为失败。
- `learning/skill_selector.py` 实现版本化效果记忆、最近邻检索和成功率减成本／违规惩罚的选择。
- `EffectMemory.save/load` 以带 schema 的 JSON 持久化版本化效果记录，恢复后仍执行 embedding shape 和版本过滤。
- `scripts/build_effect_labels.py` 从冻结技能的 acceptance/test HDF5 生成版本化 JSONL；它要求显式 `violation` 字段，遇到缺失字段或不完整试验会跳过并给出原因，不把 object_lost 猜成违规。
- `configs/effect_selection.yaml` 固定网络、检索和损失超参数边界。
- `evaluation/paired_ablation.py` 可在技能版本冻结后生成同父场景配对效果分母；版本不兼容或基础设施异常会被拒绝。

## 证据与边界

接口 smoke 和测试验证了 mask、版本隔离、预测形状和选择器；当前没有真实技能效果矩阵，所以不能声称效果表征优于普通成功率预测。

效果标签必须来自冻结版本和同一父场景的实际后续结果。`EffectHistoryEncoder` 的 64D 分支与控制 GRU 分离，metric loss 经过可训练 projection；缺失技能分支保持 mask，不能填成失败。没有有效 paired labels 时训练器会拒绝 batch。

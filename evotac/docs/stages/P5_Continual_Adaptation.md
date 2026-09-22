# P5：持续适应与有限技能扩展

状态：**接纳流程代码完成；真实新条件候选尚未运行**（2026-09-20）。

## 计划

按新失败先复用、再补标签、必要时微调或增加一个候选技能；候选必须在独立接纳场景上同时满足新条件收益、旧条件保持和违规上界，证据不足时拒绝。

## 已实现

- `learning/continual_adaptation.py` 提供 50/25/25 当前／检索／均匀旧经验混合。
- `ContinualAdaptationController` 串联样本混合、配对 gate、技能容量限制、版本切换和 JSON 状态持久化；候选训练和评估结果必须由真实分支调用者提供。
- `AcceptanceGate` 使用配对差值的 95% 置信区间，默认新条件下界大于 0、旧条件下界不低于 −5 个百分点、违规上界不超过 2 个百分点。
- `configs/continual_adaptation.yaml` 固定候选数量、技能容量和接纳阈值。
- `evaluation/paired_ablation.py` 和 `report_paired_ablation.py` 可把新／旧条件的实际分支结果送入 gate，缺失配对会返回证据不足而不是自动通过。
- `scripts/accept_adaptation.py` 提供真实 JSONL 接纳入口：只接受 acceptance split，严格配对 parent/seed，并把 controller 状态和 gate 决策持久化。

## 证据与边界

CPU 测试覆盖接纳、样本不足和比例校验；没有真实候选仿真结果前，不把 gate smoke 当作持续适应效果。

控制器现在要求 baseline 版本等于当前激活版本、candidate 版本与提案一致；非零比例的数据组缺失会直接拒绝，单轮最多增加一个技能且每个条件最多两次候选评估。状态文件保存 gate 阈值，便于恢复后复核接纳规则。

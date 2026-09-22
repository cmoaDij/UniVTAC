# P6：最小充分消融实验

状态：**实验协议已冻结；真实 Isaac 消融未开始**（2026-09-20）。

## 计划

使用独立父场景、三种训练种子和配对分支，分别比较触觉／恢复、效果距离／检索和持续适应。统一物理预算、策略版本、传感器权限和失败分母，报告总体与公共失败集合成功率、违规率、适应曲线、技能数和墙钟成本。

## 已实现

- 六阶段总计划已写入 `docs/EvoTac_Phase_Implementation_Plan.md`，阶段 2 门槛写入 `docs/Phase2_Plan.md`。
- 真实运行入口、HDF5 记录、GPU 快照和性能分项计时已准备好；`profile_physics.py` 可先执行有界控制检查。
- `evaluation/paired_ablation.py` 固定父场景／seed／预算配对并计算置信区间；`report_paired_ablation.py` 拒绝重复键并明确标出从输入推断的分母，完整 P6 分母仍必须交给 `audit_phase6_report.py` 和冻结计划检查。
- 配对 runner 在真实回调提供字段时汇总墙钟、技能数量、效果 Brier、到 oracle 距离、旧条件保持和适应曲线 AUC；缺失字段不会被伪造为零。
- `configs/phase6_ablation_plan.json` 和 `validate_phase6_plan.py` 固定三组方法、三种训练 seed、每个 condition 每方法 100 个父场景、冻结测试规则与接纳阈值。
- `scripts/run_fast_ablation.py` 可用 surrogate 执行相同父场景／condition／seed／预算审计；最近触觉恢复组完成 4,800 个 episode，效果组织组 6,000 个 episode，持续适应组 4,800 个 episode，用于检查分母和报告管线。
- `scripts/audit_phase6_report.py` 会独立检查方法矩阵、condition、seed、预算、父场景独立性和完整分母；三份 surrogate JSON 报告均通过 audit。
- 消融结果必须只来自 P2–P5 冻结版本，失配重放和基础设施异常单独统计。

## 当前结论边界

P6 的三个研究问题在真实 Isaac/TacEx 上目前均为“证据不足”。surrogate smoke 只验证每个 condition 100 个父场景、3 个 seed 和冻结方法矩阵的配对循环及 JSONL 审计，不能替代触觉、效果组织或持续适应的物理配对消融。已有 FTP-1 6/10 开发集结果和 handoff 机制验证也不能替代这些消融。

严格审计还拒绝重复的 `(condition,parent,method,seed)`、不同方法使用不同父场景集合、非二值结果和推断出的分母。surrogate 通过稳定的 `sha256(parent_scene, train_seed)` 派生 reset seed，使每个 parent 真正产生不同的轻量逻辑轨迹；其 `evidence_scope` 仍明确标为 protocol smoke，不是 Isaac/TacEx 物理证据。

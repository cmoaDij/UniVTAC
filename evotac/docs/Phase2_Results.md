# 阶段 2 验证结果

完成日期：2026-09-21。本轮失败诊断、多场景基线、真实恢复交接和触觉接口准备已完成。历史正式 v3 cohort 仍为 0/10，**SAC／恢复技能有效性训练未启动**；修复后当前代码的 seed23 开发配对已通过起点和终止审计，但不能替代正式 cohort 或生成多 seed 效果标签。

## 十场景对照结果

预先固定 dev seeds：0、18、23、24、39、47、59、82、86、90，名义条件，每场景独立启动。两组均采用 taxim、livestream、GPU 0 仿真／GPU 9 模型，20 Hz 模拟控制、每控制 6 个 120 Hz 物理步、`force=False`、模型预测夹爪，最多 200 控制。唯一有意改变的执行参数是 chunk 长度。

| 指标 | B0：chunk1，正式选定集 | B1：chunk16 |
|---|---:|---:|
| 有效场景 | 10/10 | 10/10 |
| 成功 | 0/10 | **6/10** |
| 物体丢失 | 10/10 | **3/10** |
| 任务预算截断 | 0/10 | 1/10 |
| 平均执行控制数 | 60.7 | 160.0 |
| 模型推理请求数 | 607 | 105 |
| 全部请求耗时 mean / p50 / p95（秒） | 0.228 / 0.214 / 0.266 | 0.274 / 0.223 / 0.720 |
| 排除每场首次请求：mean / p50 / p95（秒） | 0.219 / 0.214 / 0.258 | 0.224 / 0.219 / 0.262 |

耗时来自实际模型 worker 流程。推理等待期间模拟时间不推进，因此不是实时 20 Hz 性能声明；首次请求在 chunk16 的较少请求中占比更大。平均执行步数混合了成功、丢失与预算截断，不能据此直接比较成功任务的效率。

| seed | B0 控制数（均丢失） | B1 控制数 | B1 终止原因 | 两次初始末端距离 mm |
|---|---:|---:|---|---:|
| 0 | 57 | 183 | success | 1.176 |
| 18 | 75 | 200 | task_budget | 0.095 |
| 23 | 57 | 167 | object_lost | 2.621 |
| 24 | 57 | 145 | object_lost | 0.971 |
| 39 | 65 | 158 | success | 0.628 |
| 47 | 70 | 148 | object_lost | 0.019 |
| 59 | 61 | 164 | success | 0.042 |
| 82 | 54 | 152 | success | 0.302 |
| 86 | 63 | 127 | success | 2.554 |
| 90 | 48 | 156 | success | 0.389 |

`task_budget` 是恰好耗尽 1200 物理步的有效预算截断，不是手工中断。全部失败及预算截断数据均保留，未剔除失败场景。

证据：[正式配对汇总](../runs/phase2/phase2_comparison_final_v1/comparison.json)、[结果图](../runs/phase2/phase2_comparison_final_v1/paired_outcomes.png)、[seed 0 轨迹对照](../runs/phase2/phase2_comparison_final_v1/seed0_mechanism.png)、[B1 批次审计和失败索引](../runs/phase2/phase2_followups_v3_chunk16/report.json)。

### 来源修正与配对限制

原始 B0 为 609 控制、平均 60.9；seed 24 原尝试为 59 控制，启动哈希与延迟导入代码存在已披露的来源例外。事先指定的新 attempt `phase2_followups_v3_seed24_provenance_repeat` 为 57 控制后丢失，重放仍不匹配，来源哈希核对通过。正式选定十场景采用该复跑，无论结果是否改善；原尝试保留为排除记录，不能额外作为第十一个场景。[原始报告](../runs/phase2/phase2_baseline_v1/report.json)、[来源说明和修改前后文件](../runs/phase2/phase2_baseline_v1/source_versions/history_mask_correction.json)。

同 seed 的独立初始化并非相同隐状态反事实：B0/B1 初始末端差最大 2.621 mm、初始关节差最大 0.04254 rad。配对汇总逐项核对模型身份、seed、场景父 ID、物理／控制／观测／预算配置，并保留这些初始差异。本结果是开发集配置对照，不是独立测试集泛化成绩或严格隔离的因果效应估计。

## 失败诊断与归因

原始 B0 的 609 个动作、seed 24 复跑的 57 个动作、夹爪对照的 46 个动作，以及 B1 的 **1600 个动作**均已对照固定官方纯映射函数和保存的原始输入／chunk。状态布局、输入重编码、原始预测到目标转换的最大误差均为零；关节增量限幅均未触发。B1 最大已测关节跟踪误差约 0.000230 rad，原 B0 约 0.000334 rad。

官方固定源码默认路径为 **120 Hz／直接写 DOF 位置／chunk16**；本实验物理接口为 **20 Hz／位置驱动**。checkpoint 的历史训练频率没有得到元数据证明。chunk1 改为 chunk16 后，开发集成功数从 0 增至 6，说明不能将原始全败直接归为 FTP-1 本身无效；但频率、驱动方式和初始化差异仍限制进一步归因，不能称为官方部署成绩复现。

视频、动作和触觉显示，部分失败伴随下压、孔口阻滞及夹持相对滑移。`object_lost` 使用原任务的 40 mm 持握相对位移判据，并不等同于已经自由落体；触觉 RGB 变化也不是已标定的力读数。成功判据还包含横向位置、朝向、插入深度及持握稳定性，报告使用实际任务判据。

**B2 夹爪单场景对照**：seed 0 将单指目标固定为初始约 5.743 mm，仍在 46 控制后丢失（原 B0 为 57）。预测与覆盖目标最大差 0.403 mm；视频仍显示下压和相对滑移。该配对初始末端差 1.50 mm、关节差 0.0182 rad，不能从一次独立重置估计总体夹爪效应；保持初始夹爪目标未解决该次失败。[夹爪诊断](../runs/phase1/phase2_followups_v3_hold_gripper_seed0/diagnosis.json)

已修复的真实接口缺陷是下一观测的 history mask 滞后一拍。当前 FTP-1 checkpoint 禁用历史，此缺陷不能解释模型全败；旧记录未回写，后续学习读取早期历史时必须按真实前缀重建 mask。

每个 baseline、复跑、夹爪及 B1 运行目录都保存 `diagnosis.json`、逐步 CSV、曲线、四路同步视频、关键帧、模型输入和原始 chunk；对应 `datasets/phase1/` 保存无损 HDF5。

## 扩大重放结果

在第 12 个基础动作后导出，完整基线结束后另建分支重建，正式选定十场景的 v3 匹配率为 **0/10**。冻结门槛未修改。B1 本轮未另做重放，不能将其零次尝试写成 0/10。

原始 B0 十场景的图像／冻结编码器诊断发现 seed 47 和 86 物体旋转误差约 **8.19°、6.04°**，对应编码器相对 L2 均值约 **5.85%、5.28%**。其他场景特征较接近，也仍未通过原物理／观测门槛。相机像素差异仅作附加诊断，v3 对相机本身检查有效性与采样时间。

这些失配记录不进入正式配对恢复效果标签。完整误差和旧 seed 24 特征表的来源限制见 [重放诊断](Phase2_Replay_Diagnosis.md)。

## 真实恢复交接

GPU 实验及独立持久化文件审计均通过：真实 chunk16 先执行四步，丢弃剩余 **12** 个目标；执行四步脚本恢复，再清队列、重新观测／推理，继续执行四个新 FTP-1 目标。

物理边界 **536→560→584→608**，共 **72** 步：基础 **48**、恢复 **24**；恢复同时计入任务总预算。恢复后请求 ID=1，首个继续动作从 584 开始。两次保存输入与实际观测、官方动作解码的误差均为零；传感器 age=0，previous_action／history mask／预算对齐，没有旧动作残留。

证据：[运行检查](../runs/phase1/phase2_followups_v3_handoff/checks.json)、[独立审计](../runs/phase1/phase2_followups_v3_handoff/handoff_independent_audit.json)、[同步视频](../runs/phase1/phase2_followups_v3_handoff/handoff_synchronized.mp4)。这是按计划人工结束的有界交接实验，不纳入完整任务成功率，不代表脚本恢复效果已验证。

## 学习型恢复准备与下一阶段

### 2026-09-21 当前代码补充证据

当前 marker RNG/noise 修复后的 seed23 standalone replay 在
`../runs/phase1/phase2_current_standalone_replay_seed23_20260921/checks.json` 中记录
`valid_match=true`、`observable_match=true`、`versions_match=true`。同一 source 的完整
no-recovery/scripted-recovery 配对在
`../runs/phase1/phase3_paired_current_seed23_full_20260921/checks.json` 中记录
`paired_valid=true`；两条 branch 分别执行 143 个 continuation controls，以及 4 个 recovery
controls 加 161 个 continuation controls，最终都为 `object_lost`、`valid_trial=true`、reward 0。
这验证了当前 P2/P3 的起点重建、恢复动作、handoff 和预算审计，但没有显示恢复收益。

当前代码的 4-parent/8-repeat fresh-process 校准见
`../runs/phase1/calibration_headless_current_v2_20260921/checks.json`。seed18/24 仍有
明显 prism/marker 漂移，生成的 current.v2 容差仅作诊断，不替换冻结 v3。

P3 的最小真实交互证据见 `../runs/phase1/phase3_actual_transition_seed23_20260921/checks.json`：controlled-trigger 采纳 3 条真实 recovery transition 并完成 3 次 batch-1 SAC 更新。train-split seed1 的 3 条 transition 已提取到 `../datasets/phase1/recovery_history_train_seed1_20260921.npz`，并由 `train_recovery_warmstart.py` 生成冻结 checkpoint。触发阈值和 batch 是 plumbing 覆盖参数，不用于报告恢复成功率。

官方 encoder 已完成固定 revision／LFS SHA256 校验、完整严格加载、旧／新帧前向、单指 512 维输出、参数及 BatchNorm 冻结、特权输入隔离。特征显式包含当前／空载／持握参考、图像差分、本体、上一接受动作、历史／时间有效性和预算；详见 [特征契约](Recovery_Feature_Interface.md)。接口正确性不等于恢复语义或训练效果已验证。

完整 CPU 测试 **92 项通过**：`python -m pytest evotac/tests -q -o cache_dir=evotac/.pytest_cache`。真实仿真与文件审计另行验收，全部批次状态完成：[调度记录](../runs/phase2/phase2_followups_v3/status.json)。

下一阶段优先固定 chunk16 作为物理恢复实验的基础配置，保留 B0 供复核；先解决配对重放可用性，或另立连续在线采样协议并以真实回报训练。随后验证单技能 SAC 的数据采集、replay buffer、一次梯度更新、终止／预算处理和 FTP-1 接续，再扩展技能。独立测试父场景不用于当前开发调参。详细门槛与顺序见 [阶段 2 计划](Phase2_Plan.md)。

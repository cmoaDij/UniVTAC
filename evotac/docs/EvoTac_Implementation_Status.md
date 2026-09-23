# EvoTac 实现状态

更新时间：2026-09-21。版本前提是作者 `isaac51` 分支：Isaac Sim 5.1.0、Isaac Lab 2.3.0、PyTorch 2.7/cu126 和仓库内 TacEx。不存在把本地实现当作 4.5 版本运行的假设。

| 项目 | 状态 | 证据/限制 |
|---|---|---|
| P0 版本、配置与路径隔离 | 已完成 | `evotac/configs/phase1_insert_hole.yaml` 固定 120/20 Hz、taxim、GS Mini；manifest 写入 Isaac 5.1/ Lab 2.3 和源码哈希 |
| P1 wrapper/动作/观测/失败落盘 | 已完成 | `evotac/tests/` CPU 契约测试；所有终止原因和不完整文件保留 |
| FTP-1 chunk16 基线 | 已完成 | 10 个 dev seed：6/10 success、3/10 object_lost、1/10 task_budget，1600 个控制动作 |
| v3 配对重放 | 旧正式 cohort 未完成；当前 seed23 开发配对已通过 | 旧 v3 继续冻结；当前 standalone replay `phase2_current_standalone_replay_seed23_20260921` 严格通过，完整 paired run `phase3_paired_current_seed23_full_20260921` 的两 branch 均 strict、均以 `object_lost` 终止，不能外推成功率 |
| 分阶段重放诊断 | 当前 seed23 可用；多 seed strict cohort 未冻结 | marker RNG/marker RGB 噪声已修复；当前 4 parents/8 fresh-process calibration 中 seed0/23 稳定、seed18/24 漂移，诊断容差不替换 v3，也不用于正式恢复标签 |
| 恢复交接流程 | 完整 seed23 paired wiring 和 terminal audit 通过，效果未证明 | no-recovery 143 continuation controls；scripted-recovery 4 recovery + 161 continuation controls；两 branch `valid_trial=true`、`object_lost`、reward 0 |
| 单技能 SAC 代码骨架 | 实现完成；真实 collector/headless、transition、更新链路通过 | `recovery_buffer.py`、`recovery_sac.py`、`recovery_rollout.py`、`recovery_training.py`、`recovery_trainer.py`、`recovery_observation.py`、`history_encoder.py`、`recovery_warmstart.py`、`runtime_monitor.py`、`trigger_calibration.py`、`recovery_state_machine.py`；GPU8 controlled-trigger run 采纳 3 条真实 transition 并完成 3 次更新；新的 train-only 校准 artifact 和 dev/test fail-closed provenance 已通过 111 项 CPU 测试 |
| SAC 真实交互训练 | 已完成最小 plumbing 更新；策略效果未验证 | seed23/seed1 真实 headless run 均完成 3 recovery actions/3 updates；train-split seed1 已提取 3 条 transition 并生成冻结 warm-start checkpoint。当前仍使用显式触发、batch1 和单父场景，不能宣称成功率提升 |
| P4 效果预测/选择/复用 | CPU 初始实现完成 | `effect_model.py`、`effect_training.py`、`skill_selector.py`、`effect_selection.yaml`、`build_effect_labels.py`；训练器支持独立效果历史分支、mask、可选 metric loss 和 checkpoint，目前没有具备显式 violation 的真实技能效果标签 |
| P5 持续适应/接纳 | CPU 初始实现完成 | `continual_adaptation.py`、`continual_adaptation.yaml`、`accept_adaptation.py`；controller 串联 50/25/25 采样、配对 gate、版本和容量状态，尚无真实仿真候选 |
| P6 消融实验 | 协议与审计代码完成，surrogate smoke 完成，物理实验未开始 | `phase6_ablation_plan.json`、配对父场景 runner、`run_fast_ablation.py`、`audit_phase6_report.py` 和 JSONL/JSON 审计已实现；surrogate 完成 4,800/6,000/4,800 episode（按三组顺序），真实结果仍需 P2–P5 冻结版本和独立父场景 |
| GPU/墙钟性能测量 | 已加入工具 | executor 保存 `wall_physics_hz`；manifest 记录 `nvidia-smi` 快照；`profile_rollout` 可从 HDF5 统计；`profile_physics` 分项记录物理／渲染／触觉／观测／日志；二进制场景哈希把 chunk16 构建平均降至约 1.55 s |
| headless 性能分支 | 已完成有界真实 profile、校准和独立数值诊断 | `--performance-profile` 保留相机/触觉但关闭 livestream；通过 `CUDA_VISIBLE_DEVICES=8` 将物理 GPU 8 映射为逻辑 `cuda:0`，headless 与 livestream 各完成 4 controls/24 physics steps；执行段分别为 27.246/4.541 Hz 与 22.523/3.754 Hz。headless calibration 使用 4 dev parents/8 repeats，seed 47 独立候选容差审计通过；完整 paired fresh reset 仍超出候选范围，不能把 diagnostic numeric pass 或短 profile 当作旧 v3 严格配对、策略成功率或正式吞吐基线 |
| GPU 忙时快速逻辑验证 | 已加入 surrogate | `simulation/fast_insert_hole.py` 保留 120/20 Hz、6 步控制边界、触觉掩码和三类条件；仅用于控制契约/消融 smoke，不替代 TacEx 物理 |

## 当前实测速度与 GPU 争用

从已保存的 chunk16 HDF5 执行记录汇总：1600 个有效控制、9600 个物理步、墙钟 442.07 秒，
`wall_control_hz=3.62`、`wall_physics_hz=21.72`。这表示当前机器在该负载下每秒只完成约 21.7 个 120 Hz 物理步；它不表示仿真时间被改成 21.7 Hz。配置的模拟时间仍按 120 Hz 前进。

采样时 `nvidia-smi` 显示 10 张 RTX 5880 Ada 均接近 100% 利用率，单卡约 32–34 GiB 已用；另有多个 `lerobot_train` 进程占用显存。因而“其他人正在使用 GPU”是历史低墙钟 FPS 的强证据。已在空闲 GPU 8（映射为逻辑 `cuda:0`）完成同 seed 的 headless/livestream 初步 A/B；两条各 4 controls/24 physics steps，reset 终点分别为 526/515，尚未严格配对隐藏状态。

2026-09-20 的 profile 预检曾在 Kit 启动前拒绝 GPU 0：仅 14,829 MiB 可用，而 profile 默认要求 20,000 MiB。直接使用物理 `cuda:8` 的尝试因 peer mapping 和 UIPC 的非 `cuda:0` 限制在 reset 超时；失败记录在 `runs/phase1/phase1_profile_gpu8_real_20260920/checks.json`。将 GPU 8 映射为逻辑 `cuda:0` 后，headless `phase1_profile_visible8_real_20260920` 与 livestream `phase1_profile_visible8_livestream_real_20260920` 各完成 4 个控制周期，执行段分别为 27.246/4.541 Hz 与 22.523/3.754 Hz。两者都是有界 profile，不能外推为策略 FPS；后续正式吞吐应增加控制周期并保持同一启动模式。

GPU 占用期间可运行 `python -m evotac.scripts.run_fast_simulation`，用于验证 observable-only policy、固定六步控制、恢复预算和 P6 方法接口。最近一次 24 个 episode 的 surrogate smoke 为约 285 episodes/s（`runs/fast_validation_smoke3/report.json`）；该数字只表示轻量逻辑执行速度，不能外推为 Isaac/TacEx FPS。

公开的 TacEx 基准也说明了为什么不能拿普通 Isaac Lab FPS 直接类比本任务：在单环境、480×640 触觉图像、10×10 marker 的实验中，height-map、GPU Taxim 和 CPU FOTS 分别约为 1.37、5.90、4.49 ms/frame；GIPC 软体物理随网格从 1,029 vertices 的 24.95 ms 增到 12,509 vertices 的 221.61 ms，而且软体配置只能稳定运行单环境并受显存限制。该表不是本机 5.1 的实测值，但足以说明 tactile/FEM 是独立于 viewport 的成本来源。

## 已知风险

- 传感器负载不是普通无相机 Isaac Lab benchmark：当前同时运行两路 480×270 RGB 相机、双 GelSight Mini 的 rgb/marker 输出和 taxim/TacEx 接触更新。
- 5.1 的 headless 不会自动跳过传感器渲染；如果关闭相机或触觉，得到的只是性能上界，不能用于 FTP-1/恢复效果比较。
- 当前代码只允许通过显式 `--performance-profile` 进入 headless，避免把已有 livestream 验收数据和无直播性能数据混在一起。

## 下一步

P2 当前 seed23 的 standalone/paired audit 已通过，但 seed18/24 的 fresh-process 校准仍暴露 UIPC/FEM/tactile 漂移；按 [分阶段计划](EvoTac_Phase_Implementation_Plan.md) 先稳定 reset 起点并取得可追溯 strict cohort，再扩大多 seed 同预算恢复对照。P3 已有 3 条真实 transition、3 次 plumbing SAC 更新和 train-split warm-start；下一步是在 GPU 空闲时用冻结 checkpoint 做一次标准配置运行，随后校准监测器、使用标准 batch 和更多训练父场景评估恢复效果。

性能拆解、headless A/B 矩阵、TacEx 公开耗时参考和独占 GPU 后的尝试顺序见 [Performance_Analysis_and_Optimization.md](Performance_Analysis_and_Optimization.md)。

## 阶段记录

按 P0–P6 的逐阶段计划、代码、证据和边界记录见 [stages/](stages/README.md)。新增的 `profile_physics.py` 只做有界真实控制和分项计时，不改变冻结配置，也不把 CPU smoke 当成物理效果。

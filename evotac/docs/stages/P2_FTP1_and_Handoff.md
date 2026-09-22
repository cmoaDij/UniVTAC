# P2：FTP-1 基线、起点重建与恢复交接

状态：**当前代码的 seed23 起点重建和完整 paired terminal run 已通过；跨 seed strict cohort 仍未冻结**（2026-09-21）。

## 计划

冻结 FTP-1 chunk16，完成 B0/B1 开发对照，定位策略前缀首次重放差异；只有通过冻结 v3 阈值的起点才可用于配对恢复标签。完成一次非空 FTP-1 队列暂停、恢复动作、重新观测／推理和继续执行。

## 已实现与结果

- 十场景 B0/B1、夹爪诊断、来源修正和失败保留已完成；chunk16 为 6/10 成功、3/10 丢失、1/10 预算截断。
- `replay_chunk16_prefix.py` 和 `diagnose_chunk16_replay.py` 提供分阶段重放报告；seed 0 的一次 physics-only 重放通过，seed 47/86 的真实重放均完成但 v3 为 `false`。seed 47 在 boundary 0 已有 tactile marker 最大 21 灰度差、boundary 1 物体姿态约 `3.59e-7 m / 0.000652°`；seed 86 对应 `23` 灰度差、`1.79e-7 m / 0.000129°`。报告分别保存在 `runs/phase2/phase2_chunk16_replay_phys_seed47_visible8_v4_diagnosis/` 和 `runs/phase2/phase2_chunk16_replay_phys_seed86_visible8_v1_diagnosis/`，正式扩大集仍为 0/10。
- `policy/handoff.py` 与 `verify_ftp1_handoff.py` 验证丢弃 12 个旧目标、恢复 24 物理步、重新推理并接续四个目标。
- `policy/tactile_features.py` 严格加载冻结触觉编码器，隔离 actor 真值和标签。
- `replay_chunk16_prefix.py` 对旧持久化场景只允许 EvoTac 自身源码哈希变化；schema、动作、配置、上游资产和 Isaac 版本仍必须逐项相等，并把比较结果写入 `version_comparison`。
- replay 检查文件同时写入 `hardware_comparison`：本轮源数据在物理 GPU 0，seed 47/86 replay 在物理 GPU 8；启动模式相同但 GPU 不同，因此该轮只能作为差异定位证据，不能升级为严格硬件配对验收。
- 为区分“跨 GPU 差异”和“同 GPU 重放不稳定”，新增 `--diagnostic-code-drift --repeat-current`。修正重复逻辑后，seed 47 从同一个不可变场景在同一进程、同一当前代码、同一物理 GPU 8 独立重放两次：两次 physics 起点均为 555，关节/末端位置在 boundary 0 完全相同，但触觉 RGB 最大差为左 34、右 32，marker 最大差为左 168、右 164；episode 的 observation、training tree 均未逐项相等，最终物体 `prism` 姿态误差约 `0.0697`（见 `runs/phase1/phase2_chunk16_replay_samegpu_seed47_v3/checks.json`）。因此当前 TacEx/UIPC reset/FEM 路径存在需要进一步隔离的同 GPU 非确定性；不能把 seed 47/86 的失配仅归因于 GPU 映射，也不能放宽 v3 阈值来接收这些样本。
- `--trace-reset` 已对同一进程的两次 reset 记录 UIPC object position/velocity、solver frame 和机器人状态。`reset_trace_report.json` 显示 `actors_reset`/`recovered` 的 object state 完全一致，第一次可观测差异出现在 marker calibration 结束（physics 27，`prism` position 最大约 `1.62e-6`）；到 prefix 的接触阶段（physics 348）`prism` position/velocity 差异增至约 `1.29e-3/2.27e-3`。这把后续排查范围收窄到 FEM/UIPC 接触演化与触觉更新顺序，而不是机器人目标写入。
- 已修复 TacEx FEM marker 后处理的隐式随机性：`VisionTactileSensorUIPC` 使用每个传感器独立的 RNG，并在 reset 时恢复种子；此前的全局 `np.random.choice` 会使 marker subset/marker RGB 随进程历史变化。该修改改变了 vendor 文件哈希，旧数据不会被伪装成同版本配对；必须重新生成基线后再做严格 v3 replay。已有 trace 是修复前的定位证据。
- 在显式 implementation-drift 模式下复测同 GPU seed 47：两次 replay 的 action tree（忽略 wall-clock 字段）完全一致；boundary `0/1/4/8/12` 的 tactile RGB 与 marker RGB 最大差都不超过 24 灰度，物体位置最大差约 `2.39e-7 m`、姿态约 `0.000122°`（`runs/phase2/phase2_markerfix_seed47_repeat_report.json`）。因此 marker 后处理的独立随机性已消除；剩余触觉 RGB 和接触状态差异仍使 observation tree 不相等，严格 v3 仍不能通过。
- 进一步修复了 `ManiSkillSimulator.draw_markers()` 无视配置、始终执行 `torch.rand_like()*80` 的问题：现在 `marker_random_noise=0` 时跳过随机数生成，非零配置才加噪声。修复后重新生成的 seed 47 headless 短基线与独立进程 replay 使用同一物理 GPU 8、同一启动模式和同一版本哈希（`runs/phase1/phase2_markerfix_noiseoff_source_seed47_v1/`、`runs/phase1/phase2_markerfix_noiseoff_replay_seed47_v1/`）；动作树仍完全一致，marker RGB 已跟随物理差异而不再出现独立随机噪声，但严格 v3 仍为 `valid_match=false`。独立 replay 的右指触觉 RGB RMSE 为 `1.100757`，超过冻结上限 `1.080226`，history 还有少量 RMSE 超限；同代码重复的 observation tree 仍不相等。
- 针对 headless profile 重新用 dev seed `0/18/23/24`、每个 parent 两次 fresh-process replay 校准了候选版本 `standard_init_headless_markerfix_dev.v1`（8 reports，`runs/phase1/calibration_headless_markerfix_v1/`）。候选上限只由这四个 dev parent 的最大重复误差乘 1.2 得到，并保留 float32 精度下限；没有改写冻结 v3。未参与校准的 seed 47 在同一物理 GPU 8、同一 headless 启动模式下做独立 replay，所有观测误差均落在候选上限内，`candidate_numeric_validation=true`，证据为 `runs/phase1/phase2_markerfix_noiseoff_replay_seed47_headless_candidate_v3/independent_tolerance_validation.json`。由于 source 使用旧代码/配置 fingerprint，该运行保留 `diagnostic_implementation_drift`、`strict_valid_match=false`；这证明候选容差的数值范围具备独立支持，不把旧持久化数据伪装成新版本严格配对。独立同代码 repeat 的 observation tree 仍为 `exact_match=false`，所以正式严格 cohort 仍需用修复后的代码重新生成。
- 在新增 counterfactual 入口后重新生成了当前 fingerprint 的 seed 23 source（`runs/phase1/phase2_headless_candidate_source_seed23_v2/`）。同版本 standalone replay 和一次 same-code repeat 曾为 `valid_match=true`、`versions_match=true`；旧版 bounded runner 产物 `runs/phase1/phase2_headless_counterfactual_seed23_v2/checks.json` 两条均以 `counterfactual_control_limit` 截断，只确认了 paired wiring。随后 v3 runner 支持在同一不可变运行中先生成 source、共享 total budget 和重置 worker RNG；16-control bounded run 的两条 prefix replay 也 strict 通过（`runs/phase1/phase2_headless_counterfactual_seed23_v3/checks.json`）。但扩大到完整剩余预算时，fresh reset 的 tactile marker/RGB 差异超过候选容差，两个 branch 都被 `valid_match=false` 拒绝（`runs/phase1/phase2_headless_counterfactual_seed23_full_v2/checks.json`）。因此候选容差目前只能作为诊断/开发版本范围，不能宣称正式 strict cohort 已稳定；需要继续隔离 UIPC/FEM reset 漂移或重新设计可审计的配对起点。
- 重放场景的完整性校验改为 `evotac.scene_binary.v1`，对 ndarray 直接规范化哈希；seed 47 chunk16 场景构建基准三次平均约 1.55 s（`runs/phase2/scene_build_benchmark.json`），避免把大触觉数组展开成 JSON 后再哈希。

## 2026-09-21 当前代码验收

- 用 marker RNG/noise 修复后的当前 fingerprint 重新生成 seed 23 source，并在独立 Kit 进程做 standalone replay：`runs/phase1/phase2_current_standalone_replay_seed23_20260921/checks.json`。`status=replayed`，`valid_match=true`、`observable_match=true`、`versions_match=true`、`history_match=true`、`references_match=true`；这是一条可用于继续开发的严格 seed23 起点证据。
- 对同一 source 做完整的 no-recovery/scripted-recovery 配对，命令产物为 `runs/phase1/phase3_paired_current_seed23_full_20260921/checks.json`。`paired_valid=true`，两条 branch 的 prefix reconstruction 均 strict 通过；no-recovery 使用 143 个 continuation controls，scripted-recovery 使用 4 个 recovery controls 加 161 个 continuation controls。两条都以 `object_lost` 终止、`valid_trial=true`、reward 为 0，因此证明了共享起点、恢复动作提交、handoff 和同预算终止记录，尚未证明恢复收益。
- 为避免把单个 seed 外推成 cohort，当前代码又做了 4 个 dev parent、每个 2 次 fresh-process replay 的小规模校准：`runs/phase1/calibration_headless_current_v2_20260921/checks.json`。seed 0/23 在现有候选范围内稳定，seed 18/24 出现明显 prism/marker 漂移；因此生成的 `standard_init_headless_markerfix_current.v2` 只作为诊断统计，不能替换冻结 v3 或扩大正式标签接纳范围。

## 证据与边界

真实 handoff 和 seed23 paired run 证明了控制边界、版本门禁、恢复交接和终止审计；它们不证明恢复提升成功率。旧 v3 仍保持冻结并拒绝越界前缀；headless 候选仍只能作为开发范围。下一步应先隔离 UIPC/FEM/传感器 reset 漂移并取得稳定 strict cohort，再扩展多 seed 恢复对照和恢复标签。

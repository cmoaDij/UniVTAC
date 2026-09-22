# P3：单恢复技能 SAC

状态：**单恢复技能代码、headless collector、真实 transition 入 buffer 和一次 SAC 更新已验证；策略有效性仍未证明**（2026-09-21）。

## 计划

先使用通过 P2 起点门槛的真实失败交互采集恢复片段，再执行单技能 SAC；恢复交还后的基础策略结果只作为最后一个恢复 transition 的实际回报，基础动作不得混入恢复 buffer。

观测路径固定为冻结 ResNet18（左右指、当前/空载/持握三组）和相机统计 → 3162D 单帧 → 八帧因果 GRU → 128D SAC 状态。控制历史和 P4 效果历史使用不同模块；输入包含采样间隔、图像年龄、参考时刻和已接受目标，不能从未来帧补齐。

## 已实现

- `learning/recovery_buffer.py` 保存真实 Transition，严格检查归一化动作、终止和 bootstrap 标记，并支持可审计 checkpoint。
- `learning/recovery_sac.py` 实现连续动作 actor、双 critic、目标网络、熵温度和一次更新。
- `learning/skill_registry.py` 保留 P3 的单技能 smoke registry，同时提供三种动作兼容的计划技能槽；槽位声明不等于真实效果验证。
- `learning/recovery_rollout.py` 保留 pending 片段，handoff 后把真实 continuation reward 加到最后一项并关闭 bootstrap。
- `learning/recovery_state_machine.py` 实现 `NORMAL → SELECT → RECOVERY → HANDOFF → NORMAL/STOP`。
- `learning/recovery_training.py` 串联真实 wrapper、基础策略、恢复策略、监测器和 collector；交还后的终止结果才写入最后恢复 transition。
- `learning/recovery_trainer.py` 将已接纳的恢复 transition 送入 SAC，按固定更新比执行梯度更新，并支持带 schema 的 learner/buffer checkpoint；基础策略动作不会被加入恢复 buffer。
- `learning/recovery_observation.py` 接受真实的 `[2,2,9]` 双参考/双指触觉 cue，将双指 latent、差分 cue、相机 RGB 统计、本体、上一接受目标和时间预算保留在固定 3162D 帧中；`history_encoder.py` 用长度 8 的 128D GRU 状态供 SAC 使用。`CachedTactileEncoder` 按内容缓存重复帧，避免同一控制周期重复前向。
- `scripts/train_recovery.py` 提供 GPU 空闲后的真实入口：冻结 FTP-1、严格触觉编码器、RecoveryCollector、SAC trainer 和带显存预检的运行记录会被串联到同一 run；`--performance-profile` 可使用 headless Kit，同时保留相机和触觉传感器。
- `scripts/extract_recovery_history.py` 从 HDF5 的部署观测和已执行 `recovery_delta` 动作生成 `[N,8,3162]` 窗口，并拒绝混用 split、重复父场景和无效动作；`scripts/train_recovery_warmstart.py` 只接受显式 `split=train` provenance，训练后导出冻结 HistoryEncoder/技能 actor checkpoint。
- 在线驱动默认要求连续三个有效控制周期稳定后才 handoff；`stable_cycles` 可在契约 smoke 中显式缩短，但正式配置不应绕过该门槛。
- 真实入口默认拒绝未暖启动的随机历史编码器；须提供训练父场景得到的 `HistoryEncoder` checkpoint，只有显式 `--allow-uninitialized-history` 才能运行接口 plumbing smoke。

## 2026-09-21 当前运行

- `runs/phase1/phase3_real_collector_seed23_20260921/checks.json` 是先前使用默认风险阈值的自然触发基线：GPU8（通过 `CUDA_VISIBLE_DEVICES=8` 映射为逻辑 `cuda:0`）上的真实 FTP-1、双 GelSight/UIPC、collector 和 SAC trainer 接线正常，GPU 预检记录 49,114 MiB 可用；episode 以 `object_lost` 终止，`valid_trial=true`，但自然阈值下 `recovery_actions=0`、`admitted_transitions=0`。后续 controlled-trigger 运行已单独验证 transition 链路，没有把基础策略失败伪装成恢复样本。
- `runs/phase1/phase3_paired_current_seed23_full_20260921/checks.json` 的 scripted branch 已实际提交 4 个恢复控制并完成 handoff；这验证了 P3 所需的动作边界、恢复预算、重新观测/推理和 continuation 记录。该 branch 与 no-recovery branch 都是 `object_lost`，不能据此声称 SAC 或恢复策略有效。

## 2026-09-21 当前运行（补充）

- `runs/phase1/phase3_actual_transition_seed23_20260921/checks.json` 使用显式可观测风险触发（阈值 0，仅 plumbing）和 batch size 1，执行 3 个恢复控制、handoff、采纳 3 条真实 transition，并完成 3 次 SAC 更新；终止结果仍为 `object_lost`，因此这是数据/优化链路证据，不是收益证据。
- 为得到合法的训练 provenance，在 train split 的 seed1 上重复一次单 episode 采集：`runs/phase1/phase3_train_transition_seed1_20260921/checks.json`。从其 HDF5 提取的 3 条 transition 位于 `datasets/phase1/recovery_history_train_seed1_20260921.npz`，并训练出 `runs/phase1/phase3_train_transition_seed1_20260921/history_warmstart.pt`；加载检查确认 `frozen_warmstart_checkpoint`、skill actor 和 3162D→128D HistoryEncoder 一致。
- 随后尝试用该 frozen warm-start 在 GPU8 做一次真实在线运行，但 GPU 预检发现可用显存仅 15,544 MiB（要求 20,000 MiB），在 Kit 启动前安全拒绝；没有产生半成品仿真数据。

## 证据与边界

CPU smoke 已完成一次梯度更新，92 项测试通过；这只证明数据和优化契约。headless 单 episode 接线 smoke 已完成：`runs/phase1/phase3_plumbing_headless_gpu8_20260920/checks.json` 记录成功的真实 FTP-1 链路。随后 controlled-trigger 运行已经采纳真实 transition 并完成 SAC 更新，但仍使用 plumbing 触发和小 batch；正式训练还需要冻结 warm-start、校准监测器、标准 batch 和多父场景效果对照。

当前真实 HDF5 提取记录为 `runs/phase2/recovery_history_extract_dev0/report.json`：4 个 transition、单一 dev 父场景、`train_checkpoint_allowed=false`。它验证了提取链路，不构成训练数据或效果证据。

真实运行前置检查：`train_recovery.py --dry-run` 只校验配置；正式恢复训练还需要稳定的 P2 匹配失败集合、更多训练父场景、经过校准的监测器阈值、冻结 `HistoryEncoder` checkpoint 和标准 batch 的更新。`extract_recovery_history.py` 读取 wrapper 每帧记录的恢复预算（handoff 后保持冻结），再用 `train_recovery_warmstart.py` 生成 checkpoint；`--allow-uninitialized-history` 与风险阈值覆盖只用于 plumbing smoke，产生的结果不能直接进入效果标签或 P6。

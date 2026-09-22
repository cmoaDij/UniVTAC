# P0：版本、配置与速度基线

状态：**已实现；两种启动模式初步测速完成，严格配对 A/B 待验证**（2026-09-20）。

## 计划

锁定 isaac51、Isaac Sim 5.1、Isaac Lab 2.3、本地 TacEx、120 Hz 物理、20 Hz 控制、taxim、GelSight Mini 和单环境；每次运行写入配置哈希、源码哈希、资产哈希、GPU 快照和真实墙钟速率。速度测量区分控制执行段和端到端策略时间。

## 已实现

- `evotac/config.py` 拒绝不符合冻结频率、后端、传感器和控制点的配置。
- `evotac/scripts/runtime.py` 生成 manifest，记录 GPU 与启动模式。
- `evotac/perf/runtime_probe.py` 统计 `wall_physics_hz` 和 `wall_control_hz`。
- `evotac/perf/scoped_timing.py` 与 `evotac/scripts/profile_physics.py` 增加真实 Isaac/TacEx 的分项 CPU 墙钟计时，保留相机和触觉输出。
- `profile_physics --dry-run` 可在 GPU 忙时先校验 120/20 Hz、六步 decimation、相机和双触觉配置，不触发 Isaac/EULA。
- 性能 profile 可用 `--flush-every 16` 或更高值减少 HDF5 flush 次数；该选项只改变日志持久化频率，manifest 会记录它，正式验证仍使用配置冻结的每步 flush。
- `simulation/fast_insert_hole.py` 和 `run_fast_simulation.py` 提供 GPU 忙时的快速控制/消融 smoke，仍固定 120/20 Hz 和六步 decimation，并明确标记为 surrogate 证据。
- 2026-09-20 的真实 profile 在导入 Kit 前被 GPU 预检拒绝：GPU 0 只有 14,829 MiB 空闲、利用率 100%，低于 20,000 MiB 门槛；因此没有浪费一次长 reset 或改变其他训练进程。

## 最新真实启动尝试

直接把物理 GPU 8 传给 Kit 的尝试仍会触发 `cudaErrorTooManyPeers`，并被 TacEx/UIPC 拒绝（当前实现要求逻辑 `cuda:0`）；该失败保留在 `runs/phase1/phase1_profile_gpu8_real_20260920/checks.json` 作为启动兼容性证据。随后用 `CUDA_VISIBLE_DEVICES=8` 将物理 GPU 8 映射为逻辑 `cuda:0`，完成了真实 headless profile：`phase1_profile_visible8_real_20260920`，4 个控制周期、24 个物理步、全部 `valid_trial=true`，执行段 `27.246 wall physics Hz`、`4.541 wall control Hz`，控制循环（含观测和落盘，不含 reset）`2.594 control Hz`。检查文件为 `runs/phase1/phase1_profile_visible8_real_20260920/checks.json`，HDF5 episode 与 manifest 保存在 `datasets/phase1/phase1_profile_visible8_real_20260920/`。

同一映射方式下的 livestream 对照为 `phase1_profile_visible8_livestream_real_20260920`，4 个控制周期、24 个物理步、`22.523 wall physics Hz`、`3.754 wall control Hz`；检查文件和 HDF5 位于同名 `runs/phase1/` 与 `datasets/phase1/` 目录。两次 reset 终点分别为 526 和 515，因此这是一条初步启动模式对照，不是严格隐藏状态配对。

## 证据与边界

历史 chunk16 执行段为 21.72 wall physics Hz、3.62 wall control Hz；采样时 GPU 全部高负载。同一物理 GPU、seed 和传感器配置下，headless profile 为 27.246 wall physics Hz、4.541 wall control Hz，livestream profile 为 22.523 wall physics Hz、3.754 wall control Hz；两者各覆盖 4 个有界控制周期且全部动作有效。headless 控制循环（含观测和落盘，不含 reset）为 2.594 control Hz。该 A/B 只说明启动模式的执行段差异，不能直接外推策略 FPS 或成功率。

独立从已保存的 `phase2_followups_v3_chunk16_seed0` HDF5 重算得到 183 controls、1,098 physics steps、50.889 s、21.576 physics Hz 和 3.596 control Hz，说明统计工具可复核保存记录。

最近的 surrogate sweep 覆盖 3 个 seed、4 个 condition 和 base/EvoTac 两种策略，共 24 个 episode，逻辑执行速度约 285 episode/s；其成功率只用于检查接口和报告管线，不用于物理结论。

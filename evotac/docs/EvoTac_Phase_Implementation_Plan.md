# EvoTac 分阶段实现计划

本文把 `myideas/` 中的方法设计转换为可执行的仓库阶段，并以作者的
`isaac51` 分支（Isaac Sim 5.1 / Isaac Lab 2.3）为唯一当前版本前提。原始设计仍保留在
[方法与 UniVTAC 方案](../../myideas/EvoTac_Method_and_UniVTAC_Plan.md) 和
[阶段 1 方案](../../myideas/EvoTac_Phase1_Implementation_Plan.md)；阶段 2 的实验顺序以
[Phase2_Plan.md](Phase2_Plan.md) 为准。

## 冻结条件

后续可比较实验固定 `FTP-1 chunk16`、120 Hz 物理、20 Hz 控制、每次控制推进 6 个物理步、
`force=False`、taxim/GelSight Mini、相同模型权重、预算和 v3 重放阈值。所有成功、丢失、预算耗尽和
基础设施异常都保留在分母中。墙钟频率单独记录为 `wall_physics_hz` 与 `wall_control_hz`，不能把
配置的 120/20 Hz 当作机器实际 FPS。

## 阶段与门槛

| 阶段 | 目标 | 当前交付物 | 进入下一阶段的门槛 |
|---|---|---|---|
| P0 版本和运行基线 | 锁定 Isaac 5.1 分支、配置、设备和性能测量方式 | 版本清单、GPU 快照、固定步长执行器 | 每次运行都能复现配置并记录 GPU 与墙钟速率 |
| P1 控制/观测/数据契约 | 独立 wrapper、动作类型、传感器时间、失败落盘 | `evotac/envs/`、`evotac/data/`、阶段 1 验收文档 | CPU 契约测试通过；真实 reset/step/失败结果完整落盘 |
| P2 FTP-1 基线与起点重建 | 冻结 chunk16，定位首次重放差异并验证恢复是否改善失败 | B0/B1 对照、v3 阈值、分阶段诊断、handoff、seed23 完整 paired runner | 当前 seed23 standalone strict replay 和 paired terminal audit 已通过；多 seed strict cohort 仍需稳定后才能生成正式标签 |
| P3 单恢复技能 SAC | 冻结 FTP-1 和触觉编码器，完成一次真实交互、入缓冲和一次梯度更新 | `evotac/learning/` 的单技能 SAC、collector、buffer、monitor、真实 handoff | 真实 collector/headless 链路已通；还需要真实恢复 transition、冻结 HistoryEncoder 和一次不使用特权标签的 SAC 更新 |
| P4 效果表征、技能选择与经验复用 | 冻结技能和基础策略，采集带版本的实际效果标签，训练效果预测/检索/选择 | `effect_model.py`、`skill_selector.py`、`effect_selection.yaml` | 固定技能矩阵中效果选择优于仅成功率预测；版本和缺失标签正确隔离 |
| P5 持续适应与有限扩展 | 按新失败先复用、再更新、必要时增加候选，并用独立接纳集决定保留 | `continual_adaptation.py`、`continual_adaptation.yaml` | 完成一轮新失败→更新→刷新标签→接纳/拒绝；不自动接纳 |
| P6 最小充分实验 | 用三组配对消融回答触觉、效果组织和持续适应三个问题 | 计划与评估协议已写入 `myideas/EvoTac_UniVTAC_Implementation_Plan.md` | 三组实验均有独立父场景、预算、置信区间和证据等级 |

## P2 的下一轮执行顺序

1. 先跑 seed 0、47、86 的边界 `0/1/4/8/12`，分别检查空载、抓取结束、接近结束、策略前缀和物体姿态、触觉形变以及 history mask。
2. 对同一不可变 scene 在同一 GPU/代码下重复 reset + prefix，先隔离 reset、UIPC/FEM 和触觉缓存的非确定性；marker 后处理必须使用 reset 重置的本地 RNG；不放宽 v3 容差来掩盖差异。
3. 当前 seed23 已完成 standalone strict replay 和完整 paired terminal audit；扩大前仍需先隔离 UIPC/FEM/传感器 reset 漂移。现有 4-parent/8-repeat 校准中 seed18/24 不稳定，不能把诊断容差升级成正式 cohort。
4. P3 真实 collector 已在空闲 GPU 完成 controlled-trigger 端到端运行，scripted branch 已验证 4 个恢复控制和 handoff；已有 3 条真实 transition 和 train-split warm-start。当前源码已加入 train-only trigger calibration artifact、哈希冻结和 dev/test fail-closed 检查；新的标准真实训练仍需等待独占 GPU。
5. P3 冻结技能后才生成 P4 效果标签；P4 通过版本兼容和选择验收后，才进入 P5 的新条件更新。

## 性能验证轨道

作者的 UniVTAC `isaac51` 分支支持 Isaac Sim 5.1；官方 README 给出相对 4.5 最高约 5 倍的数据采集吞吐提升，但没有发布本 insert_hole/taxim 配置的逐任务 FPS。因此本项目以本机 A/B 为准：

- **验证模式**：`--livestream 2`，相机和触觉全部打开，结果才可与现有 v3/FTP-1 数据比较。
- **性能模式**：`--performance-profile`（等价 headless、livestream=0），仍保留相机和触觉传感器，测量去掉 viewport/streaming 后的真实收益。
- **上界模式**：另建无相机或无触觉的实验，仅用于拆分渲染和 TacEx 成本，不得与策略结果混合。

headless 只移除交互视口/直播相关负担；相机传感器仍由渲染器生成，taxim/TacEx 的触觉计算也仍在。任何降低相机更新率、分辨率、传感器字段或更换后端的改动都必须单独版本化，因为会改变观测契约和策略可比性。

建议命令（GPU 空闲后执行）：

```bash
PYTHONPATH=. python -m evotac.scripts.replay_chunk16_prefix \
  --performance-profile --source-run-id phase2_followups_v3_chunk16_seed0 \
  --seed 0 --run-id profile_headless_seed0
PYTHONPATH=. python -m evotac.scripts.profile_rollout \
  evotac/datasets/phase1/profile_headless_seed0
```

性能报告必须同时保存 `gpu_snapshot_before/after_start`、总墙钟秒数、`wall_physics_hz`、
`wall_control_hz` 和传感器/后端配置。

低速的分项假设、TacEx 公开耗时、headless 语义以及 V/H/C/T/R A/B 矩阵见
[性能分析与优化方案](Performance_Analysis_and_Optimization.md)。

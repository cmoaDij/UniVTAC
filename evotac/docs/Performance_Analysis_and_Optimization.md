# UniVTAC 5.1 仿真速度分析与待尝试方案

更新时间：2026-09-20。本文只分析当前作者 `isaac51` 版本（Isaac Sim 5.1.0、Isaac Lab 2.3.0、本地 TacEx），不把 4.5 的公开 checkpoint 或旧分支运行速度当作当前实现的证据。

## 目前观测到的速度

已保存的 chunk16 轨迹汇总了 1600 个有效控制、9600 个物理步和 442.07 秒执行器墙钟时间，得到 `wall_physics_hz=21.72`、`wall_control_hz=3.62`。这两个数字只覆盖 `FixedStepExecutor.execute()` 从命令适配到物理推进、渲染和状态读取的墙钟时间；模型推理、策略前处理、HDF5 flush 和 reset 不在这两个数字中。因此它们是执行段吞吐，不能称为端到端 FPS，也不改变配置的 120 Hz 物理时钟和 20 Hz 控制时钟。

同一批数据运行期间，`nvidia-smi` 看到 10 张 RTX 5880 Ada 均接近 100% 利用率，外部 `lerobot_train` 进程在每张卡上占用约 32–34 GiB。GPU 争用是历史慢速的强证据。2026-09-20 用 `CUDA_VISIBLE_DEVICES=8` 把物理 GPU 8 映射为 TacEx 支持的逻辑 `cuda:0`，完成了同 seed 的 headless/livestream 初版 A/B：两条各 4 个控制周期、24 个物理步全部有效，执行段分别为 headless `27.246 physics Hz` / `4.541 control Hz` 与 livestream `22.523 physics Hz` / `3.754 control Hz`。它们仍是有界 profile，不能作为正式吞吐或策略效果结论。

## 负载拆解

当前 EvoTac `factory.py` 设置 `sim.dt=1/120`、`decimation=6`、`render_interval=6`，也就是每个控制动作推进六个物理步，并在控制边界更新一次 render、场景、UIPC 网格和两路触觉。任务仍启用：

- 两路 480×270 RGB 相机；
- 双 GelSight Mini 的 tactile RGB 和 marker RGB；
- taxim optical simulation、marker simulation 和本地 TacEx/UIPC 接触更新；
- `force=False` 的物理驱动控制。

因此不能把它与无相机、无触觉的普通 Isaac Lab benchmark 直接比较。Isaac Lab 文档明确区分 camera-sensor renderer 与人机交互 visualizer；headless 会减少后者，但不会自动取消 camera sensor buffer。[Isaac Lab Renderers](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/renderers.html)

TacEx 的公开论文提供了负载量级参考，而不是本机 5.1 的承诺：单环境、480×640 tactile 图像和 10×10 marker 的刚性 gelpad 实验中，height-map、GPU Taxim、CPU FOTS 约为 1.37、5.90、4.49 ms/frame；GIPC 软体计算从 1029 vertices 的 24.95 ms/frame 增到 12509 vertices 的 221.61 ms/frame，且软体配置受显存限制。[TacEx 论文](https://arxiv.org/pdf/2411.04776) 这说明 tactile optical/marker/FEM 是独立于 viewport 的成本来源。

UniVTAC 官方 README 只给出 Isaac Sim 5.1 相对 4.5 “最高约 5 倍数据采集吞吐”的版本级描述，没有给出当前 insert_hole、单环境、双触觉、taxim 配置的 FPS 表。因此本项目必须用同一机器、同一 seed 和同一观测契约做 A/B。[UniVTAC README](https://github.com/univtac/UniVTAC)

## headless 能解决什么

当前新增 `--performance-profile`，它会设置 `headless=True`、`livestream=0`，同时保留 `enable_cameras=True`。Isaac Lab 5.1 的 `AppLauncher` 在这个组合下使用 headless rendering experience、offscreen camera rendering，并关闭活动 viewport；这是适合测速的模式。验证结果仍使用 `--livestream 2`，因为改变启动体验会改变比较条件。

性能 profile 还允许显式设置 `--flush-every 16`（范围 1–128）以测量 HDF5 持久化开销；每步 flush 的冻结验证配置不被修改，manifest 会记录 profile 的 flush 值。该选项只影响落盘时机，不丢弃动作、观测或失败记录。

Isaac Sim 5.1 性能手册指出，headless Python 运行仍可能更新 viewport，应该显式禁用 viewport updates；但该选项不适用于 streaming 场景。[Isaac Sim 5.1 Performance Handbook](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html) 官方 livestream 文档也说明 WebRTC 是 headless 实例的远程交互机制，不能把直播服务当作纯仿真模式。[Livestream Clients](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)

结论是：headless 值得测试，预期收益来自 viewport/UI/streaming；如果瓶颈在 taxim、marker 或 UIPC，它不会带来同等比例的加速。关闭 camera/tactile 只能作为上界测试，不能用于 FTP-1 或恢复策略的效果比较。

## GPU 空闲后的最小 A/B 矩阵

固定 seed 0、`phase2_followups_v3_chunk16` 前缀、120/20 Hz、taxim、相机和触觉字段，依次测量：

| 组 | 启动 | 传感器 | 用途 |
|---|---|---|---|
| V | `--livestream 2` | 全部开启 | 现有验证可比基线 |
| H | `--performance-profile` | 全部开启 | headless 去除 viewport/streaming 的收益 |
| C | headless | 相机关闭、触觉保留 | 相机 renderer 成本上界；不可作策略结果 |
| T | headless | 相机保留、触觉输出关闭 | tactile/TacEx 成本上界；不可作策略结果 |
| R | headless | 相机和触觉均关闭 | 纯物理/控制上界；不可作策略结果 |

每组都记录启动前后 GPU snapshot、reset 秒数、执行器物理步/秒、完整控制段秒数、渲染调用耗时、`sim.step` 耗时、`uipc_sim.update_render_meshes` 耗时、tactile update 耗时、观测拷贝/编码耗时和 HDF5 flush 耗时。只有 V/H 使用完全相同的观测与成功判定，才可以回答 headless 是否保持效果可比。

## 如果独占 GPU 后仍慢，优先尝试的方案

先不修改冻结代码，按以下顺序做小实验：

1. **先分项 profile，不先调频率。** 用 Isaac Sim/Kit 的 Tracy 或官方性能 benchmark 记录 CPU/GPU frame time，并在 `FixedStepExecutor` 的边界计时 UIPC、render、tactile 和观测读取。若 GPU frame time 不高而单个 CPU 线程高，问题更可能在 Python/FOTS/同步；若 GPU frame time 集中在 tactile camera/Taxim，问题是传感器负载。
2. **确认触觉是否每个物理步重复计算。** 当前策略控制每六个物理步调用一次 `_update_render()`；代码中的 tactile `update_period` 仍初始化为 `1/120`，实际是否在 `dt=6/120` 时只计算一次需要运行时计数验证。若它在控制边界被重复触发多次，应优先修复更新节奏，而不是降低物理频率。
3. **尝试相机更新节奏与观测节奏一致。** Isaac Lab 相机文档建议 `update_period` 匹配需要的 observation cadence；当前策略观测为 20 Hz，若验证证明 120 Hz camera buffer 没有被策略消费，可以建立独立 20 Hz profile。这个 profile 会改变传感器时间契约，不能混入 FTP-1 正式结果。[Isaac Lab Camera Sensors](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/sensors/camera.html)
4. **保留 RGB/marker 必需输出，剥离只用于诊断的字段。** 当前 `tactile_data_types` 初始化还包含 depth/marker_motion，wrapper 观测白名单只取 RGB/marker RGB。需要 profile 证明底层额外输出是否仍计算；若是，建立只启用训练所需字段的版本化配置。
5. **比较 taxim 与 pix2pix 只作为速度/保真度实验。** 后端改变触觉图像分布，不能用于当前 FTP-1 重放匹配；必须单独报告速度和观测差异。
6. **最后才考虑低分辨率、降低物理频率或切换软体模型。** 这些会改变接触动力学、触觉图像或预算单位。它们可以作为“快速训练 profile”，但不能宣称与冻结 120/20 Hz 结果等价。

## 判断规则

- H 比 V 明显更快、C/T/R 变化很小：主要是 viewport/streaming 或相机 renderer，使用 headless 进行大规模采集，V 保留少量验证。
- H 与 V 接近、T 明显更快：瓶颈主要是触觉仿真，优先检查 taxim/FOTS/UIPC 更新频率和输出字段。
- H 与 V 接近、R 明显更快但 C/T 难区分：需要细粒度 Tracy 和 CPU/GPU 同步 profile，不能仅凭 `nvidia-smi` 判断。
- 独占 GPU 后所有组均显著快于当前 21.7 physics Hz：外部 GPU 争用是主要原因；仍需保留独占结果作为正式性能基线。

当前已有同 seed 的 headless/livestream 执行段 A/B，但每条只有 4 个控制周期；在更长 profile 完成前，不能把 27.246 physics Hz 或 22.523 physics Hz 外推为正式采集吞吐。现阶段最准确的结论仍是：低速同时可能包含 GPU 争用与触觉/渲染负载，headless 是可逆实验，但不能预先承诺它会解决 TacEx 限制。

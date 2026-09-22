# EvoTac 阶段 1 实现计划：独立控制接口、观测契约与经验采集

日期：2026-09-17。状态：现有任务数据审计、接触数据盘点及源码检查已完成；阶段 1 控制接口与 GPU 验收待实现。公开权重尚未在本地验证加载。

依据：[方法方案](EvoTac_Method_and_UniVTAC_Plan.md)第 3、9、12 节及附录 A/B；[多阶段实施方案](EvoTac_UniVTAC_Implementation_Plan.md)阶段一；当前仓库源码。

## 1. 目标与隔离原则

阶段 1 交付一个可供后续学习模块使用的单环境接口：能够每 50 ms 仿真时间接受一次控制、返回同步观测、保存失败过程，并通过前缀重放重建恢复起点。

**采用根目录 `evotac/` 独立包 + 专用 `insert_hole` 子类 + 专用脚本。按用户要求，新增代码、配置、实验数据、模型和运行结果统一放在 `evotac/` 内。阶段 1 以不修改现有程序文件为实施目标。**

固定时长入口实现于 EvoTac 扩展层，复用 `envs/_base_task.py` 和机器人管理器已有底层方法，不直接更改共享基类。通过独立入口启用，原来的采集、重放和评估继续走原有路径；总计划采用相同边界。

范围固定为 Isaac Sim 5.1、Isaac Lab 2.3、本地 TacEx、Franka、GelSight Mini、`taxim`、`num_envs=1`。阶段 1 不训练 ACT/SAC，不实现学习型监测、效果表征、技能选择或持续适应；为阶段 2 预留参考、历史和动作缓存接口。

本阶段先复用现有数据验证读取/转换契约，再通过少量新运行验证真实控制、同步及重放。编码器预训练数据已在本地，权重验证安排在阶段 2；不为阶段 1 重采通用接触数据，也不将下载权重或模型训练作为接口验收的前置条件。

隔离要求：

- 依赖方向只有 `evotac → envs/现有工具`；原模块不导入 `evotac`。
- 不修改 `envs/`、`policy/ACT/`、`encoder/`、现有 `scripts/*.py`、根目录 shell 脚本、`task_config/*.yml`、资产或 `third_party/TacEx/`。
- 不全局替换类/方法，不修改共享配置对象；每次创建独立配置实例。专用子类只由 EvoTac 工厂显式实例化。
- 不覆盖原数据、模型、种子缓存或 UIPC workspace；全部项目输出使用 `evotac/` 内带运行编号的子目录。
- 不替换依赖或运行安装脚本；复用已有仿真环境。若确实遇到底层能力缺口，先记录具体证据，再另立最小兼容补丁，不提前扩大范围。

“不影响原程序”指原代码、默认配置、入口及输出均保持隔离；子类仍依赖底层接口，不能据此承诺未来仓库升级完全无适配成本。

### 1.1 已有资源：可复用什么，尚未验证什么

| 资源 | 本次已核查的状态 | 阶段 1 如何处理 |
|---|---|---|
| `data/isaac51/` | 8 个任务、800 条成功记录，约 242.11 GiB；全量结构/小型状态检查和 432 张图片抽检通过 | 只读引用，复用已完成的审计，不重新下载或全量复制 |
| `data/isaac51/insert_hole/` | 100 条、19,151 帧，约 24.18 GiB；没有独立动作命令、完整前缀及参考图 | 作为 ACT 候选示范，验证字段映射和少量轨迹的控制兼容性；不放入恢复 RL buffer |
| `data/contact/` | 已有 14 类形状、638 条 HDF5，约 124.36 GiB；仅盘点数量/体积 | 登记为阶段 2/3 编码器适配候选；尚未完成内容质量、采集版本和加载验证 |
| 官方 `checkpoints/encoder.pth` | 约 165 MB，公开文件存在；检查的本地目录未发现，未验证加载 | 记录下载来源及待验证项，后续写入 `evotac/checkpoints/`；不为此阻塞控制接口 |
| 官方 insert_hole 策略权重 | 有 `vision_only`、`univtac` 两种；官方策略说明对应 Isaac Sim 4.5 | 仅作阶段 2 可选初始化，不能认定是 5.1 可直接部署策略 |

任务数据的检查方法与证据见 [已有数据分析](../evotac/docs/Isaac51_Existing_Data_Assessment.md)。公开发布状态依据 2026-09-17 查阅的 [官方说明](https://github.com/univtac/UniVTAC/tree/isaac51)、[权重目录](https://huggingface.co/datasets/byml/UniVTAC/tree/main/checkpoints)及 [接触数据目录](https://huggingface.co/datasets/byml/UniVTAC/tree/main/contact)。5.1 任务数据发布不等于 5.1 策略权重发布；contact 和独立 encoder 权重的采集/训练版本也需单独核验。

### 1.2 必须补采的数据及阶段边界

| 数据 | 已有数据的缺口 | 本阶段要求 |
|---|---|---|
| 空载/稳定抓取参考 | 旧插入首帧已在抓取和接近之后 | 在正确阶段记录原图、采样时刻、参考类型和有效性 |
| 完整抓取—接近前缀 | 旧 HDF5 未保存 reset 内完整动作链 | 保存命令、保持物理步数、参考/渲染事件及重放误差 |
| 同步控制 transition | 旧数据是状态序列，缺命令接受状态与逐模态时间 | 记录观测、原始/限幅/接受命令、实测变化和时间/有效性 |
| 失败、超时和异常 | 已保存的 100 条 insert_hole 全为成功 | 先采少量可控样例，确认全结果落盘及终止语义；不追求大规模训练集 |
| ACT 自主失败和局部恢复示范 | 当前基础 ACT 尚未在新接口中冻结 | 阶段 2/3 再采，接口验收可先使用脚本动作 |
| SAC 恢复交互、冻结技能效果标签、新条件数据 | 旧数据没有这些内容 | 阶段 3–5 边训练边采，不列为阶段 1 完成条件 |

成功示范优先复用，通用接触数据不重复采集；只有旧示范在新控制下不兼容或覆盖不足时，再补采匹配的成功示范。

## 2. 已核对的源码与实现影响

| 当前位置 | 当前行为 | 阶段 1 的处理 |
|---|---|---|
| `envs/_base_task.py::take_action/_take_action` | `qpos` 分支设置目标后调用 `_step`；`ee/delta_ee` 分支调用 `move`，执行规划轨迹 | ACT 保留关节目标语义；恢复另写局部 IK，禁止经 `delta_ee → move` 作为 RL 单步 |
| `envs/_base_task.py::_step/_configured_decimation` | 已有 `_active_decimation`、`_physics_step_count`；一次 `_step` 可推进多个物理步 | 正常控制复用该能力，配置 `decimation=6`；不重复乘以六 |
| `envs/_base_task.py::_update_render` | 依次更新 UIPC 渲染网格、render、scene、actor、tactile | 保留该同步顺序，以物理步编号跟踪采样；避免 wrapper 再无条件渲染一次 |
| `envs/robot/robot.py::set_arm/set_gripper` | 默认 `force=True` 会直接调用 PhysX 写关节位置 | EvoTac 控制和接触前缀显式使用 `force=False`；保留 reset 的初始化写状态 |
| `envs/robot/robot.py::get_ee_pose` | 返回机器人基座系位姿，虽然内部先读世界系状态 | 世界系增量与基座系位姿不能直接相加；适配器负责坐标转换 |
| `envs/robot/robot_cfg.py::_resolve_arm_pd` | collect 与 eval 默认 PD 增益不同 | EvoTac 数据采集、控制、重放显式固定同一组 PD 参数并写入版本清单 |
| `envs/_base_task.py::reset` | 先取得 `super().reset()` 返回值，再执行稳定、`pre_move()` 等操作，最后返回早先结果 | wrapper 在完整初始化结束后重新取得观测；不能直接转发原返回值 |
| `envs/insert_hole.py::pre_move` | reset 内已经完成抓取及接近，含随机参数、规划和自适应夹持 | 前缀记录必须覆盖 reset 内部；空载参考必须在抓取前挂接 |
| `envs/_base_task.py::_get_observations` | 可返回 `actor` 真值、`atom` 脚本标记等字段 | 以白名单重新构造部署观测，训练真值走单独通道 |
| `scripts/collect_data.py::run` | 仅对成功回合调用 `save_to_hdf5`，失败会清理缓存 | 独立记录器保存所有结果，不依赖旧成功示范管线 |
| `scripts/eval_policy.py::run` | 存在专家筛选与失败种子缓存 | 新入口按显式场景清单运行，保留所有尝试及其分母 |
| `envs/utils/env_parser.py` | 原工厂按 `envs.{task_name}` 动态导入任务，频率配置也有整除校验 | 独立工厂加载 EvoTac 子类；不修改原注册逻辑，不把 20 Hz 简单叠加到旧 60 Hz 配置 |
| `policy/ACT/deploy_policy.py::Policy.eval` | 自行调用 `task.take_action`；编码器还会读取触觉 | 后续只复用模型推理能力，EvoTac 执行器拥有步进权；阶段 1 不改这个适配器 |
| `envs/utils/data.py::batch_gather_hdf5` | 先配对 `q[i] → q[i+1]` 再按 stride 取样，未保存原执行命令 | EvoTac 转换器明确预测时距；不能用 `downsample_factor=3` 代替 60→20 Hz 标签对齐 |
| `encoder/train.py`、`encoder/dataloader.py` | 旧 `contact-gs` 路径；解码调用参数与当前函数不匹配；按单帧/单指随机划分 | 登记为阶段 2 适配任务；在 `evotac/` 新加载器中修正，不修改原脚本 |
| `policy/ACT/detr/models/backbone.py::TactileBackbone` | 只有权重路径存在时才加载，否则继续使用初始化网络 | 阶段 2 加载器要求缺失/不匹配时显式失败，不能把“能推理”当作“已加载预训练权重” |

`first_frame` 的 UIPC 保存/重放仅是现有 reset 机制的一部分，不构成任意恢复点的完整机器人、触觉、参考与策略状态快照。

## 3. 建议代码放置位置

目录结构包含已存在的检查脚本和局部忽略规则，其余是拟实现位置；不为文档任务创建空实现。学习模型和训练入口放在阶段 2 之后，不提前扩展阶段 1。

```text
evotac/
    .gitignore                        # 已有，仅忽略本目录的生成文件
    __init__.py                         # 保持轻量，不在 import 时启动仿真
    configs/
        phase1_insert_hole.yaml         # 独立配置，包含控制、传感器、预算与输出
        legacy_insert_hole.yaml         # 来源、关节/相机映射、时间配对及划分
    envs/
        __init__.py
        factory.py                     # AppLauncher 启动后构造专用任务/配置
        univtac_rl_wrapper.py          # reset / step / replay_start / close
        fixed_step_executor.py        # 固定时长关节目标执行、结果与计步
        action_adapter.py             # 两种动作契约、局部 IK、限位与坐标转换
        observation_contract.py       # 部署观测白名单、时间戳、有效性
        state_replay.py                # 前缀重放、状态重建、误差报告
        tasks/
            __init__.py
            insert_hole.py             # EvoTacInsertHoleTask、专用 TaskCfg
    data/
        __init__.py
        schemas.py                    # 动作、transition、scene_ref、版本字段
        rollout_logger.py             # 全结果记录与异常时落盘
        legacy_demonstrations.py       # 只读旧数据与 ACT 候选时间配对
    scripts/
        __init__.py
        audit_legacy_data.py           # 已有，已完成 800 条任务轨迹检查
        smoke_phase1.py                # 单次短集成验收，可复用于阶段 2
        collect_rollouts.py            # 显式场景清单采集，不筛掉失败
    tests/
        test_control_contract.py       # 动作/坐标、计步、异常边界
        test_observation_contract.py   # 真值隔离与时间对齐
        test_rollout_and_replay.py     # 保存、父场景、重放误差与有效性
        test_legacy_demonstrations.py  # 预测时距、来源种子、父场景隔离
    docs/
        Isaac51_Existing_Data_Assessment.md  # 已有数据分析
    datasets/                         # 生成的示范、rollout、前缀和场景清单
    checkpoints/                      # 后续阶段的模型权重
    runs/                             # 指标、日志、视频、仿真 workspace
    .cache/                           # 本项目可配置的临时缓存
```

| 位置 | 放置理由 |
|---|---|
| `evotac/envs/` | 将对私有仿真接口的依赖限制在环境适配层；学习模块只依赖 wrapper 契约 |
| `evotac/envs/tasks/insert_hole.py` | 继承原任务，复用资产、成功条件；仅扩展前缀、参考挂接、受控条件与记录 |
| `evotac/data/` | EvoTac 需要失败轨迹、动作接受状态、版本及分支信息，不能直接套用成功示范格式 |
| `evotac/configs/` | 与原 YAML 分离，避免改变原任务默认频率、PD 增益或观测字段 |
| `evotac/scripts/` | 支持 `python -m evotac.scripts.…`，不需要修改原 launcher，也不依赖全局任务注册 |
| `evotac/policy/` | 留到阶段 2 放基础策略适配器与部署代码；复用原模型，执行仍由 EvoTac 独立入口负责，不在根 `policy/` 新建文件 |
| `evotac/models/tactile_encoder.py`、`evotac/data/contact_dataset.py` | 阶段 2 的权重加载/前向适配及接触数据加载；修复兼容性但不改原 `encoder/` |
| `evotac/datasets/` | 保存实际数据，与存放 Python 数据处理模块的 `evotac/data/` 区分 |
| `evotac/checkpoints/`、`evotac/runs/` | 分别保存模型权重和运行产物，均不提交 Git |

输出约定：

```text
evotac/datasets/phase1/<run_id>/
    manifest.json                     # 版本、完整配置、命令、种子与资产标识
    scenes.jsonl                      # 父场景/前缀引用；所有尝试均有记录
    episodes/<episode_id>.h5           # 图像、状态、动作、结果；无损存储
evotac/runs/phase1/<run_id>/
    simulator/                        # 独立 UIPC workspace 及旧基类辅助输出
    logs/
    videos/
    checks.json                       # 验收结果与未通过原因
    reconstruction.jsonl              # 起点重建误差、容差版本与有效性
    legacy_compatibility.json          # 旧示范在新控制下的检查结论及不足
evotac/datasets/legacy_insert_hole/<conversion_version>/
    manifest.json                     # 源路径/source_seed、划分、频率证据与转换语义
    episodes/                         # 按需生成的候选 ACT 数据，不是恢复 transition
evotac/checkpoints/univtac_release/    # 后续下载的公开权重与来源记录
```

这些新路径不受原 `/data/*`、`/eval_result` 忽略规则覆盖。已有 `evotac/.gitignore` 按以下规则处理，不修改根 `.gitignore`：

```gitignore
/datasets/
/checkpoints/
/runs/
/.cache/
/.pytest_cache/
**/__pycache__/
```

源码 `evotac/data/` 不在忽略列表中。用于复现实验的小型手写场景配置放 `evotac/configs/`；生成的大型场景清单放 `evotac/datasets/`。新增说明文档可放 `evotac/docs/`，本文作为已有方案文档继续保留在 `myideas/`。

所有输出路径以 `evotac/` 的绝对路径为基准解析，不依赖进程当前工作目录。工厂必须显式设置 `cfg.save_dir`，从而将基类辅助输出和 UIPC workspace 定位到 `evotac/runs/…/simulator/`；不能使用 `save_dir="auto"`。Logger、视频、模型保存和可配置缓存也统一传入专用路径。

集中存放不意味着复制 UniVTAC：仍从原 `envs/`、`policy/ACT/` 和本地 TacEx 导入能力、读取现有资产。这是仓库内独立扩展目录，不是拷走即可运行的独立项目。Isaac/驱动等依赖自身的系统缓存不属于 EvoTac 实验数据；可配置的项目缓存按上述路径管理。

## 4. 对外接口与数据契约

### 4.1 环境 API

```python
reset(seed, condition) -> (deploy_obs, info)
step(action) -> (next_deploy_obs, reward, terminated, truncated, info)
replay_start(scene_ref) -> (deploy_obs, reconstruction_report)
close() -> None
```

采用五元组 step 语义即可，阶段 1 无须引入完整 RL 框架或直接调用继承链中未实现的通用 RL `step()`。`info` 内明确区分 `execution_info`、`training_info`、`scene_metadata`。传给模型的函数只接收 `deploy_obs`，不能接收完整 `info` 或任务实例。

`condition` 在环境侧生成受控条件，默认先支持名义条件和一类横向接近偏差；角度、抓取偏移在后续扩展。扰动通过物理动作执行，不在接触过程中瞬移物体。隐藏条件值和条件类别只进入场景记录。

### 4.2 动作契约

统一使用带类型的动作对象，明确区分两种输入，禁止仅根据向量长度猜测语义：

| 类型 | 内容 | 执行方式 |
|---|---|---|
| `joint_target` | 7 个臂关节目标 + 1 个夹爪标量，共 8 维 | ACT/脚本关节目标，经限位后保持一个控制周期 |
| `recovery_delta` | `[dx, dy, dz, rx, ry, rz, dg]`，7 维，归一化到 `[-1, 1]` | 缩放、坐标转换、阻尼最小二乘局部 IK，再变为关节目标 |

约定：平移为世界系增量，单位 m；旋转采用世界系旋转向量，缩放后单位 rad，用指数映射组合旋转；夹爪为单指关节位置增量，单位 m，同时映射到两指。旋转定义写入动作版本，不能沿用旧 `delta_ee` 的 Euler 实现后又将其标为旋转向量。

第一版控制点选机器人 `panda_hand`，与对应 Jacobian 保持一致。若以后改到夹爪中心，必须增加工具偏移的 Jacobian 修正并升级动作版本。不能只平移末端位置、不改变 Jacobian。

局部 IK 直接从当前机器人状态和 Jacobian 计算关节增量，不调用 Curobo 规划整段恢复轨迹。实现时核对 body 索引、固定基座偏移、臂关节列顺序及世界/基座坐标转换；可参考本地 `third_party/TacEx/source/tacex_uipc/examples/single_uipc_attachment.py`，但不修改该文件。

尺度、IK 阻尼、关节范围、每步关节变化、夹爪范围和奇异/残差阈值全部配置化。具体幅度由阶段 1 小幅控制验证后冻结，不把未经验证的数值称为可用默认值。拒绝 NaN、Inf 和不合法维度；IK 求解失败或目标无法合法执行时显式报告失败。

每步记录四层信息：

1. `proposed_action`：策略原始输出；恢复 critic 使用这一契约下的归一化动作。
2. `commanded_action`：限幅、缩放和 IK 后的关节目标，以及中间限幅量。
3. `applied_action`：控制器已接受的目标及位置/速度控制设置；不是测得的运动量。接口无法确认时写 `accepted=false/unknown`，不能伪造成功。
4. `measured_state_change`：前后关节、末端变化，作为执行响应和下一观测。

actor、critic 与 replay buffer 始终使用同一动作定义；适配器是环境转换的一部分。8 维基础动作和 7 维恢复动作分别标注，后续恢复 actor 的训练不能混入基础动作样本。

### 4.3 部署观测与训练信息

| `deploy_obs` 允许项 | 约束 |
|---|---|
| 头部/腕部相机 RGB、双指触觉 RGB/marker RGB | 明确相机/传感器身份、图像格式；先保留原始数据，不在阶段 1 训练编码器 |
| 关节位置/速度、两指位置、末端位姿 | 按名称映射关节，注明单位与坐标系；不假设 `[:8]` 就是所需 8 维 |
| 上一周期已接受的控制命令及有效性 | 与当前状态变化分开，避免把未执行命令当成执行历史 |
| 模态时间戳、frame id、age、validity | 由真实更新记录生成，不能在每次读取时给旧图像贴新时间 |
| 任务目标、已知控制模式、剩余预算 | 仅包含部署时确知内容；无隐藏扰动标签或仿真精确目标位姿 |
| 触觉参考与历史引用/掩码 | 尚未获得参考时保持 invalid；历史不足时保留有效性掩码 |

`training_info` 单独保存物体/孔位真值、隐藏扰动、任务成功、真值进度、重建物体误差和奖励依据。`actor`、`atom.id/tag`、任务 metadata 不得整体并入部署观测。仿真形变/深度诊断先放入训练侧，未经部署可获得性确认不得自动作为触觉输入。

原任务 `check_success()` 和 `check_early_stop()` 可用于环境结果判定；它们读取真值，因此不能作为后续运行时监测器或交还规则的输入。阶段 1 不引入特权 critic。

### 4.4 旧数据转换契约

manifest 使用 `data_kind` 区分 `legacy_demonstration`、`contact_pretrain`、`control_rollout` 和后续 `recovery_outcome`。对旧数据保存 `source_path / source_episode_id / source_seed / source_version / source_rate_evidence / split / conversion_version / action_semantics`；未知项保留 unknown，不复制新 wrapper 的版本冒充历史采集版本。

1. 文件编号只作样本 ID。insert_hole 有 60 条被重编号，原种子范围为 0–761，例如 `99.hdf5` 的 `source_seed=761`；重建时读取 metadata 中的原种子。旧种子只能给出场景候选，不能据此声称恢复了触觉状态。
2. 保存步号间隔均为 2，与 120 Hz 物理/60 Hz 保存配置一致，但文件没有历史频率快照。确认源频率后才生成按秒标注的训练版本；未确认时只做探索性转换，标记不可用于正式训练。
3. 若源保存频率确认为 60 Hz，取 `i=0,3,6,…`，用 `(o[i], q[i]) → q[i+3]` 构成 20 Hz 候选样本，要求 `i+3<N`。当前 100 条可形成 6,315 对。该目标明确标为 `future_measured_qpos`，不写为 `applied_action`，不进入 SAC replay buffer。
4. 按关节名称确认原 9 维顺序，映射为 7 个臂关节和夹爪标量；同步映射相机名称，如原 `head` 与 ACT 的 `cam_high`。原统计量、chunk 长度、颜色/尺寸预处理都需重新核对，不直接套用 4.5 配置。
5. 先按父场景划分再转换，例如 80/10/10；同一来源轨迹的所有抽样相位/裁剪/重放保持同一划分，归一化只在训练划分计算。可按 `rotate=0/π` 分层，但隐藏标签不进入部署输入。
6. 缺少接受命令、参考或传感器时间的旧样本保留缺失标记，不填可靠零值或伪造时间；JPEG 原图不可冒充无损触觉记录，也不用于直接标定新系统的重建容差。

阶段 1 只需小样本转换与关节目标执行检查，结果记录为可复用、仅适合暖启动、不兼容或证据不足。旧数据不兼容不阻止正确的新 wrapper 通过阶段 1，但必须明确阶段 2 需要补采什么；ACT 闭环效果留阶段 2 验证。

### 4.5 给阶段 2 的触觉编码器接口

按现有源码，优先对接单张三通道 `rgb_marker → ResNet18 → 512 维特征`，双指共享权重并保留身份。原预训练利用 RGB/深度/标记点/相对位姿等解码监督；部署仅运行允许图像到特征的前向，不读取物体真值。实际公开权重的结构和监督配置尚未加载确认。

阶段 1 保存原图、双参考及有效性即可，不预先将有符号差分拼为九通道。阶段 2 保持预训练前端的三通道输入，将差分派生低维线索送入历史网络；新增差分图像分支若有需要单独训练和版本化。完整原图与差分来源仍保留，便于后续研究。

阶段 2 必须核对 checkpoint 的哈希/revision、参数覆盖、输入尺寸/颜色/范围、BatchNorm 和单指输出形状，并比较旧图像与新 wrapper 图像上的前向行为。通过后显式冻结参数并保持 `eval()`，包括 BatchNorm 缓冲区。权重缺失/关键参数不匹配应报错；不使用无检查的 `strict=False` 或静默随机初始化。

只有适配不足才利用已有 contact 训练划分微调；修正路径和解码 API、复核 24–34 mm 深度归一化及图像尺寸，按来源轨迹同时隔离双指和全部帧。图像前端最迟在阶段 3 正式技能训练前冻结，效果分支的后续训练不得改变它。公开权重未验证时不标记为“编码器已可用”；contact 未审计时不标记为“预训练数据已就绪”。若直接复用已验证权重，无须先完成全部 contact 数据审计。

## 5. 分步实现与验收

### P1-A：配置、契约和独立任务工厂

新增 `schemas.py`、专用 YAML、`factory.py` 和子类骨架。约定版本号、动作类型、观测字段、终止原因、父场景/分支标识、预算单位和输出目录。沿用已有任务审计报告，登记 contact 的待验证状态；实现旧示范读取、来源清单和少量时间配对检查，不在此训练模型。

启动顺序必须为：读取纯配置 → `AppLauncher`（开启 camera）→ 导入 Isaac/TacEx/任务模块 → 构造独立任务。`evotac/__init__.py` 和纯数据契约不能提前导入仿真模块。

专用配置至少含：

| 配置组 | 必要内容 |
|---|---|
| simulation | `physical_hz=120`、`control_hz=20`、`num_envs=1`、`taxim`、相机启用 |
| controller | 固定 PD 参数、`force=false`、动作尺度、关节限制、IK 参数 |
| observation | 白名单、模态最大帧龄、参考有效性、历史长度预留 8 |
| budgets | 任务/恢复物理步数、重放稳定步数、异常停止规则 |
| replay | 前缀协议版本、误差统计方法、容差文件/版本 |
| logging | 输出根目录、运行 ID、flush 周期、图像保存方式 |
| legacy_data | 只读源目录、source_seed 字段、源频率及证据、关节/相机映射、父场景划分、转换语义 |
| resources | contact 路径、候选权重 URI/本地路径与验证状态；允许权重尚未下载 |

初始只构造一个环境，不增加并行仿真或 Gym 向量封装。复用现有配置工具时仅调用明确适用的纯函数；不让原任务工厂自动创建 `envs.insert_hole.Task` 后再替换类。

**验收：** 配置解析和数据结构能在不启动 Isaac 的情况下使用；原配置无变更；非法频率、路径冲突和动作契约会给出明确错误。用合成序列确认 20 Hz 标签领先三个源帧而不是一个，检查 `source_seed` 映射和父场景划分，无法确认来源的字段不被补造。

### P1-B：固定控制周期与动作适配器

实现 `fixed_step_executor.py` 和 `action_adapter.py`。正常一步的流程：

```text
检查 episode 未结束、预算充足、动作合法
→ 读取当前本体状态，完成限幅/局部 IK
→ 设置臂与夹爪目标，force=False
→ 明确设置速度目标，避免继承前缀残留速度命令
→ 在 _configured_decimation() 作用域内调用一次 _step(is_save=False)
→ 按已有同步路径取得传感器/机器人状态
→ 检查物理步差、成功/失败、预算并返回
```

专用任务采用明确的 eval 控制配置：`cfg.decimation=6`，旧数据/视频自动保存关闭，PD 增益显式固定。已有 eval `_step` 会渲染同步；执行器只在尚未同步到目标物理步时补同步，不能再无条件调用 `_update_render()`。

底层 UIPC 已通过物理 callback 随 `sim.step()` 推进，不额外手动调用 `uipc_sim.step()`。抓取初始化仍按原 reset 的单物理步节奏运行；20 Hz 约束针对策略控制阶段，不能把初始化轨迹每个点也保持六步。

为阶段 2 的 ACT 数据保留统一时间轴：新示范记录 20 Hz 决策边界的观测与随后保持的关节目标，不能把执行后的实测关节位置当原命令。旧示范按 4.4 转换为明确标注的代理目标，记录目标时距并验证回放；旧文件缺失的原命令持续时间不能恢复或伪造。先复用旧示范，只有不兼容或覆盖不足时再通过新接口补采。

关键边界：

- 正常执行 `Δphysics_steps=6`，`Δsim_time=6/120=0.05 s`；`step_count` 和墙钟耗时都不作为时间轴。
- `plan_success=False` 时 `_step` 会直接返回。必须检查步数实际增量，不能记录一个虚假的完整 transition。
- 一次仅执行当前目标，不预执行未来动作块。中断粒度先定义为控制周期边界，最多等待当前 50 ms **仿真时间**；墙钟等待时间另报，不声称实现实时急停。
- IK 在推进前失败时记 0 物理步和执行拒绝事件；仿真途中异常按实际已推进步数记录。不补空步凑满六步。
- 任务/恢复预算默认设为六的整数倍。恢复预算为任务执行步数的子集，不重复计入总交互量；边界不足一个完整周期时停止，不越预算执行。
- STOP 取消后续目标，保持合法关节/夹爪目标或结束回合，不自动松夹爪。

**验收：** 零增量、小幅单轴平移、旋转和夹爪动作均能按契约执行；用基座旋转的合成案例检查坐标转换，用 GPU 运行检查实际方向和物理步数。后续 SAC 只接入这个入口。

在 P1-C 完成 reset/同步后，使用少量旧训练/开发示范的代理目标检查新控制器跟踪与环境成功判定；记录 `source_seed`、初始状态差异、限幅/跟踪误差及结果。不为追求旧轨迹成功而改回 `force=True`，不把轨迹回放成功等同于 ACT 闭环成功。

### P1-C：同步观测、reset 与任务扩展

实现 `observation_contract.py`、wrapper 的 `reset/step`，完善专用任务子类。

1. reset 前设置本回合 condition、独立输出上下文和 RNG 协议；清理上回合目标、历史、参考、IK 状态及预算。
2. 继承原任务资产与成功定义；将原 `pre_move` 的抓取/提起/接近顺序在专用子类中局部拆为可记录阶段。只复制必要的任务流程，标注来源和上游版本，不复制整个 BaseTask。
3. 专用前缀中通过窄方法覆写确保 `move` 向低层传递 `force=False`，避免原默认 `force=True` 造成接触中位置强写；reset 的初始关节/物体放置保留原语义。
4. 抓取前留出空载参考采样点；稳定持握且未接近孔壁时留出抓取参考采样点。阶段 1 负责原图、来源时间及有效性，阶段 2 实现差分、特征和稳定性逻辑。不能把最后接近位置的图像冒充抓取参考。
5. 完整 reset、前缀和固定稳定过程结束后重新采集 `deploy_obs`，不用原 reset 的早期返回观测。
6. 用同步边界处的物理步 ID 和传感器实际刷新记录时间，令 `sensor_time <= observation_time`。低频模态可使用旧帧，但必须保留原采样时间和帧龄；过期即 invalid。
7. 同物理状态上的多次 render 不算新物理时刻，不用 `_update_render` 的最小 `dt` 推算时间前进。传感器缺帧与无接触分别标注。

**验收：** 记录 reset 后首帧与抓取阶段；检查字段白名单、时间非前视、旧帧有效性；改变 `training_info` 的物体真值字段不能改变 actor 输入。固定 `force=False` 后前缀是否仍能抓稳，需要 GPU 实测，失败不通过放宽语义掩盖。

### P1-D：独立 rollout 记录与终止语义

实现 `rollout_logger.py`，在 reset 前就创建 episode 记录。每次尝试具有 `parent_scene_id / episode_id / branch_id / seed / split / version`，保存 `obs_t → action_t → obs_t+1`，不能将动作后的图像写为动作前输入。

HDF5 按 `/observations`、`/actions`、`/transitions`、`/training_info`、`/references` 分组；场景与版本放 manifest。图像先用无损保存，避免重建误差受 JPEG 编码影响。异步保存前复制/固化张量，防止引用仿真内部可变缓冲区。

| 结束原因 | 记录方式 | 后续训练解释 |
|---|---|---|
| 任务成功、物体丢失等真实终止 | `terminated=true`，明确 reason | 最终结果有效，终止不 bootstrap |
| 任务/恢复预算耗尽 | `truncated=true`、`bootstrap_allowed=false` | 有限预算任务的真实失败，不能按普通时间截断继续 bootstrap |
| 控制指令合法但 IK/执行失败 | 明确执行失败 reason 与实际物理步数 | 控制失败可作为有效失败；未执行动作不可伪造成已执行 transition |
| 传感器/渲染/仿真基础设施异常 | `valid_trial=false`，异常类型与最后有效帧 | 保留尝试和成本，不当作技能失败标签 |
| 用户停止或进程可捕获中断 | interrupted 状态及不完整标记 | 不假定任务失败标签有效 |
| 文件分片/存储截断 | 仅标记存储边界，不标成环境终止 | 保留 next observation 和非终端语义 |

阶段 1 reward 仅实现可审计的稀疏任务结果和组件字段；例如终止成功奖励 1、其余 0。不在此时提前决定阶段 3 的恢复 shaping、交还归因或 critic 目标。

采用逐步追加、定期 flush、`try/finally` 关闭以及完成标记；未完成文件仍可定位和检查。进程强制终止最多保证已 flush 的前缀可用，不承诺保住所有内存数据。不得调用旧清理流程删除 EvoTac episode 文件。

**验收：** 成功、真实失败、超时、零步执行拒绝、reset 异常均有记录；重开文件后动作与前后观测数量、索引、状态一致。

### P1-E：匹配恢复起点重放

实现 `state_replay.py` 和 `replay_start(scene_ref)`，采用“相同种子 + 保存的执行前缀 + 固定稳定过程”，不以刚体位姿写回替代接触历史。

`scene_ref` 必须关联：父场景与数据划分、环境/资产/PD/适配器版本、种子与条件协议、reset 起点、前缀命令及持续物理步数、传感器更新节奏、参考采样事件、末段观测/动作历史和剩余预算。同一父场景的所有重放、分支及扰动副本必须属于同一数据划分。

实施细节：

- 专用任务在 reset 的抓取与接近期间记录实际接受的位置/速度目标、每段保持步数和 render/参考事件；仅记录高层 `move` 目标不能保证重放相同前缀。
- 重放路径在完成标准初始化后，由专用 `pre_move` 分支消费已记录前缀，避免先自动执行一遍原前缀再重放第二遍。标准初始化也要校验版本与起始误差。
- 以物理步驱动回放；恢复点前至少重建 8 个有效采样点及其动作上下文，历史不足时给出掩码。
- 参考图应在本次重放的同一阶段重新采样，验证其一致性；原参考保存为比较基准。不能仅复制旧参考来掩盖本次接触不同。
- 稳定过程使用固定命令和固定步数，写入协议；若首次运行与后续 reset 的初始化步数不同，分别记录，不能隐去额外成本。
- 阶段 1 预留 `reset/export/import/rebuild` 式策略状态接口并用测试替身检验。阶段 2 接入真实 ACT 后，前缀缓存按协议重建，进入恢复时清空基础动作块；交还时清空并重新推理，绝不恢复待执行旧动作。
- 每次重放输出关节/末端误差、双指触觉误差、参考误差、时间/历史匹配情况、离线物体误差与 `valid_match`。

容差先由独立开发父场景的重复前缀重放统计确定，再冻结版本。校准前报告 `uncalibrated`，不能标为已通过匹配。超容差分支不进入有效效果标签，但保留记录及交互成本；所有方法使用同一容差与重试协议。

预算分两层：父场景定义任务执行起点及任务剩余预算；每个物理步还进入全局交互账本。初始化、抓取/接近前缀、恢复、后续基础执行按互斥阶段计费，重复重放也计费。reset 内计数器归零前先结转已用成本，不能通过多次 reset 漏记交互。

**验收：** 同一个父场景能重放并产生完整误差报告；故意改变前缀或版本时会被拒绝/标为不兼容；超容差不转成技能失败。匹配试验不宣称严格相同隐藏状态的反事实试验。

### P1-F：短集成验收与阶段 2 交接

在同一个 smoke 程序中串起：reset → 短关节目标前缀 → 小幅恢复动作 → 保存 → reset/replay_start → 误差报告。失败分支通过可控预算耗尽或执行拒绝覆盖，不依赖训练模型制造失败。

验收清单：

| 必须成立的结论 | 证据 |
|---|---|
| 正常一步是 6 个物理步 | 开始/结束 physics ID、仿真时间差；另报墙钟吞吐 |
| 可在周期边界更换控制源 | 脚本动作队列取消后不再消费旧命令；真实 ACT 缓存留阶段 2 验证 |
| 控制经物理驱动 | `force=False` 的命令记录、目标与测量值分别保存 |
| 观测时间正确且无真值输入 | 模态时间/帧龄检查、部署字段白名单、真值隔离测试 |
| 失败不会丢失 | 回合文件、终止原因、实际已执行步数、异常有效性 |
| 起点可检验地重建 | 触觉/本体/离线物体误差、容差版本、valid_match |
| 旧数据复用结论明确 | 来源清单、时间配对测试、少量新控制兼容性报告；允许明确结论为不兼容/证据不足 |
| 原流程仍可使用 | 源码与默认配置差异检查；原 smoke/单种子采集结果 |

已执行的数据检查可按需复现；无数据变化时无需再次全量运行：

```bash
/data/ZED/conda/envs/UniVTAC/bin/python evotac/scripts/audit_legacy_data.py
```

该脚本仅覆盖 `data/isaac51`，不能将其通过解释为 contact 数据或编码器权重也已通过。建议实现后运行的命令（以下测试及新增模块入口尚不存在）：

```bash
python -m pytest evotac/tests -q -o cache_dir=evotac/.pytest_cache
python -m evotac.scripts.smoke_phase1 --config evotac/configs/phase1_insert_hole.yaml --seed 0 --headless
python -m evotac.scripts.collect_rollouts --config evotac/configs/phase1_insert_hole.yaml --scene-manifest <显式场景清单> --headless
```

原流程最小回归使用仓库已有命令：

```bash
python scripts/smoke_isaac51.py --backend taxim --headless --output-dir evotac/runs/phase1_legacy_smoke
python scripts/collect_data.py insert_hole demo --start_seed 0 --max_seed 0 --gpu 0 --config-overrides collect_settings.episode_num=1 collect_settings.save_root_dir=evotac/runs/phase1_legacy_regression
```

原采集的输出根目录通过 CLI 隔离，避免覆盖已有 demo。单种子失败需结合异常、规划和成功检查判断，不应仅看脚本是否返回 0；保存命令、种子、后端、版本及实际结果。未修改触觉后端时无需额外展开 `pix2pix` 实验；若最终涉及后端修改，则按仓库规范补测两种后端。

CPU 测试聚焦有风险的契约和异常处理，GPU 集成验证真实仿真。阶段 2 复用同一 smoke 入口，补齐真实 ACT 的“触发—暂停—脚本恢复—交还—重新规划”；学习型恢复在阶段 3 接入。阶段 1 的通过不依赖阶段 2 尚未训练的模型。

## 6. 依赖顺序、交付边界与未决项

执行顺序为 **P1-A（含旧数据契约）→ P1-B → P1-C → P1-D → P1-E → P1-F**。P1-B/C 完成后补做少量旧示范兼容性运行，结果随 P1-F 交接。先把单环境链路跑通，再增加采集规模；不新增一个必须先完成编码器预训练的阶段。

阶段 1 完成时应交付：

1. 独立 `evotac/` 包及冻结后的阶段 1 YAML。
2. 可运行的 `reset/step/replay_start/close`，统一动作和部署观测契约。
3. 包含真实失败与异常状态的示例 rollout、版本清单与物理步成本。
4. 重建误差报告及开发集容差校准结果。
5. GPU 短集成与原流程最小回归记录。
6. 已有示范/contact/公开权重的资源清单、少量旧示范转换及兼容性结论；明确未完成的编码器验证和需要补采的成功示范条件。

仍需实施时实测的项目：物理驱动前缀在固定 PD 下能否稳定抓持；Jacobian 与控制点/坐标的正确映射；相机和触觉的实际刷新延迟；前缀重放的可重复精度。若任何一项未达到契约，阶段 1 应明确标为未完成，不能用网络训练代替接口验证。

当前已存在 `evotac/.gitignore`、任务数据只读检查脚本、机器可读审计结果和分析文档；contact 已盘点，但未完成内容审计。控制/环境、旧数据转换、编码器加载及训练模块仍未实现，公开权重未完成本地验证，未执行 GPU 仿真。以上已完成的盘点无需重复，后续从 P1-A 的契约与工厂实现继续。

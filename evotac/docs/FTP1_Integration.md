# FTP-1 / Pi0.5 在 UniVTAC 中运行

本实现采用用户允许的 **FTP-1 路径**：运行官方基于 Pi0.5 的视觉触觉模型及其 `UniVTAC_insert_hole` 微调权重。它不是另一个已经验证的纯 Pi0.5 基线，也不使用随机模型替代真实权重。

## 来源与隔离

- [官方代码](https://github.com/michaelyuancb/ftp1-policy)，固定提交 `89fa681d6c014cce28300946b7526db808e0b1c1`。
- [官方权重](https://huggingface.co/MJJJJ1064/ftp1_univtac_finetune)，固定版本 `620ac69b4fffd2341300cfef1b1d224d56710ed3`，子目录 `FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1/19999`。
- 大文件经官方 ModelScope 镜像下载，逐文件校验固定 Hugging Face 版本的 LFS SHA256；完整记录在 `evotac/checkpoints/ftp1_univtac/verified_source.json`。主模型 SHA256 为 `bd6a75220a5c7de5e9704b898e204ad4b259e38660c6a25e34802ef432025e10`。
- 源码、权重、依赖、缓存和运行记录全部位于 `evotac/`。模型进程使用 `.cache/venvs/ftp1`，只读复用基础环境的 Torch 2.7.0+cu126；Transformers 4.53.2 及官方补丁仅安装于该 venv。Isaac Sim 继续使用原 UniVTAC Python。
- 模型和渲染分别使用物理 GPU 2 / GPU 1，可通过配置与启动环境调整。所有仿真启动均要求 livestream，拒绝 `--headless`。

## 启动

以下命令从仓库根目录执行。本机已经完成依赖与权重准备；前两条仅用于复现安装。

```bash
/data/ZED/conda/envs/UniVTAC/bin/python -m evotac.scripts.setup_ftp1
/data/ZED/conda/envs/UniVTAC/bin/python -m evotac.scripts.download_ftp1_checkpoint

OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/data/ZED/conda/envs/UniVTAC/lib \
/data/ZED/conda/envs/UniVTAC/bin/python -u -m evotac.scripts.run_ftp1 \
  --livestream 2 --device cuda:0 --model-gpu 2 \
  --seed 0 --max-controls 200 --run-id ftp1_my_run
```

用户已授权本机接受协议。`run-id` 必须未使用。控制、图像和事件记录在 `evotac/datasets/phase1/<run-id>/`；检查报告与模型输入、预测动作、推理日志记录在 `evotac/runs/phase1/<run-id>/`。程序结束后 streaming 服务随仿真退出。无本地显示器时可能出现 GLFW 告警，实际渲染是否成功应以有效 RGB 帧、传感器时间戳和渲染结果为准。

独立离线推理入口为 `python -m evotac.scripts.smoke_ftp1_model --episode <EvoTac HDF5> --run-id <新名称>`。它仍然加载真实模型，只是不启动 Isaac。

## 观测与动作接口

适配依据固定版本的 `UniVTAC/policy/FTP1/deploy_policy.py`：头部、腕部 RGB 各缩放至 224×224；左右指 `rgb_marker` 按左、右顺序组成触觉输入，保持 RGB 和 0–255 值域。模型只能读取部署观测，不持有 task 实例，也不读取成功标签或物体真值。

120 维状态中，索引 `9:16` 为七个机械臂关节，索引 `44` 为第一个夹爪手指的位置，其余补零。使用检查点自带的 `UniVTAC_insert_hole` 归一化统计和训练过的 HPT 触觉编码器。提示词为 `insert the stick to the hole.`。

模型返回 32×120 动作。检查点声明 `mix`：臂关节为相对于**产生该 chunk 时状态**的增量，夹爪为绝对目标。跳过表示当前帧的索引 0，从索引 1 执行。默认每次仅执行一个目标再重新观测；配置中的 `execute_chunk_steps` 可执行更长前缀，每个目标始终相对同一个推理时刻状态解码。控制切换会丢弃缓存动作。

目标通过 EvoTac `joint_target`、`force=False` 接口执行，每次固定 6 个 120 Hz 物理步，即模拟控制频率 20 Hz。已有位置和增量限制仍适用，原始预测与实际执行目标均留存。模型推理的墙钟耗时不推进物理时间，因此这不是实时 20 Hz 部署声明。

启动时检查权重加载告警：只允许官方明确跳过的两个未参与推理的 expert `lm_head`，以及经过实际对象共享验证的词嵌入别名；任何其他缺失/多余键均报错。HPT 文件必须存在并由官方严格加载器加载。

## 验证与边界

- `model_smoke_02`：真实权重离线推理通过，输出 32×120 有限动作，首次推理约 0.493 秒；共享词嵌入验证、归一化与触觉权重加载通过。`model_smoke_01` 的加载检查失败记录保留，未删除。
- CPU 测试：36 项通过，覆盖 Phase 1 合约、输入布局、非法传感器、动作 chunk 基准和控制切换缓存清除。
- `ftp1_livestream_01`（场景 seed 0）：8 次真实模型控制、48 个物理步全部执行，六路图像有效且非恒定、age=0；手动达到短测上限，明确记录为未完成的 `executed_partial`。
- `ftp1_livestream_full_01`（场景 seed 0、推理 seed 0，最大预算 200 次）：实际执行 55 次推理和控制、330 个物理步，于 `object_lost` 终止，`task_success=false`。这是完整记录的有效失败试验，没有渲染、执行或模型基础设施异常。物体丢失判定来自原任务的夹持相对位移阈值，而非模型标签。
- 完整试跑的 56 组观测中，六路图像均有效、非恒定且 age=0；HDF5 审计为 `consistent`。首次推理 0.497 秒，均值 0.177 秒；所有控制均为 6 个物理步，机械臂实际关节位置发生变化。单次运行不能估计成功率；当前结论是模型闭环运行已验证，插孔成功尚未验证。
- 官方 UniVTAC 路径原先使用 Isaac Sim 4.5；这里适配到当前 5.1/Isaac Lab 2.3。检查点没有足以确认历史控制频率的元数据，因此不能把本接口的任务表现等同于官方基准。
- `checks.json` 的 `status` 表示运行状态，`task_success` 表示实际任务成功；达到手动 `--max-controls` 上限会明确记为 `executed_partial`。请同时检查 HDF5 的终止原因、有效性和完整性，不能仅凭 Kit 进程退出码判断成功。

本机运行证据：

- [完整试跑报告](../runs/phase1/ftp1_livestream_full_01/checks.json)
- [渲染、动作与数据审计](../runs/phase1/ftp1_livestream_full_01/rendering_and_execution_report.json)
- [实际头部相机视频](../runs/phase1/ftp1_livestream_full_01/head_camera.mp4)（20 FPS，模拟时间；不包含模型推理等待）
- [末帧画面](../runs/phase1/ftp1_livestream_full_01/camera_head_rgb_last.png)
- [独立离线推理](../runs/ftp1/model_smoke_02/checks.json)

生成审计和视频：

```bash
/data/ZED/conda/envs/UniVTAC/bin/python -m evotac.scripts.report_ftp1_run --run-id ftp1_my_run
/data/ZED/conda/envs/UniVTAC/bin/python -m pytest evotac/tests -q -o cache_dir=evotac/.pytest_cache
```

基础 Conda 环境复核仍为 Torch 2.7.0+cu126、Transformers 5.17.0、protobuf 7.36.1、Isaac Sim 5.1.0.0、Isaac Lab 2.3.0；FTP-1 的依赖覆盖只在新 venv 生效。原仓库受跟踪文件的 `git diff` 为空。

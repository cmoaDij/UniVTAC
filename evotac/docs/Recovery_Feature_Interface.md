# 冻结触觉编码器与恢复特征接口

验证报告：`runs/phase2/tactile_features_v1/checks.json`。这是前向／接口验收，不是恢复任务效果验收。

## 权重与预处理

- 官方 `byml/UniVTAC` revision：`e1aee7b0c95543b535e0146b2de3ee1bc6ddaabd`。
- `checkpoints/encoder.pth`：165,482,707 bytes；SHA256 `28097c91a1a1d051f65266fc7fdd84420dc184da9dcfc2006aa40af144cf15bf`，与官方 LFS 元数据一致。
- 完整 `encoder.network.Tactile`（ResNet18、512 维及全部五个解码头）以 `strict=True` 加载，随后只保留 backbone。下载来源和加载代码哈希随报告保存。
- 输入 `rgb_marker`，三通道 uint8 HWC，转换为 float32 CHW／255，再按原 encoder 训练代码 resize 至 **H=320，W=240**，双指顺序固定为左、右。没有 ImageNet normalization。原 ACT deploy 使用 256×256，不能把二者混为同一预处理版本。
- 参数 `requires_grad=False`，`train(True)` 也保持 eval，防止 SAC 外层模块切换模式后修改 BatchNorm。

## 输出契约

入口：`evotac.policy.tactile_features.recovery_features(observation, encoder, recovery_remaining_steps=...)`。恢复预算必须由执行器提供，不从图像推测。每条输出包含 `schema=evotac.recovery_features.v2` 和 encoder／预处理版本。

| 字段 | 形状／含义 |
|---|---|
| `tactile_embeddings` | `[3,2,512]`，当前／空载参考／持握参考，各自左右指；参考独立编码，未拼成九通道输入 |
| `difference_cues` | `[2,2,9]`，当前减空载／当前减持握；左右指；三个通道各自的有符号均值、绝对均值、RMS，图像差值先除 255 |
| `proprioception` | 23 维：7 关节位置、7 关节速度、2 手指位置、世界坐标下末端 xyz、wxyz 四元数；字段顺序随输出保存 |
| `previous_accepted_target` | 9 维：7 个已接受臂关节位置目标和两个手指目标；初始无命令时零填充，另给有效位 |
| `history_mask` | 原观测历史有效位；历史条目由 wrapper 维护，当前函数不假装重建 RNN 状态 |
| `physics_step`、`tactile_age_steps` | 当前物理时刻和两个触觉采样 age |
| `reference_physics_steps` | 空载／持握参考来源时刻，要求空载先于持握，参考不在未来 |
| `remaining_task_steps`、`remaining_recovery_steps` | 实际物理步预算，进入恢复不重置任务预算 |

`recovery_observation.recovery_vector` 将上述字段组成固定 **3162D** 单帧（3072 个三参考双指 latent、36 个差分 cue、23 个本体字段、9 个上一接受目标、两路相机 RGB 均值/标准差、有效位以及年龄/时间/预算），`RecoveryHistory` 以左侧 mask 保留最近 8 帧，冻结的 128D `HistoryEncoder` 输出给恢复 SAC。P4 使用独立的 `EffectHistoryEncoder`，不会共享或更新控制历史参数。旧的 `[2,9]`/缺少相机字段只在 simulator-free v1 测试适配器中兼容；真实 wrapper 必须提供 `[2,2,9]`、两路相机年龄和恢复预算。

编码器输入只来自部署观测白名单；输出不包含 actor 位姿、成功标签、物体丢失标签或场景条件。触觉差分是图像统计，不解释为校准后的力或滑移真值。

## 实测验收

- 从旧 dev 示范的三个父轨迹各取双指帧，输出 `[6,512]`；从新失败试跑选八个时刻，当前／两类参考输出 `[8,3,2,512]`，全部有限。
- 新图像当前特征平均 L2 范数 15.020，旧图像 14.081；这是描述统计，不是域兼容性阈值。
- 重复推理最大误差为 0；切到 `train(True)` 后仍 eval，所有 BatchNorm buffers 保持不变。
- 新轨迹不同时刻的当前特征最大差异 1.689；输入取反的特征最大变化 7.587，排除了该样本上的恒定输出。
- 在输入加入 actor 真值、成功和条件字段后，输出逐项不变。
- 缺文件／错误哈希、未来传感器、参考乱序和负预算均有拒绝测试。

旧 JPEG 数据按原编解码往返读取，未擅自交换颜色通道；发布权重的完整历史训练流程仍不能仅凭文件推断。现有 contact 数据尚未完成全量内容审计。只有后续训练收益不足时，再开展按轨迹划分的前端适配。

历史 mask 的旧版滞后一拍问题见 [阶段 2 计划](Phase2_Plan.md)。使用旧轨迹训练历史模型时，应依据真实 observation/action 索引重建 mask，不直接消费旧字段。

```bash
/data/ZED/conda/envs/UniVTAC/bin/python -m evotac.scripts.download_tactile_encoder
/data/ZED/conda/envs/UniVTAC/bin/python -m evotac.scripts.verify_tactile_features --run-id <新名称>
```

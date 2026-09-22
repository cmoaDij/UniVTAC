# P1：控制、观测与数据契约

状态：**已完成并通过 CPU 契约测试；真实接口已有运行证据**（2026-09-20）。

## 计划

提供固定六物理步控制边界、八维基础动作和七维恢复动作，隔离部署观测与特权训练信息，记录每个执行、失败、重置异常和零步拒绝，支持完整前缀重建。

## 已实现

- `evotac/envs/action_adapter.py` 实现 panda_hand 世界系旋转向量、阻尼 IK、关节和夹爪限位。
- `evotac/envs/fixed_step_executor.py` 是物理步唯一推进者，并写入实际物理步、目标接受和墙钟耗时。
- `evotac/envs/observation_contract.py` 用白名单生成相机、双指触觉、本体、参考和时间字段。
- `evotac/data/rollout_logger.py` 以 HDF5 追加保存完整失败分母；`state_replay.py` 保证场景完整性和版本隔离。
- 92 项 CPU 测试覆盖动作、终止、记录、重放、传感器时序、配对评估、GPU 预检、效果标签打包、在线恢复驱动、历史编码器、暖启动、快速 surrogate 和预算边界。

## 证据与边界

真实 `smoke_phase1` 和 handoff audit 已验证六步控制、非空传感器、预算计费和交接队列清空。重放是否可用属于 P2 的更严格门槛，不能由契约测试代替。

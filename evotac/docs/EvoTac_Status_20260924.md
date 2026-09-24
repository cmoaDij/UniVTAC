# EvoTac 当前状态（2026-09-24）

## 已完成

- π0.5 base 权重已下载：14,467,165,872 bytes。
- SHA-256：`0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`。
- train-only 数据集：44 个 parent、7,082 帧；触觉未作为学生输入。
- π0.5 严格加载 smoke 已通过；唯一允许的缺失键是与 `lm_head` 共享的 tied embedding。
- 处理链 smoke 已通过：RGB、8D proprioception、tokenizer、归一化均可生成模型输入。
- 全量测试：119 passed。

## 当前未完成

- π0.5 训练入口已补齐，但 10 step smoke 暂未完成：PyTorch CUDA runtime 在设备初始化时返回 `Error 101: invalid device ordinal`，因此训练入口现已 fail-closed，不会退回 CPU 伪训练。
- 尚未运行真实 π0.5 dev。
- 严格快照审计未通过，因此正式 test 仍禁止。

GPU UUID `GPU-d86528b1-67d4-d67f-db2c-ccae580a1267` 最近一次检查无 compute app，可在启动前再次复核；但当前 PyTorch CUDA 初始化仍报 `Error 101`，需先修复运行时可见性。历史 40 次真实 dev 结果仍为 baseline_a 5/10、baseline_b 5/10、warmstart 5/10、trained 2/10；这些结果仍只是探索性信号。

诊断命令：`CUDA_VISIBLE_DEVICES=8 PYTORCH_NVML_BASED_CUDA_CHECK=1 /data/ZED/UniVTAC/evotac/.cache/venvs/pi05/bin/python -m evotac.scripts.check_cuda_runtime --device cuda:0`。当前 `nvidia-smi` 可见 GPU，但 CUDA Runtime API 和 `deviceQuery` 都返回 101；这需要管理员级驱动/GSP 恢复或节点维护，不能由模型代码安全绕过。

## 下一步

1. 再次执行 GPU UUID 独占预检。
2. 使用新 run-id 进行 π0.5 train-only 微调，并冻结 checkpoint、统计量和训练配置。
3. 先运行 dev，不运行 test；只有严格共享状态/连续 rollout 审计通过后才解锁 test。

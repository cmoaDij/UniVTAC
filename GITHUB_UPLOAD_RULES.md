# UniVTAC GitHub 上传规则

目标：让代码能直接复现，同时避免把大文件和本机运行结果上传到仓库。

## 上传内容

- 提交核心 Python 源码、YAML/JSON 配置、测试、文档和必要的小型脚本。
- 提交与 EvoTac 直接相关的 TacEx 修改，并在提交说明中写明原因。
- 保留源码中的相对路径和默认配置，避免写入本机绝对路径。

## 不上传内容

- `datasets/`、`runs/`、`.cache/`、`.pytest_cache/`、`__pycache__/`
- 模型 checkpoint、权重、HDF5 运行数据、日志、临时文件和备份文件
- API key、token、账号信息、机器专用配置
- 未经说明的第三方大文件或自动生成文件

## 上传前检查

```bash
python -m pytest -q evotac/tests
git diff --check
git status --short
```

确认暂存区只包含本次功能相关文件；发现数据、日志、权重或密钥时先移出暂存区。

## 提交和推送

- 提交标题使用简短格式：`组件: 动作`，例如 `evotac: add snapshot audit`。
- 一次提交只做一组相关改动。
- 推送前查看 `git diff --cached --stat` 和 `git diff --cached --name-only`。
- 默认推送当前工作分支；不要强制推送，不要覆盖他人的提交。
- 运行结果放在本地报告或外部存储中，代码仓库只保存复现实验所需的实现和配置。

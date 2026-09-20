# 开发

在项目根目录安装依赖并运行检查：

```powershell
uv sync --extra training
uv run --extra training pytest -q
uv run --extra training ruff check src tests
uv run --extra training ruff format --check src tests
```

自动测试使用小模型、模拟 scorer 或本地测试客户端。真实模型表现按 [评测流程](evaluation.md)验证。

## 代码入口

| 内容 | 入口 |
|---|---|
| CLI 和配置 | [cli.py](../src/necro/cli.py) 的 `main`、[config.py](../src/necro/config.py) 的 `Settings` |
| HTTP 接口和数据结构 | [api.py](../src/necro/api.py) 的 `create_app`、[schema.py](../src/necro/schema.py) |
| 提示与结果转换 | [engine.py](../src/necro/engine.py) 的 `Task.messages`、`prepare_task`、`answer_for` |
| 模型加载和候选评分 | [backend.py](../src/necro/backend.py) 的 `TransformersScorer` |
| 评测和温度拟合 | [evaluation.py](../src/necro/evaluation.py) 的 `run_evaluation`、[calibration.py](../src/necro/calibration.py) 的 `fit_temperature` |
| 训练和导出 | [training.py](../src/necro/training.py) 的 `train`、[export.py](../src/necro/export.py) 的 `export` |

## 更新文档

使用说明按 [文档索引](README.md) 分工，实验的配置、数字和结论保留在带日期的报告中。默认值与字段限制链接到定义它们的代码，避免在多页手动同步。

`release_report.py` 包含实验报告和独立模型卡的文本模板。调整这些内容时同步模板，再检查导出包中的文档副本、相对链接和校验和。

# 2026-09-20 报告与模型卡生成

本轮使用 [release_report.py](../../src/necro/release_report.py) 的 `release_report` 整理已有评测文件，生成最终报告、训练记录和模型卡。

## 输入

脚本从 `results/improvement/final/` 读取对照、校准、延迟与导出验证结果，从 `data/improvement/` 读取 manifest、测试和边界题。各轮配置与开发集结果来自对应训练目录，完整文件清单见函数中的读取和复制路径。

`SELECTED_RUN` 是含 `run_config.json` 的训练目录，第二个参数是已导出的包目录：

```powershell
uv run --extra training python -m necro.release_report SELECTED_RUN artifacts/ScarletKc-Necro-0.8b
```

## 输出

- `docs/reports/improvement-2026-09-20.md`：导出权重的评测报告。
- `docs/process/improvement-2026-09-20.md`：训练轮次、数据与选择记录。
- 包内的 `adapter/README.md`、`merged/README.md`：独立模型卡。
- 包内的 `evaluation/`：汇总、配置、逐题判断和失败样例。

生成后按 [导出与发布](../publishing.md#验证与整理材料)同步文档与运行代码，再更新校验和。

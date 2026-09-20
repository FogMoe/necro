# 2026-09-20 改进训练记录

本轮从 [首轮 LoRA](lora-pilot-2026-09-20.md) 继续实验。模型选择遵循 [事先确定的规则](improvement-protocol.md)，导出权重的结果见 [评测报告](../reports/improvement-2026-09-20.md)。

## 训练与开发集比较

各轮使用相同基模 revision、rank 16 和 BF16，只更新 LoRA。续训重新建立优化器。开发集固定为 pilot 验证集。

| 实验 | 本轮记录数 | 学习率 | 纯训练时间 | 开发集准确率 | 开发集 NLL |
|---|---:|---:|---:|---:|---:|
| round2 | 994 | 3e-05 | 3.77 分钟 | 73.00% | 1.3562 |
| round3 | 2336 | 5e-05 | 8.38 分钟 | 84.00% | 0.8219 |
| round4 | 2336 | 2e-05 | 5.51 分钟 | 85.00% | 1.0229 |

round2 重复 pilot 数据。round3 从 pilot-v1 续训，扩充公开 train 样例、平衡 NLI 类别，并加入证据不足三元组。round4 在 round3 上以较低学习率再训练一轮。

本轮选择 `round3`，具体依据记录在 `results/improvement/selection.json`。选择方法的后续核查见 [设计复审](design-review-2026-09-20.md#校准后的开发集比较)。

## 数据

pilot 与 expanded 合计呈现 3,330 条训练记录，按上下文去重为 2,432 条，来源组 1,302 个。中英文包含对应翻译，续训再次使用已有样例。标签来自公开标注和代码构造规则。

expanded 的 XNLI Choice 按语言和类别平衡，本轮每组 176 条。取样、转换与排除逻辑见 [训练指南](../training.md#continuing-training-and-expanding-data)。

## 回归

| 数据 | 导出模型 |
|---|---:|
| 旧诊断集，400 题 | 85.75% |
| 自编开发样例，24 题 | 24/24 |

此前成绩分别保存在 [基模基线](baseline-2026-09-20.md) 和 [首轮 LoRA](lora-pilot-2026-09-20.md)。原 pilot 在本轮测试切片上的准确率为 69.25%。

## 复现本轮续训

先完成 [训练指南](../training.md) 中的 pilot 数据和权重准备，再运行：

```powershell
uv run --extra training python -m necro.experiments
uv run --extra training python -m necro.probes
uv run --extra training python -m necro.training --data data/improvement/expanded --output results/improvement/round3 --initial-adapter results/lora-pilot/run1/adapter --learning-rate 0.00005 --seed 29 --model-id necro-qwen3.5-0.8b-r3
```

本轮训练配置、日志和逐题结果保存在 `results/improvement/`，数据指纹保存在 `data/improvement/manifest.json`。

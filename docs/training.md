# LoRA 训练

训练脚本使用 PyTorch 和 PEFT，只更新语言骨干中的 LoRA 参数。训练与服务共用提示模板和 tokenizer，监督目标是正确答案的 token。每轮配方和成绩见 [过程记录](process/README.md)。

## 数据准备

先安装训练依赖，并准备旧诊断集和 pilot 数据：

```powershell
uv sync --extra training
uv run necro prepare-eval
uv run --extra training python -m necro.training_data
```

先生成 `data/baseline.jsonl`，数据准备脚本才会把它纳入重叠排除。pilot 输出包含 `train.jsonl`、`validation.jsonl`、来源缓存和 `manifest.json`。

公开数据来自 XNLI 与 MASSIVE 的中英文 train split。部分 XNLI 转为是否蕴含的 Noul，部分 MASSIVE 转为包含正确类别的小候选集，其余保留完整类别。Choice 选项顺序打乱，再加入代码构造的成对 Noul 和 Score 样例。

同一来源组中，只要有上下文与受保护的评测数据重复，就移除整组。取样与转换规则见 [training_data.py](../src/necro/training_data.py) 的 `prepare`、`rule_pairs` 和 `audit_disjoint`。每次实际数量、来源和文件 SHA-256 写入 `manifest.json`。

准备命令会复用已有来源缓存并重写训练文件与 manifest。新实验需要不同数据时，通过 `--output` 指定独立目录。公开数据来源与许可见 [第三方声明](../THIRD_PARTY_NOTICES.md)。

## 训练与前后对照

以下示例从基模开始，固定通用温度，并清除 Choice 校准值：

```powershell
$env:NECRO_MODEL = 'Qwen/Qwen3.5-0.8B'
$env:NECRO_ADAPTER = ''
$env:NECRO_TEMPERATURE = '1.0'
$env:NECRO_CHOICE_TEMPERATURE = ''
uv run --extra training necro evaluate data/lora-pilot/validation.jsonl --output results/lora-pilot/before
uv run --extra training python -m necro.training --output results/lora-pilot/run1
$env:NECRO_ADAPTER = 'results/lora-pilot/run1/adapter'
uv run --extra training necro evaluate data/lora-pilot/validation.jsonl --output results/lora-pilot/after
uv run --extra training necro evaluate data/baseline.jsonl --output results/lora-pilot/after-baseline
uv run --extra training necro evaluate examples/smoke.jsonl --output results/lora-pilot/after-smoke
```

训练输出目录已存在时会报错。再次运行使用新的目录名，并为前后评测分别保留结果。

每条记录使用 [评测 JSONL 格式](evaluation.md#数据格式与来源)。训练目录同时需要 `train.jsonl` 与 `validation.jsonl`，训练前检查两者隔离。超长记录直接报错，长度限制见 [training.py](../src/necro/training.py) 的 `encode_example`。

## 训练方式

基模使用 BF16 并冻结参数，只在语言骨干的 Linear 层加入 LoRA。输入按长度分桶，启用梯度检查点。每次运行一轮，使用 warmup 后线性下降的学习率。

通过 `--objective` 选择监督目标，prompt 在两种模式下都被屏蔽：

| 目标 | 计算方式 |
|---|---|
| `answer-ce` | 正确答案 token 的全词表交叉熵 |
| `candidate-ce` | 单 token 答案使用候选内交叉熵，多 token 数字标签使用全词表答案损失 |

多 token 答案使用 teacher forcing，逐 token 的负对数似然相加。训练开始前运行少量前向和反向以测量耗时，随后更新参数。结束后禁用 adapter，用同一探针检查基模 logits。

默认配方见 [training.py](../src/necro/training.py) 的 `train`，可调整的命令行参数见 `python -m necro.training --help`。实际使用值写入输出目录的 `run_config.json`。

| 输出 | 内容 |
|---|---|
| `run_config.json` | 基模 revision、提示指纹、训练参数、数据哈希和续训来源 |
| `profile.json` | 参数更新前的性能测量 |
| `training_summary.json` | 训练耗时、loss、显存和基模探针对照 |
| `adapter/` | LoRA 权重、PEFT 配置、tokenizer 和 `necro_adapter.json` |

## 续训与扩充数据

`--initial-adapter` 从已有 LoRA 继续训练，重新建立优化器。加载时检查基模、revision、rank 和提示契约。

项目中的扩充数据流程以 pilot 数据和旧诊断集为输入：

```powershell
uv run --extra training python -m necro.experiments
uv run --extra training python -m necro.probes
```

`experiments` 准备扩充训练集、校准集和封存测试文件。它平衡 XNLI 类别，混合不同大小的 MASSIVE 候选集，并加入证据不足样例。来源范围和分组排除规则见 [experiments.py](../src/necro/experiments.py) 的 `download_sources` 与 `prepare_improvement`。已有 expanded 目录时拒绝覆盖。

`probes` 生成金额、状态、Score 阈值和不同候选数量的边界题。已有文件时拒绝覆盖。历史续训命令与选择规则见 [改进训练记录](process/improvement-2026-09-20.md)。

## 检查结果

按同一批题比较训练前后结果，同时检查任务、语言和问题类型的变化。保留训练配置、原始预测、校准文件及测试集指纹。

服务加载步骤见 [运行与配置](running.md#加载模型)，概率拟合见 [评测](evaluation.md#概率校准)，权重整理见 [导出与发布](publishing.md)。

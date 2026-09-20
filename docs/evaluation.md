# 评测

用同一份 JSONL 比较基模、LoRA 和 Jev，并在结果中记录模型、配置与数据指纹。已有成绩见 [评测报告](README.md#评测报告)，模型加载方式见 [运行与配置](running.md#加载模型)。

## 运行一次评测

从项目根目录运行：

```powershell
uv run --extra inference necro evaluate examples/smoke.jsonl --output results/smoke
uv run --extra inference necro prepare-eval
uv run --extra inference necro evaluate data/baseline.jsonl --output results/local
```

加载 LoRA 时使用 `--extra training`。用 `--limit` 可以先运行少量记录，完整参数见 `necro evaluate --help`。评测会覆盖输出目录中的同名结果文件，比较不同模型时使用不同目录。

每次生成三个文件：

| 文件 | 内容 |
|---|---|
| `summary.json` | 总体与分组指标、运行元数据 |
| `predictions.jsonl` | 每条输入对应的完整响应和预期答案 |
| `judgments.jsonl` | 逐题预测、正确性与概率指标 |

本地元数据记录 checkpoint、模型 ID、adapter 路径、revision、提示版本、温度、运行库版本、设备和数据 SHA-256。CUDA 运行另记显卡名称与峰值已分配显存。`--limit` 只限制评测记录数，数据 SHA-256 仍计算整个输入文件。

## 数据格式与来源

JSONL 每行是一条独立记录，包含 `id`、`request` 和 `expected`。`expected` 的键对应问题 ID，Choice 填选项名，Noul 填布尔值，Score 填从零开始的等级索引。`source` 和 `language` 用于分组。完整例子见 [smoke.jsonl](../examples/smoke.jsonl)。

项目附带的 smoke 样例覆盖中英文、三种问题、否定、引用、结构化描述和输入中的指令，预期答案按明示规则构造。

`necro prepare-eval` 从以下数据源抽取中英文 validation 切片：

| 来源 | 任务 | 候选项 |
|---|---|---|
| [facebook/xnli](https://huggingface.co/datasets/facebook/xnli) | 根据前提判断支持、矛盾或信息不足 | 蕴含、矛盾、中立 |
| [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) | 用户意图分类 | 保留原始完整类别集 |

MASSIVE 使用 MTEB 的列式副本，类别名称来自 [datasets.py](../src/necro/datasets.py) 的 `INTENTS`。抽样参数见 [cli.py](../src/necro/cli.py) 的 `prepare-eval`，下载与转换见 `prepare_public_eval`。

开发、校准和测试分别使用不同切片，按原始来源组排除上下文重叠。中英文中有同源翻译，统计时按来源组处理。训练数据准备流程见 [训练](training.md#数据准备)，历史实验使用的切片和数量记录在各自报告中。

## 指标

指标实现见 [evaluation.py](../src/necro/evaluation.py) 的 `summarize` 与 `metrics`。

| 指标 | 计算方式 |
|---|---|
| `accuracy` | 最大概率类别是否正确。Noul 以 0.5 为界，等于 0.5 时取 true。Score 比较最大概率等级 |
| `nll` | 正确标签的负对数概率，计算时使用 `max(p, 1e-12)` |
| `brier` | 各类别概率与 one-hot 答案的平方误差之和，再对问题取平均 |
| `ece_10_bins` | 按最大候选概率分十个等宽区间，比较区间平均概率与准确率 |
| `score_mae` | Score 的加权均值与预期等级的平均绝对误差 |
| `candidate_mass_mean`、`candidate_mass_min` | 本地后端的合法答案概率总量，含义见 [API](api.md#概率与评分) |

准确率越高越好，NLL、Brier、ECE 和 Score MAE 越低越好。阅读结果时同时看数据集、语言和问题类型分组。ECE 使用最大候选概率，API 的 `confidence` 按另一公式计算。

## 概率校准

先固定模型，再对单独的校准集拟合 Choice 温度。下面假定 `data/calibration.jsonl` 已准备好，模型路径已按运行指南设置：

```powershell
$env:NECRO_TEMPERATURE = '1.0'
$env:NECRO_CHOICE_TEMPERATURE = ''
uv run --extra training necro evaluate data/calibration.jsonl --output results/calibration-raw
uv run --extra training python -m necro.calibration results/calibration-raw/predictions.jsonl --output results/calibration.json
$calibration = Get-Content results/calibration.json -Raw | ConvertFrom-Json
$env:NECRO_CHOICE_TEMPERATURE = $calibration.temperature.ToString([Globalization.CultureInfo]::InvariantCulture)
```

拟合命令读取 Choice 预测，以最小平均 NLL 选择温度。应用后重新评测保留的测试集，记录参数和结果。单个正温度保持 Choice 类别排序，Noul 与 Score 继续使用通用温度。基模评测可以把命令中的 extra 换成 `inference`。

## 延迟与吞吐

```powershell
uv run --extra inference necro benchmark --repeats 20
uv run --extra inference necro benchmark --repeats 20 --questions 4
```

`benchmark` 在本地进程内运行固定短文本请求，模型加载和预热后开始计时，输出全部延迟样本、p50、p95 与单独的加载耗时。多问题请求重复同一道题，用于比较批量规模。

`evaluate` 的 `elapsed_seconds` 记录整批输入准备、推理和结果转换时间，`questions_per_second` 是整批吞吐。远程评测计时包含网络和客户端调度。记录 HTTP 延迟时，应另行保存请求、并发数、预热次数和服务地址类型。

## Jev 对照

在 `.env` 中设置 `TYPESAFE_API_KEY`，然后执行：

```powershell
uv run necro evaluate data/baseline.jsonl --backend jev --output results/jev
```

该命令会把评测输入发送到 `TYPESAFE_BASE_URL`。请求的模型名由 [evaluation.py](../src/necro/evaluation.py) 的 `evaluate_remote` 固定，响应的实际模型 ID 写入结果。远程错误按该函数的状态码和重试策略处理。

两边使用相同的输入和预期答案。保存逐题结果后，可以检查分歧来自哪些任务、语言或条件，再决定下一轮的数据与评测范围。

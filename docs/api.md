# API

Necro 使用 [TypeSafe HTTP API](https://docs.typesafe.ai/api) 的请求和响应格式，由本地 Qwen 模型完成判断。字段定义见 [schema.py](../src/necro/schema.py) 的 `EvaluationRequest` 和 `EvaluationResponse`。

## 发送请求

`POST /v1/systemone` 接受 JSON，请求头使用 `Authorization: Bearer <NECRO_API_KEY>`。服务启动步骤见 [运行与配置](running.md)。下面的 PowerShell 示例使用项目附带的三类型请求，密钥需与服务配置一致：

```powershell
$headers = @{ Authorization = 'Bearer necro-local' }
Invoke-RestMethod http://127.0.0.1:8000/v1/systemone -Method Post -Headers $headers -ContentType 'application/json; charset=utf-8' -InFile examples/request.json
```

请求必须包含 `model`、`state` 和非空 `questions`。`state` 与每题的 `instructions` 接受字符串、对象或数组。每个问题独立判断，问题 ID 用于对应响应，不送入模型。请求结构拒绝未定义的额外字段。

| 类型 | `criteria` | 响应 |
|---|---|---|
| `choice` | 选项名到描述的映射，描述可以是字符串、对象、数组或 `null` | `choice` 为最大概率选项，`probabilities` 包含全部选项，另有 `confidence` |
| `noul` | 可选对象，`true` 和 `false` 描述两种结果 | `noul` 为回答是的概率 |
| `score` | 从低到高排列的等级描述数组 | `score` 为等级索引的概率加权均值，`probabilities` 和 `legend` 以从零开始的字符串索引为键，另有 `confidence` |

Score 的等级描述在 `legend` 中保留原始结构。概率相同时，Choice 取请求中排在前面的选项。结果计算见 [engine.py](../src/necro/engine.py) 的 `answer_for`。

## 模型名称与其他端点

`GET /v1/models` 需要相同的 Bearer key，返回加载器标识的模型。`GET /health` 无需鉴权，返回状态与模型 ID。通过 CLI 启动时，模型加载完成后才开始监听。

请求可以使用返回的模型 ID，或 [config.py](../src/necro/config.py) 中 `MODEL_ALIASES` 定义的迁移别名。别名都路由到同一个已加载模型，包括 Jev 名称。LoRA 和本地合并权重使用各自契约中的 `model_id`，基模使用 `MODEL_ID`。

## 概率与评分

候选概率是在给定标签集合内归一化的相对概率。单 token 标签使用 logits，完整编号使用联合 log probability，然后计算 `softmax(scores / temperature)`。Choice 取最大项，Noul 返回 Yes 的概率，Score 返回 `sum(index * probability)`。

Choice 和 Score 的 `confidence` 表示分布集中度，使用 [engine.py](../src/necro/engine.py) 的 `normalized_peak`：

```text
n == 1: confidence = 1
n > 1: confidence = (n * max(probabilities) - 1) / (n - 1)
```

温度设置见 [运行配置](running.md#配置来源)，拟合方法见 [概率校准](evaluation.md#概率校准)。

本地评测额外记录 `candidate_mass`，即归一化前合法标签或完整编号路径的概率总量。它用于观察模型有多少概率分配给有效答案。

## 标签评分实现

每题的全部选项放进同一个提示，关闭 thinking。Noul 使用 Yes/No，较小候选集合使用单字母标签。超过字母范围时，使用等长、无前导零的数字编号，计算完整编号的联合概率。

加载 tokenizer 时验证标签和实际答案边界的编码。数字评分用前缀树复用缓存，分叉时复制 attention、卷积和循环状态。映射与评分实现见 [backend.py](../src/necro/backend.py) 的 `single_token_labels`、`numeric_labels`、`verify_answer_boundary` 和 `TransformersScorer`。

多问题按长度分批，单进程内串行使用模型。不同问题会重复编码 state，本机性能可按 [评测方法](evaluation.md#延迟与吞吐)测量。

## 错误与输入限制

| HTTP 状态 | 含义 |
|---|---|
| 401 | 密钥缺失或无效 |
| 422 | 字段、模型名或输入长度不符合要求，详情在响应的 `detail` 中 |
| 529 | 检测到推理显存不足 |
| 500 | 其他内部错误 |

Choice 的候选数量和 Score 的等级数量限制见 [schema.py](../src/necro/schema.py) 的 `Choice.criteria` 与 `Score.criteria`。单题与整请求 token 上限见 [config.py](../src/necro/config.py) 的 `Settings.max_sequence_tokens` 和 `Settings.max_request_tokens`。

长度使用 Qwen tokenizer 计算，包含提示包装。请求预算累计各题实际提示，state 会随每题重复计算。超长输入直接报错。批量 token 预算用于分批，超过批量预算但未超过单题上限的长题可以单独运行。

`usage.input_tokens` 统计实际输入的提示 tokens，也包括多 token 编号评分时输入的编号前缀。`usage.output_tokens` 为零，因为没有生成文本。

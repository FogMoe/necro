# API

Necro follows the [TypeSafe HTTP API](https://docs.typesafe.ai/api) request and response format, with judgments made by a local Qwen model. The field definitions are `EvaluationRequest` and `EvaluationResponse` in [schema.py](../src/necro/schema.py).

## Sending a request

`POST /v1/systemone` accepts JSON and requires `Authorization: Bearer <NECRO_API_KEY>`. Start the server using the [setup guide](running.md). This PowerShell example sends the bundled request with all three question types. Match the key to your server configuration.

```powershell
$headers = @{ Authorization = 'Bearer necro-local' }
Invoke-RestMethod http://127.0.0.1:8000/v1/systemone -Method Post -Headers $headers -ContentType 'application/json; charset=utf-8' -InFile examples/request.json
```

Each request requires `model`, `state`, and a nonempty `questions` map. `state` and each question's `instructions` accept strings, objects, or arrays. Questions are evaluated independently. Question IDs map answers back to requests and are not sent to the model. Unknown fields are rejected.

| Type | `criteria` | Response |
|---|---|---|
| `choice` | Map of option names to descriptions. Descriptions accept strings, objects, arrays, or `null` | `choice` is the highest-probability option. `probabilities` contains every option. Includes `confidence` |
| `noul` | Optional object with `true` and `false` descriptions | `noul` is the probability of yes |
| `score` | Array of level descriptions ordered from low to high | `score` is the probability-weighted mean of level indices. `probabilities` and `legend` use zero-based string indices. Includes `confidence` |

Score preserves each level description's original structure in `legend`. Choice breaks probability ties by selecting the option that appears first in the request. Response calculation is implemented by `answer_for` in [engine.py](../src/necro/engine.py).

## Model names and other endpoints

`GET /v1/models` requires the same Bearer key and returns the model identified by the loader. `GET /health` returns status and model ID without authentication. When started through the CLI, the server begins listening after the model has loaded.

Requests accept the returned model ID or a migration alias from `MODEL_ALIASES` in [config.py](../src/necro/config.py). All aliases, including Jev names, route to the same loaded model. LoRA and local merged weights use `model_id` from their contracts. The base model uses `MODEL_ID`.

## Probabilities and scores

Candidate probabilities are relative probabilities normalized over the supplied labels. Single-token labels use logits, while complete numeric labels use joint log probabilities. Both are normalized with `softmax(scores / temperature)`. Choice selects the maximum, Noul returns the probability of Yes, and Score returns `sum(index * probability)`.

For Choice and Score, `confidence` measures how concentrated the distribution is. It uses `normalized_peak` in [engine.py](../src/necro/engine.py):

```text
n == 1: confidence = 1
n > 1: confidence = (n * max(probabilities) - 1) / (n - 1)
```

See [configuration sources](running.md#configuration-sources) for temperature settings and [probability calibration](evaluation.md#probability-calibration) for fitting a temperature.

Local evaluation also records `candidate_mass`, the total probability assigned to valid labels or complete numeric paths before normalization. It shows how much of the model's probability mass falls on valid answers.

## Label scoring

All options for a question are placed in one prompt with thinking disabled. Noul uses Yes/No. Smaller candidate sets use single-letter labels. Larger sets use equal-length numeric labels without leading zeros and score the joint probability of each complete label.

Tokenizer loading validates label encodings at the actual answer boundary. Numeric scoring reuses cached prefixes through a trie, copying attention, convolution, and recurrent state at each branch. See `single_token_labels`, `numeric_labels`, `verify_answer_boundary`, and `TransformersScorer` in [backend.py](../src/necro/backend.py).

Questions are batched by length, and model access is serialized within a process. Each question encodes the state again. Use the [evaluation procedure](evaluation.md#latency-and-throughput) to measure local performance.

## Errors and input limits

| HTTP status | Meaning |
|---|---|
| 401 | Missing or invalid key |
| 422 | Invalid fields, model name, or input length. See `detail` in the response |
| 529 | Inference ran out of GPU memory |
| 500 | Other internal error |

Candidate and level count limits are defined by `Choice.criteria` and `Score.criteria` in [schema.py](../src/necro/schema.py). Per-question and per-request token limits are defined by `Settings.max_sequence_tokens` and `Settings.max_request_tokens` in [config.py](../src/necro/config.py).

Lengths are measured with the Qwen tokenizer and include prompt formatting. The request budget sums each question's prompt, including its copy of the state. Oversized inputs raise an error. The batch token budget controls batching. A question above that budget can run alone if it is within the per-question limit.

`usage.input_tokens` counts prompt tokens actually processed, including numeric label prefixes used during multi-token scoring. `usage.output_tokens` is zero because scoring does not generate text.

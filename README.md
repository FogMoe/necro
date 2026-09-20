# Necro

Necro uses Qwen3.5-0.8B to make choices, answer yes/no questions, and assign ordered scores locally. Give it context, a question, and candidate answers to get a structured result with probabilities. It scores answer labels directly, without generating an explanation or parsing generated JSON.

Run the base model, load a LoRA adapter, or use exported merged weights. The HTTP API follows TypeSafe's request and response format so existing clients can connect. Results are documented in the [evaluation report](docs/README.md#evaluation-report).

## Quick start

Run these PowerShell commands from the project root. See [setup and configuration](docs/running.md#installation) for environment requirements.

```powershell
uv sync --extra inference
Copy-Item .env.example .env
uv run --extra inference necro score examples/request.json
uv run --extra inference necro serve
```

Skip the copy step if you already have a `.env` file. With the source checkout's default configuration, the first inference run downloads the base model. `score` prints the result in your terminal, and `serve` starts the local API. The example configuration uses `http://127.0.0.1:8000` and the local test key `necro-local`. Set your key in `.env`.

Once the server is running, save and run this example in the project environment to connect with the official Python SDK:

```python
from typesafe_sdk import TypeSafeClient

with TypeSafeClient(api_key="necro-local", base_url="http://127.0.0.1:8000") as client:
    result = client.system_one(
        state="I was charged twice. Please refund the duplicate payment.",
        questions={
            "refund": {"type": "noul", "instructions": "Does this message request a refund?"},
        },
    )
    print(result.answers["refund"].noul)
```

The SDK is included in the project's development dependencies. [examples/request.json](examples/request.json) demonstrates all three question types. See [setup and configuration](docs/running.md) for model loading, device selection, and troubleshooting.

## Documentation

- [API](docs/api.md): requests, responses, probabilities, and scores.
- [Evaluation](docs/evaluation.md): datasets, model comparisons, calibration, and latency.
- [Training](docs/training.md): LoRA data preparation, training, and validation.
- [Export and publishing](docs/publishing.md): weights, runtime files, and model cards.
- [Documentation index](docs/README.md): development, evaluation reports, and process records.

## License

The project uses [Apache-2.0](LICENSE), with training code under [MIT](LICENSE-MIT). See [licensing and third-party notices](THIRD_PARTY_NOTICES.md) for file scope, fine-tuning contributions, and upstream attribution.

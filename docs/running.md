# Setup and configuration

Necro can load the base model, a local LoRA adapter, or merged weights. Run the commands on this page from the project root. Relative paths are resolved from the working directory.

## Installation

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync --extra inference
Copy-Item .env.example .env
```

Skip the copy step if `.env` already exists. The Python requirement is defined by `project.requires-python` in [pyproject.toml](../pyproject.toml), and `uv.lock` pins dependency versions. PyTorch uses the CUDA wheels configured by `pytorch-cu128`. GPU inference requires a compatible NVIDIA driver. With `NECRO_DEVICE=auto`, inference uses CUDA when available and falls back to CPU. Training requires CUDA.

To load LoRA adapters, replace `--extra inference` with `--extra training` in installation and runtime commands to include PEFT. Remote checkpoints are downloaded from Hugging Face on first use. You can download the base model ahead of time with `uv run --extra inference hf download Qwen/Qwen3.5-0.8B`.

## Configuration sources

Process environment variables take precedence over `.env`. Unset values use the program defaults. See [.env.example](../.env.example) for an example and `Settings.from_env` and `Settings` in [config.py](../src/necro/config.py) for loading behavior and defaults.

| Setting | Purpose |
|---|---|
| `NECRO_MODEL` | Hugging Face checkpoint or local merged model directory |
| `NECRO_ADAPTER` | Local LoRA directory containing `necro_adapter.json`. Leave empty to run without an adapter |
| `NECRO_DEVICE` | `auto`, `cpu`, or a PyTorch CUDA device such as `cuda:0` |
| `NECRO_API_KEY` | Bearer key for the local API. Must be nonempty |
| `NECRO_BATCH_SIZE`, `NECRO_BATCH_TOKENS` | Question batch size and token budget after padding |
| `NECRO_TEMPERATURE` | Probability temperature for all question types |
| `NECRO_CHOICE_TEMPERATURE` | Choice-specific override. Leave empty to use the general temperature |
| `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` | Key and endpoint for [Jev comparisons](evaluation.md#jev-comparison) |

Batch settings must be positive integers. Temperatures must be finite and greater than zero. Restart the server after changing settings. Git ignores `.env`, and local inference does not require a TypeSafe key.

## Loading a model

### Base model

Set the following in `.env`:

```dotenv
NECRO_MODEL=Qwen/Qwen3.5-0.8B
NECRO_ADAPTER=
NECRO_TEMPERATURE=1.0
NECRO_CHOICE_TEMPERATURE=
```

Clear or update any matching variables already set in your terminal, since they override `.env`.

### LoRA adapter

Set `NECRO_MODEL` to the base checkpoint named by `checkpoint` in the adapter contract, and `NECRO_ADAPTER` to the adapter directory. Run with `--extra training`.

The loader checks the base model name and prompt code fingerprint. It loads the model and tokenizer at the `revision` recorded in `necro_adapter.json`, then merges the adapter in memory. Restart the server to switch adapters.

### Merged weights

Set `NECRO_MODEL` to a local directory containing `necro_model.json`, and clear `NECRO_ADAPTER`. Run with `--extra inference`. Merged weights already include the LoRA changes. Applying another adapter raises an error.

### Applying exported calibration

The export's `export.json` records `choice_temperature`. The model contract also stores it as `recommended_choice_temperature`. Set `NECRO_CHOICE_TEMPERATURE` explicitly to apply it. For example, read an existing export from the project root:

```powershell
$package = 'artifacts/ScarletKc-Necro-0.8b'
$metadata = Get-Content "$package/export.json" -Raw | ConvertFrom-Json
$env:NECRO_CHOICE_TEMPERATURE = $metadata.choice_temperature.ToString([Globalization.CultureInfo]::InvariantCulture)
```

For a standalone model distribution, use its model card to set paths and temperature.

## Starting the server

```powershell
uv run --extra inference necro score examples/request.json
uv run --extra inference necro serve --host 127.0.0.1 --port 8000
```

`score` reads a JSON request and writes a JSON response to stdout. `serve` loads the model before starting the HTTP server. Set the host and port through command-line options.

Once the server is running, check it from another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The health check returns the model ID used by the server. Send the [example API request](api.md#sending-a-request) to check inference. Interactive API documentation is available at `/docs`.

## Troubleshooting

| Issue | Action |
|---|---|
| CUDA is unavailable | Check the NVIDIA driver and PyTorch CUDA installation, or use `NECRO_DEVICE=cpu` for inference |
| `peft` is missing | Install and run with `--extra training` |
| Adapter base model or prompt fingerprint mismatch | Use the runtime shipped with the adapter and the base checkpoint specified in its contract |
| Changes to `.env` have no effect | Check for matching terminal environment variables and restart the server |
| HTTP 401 | Match the client Bearer key to the server's `NECRO_API_KEY` |
| HTTP 422 | Check the error details against the [API input requirements](api.md#errors-and-input-limits) |
| HTTP 529 | Shorten the input or reduce the batch size and token budget |

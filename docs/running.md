# 运行与配置

Necro 支持基模、本地 LoRA 和合并权重三种加载方式。本页的命令都从项目根目录运行，相对路径也以运行命令的目录为准。

## 安装

安装 [uv](https://docs.astral.sh/uv/)，然后执行：

```powershell
uv sync --extra inference
Copy-Item .env.example .env
```

已有 `.env` 时跳过复制。Python 要求见 [pyproject.toml](../pyproject.toml) 的 `project.requires-python`，依赖版本由 `uv.lock` 锁定。PyTorch 使用其中 `pytorch-cu128` 配置的 CUDA 轮子，GPU 推理需要兼容的 NVIDIA 驱动。`NECRO_DEVICE=auto` 会在 CUDA 可用时使用 GPU，否则使用 CPU。训练要求 CUDA。

加载 LoRA 时，把安装和运行命令中的 `--extra inference` 换成 `--extra training`，以安装 PEFT。首次加载远程 checkpoint 会从 Hugging Face 下载模型，也可用 `uv run --extra inference hf download Qwen/Qwen3.5-0.8B` 提前下载。

## 配置来源

进程环境变量优先于 `.env`，未设置的项使用程序默认值。配置示例见 [.env.example](../.env.example)，加载逻辑和完整默认值见 [config.py](../src/necro/config.py) 的 `Settings.from_env` 与 `Settings`。

| 配置 | 用途 |
|---|---|
| `NECRO_MODEL` | Hugging Face checkpoint 或本地合并模型目录 |
| `NECRO_ADAPTER` | 含 `necro_adapter.json` 的本地 LoRA 目录，留空则不加载 LoRA |
| `NECRO_DEVICE` | `auto`、`cpu` 或 PyTorch CUDA 设备名，例如 `cuda:0` |
| `NECRO_API_KEY` | 本地 API 的 Bearer key，不能为空 |
| `NECRO_BATCH_SIZE`、`NECRO_BATCH_TOKENS` | 问题批量大小和 padding 后的 token 预算 |
| `NECRO_TEMPERATURE` | 所有问题使用的概率温度 |
| `NECRO_CHOICE_TEMPERATURE` | 单独覆盖 Choice 温度，留空沿用通用温度 |
| `TYPESAFE_API_KEY`、`TYPESAFE_BASE_URL` | [Jev 对照评测](evaluation.md#jev-对照)的密钥与地址 |

批量参数必须为正整数，温度必须为大于零的有限数。修改配置后重启服务。`.env` 被 Git 忽略，本地推理不需要 TypeSafe 密钥。

## 加载模型

### 基模

在 `.env` 中设置：

```dotenv
NECRO_MODEL=Qwen/Qwen3.5-0.8B
NECRO_ADAPTER=
NECRO_TEMPERATURE=1.0
NECRO_CHOICE_TEMPERATURE=
```

如果终端中已经设置了同名环境变量，也要清除或更新它们，否则它们会覆盖 `.env`。

### LoRA

设置 `NECRO_MODEL` 为 adapter 契约中 `checkpoint` 对应的基模，`NECRO_ADAPTER` 为 adapter 目录。使用 `--extra training` 运行。

加载器检查基模名称和提示代码指纹，按 `necro_adapter.json` 中的 `revision` 加载模型与 tokenizer，再将 LoRA 合并到内存。切换 adapter 后需要重启。

### 合并权重

将 `NECRO_MODEL` 指向含 `necro_model.json` 的本地合并模型目录，并清空 `NECRO_ADAPTER`。使用 `--extra inference` 运行。合并权重已经包含 LoRA 改动，再次叠加 adapter 会报错。

### 使用导出包的校准值

导出包的 `export.json` 记录 `choice_temperature`，模型契约中也保存了 `recommended_choice_temperature`。需要显式设置 `NECRO_CHOICE_TEMPERATURE` 才会应用该值。例如从项目根目录读取已有包：

```powershell
$package = 'artifacts/ScarletKc-Necro-0.8b'
$metadata = Get-Content "$package/export.json" -Raw | ConvertFrom-Json
$env:NECRO_CHOICE_TEMPERATURE = $metadata.choice_temperature.ToString([Globalization.CultureInfo]::InvariantCulture)
```

从独立模型目录运行时，按该目录模型卡的命令设置路径和温度。

## 启动和确认

```powershell
uv run --extra inference necro score examples/request.json
uv run --extra inference necro serve --host 127.0.0.1 --port 8000
```

`score` 读取一个 JSON 请求并在 stdout 输出 JSON 响应。`serve` 先加载模型，再启动 HTTP 服务。主机和端口通过命令行参数设置。

服务启动后，在另一个终端查看：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

健康检查返回服务使用的模型 ID。需要确认推理时，发送 [API 示例请求](api.md#发送请求)。交互式接口文档位于服务的 `/docs`。

## 常见问题

| 现象 | 处理 |
|---|---|
| `CUDA 不可用` | 检查 NVIDIA 驱动及 PyTorch CUDA 环境，或用 `NECRO_DEVICE=cpu` 运行推理 |
| 缺少 `peft` | 安装并使用 `--extra training` |
| adapter 基模或提示代码不一致 | 使用 adapter 随附的运行代码和契约指定的基模 |
| 修改 `.env` 后没有生效 | 检查终端的同名环境变量，并重启服务 |
| HTTP 401 | 让客户端 Bearer key 与服务的 `NECRO_API_KEY` 一致 |
| HTTP 422 | 按错误详情检查字段和长度，具体限制见 [API](api.md#错误与输入限制) |
| HTTP 529 | 缩短输入，或降低批量大小与 token 预算 |

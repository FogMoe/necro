# Necro

Necro 用 Qwen3.5-0.8B 在本地做选择、是非判断和分级评分。给它上下文、问题和候选项，就能拿到结构化结果和概率。模型直接计算答案标签的概率，省去生成解释和解析 JSON 的过程。

可以直接运行基模，也可以加载 LoRA 或导出的合并权重。HTTP 接口采用 TypeSafe 的请求与响应格式，方便已有客户端接入。已有实测放在 [评测报告](docs/README.md#评测报告)。

## 开始使用

在项目根目录运行以下 PowerShell 命令。环境要求见 [运行与配置](docs/running.md#安装)。

```powershell
uv sync --extra inference
Copy-Item .env.example .env
uv run --extra inference necro score examples/request.json
uv run --extra inference necro serve
```

已有 `.env` 时跳过复制。首次推理会下载基模。`score` 在终端输出结果，`serve` 启动本地 API。示例配置使用 `http://127.0.0.1:8000` 和本地测试密钥 `necro-local`，可在 `.env` 中修改密钥。

服务启动后，可以用官方 Python SDK 连接。在项目环境中保存并运行下面的脚本：

```python
from typesafe_sdk import TypeSafeClient

with TypeSafeClient(api_key="necro-local", base_url="http://127.0.0.1:8000") as client:
    result = client.system_one(
        state="我被重复扣款，请退掉多扣的一笔。",
        questions={
            "refund": {"type": "noul", "instructions": "用户是否要求退款？"},
        },
    )
    print(result.answers["refund"].noul)
```

SDK 随项目的开发依赖安装。完整的三类型请求见 [examples/request.json](examples/request.json)。加载微调模型、切换设备和排查启动问题，见 [运行与配置](docs/running.md)。

## 文档

- [API](docs/api.md)：输入、响应、概率和评分。
- [评测](docs/evaluation.md)：准备数据、比较模型、校准概率和测量延迟。
- [训练](docs/training.md)：准备 LoRA 数据、训练和验证。
- [导出与发布](docs/publishing.md)：整理权重、运行代码和模型卡。
- [文档索引](docs/README.md)：开发说明、评测报告和过程记录。

## 许可

项目采用 [Apache-2.0](LICENSE)，训练代码采用 [MIT](LICENSE-MIT)。文件范围、微调贡献和第三方来源见 [许可与第三方声明](THIRD_PARTY_NOTICES.md)。

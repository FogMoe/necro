# 导出与发布

导出会生成 LoRA adapter、合并权重和运行代码。整理好模型卡与评测材料后，可以把 adapter 或 merged 目录作为独立模型仓库上传。

## 本地导出

从项目根目录运行。将 `SELECTED_ADAPTER` 和 `CHOICE_TEMPERATURE` 替换为选中权重的目录与校准值：

```powershell
uv run --extra training python -m necro.export SELECTED_ADAPTER artifacts/ScarletKc-Necro-0.8b --temperature CHOICE_TEMPERATURE
```

模型 ID 来自 adapter 的 `necro_adapter.json`。运行时使用的 `NECRO_MODEL` 必须匹配该契约的 `checkpoint`。输出目录已存在时拒绝覆盖。

| 导出内容 | 用途 |
|---|---|
| `adapter/` | LoRA safetensors、PEFT 配置、tokenizer 和提示契约，加载时配合原始基模 |
| `merged/` | 已合并的完整模型、tokenizer 和 `necro_model.json` |
| `runtime/` | 候选评分 API、训练和评测代码、依赖锁定、测试与文档 |
| `export.json` | 模型 ID、Choice 温度、基模 revision 和权重 SHA-256 |

文件选择见 [export.py](../src/necro/export.py) 的 `export`。它按固定清单复制运行材料，原始训练数据、缓存和凭据保留在工作目录。

## 验证与整理材料

加载 adapter 和合并权重，使用相同输入与温度比较选择结果和逐项概率，运行命令见 [运行与配置](running.md#加载模型) 与 [评测](evaluation.md)。将验证结果、模型选择依据和校准参数放入包内的 `evaluation/`。

独立模型卡至少包括用途、运行命令、训练来源、评测条件、成绩、已知错误和许可。运行命令应从模型目录的 `runtime/` 启动，并明确使用 adapter 还是合并权重。

现有实验的报告生成入口见 [材料整理记录](process/publishing-2026-09-20.md)。报告只记录结果，训练轮次和选择经过放在 `docs/process/`，模型卡链接到包内报告。

报告生成后，把更新后的文档和运行代码同步到包根的 `runtime/`，再为两个独立模型目录补齐运行代码和评测材料：

```powershell
$package = 'artifacts/ScarletKc-Necro-0.8b'
Copy-Item README.md, LICENSE, LICENSE-MIT, THIRD_PARTY_NOTICES.md -Destination "$package/runtime" -Force
Get-ChildItem docs | Copy-Item -Destination "$package/runtime/docs" -Recurse -Force
Copy-Item src/necro/release_report.py -Destination "$package/runtime/src/necro" -Force
foreach ($kind in @('adapter', 'merged')) {
    Copy-Item "$package/runtime" -Destination "$package/$kind" -Recurse
    Copy-Item "$package/evaluation" -Destination "$package/$kind" -Recurse
}
```

这段复制命令用于尚未包含 `runtime/` 和 `evaluation/` 的独立模型目录。已有副本时按目录逐项同步，并移除已迁走的旧文档。整理完成后生成整个包的 `SHA256SUMS`，修改任何文件后重新生成。

## 上传 Hugging Face

登录 Hub，选择账户、仓库名与可见性，再上传准备好的独立模型目录。下面的命令会创建或更新远程仓库：

```powershell
uv run --extra training hf auth login
uv run --extra training hf upload YOUR_NAMESPACE/ScarletKc-Necro-0.8b artifacts/ScarletKc-Necro-0.8b/merged . --private
```

替换 `YOUR_NAMESPACE` 为目标账户或组织。`--private` 只在创建新仓库时设置私有，已有仓库沿用其可见性。要新建公开仓库则使用 `--no-private`。命令参数可通过 `hf upload --help` 查看。

## 许可

项目的 [Apache-2.0](../LICENSE)、训练代码的 [MIT](../LICENSE-MIT) 和上游许可证随模型与运行代码一起分发。具体范围见 [许可与第三方声明](../THIRD_PARTY_NOTICES.md)，模型卡链接到对应副本。

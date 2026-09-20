# 2026-09-20 首轮 LoRA

Qwen3.5-0.8B 使用 994 条中英文样例训练一轮，在 200 题验证集上的准确率从 38.5% 提升到 72.0%。本轮纯参数更新耗时 239.33 秒。

## 数据

XNLI 与 MASSIVE 的 train split 各取中英文 200 条。四分之一的 XNLI 转为 Noul，一半 MASSIVE 使用包含正确答案的 8 选项子集，其余保留全部 60 类。Choice 选项顺序打乱。

另加入 100 对规则样例，每对只翻转一个相关布尔字段。移除与旧诊断集及新验证集重叠的 3 个来源组后，实际训练 994 条，其中 694 Choice、200 Noul、100 Score，中英文各 497 条。

验证集取两个数据源各语言 validation 的第 100–149 行，共 200 题，MASSIVE 保留全部 60 类。转换与隔离方法见 [训练指南](../training.md#data-preparation)。

## 配方与成本

| 项目 | 本次设置或测量 |
|---|---|
| GPU | RTX 5070 Ti Laptop，12 GB |
| 基模与计算 | Qwen3.5-0.8B、BF16 |
| LoRA | rank 16、alpha 32、dropout 0.05 |
| 可训练参数 | 10,822,656 |
| 批量 | microbatch 2、累积 4、有效批量 8 |
| 学习率 | `1e-4`，10% warmup 后线性下降 |
| 梯度裁剪 | 1.0 |
| 训练轮数与优化器步数 | 1 轮、125 步 |
| 输入 tokens | 305,640，不含 padding |
| 最大输入长度 | 上限 2048，本次最长 874 |
| 纯参数更新耗时 | 239.33 秒 |
| PyTorch 峰值已分配训练显存 | 2.38 GiB |
| adapter 权重 | 43,346,432 bytes，约 41.34 MiB |

监督目标为正确答案 token 的全词表交叉熵，多 token 标签使用 teacher forcing。按长度分桶，启用梯度检查点。训练前先测 6 个 microbatch 的前向和反向，结束后禁用 adapter，探针 logits 与训练前差异为零。完整运行步骤见 [训练指南](../training.md)。

## 验证集

| 分组 | 基模 | LoRA |
|---|---:|---:|
| XNLI 英文 | 38% | 64% |
| XNLI 中文 | 38% | 64% |
| MASSIVE 英文 | 40% | 78% |
| MASSIVE 中文 | 38% | 82% |

96 道原先错误的题改对，29 道原先正确的题改错，净增加 67 道正确答案。NLL 从 2.2871 降到 1.0389，Brier 从 0.7484 降到 0.3944，ECE 从 0.1338 变为 0.1360。

## 旧诊断集与开发样例

| 数据 | LoRA |
|---|---:|
| XNLI 英文，100 题 | 65% |
| XNLI 中文，100 题 | 62% |
| MASSIVE 英文，100 题 | 72% |
| MASSIVE 中文，100 题 | 79% |
| 公开切片合计，400 题 | 69.50% |
| 自编开发样例，24 题 | 23/24 |

基模与同题 Jev 的成绩保存在 [基线记录](baseline-2026-09-20.md#判断结果)。本轮回归集 NLL 为 1.0969，Brier 为 0.4379，ECE 为 0.1677。XNLI 的单独 NLL 略有退步。

## 推理验证

adapter 保存后在新进程加载并合并到内存，200 题用时 14.86 秒，400 题用时 32.56 秒。短单题预热后运行 20 次，p50 为 38.18 ms，p95 为 42.69 ms。

官方 Python SDK 0.7.0 的 HTTP 调用验证了模型 ID `necro-qwen3.5-0.8b-lora-pilot-v1`、三种响应和中文 legend。255 选项的精确匹配样例返回完整分布，并选中预期的 `item254`。本轮 48 项自动检查通过。

## 原始记录

训练、权重、前后对照、延迟与 HTTP 检查保存在 `results/lora-pilot/`。其中 `run1/run_config.json` 记录基模 revision 和提示契约，`run1/training_summary.json` 记录训练结果。

训练 SHA-256 为 `baba10584b1b21d616ee3a68d860fb0e4ceb75c771ea3e9a28554eb76ff79cf8`，验证集 SHA-256 为 `3cafef6f57d547cc2f9ca5d6a5aecd8f1135d879618534cb0b7fbe15faf6fb8b`。后续实验见 [改进训练记录](improvement-2026-09-20.md)。

# 2026-09-20 ScarletKc-Necro-0.8b 评测报告

本报告评测导出的 `ScarletKc-Necro-0.8b` 权重。XNLI 与 MASSIVE 测试切片共 374 题，准确率为 78.61%，同题 Jev 1.13.0 为 81.28%，相差 2.67 个百分点。

模型来自 `round3`。权重来源与各轮开发集比较见 [训练记录](../process/improvement-2026-09-20.md)，使用方法见 [运行与配置](../running.md)。

## 测试数据与结果

测试数据取自两个数据集 test split 各语言的前 100 行。按来源组移除与候选训练、开发及校准上下文重叠的 13 组后，保留 374 题。中英文含同源记录，模型选择完成后读取这批测试结果。

| 分组 | 题数 | Necro | Jev 1.13.0 |
|---|---:|---:|---:|
| facebook/xnli/en/choice | 100 | 78.00% | 89.00% |
| mteb/amazon_massive_intent/en/choice | 87 | 83.91% | 85.06% |
| facebook/xnli/zh/choice | 100 | 71.00% | 66.00% |
| mteb/amazon_massive_intent/zh/choice | 87 | 82.76% | 86.21% |

按原始来源组进行 5,000 次配对 bootstrap，Necro 减 Jev 的准确率差 95% 区间为 [-7.75, 2.41] 个百分点。指标定义见 [评测](../evaluation.md#指标)。

## 概率校准

在独立 validation 切片去除重叠后得到的 198 题上，以 NLL 选择 Choice 温度 1.721103。封存测试的 NLL 从 0.9582 降到 0.7230，ECE 从 0.1340 变为 0.0444。

| 指标 | Necro | Jev API 返回值 |
|---|---:|---:|
| NLL | 0.7230 | 1.8249 |
| Brier | 0.2991 | 0.2886 |
| ECE | 0.0444 | 0.1018 |

Jev 返回的分布中存在零概率，NLL 采用 `max(p, 1e-12)` 计算。输出精度和截断方式会影响低概率样本的 NLL。

校准值随导出包保存，应用方式见 [运行与配置](../running.md#使用导出包的校准值)。

## 边界题与错误

64 道边界题覆盖 Noul 金额与状态判断、四级 Score，以及不同候选数量的精确匹配。Necro 正确 56/64，Jev 正确 64/64。

错误包括 7 道 Noul 金额比较和 1 道中文 Score 阈值判断。例如已付 77、应付 78、状态为 settled 时，模型仍判定已结清。另一次将 2 项检查失败归入了 3–4 项的等级。

逐题请求与错误保存在包内的 `evaluation/probe-failures.jsonl`。

## 延迟

在 RTX 5070 Ti Laptop GPU 上，对同一道短 Choice 请求预热后串行运行 20 次。本地记录回环 HTTP 延迟，Jev 记录含网络与服务开销的远程 HTTPS 延迟。

| 服务 | p50 | p95 |
|---|---:|---:|
| Necro | 38.72 ms | 70.97 ms |
| Jev 1.13.0 | 350.34 ms | 453.98 ms |

## 权重与接口验证

LoRA 权重约 41.34 MiB。合并 safetensors 为 1,706,030,528 bytes，约 1.59 GiB。导出后重新加载合并权重，对本轮测试逐项比较，选择结果完全一致，概率最大差异约 2.28e-7。

官方 Python SDK 0.7.0 的 HTTP 调用验证了模型名称、三种响应类型和中文 legend。本轮验证时 55 项自动测试通过。

## 复核入口

- 基模 revision：`2fc06364715b967f1860aea9cf38778875588b17`。
- 测试集 SHA-256：`9ddf80506ff04aa7ac0bf520a84c7b4df1acb496b64740f027a259e72480e3e1`。
- 本地完整结果：`results/improvement/final/`。
- 导出包中的汇总、判断和验证材料：`evaluation/`。
- [训练记录](../process/improvement-2026-09-20.md)、[数据来源与许可](../../THIRD_PARTY_NOTICES.md)。

# 开源实现参考

这些项目分别提供数据构造、训练和候选评分的实现参考。Necro 的训练流程见 [训练指南](../training.md)，实测结果见 [评测报告](../reports/improvement-2026-09-20.md)。

## Bespoke Nimble

[训练配方](https://github.com/bespokelabsai/nimble/blob/main/docs/NIMBLE_TRAINING.md) · [数据说明](https://github.com/bespokelabsai/nimble/blob/main/docs/DATASET.md) · [模型卡](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B)

可以从数据说明查看成对样例和来源分组的方法，再从训练配方核对监督目标、训练参数和模型选择过程。复现实验时，配方对应的数据版本和学习率调度需要一起读取。

## Simple Jev 与 RFDT

[项目说明](https://github.com/featherless-ai/simple-jev) · [RFDT](https://github.com/featherless-ai/simple-jev/tree/main/RFDT) · [评分代码](https://github.com/featherless-ai/simple-jev/blob/main/common/response_scoring.py)

RFDT 提供决策任务训练的入口。评分代码适合对照提示与答案的 token 边界、候选评分和响应转换，训练部分可查看 LoRA、数据划分及教师标注缓存的处理。

## jev-local

[项目说明](https://github.com/us/jev-local) · [scorer.py](https://github.com/us/jev-local/blob/main/src/jevlocal/scorer.py)

可以从 scorer 实现查看候选拼接、序列评分和后端选择，再结合接口测试检查请求与响应。比较实现时，关注完整序列概率与按 token 平均分数的差别。

更多实现可从 [awesome-jev](https://github.com/OmniJev/awesome-jev) 查找。

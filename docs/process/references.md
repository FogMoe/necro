# Open-source implementation references

These projects provide implementation references for data construction, training, and candidate scoring. Necro's workflow is in the [training guide](../training.md), and measured results are in the [evaluation report](../reports/improvement-2026-09-20.md).

## Bespoke Nimble

[Training recipe](https://github.com/bespokelabsai/nimble/blob/main/docs/NIMBLE_TRAINING.md) · [Dataset documentation](https://github.com/bespokelabsai/nimble/blob/main/docs/DATASET.md) · [Model card](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B)

The dataset documentation covers paired examples and source grouping. The training recipe describes the objective, training parameters, and model selection. When reproducing an experiment, read the recipe together with its dataset version and learning-rate schedule.

## Simple Jev and RFDT

[Project](https://github.com/featherless-ai/simple-jev) · [RFDT](https://github.com/featherless-ai/simple-jev/tree/main/RFDT) · [Scoring code](https://github.com/featherless-ai/simple-jev/blob/main/common/response_scoring.py)

RFDT provides an entry point for training decision tasks. The scoring code is useful for comparing prompt/answer token boundaries, candidate scoring, and response conversion. The training implementation covers LoRA, data splits, and cached teacher annotations.

## jev-local

[Project](https://github.com/us/jev-local) · [scorer.py](https://github.com/us/jev-local/blob/main/src/jevlocal/scorer.py)

The scorer implementation covers candidate concatenation, sequence scoring, and backend selection. API tests show request and response behavior. When comparing implementations, distinguish full-sequence probability from scores averaged across tokens.

More implementations are listed in [awesome-jev](https://github.com/OmniJev/awesome-jev).

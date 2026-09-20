# Licensing and third-party notices

## Project licenses

Original project code, documentation and fine-tuning contributions use [Apache-2.0](LICENSE), except for the MIT training files listed below.

The following files use the [MIT License](LICENSE-MIT) and carry an SPDX identifier:

- `src/necro/training.py`
- `src/necro/training_data.py`
- `src/necro/experiments.py`
- `tests/test_training.py`

The Python package includes both licenses. Model cards identify the fine-tuned weights as Apache-2.0.

## Qwen3.5-0.8B

- Source: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).
- License: Apache-2.0. The license is included as `QWEN_APACHE_LICENSE` in model distributions and `licenses/QWEN_APACHE_LICENSE` in source and runtime distributions.
- Changes: LoRA fine-tuning of the language backbone. Merged distributions incorporate those modifications.
- Base revision: recorded in `necro_adapter.json` or `necro_model.json` in each model distribution.

## Training data

| Source | Attribution | License |
|---|---|---|
| [XNLI / XNLI-MT](https://github.com/facebookresearch/XNLI), distributed through [facebook/xnli](https://huggingface.co/datasets/facebook/xnli) | Conneau et al., *XNLI: Evaluating Cross-lingual Sentence Representations* (2018) | CC BY-NC 4.0 |
| [MASSIVE](https://github.com/alexa/massive), using the [MTEB representation](https://huggingface.co/datasets/mteb/amazon_massive_intent) | FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages* (2022) | Apache-2.0, as listed in the MTEB dataset card |

The XNLI license is included as `XNLI_CC_BY_NC_4.0` in model distributions and `licenses/XNLI_CC_BY_NC_4.0` in source and runtime distributions.

Additional rule and uncertainty examples were constructed within this project. Training labels come from the public datasets and constructed rules. Training data remains separate from the distributed model package.

## Dependencies and references

Software dependencies and bundled third-party skills retain their own licenses. TypeSafe and Jev identify the API format and evaluation service used by this project.

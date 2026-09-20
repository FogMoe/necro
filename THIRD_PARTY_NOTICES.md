# Licensing and third-party notices

## Project licenses

Original project code, documentation and fine-tuning contributions use [Apache-2.0](LICENSE), except for the MIT training files listed below.

The following files use the [MIT License](LICENSE-MIT) and carry an SPDX identifier:

- `src/necro/training/trainer.py`
- `src/necro/training/__main__.py`
- `src/necro/training/data/training_data.py`
- `src/necro/training/data/experiments.py`
- `tests/training/test_training.py`

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
| [MASSIVE](https://github.com/alexa/massive), using the [MTEB representation](https://huggingface.co/datasets/mteb/amazon_massive_intent) | FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages* (2022); English seed data: Bastianelli et al., *SLURP: A Spoken Language Understanding Resource Package* (2020) | The [upstream dataset card](https://huggingface.co/datasets/AmazonScience/massive) lists CC BY 4.0; the MTEB copy's metadata lists Apache-2.0. Upstream attribution and terms are retained. |
| [PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x) | Google LLC; Yang et al., *PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification* (2019) | [Custom upstream terms](https://github.com/google-research-datasets/paws/blob/master/LICENSE), allowing use for any purpose; Google attribution retained |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | Clark et al., *BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions* (2019) | CC BY-SA 3.0, as listed in the dataset card |
| [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) | Rajpurkar et al., *SQuAD: 100,000+ Questions for Machine Comprehension of Text* (2016) | CC BY-SA 4.0, as listed in the dataset card |
| [CMRC 2018](https://github.com/ymcui/cmrc2018), via [hfl/cmrc2018](https://huggingface.co/datasets/hfl/cmrc2018) | CMRC 2018 organizers / HFL; *A Span-Extraction Dataset for Chinese Machine Reading Comprehension* | CC BY-SA 4.0, as listed in the official dataset card; official train split used for third-phase paired candidate selection |
| [SciFact, BEIR representation](https://huggingface.co/datasets/BeIR/scifact) and [qrels](https://huggingface.co/datasets/BeIR/scifact-qrels) | Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (2020); BEIR representation by Thakur et al. | CC BY-SA 4.0, as listed in the dataset card |

The XNLI license is included as `XNLI_CC_BY_NC_4.0` in model distributions and `licenses/XNLI_CC_BY_NC_4.0` in source and runtime distributions.

Additional rule and uncertainty examples were constructed within this project. Training labels come from the public datasets and constructed rules. Training data remains separate from the distributed model package.

PAWS-X, BoolQ, SQuAD and SciFact are used in second- and third-phase experiments. SQuAD questions are converted into candidate selection, including a no-match outcome when the gold answer is withheld. SciFact qrels are converted into selection among sampled documents; sampled negative candidates have no relevance judgments. Numeric and ordinal examples have programmatically computed labels. Per-run manifests identify which sources a specific adapter actually used.

[XQuAD](https://huggingface.co/datasets/google/xquad), by Artetxe et al., *On the Cross-lingual Transferability of Monolingual Representations* (2020), is used for development, calibration and evaluation, not gradient training. Its dataset card lists CC BY-SA 4.0. Source records, translations, transformed questions and derived data retain their respective source terms. Dataset repository revisions and local snapshot hashes are recorded in experiment manifests.

## Dependencies and references

Software dependencies and bundled third-party skills retain their own licenses. TypeSafe and Jev identify the API format and evaluation service used by this project.

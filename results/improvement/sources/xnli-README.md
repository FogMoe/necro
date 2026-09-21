---
language:
- ar
- bg
- de
- el
- en
- es
- fr
- hi
- ru
- sw
- th
- tr
- ur
- vi
- zh
paperswithcode_id: xnli
pretty_name: Cross-lingual Natural Language Inference
dataset_info:
- config_name: all_languages
  features:
  - name: premise
    dtype:
      translation:
        languages:
        - ar
        - bg
        - de
        - el
        - en
        - es
        - fr
        - hi
        - ru
        - sw
        - th
        - tr
        - ur
        - vi
        - zh
  - name: hypothesis
    dtype:
      translation_variable_languages:
        languages:
        - ar
        - bg
        - de
        - el
        - en
        - es
        - fr
        - hi
        - ru
        - sw
        - th
        - tr
        - ur
        - vi
        - zh
        num_languages: 15
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 1581471691
    num_examples: 392702
  - name: test
    num_bytes: 19387432
    num_examples: 5010
  - name: validation
    num_bytes: 9566179
    num_examples: 2490
  download_size: 963942271
  dataset_size: 1610425302
- config_name: ar
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 107399614
    num_examples: 392702
  - name: test
    num_bytes: 1294553
    num_examples: 5010
  - name: validation
    num_bytes: 633001
    num_examples: 2490
  download_size: 59215902
  dataset_size: 109327168
- config_name: bg
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 125973225
    num_examples: 392702
  - name: test
    num_bytes: 1573034
    num_examples: 5010
  - name: validation
    num_bytes: 774061
    num_examples: 2490
  download_size: 66117878
  dataset_size: 128320320
- config_name: de
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 84684140
    num_examples: 392702
  - name: test
    num_bytes: 996488
    num_examples: 5010
  - name: validation
    num_bytes: 494604
    num_examples: 2490
  download_size: 55973883
  dataset_size: 86175232
- config_name: el
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 139753358
    num_examples: 392702
  - name: test
    num_bytes: 1704785
    num_examples: 5010
  - name: validation
    num_bytes: 841226
    num_examples: 2490
  download_size: 74551247
  dataset_size: 142299369
- config_name: en
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 74444026
    num_examples: 392702
  - name: test
    num_bytes: 875134
    num_examples: 5010
  - name: validation
    num_bytes: 433463
    num_examples: 2490
  download_size: 50627367
  dataset_size: 75752623
- config_name: es
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 81383284
    num_examples: 392702
  - name: test
    num_bytes: 969813
    num_examples: 5010
  - name: validation
    num_bytes: 478422
    num_examples: 2490
  download_size: 53677157
  dataset_size: 82831519
- config_name: fr
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 85808779
    num_examples: 392702
  - name: test
    num_bytes: 1029239
    num_examples: 5010
  - name: validation
    num_bytes: 510104
    num_examples: 2490
  download_size: 55968680
  dataset_size: 87348122
- config_name: hi
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 170593964
    num_examples: 392702
  - name: test
    num_bytes: 2073073
    num_examples: 5010
  - name: validation
    num_bytes: 1023915
    num_examples: 2490
  download_size: 70908548
  dataset_size: 173690952
- config_name: ru
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 129859615
    num_examples: 392702
  - name: test
    num_bytes: 1603466
    num_examples: 5010
  - name: validation
    num_bytes: 786442
    num_examples: 2490
  download_size: 70702606
  dataset_size: 132249523
- config_name: sw
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 69285725
    num_examples: 392702
  - name: test
    num_bytes: 871651
    num_examples: 5010
  - name: validation
    num_bytes: 429850
    num_examples: 2490
  download_size: 45564152
  dataset_size: 70587226
- config_name: th
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 176062892
    num_examples: 392702
  - name: test
    num_bytes: 2147015
    num_examples: 5010
  - name: validation
    num_bytes: 1061160
    num_examples: 2490
  download_size: 77222045
  dataset_size: 179271067
- config_name: tr
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 71637140
    num_examples: 392702
  - name: test
    num_bytes: 934934
    num_examples: 5010
  - name: validation
    num_bytes: 459308
    num_examples: 2490
  download_size: 48509680
  dataset_size: 73031382
- config_name: ur
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 96441486
    num_examples: 392702
  - name: test
    num_bytes: 1416241
    num_examples: 5010
  - name: validation
    num_bytes: 699952
    num_examples: 2490
  download_size: 46682785
  dataset_size: 98557679
- config_name: vi
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 101417430
    num_examples: 392702
  - name: test
    num_bytes: 1190217
    num_examples: 5010
  - name: validation
    num_bytes: 590680
    num_examples: 2490
  download_size: 57690058
  dataset_size: 103198327
- config_name: zh
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': entailment
          '1': neutral
          '2': contradiction
  splits:
  - name: train
    num_bytes: 72224841
    num_examples: 392702
  - name: test
    num_bytes: 777929
    num_examples: 5010
  - name: validation
    num_bytes: 384851
    num_examples: 2490
  download_size: 48269855
  dataset_size: 73387621
configs:
- config_name: all_languages
  data_files:
  - split: train
    path: all_languages/train-*
  - split: test
    path: all_languages/test-*
  - split: validation
    path: all_languages/validation-*
- config_name: ar
  data_files:
  - split: train
    path: ar/train-*
  - split: test
    path: ar/test-*
  - split: validation
    path: ar/validation-*
- config_name: bg
  data_files:
  - split: train
    path: bg/train-*
  - split: test
    path: bg/test-*
  - split: validation
    path: bg/validation-*
- config_name: de
  data_files:
  - split: train
    path: de/train-*
  - split: test
    path: de/test-*
  - split: validation
    path: de/validation-*
- config_name: el
  data_files:
  - split: train
    path: el/train-*
  - split: test
    path: el/test-*
  - split: validation
    path: el/validation-*
- config_name: en
  data_files:
  - split: train
    path: en/train-*
  - split: test
    path: en/test-*
  - split: validation
    path: en/validation-*
- config_name: es
  data_files:
  - split: train
    path: es/train-*
  - split: test
    path: es/test-*
  - split: validation
    path: es/validation-*
- config_name: fr
  data_files:
  - split: train
    path: fr/train-*
  - split: test
    path: fr/test-*
  - split: validation
    path: fr/validation-*
- config_name: hi
  data_files:
  - split: train
    path: hi/train-*
  - split: test
    path: hi/test-*
  - split: validation
    path: hi/validation-*
- config_name: ru
  data_files:
  - split: train
    path: ru/train-*
  - split: test
    path: ru/test-*
  - split: validation
    path: ru/validation-*
- config_name: sw
  data_files:
  - split: train
    path: sw/train-*
  - split: test
    path: sw/test-*
  - split: validation
    path: sw/validation-*
- config_name: th
  data_files:
  - split: train
    path: th/train-*
  - split: test
    path: th/test-*
  - split: validation
    path: th/validation-*
- config_name: tr
  data_files:
  - split: train
    path: tr/train-*
  - split: test
    path: tr/test-*
  - split: validation
    path: tr/validation-*
- config_name: ur
  data_files:
  - split: train
    path: ur/train-*
  - split: test
    path: ur/test-*
  - split: validation
    path: ur/validation-*
- config_name: vi
  data_files:
  - split: train
    path: vi/train-*
  - split: test
    path: vi/test-*
  - split: validation
    path: vi/validation-*
- config_name: zh
  data_files:
  - split: train
    path: zh/train-*
  - split: test
    path: zh/test-*
  - split: validation
    path: zh/validation-*
---

# Dataset Card for "xnli"

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [https://www.nyu.edu/projects/bowman/xnli/](https://www.nyu.edu/projects/bowman/xnli/)
- **Repository:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Paper:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Size of downloaded dataset files:** 7.74 GB
- **Size of the generated dataset:** 3.23 GB
- **Total amount of disk used:** 10.97 GB

### Dataset Summary

XNLI is a subset of a few thousand examples from MNLI which has been translated
into a 14 different languages (some low-ish resource). As with MNLI, the goal is
to predict textual entailment (does sentence A imply/contradict/neither sentence
B) and is a classification task (given two sentences, predict one of three
labels).

### Supported Tasks and Leaderboards

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Languages

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Dataset Structure

### Data Instances

#### all_languages

- **Size of downloaded dataset files:** 483.96 MB
- **Size of the generated dataset:** 1.61 GB
- **Total amount of disk used:** 2.09 GB

An example of 'train' looks as follows.
```
This example was too long and was cropped:

{
    "hypothesis": "{\"language\": [\"ar\", \"bg\", \"de\", \"el\", \"en\", \"es\", \"fr\", \"hi\", \"ru\", \"sw\", \"th\", \"tr\", \"ur\", \"vi\", \"zh\"], \"translation\": [\"احد اع...",
    "label": 0,
    "premise": "{\"ar\": \"واحدة من رقابنا ستقوم بتنفيذ تعليماتك كلها بكل دقة\", \"bg\": \"един от нашите номера ще ви даде инструкции .\", \"de\": \"Eine ..."
}
```

#### ar

- **Size of downloaded dataset files:** 483.96 MB
- **Size of the generated dataset:** 109.32 MB
- **Total amount of disk used:** 593.29 MB

An example of 'validation' looks as follows.
```
{
    "hypothesis": "اتصل بأمه حالما أوصلته حافلة المدرسية.",
    "label": 1,
    "premise": "وقال، ماما، لقد عدت للمنزل."
}
```

#### bg

- **Size of downloaded dataset files:** 483.96 MB
- **Size of the generated dataset:** 128.32 MB
- **Total amount of disk used:** 612.28 MB

An example of 'train' looks as follows.
```
This example was too long and was cropped:

{
    "hypothesis": "\"губиш нещата на следното ниво , ако хората си припомнят .\"...",
    "label": 0,
    "premise": "\"по време на сезона и предполагам , че на твоето ниво ще ги загубиш на следващото ниво , ако те решат да си припомнят отбора на ..."
}
```

#### de

- **Size of downloaded dataset files:** 483.96 MB
- **Size of the generated dataset:** 86.17 MB
- **Total amount of disk used:** 570.14 MB

An example of 'train' looks as follows.
```
This example was too long and was cropped:

{
    "hypothesis": "Man verliert die Dinge auf die folgende Ebene , wenn sich die Leute erinnern .",
    "label": 0,
    "premise": "\"Du weißt , während der Saison und ich schätze , auf deiner Ebene verlierst du sie auf die nächste Ebene , wenn sie sich entschl..."
}
```

#### el

- **Size of downloaded dataset files:** 483.96 MB
- **Size of the generated dataset:** 142.30 MB
- **Total amount of disk used:** 626.26 MB

An example of 'validation' looks as follows.
```
This example was too long and was cropped:

{
    "hypothesis": "\"Τηλεφώνησε στη μαμά του μόλις το σχολικό λεωφορείο τον άφησε.\"...",
    "label": 1,
    "premise": "Και είπε, Μαμά, έφτασα στο σπίτι."
}
```

### Data Fields

The data fields are the same among all splits.

#### all_languages
- `premise`: a multilingual `string` variable, with possible languages including `ar`, `bg`, `de`, `el`, `en`.
- `hypothesis`: a multilingual `string` variable, with possible languages including `ar`, `bg`, `de`, `el`, `en`.
- `label`: a classification label, with possible values including `entailment` (0), `neutral` (1), `contradiction` (2).

#### ar
- `premise`: a `string` feature.
- `hypothesis`: a `string` feature.
- `label`: a classification label, with possible values including `entailment` (0), `neutral` (1), `contradiction` (2).

#### bg
- `premise`: a `string` feature.
- `hypothesis`: a `string` feature.
- `label`: a classification label, with possible values including `entailment` (0), `neutral` (1), `contradiction` (2).

#### de
- `premise`: a `string` feature.
- `hypothesis`: a `string` feature.
- `label`: a classification label, with possible values including `entailment` (0), `neutral` (1), `contradiction` (2).

#### el
- `premise`: a `string` feature.
- `hypothesis`: a `string` feature.
- `label`: a classification label, with possible values including `entailment` (0), `neutral` (1), `contradiction` (2).

### Data Splits

|    name     |train |validation|test|
|-------------|-----:|---------:|---:|
|all_languages|392702|      2490|5010|
|ar           |392702|      2490|5010|
|bg           |392702|      2490|5010|
|de           |392702|      2490|5010|
|el           |392702|      2490|5010|

## Dataset Creation

### Curation Rationale

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the source language producers?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Annotations

#### Annotation process

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the annotators?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Personal and Sensitive Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Discussion of Biases

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Other Known Limitations

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Additional Information

### Dataset Curators

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Licensing Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Citation Information

```
@InProceedings{conneau2018xnli,
  author = {Conneau, Alexis
                 and Rinott, Ruty
                 and Lample, Guillaume
                 and Williams, Adina
                 and Bowman, Samuel R.
                 and Schwenk, Holger
                 and Stoyanov, Veselin},
  title = {XNLI: Evaluating Cross-lingual Sentence Representations},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  location = {Brussels, Belgium},
}
```


### Contributions

Thanks to [@lewtun](https://github.com/lewtun), [@mariamabarham](https://github.com/mariamabarham), [@thomwolf](https://github.com/thomwolf), [@lhoestq](https://github.com/lhoestq), [@patrickvonplaten](https://github.com/patrickvonplaten) for adding this dataset.
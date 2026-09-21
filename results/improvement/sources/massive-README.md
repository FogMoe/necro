---
annotations_creators:
- human-annotated
language:
- afr
- amh
- ara
- aze
- ben
- cmo
- cym
- dan
- deu
- ell
- eng
- fas
- fin
- fra
- heb
- hin
- hun
- hye
- ind
- isl
- ita
- jav
- jpn
- kan
- kat
- khm
- kor
- lav
- mal
- mon
- msa
- mya
- nld
- nob
- pol
- por
- ron
- rus
- slv
- spa
- sqi
- swa
- swe
- tam
- tel
- tgl
- tha
- tur
- urd
- vie
license: apache-2.0
multilinguality: translated
task_categories:
- text-classification
task_ids: []
configs:
- config_name: default
  data_files:
  - path: train/*.json.gz
    split: train
  - path: test/*.json.gz
    split: test
  - path: validation/*.json.gz
    split: validation
- config_name: ta
  data_files:
  - path: train/ta.json.gz
    split: train
  - path: test/ta.json.gz
    split: test
  - path: validation/ta.json.gz
    split: validation
- config_name: is
  data_files:
  - path: train/is.json.gz
    split: train
  - path: test/is.json.gz
    split: test
  - path: validation/is.json.gz
    split: validation
- config_name: pl
  data_files:
  - path: train/pl.json.gz
    split: train
  - path: test/pl.json.gz
    split: test
  - path: validation/pl.json.gz
    split: validation
- config_name: zh-CN
  data_files:
  - path: train/zh-CN.json.gz
    split: train
  - path: test/zh-CN.json.gz
    split: test
  - path: validation/zh-CN.json.gz
    split: validation
- config_name: el
  data_files:
  - path: train/el.json.gz
    split: train
  - path: test/el.json.gz
    split: test
  - path: validation/el.json.gz
    split: validation
- config_name: ru
  data_files:
  - path: train/ru.json.gz
    split: train
  - path: test/ru.json.gz
    split: test
  - path: validation/ru.json.gz
    split: validation
- config_name: te
  data_files:
  - path: train/te.json.gz
    split: train
  - path: test/te.json.gz
    split: test
  - path: validation/te.json.gz
    split: validation
- config_name: cy
  data_files:
  - path: train/cy.json.gz
    split: train
  - path: test/cy.json.gz
    split: test
  - path: validation/cy.json.gz
    split: validation
- config_name: he
  data_files:
  - path: train/he.json.gz
    split: train
  - path: test/he.json.gz
    split: test
  - path: validation/he.json.gz
    split: validation
- config_name: de
  data_files:
  - path: train/de.json.gz
    split: train
  - path: test/de.json.gz
    split: test
  - path: validation/de.json.gz
    split: validation
- config_name: af
  data_files:
  - path: train/af.json.gz
    split: train
  - path: test/af.json.gz
    split: test
  - path: validation/af.json.gz
    split: validation
- config_name: ml
  data_files:
  - path: train/ml.json.gz
    split: train
  - path: test/ml.json.gz
    split: test
  - path: validation/ml.json.gz
    split: validation
- config_name: sl
  data_files:
  - path: train/sl.json.gz
    split: train
  - path: test/sl.json.gz
    split: test
  - path: validation/sl.json.gz
    split: validation
- config_name: vi
  data_files:
  - path: train/vi.json.gz
    split: train
  - path: test/vi.json.gz
    split: test
  - path: validation/vi.json.gz
    split: validation
- config_name: mn
  data_files:
  - path: train/mn.json.gz
    split: train
  - path: test/mn.json.gz
    split: test
  - path: validation/mn.json.gz
    split: validation
- config_name: tl
  data_files:
  - path: train/tl.json.gz
    split: train
  - path: test/tl.json.gz
    split: test
  - path: validation/tl.json.gz
    split: validation
- config_name: it
  data_files:
  - path: train/it.json.gz
    split: train
  - path: test/it.json.gz
    split: test
  - path: validation/it.json.gz
    split: validation
- config_name: jv
  data_files:
  - path: train/jv.json.gz
    split: train
  - path: test/jv.json.gz
    split: test
  - path: validation/jv.json.gz
    split: validation
- config_name: sq
  data_files:
  - path: train/sq.json.gz
    split: train
  - path: test/sq.json.gz
    split: test
  - path: validation/sq.json.gz
    split: validation
- config_name: fa
  data_files:
  - path: train/fa.json.gz
    split: train
  - path: test/fa.json.gz
    split: test
  - path: validation/fa.json.gz
    split: validation
- config_name: nb
  data_files:
  - path: train/nb.json.gz
    split: train
  - path: test/nb.json.gz
    split: test
  - path: validation/nb.json.gz
    split: validation
- config_name: km
  data_files:
  - path: train/km.json.gz
    split: train
  - path: test/km.json.gz
    split: test
  - path: validation/km.json.gz
    split: validation
- config_name: th
  data_files:
  - path: train/th.json.gz
    split: train
  - path: test/th.json.gz
    split: test
  - path: validation/th.json.gz
    split: validation
- config_name: ja
  data_files:
  - path: train/ja.json.gz
    split: train
  - path: test/ja.json.gz
    split: test
  - path: validation/ja.json.gz
    split: validation
- config_name: hi
  data_files:
  - path: train/hi.json.gz
    split: train
  - path: test/hi.json.gz
    split: test
  - path: validation/hi.json.gz
    split: validation
- config_name: id
  data_files:
  - path: train/id.json.gz
    split: train
  - path: test/id.json.gz
    split: test
  - path: validation/id.json.gz
    split: validation
- config_name: kn
  data_files:
  - path: train/kn.json.gz
    split: train
  - path: test/kn.json.gz
    split: test
  - path: validation/kn.json.gz
    split: validation
- config_name: fi
  data_files:
  - path: train/fi.json.gz
    split: train
  - path: test/fi.json.gz
    split: test
  - path: validation/fi.json.gz
    split: validation
- config_name: ur
  data_files:
  - path: train/ur.json.gz
    split: train
  - path: test/ur.json.gz
    split: test
  - path: validation/ur.json.gz
    split: validation
- config_name: my
  data_files:
  - path: train/my.json.gz
    split: train
  - path: test/my.json.gz
    split: test
  - path: validation/my.json.gz
    split: validation
- config_name: lv
  data_files:
  - path: train/lv.json.gz
    split: train
  - path: test/lv.json.gz
    split: test
  - path: validation/lv.json.gz
    split: validation
- config_name: fr
  data_files:
  - path: train/fr.json.gz
    split: train
  - path: test/fr.json.gz
    split: test
  - path: validation/fr.json.gz
    split: validation
- config_name: ko
  data_files:
  - path: train/ko.json.gz
    split: train
  - path: test/ko.json.gz
    split: test
  - path: validation/ko.json.gz
    split: validation
- config_name: sw
  data_files:
  - path: train/sw.json.gz
    split: train
  - path: test/sw.json.gz
    split: test
  - path: validation/sw.json.gz
    split: validation
- config_name: sv
  data_files:
  - path: train/sv.json.gz
    split: train
  - path: test/sv.json.gz
    split: test
  - path: validation/sv.json.gz
    split: validation
- config_name: nl
  data_files:
  - path: train/nl.json.gz
    split: train
  - path: test/nl.json.gz
    split: test
  - path: validation/nl.json.gz
    split: validation
- config_name: da
  data_files:
  - path: train/da.json.gz
    split: train
  - path: test/da.json.gz
    split: test
  - path: validation/da.json.gz
    split: validation
- config_name: ar
  data_files:
  - path: train/ar.json.gz
    split: train
  - path: test/ar.json.gz
    split: test
  - path: validation/ar.json.gz
    split: validation
- config_name: ms
  data_files:
  - path: train/ms.json.gz
    split: train
  - path: test/ms.json.gz
    split: test
  - path: validation/ms.json.gz
    split: validation
- config_name: en
  data_files:
  - path: train/en.json.gz
    split: train
  - path: test/en.json.gz
    split: test
  - path: validation/en.json.gz
    split: validation
- config_name: am
  data_files:
  - path: train/am.json.gz
    split: train
  - path: test/am.json.gz
    split: test
  - path: validation/am.json.gz
    split: validation
- config_name: pt
  data_files:
  - path: train/pt.json.gz
    split: train
  - path: test/pt.json.gz
    split: test
  - path: validation/pt.json.gz
    split: validation
- config_name: ka
  data_files:
  - path: train/ka.json.gz
    split: train
  - path: test/ka.json.gz
    split: test
  - path: validation/ka.json.gz
    split: validation
- config_name: ro
  data_files:
  - path: train/ro.json.gz
    split: train
  - path: test/ro.json.gz
    split: test
  - path: validation/ro.json.gz
    split: validation
- config_name: tr
  data_files:
  - path: train/tr.json.gz
    split: train
  - path: test/tr.json.gz
    split: test
  - path: validation/tr.json.gz
    split: validation
- config_name: hu
  data_files:
  - path: train/hu.json.gz
    split: train
  - path: test/hu.json.gz
    split: test
  - path: validation/hu.json.gz
    split: validation
- config_name: zh-TW
  data_files:
  - path: train/zh-TW.json.gz
    split: train
  - path: test/zh-TW.json.gz
    split: test
  - path: validation/zh-TW.json.gz
    split: validation
- config_name: bn
  data_files:
  - path: train/bn.json.gz
    split: train
  - path: test/bn.json.gz
    split: test
  - path: validation/bn.json.gz
    split: validation
- config_name: hy
  data_files:
  - path: train/hy.json.gz
    split: train
  - path: test/hy.json.gz
    split: test
  - path: validation/hy.json.gz
    split: validation
- config_name: es
  data_files:
  - path: train/es.json.gz
    split: train
  - path: test/es.json.gz
    split: test
  - path: validation/es.json.gz
    split: validation
- config_name: az
  data_files:
  - path: train/az.json.gz
    split: train
  - path: test/az.json.gz
    split: test
  - path: validation/az.json.gz
    split: validation
tags:
- mteb
- text
---
<!-- adapted from https://github.com/huggingface/huggingface_hub/blob/v0.30.2/src/huggingface_hub/templates/datasetcard_template.md -->

<div align="center" style="padding: 40px 20px; background-color: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); max-width: 600px; margin: 0 auto;">
  <h1 style="font-size: 3.5rem; color: #1a1a1a; margin: 0 0 20px 0; letter-spacing: 2px; font-weight: 700;">MassiveIntentClassification</h1>
  <div style="font-size: 1.5rem; color: #4a4a4a; margin-bottom: 5px; font-weight: 300;">An <a href="https://github.com/embeddings-benchmark/mteb" style="color: #2c5282; font-weight: 600; text-decoration: none;" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">MTEB</a> dataset</div>
  <div style="font-size: 0.9rem; color: #2c5282; margin-top: 10px;">Massive Text Embedding Benchmark</div>
</div>

MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages

|               |                                             |
|---------------|---------------------------------------------|
| Task category | t2c                              |
| Domains       | Spoken                               |
| Reference     | https://arxiv.org/abs/2204.08582 |


## How to evaluate on this task

You can evaluate an embedding model on this dataset using the following code:

```python
import mteb

task = mteb.get_tasks(["MassiveIntentClassification"])
evaluator = mteb.MTEB(task)

model = mteb.get_model(YOUR_MODEL)
evaluator.run(model)
```

<!-- Datasets want link to arxiv in readme to autolink dataset with paper -->
To learn more about how to run models on `mteb` task check out the [GitHub repitory](https://github.com/embeddings-benchmark/mteb). 

## Citation

If you use this dataset, please cite the dataset as well as [mteb](https://github.com/embeddings-benchmark/mteb), as this dataset likely includes additional processing as a part of the [MMTEB Contribution](https://github.com/embeddings-benchmark/mteb/tree/main/docs/mmteb).

```bibtex

@misc{fitzgerald2022massive,
  archiveprefix = {arXiv},
  author = {Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
  eprint = {2204.08582},
  primaryclass = {cs.CL},
  title = {MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages},
  year = {2022},
}


@article{enevoldsen2025mmtebmassivemultilingualtext,
  title={MMTEB: Massive Multilingual Text Embedding Benchmark},
  author={Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2502.13595},
  year={2025},
  url={https://arxiv.org/abs/2502.13595},
  doi = {10.48550/arXiv.2502.13595},
}

@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},
  year = {2022}
  url = {https://arxiv.org/abs/2210.07316},
  doi = {10.48550/ARXIV.2210.07316},
}
```

# Dataset Statistics
<details>
  <summary> Dataset Statistics</summary>

The following code contains the descriptive statistics from the task. These can also be obtained using:

```python
import mteb

task = mteb.get_task("MassiveIntentClassification")

desc_stats = task.metadata.descriptive_stats
```

```json
{
    "validation": {
        "num_samples": 103683,
        "number_of_characters": 3583467,
        "number_texts_intersect_with_train": 5457,
        "min_text_length": 1,
        "average_text_length": 34.56176036573016,
        "max_text_length": 224,
        "unique_text": 102325,
        "unique_labels": 59,
        "labels": {
            "iot_hue_lightoff": {
                "count": 867
            },
            "iot_hue_lightdim": {
                "count": 867
            },
            "iot_cleaning": {
                "count": 969
            },
            "general_quirky": {
                "count": 5355
            },
            "takeaway_query": {
                "count": 1224
            },
            "play_music": {
                "count": 6273
            },
            "music_query": {
                "count": 1530
            },
            "weather_query": {
                "count": 6426
            },
            "music_settings": {
                "count": 408
            },
            "audio_volume_down": {
                "count": 408
            },
            "datetime_query": {
                "count": 3264
            },
            "general_greet": {
                "count": 102
            },
            "alarm_set": {
                "count": 1581
            },
            "audio_volume_up": {
                "count": 612
            },
            "alarm_query": {
                "count": 969
            },
            "news_query": {
                "count": 4182
            },
            "iot_hue_lighton": {
                "count": 255
            },
            "iot_wemo_off": {
                "count": 255
            },
            "iot_hue_lightchange": {
                "count": 1122
            },
            "audio_volume_mute": {
                "count": 765
            },
            "alarm_remove": {
                "count": 714
            },
            "general_joke": {
                "count": 765
            },
            "datetime_convert": {
                "count": 459
            },
            "iot_wemo_on": {
                "count": 357
            },
            "iot_hue_lightup": {
                "count": 612
            },
            "iot_coffee": {
                "count": 714
            },
            "social_post": {
                "count": 2550
            },
            "music_dislikeness": {
                "count": 102
            },
            "cooking_recipe": {
                "count": 2091
            },
            "takeaway_order": {
                "count": 1020
            },
            "music_likeness": {
                "count": 816
            },
            "calendar_query": {
                "count": 5202
            },
            "qa_stock": {
                "count": 1224
            },
            "qa_factoid": {
                "count": 4590
            },
            "calendar_set": {
                "count": 6681
            },
            "recommendation_events": {
                "count": 1326
            },
            "cooking_query": {
                "count": 102
            },
            "calendar_remove": {
                "count": 2397
            },
            "email_sendemail": {
                "count": 3213
            },
            "play_radio": {
                "count": 2346
            },
            "play_audiobook": {
                "count": 1785
            },
            "play_game": {
                "count": 1122
            },
            "lists_query": {
                "count": 2550
            },
            "lists_remove": {
                "count": 1887
            },
            "lists_createoradd": {
                "count": 1275
            },
            "email_addcontact": {
                "count": 255
            },
            "play_podcasts": {
                "count": 1734
            },
            "recommendation_movies": {
                "count": 612
            },
            "recommendation_locations": {
                "count": 1581
            },
            "transport_ticket": {
                "count": 1275
            },
            "transport_query": {
                "count": 1836
            },
            "transport_taxi": {
                "count": 1377
            },
            "transport_traffic": {
                "count": 1122
            },
            "qa_definition": {
                "count": 2805
            },
            "qa_currency": {
                "count": 1632
            },
            "qa_maths": {
                "count": 663
            },
            "social_query": {
                "count": 918
            },
            "email_query": {
                "count": 3723
            },
            "email_querycontact": {
                "count": 816
            }
        }
    },
    "test": {
        "num_samples": 151674,
        "number_of_characters": 5230011,
        "number_texts_intersect_with_train": 7273,
        "min_text_length": 1,
        "average_text_length": 34.48192175323391,
        "max_text_length": 495,
        "unique_text": 148972,
        "unique_labels": 59,
        "labels": {
            "alarm_set": {
                "count": 2091
            },
            "audio_volume_mute": {
                "count": 1632
            },
            "iot_hue_lightchange": {
                "count": 1836
            },
            "iot_hue_lighton": {
                "count": 153
            },
            "iot_hue_lightoff": {
                "count": 2193
            },
            "iot_cleaning": {
                "count": 1326
            },
            "general_quirky": {
                "count": 8619
            },
            "general_greet": {
                "count": 51
            },
            "datetime_query": {
                "count": 4488
            },
            "datetime_convert": {
                "count": 765
            },
            "alarm_remove": {
                "count": 1071
            },
            "alarm_query": {
                "count": 1734
            },
            "music_likeness": {
                "count": 1836
            },
            "iot_hue_lightup": {
                "count": 1377
            },
            "takeaway_order": {
                "count": 1122
            },
            "weather_query": {
                "count": 7956
            },
            "general_joke": {
                "count": 969
            },
            "play_music": {
                "count": 8976
            },
            "iot_hue_lightdim": {
                "count": 1071
            },
            "takeaway_query": {
                "count": 1785
            },
            "news_query": {
                "count": 6324
            },
            "audio_volume_up": {
                "count": 663
            },
            "iot_wemo_off": {
                "count": 918
            },
            "iot_wemo_on": {
                "count": 510
            },
            "iot_coffee": {
                "count": 1836
            },
            "music_query": {
                "count": 1785
            },
            "audio_volume_down": {
                "count": 561
            },
            "audio_volume_other": {
                "count": 306
            },
            "music_dislikeness": {
                "count": 204
            },
            "music_settings": {
                "count": 306
            },
            "recommendation_events": {
                "count": 2193
            },
            "qa_stock": {
                "count": 1326
            },
            "calendar_set": {
                "count": 10659
            },
            "play_audiobook": {
                "count": 2091
            },
            "social_query": {
                "count": 1275
            },
            "qa_factoid": {
                "count": 7191
            },
            "transport_ticket": {
                "count": 1785
            },
            "recommendation_locations": {
                "count": 1581
            },
            "calendar_query": {
                "count": 6426
            },
            "recommendation_movies": {
                "count": 1020
            },
            "transport_query": {
                "count": 2601
            },
            "cooking_recipe": {
                "count": 3672
            },
            "play_game": {
                "count": 1785
            },
            "calendar_remove": {
                "count": 3417
            },
            "email_query": {
                "count": 6069
            },
            "email_sendemail": {
                "count": 5814
            },
            "play_radio": {
                "count": 3672
            },
            "play_podcasts": {
                "count": 3213
            },
            "lists_query": {
                "count": 2601
            },
            "lists_remove": {
                "count": 2652
            },
            "lists_createoradd": {
                "count": 1989
            },
            "transport_taxi": {
                "count": 1173
            },
            "transport_traffic": {
                "count": 765
            },
            "qa_definition": {
                "count": 2907
            },
            "qa_maths": {
                "count": 1275
            },
            "social_post": {
                "count": 4131
            },
            "qa_currency": {
                "count": 1989
            },
            "email_addcontact": {
                "count": 612
            },
            "email_querycontact": {
                "count": 1326
            }
        }
    },
    "train": {
        "num_samples": 587214,
        "number_of_characters": 20507758,
        "number_texts_intersect_with_train": null,
        "min_text_length": 1,
        "average_text_length": 34.92382334208653,
        "max_text_length": 295,
        "unique_text": 565055,
        "unique_labels": 60,
        "labels": {
            "alarm_set": {
                "count": 9282
            },
            "audio_volume_mute": {
                "count": 5610
            },
            "iot_hue_lightchange": {
                "count": 6375
            },
            "iot_hue_lightoff": {
                "count": 7803
            },
            "iot_hue_lightdim": {
                "count": 3876
            },
            "iot_cleaning": {
                "count": 4743
            },
            "calendar_query": {
                "count": 28866
            },
            "play_music": {
                "count": 32589
            },
            "general_quirky": {
                "count": 28305
            },
            "general_greet": {
                "count": 1275
            },
            "datetime_query": {
                "count": 17850
            },
            "datetime_convert": {
                "count": 2652
            },
            "takeaway_query": {
                "count": 6222
            },
            "alarm_remove": {
                "count": 3978
            },
            "alarm_query": {
                "count": 6630
            },
            "news_query": {
                "count": 25653
            },
            "music_likeness": {
                "count": 5763
            },
            "music_query": {
                "count": 7854
            },
            "iot_hue_lightup": {
                "count": 3876
            },
            "takeaway_order": {
                "count": 6885
            },
            "weather_query": {
                "count": 29223
            },
            "music_settings": {
                "count": 2601
            },
            "general_joke": {
                "count": 3672
            },
            "music_dislikeness": {
                "count": 714
            },
            "audio_volume_other": {
                "count": 918
            },
            "iot_coffee": {
                "count": 6324
            },
            "audio_volume_up": {
                "count": 5610
            },
            "iot_wemo_on": {
                "count": 2448
            },
            "iot_hue_lighton": {
                "count": 1122
            },
            "iot_wemo_off": {
                "count": 2652
            },
            "audio_volume_down": {
                "count": 2652
            },
            "qa_stock": {
                "count": 7752
            },
            "play_radio": {
                "count": 14433
            },
            "recommendation_locations": {
                "count": 8823
            },
            "qa_factoid": {
                "count": 27744
            },
            "calendar_set": {
                "count": 41310
            },
            "play_audiobook": {
                "count": 7650
            },
            "play_podcasts": {
                "count": 9843
            },
            "social_query": {
                "count": 5508
            },
            "transport_query": {
                "count": 11577
            },
            "email_sendemail": {
                "count": 18054
            },
            "recommendation_movies": {
                "count": 3570
            },
            "lists_query": {
                "count": 10098
            },
            "play_game": {
                "count": 5712
            },
            "transport_ticket": {
                "count": 6477
            },
            "recommendation_events": {
                "count": 9690
            },
            "email_query": {
                "count": 21318
            },
            "transport_traffic": {
                "count": 5967
            },
            "cooking_query": {
                "count": 204
            },
            "qa_definition": {
                "count": 13617
            },
            "calendar_remove": {
                "count": 15912
            },
            "lists_remove": {
                "count": 8364
            },
            "cooking_recipe": {
                "count": 10557
            },
            "email_querycontact": {
                "count": 6477
            },
            "lists_createoradd": {
                "count": 9027
            },
            "transport_taxi": {
                "count": 5100
            },
            "qa_maths": {
                "count": 3978
            },
            "social_post": {
                "count": 14433
            },
            "qa_currency": {
                "count": 7242
            },
            "email_addcontact": {
                "count": 2754
            }
        }
    }
}
```

</details>

---
*This dataset card was automatically generated using [MTEB](https://github.com/embeddings-benchmark/mteb)*
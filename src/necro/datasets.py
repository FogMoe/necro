"""下载评测切片，不把验证数据混入训练。"""

import json
import time
from pathlib import Path

import httpx

from necro.config import MODEL_ID

# 类别来自 AmazonScience/massive 的 _INTENTS；描述仅解释原有类别。
INTENTS = {
    "datetime_query": "Ask the current date or time",
    "iot_hue_lightchange": "Change the color of lights",
    "transport_ticket": "Book or buy a transport ticket",
    "takeaway_query": "Ask about takeaway food or delivery",
    "qa_stock": "Ask stock prices or stock market information",
    "general_greet": "Greet the assistant",
    "recommendation_events": "Ask for events to attend",
    "music_dislikeness": "Express dislike for music",
    "iot_wemo_off": "Switch off a smart plug or connected appliance",
    "cooking_recipe": "Ask for a recipe or cooking instructions",
    "qa_currency": "Ask about currencies or exchange rates",
    "transport_traffic": "Ask about road traffic",
    "general_quirky": "Casual conversation with the assistant",
    "weather_query": "Ask about the weather",
    "audio_volume_up": "Increase audio volume",
    "email_addcontact": "Add an email contact",
    "takeaway_order": "Order takeaway food",
    "email_querycontact": "Ask for contact details",
    "iot_hue_lightup": "Make lights brighter",
    "recommendation_locations": "Recommend places to visit or nearby businesses",
    "play_audiobook": "Play an audiobook",
    "lists_createoradd": "Create a list or add items to a list",
    "news_query": "Ask about news",
    "alarm_query": "Ask about existing alarms",
    "iot_wemo_on": "Switch on a smart plug or connected appliance",
    "general_joke": "Ask for a joke",
    "qa_definition": "Ask for a definition or explanation of a term",
    "social_query": "Read or check social media",
    "music_settings": "Change music playback settings such as shuffle or repeat",
    "audio_volume_other": "Set or ask about a specific audio volume",
    "calendar_remove": "Delete a calendar event",
    "iot_hue_lightdim": "Dim the lights",
    "calendar_query": "Ask about calendar events or appointments",
    "email_sendemail": "Send an email",
    "iot_cleaning": "Control a cleaning robot",
    "audio_volume_down": "Decrease audio volume",
    "play_radio": "Play a radio station",
    "cooking_query": "Ask about a cooking timer or cooking duration",
    "datetime_convert": "Convert time between time zones",
    "qa_maths": "Ask a mathematics question",
    "iot_hue_lightoff": "Turn lights off",
    "iot_hue_lighton": "Turn lights on",
    "transport_query": "Ask about transport routes or schedules",
    "music_likeness": "Express liking for music",
    "email_query": "Read or check emails",
    "play_music": "Play music or a song",
    "audio_volume_mute": "Mute audio",
    "social_post": "Publish a social media post",
    "alarm_set": "Set an alarm",
    "qa_factoid": "Ask a factual general-knowledge question",
    "calendar_set": "Create or update a calendar event or reminder",
    "play_game": "Play a game",
    "alarm_remove": "Delete an alarm",
    "lists_remove": "Delete a list or remove an item from a list",
    "transport_taxi": "Book or request a taxi",
    "recommendation_movies": "Ask for movie recommendations",
    "iot_coffee": "Control a coffee machine",
    "music_query": "Ask information about music, artists or a playing song",
    "play_podcasts": "Play a podcast",
    "lists_query": "Read or check a list",
}


def fetch_rows(
    client: httpx.Client,
    dataset: str,
    config: str,
    count: int,
    split: str = "validation",
    start: int = 0,
):
    records = []
    for offset in range(start, start + count, 100):
        for attempt in range(4):
            response = client.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "offset": offset,
                    "length": min(100, count - (offset - start)),
                },
            )
            if response.status_code in {429, 502, 503, 504} and attempt < 3:
                time.sleep(2**attempt)
                continue
            break
        response.raise_for_status()
        payload = response.json()
        for row in payload["rows"]:
            if row.get("truncated_cells"):
                raise ValueError(f"数据行被截断：{dataset}/{config}/{row['row_idx']}")
            records.append(row)
    if len(records) != count:
        raise ValueError(f"{dataset}/{config} 只取得 {len(records)}/{count} 行。")
    return records


def prepare_public_eval(
    output: Path, per_language: int = 100, split: str = "validation", start: int = 0
):
    if not 1 <= per_language <= 1000:
        raise ValueError("per-language 必须在 1–1000 之间。")
    examples = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for language in ("en", "zh"):
            dataset = "facebook/xnli"
            for item in fetch_rows(client, dataset, language, per_language, split, start):
                row = item["row"]
                criteria = {
                    "entailment": "The premise supports the hypothesis.",
                    "neutral": "The premise neither supports nor contradicts the hypothesis.",
                    "contradiction": "The premise contradicts the hypothesis.",
                }
                instruction = (
                    "根据 premise 判断 hypothesis 是否成立，仅依据给定前提，不补充外部事实。"
                    if language == "zh"
                    else "Does the premise support, contradict, or leave the hypothesis "
                    "undetermined? "
                    "Use only the given premise."
                )
                examples.append(
                    {
                        "id": f"xnli/{language}/{split}/{item['row_idx']}",
                        "group_id": f"xnli/{split}/{item['row_idx']}",
                        "source": dataset,
                        "language": language,
                        "request": {
                            "model": MODEL_ID,
                            "state": {"premise": row["premise"], "hypothesis": row["hypothesis"]},
                            "questions": {
                                "decision": {
                                    "type": "choice",
                                    "instructions": instruction,
                                    "criteria": criteria,
                                }
                            },
                        },
                        "expected": {"decision": list(criteria)[row["label"]]},
                    }
                )
            # 官方脚本型数据集无法由 Viewer 读取，使用 MTEB 的同源列式副本。
            dataset = "mteb/amazon_massive_intent"
            config = "zh-CN" if language == "zh" else "en"
            for item in fetch_rows(client, dataset, config, per_language, split, start):
                row = item["row"]
                if row["label"] not in INTENTS:
                    raise ValueError(f"未知 MASSIVE 类别：{row['label']}")
                examples.append(
                    {
                        "id": f"massive/{language}/{split}/{row['id']}",
                        "group_id": f"massive/{row['id']}",
                        "source": dataset,
                        "language": language,
                        "request": {
                            "model": MODEL_ID,
                            "state": row["text"],
                            "questions": {
                                "decision": {
                                    "type": "choice",
                                    "criteria": INTENTS,
                                    "instructions": "选择用户最主要的意图。"
                                    if language == "zh"
                                    else "Select the user's primary intent.",
                                }
                            },
                        },
                        "expected": {"decision": row["label"]},
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples),
        encoding="utf-8",
    )
    return {"examples": len(examples), "output": str(output), "split": split, "start": start}

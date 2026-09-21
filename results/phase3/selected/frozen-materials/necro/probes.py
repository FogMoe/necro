"""评测专用的结构与边界样例；不加入训练。"""

import json
from pathlib import Path

from necro.config import MODEL_ID
from necro.training_data import write_jsonl


def heldout_probes():
    rows = []
    for language in ("zh", "en"):
        for index in range(16):
            due = 50 + index * 7
            paid = due + [-1, 0, 1, 2][index % 4]
            settled = index % 4 != 2
            state = {
                "amount_paid": paid,
                "amount_due": due,
                "status": "settled" if settled else "pending",
            }
            question = {
                "type": "noul",
                "instructions": "仅当 amount_paid 不小于 amount_due 且 status 等于 settled，"
                "账单才已结清。账单是否已结清？"
                if language == "zh"
                else "The bill is cleared only if amount_paid is at least amount_due and status "
                "equals settled. Is the bill cleared?",
            }
            rows.append((language, f"bill/{index}", state, question, paid >= due and settled))
        for index in range(8):
            failed = [0, 1, 2, 3, 4, 5, 8, 0][index]
            question = {
                "type": "score",
                "instructions": "只按 failed_checks 的数量评估故障等级。"
                if language == "zh"
                else "Rate the incident using only the number of failed_checks.",
                "criteria": ["0 项失败", "1–2 项失败", "3–4 项失败", "至少 5 项失败"]
                if language == "zh"
                else ["0 failures", "1–2 failures", "3–4 failures", "At least 5 failures"],
            }
            rows.append(
                (
                    language,
                    f"severity/{index}",
                    {"failed_checks": failed, "ignored_notes": "ordinary scheduled inspection"},
                    question,
                    0 if failed == 0 else 1 if failed <= 2 else 2 if failed <= 4 else 3,
                )
            )
        for count in (8, 27, 60, 255):
            for side, index in enumerate((0, count - 1)):
                key = f"route_{index:03d}"
                question = {
                    "type": "choice",
                    "instructions": "选择名字与 state.selected_route 完全相同的选项。"
                    if language == "zh"
                    else "Select the option whose key exactly matches state.selected_route.",
                    "criteria": {f"route_{i:03d}": f"Destination {i:03d}" for i in range(count)},
                }
                rows.append(
                    (language, f"size/{count}/{side}", {"selected_route": key}, question, key)
                )
    return [
        {
            "id": f"probe/{language}/{key}",
            "group_id": f"probe/{key}",
            "source": "constructed-heldout-probes",
            "language": language,
            "request": {"model": MODEL_ID, "state": state, "questions": {"decision": question}},
            "expected": {"decision": expected},
        }
        for language, key, state, question, expected in rows
    ]


if __name__ == "__main__":
    path = Path("data/improvement/probes-sealed.jsonl")
    if path.exists():
        raise ValueError("评测文件已存在，不覆盖。")
    rows = heldout_probes()
    write_jsonl(path, rows)
    print(json.dumps({"examples": len(rows), "output": str(path)}))

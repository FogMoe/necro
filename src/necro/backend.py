import json
import logging
import string
import threading
import time
from pathlib import Path

from necro.config import MODEL_ID, Settings
from necro.engine import InputError, ScoredTask, Task, prompt_fingerprint
from necro.schema import Noul

logger = logging.getLogger(__name__)


def single_token_labels(tokenizer, count: int = 26) -> tuple[list[str], list[int]]:
    """短选项使用无前缀歧义的 A–Z，大选项走完整序列评分。"""
    labels, ids = [], []
    seen = set(tokenizer.all_special_ids)
    for label in string.ascii_uppercase:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in seen:
            labels.append(label)
            ids.append(encoded[0])
            seen.add(encoded[0])
            if len(labels) == count:
                return labels, ids
    raise RuntimeError(f"词表中只有 {len(labels)} 个可用标签，无法支持 {count} 个选项。")


def numeric_labels(count: int) -> list[str]:
    """等长、无前导零、互不为前缀的编号，防止短编号占据不完整概率。"""
    if not 1 <= count <= 255:
        raise ValueError("候选数量必须在 1–255 之间。")
    width = 1
    while 9 * 10 ** (width - 1) < count:
        width += 1
    start = 10 ** (width - 1)
    return [str(start + i) for i in range(count)]


def verify_answer_boundary(tokenizer, prompt: str, mapping: dict[str, int]):
    """单独编码是单 token 仍不够；必须验证它在实际答案边界的编码。"""
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    for label, token_id in mapping.items():
        actual = tokenizer.encode(prompt + label, add_special_tokens=False)
        if actual != [*prefix, token_id]:
            raise RuntimeError(f"标签 {label!r} 在答案边界不是一个独立 token。")


class TransformersScorer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.last_candidate_masses: list[float] = []
        self.load_seconds = 0.0
        self.model_id = MODEL_ID
        self.adapter_contract = None
        self.model_contract = None
        model_manifest = Path(settings.checkpoint) / "necro_model.json"
        if model_manifest.is_file():
            contract = json.loads(model_manifest.read_text(encoding="utf-8"))
            if contract["prompt_sha256"] != prompt_fingerprint():
                raise ValueError("合并模型与当前提示代码不一致，不能直接加载。")
            self.model_contract = contract
            self.model_id = contract["model_id"]
            if settings.adapter:
                raise ValueError("已合并的 Necro 模型不能再次叠加 adapter。")
        if settings.adapter:
            path = Path(settings.adapter) / "necro_adapter.json"
            contract = json.loads(path.read_text(encoding="utf-8"))
            if contract["checkpoint"] != settings.checkpoint:
                raise ValueError("Adapter 的基模与 NECRO_MODEL 不一致。")
            if contract["prompt_sha256"] != prompt_fingerprint():
                raise ValueError("Adapter 与当前提示代码不一致，不能直接加载。")
            self.adapter_contract = contract
            self.model_id = contract["model_id"]

    def load(self):
        with self.lock:
            self._load()

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        started = time.perf_counter()
        device = self.settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请安装支持显卡的 PyTorch 或选择 NECRO_DEVICE=cpu。")
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        logger.info("加载 %s 到 %s", self.settings.checkpoint, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.checkpoint,
            revision=self.adapter_contract["revision"] if self.adapter_contract else None,
            padding_side="left",
        )
        self.labels, self.label_ids = single_token_labels(self.tokenizer)
        self.noul_labels = ["Yes", "No"]
        noul_ids = [
            self.tokenizer.encode(label, add_special_tokens=False) for label in self.noul_labels
        ]
        if any(len(ids) != 1 for ids in noul_ids):
            raise RuntimeError("Noul 的 Yes/No 标签必须是单 token。")
        self.noul_ids = [ids[0] for ids in noul_ids]
        digits = {
            str(i): self.tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)
        }
        if any(len(ids) != 1 for ids in digits.values()):
            raise RuntimeError("数字标签需要每个数字编码为单 token。")
        self.digit_ids = {key: value[0] for key, value in digits.items()}
        answer_prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Return one option label."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        verify_answer_boundary(
            self.tokenizer,
            answer_prefix,
            {
                **dict(zip(self.labels, self.label_ids, strict=True)),
                **dict(zip(self.noul_labels, self.noul_ids, strict=True)),
                **self.digit_ids,
            },
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                self.settings.checkpoint,
                revision=self.adapter_contract["revision"] if self.adapter_contract else None,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )
        if self.settings.adapter:
            from peft import PeftModel

            logger.info("加载 LoRA 并在内存中合并；基础权重文件保持原样。")
            adapted = PeftModel.from_pretrained(self.model, self.settings.adapter)
            self.model = adapted.merge_and_unload(safe_merge=True).eval()
        # Match the frozen parameters produced by PEFT merging on every inference path.
        self.model.requires_grad_(False)
        self.load_seconds = time.perf_counter() - started
        logger.info(
            "模型加载完成，耗时 %.2f 秒；已验证 %d 个单 token 标签。",
            self.load_seconds,
            len(self.labels),
        )

    def score(self, tasks: list[Task]) -> list[ScoredTask]:
        with self.lock:
            return self._score(tasks)

    def _score(self, tasks: list[Task]) -> list[ScoredTask]:
        import torch

        self._load()
        rows = []
        candidate_token_ids = []
        sequence_tasks = {}
        request_budgets = {}
        for task in tasks:
            if task.request_group not in request_budgets:
                request_budgets[task.request_group] = len(
                    self.tokenizer.encode(
                        json.dumps(task.state, ensure_ascii=False),
                        add_special_tokens=False,
                    )
                )
            request_budgets[task.request_group] += len(
                self.tokenizer.encode(
                    task.question.model_dump_json(),
                    add_special_tokens=False,
                )
            )
            if request_budgets[task.request_group] > self.settings.max_request_tokens:
                raise InputError(
                    f"请求超过 {self.settings.max_request_tokens} tokens；请拆分问题。"
                )
            if isinstance(task.question, Noul):
                labels, token_ids = self.noul_labels, self.noul_ids
            elif len(task.keys) > len(self.labels):
                labels, token_ids = numeric_labels(len(task.keys)), None
                sequence_tasks[len(rows)] = labels
            else:
                labels = self.labels[: len(task.keys)]
                token_ids = self.label_ids[: len(task.keys)]
            candidate_token_ids.append(token_ids)
            messages = task.messages(labels)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) > self.settings.max_sequence_tokens:
                raise InputError(
                    f"单题输入 {len(ids)} tokens，超过 {self.settings.max_sequence_tokens}；"
                    "请缩短 state 或 criteria。"
                )
            rows.append(ids)
        # 按长度分批减少 padding；结果再还原到原始问题顺序。
        order = sorted(
            (i for i in range(len(rows)) if i not in sequence_tasks), key=lambda i: len(rows[i])
        )
        results: list[ScoredTask | None] = [None] * len(rows)
        position = 0
        with torch.inference_mode():
            while position < len(order):
                indices = [order[position]]
                position += 1
                while position < len(order) and len(indices) < self.settings.batch_size:
                    next_index = order[position]
                    if len(rows[next_index]) * (len(indices) + 1) > self.settings.batch_tokens:
                        break
                    indices.append(next_index)
                    position += 1
                padded = self.tokenizer.pad(
                    {"input_ids": [rows[i] for i in indices]},
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(self.device)
                output = self.model(**padded, logits_to_keep=1, use_cache=False)
                logits = output.logits[:, -1, :].float()
                normalizers = torch.logsumexp(logits, dim=-1)
                for batch_index, task_index in enumerate(indices):
                    selected_ids = torch.tensor(candidate_token_ids[task_index], device=self.device)
                    scores = logits[batch_index].index_select(0, selected_ids)
                    temperature = self.settings.temperature_for(tasks[task_index].question.type)
                    probabilities = torch.softmax(scores / temperature, dim=-1)
                    mass = torch.exp(torch.logsumexp(scores, dim=-1) - normalizers[batch_index])
                    results[task_index] = ScoredTask(
                        probabilities=probabilities.cpu().tolist(),
                        input_tokens=len(rows[task_index]),
                        candidate_mass=float(mass),
                    )
                del output, logits, padded
            for task_index, labels in sequence_tasks.items():
                results[task_index] = self._score_sequence(rows[task_index], labels)
        self.last_candidate_masses = [result.candidate_mass for result in results]
        return results

    def _score_sequence(self, input_ids: list[int], labels: list[str]) -> ScoredTask:
        """复用输入缓存，对数字编号的完整路径求概率；中间节点不重新归一化。"""
        import torch

        length = len(input_ids)
        output = self.model(
            input_ids=torch.tensor([input_ids], device=self.device),
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        log_probabilities = output.logits[:, -1, :].float().log_softmax(-1)
        prefixes = [""]
        totals = torch.zeros(1, device=self.device)
        extra_tokens = 0
        width = len(labels[0])
        for depth in range(width):
            children = sorted({label[: depth + 1] for label in labels})
            parent_lookup = {prefix: i for i, prefix in enumerate(prefixes)}
            parents = torch.tensor(
                [parent_lookup[child[:-1]] for child in children],
                device=self.device,
            )
            tokens = torch.tensor(
                [self.digit_ids[child[-1]] for child in children], device=self.device
            )
            totals = totals[parents] + log_probabilities[parents, tokens]
            prefixes = children
            if depth + 1 < width:
                # reorder_cache 同时复制 Qwen 混合结构的 KV、卷积及循环状态。
                # 重复父索引让兄弟分支独立，不能仅复制 attention 的 KV。
                cache.reorder_cache(parents)
                batch = len(children)
                output = self.model(
                    input_ids=tokens[:, None],
                    attention_mask=torch.ones(
                        (batch, length + depth + 1), device=self.device, dtype=torch.long
                    ),
                    position_ids=torch.full(
                        (3, batch, 1), length + depth, device=self.device, dtype=torch.long
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                extra_tokens += batch
                cache = output.past_key_values
                log_probabilities = output.logits[:, -1, :].float().log_softmax(-1)
        ordered = torch.tensor([prefixes.index(label) for label in labels], device=self.device)
        totals = totals.index_select(0, ordered)
        temperature = self.settings.temperature_for("choice")
        probabilities = (totals / temperature).softmax(-1).cpu().tolist()
        return ScoredTask(
            probabilities=probabilities,
            input_tokens=length + extra_tokens,
            candidate_mass=float(totals.logsumexp(0).exp()),
        )

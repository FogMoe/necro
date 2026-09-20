import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

MODEL_ID = "necro-qwen3.5-0.8b"
DEFAULT_CHECKPOINT = "Qwen/Qwen3.5-0.8B"
MODEL_ALIASES = frozenset(
    {
        MODEL_ID,
        DEFAULT_CHECKPOINT,
        "jev-latest",
        "jev-preview",
        "jev-1.13.0",
    }
)


@dataclass(frozen=True)
class Settings:
    checkpoint: str = DEFAULT_CHECKPOINT
    adapter: str | None = None
    device: str = "auto"
    batch_size: int = 4
    batch_tokens: int = 8192
    temperature: float = 1.0
    choice_temperature: float | None = None
    max_sequence_tokens: int = 32768
    max_request_tokens: int = 65536
    api_key: str = field(default="necro-local", repr=False)

    def __post_init__(self):
        import math

        if self.batch_size < 1 or self.batch_tokens < 1:
            raise ValueError("批量大小和批量 token 预算必须大于 0。")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("NECRO_TEMPERATURE 必须是大于 0 的有限数。")
        if self.choice_temperature is not None and (
            not math.isfinite(self.choice_temperature) or self.choice_temperature <= 0
        ):
            raise ValueError("NECRO_CHOICE_TEMPERATURE 必须是大于 0 的有限数。")
        if not self.api_key:
            raise ValueError("NECRO_API_KEY 不能为空。")

    @classmethod
    def from_env(cls):
        load_dotenv(override=False)
        return cls(
            checkpoint=os.getenv("NECRO_MODEL", DEFAULT_CHECKPOINT),
            adapter=os.getenv("NECRO_ADAPTER") or None,
            device=os.getenv("NECRO_DEVICE", "auto"),
            batch_size=int(os.getenv("NECRO_BATCH_SIZE", "4")),
            batch_tokens=int(os.getenv("NECRO_BATCH_TOKENS", "8192")),
            temperature=float(os.getenv("NECRO_TEMPERATURE", "1.0")),
            choice_temperature=float(os.environ["NECRO_CHOICE_TEMPERATURE"])
            if os.getenv("NECRO_CHOICE_TEMPERATURE")
            else None,
            api_key=os.getenv("NECRO_API_KEY", "necro-local"),
        )

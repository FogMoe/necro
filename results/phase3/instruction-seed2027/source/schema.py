"""TypeSafe v1 请求与响应的结构；不复制 Jev 的权重或数值行为。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Content = str | dict[str, JsonValue] | list[JsonValue]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoulCriteria(StrictModel):
    true: Content | None = None
    false: Content | None = None


class Noul(StrictModel):
    type: Literal["noul"]
    instructions: Content
    criteria: NoulCriteria | None = None


class Choice(StrictModel):
    type: Literal["choice"]
    instructions: Content
    criteria: dict[str, Content | None] = Field(min_length=1, max_length=255)


class Score(StrictModel):
    type: Literal["score"]
    instructions: Content
    criteria: list[Content] = Field(min_length=2, max_length=10)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class EvaluationRequest(StrictModel):
    state: Content
    model: str
    questions: dict[str, Question] = Field(min_length=1)


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1, allow_inf_nan=False)


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float = Field(allow_inf_nan=False)
    legend: dict[str, Content]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(StrictModel):
    input_tokens: int
    output_tokens: int = 0


class EvaluationResponse(StrictModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage

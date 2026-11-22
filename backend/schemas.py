from __future__ import annotations

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    model: str
    answer: str | None = Field(None, description="'B. option text' for multiple choice, free text otherwise")
    answer_index: int | None = None
    explanation: str = ""
    extraction: str
    raw_output: str
    latency_s: float
    new_tokens: int


class CompareResponse(BaseModel):
    predictions: list[Prediction]


class Feedback(BaseModel):
    model: str
    question: str = Field(..., max_length=2000)
    answer: str | None = None
    correct: bool
    comment: str | None = Field(None, max_length=2000)

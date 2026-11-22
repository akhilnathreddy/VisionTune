"""Prompt and target formatting. One function builds every prompt, so baseline,
LoRA and QLoRA are always evaluated with identical inputs (blueprint 7.1)."""

from __future__ import annotations

from typing import Any

from visiontune.data.records import LETTERS

INSTRUCTION = (
    "Choose the correct option. Reply in exactly this format:\n"
    "Answer: <letter>\n"
    "Explanation: <one or two sentences>"
)
INSTRUCTION_NO_EXPLANATION = "Choose the correct option. Reply in exactly this format:\nAnswer: <letter>"


def format_options(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def build_user_text(record: dict[str, Any], include_explanation: bool = True) -> str:
    parts = []
    if record.get("context"):
        parts.append(f"Context: {record['context']}")
    parts.append(f"Question: {record['question']}")
    parts.append(f"Options:\n{format_options(record['choices'])}")
    parts.append(INSTRUCTION if include_explanation else INSTRUCTION_NO_EXPLANATION)
    return "\n".join(parts)


def build_target(record: dict[str, Any], include_explanation: bool = True) -> str:
    idx = record["answer_index"]
    text = f"Answer: {LETTERS[idx]}. {record['choices'][idx]}"
    if include_explanation and record.get("explanation"):
        text += f"\nExplanation: {record['explanation']}"
    return text


def build_messages(record: dict[str, Any], system: str | None, include_explanation: bool = True,
                   with_answer: bool = False) -> list[dict[str, Any]]:
    """Chat messages in the Hugging Face multimodal format. The image itself is passed
    to the processor separately; here it is only a placeholder."""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    user_content: list[dict[str, Any]] = []
    if record.get("has_image", True):
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": build_user_text(record, include_explanation)})
    messages.append({"role": "user", "content": user_content})
    if with_answer:
        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": build_target(record, include_explanation)}]})
    return messages


OPEN_INSTRUCTION = (
    "Answer the question about the image. Reply in exactly this format:\n"
    "Answer: <short answer>\n"
    "Explanation: <one or two sentences>"
)


def build_open_messages(question: str, context: str | None, system: str | None,
                        has_image: bool = True) -> list[dict[str, Any]]:
    """Free-form question without options (demo app only; the adapters were trained on
    multiple-choice questions, so open answers are outside the evaluated setting)."""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    parts = [f"Context: {context}"] if context else []
    parts += [f"Question: {question}", OPEN_INSTRUCTION]
    content: list[dict[str, Any]] = [{"type": "image"}] if has_image else []
    content.append({"type": "text", "text": "\n".join(parts)})
    messages.append({"role": "user", "content": content})
    return messages


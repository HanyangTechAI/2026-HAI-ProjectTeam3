import re
from typing import Dict


def infer_task_type(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.lower()).strip()

    if looks_like_math(compact):
        return "math"
    if contains_any(compact, ["summarize", "summary", "tl;dr", "요약", "줄여줘", "한 문장으로"]):
        return "summarization"
    if contains_any(compact, ["classify", "classification", "label", "category", "sentiment", "분류", "라벨"]):
        return "classification"
    if contains_any(
        compact,
        ["write", "draft", "rewrite", "email", "slack", "message", "제안해줘", "작성", "고쳐", "공지", "문장"],
    ):
        return "writing"
    return "general"


def contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)


def looks_like_math(text: str) -> bool:
    math_words = [
        "calculate",
        "solve",
        "discount",
        "percent",
        "ratio",
        "total",
        "how many",
        "how much",
        "sum",
        "difference",
        "product",
        "sale price",
        "new price",
        "계산",
        "몇",
        "합",
        "퍼센트",
        "할인",
        "비율",
        "가격",
    ]
    number_count = len(re.findall(r"[-+]?\d*\.?\d+", text))
    has_operator = bool(re.search(r"[+\-*/=]", text))
    return (contains_any(text, math_words) and number_count > 0) or (number_count >= 2 and has_operator)


def task_metadata(text: str) -> Dict[str, str]:
    task_type = infer_task_type(text)
    return {
        "task_type": task_type,
        "task_family": "math" if task_type == "math" else "general_llm",
    }

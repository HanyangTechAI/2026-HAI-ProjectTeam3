import re
import hashlib
from functools import lru_cache
from typing import Dict, List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised only in minimal demo envs
    SentenceTransformer = None


RATIO_WORDS = [
    "half",
    "twice",
    "third",
    "triple",
    "triple",
    "double",
    "times",
    "each",
    "per",
    "every",
    "remain",
    "remaining",
    "left",
]

MULTISTEP_HINT_WORDS = [
    "then",
    "after",
    "before",
    "altogether",
    "total",
    "in total",
    "how many more",
    "how much more",
    "if",
    "yesterday",
    "today",
    "tomorrow",
    "buy",
    "buys",
    "bought",
    "give",
    "gives",
    "gave",
]


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str):
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Use embedding_model_name='hashing:384' for offline demos."
        )
    return SentenceTransformer(model_name)


def extract_hashing_embedding(question: str, dim: int = 384) -> List[float]:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[A-Za-z0-9]+", question.lower())

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    return vec.astype(np.float32).tolist()


def count_digits(text: str) -> int:
    return len(re.findall(r"\d", text))


def contains_percent(text: str) -> bool:
    return "%" in text or "percent" in text.lower()


def contains_money(text: str) -> bool:
    lowered = text.lower()
    return "$" in text or "dollar" in lowered or "cents" in lowered


def contains_ratio_words(text: str) -> bool:
    lowered = text.lower()
    return any(contains_keyword(lowered, word) for word in RATIO_WORDS)


def contains_multistep_hint(text: str) -> bool:
    lowered = text.lower()
    return any(contains_keyword(lowered, word) for word in MULTISTEP_HINT_WORDS)


def contains_keyword(lowered_text: str, keyword: str) -> bool:
    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
    return re.search(pattern, lowered_text) is not None


def normalize_length(x: int, max_len: int = 200) -> float:
    return min(float(x) / float(max_len), 1.0)


def normalize_word_count(x: int, max_words: int = 100) -> float:
    return min(float(x) / float(max_words), 1.0)


def normalize_digit_count(x: int, max_digits: int = 10) -> float:
    return min(float(x) / float(max_digits), 1.0)


def extract_handcrafted_features(question: str) -> Dict[str, float]:
    question = question.strip()
    words = question.split()

    char_len = len(question)
    word_count = len(words)
    digit_count = count_digits(question)

    features = {
        "normalized_length": normalize_length(char_len, max_len=200),
        "normalized_word_count": normalize_word_count(word_count, max_words=100),
        "normalized_digit_count": normalize_digit_count(digit_count, max_digits=10),
        "has_percent": 1.0 if contains_percent(question) else 0.0,
        "has_money": 1.0 if contains_money(question) else 0.0,
        "has_ratio_words": 1.0 if contains_ratio_words(question) else 0.0,
        "has_multistep_hint": 1.0 if contains_multistep_hint(question) else 0.0,
    }
    return features


def extract_embedding(
    question: str,
    model_name: str,
    normalize_embedding: bool = True,
) -> List[float]:
    if model_name.startswith("hashing:"):
        dim_text = model_name.split(":", 1)[1]
        dim = int(dim_text) if dim_text else 384
        return extract_hashing_embedding(question, dim=dim)

    model = get_embedding_model(model_name)

    embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embedding,
    )

    if isinstance(embedding, np.ndarray):
        embedding = embedding.astype(np.float32).tolist()

    return embedding


def build_full_state(
    question: str,
    model_name: str,
    normalize_embedding: bool = True,
):
    handcrafted = extract_handcrafted_features(question)
    embedding = extract_embedding(
        question=question,
        model_name=model_name,
        normalize_embedding=normalize_embedding,
    )

    return {
        "question": question,
        "handcrafted": handcrafted,
        "embedding": embedding,
    }

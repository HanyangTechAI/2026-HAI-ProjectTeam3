import re

from .schemas import Domain, PromptAnalysis, ReasoningDepth, RiskLevel, TaskType


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def analyze_prompt(prompt: str) -> PromptAnalysis:
    text = prompt.strip()
    lowered = text.lower()
    words = re.findall(r"\w+", text)
    word_count = len(words)
    prompt_tokens = estimate_tokens(text)
    signals: list[str] = []

    task_type = detect_task_type(lowered)
    domain = detect_domain(lowered)
    risk_score, risk_level = detect_risk(lowered, domain)

    complexity = 0.0
    if prompt_tokens > 250:
        complexity += 0.25
        signals.append("long_prompt")
    if prompt_tokens > 1000:
        complexity += 0.25
        signals.append("very_long_prompt")
    if task_type in {TaskType.MATH, TaskType.CODING, TaskType.STOCK}:
        complexity += 0.25
        signals.append(f"{task_type.value}_task")
    if any(marker in lowered for marker in ["step by step", "prove", "debug", "optimize", "compare", "trade-off", "tradeoff"]):
        complexity += 0.2
        signals.append("explicit_reasoning_request")
    if any(marker in lowered for marker in ["multiple", "constraints", "edge case", "architecture", "design"]):
        complexity += 0.15
        signals.append("multi_constraint_request")
    if risk_level != RiskLevel.LOW:
        complexity += 0.1
        signals.append(f"{risk_level.value}_risk")

    complexity = min(1.0, complexity)
    if complexity < 0.33:
        complexity_level = "low"
        reasoning_need = ReasoningDepth.NONE
    elif complexity < 0.66:
        complexity_level = "medium"
        reasoning_need = ReasoningDepth.SHORT
    else:
        complexity_level = "high"
        reasoning_need = ReasoningDepth.LONG

    if task_type in {TaskType.MATH, TaskType.CODING, TaskType.STOCK} and reasoning_need == ReasoningDepth.NONE:
        reasoning_need = ReasoningDepth.SHORT

    return PromptAnalysis(
        charLength=len(text),
        wordCount=word_count,
        promptTokensEstimate=prompt_tokens,
        complexityScore=round(complexity, 3),
        complexityLevel=complexity_level,
        domain=domain,
        taskType=task_type,
        reasoningNeed=reasoning_need,
        riskScore=round(risk_score, 3),
        riskLevel=risk_level,
        signals=signals,
    )


def detect_task_type(text: str) -> TaskType:
    if contains_any(text, ["stock", "stocks", "share price", "ticker", "portfolio", "dividend", "earnings", "valuation", "market cap", "buy or sell", "investment", "investing", "etf"]):
        return TaskType.STOCK
    if contains_any(text, ["classify", "classification", "label", "category", "sentiment", "분류", "라벨"]):
        return TaskType.CLASSIFICATION
    if contains_any(text, ["python", "javascript", "typescript", "sql", "code", "bug", "debug", "function", "api error"]):
        return TaskType.CODING
    if (contains_any(text, ["calculate", "solve", "equation", "percent", "ratio", "how many", "how much"]) and has_number(text)) or has_formula(text):
        return TaskType.MATH
    if contains_any(text, ["summarize", "summary", "tl;dr", "요약", "줄여줘"]):
        return TaskType.SUMMARIZATION
    if contains_any(text, ["write", "draft", "rewrite", "email", "slack", "작성", "고쳐"]):
        return TaskType.WRITING
    return TaskType.GENERAL


def detect_domain(text: str) -> Domain:
    if contains_any(text, ["api", "database", "server", "deploy", "latency", "code", "bug", "프론트", "백엔드"]):
        return Domain.SOFTWARE
    if contains_any(text, ["contract", "law", "legal", "privacy", "terms", "compliance"]):
        return Domain.LEGAL
    if contains_any(text, ["doctor", "medical", "symptom", "medicine", "diagnosis", "health"]):
        return Domain.MEDICAL
    if contains_any(text, ["invoice", "payment", "subscription", "revenue", "cost", "finance", "billing", "charged", "stock", "ticker", "portfolio", "dividend", "earnings", "investment", "etf"]):
        return Domain.FINANCE
    if contains_any(text, ["lesson", "student", "exam", "homework", "education"]):
        return Domain.EDUCATION
    if contains_any(text, ["customer", "product", "sales", "marketing", "dashboard", "contract"]):
        return Domain.BUSINESS
    return Domain.GENERAL


def detect_risk(text: str, domain: Domain) -> tuple[float, RiskLevel]:
    score = 0.0
    if domain in {Domain.LEGAL, Domain.MEDICAL, Domain.FINANCE}:
        score += 0.45
    if contains_any(text, ["personal data", "password", "secret", "api key", "refund", "lawsuit", "diagnosis", "buy or sell", "investment advice"]):
        score += 0.35
    if contains_any(text, ["must", "guarantee", "critical", "production", "security", "compliance"]):
        score += 0.2
    score = min(1.0, score)
    if score >= 0.66:
        return score, RiskLevel.HIGH
    if score >= 0.33:
        return score, RiskLevel.MEDIUM
    return score, RiskLevel.LOW


def contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)


def has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def has_formula(text: str) -> bool:
    return bool(re.search(r"\d+\s*[+\-*/=]\s*\d+", text))

import re
from dataclasses import dataclass


@dataclass
class VerificationResult:
    repaired_text: str
    extracted_answer: str
    format_ok: bool


def normalize_number_string(text: str) -> str:
    return text.strip().replace(",", "")


def extract_final_answer(text: str) -> str:
    """
    1. FINAL: <number> 우선 탐색
    2. 없으면 마지막 숫자 fallback
    """
    match = re.search(r"FINAL:\s*([-+]?\d*\.?\d+)", text, flags=re.IGNORECASE)
    if match:
        return normalize_number_string(match.group(1))

    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        return normalize_number_string(numbers[-1])

    return ""


def is_final_format(text: str) -> bool:
    stripped = text.strip()
    return re.fullmatch(r"FINAL:\s*[-+]?\d*\.?\d+", stripped, flags=re.IGNORECASE) is not None


def verify_and_repair_output(text: str) -> VerificationResult:
    """
    verify action에서 사용하는 최소 verifier:
    - 형식이 맞으면 그대로 통과
    - 형식이 아니면 숫자 추출 후 FINAL 형식으로 복구
    """
    extracted = extract_final_answer(text)

    if is_final_format(text):
        return VerificationResult(
            repaired_text=text.strip(),
            extracted_answer=extracted,
            format_ok=True,
        )

    if extracted != "":
        repaired = f"FINAL: {extracted}"
        return VerificationResult(
            repaired_text=repaired,
            extracted_answer=extracted,
            format_ok=True,
        )

    return VerificationResult(
        repaired_text=text.strip(),
        extracted_answer="",
        format_ok=False,
    )
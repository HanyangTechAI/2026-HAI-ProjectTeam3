def build_prompt(question: str, reasoning_budget: str) -> str:
    """
    reasoning_budget:
        - "none"
        - "short"
        - "long"
    """

    question = question.strip()

    if reasoning_budget == "none":
        instruction = (
            "Solve the math word problem and output exactly one line in the format:\n"
            "FINAL: <number>\n"
            "Do not include any explanation."
        )

    elif reasoning_budget == "short":
        instruction = (
            "Solve the math word problem carefully.\n"
            "Think briefly if needed, but do not show reasoning.\n"
            "Output exactly one line in the format:\n"
            "FINAL: <number>"
        )

    elif reasoning_budget == "long":
        instruction = (
            "Solve the math word problem very carefully.\n"
            "Use deeper internal reasoning if needed, but do not show reasoning.\n"
            "Double-check arithmetic internally before answering.\n"
            "Output exactly one line in the format:\n"
            "FINAL: <number>"
        )

    else:
        raise ValueError(f"Unknown reasoning_budget: {reasoning_budget}")

    prompt = f"{instruction}\n\nQuestion: {question}"
    return prompt
def build_prompt(question: str, reasoning_budget: str, task_type: str = "math") -> str:
    """
    reasoning_budget:
        - "none"
        - "short"
        - "long"
    """

    question = question.strip()

    if task_type != "math":
        return build_general_prompt(question, reasoning_budget, task_type)

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

    prompt = f"{instruction}\n\nRequest: {question}"
    return prompt


def build_general_prompt(request: str, reasoning_budget: str, task_type: str) -> str:
    if reasoning_budget == "none":
        budget_instruction = "Answer directly and keep the response compact."
    elif reasoning_budget == "short":
        budget_instruction = "Use brief internal planning, then provide a clear response."
    elif reasoning_budget == "long":
        budget_instruction = "Use careful internal planning and cover the important details."
    else:
        raise ValueError(f"Unknown reasoning_budget: {reasoning_budget}")

    task_instructions = {
        "summarization": "Summarize the user's content faithfully. Preserve key facts and avoid adding unsupported details.",
        "classification": "Classify the user's content into the most useful label or labels, and include a short rationale.",
        "writing": "Draft or rewrite the requested text in a polished, useful form.",
        "general": "Respond helpfully to the user's request.",
    }
    instruction = task_instructions.get(task_type, task_instructions["general"])
    return f"{instruction}\n{budget_instruction}\n\nRequest: {request}"

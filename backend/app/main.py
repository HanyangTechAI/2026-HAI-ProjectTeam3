import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .analyzer import analyze_prompt
from .model_policy import build_policy_from_env
from .policy import selected_cost
from .pricing import estimate_cost
from .providers import build_inference_prompt, provider_for
from .schemas import OptimizeResponse, PromptRequest, ProviderUsage, StatsResponse
from .store import UsageStore


app = FastAPI(title="AI API Cost Optimizer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

policy = build_policy_from_env()
store = UsageStore()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ai-api-cost-optimizer",
        "policy": policy.__class__.__name__,
    }


@app.post("/api/optimize", response_model=OptimizeResponse)
def optimize(request: PromptRequest) -> OptimizeResponse:
    request_id = str(uuid.uuid4())
    analysis = analyze_prompt(request.prompt)
    strategy, candidates = policy.choose(
        analysis=analysis,
        max_completion_tokens=request.maxCompletionTokens,
        force_mock=request.forceMock,
    )
    estimated = selected_cost(analysis, strategy, request.maxCompletionTokens)
    inference_prompt = build_inference_prompt(request.prompt, strategy)
    provider = provider_for(strategy, force_mock=request.forceMock)
    output, usage, provider_mode = provider.generate(
        prompt=inference_prompt,
        strategy=strategy,
        max_completion_tokens=request.maxCompletionTokens,
    )

    if usage.estimatedCostUsd == 0.0 and strategy.modelRoute.value != "local":
        actualish_cost = estimate_cost(strategy.modelRoute, usage.promptTokens, usage.completionTokens)
        usage = ProviderUsage(
            promptTokens=usage.promptTokens,
            completionTokens=usage.completionTokens,
            totalTokens=usage.totalTokens,
            estimatedCostUsd=actualish_cost.totalCostUsd,
        )

    response = OptimizeResponse(
        requestId=request_id,
        analysis=analysis,
        strategy=strategy,
        estimatedCost=estimated,
        candidates=candidates,
        output=output,
        providerMode=provider_mode,
        usage=usage,
    )
    store.insert(
        {
            "request_id": request_id,
            "route": strategy.modelRoute.value,
            "task_type": analysis.taskType.value,
            "prompt_tokens": usage.promptTokens,
            "completion_tokens": usage.completionTokens,
            "total_tokens": usage.totalTokens,
            "estimated_cost_usd": usage.estimatedCostUsd,
            "payload": response.model_dump(mode="json"),
        }
    )
    return response


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(**store.stats())

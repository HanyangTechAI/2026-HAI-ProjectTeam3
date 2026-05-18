# AI API Cost Optimizer SaaS

사용자 요청을 분석해서 OpenAI, Gemini, local/mock 중 비용 대비 효율이 좋은 추론 전략을 선택하는 AI Gateway 서비스입니다.

프로젝트 목표는 단순 API 호출 래퍼가 아니라, 실제 요청 로그와 피드백을 바탕으로 모델 라우팅 정책을 계속 개선하는 것입니다.

## Features

- Prompt analysis: task type, domain, risk, complexity, estimated tokens
- Routing strategy: model route, reasoning depth, verification, retry, context compression
- Cost estimation: route별 예상 input/output cost
- Provider execution: OpenAI, Gemini, mock/local fallback
- Usage tracking: SQLite or PostgreSQL
- Frontend dashboard: prompt input, selected strategy, response, usage stats
- Learned policy: trainable linear routing model with rule-based fallback

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start backend:

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Start frontend:

```bash
python -m http.server 3000 --directory frontend
```

Open:

- Frontend: `http://127.0.0.1:3000`
- Backend: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`

If API keys are not set, the backend automatically falls back to mock providers where needed.

Environment variables:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
DATABASE_URL=postgresql://optimizer:optimizer@localhost:5432/optimizer
ROUTING_POLICY_PATH=outputs/routing_policy.json
```

Without `DATABASE_URL`, usage is stored in `outputs/usage.db`.

## Docker Compose

```bash
docker compose up --build
```

Services:

- Frontend: `http://127.0.0.1:3000`
- Backend: `http://127.0.0.1:8000`
- PostgreSQL: `localhost:5432`

## API

Optimize and execute:

```bash
curl -X POST http://127.0.0.1:8000/api/optimize \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Write a short follow-up email about contract review.\",\"forceMock\":true}"
```

Stats:

```bash
curl http://127.0.0.1:8000/api/stats
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

## Learned Routing Model

The first service model is a pure-Python linear routing policy in `backend/app/model_policy.py`.

It receives `PromptAnalysis` features and chooses one strategy action:

- `modelRoute`
- `reasoningDepth`
- `verify`
- `retry`
- `contextCompression`

Train the initial model:

```bash
python run_train_routing_model.py \
  --suite_path data/service_request_suite.json \
  --output_path outputs/routing_policy.json \
  --metrics_path outputs/routing_policy_metrics.json
```

The server automatically uses `outputs/routing_policy.json` if it exists. If the model file is missing or incompatible, the service falls back to `RuleBasedPolicy`.

Current initial training uses the rule-based policy as a teacher. The next research step is to replace teacher labels with service logs, user feedback, observed cost, latency, and quality scores.

## RL Routing Policy

The first RL direction is an offline contextual bandit. Each prompt analysis is the context, each routing strategy is an action, and the reward balances:

- heuristic expected quality
- estimated API cost
- latency/retry/verification overhead

Train the RL-style policy:

```bash
python run_train_rl_routing_policy.py \
  --suite_path data/service_request_suite.json \
  --output_path outputs/rl_routing_policy.json \
  --metrics_path outputs/rl_routing_policy_metrics.json \
  --history_path outputs/rl_routing_policy_history.json
```

Run the backend with the RL policy:

```bash
ROUTING_POLICY_PATH=outputs/rl_routing_policy.json \
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

On Windows PowerShell:

```powershell
$env:ROUTING_POLICY_PATH="outputs/rl_routing_policy.json"
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

This is still offline RL with a simulated reward. The next production step is to log explicit user feedback or evaluator scores, then train the same bandit update from observed rewards instead of the heuristic reward.

## RLHF Feedback Loop

The service can now collect human feedback for each generated response:

```bash
curl -X POST http://127.0.0.1:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d "{\"requestId\":\"REQUEST_ID_FROM_OPTIMIZE\",\"reviewerId\":\"rater_01\",\"rating\":1,\"qualityScore\":1.0,\"comment\":\"Good answer\"}"
```

Feedback fields:

- `reviewerId`: required reviewer identifier, such as `rater_01`
- `rating`: `1` for good, `-1` for bad, `0` for neutral
- `qualityScore`: optional normalized human reward from `0.0` to `1.0`
- `comment`: optional annotation for later analysis

Multiple reviewers can evaluate the same `requestId`. During RLHF training, feedback is aggregated per request by mean reward. If the same reviewer submits multiple ratings for the same request, the latest stored row is used and older rows are ignored for that reviewer/request pair.

Train from collected human feedback:

```bash
python run_train_rlhf_policy.py \
  --sqlite_path outputs/usage.db \
  --initial_model_path outputs/rl_routing_policy.json \
  --output_path outputs/rlhf_routing_policy.json \
  --metrics_path outputs/rlhf_routing_policy_metrics.json \
  --history_path outputs/rlhf_routing_policy_history.json
```

Run the backend with the RLHF-trained policy:

```powershell
$env:ROUTING_POLICY_PATH="outputs/rlhf_routing_policy.json"
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

This is RLHF for routing policy optimization, not LLM weight fine-tuning. Human feedback updates which provider/strategy the gateway selects for future requests.

## Structure

```text
backend/app/
  analyzer.py        # prompt analysis
  model_policy.py    # learned linear routing policy
  policy.py          # Policy interface and RuleBasedPolicy baseline
  pricing.py         # token and route cost estimation
  providers.py       # OpenAI, Gemini, Mock providers
  schemas.py         # Pydantic request/response models
  store.py           # SQLite/PostgreSQL usage store
  main.py            # FastAPI app
frontend/
  index.html
  styles.css
  app.js
data/
  service_request_suite.json
run_train_routing_model.py
run_train_rl_routing_policy.py
run_train_rlhf_policy.py
docker-compose.yml
```

## Research Roadmap

- Add offline evaluation against `data/service_request_suite.json`
- Convert usage logs and Good/Bad feedback into preference pairs
- Add reward terms for cost, latency, retry rate, and user satisfaction
- Train a reranker or contextual bandit policy from real traffic
- Add organization-level budget limits and route constraints

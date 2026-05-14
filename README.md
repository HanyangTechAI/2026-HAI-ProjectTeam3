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
docker-compose.yml
```

## Research Roadmap

- Add offline evaluation against `data/service_request_suite.json`
- Convert usage logs and Good/Bad feedback into preference pairs
- Add reward terms for cost, latency, retry rate, and user satisfaction
- Train a reranker or contextual bandit policy from real traffic
- Add organization-level budget limits and route constraints

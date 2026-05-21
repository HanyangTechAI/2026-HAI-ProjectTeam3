# AI API Cost Optimizer

AI API Cost Optimizer is a small gateway service that analyzes each prompt, selects a cost-aware inference strategy, calls an LLM provider when available, and records usage plus human feedback for later routing-policy training.

The service supports OpenAI, Gemini, and local/mock execution. If provider credentials are missing, routes fall back to mock output. If a Gemini request fails and an OpenAI key is available, the backend retries once through the matching OpenAI route.

## Features

- Prompt analysis: task type, domain, risk, complexity, and token estimates
- Routing strategy: model route, reasoning depth, verification, retry, and context compression
- Cost estimation for OpenAI, Gemini, and local/mock routes
- Provider execution through OpenAI, Gemini, or mock fallback
- Usage tracking in SQLite by default or PostgreSQL through Docker Compose
- Frontend dashboard for prompts, selected strategy, responses, feedback, and stats
- Trainable linear routing policies, including offline RL and RLHF-style feedback training

## Environment

Create a local `.env` from the committed template:

```bash
cp .env.example .env
```

`.env` is ignored by git. Put real provider keys there only if you want live API calls:

```bash
OPENAI_API_KEY=
GEMINI_API_KEY=
GEMINI_SMALL_MODEL=gemini-2.5-flash-lite
GEMINI_LARGE_MODEL=gemini-2.5-flash
DATABASE_URL=postgresql://optimizer:optimizer@localhost:5432/optimizer
ROUTING_POLICY_PATH=outputs/routing_policy.json
```

Notes:

- `OPENAI_API_KEY` and `GEMINI_API_KEY` are optional. Without them, the backend uses mock providers where needed.
- `GEMINI_LARGE_MODEL` defaults to `gemini-2.5-flash` because `gemini-2.5-pro` may have no free-tier quota.
- Without `DATABASE_URL`, usage is stored in `outputs/usage.db`.
- `ROUTING_POLICY_PATH` is optional. If unset, the server tries `outputs/routing_policy.json` and otherwise falls back to the rule-based policy.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The backend serves the frontend at:

- App: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`
- API docs: `http://127.0.0.1:8000/docs`

You can also serve the frontend separately during UI work:

```bash
python -m http.server 3000 --directory frontend
```

Then open `http://127.0.0.1:3000`. The frontend will call the backend on port `8000`.

## Docker Compose

Start the full stack:

```bash
docker compose up --build
```

Services:

- Frontend: `http://127.0.0.1:3000`
- Backend: `http://127.0.0.1:8000`
- Backend also mapped on: `http://127.0.0.1`
- PostgreSQL: `localhost:5432`

If you change backend or frontend source, rebuild the containers:

```bash
docker compose up --build
```

## API

Optimize and execute a prompt:

```bash
curl -X POST http://127.0.0.1:8000/api/optimize \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"A jacket costs 80 dollars and is discounted by 25 percent. What is the sale price?\",\"maxCompletionTokens\":512,\"forceMock\":false}"
```

Force mock mode:

```bash
curl -X POST http://127.0.0.1:8000/api/optimize \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Write a short follow-up email about contract review.\",\"forceMock\":true}"
```

Submit feedback:

```bash
curl -X POST http://127.0.0.1:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d "{\"requestId\":\"REQUEST_ID_FROM_OPTIMIZE\",\"reviewerId\":\"rater_01\",\"rating\":1,\"qualityScore\":1.0,\"comment\":\"Good answer\"}"
```

Stats and health:

```bash
curl http://127.0.0.1:8000/api/stats
curl http://127.0.0.1:8000/health
```

## Provider Behavior

Provider routes are selected by the active routing policy:

- OpenAI small: `gpt-4.1-mini`
- OpenAI large: `gpt-4.1`
- Gemini small: `GEMINI_SMALL_MODEL`, default `gemini-2.5-flash-lite`
- Gemini large: `GEMINI_LARGE_MODEL`, default `gemini-2.5-flash`
- Local: mock provider

The API returns structured JSON errors for provider failures. Example:

```json
{
  "detail": "Provider request failed for gemini-large: ..."
}
```

If Gemini fails and `OPENAI_API_KEY` is set, the backend retries once with the matching OpenAI route and marks `providerMode` as `openai-fallback`.

High-risk requests detected by the analyzer bypass learned routing and use the rule-based verified large-model route.

## Learned Routing Model

The learned routing policy is implemented in `backend/app/model_policy.py`. It receives `PromptAnalysis` features and chooses:

- `modelRoute`
- `reasoningDepth`
- `verify`
- `retry`
- `contextCompression`

Train the initial supervised routing model:

```bash
python run_train_routing_model.py \
  --suite_path data/service_request_suite.json \
  --output_path outputs/routing_policy.json \
  --metrics_path outputs/routing_policy_metrics.json
```

The server automatically uses `outputs/routing_policy.json` if it exists. If the model file is missing or incompatible, the service falls back to `RuleBasedPolicy`.

## RL Routing Policy

The offline RL direction is a contextual bandit. Each prompt analysis is the context, each routing strategy is an action, and the reward balances quality, estimated cost, latency, retry overhead, and verification overhead.

Train the RL-style policy:

```bash
python run_train_rl_routing_policy.py \
  --suite_path data/service_request_suite.json \
  --output_path outputs/rl_routing_policy.json \
  --metrics_path outputs/rl_routing_policy_metrics.json \
  --history_path outputs/rl_routing_policy_history.json
```

Run with the RL policy:

```powershell
$env:ROUTING_POLICY_PATH="outputs/rl_routing_policy.json"
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## RLHF Feedback Loop

The service collects human feedback for each generated response through `/api/feedback`.

Feedback fields:

- `reviewerId`: reviewer identifier, such as `rater_01`
- `rating`: `1` for good, `-1` for bad, `0` for neutral
- `qualityScore`: optional normalized reward from `0.0` to `1.0`
- `comment`: optional annotation for later analysis

Multiple reviewers can evaluate the same `requestId`. During RLHF training, feedback is aggregated per request by mean reward. If the same reviewer submits multiple ratings for the same request, the latest row is used for that reviewer/request pair.

Train from collected human feedback:

```bash
python run_train_rlhf_policy.py \
  --sqlite_path outputs/usage.db \
  --initial_model_path outputs/rl_routing_policy.json \
  --output_path outputs/rlhf_routing_policy.json \
  --metrics_path outputs/rlhf_routing_policy_metrics.json \
  --history_path outputs/rlhf_routing_policy_history.json
```

If feedback was collected in the Docker PostgreSQL database, export both feedback and usage payloads first:

```bash
mkdir -p outputs

docker compose exec -T db psql -U optimizer -d optimizer \
  -t -A \
  -c "select coalesce(json_agg(row_to_json(t)), '[]'::json) from (select id, request_id, reviewer_id, rating, quality_score, reward, comment, created_at from feedback_events order by id desc) t;" \
  > outputs/feedback_events.json

docker compose exec -T db psql -U optimizer -d optimizer \
  -t -A \
  -c "select coalesce(json_agg(row_to_json(t)), '[]'::json) from (select request_id, payload, created_at from usage_events order by created_at desc) t;" \
  > outputs/usage_events.json
```

Then train from the exported JSON files:

```bash
python run_train_rlhf_policy.py \
  --feedback_json_path outputs/feedback_events.json \
  --usage_json_path outputs/usage_events.json \
  --initial_model_path outputs/rl_routing_policy.json \
  --output_path outputs/rlhf_routing_policy.json \
  --metrics_path outputs/rlhf_routing_policy_metrics.json \
  --history_path outputs/rlhf_routing_policy_history.json \
  --lr 0.02 \
  --epochs 8
```

The lower learning rate and shorter run keep the RLHF policy from collapsing all traffic into a single route.

Run with the RLHF-trained policy:

```powershell
$env:ROUTING_POLICY_PATH="outputs/rlhf_routing_policy.json"
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

This is RLHF for routing policy optimization, not LLM weight fine-tuning.

## Project Structure

```text
backend/app/
  analyzer.py        # prompt analysis
  model_policy.py    # learned linear routing policy
  policy.py          # policy interface and rule-based baseline
  pricing.py         # token and route cost estimation
  providers.py       # OpenAI, Gemini, and mock providers
  schemas.py         # Pydantic request/response models
  store.py           # SQLite/PostgreSQL usage and feedback store
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

## Roadmap

- Add offline evaluation against `data/service_request_suite.json`
- Convert usage logs and Good/Bad feedback into preference pairs
- Add reward terms for cost, latency, retry rate, and user satisfaction
- Train a reranker or contextual bandit policy from real traffic
- Add organization-level budget limits and route constraints

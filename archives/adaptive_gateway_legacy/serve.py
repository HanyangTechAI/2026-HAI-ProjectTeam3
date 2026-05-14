import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from configs import TrainConfig
from src.demo_samples import DEMO_SAMPLES
from src.inference_service import InferencePolicyService, ServiceConfig


INDEX_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adaptive AI Gateway</title>
  <style>
    :root { --ink:#16202a; --muted:#66717d; --line:#d8e0e8; --panel:#f7fafc; --accent:#0f766e; }
    body { margin:0; font-family:Arial, Helvetica, sans-serif; color:var(--ink); background:#fff; }
    header { padding:28px 36px; border-bottom:1px solid var(--line); background:#f8fbfb; }
    main { max-width:1080px; margin:0 auto; padding:28px 20px 44px; }
    h1 { margin:0 0 8px; font-size:28px; letter-spacing:0; }
    p { color:var(--muted); line-height:1.5; }
    textarea, input, select { width:100%; box-sizing:border-box; border:1px solid var(--line); border-radius:8px; padding:10px 12px; font:inherit; }
    textarea { min-height:110px; resize:vertical; }
    label { display:block; font-weight:700; margin:16px 0 8px; }
    button { border:0; border-radius:8px; background:var(--accent); color:white; padding:11px 16px; font-weight:700; cursor:pointer; }
    button.secondary { background:#334155; }
    .grid { display:grid; grid-template-columns:2fr 1fr; gap:18px; }
    .panel { border:1px solid var(--line); border-radius:8px; padding:18px; background:var(--panel); }
    .actions { display:flex; gap:10px; margin-top:16px; flex-wrap:wrap; }
    .section-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:26px; }
    pre { white-space:pre-wrap; background:#111827; color:#f8fafc; border-radius:8px; padding:16px; overflow:auto; min-height:220px; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:14px 0; }
    .metric { border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; overflow-wrap:anywhere; }
    .metric span { display:block; color:var(--muted); font-size:12px; margin-bottom:6px; }
    .metric strong { font-size:20px; }
    @media (max-width:760px) { .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Adaptive AI Gateway</h1>
    <p>요청 유형과 난이도를 보고 추론 길이, 모델 라우팅, 검증 사용 여부를 선택합니다. 수학 문제뿐 아니라 요약, 작성, 분류, 일반 요청을 같은 정책 실험 루프로 기록합니다.</p>
  </header>
  <main>
    <div class="grid">
      <section class="panel">
        <label for="question">Request</label>
        <textarea id="question">다음 공지 문장을 더 정중하고 간결하게 고쳐줘: 내일까지 보고서 보내주세요.</textarea>
        <label for="gold">Expected answer (optional, mostly for evaluation)</label>
        <input id="gold" value="">
        <label for="topk">Top K actions</label>
        <select id="topk"><option>3</option><option>5</option><option>8</option></select>
        <div class="actions">
          <button onclick="predict()">Predict</button>
          <button class="secondary" onclick="demo()">Run Math Demo Batch</button>
        </div>
      </section>
      <section class="panel">
        <h2>Result</h2>
        <div class="metrics" id="metrics"></div>
        <div class="actions">
          <button class="secondary" onclick="feedback('up')">Good</button>
          <button class="secondary" onclick="feedback('down')">Bad</button>
        </div>
      </section>
    </div>
    <div class="section-head">
      <h2>Gateway Dashboard</h2>
      <div class="actions">
        <button class="secondary" onclick="metrics()">Refresh Metrics</button>
        <button class="secondary" onclick="logs()">Recent Logs</button>
      </div>
    </div>
    <div class="metrics" id="dashboard"></div>
    <h2>Raw JSON</h2>
    <pre id="output">Ready.</pre>
  </main>
  <script>
    async function postJson(path, body) {
      const res = await fetch(path, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }
    let lastResult = null;
    function render(data) {
      if (data && data.request_id) lastResult = data;
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      const target = data.summary || data;
      const metrics = [];
      if (target.pred !== undefined) metrics.push(['Prediction', target.pred]);
      if (target.task && target.task.task_type) metrics.push(['Task', target.task.task_type]);
      if (target.chosen_action_label) metrics.push(['Action', target.chosen_action_label]);
      if (target.accuracy !== undefined) metrics.push(['Accuracy', Number(target.accuracy).toFixed(3)]);
      if (target.avg_reward !== undefined) metrics.push(['Avg Reward', Number(target.avg_reward).toFixed(3)]);
      if (target.total_tokens !== undefined) metrics.push(['Tokens', target.total_tokens]);
      if (target.avg_total_tokens !== undefined) metrics.push(['Avg Tokens', Number(target.avg_total_tokens).toFixed(1)]);
      document.getElementById('metrics').innerHTML = metrics.map(([k,v]) => `<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join('');
    }
    function renderDashboard(data) {
      const metrics = [];
      metrics.push(['Requests', data.total_requests ?? 0]);
      metrics.push(['Feedback', data.feedback_count ?? 0]);
      metrics.push(['Labeled', data.labeled_requests ?? 0]);
      if (data.accuracy !== null && data.accuracy !== undefined) metrics.push(['Accuracy', Number(data.accuracy).toFixed(3)]);
      if (data.avg_reward !== null && data.avg_reward !== undefined) metrics.push(['Avg Reward', Number(data.avg_reward).toFixed(3)]);
      if (data.avg_total_tokens !== null && data.avg_total_tokens !== undefined) metrics.push(['Avg Tokens', Number(data.avg_total_tokens).toFixed(1)]);
      metrics.push(['Cost Units', Number(data.estimated_cost_units || 0).toFixed(1)]);
      if (data.estimated_savings_rate !== null && data.estimated_savings_rate !== undefined) {
        metrics.push(['Savings vs Large', `${(Number(data.estimated_savings_rate) * 100).toFixed(1)}%`]);
      }
      if (data.task_type_hist) metrics.push(['Tasks', JSON.stringify(data.task_type_hist)]);
      document.getElementById('dashboard').innerHTML = metrics.map(([k,v]) => `<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join('');
    }
    async function predict() {
      try {
        render(await postJson('/api/predict', {
          question: document.getElementById('question').value,
          gold: document.getElementById('gold').value,
          topk: Number(document.getElementById('topk').value)
        }));
      } catch (err) { render({ error: String(err.message || err) }); }
    }
    async function demo() {
      try { render(await postJson('/api/demo', { num_samples: 5 })); }
      catch (err) { render({ error: String(err.message || err) }); }
    }
    async function metrics() {
      try {
        const res = await fetch('/api/metrics');
        const data = await res.json();
        renderDashboard(data);
        render(data);
      } catch (err) { render({ error: String(err.message || err) }); }
    }
    async function logs() {
      try {
        const res = await fetch('/api/logs?limit=20');
        render(await res.json());
      } catch (err) { render({ error: String(err.message || err) }); }
    }
    async function feedback(rating) {
      if (!lastResult) {
        render({ error: 'No prediction result to rate yet.' });
        return;
      }
      try {
        render(await postJson('/api/feedback', {
          request: lastResult,
          feedback: { rating, source: 'web_ui' }
        }));
        metrics();
      } catch (err) { render({ error: String(err.message || err) }); }
    }
    metrics();
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    service: InferencePolicyService

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
        elif parsed.path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "model_path": self.service.model_path,
                    "api_mode": self.service.resolved_api_mode,
                    "requested_api_mode": self.service.requested_api_mode,
                }
            )
        elif parsed.path == "/api/samples":
            self._send_json({"math_samples": DEMO_SAMPLES, "general_samples": load_general_samples()})
        elif parsed.path == "/api/metrics":
            self._send_json(self.service.metrics())
        elif parsed.path == "/api/logs":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["50"])[0])
            self._send_json(self.service.recent_logs(limit=limit))
        else:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self._read_json()
            if parsed.path == "/api/predict":
                self._send_json(
                    self.service.predict(
                        question=str(body.get("question") or body.get("request") or ""),
                        gold=str(body.get("gold", "")),
                        topk=int(body.get("topk", self.service.service_config.topk)),
                    )
                )
            elif parsed.path == "/api/demo":
                self._send_json(self.service.demo_batch(num_samples=int(body.get("num_samples", 5))))
            elif parsed.path == "/api/feedback":
                self._send_json(
                    self.service.add_feedback(
                        request_record=dict(body.get("request", {})),
                        feedback=dict(body.get("feedback", {})),
                    )
                )
            else:
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format, *args):
        print(f"[HTTP] {self.address_string()} - {format % args}")

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_json(self, data, status=HTTPStatus.OK):
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html: str, status=HTTPStatus.OK):
        payload = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def load_general_samples():
    path = os.path.join("data", "service_general_requests.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Serve the adaptive AI gateway.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api_mode",
        choices=["config", "auto", "mock", "openai"],
        default="config",
        help="Use configs.py by default. Override with auto/mock/openai.",
    )
    parser.add_argument("--embedding_model", default=cfg.demo_embedding_model_name)
    parser.add_argument("--policy_source", choices=["auto", "checkpoint", "heuristic"], default="auto")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--log_path", default="outputs/service_requests.jsonl")
    args = parser.parse_args()
    api_mode = cfg.api_mode if args.api_mode == "config" else args.api_mode

    service_config = ServiceConfig(
        api_mode=api_mode,
        embedding_model=args.embedding_model,
        policy_source=args.policy_source,
        checkpoint=args.checkpoint,
        topk=args.topk,
        log_path=args.log_path,
    )
    RequestHandler.service = InferencePolicyService(service_config)
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"[INFO] serving on http://{args.host}:{args.port}")
    print(f"[INFO] policy={RequestHandler.service.model_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()

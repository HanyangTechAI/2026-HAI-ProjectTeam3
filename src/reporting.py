import html
import json
from datetime import datetime


def _fmt(value, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _json_block(data) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return html.escape(text)


def render_demo_report(output: dict) -> str:
    summary = output.get("summary", {})
    baselines = output.get("baselines", [])
    records = output.get("records", [])
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    baseline_rows = []
    for row in baselines:
        baseline_rows.append(
            "<tr>"
            f"<td>{html.escape(row['name'])}</td>"
            f"<td>{html.escape(row['action_label'])}</td>"
            f"<td>{_fmt(row['accuracy'])}</td>"
            f"<td>{_fmt(row['avg_reward'])}</td>"
            f"<td>{_fmt(row['avg_total_tokens'])}</td>"
            "</tr>"
        )

    record_rows = []
    for i, row in enumerate(records, start=1):
        reward = row["reward_breakdown"]["total_reward"] if row.get("reward_breakdown") else None
        record_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{html.escape(row['difficulty']['difficulty_level'])}</td>"
            f"<td>{html.escape(row['chosen_action_label'])}</td>"
            f"<td>{html.escape(str(row['pred']))}</td>"
            f"<td>{html.escape(str(row['gold']))}</td>"
            f"<td>{html.escape(str(row['correct']))}</td>"
            f"<td>{_fmt(reward) if reward is not None else ''}</td>"
            f"<td>{row['total_tokens']}</td>"
            f"<td>{html.escape(row['question'])}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RL Inference Policy Demo Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5f6b76;
      --line: #d7dee5;
      --panel: #f7f9fb;
      --accent: #0f766e;
      --accent-2: #9a3412;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 32px 40px 22px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f7fbfb, #ffffff);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 34px 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--panel);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric strong {{
      display: block;
      font-size: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef4f4;
      font-weight: 700;
    }}
    tr:nth-child(even) td {{
      background: #fafbfc;
    }}
    pre {{
      white-space: pre-wrap;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #111827;
      color: #f8fafc;
      padding: 16px;
      overflow: auto;
    }}
    .tag {{
      display: inline-block;
      border-radius: 999px;
      background: #dff4f1;
      color: var(--accent);
      padding: 3px 10px;
      font-size: 12px;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <header>
    <span class="tag">RL-based LLM inference policy optimization</span>
    <h1>추론 정책 시연 리포트</h1>
    <p>생성 시각: {html.escape(generated_at)} | checkpoint: {html.escape(summary.get('model_path', ''))}</p>
    <div class="metrics">
      <div class="metric"><span>Accuracy</span><strong>{_fmt(summary.get('accuracy', 0.0))}</strong></div>
      <div class="metric"><span>Average Reward</span><strong>{_fmt(summary.get('avg_reward', 0.0))}</strong></div>
      <div class="metric"><span>Average Tokens</span><strong>{_fmt(summary.get('avg_total_tokens', 0.0), 1)}</strong></div>
      <div class="metric"><span>Samples</span><strong>{summary.get('num_samples', 0)}</strong></div>
    </div>
  </header>
  <main>
    <h2>Baseline Comparison</h2>
    <table>
      <thead><tr><th>Policy</th><th>Action</th><th>Accuracy</th><th>Avg Reward</th><th>Avg Tokens</th></tr></thead>
      <tbody>{''.join(baseline_rows)}</tbody>
    </table>

    <h2>Policy Rollouts</h2>
    <table>
      <thead><tr><th>#</th><th>Difficulty</th><th>Chosen Action</th><th>Pred</th><th>Gold</th><th>Correct</th><th>Reward</th><th>Tokens</th><th>Question</th></tr></thead>
      <tbody>{''.join(record_rows)}</tbody>
    </table>

    <h2>Raw Summary</h2>
    <pre>{_json_block(summary)}</pre>
  </main>
</body>
</html>
"""

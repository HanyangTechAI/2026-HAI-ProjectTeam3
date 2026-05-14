const API_BASE = window.API_BASE || "http://127.0.0.1:8000";

const promptEl = document.getElementById("prompt");
const maxTokensEl = document.getElementById("maxTokens");
const forceMockEl = document.getElementById("forceMock");
const runBtn = document.getElementById("runBtn");
const strategyEl = document.getElementById("strategy");
const outputEl = document.getElementById("output");
const statsEl = document.getElementById("stats");

runBtn.addEventListener("click", run);
refreshStats();

async function run() {
  runBtn.disabled = true;
  outputEl.textContent = "Running...";
  try {
    const response = await fetch(`${API_BASE}/api/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: promptEl.value,
        maxCompletionTokens: Number(maxTokensEl.value || 512),
        forceMock: forceMockEl.checked,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || response.statusText);
    renderStrategy(data);
    outputEl.textContent = data.output + "\n\n" + JSON.stringify(data, null, 2);
    await refreshStats();
  } catch (err) {
    outputEl.textContent = String(err.message || err);
  } finally {
    runBtn.disabled = false;
  }
}

async function refreshStats() {
  try {
    const response = await fetch(`${API_BASE}/api/stats`);
    const data = await response.json();
    renderStats(data);
  } catch (err) {
    statsEl.innerHTML = card("Stats", "Unavailable");
  }
}

function renderStrategy(data) {
  strategyEl.innerHTML = [
    card("Task", data.analysis.taskType),
    card("Domain", data.analysis.domain),
    card("Risk", data.analysis.riskLevel),
    card("Model Route", data.strategy.modelRoute),
    card("Reasoning", data.strategy.reasoningDepth),
    card("Verify", String(data.strategy.verify)),
    card("Retry", data.strategy.retry),
    card("Compression", String(data.strategy.contextCompression)),
    card("Est. Cost", `$${data.estimatedCost.totalCostUsd.toFixed(8)}`),
    card("Provider Mode", data.providerMode),
    card("Decision", data.strategy.decisionReason),
  ].join("");
}

function renderStats(data) {
  statsEl.innerHTML = [
    card("Requests", data.totalRequests),
    card("Tokens", data.totalTokens),
    card("Estimated Cost", `$${Number(data.estimatedCostUsd || 0).toFixed(8)}`),
    card("Routes", JSON.stringify(data.routeCounts)),
    card("Tasks", JSON.stringify(data.taskCounts)),
  ].join("");
}

function card(label, value) {
  return `<div class="card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

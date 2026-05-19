const API_BASE = window.API_BASE || defaultApiBase();

function defaultApiBase() {
  if (!window.location.hostname) return "http://127.0.0.1:8000";
  if (window.location.port === "3000") {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return window.location.origin;
}

const promptEl = document.getElementById("prompt");
const maxTokensEl = document.getElementById("maxTokens");
const forceMockEl = document.getElementById("forceMock");
const runBtn = document.getElementById("runBtn");
const strategyEl = document.getElementById("strategy");
const responseTextEl = document.getElementById("responseText");
const outputEl = document.getElementById("output");
const statsEl = document.getElementById("stats");
const feedbackPanelEl = document.getElementById("feedbackPanel");
const reviewerIdEl = document.getElementById("reviewerId");
const feedbackCommentEl = document.getElementById("feedbackComment");
const feedbackStatusEl = document.getElementById("feedbackStatus");
const goodBtn = document.getElementById("goodBtn");
const badBtn = document.getElementById("badBtn");

let lastResponse = null;

reviewerIdEl.value = localStorage.getItem("reviewerId") || "";
runBtn.addEventListener("click", run);
goodBtn.addEventListener("click", () => sendFeedback(1));
badBtn.addEventListener("click", () => sendFeedback(-1));
refreshStats();

async function run() {
  runBtn.disabled = true;
  feedbackPanelEl.hidden = true;
  feedbackStatusEl.textContent = "";
  responseTextEl.textContent = "Running...";
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
    const data = await parseApiResponse(response);
    if (!response.ok) throw new Error(data.detail || response.statusText);
    lastResponse = data;
    renderStrategy(data);
    responseTextEl.textContent = data.output || "(empty response)";
    outputEl.textContent = JSON.stringify(data, null, 2);
    feedbackPanelEl.hidden = false;
    await refreshStats();
  } catch (err) {
    responseTextEl.textContent = String(err.message || err);
    outputEl.textContent = "";
  } finally {
    runBtn.disabled = false;
  }
}

async function sendFeedback(rating) {
  if (!lastResponse) return;
  const reviewerId = reviewerIdEl.value.trim();
  if (!reviewerId) {
    feedbackStatusEl.textContent = "Enter a reviewer ID first.";
    reviewerIdEl.focus();
    return;
  }
  localStorage.setItem("reviewerId", reviewerId);
  goodBtn.disabled = true;
  badBtn.disabled = true;
  feedbackStatusEl.textContent = "Saving feedback...";
  try {
    const response = await fetch(`${API_BASE}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        requestId: lastResponse.requestId,
        reviewerId,
        rating,
        qualityScore: rating === 1 ? 1 : 0,
        comment: feedbackCommentEl.value,
      }),
    });
    const data = await parseApiResponse(response);
    if (!response.ok) throw new Error(data.detail || response.statusText);
    feedbackStatusEl.textContent = `Saved feedback reward=${Number(data.reward).toFixed(2)}`;
    feedbackCommentEl.value = "";
    await refreshStats();
  } catch (err) {
    feedbackStatusEl.textContent = String(err.message || err);
  } finally {
    goodBtn.disabled = false;
    badBtn.disabled = false;
  }
}

async function refreshStats() {
  try {
    const response = await fetch(`${API_BASE}/api/stats`);
    const data = await parseApiResponse(response);
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

async function parseApiResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  const text = await response.text();
  return {
    detail: text || response.statusText || `HTTP ${response.status}`,
  };
}

function renderStats(data) {
  statsEl.innerHTML = [
    card("Requests", data.totalRequests),
    card("Feedback", data.totalFeedback || 0),
    card("Tokens", data.totalTokens),
    card("Estimated Cost", `$${Number(data.estimatedCostUsd || 0).toFixed(8)}`),
    card("Routes", JSON.stringify(data.routeCounts)),
    card("Tasks", JSON.stringify(data.taskCounts)),
    card("Ratings", JSON.stringify(data.feedbackCounts || {})),
    card("Reviewers", JSON.stringify(data.reviewerCounts || {})),
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

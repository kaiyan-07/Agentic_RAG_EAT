const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const statusEl = document.querySelector("#status");
const traceBody = document.querySelector("#trace-body");
const clearButton = document.querySelector("#clear-button");

let sessionId = window.localStorage.getItem("recipe-agent-session") || null;

function setStatus(text, busy = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("busy", busy);
}

function addMessage(role, content) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = content;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

function appendMessage(node, content) {
  node.textContent += content;
  messages.scrollTop = messages.scrollHeight;
}

function renderLiveEvent(event) {
  if (event.type !== "rag_step" && event.type !== "agent_step") return;
  const step = event.step || {};
  const label = step.label || step.name || event.type;
  const detail = step.detail || "";
  const icon = step.icon ? `${step.icon} ` : "";
  const current = traceBody.querySelector("ul.steps");
  if (!current) {
    traceBody.innerHTML = '<div class="trace-section"><h3>Live Steps</h3><ul class="steps"></ul></div>';
  }
  const list = traceBody.querySelector("ul.steps");
  const item = document.createElement("li");
  item.innerHTML = `<strong>${escapeHtml(icon + label)}</strong><br>${escapeHtml(detail)}`;
  list.appendChild(item);
}

function renderTrace(payload) {
  const trace = payload.tool_trace;
  const steps = payload.agent_steps || [];
  const stepItems = steps
    .map((step) => `<li><strong>${escapeHtml(step.name)}</strong> · ${escapeHtml(step.status)}<br>${escapeHtml(step.detail)}</li>`)
    .join("");
  const retrievedDocs = trace?.retrieved_docs || [];
  const docItems = retrievedDocs
    .map((doc) => `<li>#${doc.rank || "-"} ${escapeHtml(doc.dish_name || "-")}<br>${escapeHtml(doc.source || "-")}</li>`)
    .join("");

  const toolHtml = trace
    ? `
      <div class="trace-section">
        <h3>RAG Tool</h3>
        <dl>
          <dt>tool</dt><dd>${escapeHtml(trace.tool_name)}</dd>
          <dt>used</dt><dd>${trace.tool_used ? "yes" : "no"}</dd>
          <dt>hit</dt><dd>${trace.hit ? "yes" : "no"}</dd>
          <dt>query</dt><dd>${escapeHtml(trace.tool_query)}</dd>
          <dt>route</dt><dd>${escapeHtml(trace.route_type || "-")}</dd>
          <dt>rewrite</dt><dd>${escapeHtml(trace.rewritten_query || "-")}</dd>
          <dt>grade</dt><dd>${escapeHtml(JSON.stringify(trace.retrieval_grade || {}))}</dd>
          <dt>rewrite retrieval</dt><dd>${trace.rewrite_triggered ? escapeHtml(trace.rewrite_strategy || "-") : "no"}</dd>
          <dt>expanded query</dt><dd>${escapeHtml(trace.expanded_query || "-")}</dd>
          <dt>step-back question</dt><dd>${escapeHtml(trace.step_back_question || "-")}</dd>
          <dt>hyde doc</dt><dd>${escapeHtml(trace.hypothetical_document || "-")}</dd>
          <dt>filters</dt><dd>${escapeHtml(JSON.stringify(trace.filters || {}))}</dd>
          <dt>retrieval</dt><dd>${escapeHtml(trace.retrieval_mode || "-")}</dd>
          <dt>rerank</dt><dd>${trace.rerank_applied ? "yes" : "no"}</dd>
          <dt>elapsed</dt><dd>${trace.elapsed_ms ?? "-"} ms</dd>
          <dt>chunks</dt><dd>${trace.retrieved_chunk_count}</dd>
          <dt>sections</dt><dd>${escapeHtml((trace.retrieved_chunk_summaries || []).join(", ") || "-")}</dd>
          <dt>parents</dt><dd>${trace.retrieved_parent_count}</dd>
          <dt>dishes</dt><dd>${escapeHtml((trace.retrieved_dish_names || []).join(", ") || "-")}</dd>
          <dt>sources</dt><dd>${escapeHtml((trace.retrieved_sources || []).join("\n") || "-")}</dd>
        </dl>
        ${docItems ? `<ul class="steps">${docItems}</ul>` : ""}
      </div>
    `
    : `
      <div class="trace-section">
        <h3>RAG Tool</h3>
        <p class="muted">本轮没有调用知识库工具。</p>
      </div>
    `;

  traceBody.innerHTML = `
    <div class="trace-section">
      <h3>Agent Steps</h3>
      <ul class="steps">${stepItems}</ul>
    </div>
    ${toolHtml}
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  input.value = "";
  addMessage("user", message);
  setStatus("thinking", true);

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        include_trace: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const assistantNode = addMessage("assistant", "");
    let buffer = "";
    const decoder = new TextDecoder();
    const reader = response.body.getReader();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const rawEvent of events) {
        const line = rawEvent.split("\n").find((item) => item.startsWith("data: "));
        if (!line) continue;
        const event = JSON.parse(line.slice(6));

        if (event.type === "token") {
          appendMessage(assistantNode, event.content || "");
        } else if (event.type === "rag_step" || event.type === "agent_step") {
          renderLiveEvent(event);
        } else if (event.type === "final") {
          const payload = event.response;
          sessionId = payload.session_id;
          window.localStorage.setItem("recipe-agent-session", sessionId);
          assistantNode.textContent = payload.answer;
          renderTrace(payload);
        } else if (event.type === "error") {
          throw new Error(event.message || "stream error");
        }
      }
    }
    setStatus("ready");
  } catch (error) {
    addMessage("assistant", "请求失败，请确认后端服务已启动。");
    setStatus("error");
  }
});

clearButton.addEventListener("click", () => {
  messages.innerHTML = "";
  traceBody.innerHTML = '<p class="muted">等待一次对话后显示 Agent 步骤和 RAG 检索路径。</p>';
  sessionId = null;
  window.localStorage.removeItem("recipe-agent-session");
  setStatus("ready");
});

addMessage("assistant", "你好，我已经把现有 RAG 作为工具接入。可以问我：今晚想吃清淡一点，有什么简单菜？");

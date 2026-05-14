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
}

function renderTrace(payload) {
  const trace = payload.tool_trace;
  const steps = payload.agent_steps || [];
  const stepItems = steps
    .map((step) => `<li><strong>${escapeHtml(step.name)}</strong> · ${escapeHtml(step.status)}<br>${escapeHtml(step.detail)}</li>`)
    .join("");

  const toolHtml = trace
    ? `
      <div class="trace-section">
        <h3>RAG Tool</h3>
        <dl>
          <dt>tool</dt><dd>${escapeHtml(trace.tool_name)}</dd>
          <dt>query</dt><dd>${escapeHtml(trace.tool_query)}</dd>
          <dt>route</dt><dd>${escapeHtml(trace.route_type || "-")}</dd>
          <dt>rewrite</dt><dd>${escapeHtml(trace.rewritten_query || "-")}</dd>
          <dt>filters</dt><dd>${escapeHtml(JSON.stringify(trace.filters || {}))}</dd>
          <dt>chunks</dt><dd>${trace.retrieved_chunk_count}</dd>
          <dt>parents</dt><dd>${trace.retrieved_parent_count}</dd>
          <dt>dishes</dt><dd>${escapeHtml((trace.retrieved_dish_names || []).join(", ") || "-")}</dd>
          <dt>sources</dt><dd>${escapeHtml((trace.retrieved_sources || []).join("\n") || "-")}</dd>
        </dl>
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
    const response = await fetch("/api/chat", {
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

    const payload = await response.json();
    sessionId = payload.session_id;
    window.localStorage.setItem("recipe-agent-session", sessionId);
    addMessage("assistant", payload.answer);
    renderTrace(payload);
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

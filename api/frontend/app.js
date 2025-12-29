// api/frontend/app.js
// Queue-first: POST /image/jobs -> poll /image/jobs/{id} -> show output_url
// Chat: POST /chat?prompt=...

const API_BASE = ""; // same-origin recommended

const els = {
  apiDot: document.getElementById("apiDot"),
  apiText: document.getElementById("apiText"),

  // image
  prompt: document.getElementById("prompt"),
  steps: document.getElementById("steps"),
  guidance: document.getElementById("guidance"),
  width: document.getElementById("width"),
  height: document.getElementById("height"),
  seed: document.getElementById("seed"),
  model: document.getElementById("model"),

  btnGenerate: document.getElementById("btnGenerate"),
  btnClear: document.getElementById("btnClear"),
  imgResult: document.getElementById("imgResult"),
  imgMeta: document.getElementById("imgMeta"),
  jobMeta: document.getElementById("jobMeta"),

  // chat
  chatInput: document.getElementById("chatInput"),
  btnChatSend: document.getElementById("btnChatSend"),
  btnChatClear: document.getElementById("btnChatClear"),
  chatLog: document.getElementById("chatLog"),
};

const state = {
  lastJobId: null,
  lastImageUrl: null,
  healthOk: false,
  polling: false,
};

function setStatus(ok, text) {
  els.apiDot.classList.toggle("ok", !!ok);
  els.apiDot.classList.toggle("bad", !ok);
  els.apiText.textContent = text;
}

async function healthCheck() {
  try {
    const r = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    setStatus(true, "ok");
    state.healthOk = true;
  } catch {
    setStatus(false, "down");
    state.healthOk = false;
  }
}

function escapeHtml(s) {
  return (s ?? "")
    .toString()
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function nowStr() {
  return new Date().toLocaleString();
}

function setBusyImage(isBusy, label) {
  els.btnGenerate.disabled = !!isBusy;
  els.btnGenerate.textContent = isBusy ? (label || "Working…") : "Generate";
}

function imageHint(html) {
  els.imgResult.innerHTML = html;
}

function renderImage(url, jobId) {
  state.lastImageUrl = url;

  const safeUrl = escapeHtml(url);
  const safeJob = escapeHtml(jobId || "");

  imageHint(`
    <div class="hint">Done • ${safeJob}</div>
    <img class="img" src="${safeUrl}" alt="generated"/>
    <div class="hint small">Direct link: <span style="font-family:var(--mono)">${safeUrl}</span></div>
  `);
  els.imgMeta.textContent = `OK • ${nowStr()}`;
}

function renderError(err, jobId) {
  const safe = escapeHtml(err);
  const safeJob = escapeHtml(jobId || "");
  imageHint(`
    <div class="error">Job failed • ${safeJob}</div>
    <pre class="pre">${safe}</pre>
  `);
  els.imgMeta.textContent = `ERROR • ${nowStr()}`;
}

function toInt(val, fallback) {
  const n = parseInt(val, 10);
  return Number.isFinite(n) ? n : fallback;
}
function toFloat(val, fallback) {
  const n = parseFloat(val);
  return Number.isFinite(n) ? n : fallback;
}

function chatAppend(role, text, extra = {}) {
  const wrap = document.createElement("div");
  wrap.className = `chatmsg ${role}`;

  const head = document.createElement("div");
  head.className = "chatmsg-head";
  head.textContent = role === "user" ? "You" : "SyMon";

  const body = document.createElement("div");
  body.className = "chatmsg-body";
  body.innerHTML = escapeHtml(text);

  wrap.appendChild(head);
  wrap.appendChild(body);

  // Optional: clickable suggestions
  if (extra && Array.isArray(extra.suggestions) && extra.suggestions.length) {
    const sugWrap = document.createElement("div");
    sugWrap.style.marginTop = "8px";
    sugWrap.style.display = "flex";
    sugWrap.style.flexWrap = "wrap";
    sugWrap.style.gap = "8px";

    extra.suggestions.slice(0, 8).forEach((s) => {
      const b = document.createElement("button");
      b.className = "btn";
      b.style.padding = "8px 10px";
      b.style.fontSize = "12px";
      b.textContent = s;
      b.addEventListener("click", () => {
        els.prompt.value = s;
        els.prompt.focus();
        chatAppend("bot", `Loaded prompt into Image box.`);
      });
      sugWrap.appendChild(b);
    });

    wrap.appendChild(sugWrap);
  }

  els.chatLog.appendChild(wrap);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function extractPromptSuggestions(text) {
  // very light heuristic: lines starting with - or numbered
  const lines = (text || "").split("\n").map((l) => l.trim()).filter(Boolean);
  const sug = [];
  for (const l of lines) {
    const m = l.match(/^(\d+[\).\]]\s+|-+\s+)(.+)$/);
    if (m && m[2]) sug.push(m[2].trim());
  }
  return sug;
}

async function createImageJob(payload) {
  const r = await fetch(`${API_BASE}/image/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!r.ok) {
    let t = "";
    try { t = await r.text(); } catch {}
    throw new Error(t || `HTTP ${r.status}`);
  }

  return await r.json();
}

async function fetchJob(jobId) {
  const r = await fetch(`${API_BASE}/image/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
  if (!r.ok) {
    let t = "";
    try { t = await r.text(); } catch {}
    throw new Error(t || `HTTP ${r.status}`);
  }
  return await r.json();
}

async function pollJob(jobId, { intervalMs = 1200, timeoutMs = 5 * 60 * 1000 } = {}) {
  if (state.polling) return;
  state.polling = true;

  const t0 = Date.now();
  let spins = 0;

  try {
    while (true) {
      spins++;
      els.jobMeta.textContent = `Job: ${jobId} • polling… (${spins})`;

      const data = await fetchJob(jobId);
      const job = data?.job || {};
      const st = job.status;

      if (st === "done") {
        const url = job.output_url;
        if (!url) {
          renderError("Job marked done but no output_url returned.", jobId);
        } else {
          renderImage(url, jobId);
        }
        return;
      }

      if (st === "error") {
        renderError(job.error || "Unknown worker error", jobId);
        return;
      }

      // queued/running
      const elapsed = Math.floor((Date.now() - t0) / 1000);
      imageHint(`<div class="hint">Working… (${escapeHtml(st || "queued")}) • ${elapsed}s</div>`);
      els.imgMeta.textContent = `IN PROGRESS • ${nowStr()}`;

      if (Date.now() - t0 > timeoutMs) {
        renderError(`Timed out after ${Math.floor(timeoutMs / 1000)}s. Job still ${st}.`, jobId);
        return;
      }

      await new Promise((res) => setTimeout(res, intervalMs));
    }
  } finally {
    state.polling = false;
    setBusyImage(false);
  }
}

async function doGenerate() {
  const prompt = (els.prompt.value || "").trim();
  if (!prompt) return;

  // Build params
  const steps = toInt(els.steps.value, 28);
  const guidance = toFloat(els.guidance.value, 7.5);
  const width = toInt(els.width.value, 512);
  const height = toInt(els.height.value, 512);
  const seedRaw = (els.seed.value || "").trim();
  const seed = seedRaw ? toInt(seedRaw, null) : null;
  const model_id = (els.model.value || "").trim() || null;

  setBusyImage(true, "Queued…");
  els.imgMeta.textContent = "";
  els.jobMeta.textContent = "";
  imageHint(`<div class="hint">Submitting job to queue…</div>`);

  try {
    const payload = { prompt, steps, guidance, width, height, seed, model_id };
    const created = await createImageJob(payload);

    const jobId = created.job_id;
    state.lastJobId = jobId;

    els.imgMeta.textContent = `QUEUED • ${nowStr()}`;
    els.jobMeta.textContent = `Job: ${jobId}`;

    setBusyImage(true, "Running…");
    await pollJob(jobId);
  } catch (e) {
    renderError(String(e?.message || e), state.lastJobId || "");
    setBusyImage(false);
  }
}

async function doChat() {
  const msg = (els.chatInput.value || "").trim();
  if (!msg) return;

  chatAppend("user", msg);
  els.chatInput.value = "";
  els.btnChatSend.disabled = true;
  els.btnChatSend.textContent = "Sending…";

  try {
    const url = `${API_BASE}/chat?prompt=${encodeURIComponent(msg)}`;
    const r = await fetch(url, { method: "POST" });

    if (!r.ok) {
      const t = await r.text();
      throw new Error(t || `HTTP ${r.status}`);
    }

    const data = await r.json();
    const reply =
      data.reply ||
      data.assistant ||
      data.message ||
      data.text ||
      JSON.stringify(data);

    const suggestions = extractPromptSuggestions(reply);
    chatAppend("bot", reply, { suggestions });
  } catch (e) {
    chatAppend("bot", `Error: ${String(e?.message || e)}`);
  } finally {
    els.btnChatSend.disabled = false;
    els.btnChatSend.textContent = "Send";
  }
}

// Wire up
els.btnGenerate.addEventListener("click", doGenerate);
els.btnClear.addEventListener("click", () => {
  els.prompt.value = "";
  els.imgMeta.textContent = "No job yet.";
  els.jobMeta.textContent = "";
  imageHint(`<div class="hint">Generated image will appear here after the worker completes the job.</div>`);
});

els.btnChatSend.addEventListener("click", doChat);
els.btnChatClear.addEventListener("click", () => {
  els.chatLog.innerHTML = "";
  chatAppend("bot", "Cleared.");
});

els.chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    doChat();
  }
});

// Start
healthCheck();
setInterval(healthCheck, 5000);
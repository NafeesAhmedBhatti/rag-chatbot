/**
 * RAG Chatbot — Frontend Application
 *
 * Features:
 *   - Three-panel layout (sidebar + chat + context)
 *   - Chat with streaming-effect answer display
 *   - Markdown rendering (lightweight)
 *   - Citation badges with filename + page
 *   - Retrieved chunks panel with scores, expand/collapse
 *   - PDF upload via drag-and-drop or file picker
 *   - Pipeline stats (live from /api/stats)
 *   - Response latency display
 */

(function () {
  "use strict";

  // ============ DOM ============
  const chatMessages = document.getElementById("chat-messages");
  const welcomeScreen = document.getElementById("welcome-screen");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const chunksContainer = document.getElementById("chunks-container");
  const chunkCountBadge = document.getElementById("chunk-count-badge");
  const latencyInfo = document.getElementById("latency-info");
  const uploadInput = document.getElementById("upload-input");
  const uploadStatus = document.getElementById("upload-status");
  const pdfList = document.getElementById("pdf-list");

  // ============ State ============
  let isProcessing = false;

  // ============ Init ============
  init();

  function init() {
    sendBtn.addEventListener("click", sendMessage);
    chatInput.addEventListener("keydown", handleInputKeydown);
    chatInput.addEventListener("input", autoResize);
    uploadInput.addEventListener("change", handleUpload);

    // Example questions
    document.querySelectorAll(".example-q").forEach((btn) => {
      btn.addEventListener("click", () => {
        chatInput.value = btn.dataset.q;
        autoResize();
        sendMessage();
      });
    });

    loadStats();
    loadPdfList();
    setInterval(loadStats, 30000);
  }

  // ============ Stats ============

  async function loadStats() {
    try {
      const resp = await fetch("/api/stats");
      if (!resp.ok) return;
      const data = await resp.json();

      document.getElementById("stat-docs").textContent = data.total_documents;
      document.getElementById("stat-chunks").textContent = formatNumber(data.total_chunks);
      document.getElementById("stat-size").textContent = data.index_size_mb.toFixed(2);
      document.getElementById("stat-dims").textContent = data.vector_dimensions;
      document.getElementById("model-llm").textContent = data.llm_model;
    } catch (err) {
      // Silent fail — stats are non-critical
    }
  }

  async function loadPdfList() {
    try {
      const resp = await fetch("/api/documents");
      if (!resp.ok) return;
      const data = await resp.json();

      if (data.total === 0) {
        pdfList.innerHTML = '<div class="pdf-item-loading">No PDFs ingested yet</div>';
        return;
      }

      pdfList.innerHTML = "";
      data.documents.forEach((doc) => {
        const item = document.createElement("div");
        item.className = "pdf-item";
        item.innerHTML =
          '<span class="pdf-icon">📄</span>' +
          '<span class="pdf-name" title="' + escapeHtml(doc.filename) +
          '">' + escapeHtml(doc.filename) + "</span>" +
          '<span class="pdf-pages">' + doc.pages + "p</span>";
        pdfList.appendChild(item);
      });
    } catch (err) {
      pdfList.innerHTML = '<div class="pdf-item-loading">Unable to load</div>';
    }
  }

  // ============ Chat ============

  function handleInputKeydown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function autoResize() {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
    sendBtn.disabled = !chatInput.value.trim() || isProcessing;
  }

  async function sendMessage() {
    const question = chatInput.value.trim();
    if (!question || isProcessing) return;

    // Hide welcome screen
    if (welcomeScreen) welcomeScreen.style.display = "none";

    isProcessing = true;
    sendBtn.disabled = true;

    // Render user message
    renderMessage("user", question);

    // Clear input
    chatInput.value = "";
    autoResize();

    // Show typing indicator
    const typingEl = renderTypingIndicator();

    // Clear chunks panel
    chunksContainer.innerHTML = "";

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || ("HTTP " + resp.status));
      }

      const data = await resp.json();

      // Remove typing indicator
      typingEl.remove();

      // Render answer with streaming effect
      await renderAnswer(data);

      // Render chunks
      renderChunks(data.retrieved_chunks);

      // Render latency info
      renderLatency(data);

    } catch (err) {
      typingEl.remove();
      renderMessage("assistant", "⚠️ Error: " + err.message);
    } finally {
      isProcessing = false;
      sendBtn.disabled = false;
      chatInput.focus();
    }
  }

  function renderMessage(role, content) {
    const el = document.createElement("div");
    el.className = "msg";

    const avatar = role === "user" ? "🧑" : "🤖";
    const label = role === "user" ? "You" : "Assistant";

    el.innerHTML =
      '<div class="msg-label ' + role + '">' +
      '<div class="avatar">' + avatar + '</div>' +
      label +
      "</div>" +
      '<div class="msg-bubble ' + role + '">' +
      escapeHtml(content) +
      "</div>";

    chatMessages.appendChild(el);
    scrollToBottom();
    return el;
  }

  function renderTypingIndicator() {
    const el = document.createElement("div");
    el.className = "msg";
    el.innerHTML =
      '<div class="msg-label assistant">' +
      '<div class="avatar">🤖</div>Assistant</div>' +
      '<div class="msg-bubble assistant">' +
      '<div class="typing-indicator">' +
      '<div class="typing-dot"></div>' +
      '<div class="typing-dot"></div>' +
      '<div class="typing-dot"></div>' +
      "</div></div>";
    chatMessages.appendChild(el);
    scrollToBottom();
    return el;
  }

  async function renderAnswer(data) {
    const el = document.createElement("div");
    el.className = "msg";

    el.innerHTML =
      '<div class="msg-label assistant">' +
      '<div class="avatar">🤖</div>Assistant</div>' +
      '<div class="msg-bubble assistant" id="answer-bubble"></div>';

    chatMessages.appendChild(el);
    const bubble = el.querySelector("#answer-bubble");
    bubble.id = "";

    // Stream the answer text character by character
    await streamText(bubble, data.answer);

    // Add citation badges
    if (data.citations && data.citations.length > 0) {
      const badgesDiv = document.createElement("div");
      badgesDiv.className = "citation-badges";

      data.citations.forEach((cite) => {
        const badge = document.createElement("span");
        badge.className = "citation-badge";
        badge.innerHTML =
          '<svg class="cite-icon" viewBox="0 0 12 12" fill="none">' +
          '<path d="M2 4C2 2.9 2.9 2 4 2H5M10 8C10 9.1 9.1 10 8 10H7" ' +
          'stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>' +
          "</svg>" +
          escapeHtml(cite.filename) + " · p." + cite.page;
        badgesDiv.appendChild(badge);
      });

      bubble.appendChild(badgesDiv);
    }

    // Add copy button
    const actionsDiv = document.createElement("div");
    actionsDiv.className = "msg-actions";
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.innerHTML =
      '<svg width="14" height="14" viewBox="0 0 14 14" fill="none">' +
      '<rect x="4" y="4" width="8" height="8" rx="1.5" stroke="currentColor" stroke-width="1.2"/>' +
      '<path d="M2 9V2.5C2 2.224 2.224 2 2.5 2H9" stroke="currentColor" stroke-width="1.2"/>' +
      "</svg> Copy";
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(data.answer).then(() => {
        copyBtn.classList.add("copied");
        copyBtn.innerHTML =
          '<svg width="14" height="14" viewBox="0 0 14 14" fill="none">' +
          '<path d="M3 7L6 10L11 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
          "</svg> Copied!";
        setTimeout(() => {
          copyBtn.classList.remove("copied");
          copyBtn.innerHTML =
            '<svg width="14" height="14" viewBox="0 0 14 14" fill="none">' +
            '<rect x="4" y="4" width="8" height="8" rx="1.5" stroke="currentColor" stroke-width="1.2"/>' +
            '<path d="M2 9V2.5C2 2.224 2.224 2 2.5 2H9" stroke="currentColor" stroke-width="1.2"/>' +
            "</svg> Copy";
        }, 2000);
      });
    });
    actionsDiv.appendChild(copyBtn);
    el.appendChild(actionsDiv);

    scrollToBottom();
  }

  async function streamText(element, text) {
    const words = text.split(" ");
    const chunkSize = 2; // words per tick
    let i = 0;

    return new Promise((resolve) => {
      function tick() {
        if (i >= words.length) {
          resolve();
          return;
        }
        const slice = words.slice(i, i + chunkSize).join(" ");
        element.innerHTML = renderMarkdown(words.slice(0, i + chunkSize).join(" "));
        i += chunkSize;
        scrollToBottom();
        setTimeout(tick, 20);
      }
      tick();
    });
  }

  // ============ Chunks Panel ============

  function renderChunks(chunks) {
    if (!chunks || chunks.length === 0) {
      chunksContainer.innerHTML =
        '<div class="chunks-placeholder"><p>No chunks met the score threshold for this query.</p></div>';
      chunkCountBadge.style.display = "none";
      return;
    }

    chunkCountBadge.style.display = "inline-flex";
    chunkCountBadge.textContent = chunks.length;

    chunksContainer.innerHTML = "";

    chunks.forEach((chunk, idx) => {
      const rank = idx + 1;
      const scoreClass = chunk.score >= 0.5 ? "high" : chunk.score >= 0.35 ? "mid" : "low";

      const card = document.createElement("div");
      card.className = "chunk-card";
      card.innerHTML =
        '<div class="chunk-header">' +
        '<div class="chunk-header-left">' +
        '<div class="chunk-rank r' + rank + '">' + rank + "</div>" +
        '<span class="chunk-source" title="' + escapeHtml(chunk.source_file) + '">' +
        escapeHtml(chunk.source_file) + "</span>" +
        '<span class="chunk-page">p.' + chunk.page + "</span>" +
        "</div>" +
        '<div class="chunk-score-bar">' +
        '<div class="score-dot ' + scoreClass + '"></div>' +
        '<span class="chunk-score-value">' + chunk.score.toFixed(4) + "</span>" +
        '<svg class="chunk-expand-icon" viewBox="0 0 16 16" fill="none">' +
        '<path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' +
        "</svg>" +
        "</div>" +
        "</div>" +
        '<div class="chunk-body">' +
        '<p>' + escapeHtml(chunk.text) + "</p>" +
        '<div class="chunk-meta">' +
        '<span class="chunk-meta-item">tokens: ' + (chunk.token_count || "—") + "</span>" +
        '<span class="chunk-meta-item">chunk: ' + escapeHtml(chunk.chunk_id) + "</span>" +
        "</div>" +
        "</div>";

      // Expand/collapse
      const header = card.querySelector(".chunk-header");
      header.addEventListener("click", () => {
        card.classList.toggle("expanded");
      });

      chunksContainer.appendChild(card);
    });
  }

  // ============ Latency Info ============

  function renderLatency(data) {
    latencyInfo.innerHTML =
      '<span class="latency-pill">⏱ ' + data.latency_ms.toFixed(0) + "ms total</span>" +
      '<span class="latency-pill">🔍 ' + data.retrieval_latency_ms.toFixed(0) + "ms retrieve</span>" +
      '<span class="latency-pill">🧠 ' + data.generation_latency_ms.toFixed(0) + "ms generate</span>" +
      '<span class="latency-pill">top-k: ' + data.top_k + "</span>" +
      '<span class="latency-pill">model: ' + escapeHtml(data.model) + "</span>";
  }

  // ============ Upload ============

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    uploadStatus.textContent = "Uploading " + file.name + "…";
    uploadStatus.className = "upload-status loading";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch("/api/ingest", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || ("HTTP " + resp.status));
      }

      const data = await resp.json();
      uploadStatus.textContent =
        "✓ " + data.filename + ": " + data.pages + " pages, " + data.chunks + " chunks";
      uploadStatus.className = "upload-status success";

      // Refresh stats
      loadStats();
      loadPdfList();
    } catch (err) {
      uploadStatus.textContent = "✗ " + err.message;
      uploadStatus.className = "upload-status error";
    } finally {
      uploadInput.value = "";
    }
  }

  // ============ Utilities ============

  function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function renderMarkdown(text) {
    if (!text) return "";
    // Lightweight markdown: bold, italic, code, line breaks
    let html = escapeHtml(text);
    // Code blocks (triple backtick)
    html = html.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // Italic
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    // Line breaks
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  function formatNumber(n) {
    if (n >= 1000) return (n / 1000).toFixed(1) + "K";
    return String(n);
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }
})();

/**
 * app.js - Omago AI Interactive Frontend Controller
 * Connects UI actions to the LangGraph Python FastAPI backend on localhost.
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- State ---
  let currentThreadId = `omago-session-${Math.random().toString(36).substring(2, 9)}`;
  let selectedMode = "default";
  let deepSearchActive = false;
  let webSearchActive = false;
  let attachedFileContent = null;
  let isSending = false;
  const conversationHistory = [];

  // --- DOM Elements ---
  const chatHero = document.getElementById("chat-hero");
  const chatMessages = document.getElementById("chat-messages");
  const chatTextarea = document.getElementById("chat-textarea");
  const sendBtn = document.getElementById("send-btn");
  const attachBtn = document.getElementById("attach-btn");
  const fileInputHidden = document.getElementById("file-input-hidden");
  const attachmentPreview = document.getElementById("attachment-preview");
  const removeAttachBtn = document.getElementById("remove-attach-btn");
  const deepSearchPill = document.getElementById("deep-search-pill");
  const webSearchPill = document.getElementById("web-search-pill");
  const voiceBtn = document.getElementById("voice-btn");
  const modePills = document.querySelectorAll(".mode-pill");
  const newChatBtn = document.getElementById("new-chat-btn");
  const threadDisplay = document.getElementById("current-thread-display");
  const navLinks = document.querySelectorAll(".nav-link");
  const tabViews = document.querySelectorAll(".tab-view");
  const preferencesList = document.getElementById("preferences-list");
  const leadsTableBody = document.getElementById("leads-table-body");
  const queriesCounter = document.getElementById("queries-counter");
  const leadsThreadId = document.getElementById("leads-thread-id");
  const copyEmbedBtn = document.getElementById("copy-embed-btn");
  const floatingLauncher = document.getElementById("floating-launcher");
  const shortcutsModal = document.getElementById("shortcuts-modal");
  const closeModalBtn = document.getElementById("close-modal-btn");
  const chipButtons = document.querySelectorAll(".chip-btn");

  // Initialize display
  updateThreadDisplay();
  initEarthBackgroundCanvas();

  function initEarthBackgroundCanvas() {
    const canvas = document.getElementById("earth-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    const TOTAL_FRAMES = 400;
    const FRAME_RATE = 24;
    const frameInterval = 1000 / FRAME_RATE;
    const images = new Array(TOTAL_FRAMES);
    let currentFrameIndex = 0;
    let lastTimestamp = 0;

    const getFrameUrl = (idx) => `/earth-frames/frame_${String(idx + 1).padStart(4, "0")}.jpg`;

    const INITIAL_CHUNK = 40;
    const loadImage = (i) => {
      if (images[i]) return;
      const img = new Image();
      img.src = getFrameUrl(i);
      img.onload = () => { images[i] = img; };
    };

    for (let i = 0; i < INITIAL_CHUNK; i++) loadImage(i);
    setTimeout(() => {
      for (let i = INITIAL_CHUNK; i < TOTAL_FRAMES; i++) loadImage(i);
    }, 400);

    const handleResize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = window.innerWidth + "px";
      canvas.style.height = window.innerHeight + "px";
    };
    window.addEventListener("resize", handleResize);
    handleResize();

    const render = (timestamp) => {
      if (!lastTimestamp) lastTimestamp = timestamp;
      const elapsed = timestamp - lastTimestamp;

      if (elapsed >= frameInterval) {
        lastTimestamp = timestamp - (elapsed % frameInterval);

        let frameToDraw = images[currentFrameIndex];
        if (!frameToDraw) {
          for (let k = currentFrameIndex - 1; k >= 0; k--) {
            if (images[k]) { frameToDraw = images[k]; break; }
          }
        }

        if (frameToDraw && frameToDraw.complete && frameToDraw.naturalWidth) {
          const cw = canvas.width;
          const ch = canvas.height;
          const iw = frameToDraw.naturalWidth;
          const ih = frameToDraw.naturalHeight;
          const scale = Math.max(cw / iw, ch / ih);
          const dw = iw * scale;
          const dh = ih * scale;
          const dx = (cw - dw) / 2;
          const dy = (ch - dh) / 2;
          ctx.drawImage(frameToDraw, dx, dy, dw, dh);
        }

        currentFrameIndex = (currentFrameIndex + 1) % TOTAL_FRAMES;
      }
      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  }

  // --- Auto-resize Textarea ---
  chatTextarea.addEventListener("input", () => {
    chatTextarea.style.height = "auto";
    chatTextarea.style.height = Math.min(chatTextarea.scrollHeight, 180) + "px";
  });

  // --- Enter to Submit (Shift+Enter for newline) ---
  chatTextarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  sendBtn.addEventListener("click", handleSendMessage);

  // --- Mode Pills Handling ---
  modePills.forEach((pill) => {
    pill.addEventListener("click", () => {
      const mode = pill.getAttribute("data-mode");
      if (selectedMode === mode) {
        selectedMode = "default";
        pill.classList.remove("selected");
      } else {
        modePills.forEach((p) => p.classList.remove("selected"));
        pill.classList.add("selected");
        selectedMode = mode;
      }
    });
  });

  // --- Deep Search & Web Search Toggles ---
  deepSearchPill.addEventListener("click", () => {
    deepSearchActive = !deepSearchActive;
    deepSearchPill.classList.toggle("active", deepSearchActive);
  });

  webSearchPill.addEventListener("click", () => {
    webSearchActive = !webSearchActive;
    webSearchPill.classList.toggle("active", webSearchActive);
  });

  // --- File Attachment ---
  attachBtn.addEventListener("click", () => fileInputHidden.click());

  fileInputHidden.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      attachedFileContent = {
        name: file.name,
        text: event.target.result,
      };
      attachmentPreview.querySelector(".attachment-name").textContent = `📄 ${file.name}`;
      attachmentPreview.style.display = "inline-flex";
    };
    reader.readAsText(file);
  });

  removeAttachBtn.addEventListener("click", () => {
    attachedFileContent = null;
    fileInputHidden.value = "";
    attachmentPreview.style.display = "none";
  });

  // --- Voice Input (Web Speech API) ---
  let recognition = null;
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      chatTextarea.value = (chatTextarea.value + " " + transcript).trim();
      chatTextarea.dispatchEvent(new Event("input"));
    };

    recognition.onend = () => {
      voiceBtn.classList.remove("recording");
    };

    voiceBtn.addEventListener("click", () => {
      if (voiceBtn.classList.contains("recording")) {
        recognition.stop();
        voiceBtn.classList.remove("recording");
      } else {
        try {
          recognition.start();
          voiceBtn.classList.add("recording");
        } catch (err) {
          console.warn("Speech recognition error:", err);
        }
      }
    });
  } else {
    voiceBtn.addEventListener("click", () => {
      alert("Speech Recognition is not supported in this browser. Please type your query.");
    });
  }

  // --- Send Message & LangGraph Stream Interaction ---
  async function handleSendMessage() {
    if (isSending) return;
    const query = chatTextarea.value.trim();
    if (!query) return;

    isSending = true;
    chatTextarea.value = "";
    chatTextarea.style.height = "auto";

    // Build complete message payload (including attachment if present)
    let fullMessage = query;
    if (attachedFileContent) {
      fullMessage = `[Attached Context from file '${attachedFileContent.name}']:\n\`\`\`\n${attachedFileContent.text}\n\`\`\`\n\nQuestion: ${query}`;
      // Clear attachment after sending
      attachedFileContent = null;
      attachmentPreview.style.display = "none";
      fileInputHidden.value = "";
    }

    // Switch view to conversation stream
    showConversationView();

    // Render User Bubble
    appendUserMessage(query);

    // Render Loading Indicator
    const loadingElem = appendLoadingIndicator();

    // Scroll to bottom
    scrollToBottom();

    // Add to Leads log table
    addQueryToLeads(query, selectedMode);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: fullMessage,
          thread_id: currentThreadId,
          mode: selectedMode,
          deep_search: deepSearchActive,
          web_search: webSearchActive,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      currentThreadId = data.thread_id;
      updateThreadDisplay();

      // Remove loading indicator
      loadingElem.remove();

      // Render Tutor Reply
      appendTutorReply(data.reply, data.mode);

      // Update session preferences in Training Tab if returned
      if (data.session_preferences && data.session_preferences.length > 0) {
        updatePreferencesUI(data.session_preferences);
      }

    } catch (err) {
      console.error("Chat error:", err);
      loadingElem.remove();
      appendTutorReply(
        `**Connection / Engine Note:** Could not retrieve response from the tutor.\n\n*Error details:* \`${err.message}\`\n\nPlease ensure your \`.env\` has a valid \`GROQ_API_KEY\` and that the localhost server is active.`,
        "error"
      );
    } finally {
      isSending = false;
      scrollToBottom();
      chatTextarea.focus();
    }
  }

  // --- UI Render Helpers ---
  function showConversationView() {
    chatHero.style.display = "none";
    chatMessages.style.display = "flex";
  }

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(row);
  }

  function appendLoadingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.innerHTML = `
      <div class="typing-indicator-box">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    `;
    chatMessages.appendChild(row);
    return row;
  }

  function appendTutorReply(markdownContent, mode) {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    // Format mode display tag
    let modeLabel = "First-Principles AI Tutor";
    if (mode === "fast") modeLabel = "⚡ Fast / TL;DR Mode";
    else if (mode === "in-depth") modeLabel = "💡 Deep Dive Mode";
    else if (mode === "step-by-step") modeLabel = "🔀 Step-by-Step Mode";
    else if (mode === "holistic") modeLabel = "🧊 Holistic First-Principles";

    // Parse markdown safely
    const parsedHtml = (typeof marked !== "undefined")
      ? marked.parse(markdownContent)
      : `<p>${escapeHtml(markdownContent)}</p>`;

    row.innerHTML = `
      <div class="tutor-card">
        <div class="tutor-header">
          <div class="tutor-identity">
            <div class="tutor-badge-orb">
              <svg viewBox="0 0 100 100" fill="none">
                <circle cx="50" cy="50" r="46" fill="white"/>
                <path d="M50 14C30.1178 14 14 30.1178 14 50C14 69.8822 30.1178 86 50 86C69.8822 86 86 69.8822 86 50C86 30.1178 69.8822 14 50 14ZM50 26C63.2548 26 74 36.7452 74 50C74 63.2548 63.2548 74 50 74C40.6724 74 32.6127 68.6835 28.6758 60.916C33.1554 62.883 38.3585 64 43.8 64C60.3685 64 73.8 50.5685 73.8 34C73.8 31.1895 73.4132 28.4688 72.6904 25.8887C66.1265 25.9616 60.2743 28.3248 55.7725 32.2275C51.6504 28.3613 46.0967 26 50 26Z" fill="#7c3aed"/>
              </svg>
            </div>
            <span class="tutor-name">Omago AI</span>
          </div>
          <span class="mode-tag">${modeLabel}</span>
        </div>
        <div class="tutor-body">${parsedHtml}</div>
        <div class="tutor-actions">
          <button class="action-btn tts-btn" title="Read explanation aloud">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
            <span>Listen</span>
          </button>
          <button class="action-btn copy-btn" title="Copy response to clipboard">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            <span>Copy</span>
          </button>
        </div>
      </div>
    `;

    chatMessages.appendChild(row);

    // Apply syntax highlighting
    if (typeof hljs !== "undefined") {
      row.querySelectorAll("pre code").forEach((block) => hljs.highlightElement(block));
    }

    // Bind Listen TTS
    const ttsBtn = row.querySelector(".tts-btn");
    ttsBtn.addEventListener("click", () => speakText(markdownContent));

    // Bind Copy button
    const copyBtn = row.querySelector(".copy-btn");
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(markdownContent).then(() => {
        copyBtn.querySelector("span").textContent = "Copied!";
        setTimeout(() => (copyBtn.querySelector("span").textContent = "Copy"), 2000);
      });
    });
  }

  // --- Text-to-Speech (TTS) ---
  function speakText(rawText) {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    // Strip markdown formatting for cleaner speech
    const cleanText = rawText.replace(/[#*`_~\[\]()]/g, "").replace(/\n+/g, " ");
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }

  // --- New Chat Reset ---
  newChatBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/reset", { method: "POST" });
      const data = await res.json();
      currentThreadId = data.thread_id;
    } catch {
      currentThreadId = `omago-session-${Math.random().toString(36).substring(2, 9)}`;
    }
    chatMessages.innerHTML = "";
    chatMessages.style.display = "none";
    chatHero.style.display = "flex";
    updateThreadDisplay();
  });

  function updateThreadDisplay() {
    threadDisplay.textContent = `Thread: ${currentThreadId}`;
    if (leadsThreadId) leadsThreadId.textContent = currentThreadId;
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function escapeHtml(text) {
    return text.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[m]));
  }

  // --- Tab Navigation ---
  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetTab = link.getAttribute("data-tab");

      navLinks.forEach((l) => l.classList.remove("active"));
      link.classList.add("active");

      tabViews.forEach((tab) => {
        tab.style.display = tab.id === `tab-${targetTab}` ? "flex" : "none";
      });
    });
  });

  // --- Session Preferences UI ---
  function updatePreferencesUI(prefs) {
    if (!preferencesList) return;
    preferencesList.innerHTML = "";
    prefs.forEach((p) => {
      const li = document.createElement("li");
      li.textContent = p;
      preferencesList.appendChild(li);
    });
  }

  // --- Leads / Queries Tracker ---
  let queryCount = 0;
  function addQueryToLeads(query, mode) {
    queryCount++;
    if (queriesCounter) queriesCounter.textContent = queryCount.toString();

    if (queryCount === 1 && leadsTableBody) {
      leadsTableBody.innerHTML = "";
    }

    const tr = document.createElement("tr");
    const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    tr.innerHTML = `
      <td>${timeStr}</td>
      <td><strong>${escapeHtml(query.length > 50 ? query.substring(0, 50) + "..." : query)}</strong></td>
      <td><span class="mode-tag">${mode}</span></td>
      <td><span style="color:#22c55e; font-weight:600;">Completed</span></td>
    `;
    leadsTableBody.prepend(tr);
  }

  // --- Embed Snippet Copy ---
  if (copyEmbedBtn) {
    copyEmbedBtn.addEventListener("click", () => {
      const snippet = `<script src="http://localhost:8000/embed.js" data-tutor-theme="purple" async></script>\n<div id="omago-ai-widget"></div>`;
      navigator.clipboard.writeText(snippet).then(() => {
        copyEmbedBtn.textContent = "Copied!";
        setTimeout(() => (copyEmbedBtn.textContent = "Copy Snippet"), 2000);
      });
    });
  }

  // --- Floating Button & Shortcuts Modal ---
  floatingLauncher.addEventListener("click", () => {
    shortcutsModal.style.display = "flex";
  });

  closeModalBtn.addEventListener("click", () => {
    shortcutsModal.style.display = "none";
  });

  shortcutsModal.addEventListener("click", (e) => {
    if (e.target === shortcutsModal) {
      shortcutsModal.style.display = "none";
    }
  });

  chipButtons.forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      chatTextarea.value = prompt;
      shortcutsModal.style.display = "none";
      chatTextarea.focus();
      chatTextarea.dispatchEvent(new Event("input"));
    });
  });
});

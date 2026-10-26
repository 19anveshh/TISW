/**
 * embed.js - Omago AI Embeddable Widget Script
 * Injects a floating Omago AI tutor button and widget into any webpage.
 */
(function() {
  const container = document.getElementById("omago-ai-widget") || document.body;
  const launcher = document.createElement("div");
  launcher.innerHTML = `
    <style>
      .omago-floating-trigger {
        position: fixed;
        bottom: 24px;
        right: 24px;
        background: linear-gradient(135deg, #a855f7, #7e22ce);
        color: white;
        border-radius: 9999px;
        padding: 12px 20px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(147, 51, 234, 0.4);
        display: flex;
        align-items: center;
        gap: 8px;
        z-index: 99999;
        transition: transform 0.2s ease;
      }
      .omago-floating-trigger:hover {
        transform: scale(1.05);
      }
    </style>
    <div class="omago-floating-trigger" onclick="window.open('http://localhost:8000', '_blank')">
      <span>✨</span> Ask Omago AI
    </div>
  `;
  container.appendChild(launcher);
})();

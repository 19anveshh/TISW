# Original TISW chat input reference

This is the complete pre-experiment input markup and the original input-specific
CSS. It is intentionally not imported by the application.

## `App.jsx` input markup

```jsx
<div className="input-area">
  {showFileZone && (
    <div className="file-zone-container">
      <FileDropZone
        onUploadSuccess={(result) => {
          triggerToast(`${result.filename} uploaded & indexed`, 'success');
          setRagEnabled(true);
          fetchDocuments(sessionToken)
            .then((data) => setDocuments(data.documents || []))
            .catch(() => {});
          setShowFileZone(false);
        }}
        onUploadError={(err) => triggerToast(err, 'error')}
        disabled={isLoading}
        sessionToken={sessionToken}
      />
    </div>
  )}

  <div className="input-card">
    <textarea
      ref={textareaRef}
      rows={1}
      value={prompt}
      onChange={(e) => setPrompt(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          handleSend();
        }
      }}
      placeholder={
        !sessionToken
          ? 'Open the sidebar to configure your Groq API key...'
          : ragEnabled
          ? `RAG Active (${documents.length} doc${documents.length === 1 ? '' : 's'}) — ask anything about your documents...`
          : 'Ask anything…'
      }
      className="input-textarea"
      disabled={isLoading}
    />

    <div className="input-controls">
      <button
        type="button"
        className={`input-icon-btn ${documents.length > 0 ? 'input-icon-btn--has-docs' : ''}`}
        onClick={() => setShowFileZone((prev) => !prev)}
        title={
          documents.length > 0
            ? `${documents.length} document(s) uploaded — click to add more`
            : 'Attach PDF, DOCX, or TXT document'
        }
        aria-label="Attach document"
      >
        <Paperclip className="w-4 h-4" />
      </button>

      <button
        type="button"
        className="send-btn"
        onClick={() => handleSend()}
        disabled={isLoading || !prompt.trim()}
        title="Send message"
        aria-label="Send message"
      >
        {isLoading ? (
          <RefreshCw className="w-4 h-4 spin" />
        ) : (
          <Send className="w-4 h-4" />
        )}
      </button>
    </div>
  </div>
</div>
```

## Original CSS from `index.css`

```css
.input-area {
  position: sticky;
  bottom: 0;
  z-index: 25;
  padding: 8px 24px calc(24px + env(safe-area-inset-bottom, 0px));
  background: transparent;
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  pointer-events: none;
}

.file-zone-container {
  pointer-events: auto;
  width: 100%;
  max-width: 760px;
  margin-bottom: 10px;
}

.input-card {
  pointer-events: auto;
  width: 100%;
  max-width: 740px;
  border-radius: 20px;
  padding: 10px 14px 10px 18px;
  display: flex;
  align-items: flex-end;
  gap: 10px;
  background: rgba(14, 11, 24, 0.65);
  backdrop-filter: blur(28px) saturate(180%);
  -webkit-backdrop-filter: blur(28px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 16px 44px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.input-card:focus-within {
  border-color: rgba(168, 85, 247, 0.45);
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.6), 0 0 0 2px rgba(124, 58, 237, 0.18);
}

.input-textarea {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  resize: none;
  color: #f1f0f5;
  font-size: 14.5px;
  line-height: 1.5;
  min-height: 24px;
  max-height: 160px;
  padding: 6px 0;
  font-family: inherit;
}

.input-textarea::placeholder { color: rgba(255, 255, 255, 0.38); }

.input-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  padding-bottom: 2px;
}

.input-icon-btn {
  width: 32px;
  height: 32px;
  border-radius: 9px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.04);
  color: rgba(255, 255, 255, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.15s ease;
}

.send-btn {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.18s ease;
  box-shadow: 0 4px 14px rgba(124, 58, 237, 0.45);
}
```

## Original mobile rule

```css
@media (max-width: 768px) {
  .input-area {
    padding: 6px 16px calc(18px + env(safe-area-inset-bottom, 0px));
  }
}
```

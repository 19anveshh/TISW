# TISW — Teaching in a Structured Way

> An AI-powered tutoring interface built with **React + Vite**, **FastAPI**, **LangGraph**, and **Groq**.

---

## Features

- 🌍 Animated Earth background with glassmorphism UI
- 💬 Persistent multi-turn conversations via LangGraph memory
- ⚡ Ultra-fast inference powered by Groq
- 📎 Document upload support
- 🔒 Secure server-side API key handling — never exposed to the browser

---

## Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Node.js 18+

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/19anveshh/TISW.git
cd TISW
```

### 2. Configure your Groq API key

> ⚠️ **Never commit your real API key to Git.** The `.env` file is already listed in `.gitignore`.

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-20b
PORT=8000
```

Get a free API key at **[console.groq.com/keys](https://console.groq.com/keys)**.

### 3. Install Python dependencies

```bash
uv sync
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running Locally

Open **two terminals**:

**Terminal 1 — Backend**
```bash
uv run python server.py
```
Backend will be available at `http://localhost:8000`.

**Terminal 2 — Frontend**
```bash
cd frontend
npm run dev
```
Frontend will be available at `http://localhost:5173`.

---

## API Key Security

This project is designed to be safely open-sourced:

| What | Status |
|------|--------|
| `.env` committed to Git | Never — blocked by `.gitignore` |
| API key in frontend code | Never — key stays on the server |
| API key in browser requests | Never — handled server-side only |
| API key in API responses | Never — not returned or logged |
| `.env.example` with placeholder | Safe to commit |

The sidebar **"Configure key"** button opens a guide explaining how to set up your own key locally. It does **not** allow runtime key changes via the browser — that would be insecure for a public deployment.

---

## Project Structure

```
TISW/
├── main.py          # LangGraph tutor workflow
├── server.py        # FastAPI backend (POST /api/chat, GET /api/health)
├── .env             # Your local secrets — NOT committed
├── .env.example     # Safe template — committed
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── EarthBackground.jsx
│   │   ├── index.css        # All styles (no Tailwind)
│   │   └── lib/api.js       # Backend API client
│   └── vite.config.js
└── pyproject.toml
```

---

## License

MIT

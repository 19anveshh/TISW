"""Security-focused API checks that do not require a live Groq request."""

import io
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import server


client = TestClient(server.app)


class FakeProbe:
    def invoke(self, _messages):
        return AIMessage(content="OK")


def run_security_tests():
    secret = "gsk_test_secret_value_that_must_not_escape"
    output = io.StringIO()
    with patch.object(server, "is_offline_mode", return_value=False), patch.object(server, "ChatGroq", return_value=FakeProbe()):
        with redirect_stdout(output), redirect_stderr(output):
            response = client.post("/api/verify-key", json={"api_key": secret})

    assert response.status_code == 200, response.text
    body = response.text
    logs = output.getvalue()
    assert secret not in body
    assert secret not in logs
    assert "api_key" not in body
    assert "api_key" not in logs.lower()

    health = client.get("/api/health")
    assert health.status_code == 200
    assert secret not in health.text

    print("✓ API keys are absent from verification responses, logs, and health output")

    run_cors_tests()


def run_cors_tests():
    allowed = [
        "https://tisw-frontend.vercel.app",
        "https://tisw-frontend-git-main-19anveshh.vercel.app",
        "http://localhost:5199",
        "http://127.0.0.1:4173",
    ]
    for origin in allowed:
        response = client.options(
            "/api/verify-key",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-client-id",
            },
        )
        assert response.headers.get("access-control-allow-origin") == origin, origin

    rejected = client.options(
        "/api/verify-key",
        headers={
            "Origin": "https://tisw-frontend.vercel.app.evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in rejected.headers

    print("✓ Deployment previews and any local dev port may call the API; look-alike hosts may not")


if __name__ == "__main__":
    run_security_tests()

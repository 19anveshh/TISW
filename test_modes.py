"""Offline checks for hidden, explicit, thread-scoped TISW behavior."""

import sys
from unittest.mock import patch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from fastapi.testclient import TestClient

from main import NORMAL_MODE_PROMPT, TISW_MODE_PROMPT, create_tutor_graph, detect_mode_command, detect_mode_request
import server


class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="mocked response")


def run_mode_tests():
    for text in (
        "What is Python?",
        "Explain Python simply.",
        "Teach me Python step by step.",
        "What does TISW stand for?",
        "Explain the difference between TISW and a normal chatbot.",
        "I use TISW at school.",
    ):
        assert detect_mode_command(text) is None, text

    for text in (
        "Only use TISW.",
        "Use TISW to explain Python.",
        "Please use TISW for this topic!",
        "Let's use TISW in this conversation.",
        "Continue using tisw?",
        "Always use TISW from now on.",
        "Explain this using TISW.",
        "Could you explain Python using TISW?",
        "Could you apply the TISW methodology?",
    ):
        assert detect_mode_command(text) == "activate", text

    for text in (
        "Switch back to normal chat!",
        "Stop using TISW and answer normally.",
        "Disable TISW for this conversation.",
        "Answer normally from now on.",
    ):
        assert detect_mode_command(text) == "deactivate", text

    assert detect_mode_request("What is Python?", "normal") == "normal"
    assert detect_mode_request("What is TISW?", "normal") == "normal"
    assert detect_mode_request("Use TISW to explain Python.", "normal") == "tisw"
    assert detect_mode_request("Switch back to normal chat!", "tisw") == "normal"

    llm = FakeLLM()
    graph = create_tutor_graph(llm, InMemorySaver())
    config = {"configurable": {"thread_id": "mode-test"}}

    normal = graph.invoke({"messages": [{"role": "user", "content": "What is Python?"}]}, config=config)
    assert normal["active_mode"] == "normal"
    assert len(llm.calls) == 1
    assert llm.calls[-1][0].content == NORMAL_MODE_PROMPT
    assert TISW_MODE_PROMPT not in llm.calls[-1][0].content

    tisw = graph.invoke(
        {"messages": [{"role": "user", "content": "Always use TISW from now on."}]},
        config=config,
    )
    assert tisw["active_mode"] == "tisw"
    assert tisw["messages"][-1].content == "Understood."
    assert len(llm.calls) == 1  # activation is not sent to the LLM

    follow_up = graph.invoke(
        {
            "messages": [{"role": "user", "content": "What about an example?"}],
            "rag_context": "[Source: python-notes.txt]\nA list stores an ordered collection.",
        },
        config=config,
    )
    assert follow_up["active_mode"] == "tisw"
    assert len(llm.calls) == 2
    assert llm.calls[-1][0].content.startswith(TISW_MODE_PROMPT)
    assert "RETRIEVED DOCUMENT EXCERPTS" in llm.calls[-1][0].content
    assert "python-notes.txt" in llm.calls[-1][0].content

    normal_again = graph.invoke(
        {"messages": [{"role": "user", "content": "Answer normally from now on."}]},
        config=config,
    )
    assert normal_again["active_mode"] == "normal"
    assert normal_again["messages"][-1].content == "Understood."
    assert len(llm.calls) == 2

    normal_follow_up = graph.invoke(
        {"messages": [{"role": "user", "content": "What is a list?"}], "rag_context": None},
        config=config,
    )
    assert normal_follow_up["active_mode"] == "normal"
    assert len(llm.calls) == 3
    assert llm.calls[-1][0].content == NORMAL_MODE_PROMPT

    new_thread = graph.invoke(
        {"messages": [{"role": "user", "content": "What is Python?"}]},
        config={"configurable": {"thread_id": "new-thread"}},
    )
    assert new_thread["active_mode"] == "normal"

    other_llm = FakeLLM()
    other_graph = create_tutor_graph(other_llm, InMemorySaver())
    other_state = other_graph.invoke(
        {"messages": [{"role": "user", "content": "What is Python?"}]},
        config={"configurable": {"thread_id": "mode-test"}},
    )
    assert other_state["active_mode"] == "normal"

    rag_config = {"configurable": {"thread_id": "rag-clear-test"}}
    graph.invoke(
        {"messages": [{"role": "user", "content": "What is in my notes?"}], "rag_context": "[Source: notes.txt]"},
        config=rag_config,
    )
    graph.invoke(
        {"messages": [{"role": "user", "content": "What is recursion?"}], "rag_context": None},
        config=rag_config,
    )
    assert "RETRIEVED DOCUMENT EXCERPTS" not in llm.calls[-1][0].content
    print("✓ Explicit activation, thread persistence/isolation, deactivation, normal chat, and RAG context clearing passed")


def run_api_mode_tests():
    """Verify the same conversation behavior through the FastAPI request path."""
    client = TestClient(server.app)
    llm = FakeLLM()
    token_a = server._issue_session_token("gsk_mode_test_a", "test-model")
    token_b = server._issue_session_token("gsk_mode_test_b", "test-model")
    headers_a = {"X-Session-Token": token_a}
    headers_b = {"X-Session-Token": token_b}

    try:
        with patch.object(server, "is_offline_mode", return_value=False), patch.object(server, "_build_llm_for_session", return_value=llm):
            normal = client.post(
                "/api/chat",
                json={"message": "What is Python?", "thread_id": "api-thread"},
                headers=headers_a,
            )
            assert normal.status_code == 200, normal.text
            assert normal.json()["mode"] == "normal"

            activate = client.post(
                "/api/chat",
                json={"message": "Use TISW to explain Python.", "thread_id": "api-thread"},
                headers=headers_a,
            )
            assert activate.status_code == 200, activate.text
            assert activate.json()["mode"] == "tisw"

            follow_up = client.post(
                "/api/chat",
                json={"message": "Show me a small example.", "thread_id": "api-thread"},
                headers=headers_a,
            )
            assert follow_up.status_code == 200, follow_up.text
            assert follow_up.json()["mode"] == "tisw"

            isolated = client.post(
                "/api/chat",
                json={"message": "What is Python?", "thread_id": "api-thread"},
                headers=headers_b,
            )
            assert isolated.status_code == 200, isolated.text
            assert isolated.json()["mode"] == "normal"
    finally:
        server._session_store.pop(token_a, None)
        server._session_store.pop(token_b, None)

    print("✓ FastAPI requests preserve TISW mode per authenticated conversation session")


if __name__ == "__main__":
    run_mode_tests()
    run_api_mode_tests()

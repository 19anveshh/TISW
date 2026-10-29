"""Offline checks for the four-part TISW walkthrough and slash commands."""

import os
import sys
import tempfile
from unittest.mock import patch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Keep this run's history out of the local development database.
os.environ["TISW_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="tisw-paging-"), "paging.db")

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from fastapi.testclient import TestClient

from main import (
    TISW_BEATS,
    TISW_MODE_PROMPT,
    TISW_PARTS,
    create_tutor_graph,
    detect_mode_command,
    detect_mode_request,
    is_paging_control,
    paging_payload,
    parse_slash_command,
)
import server


class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="mocked part")


def run_command_tests():
    assert parse_slash_command("/tisw what is OOP") == ("tisw", "what is OOP")
    assert parse_slash_command("/TISW") == ("tisw", "")
    assert parse_slash_command("/normal") == ("normal", "")
    assert parse_slash_command("/continue") == ("continue", "")
    assert parse_slash_command("/all") == ("all", "")
    assert parse_slash_command("/unknown thing") is None
    assert parse_slash_command("What is /tisw?") is None

    assert detect_mode_command("/tisw") == "activate"
    assert detect_mode_command("/normal") == "deactivate"
    assert detect_mode_command("/tisw explain OOP") is None
    assert detect_mode_request("/tisw explain OOP", "normal") == "tisw"
    assert detect_mode_request("/normal", "tisw") == "normal"
    assert detect_mode_request("What is Python?", "tisw") == "tisw"

    assert is_paging_control("/continue", "tisw", 2) is True
    assert is_paging_control("/all", "tisw", 2) is True
    assert is_paging_control("continue", "tisw", 2) is True
    assert is_paging_control("/tisw new topic", "tisw", 2) is False
    assert is_paging_control("what about inheritance?", "tisw", 2) is False
    print("OK: slash commands and paging control detection")


def run_graph_tests():
    llm = FakeLLM()
    graph = create_tutor_graph(llm, InMemorySaver())
    config = {"configurable": {"thread_id": "paging-thread"}}

    def ask(text):
        return graph.invoke({"messages": [{"role": "user", "content": text}]}, config=config)

    first = ask("/tisw what is OOP in Python?")
    assert first["active_mode"] == "tisw"
    assert first["tisw_part"] == 1
    assert paging_payload(first)["part"] == 1
    assert paging_payload(first)["complete"] is False
    prompt = llm.calls[-1][0].content
    assert prompt.startswith(TISW_MODE_PROMPT)
    assert f"DELIVER PART 1 OF {TISW_PARTS} ONLY" in prompt
    assert TISW_BEATS[0] in prompt
    assert "what is OOP in Python?" in prompt

    second = ask("/continue")
    assert second["tisw_part"] == 2
    assert paging_payload(second)["part"] == 2
    prompt = llm.calls[-1][0].content
    assert f"DELIVER PART 2 OF {TISW_PARTS} ONLY" in prompt
    assert TISW_BEATS[1] in prompt

    doubt = ask("why does inheritance matter here?")
    assert doubt["tisw_part"] == 2
    prompt = llm.calls[-1][0].content
    assert "ANSWER THIS DOUBT" in prompt
    assert "DELIVER PART" not in prompt

    third = ask("continue")
    assert third["tisw_part"] == 3
    fourth = ask("/next")
    assert fourth["tisw_part"] == TISW_PARTS
    assert paging_payload(fourth)["complete"] is True

    calls_before = len(llm.calls)
    done = ask("/continue")
    assert len(llm.calls) == calls_before
    assert "already complete" in str(done["messages"][-1].content)
    assert paging_payload(done) is None

    after = ask("and what about constructors?")
    assert after["tisw_part"] == TISW_PARTS
    assert paging_payload(after) is None
    assert "ANSWER FREELY" in llm.calls[-1][0].content

    fresh = ask("/tisw what is a database index?")
    assert fresh["tisw_part"] == 1
    assert fresh["tisw_topic"] == "what is a database index?"

    whole = ask("/all")
    assert whole["tisw_part"] == TISW_PARTS
    assert paging_payload(whole)["complete"] is True
    assert "COMPLETE ANSWER REQUESTED" in llm.calls[-1][0].content

    off = ask("/normal")
    assert off["active_mode"] == "normal"
    assert paging_payload(off) is None
    assert len(llm.calls) == calls_before + 3

    no_plan = ask("/all")
    assert "no paged walkthrough" in str(no_plan["messages"][-1].content)
    print("OK: part delivery, doubts, completion, reset, and escape hatch")


class PagingStreamGraph:
    """Streams one answer and returns walkthrough state for the done frame."""

    def __init__(self):
        self.calls = 0

    def get_state(self, _config):
        return type("State", (), {"values": {"active_mode": "normal", "messages": []}})()

    async def astream_events(self, payload, config=None, version="v2"):
        self.calls += 1
        text = "Part one of the walkthrough."
        yield {"event": "on_chain_start", "name": "LangGraph", "data": {}}
        yield {
            "event": "on_chat_model_stream",
            "name": "ChatGroq",
            "metadata": {"langgraph_node": "tutor"},
            "data": {"chunk": AIMessageChunk(content=text)},
        }
        yield {
            "event": "on_chain_end",
            "name": config["run_name"],
            "data": {"output": {
                "messages": [AIMessage(content=text)],
                "active_mode": "tisw",
                "tisw_part": 1,
                "tisw_topic": "what is OOP?",
                "tisw_paged": True,
            }},
        }


def run_stream_payload_test():
    import asyncio
    import json

    def frames(text):
        parsed = []
        for frame in [frame for frame in text.split("\n\n") if frame.strip()]:
            event = next(line[6:].strip() for line in frame.splitlines() if line.startswith("event:"))
            data = next(line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:"))
            parsed.append((event, json.loads(data)))
        return parsed

    async def collect():
        return frames("".join([
            frame async for frame in server._mock_stream_events(
                server.ChatRequest(message="/tisw what is OOP?", thread_id="mock-paging")
            )
        ]))

    events = asyncio.run(collect())
    assert events[-1][0] == "done"
    done = events[-1][1]
    assert done["mode"] == "tisw"
    assert done["paging"]["part"] == 1
    assert done["paging"]["parts_total"] == TISW_PARTS
    assert done["paging"]["complete"] is False

    graph = PagingStreamGraph()

    async def collect_live():
        return frames("".join([
            frame async for frame in server._stream_chat_events(
                server.ChatRequest(message="/tisw what is OOP?", thread_id="live-paging", turn_id="live-1"),
                None,
            )
        ]))

    with patch.object(server, "is_offline_mode", return_value=False), \
         patch.object(server, "tutor_stream_app", graph), \
         patch.object(server, "_request_tracing_context", return_value=__import__("contextlib").nullcontext()):
        events = asyncio.run(collect_live())
    assert events[-1][0] == "done"
    assert events[-1][1]["paging"]["title"] == TISW_BEATS[0]
    print("OK: streaming mock and live done frames carry paging metadata")


def run_api_tests():
    client = TestClient(server.app)
    llm = FakeLLM()
    token = server._issue_session_token("gsk_paging_test", "test-model")
    headers = {"X-Session-Token": token, "X-Client-Id": "pagingclient01"}
    try:
        with patch.object(server, "is_offline_mode", return_value=False), \
             patch.object(server, "_build_llm_for_session", return_value=llm):
            started = client.post(
                "/api/chat",
                json={"message": "/tisw explain binary search", "thread_id": "api-paging", "turn_id": "p1"},
                headers=headers,
            )
            assert started.status_code == 200, started.text
            body = started.json()
            assert body["mode"] == "tisw"
            assert body["paging"]["part"] == 1
            assert body["paging"]["topic"] == "explain binary search"

            nxt = client.post(
                "/api/chat",
                json={"message": "/continue", "thread_id": "api-paging", "turn_id": "p2"},
                headers=headers,
            )
            assert nxt.status_code == 200, nxt.text
            assert nxt.json()["paging"]["part"] == 2

            stored = client.get(
                "/api/threads/api-paging/messages",
                headers=headers,
            )
            assert stored.status_code == 200, stored.text
            rows = stored.json()["messages"]
            assert rows[0]["paging"] is None
            assert rows[1]["paging"]["part"] == 1
            assert rows[3]["paging"]["part"] == 2

            off = client.post(
                "/api/chat",
                json={"message": "/normal", "thread_id": "api-paging", "turn_id": "p3"},
                headers=headers,
            )
            assert off.status_code == 200, off.text
            assert off.json()["mode"] == "normal"
            assert off.json()["paging"] is None
    finally:
        server._session_store.pop(token, None)
    print("OK: API responses and stored transcripts expose walkthrough parts")


if __name__ == "__main__":
    run_command_tests()
    run_graph_tests()
    run_stream_payload_test()
    run_api_tests()

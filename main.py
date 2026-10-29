"""
main.py - TISW AI Tutor powered by LangGraph, LangChain, and Groq.

Supports two distinct response modes:
1. NORMAL MODE (Default): Direct, natural, and clear explanations in simple,
   everyday English friendly to Indian college students. No forced rigid templates.
2. TISW MODE: Activated only when explicitly requested (e.g., 'explain using TISW').
   Provides deep first-principles structured teaching with flexible guidance.
"""

import os
import re
import sys
import uuid
from typing import Annotated, Any, List, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from runtime_config import configure_runtime, is_offline_mode


# Apply tracing policy before any graph is constructed. The evaluator sets
# TISW_OFFLINE before importing this module for local runs.
load_dotenv()
RUNTIME_CONFIG = configure_runtime()


# =====================================================================
# 1. SYSTEM PROMPTS: NORMAL MODE & TISW MODE
# =====================================================================

NORMAL_MODE_PROMPT = """# IDENTITY & PURPOSE
You are TISW AI, a knowledgeable, clear, and friendly AI study assistant.
You are currently operating in NORMAL MODE (the default mode for every user).

# CORE PRINCIPLES FOR NORMAL MODE:
1. Direct, Natural Answers:
   - Answer the student's question directly and immediately without unnecessary preamble, greetings, or filler.
   - For simple or factual questions (e.g., "What is the capital of India?"), provide a direct, concise answer.
   - For programming questions (e.g., "Write Python code to reverse a string"), provide clean, working code with a short, easy-to-follow explanation.
   - For conceptual questions (e.g., "What is recursion?" or "Explain gradient descent"), explain the concept clearly and conversationally.

2. Simple, Everyday English (Tailored for Indian Students):
   - Use clear, everyday English that is easy to follow for college students.
   - Avoid unnecessarily formal, academic, or high-sounding vocabulary (for example, do NOT say "Recursion constitutes a computational paradigm wherein..."; say "Recursion is when a function calls itself to solve a smaller version of the same problem. It needs a stopping condition so it does not continue forever.").
   - Prefer short sentences and familiar words.
   - Explain technical terms immediately when they first appear.
   - Sound like a friendly, helpful senior explaining things to a college student.

3. Everyday Relatable Examples:
   - When an example makes a concept easier to understand, use familiar real-world Indian examples (e.g., college life, cricket, UPI payments, train tickets/IRCTC, chai stall, mobile recharge, online shopping).
   - Do NOT force Indian examples into every answer—use them only when they naturally aid understanding.

4. NO FORCED TEACHING TEMPLATES:
   - Never force ordinary answers into rigid teaching templates or mandatory sections like "Intuition", "First Principles", "Step-by-Step Derivation", "Under the Hood", "Summary", or "Practice Questions".
   - Use Markdown headings and bullet points ONLY when they genuinely improve readability.
   - Keep answers proportionate: a simple question gets a short, crisp answer; a complex question gets more detail as needed.
   - Never lecture or add unsolicited study advice.
"""

TISW_MODE_PROMPT = """# IDENTITY & PURPOSE
You are TISW AI. The student explicitly requested the internal TISW (Teaching in a Structured Way) teaching methodology for this conversation. Apply it naturally without announcing an internal mode or forcing a visible template.

# TEACHING PRINCIPLE
Help the learner understand the idea before introducing unfamiliar terminology. Then connect that intuition to the accurate real concept and help the learner explore it deeply.

# HOW TO TEACH
- Start with a familiar idea, observation, or small example when it genuinely clarifies the topic. Use an analogy only when it is accurate and useful; never force a misleading analogy.
- Introduce the correct technical or academic vocabulary naturally after the learner has enough context. Define unfamiliar terms in plain English instead of stacking unexplained definitions.
- Explain not only what something is, but why it exists, what problem it solves, how it works, why it works that way, when it is useful, its limitations, and meaningful alternatives when relevant.
- Move naturally through the layers that help this particular question: basic intuition, the real concept, mechanism, reasoning, examples or applications, and deeper details. Do not force every layer into a simple answer or make every response long.
- Use simple, natural English by default. Be suitable for a beginner, including a 12-year-old, without sacrificing accuracy or becoming condescending. If the learner shows prior knowledge, skip repeated basics and go deeper.
- When appropriate, ask one short prediction, checking question, or practical exercise and use the response to correct misunderstandings. Do not turn every answer into a quiz.

# CODING AND TECHNICAL TOPICS
- Establish the problem and intended outcome before presenting a large code block.
- Build solutions in small, meaningful, runnable steps. For important lines or blocks, explain what they do, why they belong there, how they connect to the previous step, and what would change if they were removed or modified.
- Explain important errors and debugging decisions. Mention alternative approaches and trade-offs when they matter, while distinguishing requirements from optional preferences.
- Avoid mechanically explaining every character when the learner does not need that level of detail.

# SUBJECT ADAPTATION
- Science: connect observations to the underlying principle.
- Mathematics: build intuition with examples before formulas when appropriate.
- History: connect causes, context, events, and consequences.
- Economics: use familiar decisions and real-life situations.
- AI and technology: explain the basic idea before technical mechanisms.
- Reading and theory: unpack difficult passages and connect details to the author's larger argument.

Remain technically honest, acknowledge uncertainty, and adapt depth to the learner's question. The user can explicitly switch back to normal chat; after that, use the normal response style.
"""

# Alias for backwards compatibility
TEACHING_METHODOLOGY_PROMPT = TISW_MODE_PROMPT


# =====================================================================
# 2. PAGED DELIVERY, SLASH COMMANDS & MODE DETECTION
# =====================================================================

# A paged TISW answer is delivered one quarter at a time. The four beats are
# fixed so the walkthrough never needs an extra planning model call.
TISW_BEATS = (
    "Intuition and the problem",
    "The real concept and how it works",
    "A worked example",
    "Depth, limits, and practice",
)
TISW_PARTS = len(TISW_BEATS)

SLASH_COMMANDS = frozenset({"tisw", "normal", "continue", "next", "all"})

_CONTINUE_TEXTS = frozenset({"continue", "next", "next part", "go on", "keep going"})
_ALL_TEXTS = frozenset({
    "give me all",
    "give me everything",
    "all at once",
    "show everything",
    "all parts",
    "the whole thing",
})


def parse_slash_command(user_text: str) -> Optional[tuple[str, str]]:
    """Split a leading supported slash command into (command, remainder).

    Unknown slash text and ordinary questions return None so a missed command
    is answered as a normal message instead of being swallowed.
    """
    text = (user_text or "").strip()
    if not text.startswith("/"):
        return None
    head, _, rest = text[1:].partition(" ")
    command = head.lower()
    if command not in SLASH_COMMANDS:
        return None
    return command, rest.strip()


def detect_mode_command(user_text: str) -> Optional[str]:
    """Recognize explicit TISW activation/deactivation requests only.

    The matcher intentionally requires an imperative/request shape around the
    acronym. This keeps incidental questions such as "What does TISW stand for?"
    and comparisons with normal chat in the default mode.
    """
    slash = parse_slash_command(user_text)
    if slash:
        command, remainder = slash
        if command == "normal":
            return "deactivate"
        if command == "tisw" and not remainder:
            return "activate"
        return None

    normalized = re.sub(r"[^a-z0-9\s]", " ", (user_text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    deactivate_patterns = (
        r"^(?:please\s+)?stop\s+using\s+tisw\b",
        r"^(?:please\s+)?disable\s+tisw\b",
        r"^(?:please\s+)?(?:turn\s+off|switch\s+off)\s+tisw\b",
        r"^(?:please\s+)?switch\s+back\s+to\s+normal(?:\s+(?:chat|mode))?\b",
        r"^(?:please\s+)?switch\s+to\s+normal(?:\s+(?:chat|mode))?\b",
        r"^(?:please\s+)?(?:answer|respond|reply)\s+normally\b",
        r"^(?:please\s+)?use\s+normal(?:\s+(?:chat|mode))?\b",
    )
    if any(re.match(pattern, normalized) for pattern in deactivate_patterns):
        return "deactivate"

    activate_patterns = (
        r"^(?:please\s+)?(?:only\s+)?use\s+tisw\b",
        r"^(?:can|could|would)\s+you\s+use\s+tisw\b",
        r"^let(?:s|\s+s)\s+use\s+tisw\b",
        r"^(?:continue|keep)\s+using\s+tisw\b",
        r"^(?:from\s+now\s+on\s+)?always\s+use\s+tisw\b",
        r"^from\s+now\s+on\s+use\s+tisw\b",
        r"^(?:please\s+)?(?:switch\s+to|activate|enable|turn\s+on)\s+tisw\b",
        r"^(?:please\s+)?(?:apply|follow)\s+(?:the\s+)?tisw\b",
        r"^(?:can|could|would)\s+you\s+(?:apply|follow)\s+(?:the\s+)?tisw\b",
    )
    if any(re.match(pattern, normalized) for pattern in activate_patterns):
        return "activate"

    if re.match(
        r"^(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?:explain|teach|answer|respond|reply)\b.*\b(?:using|with)\s+tisw\b",
        normalized,
    ):
        return "activate"

    return None


def detect_mode_request(user_text: str, current_mode: str = "normal") -> str:
    """Apply an explicit mode command, otherwise preserve thread state."""
    slash = parse_slash_command(user_text)
    if slash:
        command = slash[0]
        if command == "tisw":
            return "tisw"
        if command == "normal":
            return "normal"
    command = detect_mode_command(user_text)
    if command == "activate":
        return "tisw"
    if command == "deactivate":
        return "normal"
    return current_mode or "normal"


def detect_persistent_preference(user_text: str) -> Optional[str]:
    """
    Checks if the student is explicitly asking to set a persistent preference
    for future responses in this session (e.g., 'from now on...', 'always use...').
    """
    lowered = user_text.lower()
    persistence_markers = (
        "from now on",
        "in future responses",
        "for all future",
        "always explain",
        "always reply",
        "always use",
        "remember that i prefer",
        "keep all future answers",
    )
    if any(marker in lowered for marker in persistence_markers):
        return user_text.strip()
    return None


def is_paging_control(user_text: str, current_mode: str = "normal", part: int = 0) -> bool:
    """True when a message only steers an existing walkthrough (no new question)."""
    if detect_mode_command(user_text):
        return True
    slash = parse_slash_command(user_text)
    if slash:
        command, remainder = slash
        if command in {"continue", "next", "all"}:
            return True
        if command in {"tisw", "normal"}:
            return not remainder
    if current_mode == "tisw" and part > 0:
        normalized = _normalized_for_command(user_text)
        return normalized in _CONTINUE_TEXTS or normalized in _ALL_TEXTS
    return False


RAG_GROUNDING_PROMPT = """
# RETRIEVED DOCUMENT CONTEXT & GROUNDING GUIDELINES
The following context excerpts were retrieved from the student's uploaded study documents.

RULES FOR USING RETRIEVED EXCERPTS:
1. Evidence-Based Grounding: Base your answer primarily on the retrieved document excerpts below whenever relevant.
2. Accurate Citations: Cite your sources clearly using the exact filename and page number when available (e.g., "[Source: notes.pdf, Page 3]" or "[Source: guide.docx]"). Do NOT invent or fabricate page numbers for TXT or DOCX files.
3. Honest Limits: If the uploaded documents do not contain sufficient information to answer the question, clearly state: "Based on your uploaded documents, this information is not covered..." You may then provide a general answer if appropriate, explicitly noting that it comes from general knowledge rather than the uploaded document.
4. Never Fabricate Citations: Do not invent facts, quotes, or citations that are not in the retrieved excerpts.
5. Respect Active Mode: Follow the active response mode (Normal Mode: direct, natural English; TISW Mode: structured, intuitive first-principles teaching).
"""


def build_system_message(
    active_mode: str = "normal",
    session_preferences: Optional[List[str]] = None,
    rag_context: Optional[str] = None,
) -> SystemMessage:
    """
    Constructs the authoritative SystemMessage based on the active mode (normal vs tisw),
    any active session preferences, and optional retrieved document context.
    """
    if active_mode == "tisw":
        content = TISW_MODE_PROMPT
    else:
        content = NORMAL_MODE_PROMPT

    if session_preferences:
        prefs_formatted = "\n".join(f"- {pref}" for pref in session_preferences)
        content += (
            "\n\n# ACTIVE SESSION PREFERENCES\n"
            "The student explicitly requested the following preferences for future responses in this session:\n"
            f"{prefs_formatted}\n"
        )

    if rag_context and rag_context.strip():
        content += f"\n\n{RAG_GROUNDING_PROMPT}\n\n# RETRIEVED DOCUMENT EXCERPTS:\n{rag_context.strip()}\n"

    return SystemMessage(content=content)


# =====================================================================
# 3. LANGGRAPH CONVERSATION STATE & WORKFLOW
# =====================================================================

class TutorState(TypedDict, total=False):
    """State schema for the Personalized AI Study Tutor graph."""
    messages: Annotated[List[AnyMessage], add_messages]
    session_preferences: List[str]
    active_mode: str  # 'normal' | 'tisw'
    rag_context: Optional[str]
    tisw_part: int    # parts of the current walkthrough already delivered
    tisw_topic: str   # question the current walkthrough is teaching
    tisw_paged: bool  # whether the latest answer belongs to the walkthrough


def paging_payload(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Describe the active walkthrough for API clients, or None if there is none."""
    if (state.get("active_mode") or "normal") != "tisw":
        return None
    part = int(state.get("tisw_part") or 0)
    topic = state.get("tisw_topic") or ""
    if part <= 0 or not topic or not state.get("tisw_paged", True):
        return None
    delivered = min(part, TISW_PARTS)
    return {
        "part": delivered,
        "parts_total": TISW_PARTS,
        "title": TISW_BEATS[delivered - 1],
        "topic": topic,
        "complete": delivered >= TISW_PARTS,
    }


def _normalized_for_command(user_text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (user_text or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _paging_block(action: str, part: int, topic: str) -> str:
    """System-prompt block that turns one TISW answer into one walkthrough part."""
    if action == "all":
        return (
            "\n\n# PAGED WALKTHROUGH — COMPLETE ANSWER REQUESTED\n"
            f"Topic: {topic}\n"
            "The student asked for the whole walkthrough at once. Give the complete TISW answer now, "
            "covering these four beats in order: intuition and the problem, the real concept and how it "
            "works, a worked example, and depth with limits and practice. Do not number parts and do not "
            "offer to continue later."
        )

    if action == "doubt":
        return (
            "\n\n# PAGED WALKTHROUGH IN PROGRESS — ANSWER THIS DOUBT\n"
            f"Topic: {topic}\n"
            f"Parts 1-{part} of {TISW_PARTS} were already delivered in this conversation. The student is "
            "now asking a follow-up doubt, not asking to continue. Answer this doubt completely and "
            "directly in TISW style. Do not restart or advance the walkthrough, do not repeat earlier "
            "parts, and do not hand out part numbers."
        )

    if action == "answer":
        return (
            "\n\n# WALKTHROUGH COMPLETE — ANSWER FREELY\n"
            f"The four-part walkthrough of {topic} is complete. Answer the student's latest message "
            "completely in TISW style without part numbers and without offering another part. If they "
            "name a new topic, teach it fully; paged delivery restarts only when they ask for it."
        )

    title = TISW_BEATS[part - 1]
    if part < TISW_PARTS:
        closing = (
            f"End with one short sentence offering part {part + 1} when the student is ready. "
            "Do not ask more than one question."
        )
    else:
        closing = (
            "This is the final part. End with one short sentence saying the walkthrough is complete "
            "and inviting a new topic."
        )
    return (
        f"\n\n# PAGED WALKTHROUGH — DELIVER PART {part} OF {TISW_PARTS} ONLY\n"
        f"Topic: {topic}\n"
        f"This part: {title}.\n"
        "The full teaching for this topic is split into four roughly equal parts. Deliver only this "
        "quarter now: do not preview later parts beyond a single short signpost, do not repeat material "
        "already delivered, and keep it to roughly 250-350 words (code, tables, and short lists are "
        "excepted). Build on what earlier parts taught when there are any. "
        f"{closing}"
    )


def _prepare_tutor_turn(state: TutorState) -> dict[str, Any]:
    """Prepare one turn for both synchronous and native streaming graphs."""
    messages: List[BaseMessage] = state.get("messages", [])
    current_prefs: List[str] = list(state.get("session_preferences") or [])
    current_mode: str = state.get("active_mode") or "normal"
    rag_context: Optional[str] = state.get("rag_context")
    tisw_part: int = int(state.get("tisw_part") or 0)
    tisw_topic: str = state.get("tisw_topic") or ""
    mode_command: Optional[str] = None
    fixed_response: Optional[str] = None
    paging_action: Optional[str] = None

    latest_text = ""
    if messages and isinstance(messages[-1], HumanMessage):
        latest_text = str(messages[-1].content).strip()
        mode_command = detect_mode_command(latest_text)
        if not mode_command:
            new_pref = detect_persistent_preference(latest_text)
            if new_pref and new_pref not in current_prefs:
                current_prefs.append(new_pref)
        current_mode = detect_mode_request(latest_text, current_mode)

    slash = parse_slash_command(latest_text)
    command, remainder = slash if slash else ("", "")
    normalized = _normalized_for_command(latest_text)

    if mode_command:
        if mode_command == "deactivate":
            tisw_part, tisw_topic, paging_action = 0, "", None
        fixed_response = "Understood."

    if not fixed_response and command == "tisw" and remainder:
        tisw_topic, tisw_part = remainder[:240], 1
    elif not fixed_response and command in {"continue", "next"}:
        if current_mode != "tisw" or tisw_part <= 0:
            fixed_response = (
                "There is no paged walkthrough in progress. Ask a question, or start one "
                "with /tisw <topic>."
            )
        elif tisw_part >= TISW_PARTS:
            fixed_response = (
                f'The walkthrough of "{tisw_topic}" is already complete. Ask a new question, '
                "or start another paged walkthrough with /tisw <topic>."
            )
        else:
            tisw_part += 1
    elif not fixed_response and command == "all":
        if current_mode != "tisw" or tisw_part <= 0:
            fixed_response = (
                "There is no paged walkthrough in progress. Ask a question, or start one "
                "with /tisw <topic>."
            )
        else:
            tisw_part = TISW_PARTS
    elif not fixed_response and current_mode == "tisw" and tisw_part > 0:
        if normalized in _CONTINUE_TEXTS:
            if tisw_part < TISW_PARTS:
                tisw_part += 1
        elif normalized in _ALL_TEXTS:
            tisw_part = TISW_PARTS

    # Classify what this answer is relative to the paged walkthrough.
    if not mode_command and not fixed_response and current_mode == "tisw":
        started = command == "tisw" and bool(remainder)
        if started:
            paging_action = "part"
        elif command in {"continue", "next"} or (tisw_part > 1 and normalized in _CONTINUE_TEXTS):
            paging_action = "part"
        elif command == "all" or normalized in _ALL_TEXTS:
            paging_action = "all"
        elif tisw_part <= 0:
            tisw_topic, tisw_part = latest_text[:240], 1
            paging_action = "part"
        elif tisw_part < TISW_PARTS:
            paging_action = "doubt"
        else:
            paging_action = "answer"

    tisw_paged = paging_action in {"part", "all", "doubt"}
    paging_block = ""
    if paging_action:
        paging_block = _paging_block(paging_action, tisw_part, tisw_topic)

    system_msg = build_system_message(current_mode, current_prefs, rag_context)
    if paging_block:
        system_msg = SystemMessage(content=system_msg.content + paging_block)

    return {
        "session_preferences": current_prefs,
        "active_mode": current_mode,
        "rag_context": rag_context,
        "tisw_part": tisw_part,
        "tisw_topic": tisw_topic,
        "tisw_paged": tisw_paged,
        "mode_command": mode_command,
        "fixed_response": fixed_response,
        "paging_action": paging_action,
        "model_input": None if fixed_response else [system_msg] + messages,
    }


def _tutor_state_update(prepared: dict[str, Any], response: Optional[BaseMessage] = None) -> dict[str, Any]:
    update: dict[str, Any] = {
        "session_preferences": prepared["session_preferences"],
        "active_mode": prepared["active_mode"],
        "rag_context": prepared["rag_context"],
        "tisw_part": prepared["tisw_part"],
        "tisw_topic": prepared["tisw_topic"],
        "tisw_paged": prepared["tisw_paged"],
    }
    if response is not None:
        update["messages"] = [response]
    return update


def _message_chunk_text(chunk: Any) -> str:
    """Return only user-visible text from a streamed model chunk."""
    content = getattr(chunk, "content", chunk if isinstance(chunk, str) else "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            text = block.get("text") or block.get("content")
            if isinstance(text, str):
                text_parts.append(text)
    return "".join(text_parts)


def create_tutor_graph(llm: ChatGroq, checkpointer: InMemorySaver):
    """
    Builds and compiles the LangGraph workflow with a single `tutor` node
    and an in-memory checkpointer to preserve multi-turn conversation and mode state.
    """

    def tutor_node(state: TutorState) -> dict:
        prepared = _prepare_tutor_turn(state)
        if prepared["fixed_response"]:
            return _tutor_state_update(prepared, AIMessage(content=prepared["fixed_response"]))
        return _tutor_state_update(prepared, llm.invoke(prepared["model_input"]))

    workflow = StateGraph(TutorState)
    workflow.add_node("tutor", tutor_node)
    workflow.add_edge(START, "tutor")
    workflow.add_edge("tutor", END)

    return workflow.compile(checkpointer=checkpointer)


def create_tutor_streaming_graph(llm: ChatGroq, checkpointer: InMemorySaver):
    """Build the same tutor workflow with native async model streaming.

    The node consumes ``llm.astream`` once and returns one final AIMessage, so
    LangGraph checkpoints the completed answer exactly once. The underlying
    model callbacks remain visible to ``astream_events`` for token forwarding.
    """

    async def tutor_node(state: TutorState, config: Optional[RunnableConfig] = None) -> dict:
        prepared = _prepare_tutor_turn(state)
        if prepared["fixed_response"]:
            return _tutor_state_update(prepared, AIMessage(content=prepared["fixed_response"]))

        parts: list[str] = []
        async for chunk in llm.astream(prepared["model_input"], config=config):
            text = _message_chunk_text(chunk)
            if text:
                parts.append(text)
        return _tutor_state_update(prepared, AIMessage(content="".join(parts)))

    workflow = StateGraph(TutorState)
    workflow.add_node("tutor", tutor_node)
    workflow.add_edge(START, "tutor")
    workflow.add_edge("tutor", END)
    return workflow.compile(checkpointer=checkpointer)


# =====================================================================
# 4. GROQ CONFIGURATION & VALIDATION
# =====================================================================

def load_and_validate_groq_llm() -> ChatGroq:
    """
    Loads GROQ_API_KEY and GROQ_MODEL from environment variables,
    validates that both are present and non-empty, and returns an
    initialized ChatGroq client.
    """
    load_dotenv()
    configure_runtime()
    if is_offline_mode():
        raise RuntimeError("Groq client initialization is disabled while TISW_OFFLINE=true.")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model_name = os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-20b"

    missing_vars = []
    if not api_key:
        missing_vars.append("GROQ_API_KEY")

    if missing_vars:
        missing = ", ".join(missing_vars)
        raise ValueError(
            f"Missing required environment variable(s): {missing}. "
            "Configure them in .env or the process environment."
        )

    if api_key in {"your_groq_api_key_here", "gsk_...", "changeme"}:
        raise ValueError(
            "GROQ_API_KEY appears to be a placeholder value. "
            "Set a valid Groq API key in .env."
        )

    try:
        llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.3,
        )
        return llm
    except Exception as exc:
        print(
            f"\n[Configuration Error] Failed to initialize ChatGroq with model '{model_name}'.",
            file=sys.stderr,
        )
        raise ValueError(f"Could not initialize the configured Groq model '{model_name}'.") from exc


# Global checkpointer and compiled tutor graph instance for API / server import
global_checkpointer = InMemorySaver()
tutor_stream_app = None
if is_offline_mode():
    tutor_app = None
else:
    try:
        _default_llm = load_and_validate_groq_llm()
        tutor_app = create_tutor_graph(_default_llm, global_checkpointer)
        tutor_stream_app = create_tutor_streaming_graph(_default_llm, global_checkpointer)
    except Exception:
        tutor_app = None


# =====================================================================
# 5. INTERACTIVE TERMINAL CHAT INTERFACE
# =====================================================================

def print_welcome_banner(thread_id: str) -> None:
    print("=" * 68)
    print("  TISW — Teaching in a Structured Way (AI Tutor)")
    print("=" * 68)
    print("Modes:")
    print("  - NORMAL MODE (Default): Direct, natural answers in simple English.")
    print("  - TISW MODE: Activated when you ask (e.g., 'explain using TISW').")
    print("    Answers arrive in four parts; say 'continue' for the next quarter.")
    print("    Returns to Normal Mode when you say 'now answer normally'.")
    print("-" * 68)
    print("Commands:")
    print("  - /tisw <question>  start a paged TISW walkthrough")
    print("  - /continue         next quarter of the walkthrough")
    print("  - /all              the whole walkthrough in one answer")
    print("  - /normal           leave TISW mode")
    print("  - Type 'new' to start a fresh conversation thread.")
    print("  - Type 'exit' or 'quit' to close the tutor.")
    print(f"  - Active Thread ID: {thread_id}")
    print("=" * 68)


def run_chat_loop() -> None:
    llm = load_and_validate_groq_llm()
    checkpointer = InMemorySaver()
    tutor_app = create_tutor_graph(llm, checkpointer)

    thread_id = f"study-session-{uuid.uuid4().hex[:8]}"
    print_welcome_banner(thread_id)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting Study Tutor. Happy learning!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q", ":q"}:
            print("\nExiting Study Tutor. Happy learning!")
            break

        if user_input.lower() in {"new", "/new", "reset", "/reset"}:
            thread_id = f"study-session-{uuid.uuid4().hex[:8]}"
            print(f"\n[Started a new conversation with Thread ID: {thread_id}]")
            continue

        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = tutor_app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            messages = result.get("messages", [])
            active_mode = result.get("active_mode", "normal")
            part = int(result.get("tisw_part") or 0)
            paged = paging_payload(result)
            if messages:
                tutor_reply = messages[-1].content
                if paged and paged["part"] < paged["parts_total"]:
                    mode_tag = f"[TISW PART {paged['part']}/{paged['parts_total']}]"
                else:
                    mode_tag = f"[{active_mode.upper()} MODE]"
                print(f"\nTutor {mode_tag}:\n{tutor_reply}")
                if paged and paged["part"] < paged["parts_total"]:
                    print(f"\n[Say 'continue' or /continue for part {paged['part'] + 1}.]")
            else:
                print("\n[Warning] Received an empty response from the tutor.")

        except KeyboardInterrupt:
            print("\n[Generation interrupted by user. You can ask another question or type 'exit' to quit.]")
        except Exception as exc:
            print(f"\n[Error] Groq API error: {type(exc).__name__}.\n", file=sys.stderr)


if __name__ == "__main__":
    run_chat_loop()

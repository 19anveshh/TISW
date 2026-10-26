"""
main.py - TISW AI Tutor powered by LangGraph, LangChain, and Groq.

Supports two distinct response modes:
1. NORMAL MODE (Default): Direct, natural, and clear explanations in simple,
   everyday English friendly to Indian college students. No forced rigid templates.
2. TISW MODE: Activated only when explicitly requested (e.g., 'explain using TISW').
   Provides deep first-principles structured teaching with flexible guidance.
"""

import os
import sys
import uuid
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages


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
You are TISW AI, operating in TISW MODE (Teaching in a Structured Way).
The student has explicitly requested the TISW structured teaching method.

Your goal is to build deep, rock-solid conceptual understanding from the ground up, using intuition and first principles.

# CORE METHODOLOGY (TISW MODE):
Structure the explanation so the student deeply understands the "Why", the "What", and the "How".
Depending on the topic, select and include the sections that genuinely help explain it:

1. The Core Idea: Explain the concept simply in plain, everyday words ("In a nutshell...").
2. Intuitive Example / Analogy: Use a relatable everyday scenario (such as cricket, UPI, college life, or daily activities) to make the mechanics click immediately before introducing heavy jargon.
3. Step-by-Step Explanation: Walk through the mechanism or logic in clear, sequential steps.
4. Why it Works (Under the Hood): Explain what is happening behind the scenes or why this approach was created.
5. Practical Example or Code: Provide a clean, minimal code snippet or practical calculation with clear comments.
6. Short Recap: Summarize the essential takeaway in 1–2 crisp sentences.

# IMPORTANT GUIDELINES FOR TISW MODE:
- These sections are GUIDELINES, not rigid mandatory checkboxes. Adapt the structure naturally to the topic. Do not make the answer unnecessarily bloated.
- Language: Keep English simple, clear, and easy to follow for Indian students. Avoid pompous or overly academic words. Explain technical terms immediately.
- Explaining Code: Decompose the logic before writing code, build from the simplest working version, and explain trade-offs.
- Maintain technical rigor while keeping the tone warm, humble, and supportive—like a great mentor sitting right next to the student.
- If the student says "Now answer normally" or "Switch to normal mode", gracefully return to direct, unformatted Normal Mode.
"""

# Alias for backwards compatibility
TEACHING_METHODOLOGY_PROMPT = TISW_MODE_PROMPT


# =====================================================================
# 2. INTELLIGENT MODE DETECTION
# =====================================================================

def detect_mode_request(user_text: str, current_mode: str = "normal") -> str:
    """
    Intelligently determines whether the student wants 'normal' or 'tisw' mode.

    NORMAL MODE: Default. Direct, natural, simple answers without forced templates.
    TISW MODE: Activated only when user explicitly asks for TISW / structured teaching.

    Returns:
        'normal' | 'tisw'
    """
    if not user_text:
        return current_mode or "normal"

    lowered = user_text.lower().strip()

    # Informational queries ABOUT TISW itself should be answered in Normal Mode
    info_tisw_queries = (
        "what is tisw",
        "what does tisw stand for",
        "what does tisw mean",
        "who is tisw",
        "tell me about tisw",
        "about tisw",
    )
    if any(q in lowered for q in info_tisw_queries):
        return current_mode or "normal"

    # 1. Check for explicit switch back to Normal Mode first
    normal_triggers = (
        "answer normally",
        "explain normally",
        "normal mode",
        "switch to normal",
        "back to normal",
        "reply normally",
        "respond normally",
        "regular mode",
        "regular answer",
        "direct answer",
        "simple answer",
        "standard mode",
        "just answer",
        "no tisw",
        "stop tisw",
        "stop using tisw",
        "without tisw",
        "don't use tisw",
        "dont use tisw",
        "do not use tisw",
        "speak normally",
        "talk normally",
    )
    if any(trigger in lowered for trigger in normal_triggers):
        return "normal"

    # 2. Check for explicit request for TISW Mode
    tisw_triggers = (
        "tisw",
        "teaching in a structured way",
        "structured teaching",
        "structured method",
        "structured approach",
        "structured way",
        "tisw method",
        "tisw approach",
        "using tisw",
        "with tisw",
        "in tisw",
        "tisw mode",
        "use tisw",
        "explain using tisw",
        "teach me using your method",
        "teach using your method",
        "explain using your method",
        "use your method",
        "use your structured method",
        "first principles using tisw",
        "teach me from first principles",
    )
    if any(trigger in lowered for trigger in tisw_triggers):
        return "tisw"

    # 3. Inherit active mode for follow-ups in the active conversation
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


def create_tutor_graph(llm: ChatGroq, checkpointer: InMemorySaver):
    """
    Builds and compiles the LangGraph workflow with a single `tutor` node
    and an in-memory checkpointer to preserve multi-turn conversation and mode state.
    """

    def tutor_node(state: TutorState) -> dict:
        messages: List[BaseMessage] = state.get("messages", [])
        current_prefs: List[str] = list(state.get("session_preferences") or [])
        current_mode: str = state.get("active_mode") or "normal"
        rag_context: Optional[str] = state.get("rag_context")

        # Check if latest user message establishes a preference or changes mode
        if messages and isinstance(messages[-1], HumanMessage):
            latest_text = str(messages[-1].content)

            # Detect persistent preferences (e.g. Hinglish, brevity)
            new_pref = detect_persistent_preference(latest_text)
            if new_pref and new_pref not in current_prefs:
                current_prefs.append(new_pref)

            # Detect mode transition (normal vs tisw)
            current_mode = detect_mode_request(latest_text, current_mode)

        # Build mode-specific SystemMessage with optional RAG grounding
        system_msg = build_system_message(current_mode, current_prefs, rag_context)
        model_input = [system_msg] + messages

        response = llm.invoke(model_input)

        return {
            "messages": [response],
            "session_preferences": current_prefs,
            "active_mode": current_mode,
            "rag_context": rag_context,
        }

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

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model_name = os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-20b"

    missing_vars = []
    if not api_key:
        missing_vars.append("GROQ_API_KEY")

    if missing_vars:
        print("\n[Configuration Error] Missing required environment variable(s):", file=sys.stderr)
        for var in missing_vars:
            print(f"  - {var}", file=sys.stderr)
        print(
            "\nPlease configure them in your .env file or environment before running:\n"
            "  GROQ_API_KEY=your_groq_api_key_here\n"
            "  GROQ_MODEL=your_chosen_groq_model_name\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if api_key in {"your_groq_api_key_here", "gsk_...", "changeme"}:
        print(
            "\n[Configuration Error] GROQ_API_KEY appears to be a placeholder value.\n"
            "Please set a valid Groq API key in your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.3,
        )
        return llm
    except Exception as exc:
        print(
            f"\n[Configuration Error] Failed to initialize ChatGroq with model '{model_name}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


# Global checkpointer and compiled tutor graph instance for API / server import
global_checkpointer = InMemorySaver()
try:
    _default_llm = load_and_validate_groq_llm()
    tutor_app = create_tutor_graph(_default_llm, global_checkpointer)
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
    print("    Returns to Normal Mode when you say 'now answer normally'.")
    print("-" * 68)
    print("Commands:")
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
            if messages:
                tutor_reply = messages[-1].content
                mode_tag = f"[{active_mode.upper()} MODE]"
                print(f"\nTutor {mode_tag}:\n{tutor_reply}")
            else:
                print("\n[Warning] Received an empty response from the tutor.")

        except KeyboardInterrupt:
            print("\n[Generation interrupted by user. You can ask another question or type 'exit' to quit.]")
        except Exception as exc:
            error_msg = str(exc)
            print(f"\n[Error] Groq API error: {error_msg}\n", file=sys.stderr)


if __name__ == "__main__":
    run_chat_loop()
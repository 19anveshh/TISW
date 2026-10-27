"""Stable scoring rubrics used by the optional LLM judge.

Scores are 1--5 and are based on observable evidence.  Length, headings, or
the presence of an analogy are never sufficient for a high score.
"""

RUBRIC_VERSION = "1.0.0"

GENERAL_RUBRIC = {
    "factual_correctness": {
        "description": "Claims agree with the reference facts and do not invent specifics.",
        "low": "Contains a material false claim, contradiction, or fabricated detail.",
        "medium": "Mostly correct but omits an important qualification or contains a minor error.",
        "high": "Accurate, appropriately qualified, and distinguishes fact from inference.",
    },
    "relevance": {
        "description": "Directly addresses the user's question and intended task.",
        "low": "Does not answer the request or spends most of the answer elsewhere.",
        "medium": "Answers the main request but includes avoidable tangents or misses a requested part.",
        "high": "Focused answer that covers the requested task at the appropriate scope.",
    },
    "completeness": {
        "description": "Covers the reference facts and important constraints for this case.",
        "low": "Missing the central answer or a required constraint.",
        "medium": "Covers the central answer but misses one meaningful supporting point.",
        "high": "Covers the case-specific requirements without padding.",
    },
    "instruction_following": {
        "description": "Follows explicit output, format, safety, and mode instructions.",
        "low": "Ignores or contradicts an explicit instruction.",
        "medium": "Follows the main instruction but misses a secondary constraint.",
        "high": "Follows all applicable instructions and does not invent requirements.",
    },
    "uncertainty_and_honesty": {
        "description": "Expresses uncertainty when evidence is insufficient and avoids unsupported claims.",
        "low": "States unsupported information confidently or pretends to have performed an action.",
        "medium": "Some qualification is present but confidence or evidence boundaries are imperfect.",
        "high": "Clearly separates known facts, assumptions, and limits; asks for missing information when needed.",
    },
    "clarity_and_coherence": {
        "description": "Readable, logically organized, and understandable for the intended learner.",
        "low": "Confusing, contradictory, or too opaque to use.",
        "medium": "Understandable with awkward structure or unexplained jumps.",
        "high": "Natural progression, clear wording, and proportionate detail.",
    },
}

TISW_RUBRIC = {
    "beginner_clarity": {
        "description": "A beginner can form a correct mental model without losing technical accuracy.",
        "low": "Uses jargon or abstractions without making the core idea understandable.",
        "medium": "Core idea is understandable but one important mental-model gap remains.",
        "high": "Builds an accurate, accessible mental model in simple language.",
    },
    "example_quality": {
        "description": "Examples or analogies are used when useful, accurately, and with limits when needed.",
        "low": "Uses a misleading analogy or forces an irrelevant example.",
        "medium": "Example helps but is generic, incomplete, or not clearly connected to the concept.",
        "high": "A precise example materially improves understanding; no example is penalized when direct explanation is better.",
    },
    "terminology_progression": {
        "description": "Real terminology is introduced after enough intuition and explained naturally.",
        "low": "Jargon is unexplained or the answer avoids necessary precise terms.",
        "medium": "Terminology is mostly accessible but a few terms arrive abruptly.",
        "high": "Plain-language intuition connects cleanly to accurate terminology.",
    },
    "reasoning_and_why": {
        "description": "Explains purpose, mechanism, and reasoning rather than only definitions.",
        "low": "Lists what something is without explaining why or how.",
        "medium": "Explains mechanism or purpose but not the most important reasoning.",
        "high": "Makes the causal logic and design reasoning clear for this question.",
    },
    "depth_and_tradeoffs": {
        "description": "Adds alternatives, limits, or trade-offs when relevant and stays proportionate.",
        "low": "Shallow, misleadingly absolute, or needlessly verbose.",
        "medium": "Useful depth but misses a relevant limitation or includes some padding.",
        "high": "Depth matches the learner and topic; limitations and alternatives appear only when useful.",
    },
    "learner_adaptation": {
        "description": "Uses demonstrated knowledge and responds to confusion or follow-up context.",
        "low": "Repeats basics despite evidence or fails to address the learner's misconception.",
        "medium": "Acknowledges context but adapts only partially.",
        "high": "Builds directly on prior turns and resolves the learner's actual gap.",
    },
}

CODING_RUBRIC = {
    "code_correctness": {
        "description": "The proposed code is syntactically valid and solves the stated task.",
        "low": "Does not run or produces materially wrong output.",
        "medium": "Works for the main example but has a relevant edge-case or design defect.",
        "high": "Runs and meets the stated behavior, including the case-specific constraints.",
    },
    "explanation_accuracy": {
        "description": "Explanation correctly describes the code and its behavior.",
        "low": "Claims behavior the code does not have or misses the actual bug.",
        "medium": "Mostly accurate but omits an important dependency or caveat.",
        "high": "Accurately connects important lines, state changes, and outputs to the reasoning.",
    },
    "debugging_reasoning": {
        "description": "When debugging is requested, identifies the actual cause and a verifiable fix.",
        "low": "Suggests unrelated changes or misdiagnoses the error.",
        "medium": "Finds the symptom and a workable fix without explaining the cause fully.",
        "high": "Traces the cause, explains why the fix works, and distinguishes alternatives.",
    },
    "transferability": {
        "description": "A learner could apply the idea to a nearby new problem.",
        "low": "Only gives an opaque snippet or memorized answer.",
        "medium": "Some reusable reasoning is present but dependencies are unclear.",
        "high": "Explains the general pattern, boundaries, and how to adapt it.",
    },
}


def rubric_for_case(case: dict) -> dict:
    """Return the stable rubric groups applicable to a dataset case."""
    groups = {"general": GENERAL_RUBRIC}
    if case.get("category") in {"explanation", "deep_concept", "multi_turn", "paired_teaching"} or case.get("mode") == "tisw":
        groups["tisw"] = TISW_RUBRIC
    if case.get("category") == "coding":
        groups["coding"] = CODING_RUBRIC
    if case.get("rag"):
        groups["rag"] = {
            "faithfulness": {
                "description": "Claims are supported by the retrieved excerpts.",
                "low": "Adds unsupported or contradictory document claims.",
                "medium": "Mostly grounded but includes an unsupported detail.",
                "high": "Claims and citations are traceable to the supplied excerpts.",
            },
            "citation_accuracy": {
                "description": "Citations use only returned source metadata and support the nearby claim.",
                "low": "Fabricates a source, page, or citation.",
                "medium": "Cites a relevant file but citation placement or specificity is incomplete.",
                "high": "Citations accurately identify the evidence used and acknowledge missing evidence.",
            },
        }
    return groups


def rubric_prompt(case: dict) -> str:
    """Serialize only the applicable rubric, keeping judge prompts stable."""
    import json

    return json.dumps(rubric_for_case(case), sort_keys=True, indent=2)

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from langchain_openai import ChatOpenAI

from services.chatbot_runner import (
    ChatbotResources,
    ChatbotRunResult,
    run_cchn_chatbot,
)

SHORT_TERM_TURNS_DEFAULT = 4
SUMMARY_WORD_LIMIT = 120


def update_short_term(
    history: Sequence[Dict[str, str]],
    max_turns: int = SHORT_TERM_TURNS_DEFAULT,
) -> List[Dict[str, str]]:
    """Return only the most recent user/assistant turns for prompt conditioning."""

    if max_turns <= 0:
        max_turns = SHORT_TERM_TURNS_DEFAULT

    filtered = [
        {"role": entry.get("role", ""), "content": entry.get("content", "")}
        for entry in history
        if entry.get("role") in {"user", "assistant"}
    ]

    return filtered[-max_turns * 2 :]


def update_long_term(
    existing_summary: str | None,
    history_slice: Sequence[Dict[str, str]],
    llm: ChatOpenAI,
    max_words: int = SUMMARY_WORD_LIMIT,
) -> str:
    """Refresh the rolling long-term summary using the most recent dialogue window."""

    if not history_slice:
        return existing_summary or ""

    formatted_history = _format_history(history_slice)
    summary_seed = existing_summary.strip() if existing_summary else "None yet."

    prompt = f"""
You maintain a concise running summary of a humanitarian negotiation coaching conversation.
Blend the previous summary with the recent messages and keep the result under {max_words} words.
Focus on situational context, stakeholder goals, and any unresolved needs.

Previous summary:
{summary_seed}

Recent dialogue:
{formatted_history}

Return only the refreshed summary in plain sentences.
""".strip()

    response = llm.invoke(prompt)
    return _coerce_content(response)


def run_chatbot_with_context(
    user_message: str,
    short_term: Sequence[Dict[str, str]],
    long_term_summary: str | None,
    resources: ChatbotResources,
) -> Tuple[str, str, ChatbotRunResult]:
    """Run the existing chatbot with additional conversational context."""

    trimmed_message = (user_message or "").strip()
    if not trimmed_message:
        raise ValueError("user_message must be a non-empty string.")

    context_sections: List[str] = []
    if long_term_summary:
        context_sections.append(f"Long-term summary:\n{long_term_summary.strip()}")
    if short_term:
        context_sections.append("Recent turns:\n" + _format_history(short_term))

    contextual_question = (
        "You are an AI negotiation coach. Use the provided context to answer the latest "
        "question while grounding everything in the CCHN Field Manual.\n"
    )
    if context_sections:
        contextual_question += "\n".join(context_sections) + "\n"
    contextual_question += f"\nLatest user question:\n{trimmed_message}"

    result = run_cchn_chatbot(contextual_question, resources)
    follow_up = _generate_follow_up_question(trimmed_message, result.answer, resources.llm)

    answer_with_follow_up = f"{result.answer.strip()}\n\n_Follow-up question:_ {follow_up}"
    return answer_with_follow_up, follow_up, result


def _format_history(history: Sequence[Dict[str, str]]) -> str:
    lines: List[str] = []
    for entry in history:
        role = entry.get("role", "assistant").capitalize()
        content = entry.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _generate_follow_up_question(
    user_message: str,
    answer: str,
    llm: ChatOpenAI,
) -> str:
    prompt = f"""
You just answered a humanitarian negotiation question.
User message: {user_message.strip()}
Your answer: {answer.strip()}

Ask exactly one short follow-up question (<=25 words) that would help you
offer more tailored guidance next. The question must sound natural and
be directly useful to the negotiator.

Return only the question.
""".strip()

    response = llm.invoke(prompt)
    return _coerce_content(response)


def _coerce_content(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(parts).strip()
    return str(message).strip()



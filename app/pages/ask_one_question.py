"""
Single-mode chatbot page with robust rewriting, intent-aware retrieval, and grounded answers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from services.chatbot_runner import (
    ChatbotResources,
    get_chat_resources,
    run_cchn_chatbot,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

CHAT_HISTORY_KEY = "cchn_chat_history"
DEBUG_INFO_KEY = "cchn_debug_info"


def get_chat_history() -> List[Dict[str, str]]:
    if CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[CHAT_HISTORY_KEY] = []
    return st.session_state[CHAT_HISTORY_KEY]


def get_debug_info() -> Dict[str, Any] | None:
    return st.session_state.get(DEBUG_INFO_KEY)


def update_debug_info(payload: Dict[str, Any]) -> None:
    st.session_state[DEBUG_INFO_KEY] = payload


def append_to_history(role: str, content: str) -> None:
    history = get_chat_history()
    history.append({"role": role, "content": content})
    st.session_state[CHAT_HISTORY_KEY] = history


def render_chat_history(history: List[Dict[str, str]]) -> None:
    if not history:
        st.info("Ask a negotiation question to start the conversation.")
        return

    for entry in history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])


def render_debug_panel(debug_info: Dict[str, Any] | None) -> None:
    with st.expander("Advanced debug", expanded=False):
        if not debug_info:
            st.write("Run the chatbot to inspect rewriting, intents, and retrieval details.")
            return

        st.markdown("**Rewritten query**")
        st.code(debug_info.get("rewritten_query", ""))

        st.markdown("**Detected intent**")
        st.write(debug_info.get("intent", ""))

        st.markdown("**Concept keywords**")
        keywords = debug_info.get("concept_keywords") or []
        st.write(", ".join(keywords) if keywords else "—")

        st.markdown("**Subqueries used for retrieval**")
        for query in debug_info.get("subqueries", []):
            st.write(f"- {query}")

        st.markdown("**Top retrieved chunks**")
        chunks = debug_info.get("chunks", [])
        if not chunks:
            st.write("No chunks were retrieved.")
        else:
            for chunk in chunks:
                st.write(
                    f"- Page {chunk.get('page', 'NA')} • {chunk.get('section', 'Passage')} "
                    f"(score {chunk.get('score')}, query: {chunk.get('query')})"
                )


def handle_user_question(
    question: str,
    resources: ChatbotResources,
) -> str:
    with st.spinner("Running the grounded negotiation pipeline..."):
        result = run_cchn_chatbot(question, resources)

    update_debug_info(result.debug_info)
    return result.answer


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Ask one question", page_icon="🤝")

    st.title("Ask one question")
    st.write(
        "Ask a negotiation question and receive a careful answer grounded in the CCHN Field Manual."
    )
    st.caption("Internally the assistant rewrites your question, runs intent-guided retrieval, and composes a grounded response.")

    with st.form("chatbot_question_form", clear_on_submit=True):
        question = st.text_area(
            "Your question",
            placeholder=(
                "For example: I am facing hostility from a local commander—how should I "
                "approach the next meeting?"
            ),
            height=140,
        )
        send_clicked = st.form_submit_button("Send", type="primary")

    try:
        resources = get_chat_resources()
    except Exception as exc:  # pragma: no cover - surfaced to UI
        st.error(str(exc))
        st.stop()

    if send_clicked:
        trimmed_question = (question or "").strip()
        if not trimmed_question:
            st.warning("Please provide a question before sending.")
        else:
            try:
                answer = handle_user_question(
                    trimmed_question,
                    resources,
                )
            except Exception as exc:  # pragma: no cover - surfaced to UI
                st.error(f"Something went wrong: {exc}")
            else:
                append_to_history("user", trimmed_question)
                append_to_history("assistant", answer)

    st.subheader("Conversation history")
    render_chat_history(get_chat_history())

    render_debug_panel(get_debug_info())


if __name__ == "__main__":
    main()


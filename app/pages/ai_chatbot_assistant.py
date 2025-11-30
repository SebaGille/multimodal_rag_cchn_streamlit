from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

from services.chatbot_runner import ChatbotResources, get_chat_resources
from services.conversation_agent import (
    run_chatbot_with_context,
    update_long_term,
    update_short_term,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HISTORY_KEY = "ai_assistant_history"
SHORT_TERM_KEY = "ai_assistant_short_term"
LONG_TERM_KEY = "ai_assistant_long_term"


def _ensure_session_state() -> None:
    st.session_state.setdefault(HISTORY_KEY, [])
    st.session_state.setdefault(SHORT_TERM_KEY, [])
    st.session_state.setdefault(LONG_TERM_KEY, "")


def _inject_chat_styles() -> None:
    """Lightweight visual cues for user turns."""
    st.markdown(
        """
        <style>
        div[data-testid="stChatMessage"][data-agent="user"] {
            justify-content: flex-end;
        }
        div[data-testid="stChatMessage"][data-agent="user"] div[data-testid="stChatMessageContent"] {
            background: #f2f7eb;
            border: 1px solid #dbe8cc;
        }
        div[data-testid="stChatMessage"][data-agent="user"] div[data-testid="stMarkdownContainer"] {
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_history() -> List[Dict[str, str]]:
    return st.session_state[HISTORY_KEY]


def _append_message(role: str, content: str) -> None:
    st.session_state[HISTORY_KEY].append({"role": role, "content": content})


def _render_chat_history(history: List[Dict[str, str]]) -> None:
    if not history:
        st.caption("No messages yet. Share what is happening in your negotiation to begin.")
        return

    for entry in history:
        role = entry.get("role", "assistant")
        with st.chat_message(role if role in {"user", "assistant"} else "assistant"):
            st.markdown(entry.get("content", ""))


def _handle_new_message(user_input: str, resources: ChatbotResources) -> bool:
    trimmed = (user_input or "").strip()
    if not trimmed:
        st.warning("Please enter a message before sending.")
        return False

    _append_message("user", trimmed)
    st.session_state[SHORT_TERM_KEY] = update_short_term(_get_history())
    st.session_state[LONG_TERM_KEY] = update_long_term(
        st.session_state[LONG_TERM_KEY],
        st.session_state[SHORT_TERM_KEY],
        resources.llm,
    )

    with st.spinner("Thinking with your conversation context..."):
        try:
            assistant_message, _follow_up, result = run_chatbot_with_context(
                trimmed,
                st.session_state[SHORT_TERM_KEY],
                st.session_state[LONG_TERM_KEY],
                resources,
            )
        except Exception as exc:  # pragma: no cover
            st.error(f"Something went wrong: {exc}")
            return False

    _append_message("assistant", assistant_message)
    st.session_state[SHORT_TERM_KEY] = update_short_term(_get_history())
    return True




def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="AI Chatbot Assistant", page_icon="💬", layout="wide")
    _ensure_session_state()
    _inject_chat_styles()

    st.title("AI Chatbot Assistant")
    st.caption(
        "Share your negotiation updates. The assistant keeps recent turns and a rolling summary "
        "to stay grounded in your evolving context."
    )

    try:
        resources = get_chat_resources()
    except Exception as exc:  # pragma: no cover
        st.error(str(exc))
        st.stop()

    _render_chat_history(_get_history())

    user_input = st.chat_input("Describe what is happening or what you need help with...")
    if user_input is not None:
        if _handle_new_message(user_input, resources):
            st.rerun()


if __name__ == "__main__":
    main()


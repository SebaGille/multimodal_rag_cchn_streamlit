"""
Streamlit entrypoint for the CCHN RAG prototype.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def main() -> None:
    chatbot_page = st.Page(
        "pages/chatbot.py",
        title="Chatbot prototype",
        default=True,
    )
    auto_eval_page = st.Page(
        "pages/chatbot_auto_evaluation.py",
        title="Chatbot auto evaluation",
    )
    learnings_page = st.Page(
        "pages/chatbot_learnings.py",
        title="Chatbot learnings",
    )

    navigation = st.navigation([chatbot_page, auto_eval_page, learnings_page])
    navigation.run()


if __name__ == "__main__":
    main()


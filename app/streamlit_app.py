"""
Streamlit entrypoint for the CCHN RAG prototype.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def main() -> None:
    # Temporarily hide the chatbot assistant page from navigation.
    # assistant_page = st.Page(
    #     "pages/ai_chatbot_assistant.py",
    #     title="AI Chatbot Assistant",
    #     default=True,
    # )
    ask_one_page = st.Page(
        "pages/ask_one_question.py",
        title="Ask one question",
        default=True,
    )
    agentic_page = st.Page(
        "pages/agentic_ai_assistant.py",
        title="Agentic AI assistant",
    )
    auto_eval_page = st.Page(
        "pages/chatbot_auto_evaluation.py",
        title="LLM-as-a-Judge",
    )
    eval_analytics_page = st.Page(
        "pages/evaluation_analytics.py",
        title="Evaluation analytics",
    )
    learnings_page = st.Page(
        "pages/project_insights_and_learnings.py",
        title="Project insights and learnings",
    )
    pipeline_page = st.Page(
        "pages/pipeline_details.py",
        title="Pipeline details",
    )

    navigation = st.navigation(
        [
            ask_one_page,
            agentic_page,
            auto_eval_page,
            eval_analytics_page,
            learnings_page,
            pipeline_page,
        ]
    )
    navigation.run()


if __name__ == "__main__":
    main()


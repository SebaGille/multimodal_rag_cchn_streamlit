"""
Single-mode chatbot page with robust rewriting, intent-aware retrieval, and grounded answers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from chat_pipeline import generate_answer, retrieve_passages, rewrite_query
from rag_state import get_rag_settings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

CHAT_HISTORY_KEY = "cchn_chat_history"
DEBUG_INFO_KEY = "cchn_debug_info"


@st.cache_resource(show_spinner=False)
def load_vectorstore(path: str, embedding_model: str) -> FAISS:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def build_llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=0)


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
    store: FAISS,
    llm: ChatOpenAI,
    *,
    per_query_k: int,
    final_k: int,
) -> str:
    with st.spinner("Rewriting and classifying your question..."):
        try:
            rewrite_result = rewrite_query(question, llm)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            raise RuntimeError(f"Query rewriting failed: {exc}") from exc

    with st.spinner("Retrieving focused passages from the manual..."):
        try:
            chunks, debug_info = retrieve_passages(
                store,
                rewrite_result.rewritten_query,
                rewrite_result.intent,
                rewrite_result.concept_keywords,
                per_query_k=per_query_k,
                final_k=final_k,
            )
        except Exception as exc:  # pragma: no cover - surfaced to UI
            raise RuntimeError(f"Retrieval failed: {exc}") from exc

    with st.spinner("Generating a grounded response..."):
        try:
            answer = generate_answer(
                user_question=question,
                rewritten_query=rewrite_result.rewritten_query,
                context_chunks=chunks,
                llm=llm,
            )
        except Exception as exc:  # pragma: no cover - surfaced to UI
            raise RuntimeError(f"Answer generation failed: {exc}") from exc

    update_debug_info(
        {
            "rewritten_query": rewrite_result.rewritten_query,
            "intent": rewrite_result.intent,
            "concept_keywords": rewrite_result.concept_keywords,
            "subqueries": debug_info.get("queries", []),
            "chunks": debug_info.get("chunks", []),
        }
    )

    return answer


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="CCHN Negotiation Chatbot", page_icon="🤝")

    settings = get_rag_settings()
    vectorstore_path = Path(settings["vectorstore_path"]).expanduser()

    st.title("CCHN Negotiation Chatbot")
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

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.error("Set OPENAI_API_KEY in your environment before using the chatbot.")
        st.stop()

    if not vectorstore_path.exists():
        st.error(f"Vectorstore directory '{vectorstore_path}' was not found.")
        st.stop()

    try:
        store = load_vectorstore(str(vectorstore_path), settings["embedding_model"])
    except Exception as exc:  # pragma: no cover - surface to UI
        st.error(f"Failed to load vectorstore: {exc}")
        st.stop()

    llm = build_llm(settings["chat_model"])

    per_query_k = max(12, settings.get("top_k", 4) * 3)
    final_k = max(10, settings.get("top_k", 4) * 2)

    if send_clicked:
        trimmed_question = (question or "").strip()
        if not trimmed_question:
            st.warning("Please provide a question before sending.")
        else:
            try:
                answer = handle_user_question(
                    trimmed_question,
                    store,
                    llm,
                    per_query_k=per_query_k,
                    final_k=final_k,
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


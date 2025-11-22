from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import streamlit as st

DEFAULT_PROMPT_TEMPLATE = """You are an assistant helping users understand the full CCHN Field Manual.

You will receive:
1) A user question.
2) Several passages from the CCHN Field Manual (the context).

Your task:
- Answer the question using only the information from the provided context.
- If the context does not contain enough information to answer reliably, say that the manual does not provide a clear answer.
- Be concise, clear, and neutral in tone.
- If relevant, refer to the content as "the manual", not by page or chunk ID, unless page numbers are provided in the context.
"""

DEFAULT_SETTINGS: Dict[str, Any] = {
    "vectorstore_path": str(Path("vectorstores/full_manual_faiss")),
    "embedding_model": "text-embedding-3-small",
    "chat_model": "gpt-4o-mini",
    "top_k": 4,
    "prompt_template": DEFAULT_PROMPT_TEMPLATE.strip(),
}

DEFAULT_RUN_STATE: Dict[str, Any] = {
    "question": "",
    "selected_modes": [],
    "mode_results": [],
    "retrieved_count": 0,
    "error": "",
    "last_run_settings": {},
    "planning_artifacts": {},
}


def get_rag_settings() -> Dict[str, Any]:
    if "rag_settings" not in st.session_state:
        st.session_state["rag_settings"] = deepcopy(DEFAULT_SETTINGS)
    return st.session_state["rag_settings"]


def get_run_state() -> Dict[str, Any]:
    if "rag_run" not in st.session_state:
        st.session_state["rag_run"] = deepcopy(DEFAULT_RUN_STATE)
    return st.session_state["rag_run"]


def update_rag_settings(new_settings: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_rag_settings()
    settings.update(new_settings)
    st.session_state["rag_settings"] = settings
    return settings


def update_run_state(**kwargs: Any) -> Dict[str, Any]:
    run_state = get_run_state()
    run_state.update(kwargs)
    st.session_state["rag_run"] = run_state
    return run_state


from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from chat_pipeline import (
    RetrievedChunk,
    RewriteResult,
    generate_answer,
    retrieve_passages,
    rewrite_query,
)
from rag_state import get_rag_settings


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass(frozen=True)
class ChatbotResources:
    store: FAISS
    llm: ChatOpenAI
    per_query_k: int
    final_k: int


@dataclass
class ChatbotRunResult:
    answer: str
    rewrite_result: RewriteResult
    chunks: List[RetrievedChunk]
    debug_info: Dict[str, Any]


@st.cache_resource(show_spinner=False)
def _load_vectorstore(path: str, embedding_model: str) -> FAISS:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def _build_llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=0)


def get_chat_resources() -> ChatbotResources:
    """Load reusable chatbot resources according to current RAG settings."""

    settings = get_rag_settings()
    vectorstore_path = Path(settings["vectorstore_path"]).expanduser()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not vectorstore_path.exists():
        raise FileNotFoundError(f"Vectorstore directory '{vectorstore_path}' was not found.")

    store = _load_vectorstore(str(vectorstore_path), settings["embedding_model"])
    llm = _build_llm(settings["chat_model"])

    base_top_k = max(1, settings.get("top_k", 4))
    per_query_k = max(12, base_top_k * 3)
    final_k = max(10, base_top_k * 2)

    return ChatbotResources(
        store=store,
        llm=llm,
        per_query_k=per_query_k,
        final_k=final_k,
    )


def run_cchn_chatbot(question: str, resources: ChatbotResources) -> ChatbotRunResult:
    """Run the existing RAG pipeline for a single user question."""

    trimmed_question = (question or "").strip()
    if not trimmed_question:
        raise ValueError("Question must be a non-empty string.")

    rewrite_result = rewrite_query(trimmed_question, resources.llm)
    chunks, retrieval_debug = retrieve_passages(
        resources.store,
        rewrite_result.rewritten_query,
        rewrite_result.intent,
        rewrite_result.concept_keywords,
        per_query_k=resources.per_query_k,
        final_k=resources.final_k,
    )
    answer = generate_answer(
        user_question=trimmed_question,
        rewritten_query=rewrite_result.rewritten_query,
        context_chunks=chunks,
        llm=resources.llm,
    )

    debug_payload = {
        "rewritten_query": rewrite_result.rewritten_query,
        "intent": rewrite_result.intent,
        "concept_keywords": rewrite_result.concept_keywords,
        "subqueries": retrieval_debug.get("queries", []),
        "chunks": retrieval_debug.get("chunks", []),
    }

    return ChatbotRunResult(
        answer=answer,
        rewrite_result=rewrite_result,
        chunks=chunks,
        debug_info=debug_payload,
    )


"""
Main chatbot page for the CCHN RAG prototype.
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import shorten
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_state import get_rag_settings, get_run_state, update_run_state

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@st.cache_resource
def load_vectorstore(path: str, embedding_model: str):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_chain(model_name: str):
    return ChatOpenAI(model=model_name, temperature=0)


def retrieve_documents(query: str, store, top_k: int) -> List[Document]:
    retriever = store.as_retriever(search_kwargs={"k": top_k})
    return retriever.invoke(query)


def format_context_for_prompt(documents: List[Document]) -> str:
    sections = []
    for doc in documents:
        meta = doc.metadata or {}
        page = meta.get("page_number", "NA")
        element = meta.get("element_type", "Passage")
        snippet = doc.page_content.strip()
        sections.append(f"Page {page} ({element}):\n{snippet}")
    return "\n\n".join(sections)


def summarize_document(doc: Document, width: int = 420) -> Dict[str, str]:
    meta = doc.metadata or {}
    page = meta.get("page_number", "NA")
    element = meta.get("element_type", "Passage")
    snippet = " ".join(doc.page_content.strip().split())
    excerpt = shorten(snippet, width=width, placeholder="...")
    return {"page": page, "element": element, "excerpt": excerpt}


def build_prompt(template: str, context_text: str, question: str) -> str:
    template = template.strip()
    return (
        f"{template}\n\n"
        f"Context from the manual:\n{context_text}\n\n"
        f"User question:\n{question}\n\n"
        "Answer in clear English using only the provided context."
    )


def run_inference(question: str, settings: Dict[str, Any]) -> None:
    path = Path(settings["vectorstore_path"]).expanduser()
    update_run_state(
        question=question,
        answer="",
        references=[],
        retrieved_count=0,
        error="",
    )

    if not path.exists():
        error_msg = f"Vectorstore directory '{path}' was not found."
        update_run_state(error=error_msg)
        st.error(error_msg)
        return

    try:
        with st.spinner("Retrieving passages..."):
            store = load_vectorstore(str(path), settings["embedding_model"])
            retrieved_documents = retrieve_documents(question, store, settings["top_k"])
    except Exception as exc:
        error_msg = f"Failed to retrieve passages: {exc}"
        update_run_state(answer="", references=[], retrieved_count=0, error=error_msg)
        st.error(error_msg)
        return

    context_text = format_context_for_prompt(retrieved_documents)
    prompt = build_prompt(settings["prompt_template"], context_text, question)
    llm = build_chain(settings["chat_model"])

    try:
        with st.spinner("Generating answer..."):
            response = llm.invoke(
                [
                    (
                        "system",
                        "You help users understand the CCHN Field Manual.",
                    ),
                    (
                        "human",
                        prompt,
                    ),
                ]
            )
    except Exception as exc:
        error_msg = f"Failed to generate answer: {exc}"
        update_run_state(
            answer="",
            references=[],
            retrieved_count=len(retrieved_documents),
            error=error_msg,
        )
        st.error(error_msg)
        return

    references = [summarize_document(doc) for doc in retrieved_documents]
    update_run_state(
        answer=response.content,
        references=references,
        retrieved_count=len(retrieved_documents),
        error="",
        last_run_settings={
            "vectorstore_path": str(path),
            "embedding_model": settings["embedding_model"],
            "chat_model": settings["chat_model"],
            "top_k": settings["top_k"],
        },
    )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Chatbot prototype", page_icon="💬")

    settings = get_rag_settings()
    run_state = get_run_state()

    if "chatbot_question_input" not in st.session_state:
        st.session_state["chatbot_question_input"] = run_state["question"]

    st.title("CCHN Manual Assistant")
    st.write(
        "Ask a question about the CCHN Field Manual and this assistant will "
        "look up the most relevant passages before answering."
    )
    st.info("Need to tweak the engine? Head to the Engine page in the sidebar.")

    question = st.text_input(
        "Your question",
        placeholder="For example: What is a frontline negotiator?",
        key="chatbot_question_input",
    )
    submit = st.button("Get answer", type="primary")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.error("Set OPENAI_API_KEY in your environment before using the app.")
        st.stop()

    if submit:
        trimmed_question = question.strip()
        if not trimmed_question:
            st.warning("Please enter a question before requesting an answer.")
        else:
            run_inference(trimmed_question, settings)
            run_state = get_run_state()

    if run_state.get("error"):
        st.error(run_state["error"])

    if run_state.get("answer"):
        st.subheader("Answer")
        st.write(run_state["answer"])

    references = run_state.get("references", [])
    if references:
        st.subheader("Passages referenced")
        for ref in references:
            st.markdown(f"**Page {ref['page']} | {ref['element']}**")
            st.write(ref["excerpt"])


if __name__ == "__main__":
    main()


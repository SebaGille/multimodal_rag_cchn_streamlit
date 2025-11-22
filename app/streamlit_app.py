"""
Streamlit UI for the CCHN RAG prototype.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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


def format_chunks(documents: List[Document]) -> List[str]:
    items = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        page = meta.get("page_number", "NA")
        element = meta.get("element_type", "unknown")
        snippet = doc.page_content.strip()
        items.append(f"[Chunk {idx}] page {page} ({element})\n{snippet}")
    return items


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="CCHN Manual Assistant")

    st.title("CCHN Manual Assistant")
    st.caption("Retrieval-augmented QA over the CCHN Field Manual extract.")

    default_store = Path("vectorstores/short_extract_faiss")

    with st.sidebar:
        st.header("Configuration")
        vectorstore_path = st.text_input(
            "Vectorstore directory",
            value=str(default_store),
        )
        embedding_model = st.text_input("Embedding model", value="text-embedding-3-small")
        chat_model = st.text_input("Chat model", value="gpt-4o-mini")
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.error("Set OPENAI_API_KEY in your environment before using the app.")
        st.stop()

    store = load_vectorstore(vectorstore_path, embedding_model)
    llm = build_chain(chat_model)

    question = st.text_input("Ask a question about the manual")
    submit = st.button("Ask", type="primary")

    if submit and question:
        with st.spinner("Retrieving context..."):
            documents = retrieve_documents(question, store, top_k)
        context_text = "\n\n".join(format_chunks(documents))
        with st.spinner("Generating answer..."):
            response = llm.invoke(
                [
                    (
                        "system",
                        "You are a grounded assistant answering questions using the provided context.",
                    ),
                    (
                        "human",
                        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer with citations like [Chunk x].",
                    ),
                ]
            )

        st.subheader("Answer")
        st.write(response.content)

        st.subheader("Retrieved chunks")
        for chunk in format_chunks(documents):
            st.code(chunk)


if __name__ == "__main__":
    main()


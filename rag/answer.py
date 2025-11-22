"""
Simple retrieval-augmented answering helper for the CCHN RAG prototype.

Usage:
    python rag/answer.py \
        --question "What is de-escalation?" \
        --vectorstore vectorstores/short_extract_faiss
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def format_context(documents: List[Document]) -> str:
    """Combine retrieved chunks with lightweight metadata for grounding."""
    context_blocks = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        page = meta.get("page_number", "NA")
        element = meta.get("element_type", "unknown")
        context_blocks.append(
            f"[Chunk {idx}] (page {page}, {element})\n{doc.page_content.strip()}"
        )
    return "\n\n".join(context_blocks)


def format_citations(documents: List[Document]) -> str:
    citations = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        snippet = doc.page_content[:80].strip().replace("\n", " ")
        citations.append(
            f"[Chunk {idx}] page {meta.get('page_number', 'NA')} • {snippet}..."
        )
    return "\n".join(citations)


def load_retriever(
    vectorstore_dir: Path,
    embedding_model: str,
    k: int,
):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_chain(model_name: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grounded assistant answering questions using the provided context.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\nAnswer with citations like [Chunk x].",
            ),
        ]
    )
    llm = ChatOpenAI(model=model_name, temperature=0)
    return prompt | llm


def answer_question(
    question: str,
    vectorstore_dir: Path,
    embedding_model: str = "text-embedding-3-small",
    chat_model: str = "gpt-4o-mini",
    top_k: int = 4,
) -> Dict[str, str]:
    retriever = load_retriever(vectorstore_dir, embedding_model, top_k)
    documents = retriever.invoke(question)
    chain = build_chain(chat_model)
    context = format_context(documents)
    response = chain.invoke({"context": context, "question": question})
    return {
        "answer": response.content,
        "citations": format_citations(documents),
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question.")
    parser.add_argument("--question", type=str, required=True, help="User question.")
    parser.add_argument(
        "--vectorstore",
        type=Path,
        default=Path("vectorstores/short_extract_faiss"),
        help="Directory containing FAISS index.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model.",
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat completion model.",
    )
    parser.add_argument("--k", type=int, default=4, help="Retriever top-k.")

    args = parser.parse_args()

    result = answer_question(
        question=args.question,
        vectorstore_dir=args.vectorstore,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        top_k=args.k,
    )
    print("Answer:\n", result["answer"])
    print("\nCitations:\n", result["citations"])


if __name__ == "__main__":
    main()


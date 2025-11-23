"""
Regression harness that mirrors the single-mode chatbot pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.chat_pipeline import (  # pylint: disable=wrong-import-position
    RetrievedChunk,
    generate_answer,
    retrieve_passages,
    rewrite_query,
)
from app.rag_state import DEFAULT_SETTINGS  # pylint: disable=wrong-import-position

LOG_DIR = PROJECT_ROOT / "logs"

TEST_QUESTIONS = [
    "What is a frontline negotiator?",
    (
        "I am a EU diplomat in Mali, french citizen. Since the situation between Mali and "
        "France is currently complicated, should I avoid going to the ministry to negotiate, "
        "being french, and send someone else from my team to avoid tension during the negotiation."
    ),
]


def configure_logging() -> Tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = LOG_DIR / f"mode_option_regression_{timestamp}.log"
    logger = logging.getLogger("single-mode-regression")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_file


def resolve_vectorstore_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_vectorstore(path: Path, embedding_model: str) -> FAISS:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=0)


def summarize_chunk(chunk: RetrievedChunk, width: int = 360) -> Dict[str, Any]:
    document = chunk.document
    meta = document.metadata or {}
    snippet = " ".join(document.page_content.strip().split())
    return {
        "page": meta.get("page_number", "NA"),
        "section": meta.get("element_type", "Passage"),
        "score": round(chunk.score, 3),
        "query": chunk.query,
        "excerpt": shorten(snippet, width=width, placeholder="..."),
    }


def run_single_question(
    question: str,
    store: FAISS,
    llm: ChatOpenAI,
    *,
    per_query_k: int,
    final_k: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "question": question,
            "answer": "",
        "rewrite": {},
        "chunks": [],
        "debug": {},
            "error": "",
        }

    try:
        rewrite_result = rewrite_query(question, llm)
        chunks, debug_info = retrieve_passages(
            store,
            rewrite_result.rewritten_query,
            rewrite_result.intent,
            rewrite_result.concept_keywords,
            per_query_k=per_query_k,
            final_k=final_k,
        )
        answer = generate_answer(
            user_question=question,
            rewritten_query=rewrite_result.rewritten_query,
            context_chunks=chunks,
            llm=llm,
        )
    except Exception as exc:  # pragma: no cover - surface errors to logs
        result["error"] = str(exc)
        return result

    result["rewrite"] = {
        "rewritten_query": rewrite_result.rewritten_query,
        "intent": rewrite_result.intent,
        "concept_keywords": rewrite_result.concept_keywords,
    }
    result["chunks"] = [summarize_chunk(chunk) for chunk in chunks]
    result["debug"] = debug_info
    result["answer"] = answer
    return result


def log_run(logger: logging.Logger, run_data: Dict[str, Any]) -> None:
    logger.info("=" * 80)
    logger.info("Question: %s", run_data["question"])

    if run_data.get("error"):
        logger.error("Error: %s", run_data["error"])
        return

    rewrite = run_data.get("rewrite", {})
    logger.info("Rewritten query: %s", rewrite.get("rewritten_query", ""))
    logger.info("Intent: %s", rewrite.get("intent", ""))
    keywords = rewrite.get("concept_keywords") or []
    if keywords:
        logger.info("Concept keywords: %s", ", ".join(keywords))

    debug = run_data.get("debug") or {}
    queries = debug.get("queries") or []
    if queries:
        logger.info("Retrieval queries:")
        for query in queries:
            logger.info("  - %s", query)

    chunks = run_data.get("chunks") or []
    if chunks:
        logger.info("Top retrieved chunks:")
        for idx, chunk in enumerate(chunks, start=1):
            logger.info(
                "  %d) page %s (%s) | score %.3f | query: %s",
                idx,
                chunk["page"],
                chunk["section"],
                chunk["score"],
                chunk["query"],
            )
            logger.info("     %s", chunk["excerpt"])
    else:
        logger.info("No chunks were retrieved.")

    logger.info("Answer:\n%s", run_data.get("answer", "").strip() or "No answer returned.")


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Export it or add it to a .env file.")

    logger, log_file = configure_logging()
    logger.info("Starting single-mode regression run. Log file: %s", log_file)

    settings = DEFAULT_SETTINGS.copy()
    vectorstore_path = resolve_vectorstore_path(settings["vectorstore_path"])
    if not vectorstore_path.exists():
        raise FileNotFoundError(f"Vectorstore directory '{vectorstore_path}' was not found.")

    logger.info("Loading vectorstore from %s", vectorstore_path)
    store = load_vectorstore(vectorstore_path, settings["embedding_model"])
    llm = build_llm(settings["chat_model"])

    per_query_k = max(12, settings.get("top_k", 4) * 3)
    final_k = max(10, settings.get("top_k", 4) * 2)
    logger.info("Retrieval parameters: per_query_k=%s | final_k=%s", per_query_k, final_k)

    for question in TEST_QUESTIONS:
        run_data = run_single_question(
            question,
            store,
            llm,
            per_query_k=per_query_k,
            final_k=final_k,
        )
        log_run(logger, run_data)

    logger.info("Run complete. Detailed output written to %s", log_file)


if __name__ == "__main__":
    main()


"""
Utility script to run all four chatbot retrieval modes for control questions and
log their answers and artifacts for comparison.
"""

from __future__ import annotations

import json
import re
import logging
import os
import sys
from pathlib import Path
from textwrap import shorten
from typing import Any, Dict, List, Sequence, Tuple
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.rag_state import DEFAULT_SETTINGS

# Mode definitions are duplicated here so this script can run outside Streamlit.
MIN_SUBQUERIES = 3
MAX_SUBQUERIES = 5

MODE_DEFINITIONS = [
    {
        "id": "original_simple",
        "label": "Original query — simple search",
        "query_source": "original",
        "retrieval_variant": "simple",
        "requires_rewrite": False,
        "requires_concepts": False,
        "description": "Single retrieval pass using the raw user question.",
    },
    {
        "id": "rewritten_simple",
        "label": "Query rewriting — simple search",
        "query_source": "rewritten",
        "retrieval_variant": "simple",
        "requires_rewrite": True,
        "requires_concepts": True,
        "description": "Single retrieval pass using an LLM-refined query.",
    },
    {
        "id": "original_multi",
        "label": "Original query — multi-query retrieval",
        "query_source": "original",
        "retrieval_variant": "multi",
        "requires_rewrite": False,
        "requires_concepts": True,
        "description": "Generates subqueries from the original question.",
    },
    {
        "id": "rewritten_multi",
        "label": "Query rewriting — multi-query retrieval",
        "query_source": "rewritten",
        "retrieval_variant": "multi",
        "requires_rewrite": True,
        "requires_concepts": True,
        "description": "Generates subqueries from the LLM-refined query.",
    },
]

MODE_LOOKUP = {mode["id"]: mode for mode in MODE_DEFINITIONS}
MODE_SEQUENCE = [mode["id"] for mode in MODE_DEFINITIONS]

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
    logger = logging.getLogger("mode-regression")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Avoid duplicate handlers when re-running the script in the same interpreter.
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


def load_vectorstore(path: Path, embedding_model: str):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_chain(model_name: str) -> ChatOpenAI:
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
    return {
        "page": page,
        "element": element,
        "excerpt": excerpt,
        "content": doc.page_content.strip(),
    }


def build_prompt(template: str, context_text: str, question: str) -> str:
    template = template.strip()
    return (
        f"{template}\n\n"
        f"Context from the manual:\n{context_text}\n\n"
        f"User question:\n{question}\n\n"
        "Answer in clear English using only the provided context."
    )


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            return json.loads(match.group())
    raise ValueError("LLM response was not valid JSON.")


def _clean_keywords(candidates: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for item in candidates or []:
        text = str(item).strip()
        if text and text.lower() not in {kw.lower() for kw in cleaned}:
            cleaned.append(text)
    return cleaned


def rewrite_query_and_extract_concepts(question: str, llm: ChatOpenAI) -> Dict[str, Any]:
    response = llm.invoke(
        [
            (
                "system",
                "You rewrite humanitarian negotiation questions so they are manual-friendly "
                "and extract the negotiation concepts involved.",
            ),
            (
                "human",
                "Rewrite the user's question so it is precise, impersonal, and compatible with a "
                "field manual. Remove country names, personal anecdotes, and unrelated details. "
                "Identify 5-10 negotiation concept keywords behind the question "
                "(examples: trust, legitimacy, hostile counterpart, de-escalation, red lines). "
                "Return JSON with keys 'rewritten_query' (string) and "
                "'concept_keywords' (list of strings). "
                f"User question: ```{question}```",
            ),
        ]
    )
    data = _parse_json_object(response.content)
    rewritten_query = data.get("rewritten_query", "").strip()
    concept_keywords = _clean_keywords(data.get("concept_keywords", []))
    if not rewritten_query:
        raise ValueError("LLM did not return a rewritten query.")
    if not concept_keywords:
        raise ValueError("LLM did not return concept keywords.")
    return {"rewritten_query": rewritten_query, "concept_keywords": concept_keywords}


def extract_concept_keywords(question: str, llm: ChatOpenAI) -> List[str]:
    response = llm.invoke(
        [
            (
                "system",
                "You analyze humanitarian negotiation questions and extract negotiation concepts.",
            ),
            (
                "human",
                "List 5-10 concise negotiation concept keywords inspired by the user's question. "
                "Remove geographic references and anecdotes. "
                "Return JSON with the single key 'concept_keywords' (list of strings). "
                f"User question: ```{question}```",
            ),
        ]
    )
    data = _parse_json_object(response.content)
    concept_keywords = _clean_keywords(data.get("concept_keywords", []))
    if not concept_keywords:
        raise ValueError("No concept keywords were extracted.")
    return concept_keywords


def fallback_subqueries(base_question: str, concept_keywords: Sequence[str]) -> List[str]:
    base = base_question.strip() or "humanitarian negotiation best practices"
    unique_keywords = _clean_keywords(concept_keywords)
    subqueries: List[str] = []
    for keyword in unique_keywords:
        subqueries.append(f"How should a humanitarian negotiator address {keyword}?")
        if len(subqueries) >= MAX_SUBQUERIES:
            break
    if not subqueries:
        subqueries.append(base)
    while len(subqueries) < MIN_SUBQUERIES:
        subqueries.append(f"{base} ({len(subqueries) + 1})")
    return subqueries[:MAX_SUBQUERIES]


def ensure_original_question_subquery(
    subqueries: Sequence[str],
    original_question: str,
) -> List[str]:
    """Ensure the raw scenario is always interrogated during retrieval."""

    def _append_unique(target: List[str], candidate: str, seen: set[str]) -> None:
        text = (candidate or "").strip()
        if not text:
            return
        lowered = text.lower()
        if lowered in seen:
            return
        target.append(text)
        seen.add(lowered)

    ordered: List[str] = []
    seen_lower: set[str] = set()
    _append_unique(ordered, original_question, seen_lower)
    for item in subqueries or []:
        _append_unique(ordered, item, seen_lower)
        if len(ordered) >= MAX_SUBQUERIES:
            break
    return ordered[:MAX_SUBQUERIES]


def generate_subqueries(
    base_question: str,
    concept_keywords: Sequence[str],
    llm: ChatOpenAI,
) -> List[str]:
    keywords = ", ".join(concept_keywords)
    response = llm.invoke(
        [
            (
                "system",
                "You create precise retrieval subqueries grounded in humanitarian negotiation concepts.",
            ),
            (
                "human",
                "Using the provided negotiation concept keywords, craft 3-5 diverse retrieval "
                "subqueries that target the CCHN Field Manual. "
                "Each subquery must be standalone and focus on a single concept or angle. "
                "Return JSON with a single key 'subqueries' (list of strings). "
                f"Concept keywords: {keywords}\n"
                f"Original question context: ```{base_question}```",
            ),
        ]
    )
    try:
        data = _parse_json_object(response.content)
        subqueries = _clean_keywords(data.get("subqueries", []))
    except ValueError:
        subqueries = []

    if len(subqueries) < MIN_SUBQUERIES:
        subqueries = fallback_subqueries(base_question, concept_keywords)
    return subqueries[:MAX_SUBQUERIES]


def deduplicate_documents(documents: Sequence[Document]) -> List[Document]:
    seen = set()
    unique_documents: List[Document] = []
    for doc in documents:
        meta = doc.metadata or {}
        key = (
            meta.get("source")
            or meta.get("file_path")
            or meta.get("chunk_id")
            or meta.get("document_id")
            or ""
        )
        compound_key = (key, meta.get("page_number"), doc.page_content.strip())
        if compound_key in seen:
            continue
        seen.add(compound_key)
        unique_documents.append(doc)
    return unique_documents


def run_multi_query_search(
    subqueries: Sequence[str],
    store,
    top_k: int,
) -> List[Document]:
    aggregated: List[Document] = []
    for subquery in subqueries:
        aggregated.extend(retrieve_documents(subquery, store, top_k))
    deduped = deduplicate_documents(aggregated)
    return deduped[: top_k or len(deduped)]


def run_modes_for_question(
    question: str,
    settings: Dict[str, Any],
    store,
    llm: ChatOpenAI,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    deduped_selection: List[str] = []
    for mode_id in MODE_SEQUENCE:
        if mode_id in MODE_LOOKUP and mode_id not in deduped_selection:
            deduped_selection.append(mode_id)

    requires_rewrite = any(MODE_LOOKUP[mid]["requires_rewrite"] for mid in deduped_selection)
    requires_original_concepts = any(
        MODE_LOOKUP[mid]["query_source"] == "original" and MODE_LOOKUP[mid]["requires_concepts"]
        for mid in deduped_selection
    )

    rewriting_plan = None
    rewriting_error = ""
    if requires_rewrite:
        try:
            rewriting_plan = rewrite_query_and_extract_concepts(question, llm)
        except Exception as exc:
            rewriting_error = f"Query rewriting failed: {exc}"

    original_concepts: List[str] | None = None
    original_concepts_error = ""
    if requires_original_concepts:
        try:
            original_concepts = extract_concept_keywords(question, llm)
        except Exception as exc:
            original_concepts_error = f"Concept extraction failed: {exc}"

    subquery_cache: Dict[str, List[str]] = {}
    mode_results: List[Dict[str, Any]] = []

    for mode_id in deduped_selection:
        config = MODE_LOOKUP[mode_id]
        mode_result: Dict[str, Any] = {
            "mode_id": mode_id,
            "label": config["label"],
            "query_variant": config["query_source"],
            "retrieval_variant": config["retrieval_variant"],
            "query_used": "",
            "subqueries": [],
            "concept_keywords": [],
            "answer": "",
            "references": [],
            "retrieved_count": 0,
            "error": "",
        }

        if config["query_source"] == "rewritten":
            if rewriting_plan:
                mode_result["query_used"] = rewriting_plan["rewritten_query"]
                mode_result["concept_keywords"] = rewriting_plan["concept_keywords"]
            else:
                mode_result["error"] = rewriting_error or "Query rewriting did not return a result."
                mode_results.append(mode_result)
                continue
        else:
            mode_result["query_used"] = question
            if config["requires_concepts"]:
                if original_concepts:
                    mode_result["concept_keywords"] = original_concepts
                else:
                    mode_result["error"] = original_concepts_error or "Concept extraction returned no keywords."
                    mode_results.append(mode_result)
                    continue

        try:
            if config["retrieval_variant"] == "simple":
                documents = retrieve_documents(mode_result["query_used"], store, settings["top_k"])
                subqueries: List[str] = []
            else:
                cache_key = f"{config['query_source']}_subqueries"
                if cache_key not in subquery_cache:
                    keywords = mode_result["concept_keywords"]
                    if not keywords:
                        raise ValueError("No concept keywords available for multi-query search.")
                    generated_subqueries = generate_subqueries(
                        mode_result["query_used"],
                        keywords,
                        llm,
                    )
                    generated_subqueries = ensure_original_question_subquery(
                        generated_subqueries,
                        question,
                    )
                    subquery_cache[cache_key] = generated_subqueries
                subqueries = subquery_cache[cache_key]
                mode_result["subqueries"] = subqueries
                documents = run_multi_query_search(subqueries, store, settings["top_k"])
        except Exception as exc:
            mode_result["error"] = str(exc)
            mode_results.append(mode_result)
            continue

        mode_result["subqueries"] = subqueries if config["retrieval_variant"] == "multi" else []

        if not documents:
            mode_result["answer"] = (
                "No relevant manual passages were retrieved for this mode, so no grounded answer is available."
            )
            mode_results.append(mode_result)
            continue

        context_text = format_context_for_prompt(documents)
        prompt = build_prompt(settings["prompt_template"], context_text, mode_result["query_used"])

        try:
            response = llm.invoke(
                [
                    ("system", "You help users understand the CCHN Field Manual."),
                    ("human", prompt),
                ]
            )
            mode_result["answer"] = response.content
        except Exception as exc:
            mode_result["error"] = f"Failed to generate answer: {exc}"
            mode_results.append(mode_result)
            continue

        mode_result["references"] = [summarize_document(doc) for doc in documents]
        mode_result["retrieved_count"] = len(mode_result["references"])
        mode_results.append(mode_result)

    artifacts = {
        "rewritten_query": (rewriting_plan or {}).get("rewritten_query"),
        "rewritten_concepts": (rewriting_plan or {}).get("concept_keywords"),
        "original_concepts": original_concepts,
        "rewriting_error": rewriting_error,
        "concept_error": original_concepts_error,
    }
    return mode_results, artifacts


def log_artifacts(logger: logging.Logger, artifacts: Dict[str, Any]) -> None:
    if artifacts.get("rewritten_query"):
        logger.info("Rewritten query: %s", artifacts["rewritten_query"])
    if artifacts.get("rewritten_concepts"):
        logger.info("Rewritten concepts: %s", ", ".join(artifacts["rewritten_concepts"]))
    if artifacts.get("original_concepts"):
        logger.info("Original concepts: %s", ", ".join(artifacts["original_concepts"]))
    if artifacts.get("rewriting_error"):
        logger.warning("Rewriting error: %s", artifacts["rewriting_error"])
    if artifacts.get("concept_error"):
        logger.warning("Concept extraction error: %s", artifacts["concept_error"])


def log_mode_result(
    logger: logging.Logger,
    question: str,
    result: Dict[str, Any],
) -> None:
    logger.info("--- Mode: %s ---", result.get("label", result.get("mode_id")))
    logger.info("Question: %s", question)
    logger.info("Query variant: %s | Retrieval: %s", result["query_variant"], result["retrieval_variant"])
    if result.get("query_used"):
        logger.info("Query used: %s", result["query_used"])
    if result.get("concept_keywords"):
        logger.info("Concept keywords: %s", ", ".join(result["concept_keywords"]))
    if result.get("subqueries"):
        for idx, subquery in enumerate(result["subqueries"], start=1):
            logger.info("Subquery %d: %s", idx, subquery)
    if result.get("error"):
        logger.error("Error: %s", result["error"])
        return
    logger.info("Answer: %s", result.get("answer", "").strip() or "No answer returned.")
    if result.get("references"):
        for ref in result["references"]:
            logger.info(
                "Reference page %s (%s): %s",
                ref.get("page"),
                ref.get("element"),
                ref.get("excerpt"),
            )
    else:
        logger.info("No passages retrieved for this mode.")


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Export it or add it to a .env file.")

    logger, log_file = configure_logging()
    logger.info("Starting mode regression run. Log file: %s", log_file)

    settings = DEFAULT_SETTINGS.copy()
    vectorstore_path = resolve_vectorstore_path(settings["vectorstore_path"])
    if not vectorstore_path.exists():
        raise FileNotFoundError(f"Vectorstore directory '{vectorstore_path}' was not found.")

    logger.info("Loading vectorstore from %s", vectorstore_path)
    store = load_vectorstore(vectorstore_path, settings["embedding_model"])
    llm = build_chain(settings["chat_model"])

    for question in TEST_QUESTIONS:
        logger.info("=" * 80)
        logger.info("Question: %s", question)
        mode_results, artifacts = run_modes_for_question(question, settings, store, llm)
        log_artifacts(logger, artifacts)
        for result in mode_results:
            log_mode_result(logger, question, result)

    logger.info("Run complete. Detailed output written to %s", log_file)


if __name__ == "__main__":
    main()



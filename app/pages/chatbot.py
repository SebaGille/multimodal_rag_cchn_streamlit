"""
Main chatbot page for the CCHN RAG prototype.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from textwrap import shorten
from typing import Any, Dict, List, Sequence

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
DEFAULT_MODE_SELECTION = [MODE_DEFINITIONS[0]["id"]]


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
    return {"page": page, "element": element, "excerpt": excerpt, "content": doc.page_content.strip()}


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
    cleaned = []
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
    try:
        data = _parse_json_object(response.content)
    except ValueError as exc:
        raise ValueError(f"Failed to parse rewrite response: {exc}") from exc

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
    try:
        data = _parse_json_object(response.content)
    except ValueError as exc:
        raise ValueError(f"Failed to parse concept extraction response: {exc}") from exc

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


def run_selected_modes(
    question: str,
    settings: Dict[str, Any],
    selected_mode_ids: Sequence[str],
) -> None:
    path = Path(settings["vectorstore_path"]).expanduser()
    deduped_selection: List[str] = []
    for mode_id in selected_mode_ids:
        if mode_id in MODE_LOOKUP and mode_id not in deduped_selection:
            deduped_selection.append(mode_id)
    update_run_state(
        question=question,
        selected_modes=deduped_selection,
        mode_results=[],
        retrieved_count=0,
        error="",
    )

    if not deduped_selection:
        st.warning("Please select at least one search mode.")
        return

    if not path.exists():
        error_msg = f"Vectorstore directory '{path}' was not found."
        update_run_state(error=error_msg)
        st.error(error_msg)
        return

    try:
        with st.spinner("Loading vectorstore..."):
            store = load_vectorstore(str(path), settings["embedding_model"])
    except Exception as exc:
        error_msg = f"Failed to load vectorstore: {exc}"
        update_run_state(error=error_msg)
        st.error(error_msg)
        return

    llm = build_chain(settings["chat_model"])
    requires_rewrite = any(MODE_LOOKUP[mid]["requires_rewrite"] for mid in deduped_selection)
    requires_original_concepts = any(
        MODE_LOOKUP[mid]["query_source"] == "original" and MODE_LOOKUP[mid]["requires_concepts"]
        for mid in deduped_selection
    )

    rewriting_plan = None
    rewriting_error = ""
    if requires_rewrite:
        try:
            with st.spinner("Rewriting question and extracting concepts..."):
                rewriting_plan = rewrite_query_and_extract_concepts(question, llm)
        except Exception as exc:
            rewriting_error = f"Query rewriting failed: {exc}"

    original_concepts: List[str] | None = None
    original_concepts_error = ""
    if requires_original_concepts:
        try:
            with st.spinner("Extracting negotiation concepts..."):
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
                    mode_result["error"] = original_concepts_error or "Concept extraction did not return any keywords."
                    mode_results.append(mode_result)
                    continue

        try:
            with st.spinner(f"Running {config['label']}"):
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
                "No relevant manual passages were retrieved for this mode, so I cannot provide a grounded answer."
            )
            mode_result["references"] = []
            mode_results.append(mode_result)
            continue

        context_text = format_context_for_prompt(documents)
        prompt = build_prompt(settings["prompt_template"], context_text, mode_result["query_used"])

        try:
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
            mode_result["answer"] = response.content
        except Exception as exc:
            mode_result["error"] = f"Failed to generate answer: {exc}"
            mode_results.append(mode_result)
            continue

        mode_result["references"] = [summarize_document(doc) for doc in documents]
        mode_result["retrieved_count"] = len(mode_result["references"])
        mode_results.append(mode_result)

    total_retrieved = sum(result.get("retrieved_count", 0) for result in mode_results)
    update_run_state(
        selected_modes=deduped_selection,
        mode_results=mode_results,
        retrieved_count=total_retrieved,
        last_run_settings={
            "vectorstore_path": str(path),
            "embedding_model": settings["embedding_model"],
            "chat_model": settings["chat_model"],
            "top_k": settings["top_k"],
        },
        planning_artifacts={
            "rewritten_query": (rewriting_plan or {}).get("rewritten_query"),
            "rewritten_concepts": (rewriting_plan or {}).get("concept_keywords"),
            "original_concepts": original_concepts,
        },
    )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Chatbot prototype", page_icon="💬")

    settings = get_rag_settings()
    run_state = get_run_state()

    if "chatbot_question_input" not in st.session_state:
        st.session_state["chatbot_question_input"] = run_state["question"]
    if "chatbot_mode_selection" not in st.session_state:
        previous_selection = run_state.get("selected_modes") or []
        st.session_state["chatbot_mode_selection"] = previous_selection or DEFAULT_MODE_SELECTION

    st.title("CCHN Manual Assistant")
    st.write(
        "Ask a question about the CCHN Field Manual and this assistant will "
        "run multiple retrieval strategies so you can compare their answers."
    )
    st.info("Need to tweak the engine? Head to the Engine page in the sidebar.")

    question = st.text_input(
        "Your question",
        placeholder="For example: What is a frontline negotiator?",
        key="chatbot_question_input",
    )
    mode_selection = st.pills(
        "Choose search modes to run",
        options=[mode["id"] for mode in MODE_DEFINITIONS],
        selection_mode="multi",
        default=st.session_state["chatbot_mode_selection"],
        format_func=lambda mode_id: MODE_LOOKUP[mode_id]["label"],
        help="Compare original vs. rewritten questions and single vs. multi-query retrieval.",
    )
    st.session_state["chatbot_mode_selection"] = mode_selection

    submit = st.button("Run selected modes", type="primary")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.error("Set OPENAI_API_KEY in your environment before using the app.")
        st.stop()

    if submit:
        trimmed_question = question.strip()
        if not trimmed_question:
            st.warning("Please enter a question before requesting an answer.")
        elif not mode_selection:
            st.warning("Select at least one search mode to continue.")
        else:
            run_selected_modes(trimmed_question, settings, mode_selection)
            run_state = get_run_state()

    if run_state.get("error"):
        st.error(run_state["error"])

    mode_results = run_state.get("mode_results", [])
    if mode_results:
        st.subheader("Mode comparison")
        for result in mode_results:
            st.divider()
            st.markdown(f"### {result.get('label', 'Search mode')}")
            st.markdown(
                f"*Query variant:* `{result.get('query_variant', '')}` &nbsp; | "
                f"*Retrieval:* `{result.get('retrieval_variant', '')}`"
            )

            st.markdown("**Query used**")
            st.code(result.get("query_used", ""))

            concept_keywords = result.get("concept_keywords") or []
            if concept_keywords:
                st.markdown("**Negotiation concepts detected**")
                st.write(", ".join(concept_keywords))

            subqueries = result.get("subqueries") or []
            if subqueries:
                st.markdown("**Subqueries executed**")
                for subquery in subqueries:
                    st.write(f"- {subquery}")

            if result.get("error"):
                st.error(result["error"])
                continue

            st.markdown("**Answer**")
            st.write(result.get("answer", "").strip() or "No answer was generated.")

            references = result.get("references", [])
            if references:
                st.markdown("**Retrieved passages**")
                for ref in references:
                    st.markdown(f"**Page {ref['page']} | {ref['element']}**")
                    st.write(ref.get("content") or ref.get("excerpt", ""))
            else:
                st.info("No passages retrieved for this mode.")
    else:
        st.info("Ask a question and choose one or more modes to see comparative results.")


if __name__ == "__main__":
    main()


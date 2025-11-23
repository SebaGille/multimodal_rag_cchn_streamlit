from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

INTENT_CODES = {
    "definition",
    "context_dilemma",
    "tactics",
    "preparation",
    "human_elements",
    "mandate_red_lines",
}
DEFAULT_INTENT = "context_dilemma"
MIN_QUERY_COUNT = 3
MAX_QUERY_COUNT = 6


@dataclass
class RewriteResult:
    rewritten_query: str
    concept_keywords: List[str]
    intent: str


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    query: str


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            return json.loads(match.group())
    raise ValueError("LLM response was not valid JSON.")


def _dedupe_list(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for entry in items:
        text = (entry or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        ordered.append(text)
        seen.add(lowered)
    return ordered


def rewrite_query(user_question: str, llm: ChatOpenAI) -> RewriteResult:
    """LLM-powered query rewriting with concept extraction and intent tagging."""

    if not user_question.strip():
        raise ValueError("Question is empty.")

    response = llm.invoke(
        [
            (
                "system",
                "You are a negotiation planning copilot trained on the CCHN Field Manual. "
                "You rewrite user questions into manual-friendly search queries, extract "
                "concept keywords, and label the negotiation intent. "
                "Never invent context that is not present in the user's scenario.",
            ),
            (
                "human",
                "You will receive a humanitarian negotiation scenario. Follow these rules:\n"
                "1. Preserve the scenario details so the final assistant answer can reuse them, "
                "but rewrite the SEARCH QUERY so it uses neutral, manual-aligned negotiation "
                "language. Remove personal identifiers, country names, and sensitive details. "
                "Focus on the underlying negotiation dilemma.\n"
                "2. Return 5-10 short concept keywords (2-4 words each) that capture the "
                "core negotiation themes. Examples: reputational risk, delegation, "
                "hostile counterpart, trust building, negotiation mandate, red lines.\n"
                "3. Detect the user's intent. Allowed codes: definition, context_dilemma, "
                "tactics, preparation, human_elements, mandate_red_lines. If the scenario "
                "describes a personal situation or dilemma, choose context_dilemma. "
                "If it is about emotions or perception, choose human_elements. "
                "If it asks what is allowed or limits, choose mandate_red_lines. "
                "If it focuses on planning or analysis, choose preparation. "
                "If it asks for negotiation moves, choose tactics. "
                "4. Respond with JSON containing keys: "
                "'rewritten_query' (string), 'concept_keywords' (list of strings), "
                "'intent' (one of the allowed codes).\n\n"
                f"User scenario: ```{user_question}```",
            ),
        ]
    )

    data = _parse_json_object(response.content)

    rewritten_query = data.get("rewritten_query", "").strip()
    concept_keywords = _dedupe_list(data.get("concept_keywords", []))
    intent = data.get("intent", DEFAULT_INTENT).strip().lower()

    if not rewritten_query:
        raise ValueError("Rewriting step did not produce a query.")
    if len(concept_keywords) < 3:
        raise ValueError("Rewriting step returned too few concept keywords.")
    if intent not in INTENT_CODES:
        intent = DEFAULT_INTENT

    return RewriteResult(
        rewritten_query=rewritten_query,
        concept_keywords=concept_keywords,
        intent=intent,
    )


def _intent_seed_queries(intent: str) -> List[str]:
    return {
        "definition": [
            "definition of {keyword}",
            "role and tasks of frontline negotiators in humanitarian contexts",
            "definition of frontline humanitarian negotiator",
        ],
        "context_dilemma": [
            "reputational risk in humanitarian negotiation",
            "delegation and choice of negotiator profile",
            "dealing with a hostile counterpart",
            "trust building in strained relations",
            "creating a conducive negotiation environment",
        ],
        "human_elements": [
            "managing emotions and hostility in humanitarian negotiation",
            "trust building in strained relations",
            "relationships with volatile counterparts",
        ],
        "mandate_red_lines": [
            "negotiation mandate and red lines",
            "institutional limits in humanitarian negotiation",
            "authorisation boundaries for negotiators",
        ],
        "tactics": [
            "tactical plan for humanitarian negotiation",
            "sequencing negotiation moves with {keyword}",
            "leveraging networks to advance negotiation objectives",
        ],
        "preparation": [
            "context analysis and network mapping in humanitarian negotiation",
            "preparation steps before frontline humanitarian engagement",
            "stakeholder analysis for {keyword}",
        ],
    }.get(intent, [
        "humanitarian negotiation best practices",
        "stabilising strained negotiation relationships",
    ])


def _expand_templates(templates: Sequence[str], concept_keywords: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for template in templates:
        if "{keyword}" in template:
            for keyword in concept_keywords:
                expanded.append(template.format(keyword=keyword))
        else:
            expanded.append(template)
    return expanded


def build_retrieval_queries(
    rewritten_query: str,
    intent: str,
    concept_keywords: Sequence[str],
) -> List[str]:
    base_queries = [rewritten_query]
    intent_queries = _expand_templates(_intent_seed_queries(intent), concept_keywords or [])

    supplemental: List[str] = []
    for keyword in concept_keywords:
        supplemental.append(f"{keyword} in humanitarian negotiation")
        if len(supplemental) >= MAX_QUERY_COUNT:
            break

    combined = _dedupe_list(base_queries + intent_queries + supplemental)

    while len(combined) < MIN_QUERY_COUNT and concept_keywords:
        combined.append(f"How should a negotiator handle {concept_keywords[len(combined) % len(concept_keywords)]}?")

    return combined[:MAX_QUERY_COUNT]


def retrieve_passages(
    store: FAISS,
    rewritten_query: str,
    intent: str,
    concept_keywords: Sequence[str],
    *,
    per_query_k: int = 15,
    final_k: int = 10,
) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
    queries = build_retrieval_queries(rewritten_query, intent, concept_keywords)
    scored_entries: Dict[Tuple[str, Any, str], RetrievedChunk] = {}

    for query in queries:
        if hasattr(store, "similarity_search_with_relevance_scores"):
            results = store.similarity_search_with_relevance_scores(query, k=per_query_k)
        else:  # pragma: no cover - compatibility fallback
            docs = store.similarity_search(query, k=per_query_k)
            results = [(doc, 0.0) for doc in docs]
        for doc, score in results:
            meta = doc.metadata or {}
            key = (
                str(meta.get("source")),
                meta.get("page_number"),
                doc.page_content.strip(),
            )
            candidate = RetrievedChunk(document=doc, score=float(score), query=query)
            existing = scored_entries.get(key)
            if not existing or candidate.score > existing.score:
                scored_entries[key] = candidate

    ranked = sorted(scored_entries.values(), key=lambda item: item.score, reverse=True)
    top_chunks = ranked[:final_k]

    debug_chunks = []
    for chunk in top_chunks:
        meta = chunk.document.metadata or {}
        debug_chunks.append(
            {
                "page": meta.get("page_number", "NA"),
                "section": meta.get("element_type", "Passage"),
                "query": chunk.query,
                "score": round(chunk.score, 3),
            }
        )

    debug_info = {
        "queries": queries,
        "chunks": debug_chunks,
    }
    return top_chunks, debug_info


def format_chunks_for_prompt(chunks: Sequence[RetrievedChunk]) -> str:
    blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.document.metadata or {}
        page = meta.get("page_number", "NA")
        section = meta.get("element_type", "Passage")
        text = chunk.document.page_content.strip()
        blocks.append(f"[Chunk {idx}] (page {page}, {section})\n{text}")
    return "\n\n".join(blocks)


def generate_answer(
    user_question: str,
    rewritten_query: str,
    context_chunks: Sequence[RetrievedChunk],
    llm: ChatOpenAI,
) -> str:
    if not context_chunks:
        return (
            "The manual does not cover this specific topic. It focuses on humanitarian "
            "negotiation practices, so there is no grounded guidance for this scenario."
        )

    context_text = format_chunks_for_prompt(context_chunks)

    prompt = (
        "Original scenario:\n"
        f"{user_question.strip()}\n\n"
        "Rewritten negotiation problem:\n"
        f"{rewritten_query.strip()}\n\n"
        "Excerpts from the CCHN Field Manual:\n"
        f"{context_text}\n\n"
        "Write a grounded answer with this structure:\n"
        "1. Problem framing – restate the generic negotiation challenge in a neutral tone.\n"
        "2. Manual guidance – summarise the key principles, tools, or sections from the manual that apply.\n"
        "3. Practical considerations – translate those principles back to the user's scenario using conditional language (\"you may\", \"one option could be\").\n"
        "If the manual is indirect, explicitly note that it suggests principles rather than definitive rules.\n"
        "Reference tools or sections when mentioned in the excerpts. Keep the tone careful and avoid prescribing organisational policy.\n"
        "Close with one sentence reminding the user that this guidance comes from the CCHN Field Manual."
    )

    response = llm.invoke(
        [
            (
                "system",
                "You are the CCHN Negotiation Chatbot. "
                "Base every answer strictly on the provided manual excerpts. "
                "Do not speculate beyond the text. Use careful, grounded language.",
            ),
            (
                "human",
                prompt,
            ),
        ]
    )
    return response.content.strip()



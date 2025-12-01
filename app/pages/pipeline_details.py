"""
Product-facing overview of the negotiation chatbot pipeline and its automation hooks.
"""

from __future__ import annotations

from textwrap import dedent

import streamlit as st

from chat_pipeline import INTENT_CODES, MAX_QUERY_COUNT, MIN_QUERY_COUNT
from eval.judge import RUBRIC_LABELS, RUBRIC_PROMPT
from eval.question_generator import GENERATOR_PROMPT
from rag_state import DEFAULT_SETTINGS

REWRITE_PROMPT_EXCERPT = dedent(
    """
    System message (chat_pipeline.rewrite_query):
    You are a negotiation planning copilot trained on the CCHN Field Manual. You rewrite user questions into manual-friendly search queries, extract concept keywords, and label the negotiation intent. Never invent context that is not present in the user's scenario.

    Human message highlights:
    1. Preserve scenario details but rewrite the search query with neutral, manual-aligned language.
    2. Return 5-10 short concept keywords covering the negotiation themes.
    3. Assign one allowed intent code (definition, context_dilemma, tactics, preparation, human_elements, mandate_red_lines).
    4. Respond as JSON with keys rewritten_query, concept_keywords, intent. The user scenario is passed verbatim in triple backticks.
    """
).strip()

ANSWER_PROMPT_EXCERPT = dedent(
    """
    System message (chat_pipeline.generate_answer):
    You are the CCHN Negotiation Chatbot. Base every answer strictly on the provided manual excerpts. Do not speculate beyond the text. Use careful, grounded language.

    Human message skeleton:
    - Provides the original scenario, rewritten negotiation problem (for grounding only), and formatted manual chunks.
    - Requests one blended paragraph-style answer that first summarises the manual guidance, then immediately translates it back to the user's situation with conditional language ("you may", "one option could be").
    - Forbids numbered headings or explicit part labels; everything lives inside a single section.
    - Requires citations of tools/sections when available plus a closing reminder that the guidance stems from the CCHN Field Manual.
    """
).strip()


def _render_parameters(items: list[tuple[str, str]]) -> None:
    for label, value in items:
        st.markdown(f"- **{label}:** {value}")


def _render_prompts(blocks: list[tuple[str, str]]) -> None:
    if not blocks:
        return
    st.markdown("**LLM prompt snapshots**")
    for title, text in blocks:
        st.caption(title)
        st.code(text, language="markdown")


def _render_step(
    *,
    title: str,
    description: str,
    parameters: list[tuple[str, str]],
    prompt_blocks: list[tuple[str, str]] | None = None,
    notes: str | None = None,
    expanded: bool = False,
) -> None:
    with st.expander(title, expanded=expanded):
        st.markdown(description)
        if parameters:
            st.markdown("**Key parameters**")
            _render_parameters(parameters)
        if prompt_blocks:
            _render_prompts(prompt_blocks)
        if notes:
            st.info(notes)


def main() -> None:
    st.set_page_config(page_title="Pipeline details", page_icon="🧩", layout="wide")
    st.title("Pipeline details")
    st.write(
        "This page summarizes every moving piece required for the CCHN negotiation chatbot and "
        "its evaluation stack. Each step highlights the owner, parameters, and (when relevant) the exact prompts that reach an LLM."
    )

    st.markdown(
        "- **Sources -> Vector store**: Parse the Field Manual PDF, chunk it, and embed it into FAISS.\n"
        "- **Runtime -> Chatbot**: Load cached resources, rewrite the user question, retrieve chunks, and draft a grounded answer.\n"
        "- **Agentic validation**: The new Agentic AI assistant page exposes a bounded plan → rewrite → retrieve → critique loop for quick experimentation.\n"
        "- **Quality loop -> Evaluation**: Auto-generate scenarios, run the chatbot end-to-end, judge answers, and visualize the logs."
    )

    default_settings = DEFAULT_SETTINGS.copy()
    fan_out_description = (
        f"per_query_k = max(12, top_k x 3) = {max(12, default_settings['top_k'] * 3)}, "
        f"final_k = max(10, top_k x 2) = {max(10, default_settings['top_k'] * 2)}"
    )

    _render_step(
        title="1. Corpus ingestion & vectorization",
        description=(
            "Transform the CCHN Field Manual into structured LangChain documents and a FAISS index "
            "that downstream components can query."
        ),
        parameters=[
            ("Owners", "`ingestion/parse_pdf.py` -> `ingestion/build_vectorstore.py`"),
            ("Source PDF", "`data/raw/CCHN-Field-Manual-EN.pdf` parsed via Unstructured"),
            (
                "JSON cache",
                "`data/full_manual_docs.json` (page content plus metadata: page number, element type, image references)",
            ),
            ("Chunking strategy", "Recursive splitter with 800 character chunks and 150 character overlap"),
            ("Embedding model", "`text-embedding-3-small`"),
            ("Vector store", "`vectorstores/full_manual_faiss` saved with LangChain FAISS"),
            (
                "Regeneration command",
                "`python ingestion/build_vectorstore.py --docs data/full_manual_docs.json --output vectorstores/full_manual_faiss`",
            ),
        ],
        notes="These artifacts must exist before the app starts; the chatbot refuses to run if the vector store path is missing.",
        expanded=True,
    )

    _render_step(
        title="2. Runtime resources & controls",
        description="Load long-lived resources once and expose knobs for operations teams inside `rag_state.py`.",
        parameters=[
            ("Default vector store path", f"`{default_settings['vectorstore_path']}`"),
            ("Embedding model", f"`{default_settings['embedding_model']}`"),
            ("Chat model", f"`{default_settings['chat_model']}` (temperature fixed at 0)"),
            ("Retrieval top_k", f"{default_settings['top_k']} chunks forwarded to the answer prompt"),
            ("Derived fan-out", fan_out_description),
            ("Caching", "`@st.cache_resource` keeps FAISS and ChatOpenAI singletons per session"),
            ("Env / guardrails", "`OPENAI_API_KEY` is mandatory; missing key or store aborts the run early"),
        ],
        notes="Operations can override any field in `st.session_state['rag_settings']` before calling `get_chat_resources()`.",
        expanded=True,
    )

    _render_step(
        title="3. Question rewriting & intent tagging",
        description="Normalize frontline questions, harvest keywords, and capture a negotiation intent code before retrieving context.",
        parameters=[
            ("Owner", "`chat_pipeline.rewrite_query`"),
            ("Model", "Inherited `ChatOpenAI` instance (`gpt-4o-mini`, temperature 0)"),
            ("Intent codes", ", ".join(sorted(INTENT_CODES))),
            ("Concept keywords", "Prompt asks for 5-10 short phrases; pipeline enforces >=3 unique entries"),
            ("Output contract", "`RewriteResult` (rewritten_query, concept_keywords, intent)"),
            (
                "Query budget",
                f"At least {MIN_QUERY_COUNT} and at most {MAX_QUERY_COUNT} distinct sub-queries feed retrieval.",
            ),
        ],
        prompt_blocks=[("Rewrite prompt", REWRITE_PROMPT_EXCERPT)],
        notes="The JSON parser is defensive: it strips whitespace, tolerates stray text around JSON, and falls back to sane defaults if the intent is invalid.",
    )

    _render_step(
        title="4. Intent-aware retrieval fan-out",
        description="Blend rewritten queries, intent seed templates, and keyword expansions to fetch high-relevance manual chunks.",
        parameters=[
            ("Owner", "`chat_pipeline.retrieve_passages`"),
            ("Query templates", "Base rewrite + intent templates + `<keyword> in humanitarian negotiation` supplements"),
            ("Deduplication", "Chunks keyed by (source, page, text) keep the highest relevance score"),
            ("Per-query fan-out", "`similarity_search_with_relevance_scores` over FAISS with the derived `per_query_k`"),
            ("Final selection", f"Top scoring chunks are trimmed to `{fan_out_description.split(', ')[1]}`"),
            ("Debug payload", "Stored in session via `run_cchn_chatbot` to power the in-app Advanced debug panel"),
        ],
        notes="Metadata preserved in the ingestion phase (page_number, element_type) is surfaced back inside chunk labels.",
    )

    _render_step(
        title="5. Grounded response drafting",
        description="Compose a careful answer using only the retrieved manual evidence and merge guidance with scenario-aware considerations into one section.",
        parameters=[
            ("Owner", "`chat_pipeline.generate_answer`"),
            ("Context formatter", "`format_chunks_for_prompt` labels each chunk with page and manual section"),
            ("Structure", "Single blended narrative: manual guidance followed by conditional scenario translation"),
            ("Safety fallback", "If no chunks survive retrieval, return a stock message that the manual has no coverage"),
        ],
        prompt_blocks=[("Answer prompt", ANSWER_PROMPT_EXCERPT)],
        notes="The same zero-temperature `ChatOpenAI` instance is reused, ensuring deterministic tone between rewriting and answering.",
    )

    _render_step(
        title="6. Streamlit surface & observability",
        description="Expose the pipeline via the Streamlit navigation so PMs and SMEs can inspect every step.",
        parameters=[
            ("Entry point", "`app/streamlit_app.py` with navigation across Ask one question, Agentic AI assistant, LLM-as-a-Judge, Evaluation analytics, Project insights and learnings, and Pipeline details"),
            ("Session state keys", "`rag_settings`, `rag_run`, chat history, debug payloads"),
            (
                "User workflow",
                "AI Chatbot Assistant relies on `st.chat_input` for multi-turn chats; Ask one question keeps the form-driven flow with spinner + history.",
            ),
            ("Debug panel", "`pages/ask_one_question.py` exposes the Advanced debug expander with rewrite/intent/queries/chunks"),
            ("Evaluation analytics", "`pages/evaluation_analytics.py` aggregates logged rubric scores and red flags from prior runs"),
            ("Insights hub", "`pages/project_insights_and_learnings.py` captures qualitative lessons from experiments"),
        ],
        notes="Because every resource loader defers to Streamlit caching, the UI stays snappy even while re-running the script on every interaction.",
    )

    rubric_dimension_text = ", ".join(f"{key} = {label}" for key, label in RUBRIC_LABELS.items())

    _render_step(
        title="7. Automated evaluation & analytics loop",
        description="Stress-test the chatbot by autogenerating realistic scenarios, running the full pipeline, judging answers, and logging everything for analytics.",
        parameters=[
            ("Owner", "`pages/chatbot_auto_evaluation.py` orchestrates the loop"),
            ("Question generation", "20 frontline dilemmas per batch, requested with `response_format={'type': 'json_object'}`"),
            ("Judge rubric", f"Integers 0-5 on {rubric_dimension_text}"),
            ("Red-flag policy", "Any violation of humanitarian principles or unsafe advice forces low scores and `red_flag=True`"),
            ("Logging", "Each run writes `.txt` plus `.json` artifacts under `eval_logs/eval_YYYYMMDD_HHMMSS.*`"),
            ("Analytics", "`pages/evaluation_analytics.py` loads the JSON logs to surface weak themes, red flags, and trends"),
        ],
        prompt_blocks=[
            ("Scenario generator prompt", dedent(GENERATOR_PROMPT).strip()),
            ("LLM judge prompt", dedent(RUBRIC_PROMPT).strip()),
        ],
        notes="Both generator and judge reuse zero-temperature `ChatOpenAI` instances; operators can override the model name via `rag_settings['judge_model']`.",
    )

    _render_step(
        title="8. Agentic AI assistant prototype",
        description="Demonstrate a lightweight agent loop that plans a handful of steps, optionally rewrites the query, critiques the draft, and (if needed) reruns retrieval once.",
        parameters=[
            ("Owner", "`pages/agentic_ai_assistant.py`"),
            (
                "Planning heuristics",
                "Length, multiple-question, and context keyword heuristics (plus a Force rewrite toggle) decide whether the LLM rewrite step runs.",
            ),
            (
                "Rewrite modes",
                "Uses `chat_pipeline.rewrite_query` when triggered; otherwise generates deterministic keywords from the raw question.",
            ),
            (
                "Critique loop",
                "A QA prompt returns `ok` vs `needs_more` plus follow-up keywords. Only one follow-up pass is allowed.",
            ),
            (
                "Follow-up retrieval",
                "Extends concept keywords with critique hints, re-calls `retrieve_passages`, merges chunks, and regenerates the answer if new evidence appears.",
            ),
            (
                "Traceability",
                "Stores the last three runs in session state and renders plan signals, chunk tables, critique notes, and a JSON trace for audits.",
            ),
        ],
        notes="This surface keeps autonomy bounded (max two retrieval rounds) while showcasing how agentic behaviors could evolve inside the existing RAG stack.",
    )

    st.caption("Updated automatically from source constants - use this page as the single reference for product reviews and onboarding.")


if __name__ == "__main__":
    main()



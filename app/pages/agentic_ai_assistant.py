"""
Mini agentic Streamlit surface that shows a bounded plan → rewrite → retrieval → critique loop.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple

import streamlit as st
from dotenv import load_dotenv

from chat_pipeline import (
    DEFAULT_INTENT,
    RetrievedChunk,
    RewriteResult,
    generate_answer,
    retrieve_passages,
    rewrite_query,
)
from services.chatbot_runner import ChatbotResources, get_chat_resources


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

AGENT_RUN_KEY = "agentic_agent_runs"


@dataclass
class AgentPlan:
    needs_rewrite: bool
    rationale: str
    signals: List[str]
    steps: List[str]


@dataclass
class CritiqueResult:
    verdict: Literal["ok", "needs_more", "error"]
    notes: str
    followup_keywords: List[str]


@dataclass
class AgenticRun:
    question: str
    plan: AgentPlan
    rewrite_mode: Literal["llm", "direct"]
    rewrite_notes: str
    rewrite_result: RewriteResult
    initial_chunks: List[RetrievedChunk]
    initial_debug: Dict[str, Any]
    draft_answer: str
    critique: CritiqueResult
    followup_chunks: List[RetrievedChunk] | None
    followup_debug: Dict[str, Any] | None
    final_answer: str

    def summary(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "question": self.question,
            "plan": {
                "needs_rewrite": self.plan.needs_rewrite,
                "rationale": self.plan.rationale,
                "signals": self.plan.signals,
                "steps": self.plan.steps,
            },
            "rewrite": {
                "mode": self.rewrite_mode,
                "rewritten_query": self.rewrite_result.rewritten_query,
                "intent": self.rewrite_result.intent,
                "concept_keywords": self.rewrite_result.concept_keywords,
                "notes": self.rewrite_notes,
            },
            "critique": {
                "verdict": self.critique.verdict,
                "notes": self.critique.notes,
                "followup_keywords": self.critique.followup_keywords,
            },
            "initial_retrieval": self.initial_debug,
            "followup_retrieval": self.followup_debug,
            "final_answer_length": len(self.final_answer),
            "was_refined": self.final_answer != self.draft_answer,
        }
        return payload


def _ensure_runs_state() -> None:
    st.session_state.setdefault(AGENT_RUN_KEY, [])


def _get_runs() -> List[AgenticRun]:
    _ensure_runs_state()
    return st.session_state[AGENT_RUN_KEY]


def _store_run(run: AgenticRun) -> None:
    runs = _get_runs()
    runs.append(run)
    st.session_state[AGENT_RUN_KEY] = runs[-3:]


def _latest_run() -> AgenticRun | None:
    runs = _get_runs()
    return runs[-1] if runs else None


def _dedupe_strings(items: Sequence[str]) -> List[str]:
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


def _extract_keywords(question: str, limit: int = 5) -> List[str]:
    tokens = re.findall(r"[A-Za-z]{4,}", question)
    keywords = []
    seen: set[str] = set()
    for token in tokens:
        lower = token.lower()
        if lower in seen:
            continue
        seen.add(lower)
        keywords.append(token.lower())
        if len(keywords) >= limit:
            break
    return keywords or ["humanitarian negotiation"]


def _decide_plan(question: str, *, force_rewrite: bool) -> AgentPlan:
    words = question.split()
    lower = question.lower()
    signals: List[str] = []

    if len(words) >= 40:
        signals.append("Long scenario (>40 words).")
    if lower.count("?") > 1:
        signals.append("Multiple explicit questions.")
    if any(token in lower for token in ("because", "context", "situation", "backstory")):
        signals.append("Contains detailed context cues.")
    if "i " in lower or "we " in lower:
        signals.append("Personal perspective detected.")
    if force_rewrite:
        signals.append("Manual override: rewrite forced in UI.")

    needs_rewrite = bool(signals) or force_rewrite
    rationale = (
        "Rewrite to clean up a lengthy/ambiguous scenario."
        if needs_rewrite
        else "Short, focused question—reuse directly for retrieval."
    )

    steps = ["rewrite" if needs_rewrite else "direct"]
    steps += ["retrieve", "draft", "critique", "optional_follow_up"]

    return AgentPlan(
        needs_rewrite=needs_rewrite,
        rationale=rationale,
        signals=signals or ["No rewrite heuristics fired."],
        steps=steps,
    )


def _merge_chunks(
    *chunk_groups: Sequence[RetrievedChunk],
    limit: int,
) -> List[RetrievedChunk]:
    scored: Dict[Tuple[str | None, Any, str], RetrievedChunk] = {}
    for group in chunk_groups:
        for chunk in group or []:
            meta = chunk.document.metadata or {}
            key = (
                str(meta.get("source")),
                meta.get("page_number"),
                chunk.document.page_content.strip(),
            )
            existing = scored.get(key)
            if not existing or chunk.score > existing.score:
                scored[key] = chunk
    ranked = sorted(scored.values(), key=lambda item: item.score, reverse=True)
    return ranked[:limit]


def _safe_json_loads(raw_text: str) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("Empty response.")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_text[start : end + 1])
    raise ValueError("Could not parse critique JSON.")


def _critique_answer(
    question: str,
    answer: str,
    chunks: Sequence[RetrievedChunk],
    llm,
) -> CritiqueResult:
    chunk_highlights = []
    for chunk in chunks[:3]:
        meta = chunk.document.metadata or {}
        snippet = chunk.document.page_content.strip().split("\n")[0]
        chunk_highlights.append(
            {
                "page": meta.get("page_number", "NA"),
                "section": meta.get("element_type", "Passage"),
                "score": round(chunk.score, 3),
                "snippet": snippet[:220],
            }
        )

    instructions = (
        "Review the grounded answer. Decide if the cited snippets are sufficient. "
        "If missing critical evidence, request one more retrieval."
    )

    response = llm.invoke(
        [
            (
                "system",
                "You are a negotiation QA checker. "
                "Return JSON with keys status ('ok' or 'needs_more'), notes, followup_keywords.",
            ),
            (
                "human",
                f"{instructions}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\n"
                f"Evidence summary:\n{json.dumps(chunk_highlights, ensure_ascii=False)}",
            ),
        ]
    )

    try:
        data = _safe_json_loads(response.content)
    except Exception as exc:  # pragma: no cover - surfaced to UI
        return CritiqueResult(
            verdict="error",
            notes=f"Critique parsing failed: {exc}",
            followup_keywords=[],
        )

    verdict = (data.get("status") or "ok").strip().lower()
    if verdict not in {"ok", "needs_more"}:
        verdict = "error"

    keywords = _dedupe_strings(data.get("followup_keywords", []))[:3]
    notes = data.get("notes", "").strip() or "No critique notes returned."

    return CritiqueResult(verdict=verdict, notes=notes, followup_keywords=keywords)


def _run_rewrite_stage(
    question: str,
    plan: AgentPlan,
    llm,
) -> Tuple[RewriteResult, str, str]:
    if plan.needs_rewrite:
        rewrite = rewrite_query(question, llm)
        return rewrite, "llm", "LLM rewrite applied based on plan signals."

    keywords = _extract_keywords(question)
    rewrite = RewriteResult(
        rewritten_query=question.strip(),
        concept_keywords=keywords,
        intent=DEFAULT_INTENT,
    )
    note = (
        "Skipped rewrite. Generated lightweight keywords from the original question "
        "to keep retrieval deterministic."
    )
    return rewrite, "direct", note


def run_agentic_assistant(
    question: str,
    resources: ChatbotResources,
    *,
    force_rewrite: bool,
    allow_followup: bool,
) -> AgenticRun:
    plan = _decide_plan(question, force_rewrite=force_rewrite)
    rewrite_result, rewrite_mode, rewrite_note = _run_rewrite_stage(
        question,
        plan,
        resources.llm,
    )

    chunks, retrieval_debug = retrieve_passages(
        resources.store,
        rewrite_result.rewritten_query,
        rewrite_result.intent,
        rewrite_result.concept_keywords,
        per_query_k=resources.per_query_k,
        final_k=resources.final_k,
    )

    draft_answer = generate_answer(
        user_question=question,
        rewritten_query=rewrite_result.rewritten_query,
        context_chunks=chunks,
        llm=resources.llm,
    )

    critique = _critique_answer(question, draft_answer, chunks, resources.llm)

    followup_chunks: List[RetrievedChunk] | None = None
    followup_debug: Dict[str, Any] | None = None
    final_chunks = chunks
    final_answer = draft_answer

    if (
        allow_followup
        and critique.verdict == "needs_more"
        and critique.followup_keywords
    ):
        extended_keywords = _dedupe_strings(
            rewrite_result.concept_keywords + critique.followup_keywords
        )
        followup_chunks, followup_debug = retrieve_passages(
            resources.store,
            rewrite_result.rewritten_query,
            rewrite_result.intent,
            extended_keywords,
            per_query_k=resources.per_query_k,
            final_k=resources.final_k,
        )
        combined = _merge_chunks(
            chunks,
            followup_chunks,
            limit=resources.final_k,
        )
        final_chunks = combined or chunks
        final_answer = generate_answer(
            user_question=question,
            rewritten_query=rewrite_result.rewritten_query,
            context_chunks=final_chunks,
            llm=resources.llm,
        )

    return AgenticRun(
        question=question,
        plan=plan,
        rewrite_mode=rewrite_mode,
        rewrite_notes=rewrite_note,
        rewrite_result=rewrite_result,
        initial_chunks=chunks,
        initial_debug=retrieval_debug,
        draft_answer=draft_answer,
        critique=critique,
        followup_chunks=followup_chunks,
        followup_debug=followup_debug,
        final_answer=final_answer,
    )


def _render_chunk_table(title: str, payload: Dict[str, Any] | None) -> None:
    with st.expander(title, expanded=False):
        if not payload:
            st.write("No retrieval diagnostics available.")
            return
        chunks = payload.get("chunks", [])
        if not chunks:
            st.write("No chunks retrieved.")
            return
        st.dataframe(
            chunks,
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Each row represents a unique chunk after scoring/deduplication.")


def _render_run(run: AgenticRun) -> None:
    st.subheader("Agent reasoning trace")
    st.markdown("### Final answer")
    st.markdown(run.final_answer or "_No answer generated._")

    if run.final_answer != run.draft_answer:
        st.info("The answer was refined after the critique. Initial draft shown below.")
        st.markdown("#### Initial draft")
        st.markdown(run.draft_answer)

    with st.expander("Step 0 · Plan", expanded=True):
        st.write(run.plan.rationale)
        st.write("**Signals considered**")
        for signal in run.plan.signals:
            st.write(f"- {signal}")
        st.write("**Step order**")
        st.write(" → ".join(run.plan.steps))

    with st.expander("Step 1 · Rewrite / intent handling", expanded=True):
        st.write(f"Mode: **{run.rewrite_mode}**")
        st.write(run.rewrite_notes)
        st.markdown("**Rewritten query**")
        st.code(run.rewrite_result.rewritten_query)
        st.markdown("**Detected intent**")
        st.write(run.rewrite_result.intent)
        st.markdown("**Concept keywords**")
        st.write(", ".join(run.rewrite_result.concept_keywords))

    st.subheader("Retrieval diagnostics")
    _render_chunk_table("Primary retrieval (first pass)", run.initial_debug)
    if run.followup_debug:
        _render_chunk_table("Follow-up retrieval (after critique)", run.followup_debug)

    st.subheader("Critique outcome")
    verdict_label = {
        "ok": "✅ No follow-up required.",
        "needs_more": "⚠️ Requested an additional pass.",
        "error": "⚠️ Critique failed.",
    }.get(run.critique.verdict, "—")
    st.write(verdict_label)
    st.write(run.critique.notes)
    if run.critique.followup_keywords:
        st.write("Suggested follow-up keywords:", ", ".join(run.critique.followup_keywords))

    st.subheader("JSON trace")
    st.code(json.dumps(run.summary(), indent=2), language="json")


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Agentic AI assistant", page_icon="🧭", layout="wide")
    _ensure_runs_state()

    st.title("Agentic AI assistant (prototype)")
    st.caption(
        "Runs a bounded plan → rewrite → retrieve → draft → critique loop. "
        "Great for validating light agent behaviors without fully autonomous tooling."
    )

    with st.form("agentic_form", clear_on_submit=False):
        question = st.text_area(
            "What negotiation scenario should the agent reason about?",
            placeholder=(
                "Describe the dilemma or question. For example: "
                "\"I need to re-engage a counterpart after an operational pause...\""
            ),
            height=160,
        )
        force_rewrite = st.checkbox(
            "Force rewrite",
            value=False,
            help="Overrides heuristics so the agent always runs the LLM rewrite step.",
        )
        allow_followup = st.checkbox(
            "Allow one critique-triggered follow-up retrieval",
            value=True,
            help="When enabled, the agent can run at most one additional retrieval pass.",
        )
        submitted = st.form_submit_button("Run agent", type="primary")

    try:
        resources = get_chat_resources()
    except Exception as exc:  # pragma: no cover - surfaced via UI
        st.error(str(exc))
        st.stop()

    if submitted:
        trimmed = (question or "").strip()
        if not trimmed:
            st.warning("Please enter a question before running the agent.")
        else:
            with st.spinner("Coordinating the mini agent..."):
                try:
                    run = run_agentic_assistant(
                        trimmed,
                        resources,
                        force_rewrite=force_rewrite,
                        allow_followup=allow_followup,
                    )
                except Exception as exc:  # pragma: no cover - surfaced via UI
                    st.error(f"Agent run failed: {exc}")
                else:
                    _store_run(run)
                    st.success("Agent run complete.")

    latest = _latest_run()
    if not latest:
        st.info("Submit a question to inspect the agent reasoning trace.")
        return

    _render_run(latest)


if __name__ == "__main__":
    main()



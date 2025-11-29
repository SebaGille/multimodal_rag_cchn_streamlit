from __future__ import annotations

import os
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from eval.judge import judge_answer
from eval.logger import write_eval_log
from eval.models import EvaluationQuestionSet, EvaluationResult, EvaluationRun
from eval.question_generator import generate_eval_questions
from services.chatbot_runner import ChatbotResources, get_chat_resources, run_cchn_chatbot

EVAL_STATE_KEY = "auto_eval_state"
RUBRIC_DIMENSIONS = [
    "context_quality",
    "counterpart_understanding",
    "tactical_tools",
    "principles_alignment",
    "operational_realism",
]


def _get_state() -> Dict[str, Any]:
    if EVAL_STATE_KEY not in st.session_state:
        st.session_state[EVAL_STATE_KEY] = {
            "questions": None,
            "results": None,
            "log_path": None,
        }
    return st.session_state[EVAL_STATE_KEY]


def _set_state(**kwargs: Any) -> None:
    state = _get_state()
    state.update(kwargs)
    st.session_state[EVAL_STATE_KEY] = state


def _build_judge_llm() -> ChatOpenAI:
    settings = st.session_state.get("rag_settings", {})
    model_name = settings.get("judge_model", settings.get("chat_model", "gpt-4o-mini"))
    return ChatOpenAI(model=model_name, temperature=0)


def _ensure_env() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Set OPENAI_API_KEY before running the evaluation.")
        st.stop()


def _render_question_list(question_set: EvaluationQuestionSet | None) -> None:
    if not question_set:
        st.info("Generate an evaluation set to see questions here.")
        return
    st.markdown(f"**Generated:** {question_set.generated_at.isoformat()} UTC")
    for idx, question in enumerate(question_set.questions, start=1):
        st.markdown(f"{idx}. {question}")


def _format_results(results: List[EvaluationResult]) -> Dict[str, float]:
    aggregates = {key: 0.0 for key in RUBRIC_DIMENSIONS}
    for result in results:
        scores = result.scores.as_dict()
        for key, value in scores.items():
            aggregates[key] += value
    count = len(results) or 1
    return {key: value / count for key, value in aggregates.items()}


def _render_results(run: EvaluationRun, log_path: str | None) -> None:
    st.subheader("Evaluation summary")
    cols = st.columns(3)
    cols[0].metric("Global average", f"{run.overall_stats.get('global_average', 0):.2f}")
    cols[1].metric("Red-flag answers", run.red_flag_count)
    cols[2].metric("Question count", len(run.results))

    st.markdown("**Per-dimension averages**")
    for key, value in run.overall_stats.items():
        if key == "global_average":
            continue
        st.write(f"- {key.replace('_', ' ').title()}: {value:.2f}")

    st.subheader("Per-question details")
    for idx, result in enumerate(run.results, start=1):
        with st.expander(f"Q{idx}: {result.question[:60]}{'...' if len(result.question) > 60 else ''}", expanded=False):
            st.markdown(f"**Question**: {result.question}")
            st.markdown(f"**Chatbot answer**:\n\n{result.answer}")
            st.markdown("**Scores**")
            score_cols = st.columns(len(RUBRIC_DIMENSIONS))
            for col, (dim, value) in zip(score_cols, result.scores.as_dict().items()):
                col.metric(dim.replace("_", " ").title(), f"{value:.1f}")
            st.metric("Final score", f"{result.final_score:.2f}")
            st.metric("Red flag", "Yes" if result.red_flag else "No")
            st.markdown(f"**Judge explanation**: {result.explanation}")
            if result.rubric_notes:
                st.markdown("**Notes**")
                for dim, note in result.rubric_notes.items():
                    if note:
                        st.write(f"- {dim.replace('_', ' ').title()}: {note}")

    if log_path:
        st.success(f"Evaluation log saved to {log_path}")


def _compute_overall_stats(results: List[EvaluationResult]) -> Dict[str, float]:
    aggregates = _format_results(results)
    global_avg = sum(aggregates.values()) / len(aggregates)
    aggregates["global_average"] = global_avg
    return aggregates


def main() -> None:
    load_dotenv()
    _ensure_env()
    st.set_page_config(page_title="LLM-as-a-Judge", page_icon="🧪")
    st.title("LLM-as-a-Judge")
    st.write("LLM-as-a-Judge generates CCHN-grounded scenario questions, runs the chatbot pipeline, and scores responses with a fixed rubric.")

    state = _get_state()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate evaluation set", type="primary"):
            try:
                judge_llm = _build_judge_llm()
                question_set = generate_eval_questions(judge_llm)
            except Exception as exc:  # pragma: no cover
                st.error(f"Failed to generate questions: {exc}")
            else:
                _set_state(questions=question_set, results=None, log_path=None)
                st.success("Generated 20 scenario questions.")

    with col2:
        if st.button("Clear questions/results"):
            _set_state(questions=None, results=None, log_path=None)
            st.info("Cleared evaluation session state.")

    st.subheader("Question set")
    _render_question_list(state.get("questions"))

    st.divider()

    resources: ChatbotResources | None = None
    try:
        resources = get_chat_resources()
    except Exception as exc:
        st.error(f"Unable to load chatbot resources: {exc}")

    if resources and state.get("questions"):
        if st.button("Run evaluation batch", type="secondary"):
            judge_llm = _build_judge_llm()
            question_set: EvaluationQuestionSet = state["questions"]
            progress = st.progress(0, "Evaluating...")
            results: List[EvaluationResult] = []

            for idx, question in enumerate(question_set.questions, start=1):
                progress.progress(idx / len(question_set.questions), text=f"Evaluating Q{idx}/{len(question_set.questions)}")
                chatbot_result = run_cchn_chatbot(question, resources)
                judge_result = judge_answer(question, chatbot_result.answer, judge_llm)
                results.append(judge_result)

            progress.empty()
            red_flags = sum(1 for item in results if item.red_flag)
            overall_stats = _compute_overall_stats(results)
            run = EvaluationRun(
                question_set=question_set,
                results=results,
                overall_stats=overall_stats,
                red_flag_count=red_flags,
            )

            log_path = write_eval_log(run)
            _set_state(results=run, log_path=str(log_path))
            st.success("Evaluation run completed.")

    if state.get("results"):
        _render_results(state["results"], state.get("log_path"))


if __name__ == "__main__":
    main()



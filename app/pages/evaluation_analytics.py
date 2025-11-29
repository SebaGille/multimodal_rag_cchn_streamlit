from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import streamlit as st

from eval.log_parser import DIMENSION_KEYS, LoggedAnswer, load_logged_answers

DIMENSION_LABELS: Dict[str, str] = {
    "context_quality": "Context analysis",
    "counterpart_understanding": "Counterpart motives",
    "tactical_tools": "Tactics & tools",
    "principles_alignment": "Principles alignment",
    "operational_realism": "Operational realism",
}

THEME_KEYWORDS: Dict[str, List[str]] = {
    "Access & security": ["access", "security", "safety", "ceasefire", "armed"],
    "Trust & legitimacy": ["trust", "legitimacy", "skeptic", "distrust"],
    "Misinformation & influence": ["misinformation", "narrative", "influence"],
    "Mandates & principles": ["principle", "mandate", "red line", "values"],
    "Stakeholder mapping": ["network", "mapping", "stakeholder", "counterpart"],
    "Resource negotiations": ["resource", "aid", "share", "demand"],
}

LOG_DIR = Path(__file__).resolve().parents[2] / "eval_logs"


def _detect_theme(question: str) -> str:
    lowered = question.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return theme
    return "Other / mixed"


def _format_timestamp(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def _compute_dimension_averages(answers: List[LoggedAnswer]) -> Dict[str, float]:
    totals = {dim: 0.0 for dim in DIMENSION_KEYS}
    for answer in answers:
        for dim in DIMENSION_KEYS:
            totals[dim] += answer.scores.get(dim, 0.0)
    count = len(answers) or 1
    return {dim: totals[dim] / count for dim in DIMENSION_KEYS}


def _compute_theme_performance(answers: List[LoggedAnswer]) -> List[Dict[str, float]]:
    aggregates: Dict[str, Dict[str, float]] = defaultdict(lambda: {"total": 0.0, "count": 0})
    for answer in answers:
        theme = _detect_theme(answer.question)
        aggregates[theme]["total"] += answer.final_score
        aggregates[theme]["count"] += 1

    rows: List[Dict[str, float]] = []
    for theme, payload in aggregates.items():
        avg = payload["total"] / payload["count"]
        rows.append({"theme": theme, "avg_score": avg, "samples": payload["count"]})
    rows.sort(key=lambda item: item["avg_score"])
    return rows


def _summarize_answers(answers: List[LoggedAnswer]) -> None:
    dim_avgs = _compute_dimension_averages(answers)
    red_flags = [item for item in answers if item.red_flag]
    red_prop = len(red_flags) / len(answers)

    st.subheader("Rubric health")
    cols = st.columns(len(DIMENSION_KEYS))
    for col, dim in zip(cols, DIMENSION_KEYS):
        col.metric(DIMENSION_LABELS[dim], f"{dim_avgs[dim]:.2f}")

    st.subheader("Quality & risk signals")
    signal_cols = st.columns(3)
    signal_cols[0].metric("Red-flag proportion", f"{red_prop:.1%}", help="Share of answers that triggered a red flag.")
    signal_cols[1].metric("Total answers", len(answers))
    signal_cols[2].metric("Evaluation runs", len({item.source_file for item in answers}))

    st.subheader("Worst performing themes")
    theme_rows = _compute_theme_performance(answers)
    if theme_rows:
        st.dataframe(
            theme_rows[:5],
            use_container_width=True,
            column_config={
                "theme": "Theme",
                "avg_score": st.column_config.NumberColumn("Avg. score", format="%.2f"),
                "samples": st.column_config.NumberColumn("Samples"),
            },
        )
    else:
        st.info("No themes to display yet.")

    st.subheader("Lowest scoring answers")
    for entry in sorted(answers, key=lambda item: item.final_score)[:3]:
        with st.expander(f"{entry.final_score:.2f} · {entry.question[:80]}{'...' if len(entry.question) > 80 else ''}"):
            st.caption(f"Run: {_format_timestamp(entry.run_timestamp)} • Log file: {entry.source_file.name}")
            st.markdown(f"**Question**: {entry.question}")
            st.markdown(f"**Answer excerpt**:\n\n{entry.answer[:600]}{'…' if len(entry.answer) > 600 else ''}")
            st.markdown(f"**Explanation**: {entry.explanation or '—'}")

    st.subheader("Recent red flags")
    if not red_flags:
        st.success("No red-flagged answers recorded.")
    else:
        for entry in sorted(red_flags, key=lambda item: item.run_timestamp, reverse=True)[:3]:
            st.write(f"- {_format_timestamp(entry.run_timestamp)} · {entry.question[:90]}{'...' if len(entry.question) > 90 else ''}")
            st.caption(f"Final score: {entry.final_score:.2f} • Explanation: {entry.explanation or '—'}")


def main() -> None:
    st.set_page_config(page_title="Evaluation analytics", page_icon="📊")
    st.title("Evaluation analytics")
    st.write("Explore how the chatbot performs across rubric dimensions, themes, and safety signals using the LLM-as-a-judge logs.")

    answers = load_logged_answers(LOG_DIR)
    if not answers:
        st.info("No evaluation logs found yet. Run the LLM-as-a-judge workflow to populate analytics.")
        return

    run_files = sorted({item.source_file for item in answers})
    first_run = min((item.run_timestamp for item in answers), default=None)
    latest_run = max((item.run_timestamp for item in answers), default=None)

    if first_run and latest_run:
        st.caption(
            f"Analytics include {len(answers)} answers from {len(run_files)} evaluation runs, collected between {_format_timestamp(first_run)} and {_format_timestamp(latest_run)}."
        )
    else:
        st.caption(f"Analytics include {len(answers)} answers from {len(run_files)} evaluation runs.")

    _summarize_answers(answers)


if __name__ == "__main__":
    main()


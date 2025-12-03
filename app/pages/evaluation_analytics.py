from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from eval.log_parser import DIMENSION_KEYS, LoggedAnswer, load_logged_answers

DIMENSION_LABELS: Dict[str, str] = {
    "context_quality": "Context analysis",
    "counterpart_understanding": "Counterpart motives",
    "tactical_tools": "Tactics & tools",
    "principles_alignment": "Principles alignment",
    "operational_realism": "Operational realism",
}
LABEL_TO_DIMENSION = {label: key for key, label in DIMENSION_LABELS.items()}

THEME_KEYWORDS: Dict[str, List[str]] = {
    "Access & security": ["access", "security", "safety", "ceasefire", "armed"],
    "Trust & legitimacy": ["trust", "legitimacy", "skeptic", "distrust"],
    "Misinformation & influence": ["misinformation", "narrative", "influence"],
    "Mandates & principles": ["principle", "mandate", "red line", "values"],
    "Stakeholder mapping": ["network", "mapping", "stakeholder", "counterpart"],
    "Resource negotiations": ["resource", "aid", "share", "demand"],
}

LOG_DIR = Path(__file__).resolve().parents[2] / "eval_logs"


@st.cache_data(show_spinner=False)
def _load_cached_answers(log_dir: Path) -> List[LoggedAnswer]:
    return load_logged_answers(log_dir)


def _safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def _safe_stdev(values: List[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


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


def _compute_dimension_stats(answers: List[LoggedAnswer]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for dim in DIMENSION_KEYS:
        values = [answer.scores.get(dim, 0.0) for answer in answers]
        stats[dim] = {"mean": _safe_mean(values), "std": _safe_stdev(values)}
    return stats


def _compute_final_score_stats(answers: List[LoggedAnswer]) -> Dict[str, float]:
    scores = [answer.final_score for answer in answers]
    return {"mean": _safe_mean(scores), "std": _safe_stdev(scores)}


def _group_answers_by_run(answers: List[LoggedAnswer]) -> Dict[Path, List[LoggedAnswer]]:
    runs: Dict[Path, List[LoggedAnswer]] = defaultdict(list)
    for answer in answers:
        runs[answer.source_file].append(answer)
    return runs


def _build_run_rows(answers: List[LoggedAnswer]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for source_file, entries in _group_answers_by_run(answers).items():
        final_scores = [entry.final_score for entry in entries]
        run_timestamp = max(entry.run_timestamp for entry in entries)
        row: Dict[str, object] = {
            "run_label": source_file.stem,
            "timestamp": run_timestamp,
            "samples": len(entries),
            "mean_final": _safe_mean(final_scores),
            "std_final": _safe_stdev(final_scores),
            "red_flag_rate": sum(1 for entry in entries if entry.red_flag) / len(entries),
        }
        row["cv_final"] = row["std_final"] / row["mean_final"] if row["mean_final"] else 0.0
        for dim in DIMENSION_KEYS:
            dim_values = [entry.scores.get(dim, 0.0) for entry in entries]
            row[f"mean_{dim}"] = _safe_mean(dim_values)
            row[f"std_{dim}"] = _safe_stdev(dim_values)
        rows.append(row)
    rows.sort(key=lambda item: item["timestamp"])
    return rows


def _render_run_stability(run_rows: List[Dict[str, object]]) -> None:
    if not run_rows:
        st.info("No evaluation runs to compare yet.")
        return

    run_df = pd.DataFrame(run_rows)
    run_df["run_time"] = run_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
    table_cols = [
        "run_label",
        "run_time",
        "samples",
        "mean_final",
        "std_final",
        "cv_final",
        "red_flag_rate",
    ]
    st.dataframe(
        run_df[table_cols],
        use_container_width=True,
        column_config={
            "run_label": "Run",
            "run_time": "Completed",
            "samples": "Samples",
            "mean_final": st.column_config.NumberColumn("Mean score", format="%.2f"),
            "std_final": st.column_config.NumberColumn("Std dev", format="%.2f"),
            "cv_final": st.column_config.NumberColumn("Coeff. variation", format="%.2f"),
            "red_flag_rate": st.column_config.ProgressColumn(
                "Red-flag rate", format="%.0f%%", min_value=0.0, max_value=1.0
            ),
        },
        hide_index=True,
    )
    st.caption("Higher coefficient of variation signals unstable evaluation runs.")

    metric_options = ["Final score"] + list(DIMENSION_LABELS.values())
    focus_label = st.selectbox("Stability focus", metric_options, index=0)
    if focus_label == "Final score":
        value_col = "mean_final"
        std_col = "std_final"
        title = "Final score trend"
    else:
        dim_key = LABEL_TO_DIMENSION[focus_label]
        value_col = f"mean_{dim_key}"
        std_col = f"std_{dim_key}"
        title = f"{focus_label} trend"

    chart_df = run_df[["timestamp", "run_label", value_col, std_col]].copy()
    chart_df.rename(columns={value_col: "value", std_col: "spread"}, inplace=True)
    chart_df["upper"] = chart_df["value"] + chart_df["spread"]
    chart_df["lower"] = (chart_df["value"] - chart_df["spread"]).clip(lower=0.0)
    chart_df["rolling"] = chart_df["value"].rolling(window=3, min_periods=1).mean()

    base = alt.Chart(chart_df).encode(x=alt.X("timestamp:T", title="Run timestamp"))
    band = base.mark_area(opacity=0.2, color="#4c78a8").encode(y="lower:Q", y2="upper:Q")
    line = base.mark_line(color="#3b7dd8").encode(
        y=alt.Y("value:Q", title="Score"),
        tooltip=[
            alt.Tooltip("run_label", title="Run"),
            alt.Tooltip("timestamp:T", title="Timestamp"),
            alt.Tooltip("value:Q", title="Mean score", format=".2f"),
            alt.Tooltip("spread:Q", title="Std dev", format=".2f"),
        ],
    )
    rolling = base.mark_line(color="#f58518", strokeDash=[6, 3]).encode(
        y="rolling:Q", tooltip=[alt.Tooltip("rolling:Q", title="Rolling mean", format=".2f")]
    )
    st.altair_chart((band + line + rolling).properties(title=title, height=320), use_container_width=True)


def _compute_error_summary(
    answers: List[LoggedAnswer],
    dim_stats: Dict[str, Dict[str, float]],
    final_stats: Dict[str, float],
) -> Dict[str, object]:
    threshold_base = final_stats["mean"] - (final_stats["std"] or 0.15)
    threshold = max(threshold_base, 0.0)
    summary = {"threshold": threshold, "buckets": {"Systematic": 0, "Random": 0}}
    systematic_dims: Dict[str, int] = defaultdict(int)
    samples: List[Dict[str, object]] = []

    for answer in answers:
        is_issue = answer.red_flag or answer.final_score < threshold
        if not is_issue:
            continue
        dims_below: List[str] = []
        for dim in DIMENSION_KEYS:
            dim_mean = dim_stats[dim]["mean"]
            dim_std = dim_stats[dim]["std"] or 0.15
            if answer.scores.get(dim, 0.0) < dim_mean - dim_std:
                dims_below.append(dim)

        classification = "Systematic" if dims_below else "Random"
        summary["buckets"][classification] += 1
        for dim in dims_below:
            systematic_dims[dim] += 1
        samples.append(
            {
                "question": answer.question,
                "score": answer.final_score,
                "classification": classification,
                "timestamp": answer.run_timestamp,
                "run": answer.source_file.stem,
                "explanation": answer.explanation or "—",
                "dimensions": [DIMENSION_LABELS[dim] for dim in dims_below],
            }
        )

    summary["issues_total"] = sum(summary["buckets"].values())
    summary["systematic_top_dims"] = [
        {"Dimension": DIMENSION_LABELS[dim], "Count": count} for dim, count in sorted(systematic_dims.items(), key=lambda item: item[1], reverse=True)
    ]
    summary["samples"] = sorted(samples, key=lambda item: item["score"])[:5]
    return summary


def _render_error_decomposition(summary: Dict[str, object]) -> None:
    if not summary["issues_total"]:
        st.success("No low-scoring or red-flagged answers that meet the error criteria.")
        return

    cols = st.columns(2)
    cols[0].metric("Systematic issues", summary["buckets"]["Systematic"])
    cols[1].metric("Likely random noise", summary["buckets"]["Random"])
    st.caption(
        f"Issues consider answers below a dynamic threshold ({summary['threshold']:.2f}) or those that triggered a red flag. "
        "Systematic = consistent rubric dimension underperformance."
    )

    if summary["systematic_top_dims"]:
        st.dataframe(
            summary["systematic_top_dims"],
            use_container_width=True,
            column_config={
                "Dimension": "Dimension",
                "Count": st.column_config.NumberColumn("Issue count"),
            },
            hide_index=True,
        )
    else:
        st.info("No repeated rubric weaknesses detected yet.")

    st.markdown("**Recent diagnostics**")
    for sample in summary["samples"]:
        label_suffix = f" · {' / '.join(sample['dimensions'])}" if sample["dimensions"] else ""
        header = f"{sample['classification']} · {sample['score']:.2f}{label_suffix}"
        with st.expander(header):
            st.caption(f"Run: {sample['run']} • {sample['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}")
            st.markdown(f"**Question:** {sample['question']}")
            st.markdown(f"**Explanation:** {sample['explanation']}")


def _summarize_answers(answers: List[LoggedAnswer]) -> None:
    dim_avgs = _compute_dimension_averages(answers)
    run_rows = _build_run_rows(answers)
    dim_stats = _compute_dimension_stats(answers)
    final_stats = _compute_final_score_stats(answers)
    error_summary = _compute_error_summary(answers, dim_stats, final_stats)
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
    signal_cols[2].metric("Evaluation runs", len(run_rows))

    st.subheader("Run stability & variance")
    _render_run_stability(run_rows)

    st.subheader("Error decomposition")
    _render_error_decomposition(error_summary)

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

    answers = _load_cached_answers(LOG_DIR)
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


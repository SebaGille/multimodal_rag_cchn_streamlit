from __future__ import annotations

import json
from pathlib import Path

from .models import EvaluationResult, EvaluationRun

LOG_DIR = Path("eval_logs")


def _format_scores(result: EvaluationResult) -> str:
    scores = result.scores.as_dict()
    parts = [f"  - {dim}: {value:.1f}" for dim, value in scores.items()]
    return "\n".join(parts)


def _write_text_log(run: EvaluationRun, path: Path) -> None:
    lines = [
        f"Evaluation run: {run.question_set.generated_at.isoformat()}",
        f"Question count: {len(run.question_set.questions)}",
        f"Difficulty mix: {run.question_set.difficulty_mix_note}",
        "",
    ]

    for idx, result in enumerate(run.results, start=1):
        lines.extend(
            [
                f"Q{idx}: {result.question}",
                f"Answer:\n{result.answer}",
                "Scores:",
                _format_scores(result),
                f"Final score: {result.final_score:.2f}",
                f"Red flag: {result.red_flag}",
                f"Explanation: {result.explanation}",
            ]
        )
        if result.rubric_notes:
            lines.append("Notes:")
            for dim, note in result.rubric_notes.items():
                if note:
                    lines.append(f"  - {dim}: {note}")
        lines.append("")

    lines.append("Overall stats:")
    for key, value in run.overall_stats.items():
        lines.append(f"- {key}: {value:.2f}")
    lines.append(f"Red-flag answers: {run.red_flag_count}")

    path.write_text("\n".join(lines), encoding="utf-8")


def _build_json_payload(run: EvaluationRun) -> dict:
    return {
        "log_version": 1,
        "metadata": {
            "generated_at": run.question_set.generated_at.isoformat(),
            "question_count": len(run.question_set.questions),
            "difficulty_note": run.question_set.difficulty_mix_note,
            "red_flag_count": run.red_flag_count,
            "overall_stats": run.overall_stats,
        },
        "questions": run.question_set.questions,
        "results": [
            {
                "index": idx,
                "question": result.question,
                "answer": result.answer,
                "scores": result.scores.as_dict(),
                "final_score": result.final_score,
                "red_flag": result.red_flag,
                "explanation": result.explanation,
                "rubric_notes": result.rubric_notes,
            }
            for idx, result in enumerate(run.results, start=1)
        ],
    }


def write_eval_log(run: EvaluationRun, *, log_dir: Path | None = None) -> Path:
    target_dir = log_dir or LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = run.question_set.generated_at.strftime("%Y%m%d_%H%M%S")
    base_path = target_dir / f"eval_{timestamp}"

    text_path = base_path.with_suffix(".txt")
    json_path = base_path.with_suffix(".json")

    _write_text_log(run, text_path)
    payload = _build_json_payload(run)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return json_path


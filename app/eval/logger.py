from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import EvaluationResult, EvaluationRun

LOG_DIR = Path("eval_logs")


def _format_scores(result: EvaluationResult) -> str:
    scores = result.scores.as_dict()
    parts = [f"  - {dim}: {value:.1f}" for dim, value in scores.items()]
    return "\n".join(parts)


def write_eval_log(run: EvaluationRun, *, log_dir: Path | None = None) -> Path:
    target_dir = log_dir or LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = run.question_set.generated_at.strftime("%Y%m%d_%H%M%S")
    file_path = target_dir / f"eval_{timestamp}.txt"

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

    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path


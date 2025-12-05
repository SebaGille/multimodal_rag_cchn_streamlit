from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

DIMENSION_KEYS = [
    "context_quality",
    "counterpart_understanding",
    "tactical_tools",
    "principles_alignment",
    "operational_realism",
]


@dataclass
class LoggedAnswer:
    question_index: int
    question: str
    answer: str
    scores: Dict[str, float]
    final_score: float
    red_flag: bool
    explanation: str
    run_timestamp: datetime
    source_file: Path


def _extract_run_timestamp(lines: Iterable[str]) -> datetime | None:
    for line in lines:
        if line.startswith("Evaluation run:"):
            _, _, value = line.partition(":")
            try:
                return datetime.fromisoformat(value.strip())
            except ValueError:
                return None
    return None


def _parse_scores(lines: List[str], start_idx: int) -> tuple[Dict[str, float], int]:
    scores: Dict[str, float] = {}
    idx = start_idx
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("- "):
            break
        key, _, value = line[2:].partition(":")
        try:
            scores[key.strip()] = float(value.strip())
        except ValueError:
            scores[key.strip()] = 0.0
        idx += 1
    return scores, idx


def _parse_text_log(path: Path) -> List[LoggedAnswer]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    run_ts = _extract_run_timestamp(lines) or datetime.fromtimestamp(path.stat().st_mtime)
    entries: List[LoggedAnswer] = []
    idx = 0
    question_pattern = re.compile(r"^Q(\d+):\s*(.+)")

    while idx < len(lines):
        match = question_pattern.match(lines[idx])
        if not match:
            idx += 1
            continue

        question_index = int(match.group(1))
        question = match.group(2).strip()
        idx += 1

        # Consume until after "Answer:" marker
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        if idx < len(lines) and lines[idx].startswith("Answer:"):
            idx += 1

        answer_lines: List[str] = []
        while idx < len(lines) and lines[idx] != "Scores:":
            answer_lines.append(lines[idx])
            idx += 1
        answer = "\n".join(answer_lines).strip()

        while idx < len(lines) and lines[idx] != "Scores:":
            idx += 1
        if idx < len(lines) and lines[idx] == "Scores:":
            idx += 1

        scores, idx = _parse_scores(lines, idx)

        final_score = 0.0
        red_flag = False
        explanation = ""

        if idx < len(lines) and lines[idx].startswith("Final score:"):
            _, _, value = lines[idx].partition(":")
            try:
                final_score = float(value.strip())
            except ValueError:
                final_score = 0.0
            idx += 1

        if idx < len(lines) and lines[idx].startswith("Red flag:"):
            _, _, value = lines[idx].partition(":")
            red_flag = value.strip().lower() == "true"
            idx += 1

        if idx < len(lines) and lines[idx].startswith("Explanation:"):
            _, _, explanation_text = lines[idx].partition(":")
            explanation = explanation_text.strip()
            idx += 1

        # Skip optional notes block
        while idx < len(lines) and lines[idx] and not lines[idx].startswith("Q"):
            idx += 1
        entries.append(
            LoggedAnswer(
                question_index=question_index,
                question=question,
                answer=answer,
                scores=scores,
                final_score=final_score,
                red_flag=red_flag,
                explanation=explanation,
                run_timestamp=run_ts,
                source_file=path,
            )
        )

    return entries


def _parse_json_log(path: Path) -> List[LoggedAnswer]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    metadata = data.get("metadata", {})
    ts_value = metadata.get("completed_at") or metadata.get("generated_at")
    if isinstance(ts_value, str):
        try:
            run_ts = datetime.fromisoformat(ts_value)
        except ValueError:
            run_ts = datetime.fromtimestamp(path.stat().st_mtime)
    else:
        run_ts = datetime.fromtimestamp(path.stat().st_mtime)

    entries: List[LoggedAnswer] = []
    results = data.get("results", [])
    for fallback_idx, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        question = item.get("question", "")
        answer = item.get("answer", "")
        scores = {dim: float(item.get("scores", {}).get(dim, 0.0)) for dim in DIMENSION_KEYS}
        final_score = float(item.get("final_score", 0.0))
        red_flag = bool(item.get("red_flag", False))
        explanation = item.get("explanation", "") or ""
        question_index = int(item.get("index", fallback_idx))
        entries.append(
            LoggedAnswer(
                question_index=question_index,
                question=question,
                answer=answer,
                scores=scores,
                final_score=final_score,
                red_flag=red_flag,
                explanation=explanation,
                run_timestamp=run_ts,
                source_file=path,
            )
        )
    return entries


def load_logged_answers(log_dir: Path) -> List[LoggedAnswer]:
    if not log_dir.exists():
        return []
    answers: List[LoggedAnswer] = []
    seen_bases = set()

    for path in sorted(log_dir.glob("eval_*.json")):
        try:
            answers.extend(_parse_json_log(path))
            seen_bases.add(path.stem)
        except OSError:
            continue

    for path in sorted(log_dir.glob("eval_*.txt")):
        if path.stem in seen_bases:
            continue
        try:
            answers.extend(_parse_text_log(path))
        except OSError:
            continue
    return answers


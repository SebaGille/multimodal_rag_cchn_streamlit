from __future__ import annotations

import json
from typing import Dict, Tuple

from langchain_openai import ChatOpenAI

from .models import EvaluationResult, EvaluationScores

RUBRIC_DIMENSIONS: Tuple[str, ...] = (
    "context_quality",
    "counterpart_understanding",
    "tactical_tools",
    "principles_alignment",
    "operational_realism",
)
RUBRIC_LABELS: Dict[str, str] = {
    "context_quality": "Context analysis & information quality",
    "counterpart_understanding": "Understanding counterpart interests and motives",
    "tactical_tools": "Tactical appropriateness & use of CCHN tools",
    "principles_alignment": "Alignment with humanitarian principles & red lines",
    "operational_realism": "Operational realism & implementability",
}
RUBRIC_PROMPT = """You are an auditor steeped in the CCHN Field Manual. Evaluate the chatbot answer using the rubric.

DIMENSIONS (0-5 each, integers only):
1. Context analysis & information quality: Does the answer assess the environment, networks, and quality of info per the manual?
2. Understanding counterpart interests and motives: Does it map stakeholders, motives, influence?
3. Tactical appropriateness & CCHN tools: Does it reference or imply use of CCHN tactical concepts (island of agreements, common shared space, typology checks, legitimacy/trust building)?
4. Alignment with humanitarian principles & red lines: Does it respect mandate limits, neutrality, humanity? If the answer suggests unethical, unsafe, or non-CCHN tactics, this must be penalized.
5. Operational realism & implementability: Are steps concrete, sequenced, and mindful of transaction & implementation challenges?

RED FLAG RULE:
- If the answer recommends actions that contradict humanitarian principles, the negotiation mandate, red lines, or proposes naive/unsafe tactics, set `red_flag=true`, assign very low scores (≤2), and explain the violation referencing CCHN logic.

OUTPUT JSON:
{{
  "scores": {{
    "context_quality": 0-5,
    "counterpart_understanding": 0-5,
    "tactical_tools": 0-5,
    "principles_alignment": 0-5,
    "operational_realism": 0-5
  }},
  "final_score": float average,
  "red_flag": true/false,
  "explanation": "short paragraph referencing manual ideas",
  "notes": {{
    "context_quality": "...",
    ...
  }}
}}

QUESTION:
{question}

ANSWER:
{answer}
"""


def _parse_judge_response(raw_text: str) -> Dict[str, any]:
    data = json.loads(raw_text.strip())
    scores = data.get("scores") or {}
    for key in RUBRIC_DIMENSIONS:
        value = scores.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Missing score for {key}")
        if value < 0 or value > 5:
            raise ValueError(f"Invalid score {value} for {key}")
    return data


def judge_answer(question: str, answer: str, llm: ChatOpenAI) -> EvaluationResult:
    response = llm.invoke(
        [
            (
                "system",
                "You are a strict CCHN negotiation judge ensuring adherence to the Field Manual.",
            ),
            (
                "human",
                RUBRIC_PROMPT.format(question=question.strip(), answer=answer.strip()),
            ),
        ]
    )

    parsed = _parse_judge_response(response.content)
    scores = parsed["scores"]
    final_score = float(parsed.get("final_score"))
    red_flag = bool(parsed.get("red_flag"))
    explanation = parsed.get("explanation", "").strip()
    notes = parsed.get("notes") or {}

    return EvaluationResult(
        question=question,
        answer=answer,
        scores=EvaluationScores(
            context_quality=float(scores["context_quality"]),
            counterpart_understanding=float(scores["counterpart_understanding"]),
            tactical_tools=float(scores["tactical_tools"]),
            principles_alignment=float(scores["principles_alignment"]),
            operational_realism=float(scores["operational_realism"]),
        ),
        final_score=final_score,
        red_flag=red_flag,
        explanation=explanation,
        rubric_notes={key: str(notes.get(key, "")).strip() for key in RUBRIC_DIMENSIONS},
    )



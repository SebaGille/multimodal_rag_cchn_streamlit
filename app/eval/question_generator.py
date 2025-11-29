from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI

from .models import EvaluationQuestionSet

GENERATOR_PROMPT = """You are an evaluator designing scenario questions to stress-test a humanitarian
negotiation chatbot trained on the CCHN Field Manual.

TASK:
- Produce exactly {count} distinct scenario questions in JSON as {{"questions": [ ... ]}}.
- Each question must describe a realistic frontline humanitarian negotiation dilemma grounded
  in the CCHN Field Manual themes: context analysis & quality of information, mapping networks
  and influence, legitimacy and trust building, negotiation typology choices, mandate/red lines,
  and transaction/implementation issues.
- Write every scenario in the first person ("I" or "we") so it feels like a field note from the
  humanitarian negotiator.
- Mention at least one specific person name plus one concrete location or country in every question,
  rotating across European, African, American, Latin American, and Asian contexts to keep things diverse.
- Use natural English between CEFR B1 and C1—clear sentences with occasional informal phrasing,
  never academic jargon.
- Avoid trivial definitions or purely theoretical asks. Each scenario should mention actors,
  context, constraints, and negotiation tension that requires judgment.
- Vary the level of difficulty implicitly (mix of easier, moderate, and very challenging dilemmas).
- Include references to tactics/tools when relevant (e.g., island of agreements, common shared space,
  network mapping, legitimacy sources) but do not quote the manual verbatim.
- Keep each question under 80 words and ensure it can be answered with grounded guidance.

Return JSON only.
"""


def _parse_question_response(raw_text: str) -> List[str]:
    import json

    raw_text = raw_text.strip()
    data = json.loads(raw_text)
    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Generator did not return a questions list.")
    cleaned = []
    for question in questions:
        text = (question or "").strip()
        if not text:
            continue
        cleaned.append(text)
    if len(cleaned) < len(questions):
        raise ValueError("Some generated questions were empty.")
    return cleaned


def generate_eval_questions(
    llm: ChatOpenAI,
    *,
    count: int = 20,
    max_attempts: int = 3,
) -> EvaluationQuestionSet:
    if count <= 0:
        raise ValueError("Count must be positive.")

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(
                [
                    (
                        "system",
                        "You design frontline humanitarian negotiation evaluation scenarios that stress-test adherence to the CCHN Field Manual.",
                    ),
                    (
                        "human",
                        GENERATOR_PROMPT.format(count=count),
                    ),
                ],
                response_format={"type": "json_object"},
            )

            raw_payload = response.content
            if not isinstance(raw_payload, str):
                import json

                raw_payload = json.dumps(raw_payload)

            questions = _parse_question_response(raw_payload)
            if len(questions) == count:
                return EvaluationQuestionSet(questions=questions)

            raise ValueError(
                f"Generator returned {len(questions)} questions, expected {count}."
            )
        except Exception as exc:  # pragma: no cover - relies on remote LLM behavior
            last_error = exc
            if attempt < max_attempts:
                continue

    raise ValueError(f"Question generation failed after {max_attempts} attempts: {last_error}")


from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class EvaluationQuestionSet:
    questions: List[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    difficulty_mix_note: str = "Includes varied frontline negotiation scenarios per CCHN guidance."


@dataclass
class EvaluationScores:
    context_quality: float
    counterpart_understanding: float
    tactical_tools: float
    principles_alignment: float
    operational_realism: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "context_quality": self.context_quality,
            "counterpart_understanding": self.counterpart_understanding,
            "tactical_tools": self.tactical_tools,
            "principles_alignment": self.principles_alignment,
            "operational_realism": self.operational_realism,
        }


@dataclass
class EvaluationResult:
    question: str
    answer: str
    scores: EvaluationScores
    final_score: float
    red_flag: bool
    explanation: str
    rubric_notes: Dict[str, str]


@dataclass
class EvaluationRun:
    question_set: EvaluationQuestionSet
    results: List[EvaluationResult]
    overall_stats: Dict[str, float]
    red_flag_count: int
    completed_at: datetime = field(default_factory=datetime.utcnow)


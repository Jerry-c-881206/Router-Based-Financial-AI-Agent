"""
Execution Planner（SDD v2.0 §4）

依 Query Understanding 的 intent 分流至對應 Pipeline：
- fact        → Fact Pipeline
- aggregation → Aggregation Pipeline
- opinion     → Opinion Pipeline（待實作）
- reasoning   → Reasoning Pipeline（待實作）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv

from aggregation_pipeline import AggregationResult, run_aggregation_pipeline
from fact_pipeline import FactResult, run_fact_pipeline
from opinion_pipeline import OpinionResult, run_opinion_pipeline
from reasoning_pipeline import ReasoningResult, run_reasoning_pipeline
from query_understanding import IntentType, QueryUnderstanding


PlannerResultKind = Literal["fact", "aggregation", "opinion", "reasoning"]


@dataclass(frozen=True)
class PlannerResult:
    """Unified wrapper for downstream Response Generator."""

    kind: PlannerResultKind
    raw: Any  # FactResult / AggregationResult / OpinionResult / ReasoningResult


def plan_and_execute(qu: QueryUnderstanding, *, user_question: str) -> PlannerResult:
    """
    SDD §4 的簡單 if-else Router：
    - 把 QueryUnderstanding 丟給對應 Pipeline
    - 目前已實作：Fact / Aggregation
    - Opinion / Reasoning 先以明確錯誤訊息佔位
    """
    load_dotenv()

    if qu.validation_error:
        error_result = FactResult(
            entity=qu.entity or "",
            time_range=qu.time_constraint.range or "",
            task=qu.task,
            data=None,
            formatted_answer=qu.validation_error,
        )
        return PlannerResult(kind="fact", raw=error_result)

    intent: IntentType = qu.intent

    if intent == "fact":
        fact = run_fact_pipeline(qu, user_question=user_question)
        return PlannerResult(kind="fact", raw=fact)

    if intent == "aggregation":
        agg = run_aggregation_pipeline(qu, user_question=user_question)
        return PlannerResult(kind="aggregation", raw=agg)

    if intent == "opinion":
        op = run_opinion_pipeline(qu, user_question=user_question)
        return PlannerResult(kind="opinion", raw=op)

    if intent == "reasoning":
        rs = run_reasoning_pipeline(qu, user_question=user_question)
        return PlannerResult(kind="reasoning", raw=rs)

    # 理論上不會到這裡，保險起見
    raise ValueError(f"Unknown intent: {intent}")


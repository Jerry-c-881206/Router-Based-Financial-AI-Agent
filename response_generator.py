"""
Response Generator（SDD v2.0 §6）

統一格式化各 Pipeline 的輸出：

🔍 分析模式：{Fact / Aggregation / Opinion / Reasoning}

📊 {entity}｜{time_range}｜{問題摘要}

{對應內容區塊}

---
資料來源：{FinMind API / Tavily Search}（搜尋/查詢日期：YYYY-MM-DD）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from execution_planner import PlannerResult
from fact_pipeline import FactResult
from aggregation_pipeline import AggregationResult
from opinion_pipeline import OpinionResult
from reasoning_pipeline import ReasoningResult


@dataclass(frozen=True)
class RenderContext:
    mode_label: str
    entity: str
    time_range: str
    question_summary: str
    body: str
    source_label: str
    date_label: str


_REALTIME_ANCHOR_KEYWORDS = [
    "昨天", "昨日",
    "上週", "上周",
    "本週", "本周",
    "今天", "今日",
    "最近幾天", "最近幾日", "近幾天", "近幾日",
    "這幾天", "這幾日",
    "近日", "近期幾天",
    "最新消息", "最新狀況", "最新情況",
]

_REALTIME_WARNING = (
    "\n\n> ⚠️ **即時性提醒**：您的問題包含短期時間錨點（如「昨天」、「上週」、「最近幾天」等）。"
    "本系統資料來源（FinMind API / Tavily Search）通常有 **1–3 個工作天的資料延遲**，"
    "查詢結果未必反映最即時的市場狀況，請參考時留意時效性。"
)


def _detect_realtime_anchor(user_question: str) -> bool:
    """偵測問題是否包含短期時間錨點關鍵詞。"""
    q = (user_question or "").replace(" ", "")
    return any(kw in q for kw in _REALTIME_ANCHOR_KEYWORDS)


def _today_str() -> str:
    return date.today().isoformat()


def _summarize_question(original_question: str, task: str) -> str:
    """
    簡易問題摘要：目前先直接用 task + 原問題，未來可以導入專門的 Summarizer。
    """
    t = (task or "").strip()
    if t:
        return f"{t}｜{original_question}"
    return original_question


def _render(ctx: RenderContext) -> str:
    return (
        f"🔍 分析模式：{ctx.mode_label}\n\n"
        f"📊 {ctx.entity}｜{ctx.time_range}｜{ctx.question_summary}\n\n"
        f"{ctx.body}\n\n"
        f"---\n"
        f"資料來源：{ctx.source_label}（查詢日期：{ctx.date_label}）"
    )


def _from_fact(result: FactResult, *, user_question: str) -> str:
    entity = result.entity or "（未指定標的）"
    time_range = result.time_range or "last_90_days"
    question_summary = _summarize_question(user_question, result.task)

    # Fact Pipeline 的 formatted_answer 已經是條列式內容，這裡直接嵌入。
    body = result.formatted_answer

    # 偵測實際有用到哪些資料來源（目前 Fact 只用 FinMind）。
    source_label = "FinMind API"

    ctx = RenderContext(
        mode_label="Fact",
        entity=entity,
        time_range=time_range,
        question_summary=question_summary,
        body=body,
        source_label=source_label,
        date_label=_today_str(),
    )
    return _render(ctx)


def _from_aggregation(result: AggregationResult, *, user_question: str) -> str:
    entity = result.entity or "（未指定標的）"
    time_range = result.time_range or "last_90_days"
    question_summary = _summarize_question(user_question, result.task)

    body = result.formatted_answer

    # Aggregation 目前只用 Tavily Search。
    source_label = "Tavily Search"

    ctx = RenderContext(
        mode_label="Aggregation",
        entity=entity,
        time_range=time_range,
        question_summary=question_summary,
        body=body,
        source_label=source_label,
        date_label=_today_str(),
    )
    return _render(ctx)


def _from_opinion(result: OpinionResult, *, user_question: str) -> str:
    entity = result.entity or "（未指定標的）"
    # Opinion 的時間範圍比較多樣，這裡簡化顯示為「依各維度預設」。
    time_range = "多維度時窗"
    question_summary = _summarize_question(user_question, "investment_eval")

    body = result.formatted_answer

    ctx = RenderContext(
        mode_label="Opinion",
        entity=entity,
        time_range=time_range,
        question_summary=question_summary,
        body=body,
        source_label="FinMind API / Tavily Search",
        date_label=_today_str(),
    )
    return _render(ctx)


def render_response(planner_result: PlannerResult, *, user_question: str) -> str:
    """
    將 Execution Planner 的結果轉成最終回應字串。
    目前已支援：Fact / Aggregation。
    Opinion / Reasoning 會在對應 Pipeline 完成後補上。
    """
    if planner_result.kind == "fact":
        response = _from_fact(planner_result.raw, user_question=user_question)  # type: ignore[arg-type]

    elif planner_result.kind == "aggregation":
        response = _from_aggregation(planner_result.raw, user_question=user_question)  # type: ignore[arg-type]

    elif planner_result.kind == "opinion":
        response = _from_opinion(planner_result.raw, user_question=user_question)  # type: ignore[arg-type]

    elif planner_result.kind == "reasoning":
        result = planner_result.raw  # type: ignore[assignment]
        if isinstance(result, ReasoningResult):
            entity = result.entity or "（未指定標的）"
            time_range = result.time_range or "last_365_days"
            question_summary = _summarize_question(user_question, "macro_impact")
            body = result.formatted_answer
            ctx = RenderContext(
                mode_label="Reasoning",
                entity=entity,
                time_range=time_range,
                question_summary=question_summary,
                body=body,
                source_label="Tavily Search",
                date_label=_today_str(),
            )
            response = _render(ctx)
        else:
            response = "目前無法產生 Reasoning 回應（結果格式不符）。"

    else:
        response = "目前無法產生回應。"

    if _detect_realtime_anchor(user_question):
        response += _REALTIME_WARNING

    return response


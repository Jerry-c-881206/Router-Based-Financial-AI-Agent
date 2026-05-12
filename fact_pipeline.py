"""
Fact Pipeline（SDD v2.0 §5.1）

設計原則：
- 只取 FinMind 結構化資料作為 context
- LLM 只負責格式化（formatting only），不得推論/估算
- FinMind 回傳空值 → 回覆「目前無法取得 {entity} 的 {time_range} 資料」
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from finmind_client import FinMindClient
from query_understanding import QueryUnderstanding
from time_utils import parse_time_range


@dataclass(frozen=True)
class FactResult:
    entity: str
    time_range: str
    task: str
    data: dict[str, Any] | None
    formatted_answer: str


_FORMAT_ONLY_SYSTEM_PROMPT = """你是一位專業的台灣股市理財助理。你只負責把「提供的結構化資料」整理成清楚的條列格式。

規則：
1. 只能使用下方提供的資料，不得自行推論、估算或捏造
2. 若資料為空或無法回答，直接輸出「目前找不到相關資訊」
3. 使用繁體中文、條列式呈現
4. 最後附上資料來源：FinMind API（查詢日期：{query_date}）

【結構化資料】{structured_data}
"""


def _today() -> date:
    return date.today()


def _time_constraint_to_bounds(time_range: str | None) -> tuple[str, date, date]:
    """
    Convert SDD v2.0 time_constraint.range into concrete dates.

    Supported:
    - YYYY-QN / YYYY-MM / YYYY (re-use time_utils.parse_time_range)
    - last_90_days / last_365_days
    - null => treat as last_90_days (Fact default)
    """
    if not time_range:
        tr = "last_90_days"
    else:
        tr = time_range.strip()

    if tr == "last_90_days":
        end = _today()
        start = end - timedelta(days=90)
        return tr, start, end
    if tr == "last_365_days":
        end = _today()
        start = end - timedelta(days=365)
        return tr, start, end

    bounds = parse_time_range(tr)
    return tr, bounds.start_date, bounds.end_date


def _select_metrics(task: str, user_question: str) -> dict[str, bool]:
    """
    Map SDD task -> which FinMind pulls are needed.
    Keep it deterministic; no LLM in this step.
    """
    t = (task or "").strip()
    q = (user_question or "").strip()

    want_eps = False
    want_rev = False
    want_fcf = False
    want_company = False

    if t == "EPS_query":
        want_eps = True
    elif t in ("revenue_query", "month_revenue_query"):
        want_rev = True
    elif t in ("cashflow_query", "fcf_query"):
        want_fcf = True
    elif t == "company_profile":
        want_company = True
    else:
        # Fallback heuristic: infer from question keywords.
        if "EPS" in q.upper() or "每股盈餘" in q:
            want_eps = True
        if "營收" in q or "月營收" in q:
            want_rev = True
        if "現金流" in q or "自由現金流" in q or "FCF" in q.upper():
            want_fcf = True
        if "公司" in q or "產業" in q or "屬於" in q:
            # For entity-description style questions.
            want_company = True

        # If still nothing, provide a minimal set for fact queries.
        if not (want_eps or want_rev or want_fcf or want_company):
            want_eps = True
            want_rev = True

    return {"eps": want_eps, "revenue": want_rev, "fcf": want_fcf, "company": want_company}


def run_fact_pipeline(qu: QueryUnderstanding, *, user_question: str, model: str = "gpt-4o-mini") -> FactResult:
    load_dotenv()

    if not qu.entity:
        return FactResult(
            entity="",
            time_range=qu.time_constraint.range or "last_90_days",
            task=qu.task,
            data=None,
            formatted_answer="請問您想查詢哪一支股票或公司？請提供股票名稱或代號。",
        )

    time_range, start_date, end_date = _time_constraint_to_bounds(qu.time_constraint.range)

    finmind = FinMindClient()
    stock_id = finmind.resolve_stock_id(qu.entity)

    if not finmind.entity_exists(stock_id):
        return FactResult(
            entity=qu.entity,
            time_range=time_range,
            task=qu.task,
            data=None,
            formatted_answer=f"查無「{qu.entity}」的上市股票資料，請確認公司名稱或股票代號是否正確。",
        )

    metrics = _select_metrics(qu.task, user_question)

    data: dict[str, Any] = {
        "source": "FinMind API",
        "entity": qu.entity,
        "stock_id": stock_id,
        "time_range": time_range,
        "task": qu.task,
        "eps": [],
        "month_revenue": [],
        "free_cash_flow": [],
        "company_profile": {},
    }

    if metrics["eps"]:
        data["eps"] = finmind.fetch_eps(stock_id, start_date=start_date, end_date=end_date)
    if metrics["revenue"]:
        data["month_revenue"] = finmind.fetch_monthly_revenue(stock_id, start_date=start_date, end_date=end_date)
    if metrics["fcf"]:
        data["free_cash_flow"] = finmind.fetch_free_cash_flow(stock_id, start_date=start_date, end_date=end_date)
    if metrics.get("company"):
        data["company_profile"] = finmind.fetch_company_profile(stock_id)

    has_any = any(bool(data[k]) for k in ("eps", "month_revenue", "free_cash_flow", "company_profile"))
    if not has_any:
        return FactResult(
            entity=qu.entity,
            time_range=time_range,
            task=qu.task,
            data=data,
            formatted_answer=f"目前無法取得 {qu.entity} 的 {time_range} 資料",
        )

    llm = ChatOpenAI(model=model, temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", _FORMAT_ONLY_SYSTEM_PROMPT)])
    chain = prompt | llm
    resp = chain.invoke(
        {
            "structured_data": data,
            "query_date": _today().isoformat(),
        }
    )
    content = getattr(resp, "content", str(resp))

    return FactResult(
        entity=qu.entity,
        time_range=time_range,
        task=qu.task,
        data=data,
        formatted_answer=content,
    )


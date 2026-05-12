"""
Query Understanding Layer（SDD v2.0 §3）

① Intent Classification
② Entity Extraction
③ Time Constraint Extraction

輸出 `QueryUnderstanding`，供 Execution Planner 使用。
Prompt 依 SDD §7.2「Query Understanding」。
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from time_utils import parse_time_range

# --- SDD §3.2 資料結構 ---------------------------------------------------------

TimeConstraintType = Literal["explicit", "implicit", "none"]
IntentType = Literal["fact", "aggregation", "opinion", "reasoning"]


class TimeConstraint(BaseModel):
    """SDD §3.2：時間約束。"""

    type: TimeConstraintType = Field(
        description="explicit：使用者明確指定時間；implicit：由語意推得；none：未指定且 opinion 等由維度處理"
    )
    range: str | None = Field(
        default=None,
        description="例：2026-Q1、2026-01、last_90_days、last_365_days；opinion 可為 null",
    )


class QueryUnderstanding(BaseModel):
    """SDD §3.2：第一層統一輸出。"""

    intent: IntentType = Field(description="fact / aggregation / opinion / reasoning")
    entity: str | None = Field(
        default=None,
        description="標的公司名稱或股票代號；無法識別時為 null（觸發 Fallback）",
    )
    time_constraint: TimeConstraint
    task: str = Field(
        description="細分行為，例：EPS_query / price_driver / investment_eval / macro_impact"
    )
    validation_error: str | None = Field(
        default=None,
        description="時間合理性驗證失敗時的錯誤訊息；None 表示驗證通過",
    )


# --- SDD §7.2 System Prompt ----------------------------------------------------

QUERY_UNDERSTANDING_SYSTEM_PROMPT = """你是一個金融查詢分析器。請從使用者的問題中提取以下欄位，以 JSON 格式回傳。

intent 定義：
- fact：查詢單一客觀數據（EPS、營收、ROE 等）
  ※ 注意：詢問公司結構、子公司、業務範疇等需要多來源整合的問題，
    即使包含「有哪些」，仍應歸類為 aggregation
- aggregation：詢問原因或多來源歸納，以及公司背景、子公司、業務介紹等需要整合性描述的問題
- opinion：要求評估或判斷（適不適合、值不值得）
- reasoning：假設情境的因果推論（如果…會怎樣）

time_constraint 規則：
- 今年 = 2026
- 未提及時間：fact/aggregation 預設 last_90_days，reasoning 預設 last_365_days
- 若 intent = opinion：time_constraint.type = none，time_constraint.range = null（由各維度自行處理）

task 規則（由 intent + 關鍵字決定，盡量精準）：
- 若 intent = fact：
  - 問「EPS / 每股盈餘 / 獲利 / 盈餘」→ task = "EPS_query"
  - 問「營收 / 月營收 / Revenue」→ task = "revenue_query"
  - 其他 fact（例如 ROE 等單一指標）→ task = "fact_query"
- 若 intent = aggregation：
  - 問「為什麼 / 原因 / 主要因素 / 驅動」→ task = "price_driver"
  - 問「是做什麼的 / 公司簡介 / 屬於哪個產業 / 主要業務」→ task = "company_profile"
  - 其他 aggregation → task = "aggregation_query"
- 若 intent = opinion：
  - 適不適合 / 值不值得 / 好不好 / 評估類 → task = "investment_eval"
- 若 intent = reasoning：
  - 如果…會怎樣 / 可能影響 → task = "macro_impact"

輸出格式（JSON only，無任何前綴）：
{{
  "intent": "fact | aggregation | opinion | reasoning",
  "entity": "公司名稱或代號，無法識別回傳 null",
  "time_constraint": {{
    "type": "explicit | implicit | none",
    "range": "2026-Q1 / last_90_days / null"
  }},
  "task": "EPS_query / revenue_query / price_driver / aggregation_query / investment_eval / macro_impact / company_profile"
}}
"""


def _normalize_entity(entity: str | None) -> str | None:
    if entity is None:
        return None
    s = entity.strip()
    return s if s else None


def _extract_entity_candidate(user_question: str) -> str | None:
    """
    Deterministic entity extraction for common MVP queries.
    Helps when LLM fails to output entity (so pipelines won't fallback).
    """
    q = (user_question or "").strip()
    if not q:
        return None

    # Remove leading markers like '#' and whitespace.
    qn = re.sub(r'^[#\s]+', '', q)

    # Stock code
    m_code = re.search(r'\b\d{4,5}\b', qn)
    if m_code:
        return m_code.group(0)

    # Common stopwords (avoid wrong extraction like "最近/原因" as entity)
    stopwords = {
        "最近",
        "近期",
        "原因",
        "主要因素",
        "驅動",
        "上漲",
        "下跌",
        "股價",
        "營收",
        "EPS",
        "現金流",
        "公司",
        "適不適合",
        "值不值得",
        "好不好",
        "如果",
        "會",
        "可能影響",
        "影響",
    }

    # Capture company name-like Chinese sequence near beginning or before verbs.
    # Examples: 台積電是間公司嗎？ / 台積電最近股價上漲的原因？
    m = re.search(r'(?P<name>[\u4e00-\u9fff]{2,10})(?=(?:是|最近|上漲|下跌|原因|的|嗎|什麼|公司|策略|營收|EPS|現金流))', qn)
    if m:
        name = m.group("name").strip()
        if name and name not in stopwords:
            return name

    # Fallback: first Chinese segment (still guarded by stopwords)
    m2 = re.search(r'[\u4e00-\u9fff]{2,10}', qn)
    if m2:
        name = m2.group(0)
        if name and name not in stopwords:
            return name

    return None


_TIME_MIN_YEAR: int = 1990
_TIME_MAX_DATE: date = date(2026, 4, 30)


def _validate_time_range(tc: TimeConstraint) -> str | None:
    """
    回傳錯誤訊息字串若時間範圍不合理，否則回傳 None。
    - 相對範圍（last_X_days）以今天為基準，永遠合理，略過驗證。
    - opinion（type=none）不驗證。
    - 明確範圍（YYYY-QN / YYYY-MM / YYYY）解析後比對上下限。
    """
    if tc.type == "none" or not tc.range:
        return None

    tr = tc.range.strip()
    if tr.startswith("last_"):
        return None

    try:
        bounds = parse_time_range(tr)
    except ValueError:
        return None

    if bounds.start_date.year < _TIME_MIN_YEAR:
        return (
            f"查詢時間範圍（{tr}）早於系統支援的最早年份（{_TIME_MIN_YEAR} 年），"
            "請重新指定時間範圍。"
        )

    if bounds.end_date > _TIME_MAX_DATE:
        return (
            f"查詢時間範圍（{tr}）超出目前可用資料的日期上限"
            f"（{_TIME_MAX_DATE.strftime('%Y-%m')}），請重新指定時間範圍。"
        )

    return None


def _keyword_intent_override(user_question: str) -> tuple[IntentType, str, TimeConstraint] | None:
    """
    When user asks a clear intent question, LLM sometimes misclassifies.
    This deterministic override reduces the chance everything routes to Fact Pipeline.
    """
    q = (user_question or "").strip()
    if not q:
        return None
    qn = q.replace(" ", "")

    # Opinion
    opinion_keys = ["適不適合", "值不值得", "好不好", "投資適合", "適合投資", "值得投資"]
    if any(k in qn for k in opinion_keys):
        return ("opinion", "investment_eval", TimeConstraint(type="none", range=None))

    # Reasoning
    reasoning_keys = ["如果", "會怎樣", "可能影響", "影響", "會影響", "可能導致", "若", "假設"]
    if any(k in qn for k in reasoning_keys):
        return ("reasoning", "macro_impact", TimeConstraint(type="implicit", range="last_365_days"))

    # Aggregation - price driver（原因/驅動類）
    aggregation_keys = ["為什麼", "原因", "主要因素", "驅動", "上漲原因", "下跌原因", "漲的原因", "跌的原因"]
    if any(k in qn for k in aggregation_keys):
        return ("aggregation", "price_driver", TimeConstraint(type="implicit", range="last_90_days"))

    # Aggregation - listing/enumeration（列舉彙整類，須在 fact_keys_company 前攔截）
    aggregation_listing_keys = ["子公司", "旗下", "關係企業", "集團旗下", "旗下品牌", "旗下業務", "旗下公司"]
    if any(k in qn for k in aggregation_listing_keys):
        return ("aggregation", "aggregation_query", TimeConstraint(type="implicit", range="last_90_days"))

    # Fact (company/profile and metrics)
    fact_keys_eps = ["EPS", "每股盈餘", "獲利", "盈餘"]
    fact_keys_rev = ["營收", "月營收", "revenue"]
    # 移除單獨的「公司」避免過度捕捉（子公司/旗下已於上方攔截）
    fact_keys_company = ["公司是什麼", "是什麼公司", "公司簡介", "屬於哪個產業", "台積電是什麼公司", "公司是做什麼的"]

    if any(k in qn for k in fact_keys_company):
        return ("fact", "company_profile", TimeConstraint(type="implicit", range="last_90_days"))

    if any(k in qn for k in fact_keys_eps):
        return ("fact", "EPS_query", TimeConstraint(type="implicit", range="last_90_days"))
    if any(k in qn for k in fact_keys_rev):
        return ("fact", "revenue_query", TimeConstraint(type="implicit", range="last_90_days"))

    # ✅ 修正：不命中任何規則時回傳 None，保留 LLM 的判斷
    return None


def understand_query(user_question: str, *, model: str = "gpt-4o-mini") -> QueryUnderstanding:
    """
    呼叫 LLM（Structured Output）產生 QueryUnderstanding。
    失敗時回傳保守結果（entity=null），避免下游崩潰。
    """
    load_dotenv()

    llm = ChatOpenAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(
        QueryUnderstanding,
        method="json_schema",
        strict=True,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUERY_UNDERSTANDING_SYSTEM_PROMPT),
            ("human", "{user_question}"),
        ]
    )
    chain = prompt | structured_llm

    try:
        out = chain.invoke({"user_question": user_question})
        if isinstance(out, QueryUnderstanding):
            qu = out
        else:
            qu = QueryUnderstanding.model_validate(out)
    except Exception as e:
        print(f"[query_understanding] LLM call failed: {e}")
        return QueryUnderstanding(
            intent="fact",
            entity=None,
            time_constraint=TimeConstraint(type="implicit", range="last_90_days"),
            task="fact_query",
        )
    
    override = _keyword_intent_override(user_question)
    if override is not None:
        intent, task, tc = override
        qu = QueryUnderstanding(
            intent=intent,
            entity=qu.entity,
            time_constraint=tc,
            task=task,
        )

    # If the LLM failed to extract entity, try deterministic extraction.
    if not qu.entity:
        candidate = _extract_entity_candidate(user_question)
        if candidate:
            qu = QueryUnderstanding(
                intent=qu.intent,
                entity=candidate,
                time_constraint=qu.time_constraint,
                task=qu.task,
            )

    final_tc = qu.time_constraint
    time_error = _validate_time_range(final_tc)

    return QueryUnderstanding(
        intent=qu.intent,
        entity=_normalize_entity(qu.entity),
        time_constraint=final_tc,
        task=qu.task.strip() if qu.task else "unknown",
        validation_error=time_error,
    )
    
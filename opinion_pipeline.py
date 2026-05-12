"""
Opinion Pipeline（SDD v2.0 §5.3）

設計原則：
- 拆成四個評估維度：
  【基本面】【成長動能】【風險】【市場觀點】
- 各維度獨立 Query 生成 + Tavily 搜尋 + Filter Layer
- 最後由 GPT‑4o 做結構化分析，並加上投資建議 Guardrail
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from aggregation_pipeline import (
    _dedup_and_rank_by_similarity,
    _filter_by_source,
    _filter_by_time,
)
from query_understanding import QueryUnderstanding
from tavily_client import search_news


@dataclass(frozen=True)
class OpinionResult:
    entity: str
    time_ranges: Dict[str, str]
    sections: Dict[str, str]
    formatted_answer: str


_DIMENSIONS = ["basic", "growth", "risk", "market"]


def _today() -> date:
    return date.today()


def _default_time_ranges() -> Dict[str, str]:
    """
    SDD §5.3 表格：
    - 基本面：最近 3 個月 → last_90_days
    - 成長動能：最近 6 個月 → last_180_days
    - 風險：最近 6 個月 → last_180_days
    - 市場觀點：最近 1 個月 → last_30_days
    """
    return {
        "basic": "last_90_days",
        "growth": "last_180_days",
        "risk": "last_180_days",
        "market": "last_30_days",
    }


def _relative_range_to_bounds(code: str) -> tuple[date, date]:
    end = _today()
    if code == "last_30_days":
        return end - timedelta(days=30), end
    if code == "last_180_days":
        return end - timedelta(days=180), end
    # default last_90_days
    return end - timedelta(days=90), end


def _dimension_queries(entity: str, user_question: str) -> Dict[str, str]:
    """
    根據 SDD §5.3 所給範例模板產生各維度 Query。
    """
    base = entity.strip()
    return {
        "basic": f"{base} 財報 營收 獲利 基本面",
        "growth": f"{base} 成長動能 市場擴張 新產品",
        "risk": f"{base} 風險 法規 競爭 下行風險",
        "market": f"{base} 分析師 評級 展望 市場觀點",
    }


_OPINION_SYSTEM_PROMPT = """你是一位嚴謹的台灣股市分析師，負責針對單一標的進行多維度的結構化分析。

維度：
- 【基本面】：財報、營收、獲利品質等
- 【成長動能】：成長趨勢、新業務、產能/市占擴張等
- 【風險】：法規、競爭、公司治理、財務槓桿等
- 【市場觀點】：分析師評級、法人看法、整體市場情緒等

規則：
1. 嚴格只使用各維度提供的搜尋資料，不得自行捏造或推測。
2. 各維度輸出 2–4 點重點，避免重複，必要時可引用具體數字或事件。
3. 不可跨維度混用來源（每個重點要能對應回該維度資料）。
4. 若某維度幾乎沒有資料，請明確說明「目前缺乏足夠公開資訊」。
5. 最後必須附上： 以上為資訊整理，不構成投資建議。
6. 不得出現「建議買入/賣出/持有」、「一定會漲/跌」等明確投資指令。

【基本面資料】
{basic_docs}

【成長動能資料】
{growth_docs}

【風險資料】
{risk_docs}

【市場觀點資料】
{market_docs}
"""


def run_opinion_pipeline(
    qu: QueryUnderstanding,
    *,
    user_question: str,
    model: str = "gpt-4o",
) -> OpinionResult:
    load_dotenv()

    if not qu.entity:
        return OpinionResult(
            entity="",
            time_ranges=_default_time_ranges(),
            sections={},
            formatted_answer="請問您想查詢哪一支股票或公司？請提供股票名稱或代號。",
        )

    entity = qu.entity.strip()
    time_ranges = _default_time_ranges()
    dim_queries = _dimension_queries(entity, user_question)

    dim_docs: Dict[str, List[Dict[str, Any]]] = {k: [] for k in _DIMENSIONS}

    # 各維度：Tavily Search + Filter Layer
    for dim in _DIMENSIONS:
        tr_code = time_ranges[dim]
        q = dim_queries[dim]
        search = search_news(q, max_results=5, time_range=tr_code)
        raw_docs: List[Dict[str, Any]] = [dict(r) for r in search.get("results", [])]

        start, end = _relative_range_to_bounds(tr_code)
        docs_time = _filter_by_time(raw_docs, (start, end))
        docs_source = _filter_by_source(docs_time)
        docs_filtered = _dedup_and_rank_by_similarity(docs_source, [q], max_docs=10)
        dim_docs[dim] = docs_filtered

    # 若四個維度都沒有任何文件
    if all(not v for v in dim_docs.values()):
        return OpinionResult(
            entity=entity,
            time_ranges=time_ranges,
            sections={},
            formatted_answer="目前找不到足夠公開資訊來進行評估。",
        )

    # GPT‑4o 結構化分析 + Guardrail
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _OPINION_SYSTEM_PROMPT),
        ]
    )
    chain = prompt | llm
    resp = chain.invoke(
        {
            "basic_docs": json.dumps(dim_docs["basic"], ensure_ascii=False, indent=2),
            "growth_docs": json.dumps(dim_docs["growth"], ensure_ascii=False, indent=2),
            "risk_docs": json.dumps(dim_docs["risk"], ensure_ascii=False, indent=2),
            "market_docs": json.dumps(dim_docs["market"], ensure_ascii=False, indent=2),
        }
    )
    content = getattr(resp, "content", str(resp))

    return OpinionResult(
        entity=entity,
        time_ranges=time_ranges,
        sections={},  # 若未來要結構化切開，可在此再做二次 parsing，目前先以 formatted_answer 為主。
        formatted_answer=content,
    )


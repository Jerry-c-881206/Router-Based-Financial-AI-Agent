"""
Reasoning Pipeline（SDD v2.0 §5.4）

流程（SDD §5.4）：
Step 1: Scenario Extraction
Step 2: Generate Causal Chain（GPT-4o）
Step 3: Generate up to 3 Tavily queries（JSON only）
Step 4: Grounding Check（比對因果鏈與檢索資料）
Step 5: Refine Answer（輸出 SDD §6.3 格式）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from query_understanding import QueryUnderstanding
from tavily_client import search_news
from time_utils import parse_time_range

try:
    from aggregation_pipeline import _filter_by_source, _filter_by_time
except Exception:  # pragma: no cover
    _filter_by_source = None  # type: ignore[assignment]
    _filter_by_time = None  # type: ignore[assignment]


_REASONING_SYSTEM_PROMPT = """你是一位嚴謹的台灣股市分析師。你會做因果推論，但必須「grounding」：

規則：
1. 嚴禁直接提供投資建議（不得出現「建議買入/賣出/持有」等指令）
2. 不得出現「一定會漲/一定會跌」等絕對性預測
3. 若資料不足，在節點說明中明確標注「【模型推斷】」
4. 嚴禁捏造數據；推論部分必須可被搜尋資料支持或標注【模型推斷】
5. 使用繁體中文輸出
"""


class ScenarioExtraction(BaseModel):
    macro_variable: str = Field(description="宏觀變數/外部情境（例如：美國升息、地緣政治、供需變化）")
    causal_direction: str = Field(description="預期傳導方向：上行/下行/中性，但用文字描述，不要絕對預測")
    entity: str = Field(description="公司/標的（來自 Query Understanding 的 entity）")
    time_hint: str | None = Field(default=None, description="時間線索（例如：最近 12 個月/2025-Q4）")


class CausalNode(BaseModel):
    node: str = Field(description="因果鏈節點名稱（例如：資金成本上升/估值收縮/需求變化）")
    from_to: str = Field(description="節點與相鄰節點的關係描述（例如 A → B）")
    rationale: str = Field(description="為何會發生（推論），但不得捏造數據")


class CausalChainDraft(BaseModel):
    causal_chain_text: str = Field(
        description="完整因果推論鏈（用「→」串接，涵蓋 宏觀變數→傳導機制→公司影響→股價影響）"
    )
    nodes: List[CausalNode] = Field(description="最多 4 個節點，後續會挑最多 3 個節點去驗證")


class CausalChainQueryGenerator(BaseModel):
    queries: List[str] = Field(description="最多 3 個 Tavily 搜尋字串")


class GroundedNodeStatus(BaseModel):
    node: str = Field(description="節點名稱（對應 causal_chain 的 node）")
    status: Literal["supported", "indirect_reference", "unsupported", "model_inferred"] = Field(
        description=(
            "supported：搜尋資料直接支撐該節點（資料與假設直接相關）；"
            "indirect_reference：資料僅間接相關，不足以直接支撐；"
            "unsupported：與資料矛盾；"
            "model_inferred：完全無支撐"
        )
    )
    explanation: str = Field(description="節點驗證結果說明（有/無支撐 + 短原因）")


class GroundedChain(BaseModel):
    causal_chain_corrected: str = Field(description="修正後因果鏈（可在關鍵節點後加註：有/無支撐/模型推斷）")
    nodes: List[GroundedNodeStatus] = Field(description="每個節點的支撐狀態")


class ReasoningResult(BaseModel):
    entity: str
    time_range: str
    causal_chain_initial: str | None = None
    causal_chain_grounded: str
    node_statuses: List[GroundedNodeStatus]
    formatted_answer: str


_SCENARIO_PROMPT = """Step 1: Scenario Extraction

使用者問題描述了一個「假設情境」。請抽取：
- 宏觀變數（macro_variable）
- 對該標的的因果方向（causal_direction）：用「偏正向/偏負向/不確定」等，不要下絕對結論
- 標的 entity
- time_hint（若題目提到時間，填入；否則填 null）

只輸出 JSON。

【使用者問題】{user_question}
【entity】{entity}
"""


_CAUSAL_CHAIN_PROMPT = """Step 2: Generate Causal Chain

根據以下 scenario，生成因果推論鏈：
宏觀變數 → 傳導機制 → 公司影響 → 股價影響

請同時輸出最多 4 個節點 nodes，每個節點包含 node / from_to / rationale。

規則：
- rationale 只能描述機制，不得捏造數字或宣稱資料已證實
- causal_chain_text 用「→」串接文字
- 若無法確定，填入不確定但仍可驗證的機制描述

只輸出 JSON。

【Scenario】{scenario_json}
"""


_QUERY_GENERATOR_PROMPT = """Step 3: Causal Chain Query Generator

你會從因果鏈中挑出「最多 3 個」最需要外部資料驗證的節點（nodes），
並為每個節點生成 Tavily 搜尋字串。

搜尋字串要求：
- 包含節點核心概念關鍵字
- 附加時間範圍（time_range）
- 繁體中文或英文皆可

只輸出 JSON（無前綴說明）：
{{
  "queries": ["...", "...", "..."]
}}

【因果鏈】{causal_chain_text}
【節點】{nodes_json}
【time_range】{time_range}
"""


_GROUNDING_CHECK_PROMPT = """Step 4: Grounding Check（比對）

你會比對以下：
- 初始因果鏈與節點（是否為可驗證推論）
- Tavily 搜尋到的支撐資料（retrieved_docs）

輸出：
- 修正後的因果鏈（因果鏈可在節點處標註 supported/indirect_reference/unsupported/model_inferred）
- 每個節點的狀態（status 欄位）
- 對每個節點說明「為何判定」（explanation 欄位）

判定規則：
1. supported：資料明確、直接支撐該節點的核心假設（數字/事件/因果關係與節點直接吻合）
2. indirect_reference：資料提及相關背景或周邊因素，但未直接驗證節點假設本身；
   間接資料不得標注為 supported，只能標注為 indirect_reference
3. unsupported：資料與節點假設矛盾，或呈現相反趨勢
4. model_inferred：完全找不到任何相關資料，純粹依賴模型知識推論
5. 禁止捏造資料內容；判定依據必須來自 retrieved_docs 或坦誠標注【模型推斷】

只輸出 JSON。

【初始因果鏈】{causal_chain_text}
【搜尋到的支撐資料】{retrieved_docs}
"""


_REFINE_ANSWER_PROMPT = """Step 5: Refine Answer

請根據以下「修正後因果鏈」與「節點驗證狀態」產生最終回應，格式需符合 SDD §6.3：

【因果推論鏈】
{causal_chain_corrected}

【節點說明】
• {{node}}: {{explanation}}（有 / 無資料支撐）
若 status 為 model_inferred，則以「【模型推斷】」結尾。

最後不需要額外投資建議文字，但需避免所有買賣指令與絕對性預測。

只輸出純文字（不用 JSON）。

【節點驗證狀態】{nodes_status_json}
"""


def _default_time_range(qu: QueryUnderstanding) -> str:
    return qu.time_constraint.range or "last_365_days"


def _time_bounds_if_possible(time_range: str) -> tuple[date, date] | None:
    if time_range in {"last_365_days", "last_90_days", "last_30_days", "last_180_days"}:
        return None
    try:
        b = parse_time_range(time_range)
        return b.start_date, b.end_date
    except Exception:
        return None


def _dedup_by_url(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for d in docs:
        url = d.get("url") or d.get("source_url")
        if url:
            if url in seen:
                continue
            seen.add(url)
        out.append(d)
    return out


def run_reasoning_pipeline(
    qu: QueryUnderstanding,
    *,
    user_question: str,
    model: str = "gpt-4o",
) -> ReasoningResult:
    load_dotenv()

    if not qu.entity:
        return ReasoningResult(
            entity="",
            time_range=_default_time_range(qu),
            causal_chain_initial=None,
            causal_chain_grounded="",
            node_statuses=[],
            formatted_answer="請問您想查詢哪一支股票或公司？請提供股票名稱或代號。",
        )

    entity = qu.entity.strip()
    time_range = _default_time_range(qu)

    llm = ChatOpenAI(model=model, temperature=0)

    # Step 1: Scenario Extraction
    scenario_prompt = ChatPromptTemplate.from_messages([("system", _REASONING_SYSTEM_PROMPT), ("human", _SCENARIO_PROMPT)])
    scenario_chain = scenario_prompt | llm.with_structured_output(ScenarioExtraction, method="json_schema", strict=True)
    scenario_resp = scenario_chain.invoke({"user_question": user_question, "entity": entity})
    scenario = scenario_resp if isinstance(scenario_resp, ScenarioExtraction) else ScenarioExtraction.model_validate(scenario_resp)

    # Step 2: Causal chain draft
    causal_prompt = ChatPromptTemplate.from_messages([("system", _REASONING_SYSTEM_PROMPT), ("human", _CAUSAL_CHAIN_PROMPT)])
    causal_chain = causal_prompt | llm.with_structured_output(CausalChainDraft, method="json_schema", strict=True)
    draft_resp = causal_chain.invoke({"scenario_json": scenario.model_dump(), "user_question": user_question})
    draft = draft_resp if isinstance(draft_resp, CausalChainDraft) else CausalChainDraft.model_validate(draft_resp)

    # Step 3: Generate search queries (up to 3)
    query_gen_prompt = ChatPromptTemplate.from_messages([("system", _REASONING_SYSTEM_PROMPT), ("human", _QUERY_GENERATOR_PROMPT)])
    query_gen_chain = query_gen_prompt | llm.with_structured_output(CausalChainQueryGenerator, method="json_schema", strict=True)
    qg_resp = query_gen_chain.invoke(
        {"causal_chain_text": draft.causal_chain_text, "nodes_json": json.dumps([n.model_dump() for n in draft.nodes], ensure_ascii=False), "time_range": time_range}
    )
    qg = qg_resp if isinstance(qg_resp, CausalChainQueryGenerator) else CausalChainQueryGenerator.model_validate(qg_resp)
    queries = [q for q in qg.queries if q and q.strip()]
    queries = queries[:3]

    # Step 4: Retrieve evidence
    retrieved_docs: List[Dict[str, Any]] = []
    for q in queries:
        search = search_news(f"{entity} {q}", max_results=5, time_range=time_range)
        retrieved_docs.extend(search.get("results", []))

    retrieved_docs = [dict(d) for d in retrieved_docs if isinstance(d, dict)]
    retrieved_docs = _dedup_by_url(retrieved_docs)

    # Optional filter by time and source to reduce noise
    bounds = _time_bounds_if_possible(time_range)
    if _filter_by_time is not None and bounds is not None:
        retrieved_docs = _filter_by_time(retrieved_docs, bounds)
    if _filter_by_source is not None:
        retrieved_docs = _filter_by_source(retrieved_docs)

    # Step 4: Grounding check
    grounding_prompt = ChatPromptTemplate.from_messages([("system", _REASONING_SYSTEM_PROMPT), ("human", _GROUNDING_CHECK_PROMPT)])
    grounding_chain = grounding_prompt | llm.with_structured_output(GroundedChain, method="json_schema", strict=True)
    grounded_resp = grounding_chain.invoke(
        {
            "causal_chain_text": draft.causal_chain_text,
            "retrieved_docs": json.dumps(retrieved_docs[:15], ensure_ascii=False, indent=2),
        }
    )
    grounded = grounded_resp if isinstance(grounded_resp, GroundedChain) else GroundedChain.model_validate(grounded_resp)

    # Step 5: Refine final answer (pure text)
    refine_prompt = ChatPromptTemplate.from_messages([("system", _REASONING_SYSTEM_PROMPT), ("human", _REFINE_ANSWER_PROMPT)])
    refine_chain = refine_prompt | llm
    refined = refine_chain.invoke(
        {
            "causal_chain_corrected": grounded.causal_chain_corrected,
            "nodes_status_json": json.dumps([n.model_dump() for n in grounded.nodes], ensure_ascii=False, indent=2),
        }
    )
    refined_text = getattr(refined, "content", str(refined)).strip()
    if not refined_text:
        refined_text = "目前找不到相關資訊"

    return ReasoningResult(
        entity=entity,
        time_range=time_range,
        causal_chain_initial=draft.causal_chain_text,
        causal_chain_grounded=grounded.causal_chain_corrected,
        node_statuses=grounded.nodes,
        formatted_answer=refined_text,
    )


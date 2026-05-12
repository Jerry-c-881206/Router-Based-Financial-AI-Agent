"""
Aggregation Pipeline（SDD v2.0 §5.2）

設計原則：
- 從多筆非結構化資料中抽取「共同原因」
- 強調：時間過濾 → 來源過濾 → 去重 → 相關性篩選
- LLM 只做「原因歸納」，不逐篇摘要
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from query_understanding import QueryUnderstanding
from tavily_client import search_news
from time_utils import parse_time_range

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AggregationResult:
    entity: str
    time_range: str
    task: str
    sub_questions: list[str]
    filtered_docs: list[dict[str, Any]]
    formatted_answer: str


_AGG_SUMMARIZER_SYSTEM_PROMPT = """【System Prompt — Aggregation Summarizer】

你是一位台灣股市分析師。請根據以下資料，歸納出最多 3 個關鍵原因。

規則：
1. 只使用提供的資料，不得自行推測
2. 找出多篇資料共同出現的核心原因
3. 合併相似資訊，不逐篇摘要
4. 用因果關係（A → B）描述每個原因
5. 若資料超過時效，標注【資料較舊，請自行確認】

【子問題一】{sub_question_1}
【子問題二】{sub_question_2}
【搜尋資料】{filtered_docs}
"""


_SUBQUESTION_SYSTEM_PROMPT = """你是一個金融問題分解器。
使用者的原始問題可能是在問「股價漲跌的原因」或是最新消息、策略、法說會等其他需要多來源歸納的問題。

請將下列問題拆成 2 個子問題，幫助下游搜尋時更精準：
- 子問題一偏向「基本面 / 公司層面」原因
- 子問題二偏向「市場 / 宏觀 / 產業 / 資金面 / 最新消息 / 策略 / 法說會」原因

輸出格式（JSON only）：
{{
  "sub_question_1": "...",
  "sub_question_2": "..."
}}

只輸出 JSON，無任何前綴說明。

【原始問題】{user_question}
"""


def _time_range_from_qu(qu: QueryUnderstanding) -> str:
    r = qu.time_constraint.range
    if not r:
        # Aggregation 預設 last_90_days（SDD §7.2）
        return "last_90_days"
    return r


def _bounds_from_time_range(time_range: str) -> tuple[date, date] | None:
    """
    For Aggregation we only need a rough cutoff; if a normalized range
    (YYYY-QN / YYYY-MM / YYYY) is provided, reuse `parse_time_range`.
    For relative tokens (last_90_days / last_365_days) we just return None
    and rely on Tavily recency.
    """
    if time_range in {"last_90_days", "last_365_days"}:
        return None
    b = parse_time_range(time_range)
    return b.start_date, b.end_date


def _parse_domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc
        return netloc.lower()
    except Exception:
        return None


_SOURCE_WHITELIST = {
    "www.cnyes.com",
    "cnyes.com",
    "udn.com",
    "money.udn.com",
    "www.chinatimes.com",
    "ctee.com.tw",
    "www.ctee.com.tw",
    "www.reuters.com",
    "www.bloomberg.com",
    "www.wsj.com",
}


def _filter_by_time(docs: Iterable[dict[str, Any]], bounds: tuple[date, date] | None) -> list[dict[str, Any]]:
    if bounds is None:
        return list(docs)
    start, end = bounds

    def ok(d: dict[str, Any]) -> bool:
        dt = d.get("published_date") or d.get("date")
        if not dt:
            return True
        from datetime import datetime

        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
            try:
                parsed = datetime.strptime(str(dt)[:19], fmt).date()
                return start <= parsed <= end
            except Exception:
                continue
        return True

    return [d for d in docs if ok(d)]


def _filter_by_source(docs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in docs:
        url = d.get("url") or d.get("source_url")
        domain = _parse_domain(url)
        if domain and domain in _SOURCE_WHITELIST:
            out.append(d)
        else:
            # 若無 domain，保守保留，交由後續 embedding 過濾
            out.append(d)
    return out


def _embed_contents(texts: list[str]) -> Any:
    if SentenceTransformer is None or np is None:
        return None
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_numpy=True)


def _dedup_and_rank_by_similarity(
    docs: list[dict[str, Any]],
    queries: list[str],
    max_docs: int = 12,
    sim_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """
    去重 + 相關性篩選（SDD §5.2 Filter Layer ③④）。
    若無 sentence-transformers，則直接截斷前 max_docs 筆。
    """
    if not docs:
        return []

    if SentenceTransformer is None or np is None:
        return docs[:max_docs]

    contents = [str(d.get("content") or d.get("snippet") or "") for d in docs]
    doc_emb = _embed_contents(contents)
    if doc_emb is None:
        return docs[:max_docs]

    # Query embedding（合併兩個子問題）
    q_emb = _embed_contents(["\n".join(queries)])
    if q_emb is None:
        return docs[:max_docs]

    doc_emb = np.array(doc_emb)
    q_vec = np.array(q_emb[0])

    # Cosine similarity
    doc_norm = np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8
    q_norm = np.linalg.norm(q_vec) + 1e-8
    sims = (doc_emb @ q_vec) / (doc_norm[:, 0] * q_norm)

    # 排序 + 去重
    idx_sorted = list(np.argsort(-sims))
    selected: list[int] = []
    for idx in idx_sorted:
        if len(selected) >= max_docs:
            break
        # 與已選文件的相似度，若太高則視為重複
        if selected:
            sel_vecs = doc_emb[selected]
            sel_norm = np.linalg.norm(sel_vecs, axis=1, keepdims=True) + 1e-8
            sim_to_sel = (sel_vecs @ doc_emb[idx]) / (sel_norm[:, 0] * (np.linalg.norm(doc_emb[idx]) + 1e-8))
            if float(sim_to_sel.max()) >= sim_threshold:
                continue
        selected.append(int(idx))

    # 按原排序回傳
    return [docs[i] for i in selected]


def _generate_sub_questions(user_question: str, *, model: str = "gpt-4o-mini") -> tuple[str, str]:
    load_dotenv()
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SUBQUESTION_SYSTEM_PROMPT),
        ]
    )
    chain = prompt | llm
    resp = chain.invoke({"user_question": user_question})
    text = getattr(resp, "content", str(resp))
    try:
        data = json.loads(text)
        q1 = str(data.get("sub_question_1") or "").strip()
        q2 = str(data.get("sub_question_2") or "").strip()
        if not q1:
            q1 = user_question
        if not q2:
            q2 = user_question
        return q1, q2
    except Exception:
        return user_question, user_question


def run_aggregation_pipeline(
    qu: QueryUnderstanding,
    *,
    user_question: str,
    model: str = "gpt-4o-mini",
) -> AggregationResult:
    load_dotenv()

    if not qu.entity:
        return AggregationResult(
            entity="",
            time_range=_time_range_from_qu(qu),
            task=qu.task,
            sub_questions=[],
            filtered_docs=[],
            formatted_answer="請問您想查詢哪一支股票或公司？請提供股票名稱或代號。",
        )

    time_range = _time_range_from_qu(qu)
    bounds = _bounds_from_time_range(time_range)

    sub_q1, sub_q2 = _generate_sub_questions(user_question, model=model)

    # Tavily Search for two sub-questions
    docs_all: list[dict[str, Any]] = []
    for q in (sub_q1, sub_q2):
        search = search_news(f"{qu.entity} {q}", max_results=8, time_range=time_range)
        for r in search.get("results", []):
            d = dict(r)
            d.setdefault("query_used", q)
            docs_all.append(d)

    # Filter Layer: 時間 → 來源 → 去重 + 相關性
    docs_time = _filter_by_time(docs_all, bounds)
    docs_source = _filter_by_source(docs_time)
    docs_filtered = _dedup_and_rank_by_similarity(docs_source, [sub_q1, sub_q2], max_docs=12)

    if not docs_filtered:
        return AggregationResult(
            entity=qu.entity,
            time_range=time_range,
            task=qu.task,
            sub_questions=[sub_q1, sub_q2],
            filtered_docs=[],
            formatted_answer="目前找不到相關資訊",
        )

    # Summarizer
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _AGG_SUMMARIZER_SYSTEM_PROMPT),
        ]
    )
    chain = prompt | llm
    resp = chain.invoke(
        {
            "sub_question_1": sub_q1,
            "sub_question_2": sub_q2,
            "filtered_docs": json.dumps(docs_filtered, ensure_ascii=False, indent=2),
        }
    )
    content = getattr(resp, "content", str(resp))

    return AggregationResult(
        entity=qu.entity,
        time_range=time_range,
        task=qu.task,
        sub_questions=[sub_q1, sub_q2],
        filtered_docs=docs_filtered,
        formatted_answer=content,
    )


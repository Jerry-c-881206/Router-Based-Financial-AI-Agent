from __future__ import annotations

import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults


def _coerce_result(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    # Sometimes Tavily returns objects/documents with attributes.
    out: dict[str, Any] = {}
    for k in ("title", "url", "content", "published_date", "score"):
        if hasattr(item, k):
            out[k] = getattr(item, k)
    # Fallback: string representation.
    if not out:
        out["content"] = str(item)
    return out


def search_news(query: str, *, max_results: int = 5, time_range: str | None = None) -> dict[str, Any]:
    """
    SDD §6.3: 搜尋結果取前 5 筆，交由 LLM 進行摘要。
    """
    load_dotenv()
    try:
        tool = TavilySearchResults(max_results=max_results)
        raw = tool.invoke({"query": query})
    except Exception as e:
        return {
            "source": "Tavily Search",
            "query": query,
            "search_date": datetime.date.today().isoformat(),
            "time_range": time_range,
            "results": [],
            "error": str(e),
        }

    if raw is None:
        results: list[dict[str, Any]] = []
    elif isinstance(raw, list):
        results = [_coerce_result(x) for x in raw]
    else:
        results = [_coerce_result(raw)]

    return {
        "source": "Tavily Search",
        "query": query,
        # Mainly for the final "資料來源標注" section.
        "search_date": datetime.date.today().isoformat(),
        "time_range": time_range,
        "results": results[:max_results],
    }


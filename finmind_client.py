from __future__ import annotations

import os
import re
from datetime import date, timedelta
from typing import Any

import requests
from dotenv import load_dotenv


class FinMindClient:
    """
    FinMind REST client (v4).

    FinMind API v4:
    - Base URL: https://api.finmindtrade.com/api/v4
    - Data endpoint: GET /data with params:
      dataset, (data_id), start_date, end_date
    """

    # Simple in-memory cache to avoid repeated TaiwanStockInfo calls.
    _stock_id_to_info: dict[str, dict[str, Any]] = {}
    _stock_name_to_id: dict[str, str] = {}
    _stock_info_loaded: bool = False

    def __init__(self, *, token: str | None = None, base_url: str = "https://api.finmindtrade.com/api/v4"):
        load_dotenv()
        self.base_url = base_url.rstrip("/")
        self.data_url = f"{self.base_url}/data"
        self.token = token or os.getenv("FINMIND_API_TOKEN") or os.getenv("FINMIND_TOKEN")

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _fetch(self, *, dataset: str, data_id: str | None = None, start_date: date | None = None, end_date: date | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"dataset": dataset}
        if data_id is not None:
            params["data_id"] = data_id
        if start_date is not None:
            params["start_date"] = start_date.isoformat()
        if end_date is not None:
            params["end_date"] = end_date.isoformat()

        resp = requests.get(self.data_url, headers=self._headers(), params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data")
        if isinstance(data, list):
            return data
        return []

    def _normalize_stock_name(self, name: str) -> str:
        s = (name or "").strip()
        # Remove common suffixes if LLM includes them.
        s = re.sub(r"(股份有限公司|有限公司|股份|公司)$", "", s)
        return s.strip()

    def _load_stock_info_cache(self, *, today: date) -> None:
        if self.__class__._stock_info_loaded:
            return

        # `/data` often expects start/end dates.
        # TaiwanStockInfo documentation lists only: dataset=TaiwanStockInfo (no explicit date params),
        # so we try without dates first for compatibility with v4.
        try:
            rows = self._fetch(dataset="TaiwanStockInfo")
        except Exception:
            # Fallback: try with a bounded window.
            start = today - timedelta(days=180)
            rows = self._fetch(dataset="TaiwanStockInfo", start_date=start, end_date=today)

        id_to_info: dict[str, dict[str, Any]] = {}
        name_to_id: dict[str, str] = {}
        for row in rows:
            sid = str(row.get("stock_id", "")).strip()
            if not sid:
                continue
            stock_name = self._normalize_stock_name(str(row.get("stock_name", "")).strip())
            info = {
                "stock_id": sid,
                "stock_name": stock_name or row.get("stock_name"),
                "industry_category": row.get("industry_category"),
                "type": row.get("type"),
                "date": row.get("date"),
            }
            id_to_info[sid] = info
            if stock_name:
                name_to_id[stock_name] = sid

        self.__class__._stock_id_to_info = id_to_info
        self.__class__._stock_name_to_id = name_to_id
        self.__class__._stock_info_loaded = True

    def resolve_stock_id(self, entity: str, *, today: date | None = None) -> str:
        """
        MVP 股票代號解析：
        - 若 entity 看起來像 4~5 位數字 => 直接視為 stock_id
        - 否則查 `TaiwanStockInfo`，比對 `stock_name`
        """
        if today is None:
            today = date.today()

        raw_entity = (entity or "").strip()
        entity_norm = self._normalize_stock_name(raw_entity)
        if re.search(r"\d{4,5}", raw_entity):
            m = re.search(r"\d{4,5}", raw_entity)
            if m:
                return m.group(0)

        try:
            self._load_stock_info_cache(today=today)
        except Exception:
            # If token/network failed, avoid crashing.
            return raw_entity

        # Direct name match
        if entity_norm in self.__class__._stock_name_to_id:
            return self.__class__._stock_name_to_id[entity_norm]

        # Fuzzy match: LLM may output "台積電是間公司" or similar.
        # If the normalized stock_name is contained within entity_norm (or vice versa), treat as match.
        for stock_name, sid in self.__class__._stock_name_to_id.items():
            if stock_name and (stock_name in entity_norm or entity_norm in stock_name):
                return sid

        return raw_entity

    def entity_exists(self, stock_id: str, *, today: date | None = None) -> bool:
        """
        Check whether stock_id is listed in TaiwanStockInfo.
        Returns True on network failure to avoid false negatives.
        """
        if today is None:
            today = date.today()
        try:
            self._load_stock_info_cache(today=today)
        except Exception:
            return True
        return str(stock_id).strip() in self.__class__._stock_id_to_info

    def fetch_company_profile(self, stock_id: str, *, today: date | None = None) -> dict[str, Any]:
        """
        台股「公司基本資料」：使用 TaiwanStockInfo 結構化資訊（SDD Fact 支援 company_profile 任務）。
        """
        if today is None:
            today = date.today()
        try:
            self._load_stock_info_cache(today=today)
        except Exception:
            return {}

        info = self.__class__._stock_id_to_info.get(str(stock_id).strip(), {})
        return info if isinstance(info, dict) else {}

    def fetch_eps(self, stock_id: str, *, start_date: date, end_date: date, limit: int = 6) -> list[dict[str, Any]]:
        """
        EPS 來源：
        - TaiwanStockFinancialStatements (綜合損益表)
        - 篩選 type == 'EPS'
        """
        try:
            rows = self._fetch(
                dataset="TaiwanStockFinancialStatements",
                data_id=stock_id,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception:
            return []

        eps_rows = [
            r
            for r in rows
            if str(r.get("type", "")).strip().upper() == "EPS"
            or "基本每股盈餘" in str(r.get("origin_name", ""))
        ]
        eps_rows.sort(key=lambda r: str(r.get("date", "")))
        return eps_rows[-limit:]

    def fetch_monthly_revenue(
        self, stock_id: str, *, start_date: date, end_date: date, limit: int = 6
    ) -> list[dict[str, Any]]:
        """
        月營收：
        - TaiwanStockMonthRevenue (月營收表)
        """
        try:
            rows = self._fetch(
                dataset="TaiwanStockMonthRevenue",
                data_id=stock_id,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception:
            return []

        rows.sort(key=lambda r: str(r.get("date", "")))
        # Callers (e.g. fact pipeline) expect keys: `revenue`, `revenue_year`, `revenue_month`.
        return rows[-limit:]

    def fetch_free_cash_flow(
        self, stock_id: str, *, start_date: date, end_date: date, limit: int = 3
    ) -> list[dict[str, Any]]:
        """
        自由現金流（FCF）：
        無專用 FreeCashFlow dataset 時，以現金流量表計算：
        FCF = 營業活動現金流 - 資本支出（CAPEX）
        """
        try:
            rows = self._fetch(
                dataset="TaiwanStockCashFlowsStatement",
                data_id=stock_id,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception:
            return []

        # Heuristics on `type`/`origin_name` based on typical FinMind line items.
        def score_operating(r: dict[str, Any]) -> int:
            t = str(r.get("type", ""))
            o = str(r.get("origin_name", ""))
            s = 0
            if "CashFlowsfromOperatingActivities" in t:
                s += 10
            if "營業活動" in o:
                s += 8
            if re.search(r"operat", t, re.IGNORECASE):
                s += 4
            if re.search(r"營業.*現金", o):
                s += 3
            return s

        def score_capex(r: dict[str, Any]) -> int:
            t = str(r.get("type", ""))
            o = str(r.get("origin_name", ""))
            s = 0
            if "CapitalExpenditures" in t:
                s += 10
            if "資本支出" in o:
                s += 9
            if re.search(r"capex", t, re.IGNORECASE):
                s += 4
            if re.search(r"不動產|廠房|設備|投資", o):
                s += 2
            return s

        direct_fcf = [
            r
            for r in rows
            if re.search(r"自由現金流", str(r.get("origin_name", "")))
            or re.search(r"free.*cash", str(r.get("type", "")), re.IGNORECASE)
        ]
        if direct_fcf:
            direct_fcf.sort(key=lambda r: str(r.get("date", "")))
            last = direct_fcf[-limit:]
            return [
                {
                    "date": r.get("date"),
                    "value": r.get("value"),
                    "origin_name": r.get("origin_name"),
                    "type": r.get("type"),
                    "computed": False,
                }
                for r in last
            ]

        # Otherwise compute from operating and capex candidates.
        operating = [r for r in rows if score_operating(r) > 0]
        capex = [r for r in rows if score_capex(r) > 0]
        if not operating or not capex:
            return []

        # Index by date for fast matching.
        op_by_date: dict[str, dict[str, Any]] = {}
        for r in operating:
            d = str(r.get("date", ""))
            if not d:
                continue
            if d not in op_by_date or score_operating(r) > score_operating(op_by_date[d]):
                op_by_date[d] = r

        capex_by_date: dict[str, dict[str, Any]] = {}
        for r in capex:
            d = str(r.get("date", ""))
            if not d:
                continue
            if d not in capex_by_date or score_capex(r) > score_capex(capex_by_date[d]):
                capex_by_date[d] = r

        common_dates = sorted(set(op_by_date.keys()) & set(capex_by_date.keys()))
        if not common_dates:
            return []

        chosen = common_dates[-limit:]
        out: list[dict[str, Any]] = []
        for d in chosen:
            op_r = op_by_date[d]
            capex_r = capex_by_date[d]
            try:
                fcf_value = float(op_r.get("value")) - float(capex_r.get("value"))
            except Exception:
                continue
            out.append(
                {
                    "date": d,
                    "value": fcf_value,
                    "origin_name": "計算得出：營業活動現金流 - 資本支出",
                    "type": "Computed_FCF",
                    "computed": True,
                }
            )
        return out


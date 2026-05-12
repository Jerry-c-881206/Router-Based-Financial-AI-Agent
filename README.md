<<<<<<< HEAD
# AI 理財助理（個股問答）

以 **Streamlit** 呈現的對話式理財助理：使用者以自然語言詢問台股相關問題，，結合 **OpenAI（LangChain）**、**FinMind** 結構化資料與 **Tavily** 新聞搜尋，產出條列化回答並標示資料來源。


> FinMind API 端點與參數說明可參考 [**llms-full.txt**](./llms-full.txt)。

---

## 功能概覽

| 分析意圖（Intent） | 說明 | 主要資料來源 |
|-------------------|------|--------------|
| **fact** | 單一客觀數據（如 EPS、營收） | FinMind |
| **aggregation** | 多來源歸納、背景與原因整理 | Tavily（可選 **sentence-transformers** 做去重／相關性篩選） |
| **opinion** | 投資適性等評估類問題 | 多維度檢索 + LLM |
| **reasoning** | 假設性情境與因果推論 | 新聞／脈絡 + LLM |

前端流程：**Query Understanding → Execution Planner（Router）→ 對應 Pipeline → Response Generator（統一版面與來源標註）**。

---

## 技術棧

- Python 3.10+（建議）
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)（`langchain-openai`、`langchain-core`、`langchain-community`）
- [OpenAI API](https://platform.openai.com/)（透過 LangChain `ChatOpenAI`）
- [FinMind API](https://finmindtrade.com/)（台股結構化資料）
- [Tavily Search](https://tavily.com/)（新聞／網頁檢索）
- [sentence-transformers](https://www.sbert.net/)（Aggregation 相似度篩選；未安裝時會降級為截斷前 N 筆）

---

## 專案結構

```
├── app.py                 # Streamlit 入口、對話 UI
├── query_understanding.py # §3 查詢理解（意圖／實體／時間）
├── execution_planner.py   # §4 依 intent 分流至各 Pipeline
├── fact_pipeline.py       # §5.1 事實查詢（FinMind + 僅格式化 LLM）
├── aggregation_pipeline.py# §5.2 聚合／原因歸納（Tavily + 篩選 + LLM）
├── opinion_pipeline.py    # §5.3 觀點／評估
├── reasoning_pipeline.py  # §5.4 推理情境
├── response_generator.py  # §6 統一輸出格式與即時性提醒
├── finmind_client.py      # FinMind REST 封裝
├── tavily_client.py       # Tavily 搜尋封裝
├── time_utils.py          # 時間範圍解析
├── requirements.txt
├── SDD.md                 # 設計文件 v2.0
├── llms-full.txt          # FinMind API 參考筆記
└── README.md
```

---

## 環境設定

### 1. 建立虛擬環境並安裝依賴

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 設定環境變數

在專案根目錄建立 `.env`（**勿將 `.env` 提交至 Git**），至少包含：

| 變數 | 說明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 金鑰 |
| `TAVILY_API_KEY` | Tavily API 金鑰 |
| `FINMIND_API_TOKEN` 或 `FINMIND_TOKEN` | （選用）FinMind 金鑰；未設定時依 FinMind 公開 API 行為可能受限 |

各模組透過 `python-dotenv` 的 `load_dotenv()` 載入。

### 3. 啟動應用

```bash
streamlit run app.py
```

瀏覽器開啟終端機顯示的本機網址（預設常為 `http://localhost:8501`）。

---

## 資料與免責聲明

- 回答會標示 **FinMind**／**Tavily** 等來源；若問題含「昨天、上週、最近幾天」等短期時間錨點，介面可能顯示 **資料延遲** 提醒（見 `response_generator.py`）。
- 本專案僅供學習與技術展示，**不構成投資建議**；投資決策請自行判斷並參考合法金融資訊來源。

---

## 與 SDD.md 的對應關係

| SDD 章節（v2.0） | 實作模組 |
|------------------|----------|
| §3 Query Understanding | `query_understanding.py` |
| §4 Execution Planner | `execution_planner.py` |
| §5.1–§5.4 Pipelines | `fact_pipeline.py`、`aggregation_pipeline.py`、`opinion_pipeline.py`、`reasoning_pipeline.py` |
| §6 Response Generator | `response_generator.py` |


---

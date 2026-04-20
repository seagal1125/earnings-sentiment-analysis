# 財報情緒密碼：法說會情緒量化 vs 公佈後報酬 — 相關性分析

> **專案目標**：透過 Gemini CLI 將法說會（Earnings Call）逐字稿大量量化為「多維度情緒指標」，再以線性回歸分析這些指標與法說會後不同時間窗口股價報酬的相關性，驗證「文字裡的祕密」能否預測股價。

---

## 1. 專案概述

### 1.1 為什麼要做這件事？

上市公司每季法說會中，CEO 和 CFO 的遣詞用字暗藏玄機：

- **「Earn Our Value」** → 台積電啟動漲價週期、充滿底氣的信號
- **「Strong Demand」** → 需求強勁、產能供不應求的利多
- **「Pricing」vs「Price」** → 前者代表「定價能力/漲價」，後者只是雜訊

傳統做法是用肉眼找關鍵字、算出現次數，然後比對股價（錢博課堂上的方法）。
**本專案的升級**：用 **Gemini CLI** 一次性提取 **12 個量化維度**，從「單一關鍵字計數」進化到「全方位情緒掃描」，並用統計方法驗證哪些維度真的有預測力。

### 1.2 資料來源

| 資料 | 來源 | 格式 |
|------|------|------|
| 法說會逐字稿 | [defeatbeta/yahoo-finance-data](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data) → `stock_earning_call_transcripts` | Parquet（結構化：speaker + content） |
| 歷史股價 | 同資料集 → `stock_prices` 或 yfinance API | Open / Close / High / Low / Volume |
| 財報行事曆 | 同資料集 → `stock_earning_calendar` | 各公司財報公佈日期 |
| 基準指數 | yfinance API → SPY（S&P 500 ETF） | 超額報酬計算用 |

### 1.3 分析標的：Magnificent 7 + TSM（共 8 支）

| # | 代碼 | 公司 | 產業 | 超額報酬基準 |
|:-:|------|------|------|:-----------:|
| 1 | **AAPL** | Apple 蘋果 | 消費電子 / 軟體生態 | SPY |
| 2 | **MSFT** | Microsoft 微軟 | 雲端 / AI 平台 | SPY |
| 3 | **GOOGL** | Alphabet 谷歌 | 廣告 / AI / 雲端 | SPY |
| 4 | **AMZN** | Amazon 亞馬遜 | 電商 / 雲端（AWS） | SPY |
| 5 | **NVDA** | NVIDIA 輝達 | AI 晶片 / GPU | SPY |
| 6 | **META** | Meta 臉書 | 社群 / 元宇宙 / AI | SPY |
| 7 | **TSLA** | Tesla 特斯拉 | 電動車 / 能源 / AI | SPY |
| 8 | **TSM** | 台積電 ADR | 晶圓代工 | **0050.TW** |

> **為什麼 TSM 用 0050.TW 當基準？**
> 台積電在台股權重超過 30%，用 SPY 當基準會忽略台股市場的獨特性。
> 0050（元大台灣50 ETF）能更精準衡量台積電相對於台灣大盤的超額報酬。
> Magnificent 7 則使用 SPY（S&P 500 ETF）作為美股大盤基準。

- **時間範圍**：2020 Q1 ~ 2025 Q4（約 24 季 × 8 支 = **~192 個樣本點**）
- **法說會逐字稿欄位結構**：

```
symbol:          VARCHAR    — 股票代碼（如 TSM）
fiscal_year:     INTEGER    — 財報年度
fiscal_quarter:  INTEGER    — 第幾季（1-4）
report_date:     VARCHAR    — 法說會日期
transcripts:     STRUCT[]   — 結構化逐字稿陣列
  ├── paragraph_number: INTEGER  — 段落序號
  ├── speaker:          VARCHAR  — 發言者（含職稱）
  └── content:          VARCHAR  — 發言內容
```

---

## 2. Gemini CLI 多維度情緒量化框架 ⚡

> 這是本專案的核心創新——不只是「樂觀 / 悲觀」二分法，而是用 LLM 一次提取 **12 個維度**的結構化指標。

### 2.1 十二維度定義

| # | 維度 ID | 維度名稱 | 數值範圍 | 說明 | Gemini 提取重點 |
|:-:|---------|---------|:--------:|------|----------------|
| 1 | `overall_sentiment` | 整體情緒得分 | [-1, 1] | 全篇法說會的整體樂觀/悲觀傾向 | 綜合語氣判斷，-1=極度悲觀，1=極度樂觀 |
| 2 | `confidence_score` | 管理層信心度 | [0, 1] | CEO/CFO 回答時的肯定程度 | "we expect" vs "we hope"、"will" vs "might" |
| 3 | `forward_guidance_tone` | 前瞻指引語氣 | [-1, 1] | 對下季/明年展望的語調 | 聚焦 Guidance 段落的措辭強度 |
| 4 | `risk_disclosure_intensity` | 風險揭露強度 | [0, 1] | 提及風險/不確定性的密度 | "uncertainty"、"challenging"、"headwinds" 等 |
| 5 | `qa_evasiveness` | QA 閃躍程度 | [0, 1] | 回答分析師提問的迴避度 | 是否正面回答、有無轉移話題 |
| 6 | `quantitative_specificity` | 數據具體度 | [0, 1] | 給出具體數字的密度 | 具體營收/毛利率/成長率 vs 模糊描述 |
| 7 | `capex_enthusiasm` | 資本支出積極度 | [-1, 1] | 對擴張/投資的態度 | "aggressive investment"、"ramp up capacity" |
| 8 | `demand_strength` | 需求強度描述 | [0, 1] | 對市場需求的描述力度 | "strong demand"、"robust"、"unprecedented" |
| 9 | `pricing_power` | 定價能力信號 | [0, 1] | 漲價/定價權的信號強度 | "earn our value"、"pricing"、"premium" |
| 10 | `competitive_moat` | 競爭優勢自信 | [0, 1] | 對競爭壁壘的描述 | "technology leadership"、"irreplaceable" |
| 11 | `innovation_momentum` | 創新動能描述 | [0, 1] | AI/新技術/研發投入的積極程度 | "breakthrough"、"next generation"、"AI" |
| 12 | `management_consistency` | 管理層言行一致 | [0, 1] | 本季說法與上季承諾的吻合度 | 需比對前一季逐字稿（跨季分析） |

### 2.2 為什麼選這 12 個維度？

```
                    ┌─────────────────────────────────┐
                    │       法說會逐字稿文本            │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────┐   ┌────────────┐   ┌────────────┐
     │  語氣層面   │   │  內容層面   │   │  行為層面   │
     │ (HOW)      │   │ (WHAT)     │   │ (HOW MUCH) │
     ├────────────┤   ├────────────┤   ├────────────┤
     │ #1 整體情緒 │   │ #7 資本支出 │   │ #5 QA閃躍  │
     │ #2 信心度   │   │ #8 需求強度 │   │ #6 數據具體 │
     │ #3 前瞻語氣 │   │ #9 定價能力 │   │ #12 言行一致│
     │ #4 風險揭露 │   │ #10 競爭優勢│   │            │
     │            │   │ #11 創新動能│   │            │
     └────────────┘   └────────────┘   └────────────┘
```

- **語氣層面**：管理層「怎麼說」— 情緒、信心、語調
- **內容層面**：管理層「說什麼」— 需求、定價、投資、創新
- **行為層面**：管理層「說多少」— 是否迴避、是否給具體數據、是否前後一致

---

## 3. Gemini CLI 批次處理流程

### 3.1 單季分析 Prompt 範本

以下是餵給 Gemini CLI 的標準化 Prompt，要求回傳 **JSON 格式**的結構化評分：

```
你是一位專業的財務分析師，專門解讀上市公司法說會逐字稿。
請分析以下 {company_name}（{symbol}）{fiscal_year}年第{fiscal_quarter}季法說會逐字稿，
並回傳嚴格的 JSON 格式評分。

## 評分維度與說明

1. overall_sentiment（-1到1）：整體情緒傾向
2. confidence_score（0到1）：管理層信心度，注意 CEO vs CFO 的差異
3. forward_guidance_tone（-1到1）：前瞻指引的語氣
4. risk_disclosure_intensity（0到1）：風險揭露的密度與嚴重度
5. qa_evasiveness（0到1）：QA 環節的閃躍程度
6. quantitative_specificity（0到1）：具體數據的密度
7. capex_enthusiasm（-1到1）：資本支出的積極度
8. demand_strength（0到1）：需求描述的強度
9. pricing_power（0到1）：定價能力的信號
10. competitive_moat（0到1）：競爭優勢的自信程度
11. innovation_momentum（0到1）：創新/AI/技術發展的積極度
12. management_consistency（0到1）：與前季承諾的一致性（若無前季資料則填0.5）

## 回傳格式

請嚴格回傳以下 JSON，不要加任何其他文字：

{
  "symbol": "{symbol}",
  "fiscal_year": {fiscal_year},
  "fiscal_quarter": {fiscal_quarter},
  "scores": {
    "overall_sentiment": 0.0,
    "confidence_score": 0.0,
    "forward_guidance_tone": 0.0,
    "risk_disclosure_intensity": 0.0,
    "qa_evasiveness": 0.0,
    "quantitative_specificity": 0.0,
    "capex_enthusiasm": 0.0,
    "demand_strength": 0.0,
    "pricing_power": 0.0,
    "competitive_moat": 0.0,
    "innovation_momentum": 0.0,
    "management_consistency": 0.0
  },
  "key_phrases": ["列出3-5個最具代表性的原文關鍵短語"],
  "reasoning": "用1-2句話說明整體判斷邏輯"
}

## 逐字稿內容

{transcript_text}
```

### 3.2 批次執行策略

#### 方法一：Shell Script 逐季呼叫 Gemini CLI

```bash
#!/bin/bash
# 02_gemini_sentiment.sh — 批次情緒量化腳本

TRANSCRIPT_DIR="./data/transcripts"
OUTPUT_DIR="./data/sentiment_scores"
mkdir -p "$OUTPUT_DIR"

for file in "$TRANSCRIPT_DIR"/*.txt; do
  basename=$(basename "$file" .txt)
  echo "Processing: $basename"
  
  # 用 Gemini CLI 分析，將 Prompt 與逐字稿一起餵入
  cat prompt_template.txt "$file" | gemini \
    > "$OUTPUT_DIR/${basename}_scores.json"
  
  echo "Done: $basename"
  sleep 2  # 避免 rate limit
done
```

#### 方法二：Python + Gemini API（大量處理推薦）

```python
# 使用 google-genai SDK 批次處理
import json
from pathlib import Path
from google import genai

client = genai.Client()

def analyze_transcript(transcript_text, symbol, year, quarter):
    """單一季度逐字稿的情緒分析"""
    prompt = build_prompt(transcript_text, symbol, year, quarter)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)
```

### 3.3 長文處理策略（Map-Reduce）

法說會逐字稿通常超過 5,000 字，建議使用分段策略：

```
法說會逐字稿（全文 8000+ 字）
         │
         ├── 段落 1：CEO 開場報告     → Gemini 分析 → 段落得分 JSON
         ├── 段落 2：CFO 財務報告     → Gemini 分析 → 段落得分 JSON
         ├── 段落 3：業務展望         → Gemini 分析 → 段落得分 JSON
         ├── 段落 4-N：QA 環節        → Gemini 分析 → 段落得分 JSON
         │
         └── 匯總（加權平均）         → 最終得分 JSON
```

> **CEO 段落 vs CFO 段落**的權重可以不同——CFO 的數據具體度（#6）權重應更高，CEO 的信心度（#2）和創新動能（#11）權重應更高。

---

## 4. 股價報酬計算

### 4.1 基準日設定

- **T 日**：法說會當日（`report_date`）
- **基準價**：建議使用 **T+1 開盤價**（避免法說會當天盤後波動干擾）
- 若法說會發生在盤後/週末，則使用下一個交易日開盤價

### 4.2 多時間窗口報酬率

| 變數名稱 | 公式 | 經濟意義 |
|---------|------|---------|
| **R₁** | (P_{T+1收盤} / P_{T+1開盤}) - 1 | 即時反應（當日漲跌） |
| **R₃** | (P_{T+3收盤} / P_{T+1開盤}) - 1 | 超短期消化 |
| **R₅** | (P_{T+5收盤} / P_{T+1開盤}) - 1 | 短期反應（一週） |
| **R₁₀** | (P_{T+10收盤} / P_{T+1開盤}) - 1 | 中短期（兩週） |
| **R₂₀** | (P_{T+20收盤} / P_{T+1開盤}) - 1 | 中期（一個月） |
| **R₆₀** | (P_{T+60收盤} / P_{T+1開盤}) - 1 | 長期（一季） |
| **R₁₈₀** | (P_{T+180收盤} / P_{T+1開盤}) - 1 | 半年基本面發酵 |

### 4.3 超額報酬（Alpha）

為排除大盤因素影響，每個 R 都要計算對應的超額報酬：

```
# Magnificent 7（AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA）
α_n = R_n(個股) − R_n(SPY)

# 台積電（TSM）
α_n = R_n(TSM) − R_n(0050.TW)
```

- **SPY**（S&P 500 ETF）作為七巨頭的美股大盤基準
- **0050.TW**（元大台灣50）作為台積電的台股基準
- 超額報酬才能說明「法說會內容」帶來的獨立效果
- 否則在牛市中，任何指標都會顯示「正相關」
- 8 支股票 × 24 季 ≈ 192 個樣本，統計顯著性遠優於單一公司

---

## 5. 線性回歸與相關性分析

### 5.1 分析矩陣

最終要建構的是一個 **12×9 的相關性矩陣**：

```
                  R₁    R₃    R₅    R₁₀   R₂₀   R₆₀   R₁₈₀   α₁₀   α₆₀
overall_sentiment  [ρ₁₁] [ρ₁₂] [ρ₁₃] ...
confidence_score   [ρ₂₁] [ρ₂₂] ...
forward_guidance   [ρ₃₁] ...
risk_disclosure    ...
qa_evasiveness     ...
quant_specificity  ...
capex_enthusiasm   ...
demand_strength    ...
pricing_power      ...
competitive_moat   ...
innovation_moment  ...
mgmt_consistency   ...
```

### 5.2 統計方法

#### (A) Pearson 相關係數

```python
from scipy import stats

# 每個維度 vs 每個時間窗口
for dimension in dimensions:
    for window in windows:
        r, p_value = stats.pearsonr(
            sentiment_df[dimension],
            return_df[f'R_{window}']
        )
        # r: 相關係數 (-1到1)
        # p_value: 統計顯著性（< 0.05 才有意義）
```

#### (B) 單變量線性回歸

```python
from sklearn.linear_model import LinearRegression

# Y = β₀ + β₁·X + ε
# X = 情緒維度得分
# Y = 股價報酬率
model = LinearRegression()
model.fit(X_sentiment, Y_return)
# 輸出：β₁（斜率）、R²（解釋力）、p-value
```

#### (C) 多變量回歸（12 維度全部放入）

```python
# Y = β₀ + β₁·X₁ + β₂·X₂ + ... + β₁₂·X₁₂ + ε
# 找出哪些維度在控制其他變數後仍有獨立預測力
import statsmodels.api as sm

X = sm.add_constant(sentiment_df[all_dimensions])
model = sm.OLS(return_df['R_10'], X).fit()
print(model.summary())
# 重點看：各 β 的 p-value、整體 R²、Adjusted R²
```

#### (D) 滾動回歸（時序穩定性檢驗）

```python
# 用滾動窗口（如 12 季）檢查相關性是否穩定
# 如果相關性只在特定時段成立，那就不可靠
rolling_corr = sentiment_series.rolling(12).corr(return_series)
```

### 5.3 視覺化輸出

| 圖表類型 | 內容 | 工具 |
|---------|------|------|
| **熱力圖** | 12×9 相關係數矩陣 | matplotlib / seaborn `heatmap()` |
| **散佈圖** | 每個維度 vs R₁₀ 報酬率 | 12 個子圖 + 回歸線 |
| **時序圖** | 情緒得分 vs 股價走勢疊圖 | 雙 Y 軸折線圖 |
| **回歸診斷** | 殘差圖、QQ Plot | 檢查模型假設 |
| **重要性排名** | 各維度的 |ρ| 或 |β| 條形圖 | 一目瞭然哪些維度最有用 |

---

## 6. Python 腳本架構

### 6.1 專案目錄結構

```
財報情緒密碼/
├── README.md                        ← 你正在讀的這份文件
├── data/
│   ├── transcripts/                 ← 法說會逐字稿（按季度存放）
│   │   ├── TSM_2020_Q1.txt
│   │   ├── TSM_2020_Q2.txt
│   │   └── ...
│   ├── sentiment_scores/            ← Gemini 回傳的 JSON 評分
│   │   ├── TSM_2020_Q1_scores.json
│   │   └── ...
│   ├── prices/                      ← 股價資料
│   │   └── TSM_prices.csv
│   └── merged/                      ← 合併後的分析用資料集
│       └── TSM_sentiment_returns.csv
├── scripts/
│   ├── 01_fetch_transcripts.py      ← 從 HuggingFace 擷取逐字稿
│   ├── 02_gemini_sentiment.sh       ← Gemini CLI 批次情緒量化
│   ├── 03_fetch_prices.py           ← 擷取股價 + 計算多窗口報酬
│   ├── 04_regression_analysis.py    ← 線性回歸 + 相關性分析
│   └── 05_visualize.py              ← 熱力圖、散佈圖等視覺化
├── output/
│   ├── correlation_heatmap.png      ← 12×9 相關係數熱力圖
│   ├── scatter_plots.png            ← 維度 vs 報酬散佈圖
│   ├── regression_summary.html      ← 回歸分析報告
│   └── sentiment_timeseries.html    ← 互動式時序圖
├── prompts/
│   └── sentiment_prompt.txt         ← Gemini Prompt 範本
└── config.yaml                      ← 設定檔（標的、時間範圍、API Key 等）
```

### 6.2 各腳本功能概要

#### `01_fetch_transcripts.py` — 抓取法說會逐字稿

```python
# 核心邏輯：
# 1. 用 DuckDB 或 HuggingFace datasets 讀取 Parquet
# 2. 篩選 symbol = 'TSM', fiscal_year >= 2020
# 3. 將 transcripts 陣列展開為純文字
# 4. 按季度存成獨立的 .txt 檔案

import duckdb

con = duckdb.connect()
df = con.execute("""
    SELECT symbol, fiscal_year, fiscal_quarter, report_date, transcripts
    FROM 'hf://datasets/defeatbeta/yahoo-finance-data/stock_earning_call_transcripts.parquet'
    WHERE symbol = 'TSM'
    AND fiscal_year >= 2020
    ORDER BY fiscal_year, fiscal_quarter
""").fetchdf()
```

#### `02_gemini_sentiment.sh` — Gemini CLI 批次分析

- 逐一讀取 `data/transcripts/` 下的逐字稿
- 套入 Prompt 範本，呼叫 `gemini` CLI
- 將 JSON 回傳存入 `data/sentiment_scores/`
- 含 retry 機制與 rate limit 控制

#### `03_fetch_prices.py` — 股價 + 報酬率計算

```python
# 核心邏輯：
# 1. 用 yfinance 下載 TSM + SPY 的歷史股價
# 2. 對每個法說會日期，計算 R₁ ~ R₁₈₀ 各窗口報酬
# 3. 計算超額報酬 α = R(TSM) - R(SPY)
# 4. 與情緒評分 merge 成最終分析資料集
```

#### `04_regression_analysis.py` — 核心分析

```python
# 核心邏輯：
# 1. 載入 merged dataset
# 2. Pearson 相關係數 + p-value（12×9 矩陣）
# 3. 單變量線性回歸（每個維度 vs 每個窗口）
# 4. 多變量回歸（全維度 → R₁₀、α₁₀）
# 5. 輸出統計摘要表
```

#### `05_visualize.py` — 視覺化

```python
# 核心邏輯：
# 1. 熱力圖：12×9 相關係數矩陣
# 2. 散佈圖 + 回歸線：重要維度 vs 報酬
# 3. 時序疊圖：情緒得分 + 股價走勢
# 4. 產出 HTML 互動式報告（echarts）
```

---

## 7. 注意事項與避坑指南

### 7.1 存活者偏差 (Survivorship Bias)

⚠️ Magnificent 7 + TSM 本身就是「贏家」，近年大多處於上升趨勢。
所以 60 日勝率可能只是反映了大趨勢，跟法說會內容沒有因果關係。

**解法**：
- 計算「超額報酬」而非「絕對報酬」（TSM 用 0050、M7 用 SPY）
- 8 支不同產業的公司互相對照，降低單一公司的偏差
- 未來可納入表現差的公司（INTC 英特爾、BABA 阿里巴巴）進一步平衡

### 7.2 樣本數

8 支股票 × 24 季 ≈ **192 個樣本點**，相較單家公司的 24 點已大幅改善。

**注意事項**：
- 跨公司回歸時需加入「公司固定效果」（Company Fixed Effects），控制個股差異
- 報告時務必附上 p-value 和信賴區間
- 使用 Bootstrap 重抽樣增加穩健性
- 可按產業分群分析：半導體（TSM, NVDA）vs 軟體（MSFT, GOOGL）vs 消費（AAPL, AMZN, META, TSLA）

### 7.3 行業基準調整

科技股法說會後通常波動大，公用事業波動小。建議：
- 報酬率扣除同期大盤報酬（SPY）
- 或扣除產業 ETF 報酬（如 SMH 半導體 ETF）

### 7.4 Token 管理策略

| 模型 | 每季成本估計 | 說明 |
|------|------------|------|
| Gemini CLI（免費額度） | $0 | 有每日請求限制 |
| Gemini 2.5 Flash | ~$0.01/季 | 最佳性價比，推薦 |
| Gemini 2.5 Pro | ~$0.05/季 | 更精準但較貴 |

**建議**：先用 Gemini CLI 免費額度跑完全部逐字稿，若額度不夠再切換到 Flash API。

### 7.5 Prompt 一致性

- **所有季度必須使用完全相同的 Prompt**，否則評分無法橫向比較
- 溫度（Temperature）設為 0，確保結果可重現
- 建議同一季度跑 3 次取平均，降低 LLM 的隨機性

---

## 8. 預期成果

完成本專案後，你將得到：

1. **12×9 相關性熱力圖** — 一眼看出哪些情緒維度與哪些時間窗口最相關
2. **回歸方程式** — 例如 `R₁₀ = 0.03 + 0.12·demand_strength + 0.08·pricing_power + ε`
3. **預測力排名** — 12 個維度按預測力排序，找出最有價值的「情緒密碼」
4. **互動式 HTML 報告** — 用 echarts 做的漂亮儀表板
5. **可重用的分析框架** — 換一家公司只需改 symbol，整套流程自動跑

---

## 9. 快速開始

```bash
# Step 1: 抓取 8 支股票的法說會逐字稿
python scripts/01_fetch_transcripts.py \
  --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,TSM \
  --start-year 2020

# Step 2: 用 Gemini CLI 批次分析情緒（這步會大量使用 Gemini）
bash scripts/02_gemini_sentiment.sh

# Step 3: 抓取股價 + 計算報酬率（TSM 用 0050.TW 基準，其餘用 SPY）
python scripts/03_fetch_prices.py \
  --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,TSM \
  --benchmarks SPY,SPY,SPY,SPY,SPY,SPY,SPY,0050.TW

# Step 4: 跑回歸分析
python scripts/04_regression_analysis.py

# Step 5: 產出視覺化報告
python scripts/05_visualize.py
```

---

## 10. 參考資源

- **資料集**：[defeatbeta/yahoo-finance-data](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data)
- **Gemini CLI**：Google DeepMind 的命令列 AI 工具
- **統計方法**：Pearson 相關係數、OLS 線性回歸、Bootstrap
- **相關專案**：`台積電法說回測/backtest_tsmc_events.py`（事件回測框架）
- **錢博課程**：Yahoo 財報情緒分析——關鍵字回測方法論

#!/usr/bin/env python3
"""
03_fetch_prices.py
擷取股價資料，計算法說會後多時間窗口報酬率與超額報酬。
TSM 的股價使用 2330.TW，基準為 0050.TW；
Magnificent 7 使用各自美股代碼，基準為 SPY。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import yfinance as yf
except ImportError:
    print("請先安裝 yfinance: pip install yfinance")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
SENTIMENT_DIR = ROOT / "data" / "sentiment_scores"
PRICE_DIR = ROOT / "data" / "prices"
MERGED_DIR = ROOT / "data" / "merged"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """下載調整後收盤價"""
    print(f"   📈 下載 {ticker} 股價 ({start} ~ {end}) ...")
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        print(f"   ⚠️ {ticker} 無資料")
        return pd.Series(dtype=float)
    # 處理多欄位或單欄位情況
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"][ticker] if ticker in data["Close"].columns else data["Close"].iloc[:, 0]
    else:
        close = data["Close"]
    return close.dropna().sort_index()


def get_report_dates_from_transcripts() -> pd.DataFrame:
    """從逐字稿檔案的 header 解析法說會日期"""
    records = []
    for f in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        meta = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("="):
                    break
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip()] = val.strip()
        if meta.get("Symbol") and meta.get("Report Date"):
            records.append({
                "transcript_symbol": meta["Symbol"],
                "fiscal_year": int(meta.get("Fiscal Year", 0)),
                "fiscal_quarter": int(meta.get("Fiscal Quarter", "Q0").replace("Q", "")),
                "report_date": meta["Report Date"],
            })
    return pd.DataFrame(records)


def get_report_dates_from_sentiment() -> pd.DataFrame:
    """從情緒評分 JSON 解析法說會資訊"""
    records = []
    for f in sorted(SENTIMENT_DIR.glob("*_scores.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            records.append({
                "transcript_symbol": data["symbol"],
                "fiscal_year": data["fiscal_year"],
                "fiscal_quarter": data["fiscal_quarter"],
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return pd.DataFrame(records)


def calc_window_return(
    price_series: pd.Series,
    event_date: pd.Timestamp,
    window: int,
    use_next_open: bool = True,
) -> float | None:
    """計算事件後 N 日的報酬率

    基準價 = T+1 開盤價（用收盤價近似）
    結束價 = T+1+window 收盤價
    """
    idx = price_series.index
    # 找到 >= event_date 的第一個交易日
    mask = idx >= event_date
    if not mask.any():
        return None

    start_pos = mask.argmax()  # T 日的位置

    if use_next_open:
        # 用 T+1 作為基準
        base_pos = start_pos + 1
    else:
        base_pos = start_pos

    end_pos = base_pos + window

    if base_pos >= len(price_series) or end_pos >= len(price_series):
        return None

    base_price = float(price_series.iloc[base_pos])
    end_price = float(price_series.iloc[end_pos])

    if base_price <= 0:
        return None

    return (end_price / base_price) - 1.0


def process_symbol(
    config_entry: dict,
    events_df: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    """處理單一股票的報酬計算"""
    t_symbol = config_entry["transcript_symbol"]
    p_symbol = config_entry["price_symbol"]
    b_symbol = config_entry["benchmark"]

    # 篩選該股票的事件
    symbol_events = events_df[events_df["transcript_symbol"] == t_symbol].copy()
    if symbol_events.empty:
        print(f"   ⚠️ {t_symbol} 無法說會事件資料")
        return pd.DataFrame()

    symbol_events["report_date"] = pd.to_datetime(symbol_events["report_date"])

    # 決定下載範圍
    min_date = symbol_events["report_date"].min() - pd.Timedelta(days=30)
    max_date = symbol_events["report_date"].max() + pd.Timedelta(days=max(windows) * 2 + 30)

    start_str = min_date.strftime("%Y-%m-%d")
    end_str = max_date.strftime("%Y-%m-%d")

    # 下載股價
    price_series = download_prices(p_symbol, start_str, end_str)
    bench_series = download_prices(b_symbol, start_str, end_str)

    if price_series.empty:
        print(f"   ❌ {p_symbol} 股價資料為空")
        return pd.DataFrame()

    # 存股價到 CSV
    price_csv = PRICE_DIR / f"{p_symbol.replace('.', '_')}_prices.csv"
    price_series.to_frame("close").to_csv(price_csv)
    print(f"   💾 股價已存: {price_csv.name}")

    # 計算每個事件的報酬率
    rows = []
    for _, event in symbol_events.iterrows():
        row = {
            "transcript_symbol": t_symbol,
            "price_symbol": p_symbol,
            "benchmark": b_symbol,
            "company_name": config_entry["company_name"],
            "fiscal_year": event["fiscal_year"],
            "fiscal_quarter": event["fiscal_quarter"],
            "report_date": event["report_date"].strftime("%Y-%m-%d"),
        }

        for w in windows:
            # 個股報酬
            ret = calc_window_return(price_series, event["report_date"], w)
            row[f"R_{w}"] = ret

            # 基準報酬
            if not bench_series.empty:
                bench_ret = calc_window_return(bench_series, event["report_date"], w)
                row[f"R_bench_{w}"] = bench_ret

                # 超額報酬
                if ret is not None and bench_ret is not None:
                    row[f"alpha_{w}"] = ret - bench_ret
                else:
                    row[f"alpha_{w}"] = None
            else:
                row[f"R_bench_{w}"] = None
                row[f"alpha_{w}"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def merge_with_sentiment(returns_df: pd.DataFrame) -> pd.DataFrame:
    """合併情緒評分與報酬率"""
    sentiment_records = []
    for f in sorted(SENTIMENT_DIR.glob("*_scores.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            flat = {
                "transcript_symbol": data["symbol"],
                "fiscal_year": data["fiscal_year"],
                "fiscal_quarter": data["fiscal_quarter"],
            }
            for k, v in data.get("scores", {}).items():
                flat[k] = v
            flat["key_phrases"] = json.dumps(
                data.get("key_phrases", []), ensure_ascii=False
            )
            flat["reasoning"] = data.get("reasoning", "")
            sentiment_records.append(flat)
        except (json.JSONDecodeError, KeyError):
            continue

    if not sentiment_records:
        print("   ⚠️ 無情緒評分資料，僅輸出報酬率")
        return returns_df

    sentiment_df = pd.DataFrame(sentiment_records)

    merged = returns_df.merge(
        sentiment_df,
        on=["transcript_symbol", "fiscal_year", "fiscal_quarter"],
        how="left",
    )
    return merged


def main():
    parser = argparse.ArgumentParser(description="擷取股價 + 計算多窗口報酬率")
    parser.add_argument("--skip-sentiment-merge", action="store_true",
                        help="跳過情緒評分合併（僅計算報酬率）")
    args = parser.parse_args()

    config = load_config()
    windows = config.get("return_windows", [1, 3, 5, 10, 20, 60, 180])
    symbol_configs = config["symbols"]

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # 載入法說會日期
    events_df = get_report_dates_from_transcripts()
    if events_df.empty:
        print("❌ 找不到法說會日期資料。請先執行: python scripts/01_fetch_transcripts.py")
        sys.exit(1)

    print(f"📊 股價報酬計算")
    print(f"   事件數: {len(events_df)}")
    print(f"   窗口:   {windows}")
    print(f"   TSM 股價代碼: 2330.TW（基準: 0050.TW）")
    print()

    all_returns = []
    for cfg in symbol_configs:
        t_sym = cfg["transcript_symbol"]
        p_sym = cfg["price_symbol"]
        print(f"\n🏢 {cfg['company_name']} ({t_sym} → 股價:{p_sym}, 基準:{cfg['benchmark']})")

        result = process_symbol(cfg, events_df, windows)
        if not result.empty:
            all_returns.append(result)
            print(f"   ✅ {len(result)} 筆報酬率計算完成")

    if not all_returns:
        print("\n❌ 無任何報酬率資料")
        sys.exit(1)

    returns_df = pd.concat(all_returns, ignore_index=True)

    # 存純報酬率
    returns_csv = MERGED_DIR / "all_returns.csv"
    returns_df.to_csv(returns_csv, index=False)
    print(f"\n💾 報酬率已存: {returns_csv}")

    # 合併情緒評分
    if not args.skip_sentiment_merge:
        merged = merge_with_sentiment(returns_df)
        merged_csv = MERGED_DIR / "sentiment_returns_merged.csv"
        merged.to_csv(merged_csv, index=False)
        print(f"💾 合併資料已存: {merged_csv}")
        print(f"   總筆數: {len(merged)}")
        print(f"   有情緒評分: {merged.iloc[:, -1].notna().sum()}")

    print("\n🎉 股價報酬計算完成！")


if __name__ == "__main__":
    main()

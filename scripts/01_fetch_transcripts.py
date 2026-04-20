#!/usr/bin/env python3

"""
01_fetch_transcripts.py
從 HuggingFace defeatbeta/yahoo-finance-data 擷取法說會逐字稿，
按公司/年度/季度存成獨立文字檔。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

try:
    import duckdb
except ImportError:
    print("請先安裝 duckdb: pip install duckdb")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data" / "transcripts"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_transcripts(symbol: str, start_year: int, end_year: int) -> list[dict]:
    """用 DuckDB 從 HuggingFace Parquet 擷取逐字稿"""
    con = duckdb.connect()

    # 安裝並載入 httpfs 以支援遠端 Parquet
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-east-1';")

    parquet_url = (
        "hf://datasets/defeatbeta/yahoo-finance-data/"
        "data/stock_earning_call_transcripts.parquet"
    )

    query = f"""
        SELECT
            symbol,
            fiscal_year,
            fiscal_quarter,
            report_date,
            transcripts
        FROM '{parquet_url}'
        WHERE symbol = ?
          AND fiscal_year >= ?
          AND fiscal_year <= ?
        ORDER BY fiscal_year, fiscal_quarter
    """

    try:
        result = con.execute(query, [symbol, start_year, end_year]).fetchall()
        columns = ["symbol", "fiscal_year", "fiscal_quarter", "report_date", "transcripts"]
        return [dict(zip(columns, row)) for row in result]
    except Exception as e:
        print(f"  ⚠️ 擷取 {symbol} 時發生錯誤: {e}")
        return []
    finally:
        con.close()


def transcripts_to_text(transcripts) -> str:
    """將結構化逐字稿陣列轉為可讀文字"""
    if transcripts is None:
        return ""

    lines = []
    for entry in transcripts:
        if isinstance(entry, dict):
            speaker = entry.get("speaker", "Unknown")
            content = entry.get("content", "")
            para_num = entry.get("paragraph_number", "")
            lines.append(f"[{para_num}] {speaker}:\n{content}\n")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
            # 結構可能是 (paragraph_number, speaker, content)
            para_num, speaker, content = entry[0], entry[1], entry[2]
            lines.append(f"[{para_num}] {speaker}:\n{content}\n")
        else:
            lines.append(str(entry) + "\n")

    return "\n".join(lines)


def save_transcript(symbol: str, year: int, quarter: int,
                    report_date: str, text: str, output_dir: Path) -> Path:
    """將逐字稿存成文字檔"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol}_{year}_Q{quarter}.txt"
    filepath = output_dir / filename

    header = (
        f"Symbol: {symbol}\n"
        f"Fiscal Year: {year}\n"
        f"Fiscal Quarter: Q{quarter}\n"
        f"Report Date: {report_date}\n"
        f"{'=' * 60}\n\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + text)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="從 HuggingFace 擷取法說會逐字稿")
    parser.add_argument(
        "--symbols",
        default=None,
        help="逗號分隔的股票代碼（逐字稿用），預設從 config.yaml 讀取",
    )
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    start_year = args.start_year or config.get("start_year", 2020)
    end_year = args.end_year or config.get("end_year", 2025)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = [s["transcript_symbol"] for s in config["symbols"]]

    print(f"📋 擷取法說會逐字稿")
    print(f"   標的: {', '.join(symbols)}")
    print(f"   時間: {start_year} ~ {end_year}")
    print(f"   輸出: {DATA_DIR}")
    print()

    total_saved = 0
    for symbol in symbols:
        print(f"🔍 擷取 {symbol} ...")
        records = fetch_transcripts(symbol, start_year, end_year)
        print(f"   找到 {len(records)} 筆逐字稿")

        for rec in records:
            text = transcripts_to_text(rec["transcripts"])
            if not text.strip():
                print(f"   ⚠️ {symbol} {rec['fiscal_year']} Q{rec['fiscal_quarter']} 逐字稿為空，跳過")
                continue

            filepath = save_transcript(
                symbol=rec["symbol"],
                year=rec["fiscal_year"],
                quarter=rec["fiscal_quarter"],
                report_date=str(rec["report_date"]),
                text=text,
                output_dir=DATA_DIR,
            )
            word_count = len(text.split())
            print(f"   ✅ 已存: {filepath.name} ({word_count} 字)")
            total_saved += 1

    print(f"\n🎉 完成！共存 {total_saved} 份逐字稿到 {DATA_DIR}")


if __name__ == "__main__":
    main()

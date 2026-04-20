import json
import time
from pathlib import Path

import requests


API_BASE_URL = "http://localhost:1234/api/v1/chat"
MODEL_ID = "mlx-community/gemma-4-26b-a4b-it"
SYSTEM_PROMPT = "You are a professional financial analyst. Return strict JSON only."

TRANSCRIPTS_DIR = Path("data/transcripts")
OUTPUT_DIR = Path("m4_pro_mlx_gemma4")
PROMPT_FILE = Path("prompts/sentiment_prompt.txt")
TARGET_SYMBOLS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "TSM"]


def parse_content(raw_content: str) -> dict:
    clean_content = raw_content.strip()
    if clean_content.startswith("```json"):
        clean_content = clean_content[7:]
    if clean_content.endswith("```"):
        clean_content = clean_content[:-3]
    return json.loads(clean_content.strip())


def analyze_transcript(file_path: Path, prompt_template: str) -> dict:
    transcript = file_path.read_text(encoding="utf-8")
    full_prompt = f"{prompt_template}\n\n{transcript}"

    payload = {
        "model": MODEL_ID,
        "system_prompt": SYSTEM_PROMPT,
        "input": full_prompt,
    }

    start_time = time.time()
    try:
        response = requests.post(API_BASE_URL, json=payload, timeout=300)
        duration = round(time.time() - start_time, 2)
        if response.status_code != 200:
            return {
                "file": file_path.name,
                "status": "error",
                "duration_seconds": duration,
                "error": f"Status {response.status_code}: {response.text}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

        result = response.json()
        raw_content = result["output"][0]["content"]

        try:
            analysis_data = parse_content(raw_content)
        except Exception:
            analysis_data = {
                "raw_content": raw_content,
                "error": "Failed to parse JSON content",
            }

        return {
            "file": file_path.name,
            "status": "success",
            "duration_seconds": duration,
            "model": MODEL_ID,
            "analysis": analysis_data,
            "stats": result.get("stats", {}),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as exc:
        return {
            "file": file_path.name,
            "status": "error",
            "duration_seconds": round(time.time() - start_time, 2),
            "error": str(exc),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
    files = sorted(
        [
            file_path
            for file_path in TRANSCRIPTS_DIR.glob("*.txt")
            if any(file_path.name.startswith(symbol) for symbol in TARGET_SYMBOLS)
        ]
    )

    print(f"找到 {len(files)} 個待處理檔案。")

    for index, file_path in enumerate(files, start=1):
        output_file = OUTPUT_DIR / f"{file_path.stem}_analysis.json"

        if output_file.exists():
            try:
                existing = json.loads(output_file.read_text(encoding="utf-8"))
                if existing.get("status") == "success":
                    print(f"[{index}/{len(files)}] 跳過已存在的檔案: {file_path.name}")
                    continue
            except Exception:
                pass

        print(f"[{index}/{len(files)}] 正在分析: {file_path.name} ... ", end="", flush=True)
        result = analyze_transcript(file_path, prompt_template)
        output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        if result["status"] == "success":
            print(f"完成! (耗時: {result['duration_seconds']}s)")
        else:
            print(f"失敗: {result['error']}")


if __name__ == "__main__":
    main()

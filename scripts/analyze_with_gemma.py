import os
import json
import time
import requests
from pathlib import Path

# 配置資訊 (正選使用 5090)
API_BASE_URL = "https://gemma4.allin-market.cc/v1/chat/completions"
MODEL_ID = "google/gemma-4-26b-a4b"

# 備援配置 (Remote M4)
# API_BASE_URL = "https://gemma4-m4.allin-market.cc/v1/chat/completions"
# MODEL_ID = "lmstudio-community/gemma-4-26b-a4b-it"

TRANSCRIPTS_DIR = Path("data/transcripts")
OUTPUT_DIR = Path("5090_Gemma4")
PROMPT_FILE = Path("prompts/sentiment_prompt.txt")
TARGET_SYMBOLS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "TSM"]

def analyze_transcript(file_path, prompt_template):
    with open(file_path, 'r', encoding='utf-8') as f:
        transcript = f.read()

    # 組合 Prompt
    full_prompt = f"{prompt_template}\n\n{transcript}"
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.1
    }

    start_time = time.time()
    try:
        response = requests.post(API_BASE_URL, json=payload, timeout=300)
        if response.status_code != 200:
            return {
                "file": file_path.name,
                "status": "error",
                "duration_seconds": round(time.time() - start_time, 2),
                "error": f"Status {response.status_code}: {response.text}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        result = response.json()
        end_time = time.time()
        
        duration = end_time - start_time
        content = result['choices'][0]['message']['content']
        
        # 嘗試解析回傳的 JSON 字串
        try:
            # 去除 Markdown 程式碼區塊 (如果有)
            clean_content = content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            analysis_data = json.loads(clean_content.strip())
        except:
            analysis_data = {"raw_content": content, "error": "Failed to parse JSON content"}

        return {
            "file": file_path.name,
            "status": "success",
            "duration_seconds": round(duration, 2),
            "model": MODEL_ID,
            "analysis": analysis_data,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        end_time = time.time()
        return {
            "file": file_path.name,
            "status": "error",
            "duration_seconds": round(end_time - start_time, 2),
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()

    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # 獲取所有相關檔案 (符合目標公司且是 .txt)
    files = sorted([f for f in TRANSCRIPTS_DIR.glob("*.txt") if any(f.name.startswith(s) for s in TARGET_SYMBOLS)])
    
    print(f"找到 {len(files)} 個待處理檔案。")

    for i, file_path in enumerate(files):
        output_file = OUTPUT_DIR / f"{file_path.stem}_analysis.json"
        
        # 如果已經處理過，跳過
        if output_file.exists():
            # 檢查檔案是否有效 (不是 error)
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("status") == "success":
                    print(f"[{i+1}/{len(files)}] 跳過已存在的檔案: {file_path.name}")
                    continue
            except:
                pass

        print(f"[{i+1}/{len(files)}] 正在分析: {file_path.name} ... ", end="", flush=True)
        result = analyze_transcript(file_path, prompt_template)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if result["status"] == "success":
            print(f"完成! (耗時: {result['duration_seconds']}s)")
        else:
            print(f"失敗: {result['error']}")

if __name__ == "__main__":
    main()

#!/bin/bash
# ============================================================
# 02_gemini_sentiment.sh
# 用 Gemini CLI 批次分析法說會逐字稿的多維度情緒
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TRANSCRIPT_DIR="$ROOT_DIR/data/transcripts"
OUTPUT_DIR="$ROOT_DIR/data/sentiment_scores"
PROMPT_FILE="$ROOT_DIR/prompts/sentiment_prompt.txt"
SLEEP_SEC=2
MAX_RETRIES=3

mkdir -p "$OUTPUT_DIR"

# 檢查必要檔案
if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ 找不到 Prompt 範本: $PROMPT_FILE"
    exit 1
fi

if [ ! -d "$TRANSCRIPT_DIR" ]; then
    echo "❌ 找不到逐字稿目錄: $TRANSCRIPT_DIR"
    echo "   請先執行: python scripts/01_fetch_transcripts.py"
    exit 1
fi

# 統計
total=0
success=0
failed=0
skipped=0

echo "🤖 Gemini CLI 批次情緒分析"
echo "   逐字稿目錄: $TRANSCRIPT_DIR"
echo "   輸出目錄:   $OUTPUT_DIR"
echo "   Prompt:     $PROMPT_FILE"
echo ""

for transcript_file in "$TRANSCRIPT_DIR"/*.txt; do
    [ -f "$transcript_file" ] || continue
    total=$((total + 1))

    basename=$(basename "$transcript_file" .txt)
    output_file="$OUTPUT_DIR/${basename}_scores.json"

    # 跳過已處理的檔案
    if [ -f "$output_file" ]; then
        echo "⏭️  跳過 (已存在): $basename"
        skipped=$((skipped + 1))
        continue
    fi

    echo "📝 處理中: $basename"

    # 重試機制
    retry=0
    while [ $retry -lt $MAX_RETRIES ]; do
        # 組合 Prompt + 逐字稿內容，餵給 Gemini CLI
        raw_result=$(cat "$PROMPT_FILE" "$transcript_file" | gemini 2>/dev/null || true)

        # 提取出純 JSON 部分
        result=$(echo "$raw_result" | python3 -c '
import sys, json, re
text = sys.stdin.read()
match = re.search(r"\{.*\}", text, re.DOTALL)
if match:
    try:
        data = json.loads(match.group(0))
        print(json.dumps(data, ensure_ascii=False))
    except Exception:
        sys.exit(1)
else:
    sys.exit(1)
' 2>/dev/null || true)

        # 檢查回傳是否為有效 JSON
        if [ -n "$result" ]; then
            echo "$result" > "$output_file"
            echo "   ✅ 完成: ${basename}_scores.json"
            success=$((success + 1))
            break
        else
            retry=$((retry + 1))
            if [ $retry -lt $MAX_RETRIES ]; then
                echo "   ⚠️ JSON 解析失敗，重試 ($retry/$MAX_RETRIES)..."
                sleep $SLEEP_SEC
            else
                # 即使 JSON 無效也存下來，方便後續除錯
                echo "$raw_result" > "${output_file}.raw"
                echo "   ❌ 失敗 (已存原始回傳): ${basename}_scores.json.raw"
                failed=$((failed + 1))
            fi
        fi
    done

    # 避免 rate limit
    sleep $SLEEP_SEC
done

echo ""
echo "📊 批次處理完成"
echo "   總計:   $total"
echo "   成功:   $success"
echo "   跳過:   $skipped"
echo "   失敗:   $failed"

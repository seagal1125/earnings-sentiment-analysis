#!/bin/bash
result=$(cat "prompts/sentiment_prompt.txt" "data/transcripts/AAPL_2020_Q1.txt" | gemini 2>/dev/null || true)
echo "RESULT:"
echo "$result"

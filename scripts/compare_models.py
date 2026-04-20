import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# 配置路徑
GEMINI_DIR = Path("data/sentiment_scores")
GEMMA_DIR = Path("localllm")

def load_data():
    gemini_data = []
    gemma_data = []
    
    # 遍歷 Gemini 的檔案
    for g_file in GEMINI_DIR.glob("*_scores.json"):
        # 解析檔名 AAPL_2024_Q4_scores.json -> AAPL_2024_Q4
        base_name = g_file.name.replace("_scores.json", "")
        gemma_file = GEMMA_DIR / f"{base_name}_analysis.json"
        
        if gemma_file.exists():
            try:
                with open(g_file, 'r', encoding='utf-8') as f1, \
                     open(gemma_file, 'r', encoding='utf-8') as f2:
                    
                    data1 = json.load(f1)
                    data2 = json.load(f2)
                    
                    # 檢查 status
                    if data2.get("status") != "success":
                        continue
                        
                    scores1 = data1.get("scores", {})
                    scores2 = data2.get("analysis", {}).get("scores", {})
                    
                    if scores1 and scores2:
                        scores1['id'] = base_name
                        scores2['id'] = base_name
                        gemini_data.append(scores1)
                        gemma_data.append(scores2)
            except Exception as e:
                print(f"Error loading {base_name}: {e}")

    return pd.DataFrame(gemini_data), pd.DataFrame(gemma_data)

def main():
    df1, df2 = load_data()
    
    if df1.empty or df2.empty:
        print("未找到可匹配的對比數據。")
        return

    # 確保 ID 對齊
    df1 = df1.set_index('id').sort_index()
    df2 = df2.set_index('id').sort_index()
    
    # 共有的列 (評分維度)
    score_cols = [c for c in df1.columns if c != 'id']
    
    results = []
    print(f"正在分析 {len(df1)} 個樣本的相關性...\n")
    
    for col in score_cols:
        if col in df2.columns:
            # 提取數據並去除 NaN
            v1 = df1[col]
            v2 = df2[col]
            
            # 計算 Pearson 相關係數
            corr, p_value = pearsonr(v1, v2)
            
            # 計算平均絕對誤差 (MAE)
            mae = np.mean(np.abs(v1 - v2))
            
            results.append({
                "維度": col,
                "相關係數 (Corr)": round(corr, 4),
                "P-Value": round(p_value, 6),
                "平均誤差 (MAE)": round(mae, 4)
            })
            
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    # 整體平均
    avg_corr = res_df["相關係數 (Corr)"].mean()
    print(f"\n平均相關係數: {round(avg_corr, 4)}")
    
    # 結論邏輯
    print("\n" + "="*50)
    print("【分析結論】")
    if avg_corr > 0.8:
        print(f"相關係數為 {round(avg_corr, 4)}，屬於【極高相關】。")
        print("結論：Gemma 4 (26B) 在 5090 本地端的量化表現與 Gemini 3.1 Pro 極度接近。")
        print("建議：後續任務可以完全移交給本地端模型處理，節省 API 成本且維持數據一致性。")
    elif avg_corr > 0.6:
        print(f"相關係數為 {round(avg_corr, 4)}，屬於【中高相關】。")
        print("結論：趨勢一致但數值精度有微小差異。")
        print("建議：可以替代用於大趨勢分析，若需極精確回測可保留 Gemini 作為校準。")
    else:
        print(f"相關係數為 {round(avg_corr, 4)}，相關性較低。")
        print("結論：本地端模型與雲端模型在解讀邏輯上有較大分歧。")
    print("="*50)

if __name__ == "__main__":
    main()

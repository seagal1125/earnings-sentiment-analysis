#!/usr/bin/env python3
"""
04_regression_analysis.py
載入合併的情緒評分 + 報酬率資料，執行：
1. Pearson 相關係數矩陣（12 維度 × N 窗口）
2. 單變量線性回歸（每個維度 vs 每個窗口）
3. 多變量回歸（全維度 → 特定窗口）
4. 產業分群分析
輸出統計摘要。
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
    from scipy import stats
    from sklearn.linear_model import LinearRegression
except ImportError:
    print("請先安裝: pip install scipy scikit-learn")
    sys.exit(1)

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️ 未安裝 statsmodels，多變量回歸將使用 sklearn 替代")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
MERGED_DIR = ROOT / "data" / "merged"
OUTPUT_DIR = ROOT / "output"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


SENTIMENT_DIMS = [
    "overall_sentiment",
    "confidence_score",
    "forward_guidance_tone",
    "risk_disclosure_intensity",
    "qa_evasiveness",
    "quantitative_specificity",
    "capex_enthusiasm",
    "demand_strength",
    "pricing_power",
    "competitive_moat",
    "innovation_momentum",
    "management_consistency",
]

# 維度中文名稱對照
DIM_NAMES_ZH = {
    "overall_sentiment": "整體情緒",
    "confidence_score": "管理層信心",
    "forward_guidance_tone": "前瞻語氣",
    "risk_disclosure_intensity": "風險揭露",
    "qa_evasiveness": "QA閃躍",
    "quantitative_specificity": "數據具體度",
    "capex_enthusiasm": "資本支出積極",
    "demand_strength": "需求強度",
    "pricing_power": "定價能力",
    "competitive_moat": "競爭優勢",
    "innovation_momentum": "創新動能",
    "management_consistency": "言行一致",
}


def load_merged_data() -> pd.DataFrame:
    """載入合併資料集"""
    merged_csv = MERGED_DIR / "sentiment_returns_merged.csv"
    if not merged_csv.exists():
        print(f"❌ 找不到合併資料: {merged_csv}")
        print("   請先依序執行 01 → 02 → 03 腳本")
        sys.exit(1)
    return pd.read_csv(merged_csv)


def get_return_columns(df: pd.DataFrame) -> list[str]:
    """取得所有報酬率欄位"""
    r_cols = [c for c in df.columns if c.startswith("R_") and not c.startswith("R_bench")]
    alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
    return r_cols + alpha_cols


def get_available_dims(df: pd.DataFrame) -> list[str]:
    """取得資料中存在的情緒維度"""
    return [d for d in SENTIMENT_DIMS if d in df.columns]


def pearson_correlation_matrix(
    df: pd.DataFrame,
    dims: list[str],
    return_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """計算 Pearson 相關係數矩陣與 p-value 矩陣"""
    corr_matrix = pd.DataFrame(index=dims, columns=return_cols, dtype=float)
    pval_matrix = pd.DataFrame(index=dims, columns=return_cols, dtype=float)

    for dim in dims:
        for ret_col in return_cols:
            valid = df[[dim, ret_col]].dropna()
            if len(valid) < 5:
                corr_matrix.loc[dim, ret_col] = np.nan
                pval_matrix.loc[dim, ret_col] = np.nan
                continue
            r, p = stats.pearsonr(valid[dim], valid[ret_col])
            corr_matrix.loc[dim, ret_col] = r
            pval_matrix.loc[dim, ret_col] = p

    return corr_matrix.astype(float), pval_matrix.astype(float)


def single_variable_regression(
    df: pd.DataFrame,
    dims: list[str],
    return_cols: list[str],
) -> pd.DataFrame:
    """對每個維度 vs 每個報酬窗口做單變量線性回歸"""
    results = []
    for dim in dims:
        for ret_col in return_cols:
            valid = df[[dim, ret_col]].dropna()
            if len(valid) < 5:
                continue

            X = valid[[dim]].values
            y = valid[ret_col].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # 計算 p-value
            r, p = stats.pearsonr(valid[dim], valid[ret_col])

            results.append({
                "dimension": dim,
                "dimension_zh": DIM_NAMES_ZH.get(dim, dim),
                "return_window": ret_col,
                "beta": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "r_squared": r_squared,
                "pearson_r": r,
                "p_value": p,
                "n_samples": len(valid),
                "significant_005": p < 0.05,
                "significant_010": p < 0.10,
            })

    return pd.DataFrame(results)


def multi_variable_regression(
    df: pd.DataFrame,
    dims: list[str],
    target_col: str,
) -> dict:
    """多變量回歸：全維度 → 特定報酬窗口"""
    valid = df[dims + [target_col]].dropna()
    if len(valid) < len(dims) + 2:
        return {"error": f"樣本不足: {len(valid)} < {len(dims) + 2}"}

    X = valid[dims].values
    y = valid[target_col].values

    if HAS_STATSMODELS:
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        coefficients = {}
        for i, dim in enumerate(dims):
            coefficients[dim] = {
                "beta": float(model.params[i + 1]),
                "p_value": float(model.pvalues[i + 1]),
                "significant": model.pvalues[i + 1] < 0.05,
            }

        return {
            "target": target_col,
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_p_value": float(model.f_pvalue),
            "n_samples": len(valid),
            "coefficients": coefficients,
            "summary_text": str(model.summary()),
        }
    else:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        coefficients = {}
        for i, dim in enumerate(dims):
            coefficients[dim] = {
                "beta": float(model.coef_[i]),
            }

        return {
            "target": target_col,
            "r_squared": r_squared,
            "n_samples": len(valid),
            "coefficients": coefficients,
        }


def industry_group_analysis(
    df: pd.DataFrame,
    dims: list[str],
    return_cols: list[str],
) -> dict:
    """按產業分群分析相關性"""
    # 定義產業分群
    groups = {
        "半導體": ["TSM", "NVDA"],
        "軟體平台": ["MSFT", "GOOGL", "META"],
        "消費科技": ["AAPL", "AMZN", "TSLA"],
    }

    results = {}
    for group_name, symbols in groups.items():
        group_df = df[df["transcript_symbol"].isin(symbols)]
        if len(group_df) < 5:
            continue

        corr_mat, pval_mat = pearson_correlation_matrix(group_df, dims, return_cols)
        results[group_name] = {
            "n_samples": len(group_df),
            "symbols": symbols,
            "correlation_matrix": corr_mat.to_dict(),
            "significant_pairs": [],
        }

        # 找出顯著相關的配對
        for dim in dims:
            for ret_col in return_cols:
                p = pval_mat.loc[dim, ret_col]
                r = corr_mat.loc[dim, ret_col]
                if pd.notna(p) and p < 0.10:
                    results[group_name]["significant_pairs"].append({
                        "dimension": dim,
                        "return_window": ret_col,
                        "pearson_r": float(r),
                        "p_value": float(p),
                    })

    return results


def print_report(
    corr_matrix: pd.DataFrame,
    pval_matrix: pd.DataFrame,
    regression_df: pd.DataFrame,
    multi_reg_results: list[dict],
    industry_results: dict,
):
    """列印分析報告"""
    print("\n" + "=" * 80)
    print("📊 法說會情緒密碼 vs 報酬率 — 相關性分析報告")
    print("=" * 80)

    # 1. 相關係數矩陣
    print("\n\n📈 一、Pearson 相關係數矩陣")
    print("-" * 60)
    display_corr = corr_matrix.copy()
    display_corr.index = [DIM_NAMES_ZH.get(d, d) for d in display_corr.index]
    print(display_corr.round(3).to_string())

    # 2. 顯著相關排名
    print("\n\n🏆 二、顯著相關排名（p < 0.10）")
    print("-" * 60)
    sig = regression_df[regression_df["significant_010"]].sort_values("p_value")
    if sig.empty:
        print("   ⚠️ 無顯著相關（p < 0.10）")
    else:
        for _, row in sig.head(20).iterrows():
            star = "***" if row["p_value"] < 0.01 else "**" if row["p_value"] < 0.05 else "*"
            print(
                f"   {row['dimension_zh']:8s} → {row['return_window']:10s} "
                f"| ρ={row['pearson_r']:+.3f} | β={row['beta']:+.4f} "
                f"| R²={row['r_squared']:.3f} | p={row['p_value']:.4f} {star}"
            )

    # 3. 多變量回歸
    print("\n\n🔬 三、多變量回歸結果")
    print("-" * 60)
    for result in multi_reg_results:
        if "error" in result:
            print(f"   {result.get('target', '?')}: {result['error']}")
            continue
        print(f"\n   目標: {result['target']}")
        print(f"   R²={result['r_squared']:.4f}, Adj R²={result.get('adj_r_squared', 'N/A')}")
        print(f"   樣本數: {result['n_samples']}")
        print(f"   顯著維度:")
        for dim, coeff in result["coefficients"].items():
            if coeff.get("significant", False):
                print(
                    f"      {DIM_NAMES_ZH.get(dim, dim):8s}: "
                    f"β={coeff['beta']:+.4f}, p={coeff.get('p_value', 'N/A')}"
                )

    # 4. 產業分群
    print("\n\n🏭 四、產業分群分析")
    print("-" * 60)
    for group_name, data in industry_results.items():
        print(f"\n   【{group_name}】({', '.join(data['symbols'])}), n={data['n_samples']}")
        pairs = data.get("significant_pairs", [])
        if not pairs:
            print("      無顯著相關")
        else:
            for pair in sorted(pairs, key=lambda x: x["p_value"])[:5]:
                dim_zh = DIM_NAMES_ZH.get(pair["dimension"], pair["dimension"])
                print(
                    f"      {dim_zh} → {pair['return_window']}: "
                    f"ρ={pair['pearson_r']:+.3f}, p={pair['p_value']:.4f}"
                )


def main():
    parser = argparse.ArgumentParser(description="線性回歸與相關性分析")
    parser.add_argument(
        "--target-windows",
        default="R_10,R_20,alpha_10,alpha_20",
        help="多變量回歸的目標報酬欄位（逗號分隔）",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 載入資料
    print("📂 載入合併資料集 ...")
    df = load_merged_data()
    print(f"   筆數: {len(df)}")
    print(f"   公司: {df['transcript_symbol'].nunique()}")

    dims = get_available_dims(df)
    return_cols = get_return_columns(df)

    if not dims:
        print("❌ 資料中無情緒維度欄位。請確認已執行 02 Gemini 情緒分析。")
        sys.exit(1)

    print(f"   情緒維度: {len(dims)} 個")
    print(f"   報酬欄位: {len(return_cols)} 個")

    # 1. Pearson 相關係數
    print("\n📈 計算 Pearson 相關係數矩陣 ...")
    corr_matrix, pval_matrix = pearson_correlation_matrix(df, dims, return_cols)

    # 存相關矩陣
    corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    pval_matrix.to_csv(OUTPUT_DIR / "pvalue_matrix.csv")
    print("   ✅ 已存: correlation_matrix.csv, pvalue_matrix.csv")

    # 2. 單變量回歸
    print("\n📐 執行單變量線性回歸 ...")
    regression_df = single_variable_regression(df, dims, return_cols)
    regression_df.to_csv(OUTPUT_DIR / "single_regression_results.csv", index=False)
    print(f"   ✅ 已存: single_regression_results.csv ({len(regression_df)} 筆)")

    # 3. 多變量回歸
    print("\n🔬 執行多變量回歸 ...")
    target_windows = [t.strip() for t in args.target_windows.split(",")]
    multi_reg_results = []
    for target in target_windows:
        if target in df.columns:
            result = multi_variable_regression(df, dims, target)
            multi_reg_results.append(result)

    with open(OUTPUT_DIR / "multi_regression_results.json", "w", encoding="utf-8") as f:
        json.dump(multi_reg_results, f, ensure_ascii=False, indent=2, default=str)
    print("   ✅ 已存: multi_regression_results.json")

    # 4. 產業分群
    print("\n🏭 產業分群分析 ...")
    industry_results = industry_group_analysis(df, dims, return_cols)
    with open(OUTPUT_DIR / "industry_group_analysis.json", "w", encoding="utf-8") as f:
        json.dump(industry_results, f, ensure_ascii=False, indent=2, default=str)
    print("   ✅ 已存: industry_group_analysis.json")

    # 5. 列印報告
    print_report(corr_matrix, pval_matrix, regression_df, multi_reg_results, industry_results)

    print("\n\n🎉 分析完成！所有結果已存至 output/ 目錄")


if __name__ == "__main__":
    main()

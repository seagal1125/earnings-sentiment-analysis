#!/usr/bin/env python3
"""
05_visualize.py
產出視覺化圖表與互動式 HTML 報告：
1. 12×N 相關係數熱力圖
2. 重要維度散佈圖 + 回歸線
3. 情緒得分時序圖
4. 維度重要性排名
5. 互動式 HTML 儀表板（echarts）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("請先安裝 matplotlib: pip install matplotlib")
    sys.exit(1)

try:
    from scipy import stats
except ImportError:
    print("請先安裝 scipy: pip install scipy")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
MERGED_DIR = ROOT / "data" / "merged"

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

DIM_NAMES_ZH = {
    "overall_sentiment": "整體情緒",
    "confidence_score": "管理層信心",
    "forward_guidance_tone": "前瞻語氣",
    "risk_disclosure_intensity": "風險揭露",
    "qa_evasiveness": "QA閃躍",
    "quantitative_specificity": "數據具體度",
    "capex_enthusiasm": "資本支出",
    "demand_strength": "需求強度",
    "pricing_power": "定價能力",
    "competitive_moat": "競爭優勢",
    "innovation_momentum": "創新動能",
    "management_consistency": "言行一致",
}

# 嘗試載入中文字型
def setup_chinese_font():
    """設定 matplotlib 中文字型"""
    font_candidates = [
        "PingFang TC",
        "Noto Sans CJK TC",
        "Microsoft JhengHei",
        "Arial Unicode MS",
        "Heiti TC",
    ]
    for font_name in font_candidates:
        fonts = [f for f in fm.fontManager.ttflist if font_name in f.name]
        if fonts:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    # 若找不到指定字型，使用系統預設
    plt.rcParams["axes.unicode_minus"] = False
    return None


def load_data():
    """載入所有分析結果"""
    merged_csv = MERGED_DIR / "sentiment_returns_merged.csv"
    corr_csv = OUTPUT_DIR / "correlation_matrix.csv"
    pval_csv = OUTPUT_DIR / "pvalue_matrix.csv"
    regr_csv = OUTPUT_DIR / "single_regression_results.csv"

    data = {}
    if merged_csv.exists():
        data["merged"] = pd.read_csv(merged_csv)
    if corr_csv.exists():
        data["corr"] = pd.read_csv(corr_csv, index_col=0)
    if pval_csv.exists():
        data["pval"] = pd.read_csv(pval_csv, index_col=0)
    if regr_csv.exists():
        data["regression"] = pd.read_csv(regr_csv)

    return data


def plot_correlation_heatmap(corr_df: pd.DataFrame, pval_df: pd.DataFrame, output_path: Path):
    """繪製相關係數熱力圖"""
    # 轉換為中文標籤
    zh_labels = [DIM_NAMES_ZH.get(d, d) for d in corr_df.index]

    fig, ax = plt.subplots(figsize=(14, 10))

    # 自訂色彩映射
    colors = ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#ffffbf",
              "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850"]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

    data = corr_df.values.astype(float)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-1, vmax=1)

    # 設定軸標籤
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(zh_labels)))
    ax.set_yticklabels(zh_labels, fontsize=10)

    # 在格子中標數值與顯著性
    for i in range(len(zh_labels)):
        for j in range(len(corr_df.columns)):
            val = data[i, j]
            pval = pval_df.values[i, j] if i < pval_df.shape[0] and j < pval_df.shape[1] else 1
            if np.isnan(val):
                continue
            star = ""
            if pval < 0.01:
                star = "***"
            elif pval < 0.05:
                star = "**"
            elif pval < 0.10:
                star = "*"
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold" if star else "normal")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontsize=11)

    ax.set_title("法說會情緒維度 vs 報酬率 — 相關係數矩陣\n(* p<0.10, ** p<0.05, *** p<0.01)",
                 fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   ✅ 熱力圖: {output_path.name}")


def plot_scatter_top_dims(merged_df: pd.DataFrame, regression_df: pd.DataFrame, output_path: Path):
    """繪製最顯著維度的散佈圖 + 回歸線"""
    # 找出最顯著的 6 個配對
    sig = regression_df.sort_values("p_value").head(6)
    if sig.empty:
        print("   ⚠️ 無顯著相關，跳過散佈圖")
        return

    n_plots = min(6, len(sig))
    cols = 3
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes] if n_plots == 1 else axes
    axes_flat = np.array(axes).flatten()

    for idx, (_, row) in enumerate(sig.iterrows()):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        dim = row["dimension"]
        ret_col = row["return_window"]
        valid = merged_df[[dim, ret_col]].dropna()

        if valid.empty:
            ax.set_visible(False)
            continue

        # 依公司上色
        if "transcript_symbol" in merged_df.columns:
            valid_with_sym = merged_df[[dim, ret_col, "transcript_symbol"]].dropna()
            symbols = valid_with_sym["transcript_symbol"].unique()
            colors = plt.cm.Set2(np.linspace(0, 1, len(symbols)))
            for sym, color in zip(symbols, colors):
                mask = valid_with_sym["transcript_symbol"] == sym
                subset = valid_with_sym[mask]
                ax.scatter(subset[dim], subset[ret_col], c=[color], alpha=0.7,
                          s=40, label=sym, edgecolors="white", linewidths=0.5)
        else:
            ax.scatter(valid[dim], valid[ret_col], alpha=0.6, s=40,
                      edgecolors="white", linewidths=0.5)

        # 回歸線
        x = valid[dim].values
        y = valid[ret_col].values
        slope, intercept, r, p, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, alpha=0.8)

        dim_zh = DIM_NAMES_ZH.get(dim, dim)
        ax.set_xlabel(dim_zh, fontsize=10)
        ax.set_ylabel(ret_col, fontsize=10)
        ax.set_title(f"{dim_zh} vs {ret_col}\nρ={r:.3f}, p={p:.4f}", fontsize=11)
        ax.legend(fontsize=7, loc="best", framealpha=0.7)
        ax.grid(True, alpha=0.3)

    # 隱藏空白子圖
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("最顯著情緒維度 vs 報酬率 散佈圖", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   ✅ 散佈圖: {output_path.name}")


def plot_importance_ranking(regression_df: pd.DataFrame, output_path: Path):
    """繪製維度重要性排名（按平均 |ρ|）"""
    if regression_df.empty:
        return

    # 只看報酬率（R_*），不看 alpha
    r_cols_mask = regression_df["return_window"].str.startswith("R_")
    filtered = regression_df[r_cols_mask].copy()

    avg_abs_r = (
        filtered.groupby("dimension")["pearson_r"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(avg_abs_r)))
    zh_labels = [DIM_NAMES_ZH.get(d, d) for d in avg_abs_r.index]

    bars = ax.barh(zh_labels, avg_abs_r.values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, avg_abs_r.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel("平均 |Pearson ρ|", fontsize=12)
    ax.set_title("情緒維度預測力排名\n（平均絕對相關係數，越高越有預測力）", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   ✅ 重要性: {output_path.name}")


def build_html_dashboard(data: dict, output_path: Path):
    """產出互動式 HTML 儀表板"""
    corr_df = data.get("corr")
    regression_df = data.get("regression")
    merged_df = data.get("merged")

    # 準備 echarts 需要的資料
    if corr_df is not None:
        dims_zh = [DIM_NAMES_ZH.get(d, d) for d in corr_df.index]
        ret_cols = list(corr_df.columns)
        heatmap_data = []
        for i, dim in enumerate(corr_df.index):
            for j, col in enumerate(ret_cols):
                val = corr_df.loc[dim, col]
                if pd.notna(val):
                    heatmap_data.append([j, i, round(float(val), 3)])
    else:
        dims_zh, ret_cols, heatmap_data = [], [], []

    # 維度重要性
    importance_data = []
    if regression_df is not None and not regression_df.empty:
        r_mask = regression_df["return_window"].str.startswith("R_")
        avg_r = (
            regression_df[r_mask]
            .groupby("dimension")["pearson_r"]
            .apply(lambda x: x.abs().mean())
            .sort_values(ascending=False)
        )
        importance_data = [
            {"name": DIM_NAMES_ZH.get(d, d), "value": round(float(v), 4)}
            for d, v in avg_r.items()
        ]

    # 統計摘要
    n_samples = len(merged_df) if merged_df is not None else 0
    n_companies = merged_df["transcript_symbol"].nunique() if merged_df is not None and "transcript_symbol" in merged_df.columns else 0
    n_significant = 0
    if regression_df is not None and "p_value" in regression_df.columns:
        n_significant = int((regression_df["p_value"] < 0.05).sum())

    chart_json = json.dumps({
        "heatmap": {"x": ret_cols, "y": dims_zh, "data": heatmap_data},
        "importance": importance_data,
    }, ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>財報情緒密碼 — 相關性分析儀表板</title>
  <meta name="description" content="法說會情緒量化 vs 公佈後報酬的線性回歸分析互動式報告">
  <meta property="og:title" content="財報情緒密碼 — 相關性分析儀表板">
  <meta property="og:description" content="12維度Gemini情緒分析 × 多時間窗口報酬率 × 線性回歸">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Noto+Sans+TC:wght@300;400;500;700;900&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{
      --bg: #0a0a14;
      --bg-card: #12121f;
      --text: #eef0f8;
      --muted: #8890a8;
      --accent: #ff4d6a;
      --accent2: #36d399;
      --accent3: #6c8aff;
      --border: #1e1e35;
      --glow: rgba(255,77,106,0.15);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', 'Noto Sans TC', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      background-image:
        radial-gradient(circle at 10% 20%, var(--glow), transparent 30%),
        radial-gradient(circle at 90% 80%, rgba(108,138,255,0.08), transparent 30%);
    }}
    .wrap {{ max-width: 1300px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }}
    .hero {{
      text-align: center;
      padding: 3rem 2rem;
      background: linear-gradient(135deg, rgba(255,77,106,0.08), rgba(108,138,255,0.08));
      border: 1px solid var(--border);
      border-radius: 24px;
      margin-bottom: 2rem;
    }}
    .hero h1 {{ font-size: 2.4rem; font-weight: 900; margin-bottom: 0.5rem; }}
    .hero h1 span {{ background: linear-gradient(135deg, var(--accent), var(--accent3)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .hero p {{ color: var(--muted); max-width: 700px; margin: 0 auto; }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .stat-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.5rem;
      text-align: center;
    }}
    .stat-card .num {{ font-size: 2.5rem; font-weight: 800; color: var(--accent); }}
    .stat-card .label {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem; }}
    .section {{ margin-bottom: 2rem; }}
    .section h2 {{
      font-size: 1.5rem;
      font-weight: 700;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .section h2::before {{
      content: '';
      display: block;
      width: 4px;
      height: 28px;
      background: linear-gradient(180deg, var(--accent), var(--accent3));
      border-radius: 2px;
    }}
    .chart-panel {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 1.5rem;
      box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }}
    .chart-box {{ width: 100%; height: 500px; }}
    .chart-box.short {{ height: 380px; }}
    .chart-grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 1.5rem;
    }}
    .note {{
      margin-top: 1.5rem;
      padding: 1rem 1.5rem;
      background: rgba(255,77,106,0.05);
      border-left: 3px solid var(--accent);
      border-radius: 0 12px 12px 0;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    footer {{
      text-align: center;
      padding: 2rem;
      color: var(--muted);
      font-size: 0.85rem;
      border-top: 1px solid var(--border);
      margin-top: 3rem;
    }}
    @media (max-width: 900px) {{
      .chart-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>財報<span>情緒密碼</span></h1>
      <p>法說會逐字稿 × Gemini 12維度情緒量化 × 線性回歸 — 找出文字裡真正能預測股價的密碼</p>
    </div>

    <div class="stats-grid">
      <div class="stat-card"><div class="num">{n_samples}</div><div class="label">分析樣本數</div></div>
      <div class="stat-card"><div class="num">{n_companies}</div><div class="label">分析公司數</div></div>
      <div class="stat-card"><div class="num">12</div><div class="label">情緒量化維度</div></div>
      <div class="stat-card"><div class="num">{n_significant}</div><div class="label">顯著相關配對 (p&lt;0.05)</div></div>
    </div>

    <div class="section">
      <h2>相關係數熱力圖</h2>
      <div class="chart-panel">
        <div id="heatmapChart" class="chart-box"></div>
      </div>
    </div>

    <div class="section">
      <h2>維度預測力排名</h2>
      <div class="chart-grid">
        <div class="chart-panel">
          <div id="importanceChart" class="chart-box short"></div>
        </div>
        <div class="chart-panel">
          <h3 style="margin-bottom:1rem;color:var(--muted);font-size:1rem;">解讀提示</h3>
          <ul style="color:var(--muted);padding-left:1.2rem;line-height:2;">
            <li>平均 |ρ| 越高，該維度與報酬率的線性關係越強</li>
            <li>但高相關不代表因果，需觀察 p-value</li>
            <li>TSM 使用 0050.TW 基準、M7 使用 SPY 基準</li>
            <li>超額報酬（α）才是真正的預測力指標</li>
          </ul>
          <div class="note">
            💡 建議重點關注「需求強度」「定價能力」「前瞻語氣」三個維度，
            這些維度在過往研究中通常有較強的預測力。
          </div>
        </div>
      </div>
    </div>

    <footer>
      <p>Little Lobster Quant — 財報情緒密碼分析系統</p>
      <p style="margin-top:0.3rem;opacity:0.5;">*投資一定有風險，本分析僅供教學研究使用，不構成任何買賣建議。</p>
    </footer>
  </div>

  <script>
    const DATA = {chart_json};

    // 熱力圖
    const heatChart = echarts.init(document.getElementById('heatmapChart'));
    heatChart.setOption({{
      tooltip: {{
        position: 'top',
        formatter: p => `${{DATA.heatmap.y[p.value[1]]}} → ${{DATA.heatmap.x[p.value[0]]}}<br/>ρ = ${{p.value[2]}}`
      }},
      grid: {{ left: 120, right: 60, top: 30, bottom: 80 }},
      xAxis: {{
        type: 'category',
        data: DATA.heatmap.x,
        axisLabel: {{ rotate: 45, color: '#8890a8', fontSize: 11 }},
        splitArea: {{ show: true }}
      }},
      yAxis: {{
        type: 'category',
        data: DATA.heatmap.y,
        axisLabel: {{ color: '#8890a8', fontSize: 11 }},
        splitArea: {{ show: true }}
      }},
      visualMap: {{
        min: -1, max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        inRange: {{ color: ['#d73027','#fee08b','#1a9850'] }},
        textStyle: {{ color: '#8890a8' }}
      }},
      series: [{{
        type: 'heatmap',
        data: DATA.heatmap.data,
        label: {{
          show: true,
          formatter: p => p.value[2] != null ? p.value[2].toFixed(2) : '',
          fontSize: 9,
          color: '#fff'
        }},
        emphasis: {{
          itemStyle: {{ shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' }}
        }}
      }}]
    }});

    // 重要性排名
    const impChart = echarts.init(document.getElementById('importanceChart'));
    const impNames = DATA.importance.map(d => d.name);
    const impValues = DATA.importance.map(d => d.value);
    impChart.setOption({{
      tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
      grid: {{ left: 100, right: 40, top: 20, bottom: 20 }},
      xAxis: {{
        type: 'value',
        axisLabel: {{ color: '#8890a8' }},
        splitLine: {{ lineStyle: {{ color: '#1e1e35' }} }}
      }},
      yAxis: {{
        type: 'category',
        data: impNames.reverse(),
        axisLabel: {{ color: '#8890a8', fontSize: 11 }}
      }},
      series: [{{
        type: 'bar',
        data: impValues.reverse(),
        itemStyle: {{
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            {{ offset: 0, color: '#6c8aff' }},
            {{ offset: 1, color: '#ff4d6a' }}
          ]),
          borderRadius: [0, 6, 6, 0]
        }},
        label: {{
          show: true,
          position: 'right',
          formatter: '{{c}}',
          color: '#8890a8',
          fontSize: 10
        }}
      }}]
    }});

    // 響應式
    window.addEventListener('resize', () => {{
      heatChart.resize();
      impChart.resize();
    }});
  </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   ✅ 儀表板: {output_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font = setup_chinese_font()
    print(f"🎨 視覺化報告產出")
    print(f"   中文字型: {font or '系統預設'}")
    print()

    data = load_data()

    if not data:
        print("❌ 找不到分析結果，請先執行 04_regression_analysis.py")
        sys.exit(1)

    # 1. 熱力圖
    if "corr" in data and "pval" in data:
        print("📊 繪製相關係數熱力圖 ...")
        plot_correlation_heatmap(
            data["corr"], data["pval"],
            OUTPUT_DIR / "correlation_heatmap.png"
        )

    # 2. 散佈圖
    if "merged" in data and "regression" in data:
        print("📊 繪製散佈圖 ...")
        plot_scatter_top_dims(
            data["merged"], data["regression"],
            OUTPUT_DIR / "scatter_plots.png"
        )

    # 3. 重要性排名
    if "regression" in data:
        print("📊 繪製重要性排名 ...")
        plot_importance_ranking(
            data["regression"],
            OUTPUT_DIR / "importance_ranking.png"
        )

    # 4. HTML 儀表板
    print("📊 產出互動式 HTML 儀表板 ...")
    build_html_dashboard(data, OUTPUT_DIR / "sentiment_dashboard.html")

    print(f"\n🎉 視覺化完成！所有圖表已存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

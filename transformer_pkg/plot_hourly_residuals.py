"""按小时分组绘制预测/真实/残差曲线，24小时生成24张图。"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = r"D:\Junshan_Project\transformer_pkg\eval_rolling\predictions_unique.csv"
OUT_DIR = r"D:\Junshan_Project\transformer_pkg\eval_rolling\hourly_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 加载数据 ──
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["date"] = df["timestamp"].dt.date
# 过滤 true=0 的行，避免除零
df = df[df["true"].abs() > 1e-6].copy()
df["mape"] = (df["true"] - df["pred"]) / df["true"] * 100  # 百分比误差

# ── 统计全局范围，保持 y 轴一致 ──
flow_min = min(df["true"].min(), df["pred"].min()) * 0.95
flow_max = max(df["true"].max(), df["pred"].max()) * 1.05
mape_max = max(abs(df["mape"].min()), abs(df["mape"].max())) * 1.1

# ── 逐小时绘图 ──
for hour in range(24):
    sub = df[df["hour"] == hour].sort_values("timestamp")
    if sub.empty:
        continue

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ts = sub["timestamp"]
    mape_mean = sub["mape"].mean()
    mape_abs_mean = sub["mape"].abs().mean()

    # ── 子图1: 真实值 vs 预测值 ──
    ax1 = axes[0]
    ax1.plot(ts, sub["true"], label="True", linewidth=1.2, marker="o",
             markersize=3, alpha=0.85)
    ax1.plot(ts, sub["pred"], label="Pred", linewidth=1.2, marker="s",
             markersize=3, alpha=0.85)
    ax1.set_ylabel("Flow (m³/h)", fontsize=11)
    ax1.set_ylim(flow_min, flow_max)
    ax1.set_title(f"Hour {hour:02d}:00 — True vs Predicted  "
                  f"(MAPE={mape_abs_mean:.1f}%)",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── 子图2: MAPE 残差 ──
    ax2 = axes[1]
    colors = ["#e74c3c" if r < 0 else "#27ae60" for r in sub["mape"]]
    ax2.bar(ts, sub["mape"], width=0.03, color=colors, alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.axhline(y=mape_mean, color="orange", linestyle="--",
                linewidth=1, label=f"mean={mape_mean:.2f}%")
    ax2.set_ylabel("Error (%)", fontsize=11)
    ax2.set_ylim(-mape_max, mape_max)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_title(f"Hour {hour:02d}:00 — Prediction Error (MAPE%)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # x 轴日期格式
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, f"hour_{hour:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {path}")

print(f"\n完成，共 {len(os.listdir(OUT_DIR))} 张图，保存在: {OUT_DIR}")

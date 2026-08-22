"""绘制 MAPE 频数直方图: 总体 + 每月各一张。"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = r"D:\Junshan_Project\transformer_pkg\eval_rolling\predictions_unique.csv"
OUT_DIR = r"D:\Junshan_Project\transformer_pkg\eval_rolling\mape_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 加载数据 ──
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df[df["true"].abs() > 1e-6].copy()
df["mape"] = (df["true"] - df["pred"]) / df["true"] * 100
df["month"] = df["timestamp"].dt.to_period("M")


def plot_mape_hist(data, title, path):
    """画单张 MAPE 直方图。"""
    residual = data["mape"]
    abs_mean = residual.abs().mean()
    std_val = residual.std()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residual, bins=80, edgecolor="white", alpha=0.7, color="steelblue")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, label="zero")
    ax.axvline(x=abs_mean, color="orange", linestyle="--", linewidth=1,
               label=f"MAPE={abs_mean:.2f}%")
    ax.axvline(x=-abs_mean, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=std_val, color="green", linestyle=":", linewidth=1,
               label=f"σ={std_val:.2f}%")
    ax.axvline(x=-std_val, color="green", linestyle=":", linewidth=1)
    ax.set_xlabel("Residual MAPE (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{title}  (n={len(residual)}, MAPE={abs_mean:.2f}%, σ={std_val:.2f}%)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: {path}")


# ── 1. 总体 MAPE 直方图 ──
print("总体 MAPE 直方图")
plot_mape_hist(df, "Overall Residual MAPE Distribution",
               os.path.join(OUT_DIR, "overall.png"))

# ── 2. 每月 MAPE 直方图 ──
print("\n每月 MAPE 直方图")
for month, sub in df.groupby("month"):
    plot_mape_hist(sub, f"Residual MAPE — {month}",
                   os.path.join(OUT_DIR, f"{month}.png"))

print(f"\n完成，共 {1 + df['month'].nunique()} 张图，保存在: {OUT_DIR}")

"""清洗前后数据对比: 按每个时间点(0~23时)画出流量随时间变化曲线, 一共24个子图"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, r"D:\Junshan_Project")
from transformer_pkg.data_processing import DataProcessor

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============ 1. Load raw data ============
file_path = r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv"
df_raw = pd.read_csv(file_path, encoding="utf-8-sig")
df_raw["时间"] = pd.to_datetime(df_raw["时间"])
df_raw = df_raw.sort_values("时间").set_index("时间")
flow_cols = [c for c in df_raw.columns if "流量" in c]
df_raw = df_raw.rename(columns={flow_cols[0]: "Total_Flow"})
df_raw = df_raw[~df_raw["Total_Flow"].isna()].copy()
print(f"Raw data: {df_raw.shape}, range: {df_raw.index.min()} ~ {df_raw.index.max()}")

# ============ 2. Load cleaned data ============
config = {
    "file_path": file_path,
    "encoding": "utf-8-sig",
    "resample_freq": "60min",
    "stride": 6,
    "hampel_cols": ["Total_Flow"],
    "hampel_window": 48,
    "spike_ratio": 2.0,
    "lookback_days": 7,
    "predict_days": 1.0,
    "test_days": 90,
    "mape_floor_ratio": 0.1,
    "target_transform": None,
}

processor = DataProcessor(config)
df_base = processor.load_raw()
df_base = processor.build_base_features(df_base)

test_days = config.get("test_days", 15)
test_start = df_base.index[-1] - pd.Timedelta(days=test_days)
df_base_train = df_base.loc[df_base.index <= test_start].copy()
df_base_test = df_base.loc[df_base.index > test_start].copy()

df_clean = pd.concat([
    processor.clean_and_resample(df_base_train),
    processor.clean_and_resample(df_base_test),
])
print(f"Cleaned data: {df_clean.shape}, range: {df_clean.index.min()} ~ {df_clean.index.max()}")

# ============ 3. Plot 24 subplots (one per hour) ============
save_dir = r"D:\Junshan_Project\data\before_after_cleaning"
os.makedirs(save_dir, exist_ok=True)

df_raw["hour"] = df_raw.index.hour
df_clean["hour"] = df_clean.index.hour

for h in range(24):
    raw_h = df_raw[df_raw["hour"] == h]["Total_Flow"]
    clean_h = df_clean[df_clean["hour"] == h]["Total_Flow"]

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(raw_h.index, raw_h.values, color="#d3d3d3", linewidth=0.5,
            alpha=0.7, label="Before cleaning (raw)")
    ax.plot(clean_h.index, clean_h.values, color="#2c3e50", linewidth=0.7,
            alpha=0.9, label="After cleaning")

    # Mark removed outliers in red
    raw_idx = set(raw_h.index)
    clean_idx = set(clean_h.index)
    removed = raw_idx - clean_idx
    if removed:
        removed_series = raw_h.loc[sorted(removed)]
        ax.scatter(removed_series.index, removed_series.values,
                   color="#e74c3c", s=8, zorder=5, label="Removed points")

    n_removed = len(raw_h) - len(clean_h)
    ax.set_title(f"Hour {h:02d}:00  (raw={len(raw_h)}, clean={len(clean_h)}, removed={n_removed})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Flow", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"flow_hour_{h:02d}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"\n图已保存: {save_dir}/flow_hour_00.png ~ flow_hour_23.png (共24张)")

diff = len(df_raw) - len(df_clean)
print(f"数据量变化: 原始 {len(df_raw)} -> 清洗后 {len(df_clean)} (去除 {diff} 条, {diff/len(df_raw)*100:.2f}%)")

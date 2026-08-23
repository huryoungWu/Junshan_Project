import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data/水厂2025年小时级汇总.csv')
df.columns = ['时间', '流量']
df['时间'] = pd.to_datetime(df['时间'])
df = df.set_index('时间').sort_index()
df = df[df['流量'] > 10]

flow = df['流量']

# ============================================================
# 突变定义: |flow[t] - flow[t-1]| > k * rolling_std
# ============================================================
diff = flow.diff().abs()
roll_std = flow.rolling(48, min_periods=12).std()

# 主方法: 2x rolling_std
spike_2x = diff > 2 * roll_std
# 3x rolling_std (更严格)
spike_3x = diff > 3 * roll_std
# 固定阈值
spike_500 = diff > 500
# 相对变化
spike_pct = (diff / flow.shift(1).replace(0, np.nan)) > 0.15

print("=== 突变检测统计 ===")
print(f"  固定 >500:      {spike_500.sum():5d} ({spike_500.mean()*100:.2f}%)")
print(f"  2x rolling_std: {spike_2x.sum():5d} ({spike_2x.mean()*100:.2f}%)")
print(f"  3x rolling_std: {spike_3x.sum():5d} ({spike_3x.mean()*100:.2f}%)")
print(f"  相对变化 >15%:  {spike_pct.sum():5d} ({spike_pct.mean()*100:.2f}%)")

spike = spike_2x
df["spike"] = spike.values
df["diff"] = diff.values
df["hour"] = df.index.hour
df["dow"] = df.index.dayofweek
df["month"] = df.index.month
df["diff_signed"] = diff.values * np.sign(flow.diff().values)

# ============================================================
# 1. 按小时统计突变率
# ============================================================
hourly_spike = df.groupby("hour")["spike"].agg(["mean", "sum", "count"])
hourly_spike.columns = ["rate", "count", "total"]
hourly_spike["rate_pct"] = hourly_spike["rate"] * 100

print("\n=== 按小时突变率 ===")
for h in range(24):
    r = hourly_spike.loc[h]
    bar = "#" * int(r["rate_pct"] * 2)
    print(f"  {h:02d}:00  {r['rate_pct']:5.1f}%  ({int(r['count']):3d}/{int(r['total'])}) {bar}")

# ============================================================
# 2. 按星期几统计突变率
# ============================================================
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_spike = df.groupby("dow")["spike"].agg(["mean", "sum", "count"])
dow_spike.columns = ["rate", "count", "total"]
dow_spike["rate_pct"] = dow_spike["rate"] * 100

print("\n=== 按星期几突变率 ===")
for d in range(7):
    r = dow_spike.loc[d]
    bar = "#" * int(r["rate_pct"] * 2)
    print(f"  {dow_names[d]}  {r['rate_pct']:5.1f}%  ({int(r['count']):3d}/{int(r['total'])}) {bar}")

# ============================================================
# 3. 月份突变率
# ============================================================
monthly_spike = df.groupby("month")["spike"].mean() * 100
print("\n=== 按月份突变率 ===")
for m in range(1, 13):
    r = monthly_spike[m]
    bar = "#" * int(r * 2)
    print(f"  {m:2d}月  {r:5.1f}% {bar}")

# ============================================================
# 4. (星期几, 小时) 突变率热力图
# ============================================================
pivot = df.groupby(["dow", "hour"])["spike"].mean().unstack(fill_value=0) * 100
# pivot shape: (7, 24), index=dow 0-6, columns=hour 0-23

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# 4a. 热力图
ax = axes[0, 0]
data = pivot.values
im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0)
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=9)
ax.set_yticks(range(7))
ax.set_yticklabels(dow_names, fontsize=10)
ax.set_xlabel("Hour", fontsize=11)
ax.set_title("Spike Rate Heatmap (DOW x Hour)", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Spike Rate (%)", shrink=0.8)
for i in range(7):
    for j in range(24):
        val = data[i, j]
        color = "white" if val > data.max() * 0.55 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=color)

# 4b. 按小时柱状图
ax = axes[0, 1]
mean_rate = hourly_spike["rate_pct"].mean()
std_rate = hourly_spike["rate_pct"].std()
colors_h = ["#e74c3c" if r > mean_rate + std_rate else "#3498db" for r in hourly_spike["rate_pct"]]
ax.bar(range(24), hourly_spike["rate_pct"], color=colors_h, alpha=0.85, edgecolor="white")
ax.axhline(mean_rate, color="red", linestyle="--", alpha=0.7, label=f"Mean={mean_rate:.1f}%")
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}" for h in range(24)])
ax.set_xlabel("Hour", fontsize=11)
ax.set_ylabel("Spike Rate (%)", fontsize=11)
ax.set_title("Spike Rate by Hour of Day", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# 4c. 按星期几柱状图
ax = axes[1, 0]
mean_dow = dow_spike["rate_pct"].mean()
std_dow = dow_spike["rate_pct"].std()
colors_d = ["#e74c3c" if r > mean_dow + std_dow else "#2ecc71" if r < mean_dow - std_dow else "#f39c12" for r in dow_spike["rate_pct"]]
ax.bar(range(7), dow_spike["rate_pct"], color=colors_d, alpha=0.85, edgecolor="white")
ax.axhline(mean_dow, color="red", linestyle="--", alpha=0.7, label=f"Mean={mean_dow:.1f}%")
ax.set_xticks(range(7))
ax.set_xticklabels(dow_names)
ax.set_xlabel("Day of Week", fontsize=11)
ax.set_ylabel("Spike Rate (%)", fontsize=11)
ax.set_title("Spike Rate by Day of Week", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# 4d. 按月份柱状图
ax = axes[1, 1]
m_colors = plt.cm.RdYlGn_r(monthly_spike.values / max(monthly_spike.values.max(), 1))
ax.bar(range(1, 13), monthly_spike.values, color=m_colors, alpha=0.85, edgecolor="white")
ax.axhline(monthly_spike.mean(), color="red", linestyle="--", alpha=0.7, label=f"Mean={monthly_spike.mean():.1f}%")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([f"{m}" for m in range(1, 13)])
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Spike Rate (%)", fontsize=11)
ax.set_title("Spike Rate by Month", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("flow_plots/spike_analysis.png", dpi=150)
plt.close()
print("\nspike_analysis.png saved")

# ============================================================
# 5. 突变方向: 向上 vs 向下
# ============================================================
spike_up = (df["spike"] & (df["diff_signed"] > 0)).groupby(df["hour"]).mean() * 100
spike_down = (df["spike"] & (df["diff_signed"] < 0)).groupby(df["hour"]).mean() * 100

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(np.arange(24) - 0.2, spike_up.values, width=0.4, color="#e74c3c", alpha=0.8, label="Spike UP (flow increases)")
ax.bar(np.arange(24) + 0.2, spike_down.values, width=0.4, color="#3498db", alpha=0.8, label="Spike DOWN (flow decreases)")
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}" for h in range(24)])
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("Spike Rate (%)", fontsize=12)
ax.set_title("Spike Direction by Hour (UP vs DOWN)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("flow_plots/spike_direction.png", dpi=150)
plt.close()
print("spike_direction.png saved")

# ============================================================
# 6. 突变幅度分布 (按小时 boxplot)
# ============================================================
spike_data = df[df["spike"]]
fig, ax = plt.subplots(figsize=(14, 6))
box_data = [spike_data[spike_data["hour"] == h]["diff"].dropna().values for h in range(24)]
bp = ax.boxplot(box_data, labels=[f"{h:02d}" for h in range(24)], showfliers=False)
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("|dFlow| (m3/h)", fontsize=12)
ax.set_title("Spike Magnitude Distribution by Hour", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("flow_plots/spike_magnitude.png", dpi=150)
plt.close()
print("spike_magnitude.png saved")

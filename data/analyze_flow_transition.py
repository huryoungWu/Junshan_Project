# -*- coding: utf-8 -*-
"""统计每个小时到下一个小时流量增/减的概率。

对每天的 24 小时流量序列, 逐对比较相邻小时:
  - h → h+1 流量增加计为 "增"
  - h → h+1 流量减少或不变计为 "减"

汇总所有天的统计, 得到每个时间点的增/减概率。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ── 加载数据 ──
csv_path = os.path.join(os.path.dirname(__file__), "水厂2025年小时级汇总.csv")
df = pd.read_csv(csv_path, encoding="utf-8-sig")
df["时间"] = pd.to_datetime(df["时间"])
df = df.set_index("时间").sort_index()

flow = df["出厂水流量"].dropna()

# ── 按天分组, 每天一条24小时序列 ──
flow.index = pd.DatetimeIndex(flow.index)
dates = sorted(set(flow.index.date))

# 统计: 对于每个起始小时 h (0~23), 记录 h → h+1 的增/减次数
# 注意: 23:00 → 次日 00:00 是跨天的, 需要特殊处理
increase_count = np.zeros(24, dtype=int)   # increase_count[h] = h→h+1 增的次数
decrease_count = np.zeros(24, dtype=int)   # decrease_count[h] = h→h+1 减的次数
total_count = np.zeros(24, dtype=int)

# 全量逐对统计 (不按天分组, 直接用连续时间序列)
hours = flow.index.hour.values
values = flow.values

for i in range(len(values) - 1):
    h_from = hours[i]
    h_to = hours[i + 1]
    # 只统计相邻小时 (h → h+1 或 23→0)
    expected_to = (h_from + 1) % 24
    if h_to != expected_to:
        continue  # 跳过不连续的 (跨天缺口等)
    total_count[h_from] += 1
    if values[i + 1] > values[i]:
        increase_count[h_from] += 1
    else:
        decrease_count[h_from] += 1

# ── 打印结果 ──
print("=" * 70)
print(" 每小时到下一小时 流量增/减概率统计")
print("=" * 70)
print(f"  {'起始小时':<12}{'→目标':<8}{'总次数':<8}{'增次数':<8}{'减次数':<8}"
      f"{'增概率':<10}{'减概率':<10}")
print(f"  {'-' * 62}")

for h in range(24):
    h_next = (h + 1) % 24
    total = total_count[h]
    if total == 0:
        continue
    inc = increase_count[h]
    dec = decrease_count[h]
    inc_pct = inc / total * 100
    dec_pct = dec / total * 100
    print(f"  {h:02d}:00→{h_next:02d}:00   {total:<8}{inc:<8}{dec:<8}"
          f"{inc_pct:<10.1f}{dec_pct:<10.1f}")

# ── 按月份分组统计 ──
print(f"\n{'=' * 70}")
print(" 各月份 每小时增/减概率")
print(f"{'=' * 70}")

months = sorted(set(pd.DatetimeIndex(flow.index).month))
for m in months:
    m_mask = pd.DatetimeIndex(flow.index).month == m
    m_flow = flow[m_mask]
    m_hours = m_flow.index.hour.values
    m_values = m_flow.values

    m_inc = np.zeros(24, dtype=int)
    m_dec = np.zeros(24, dtype=int)
    m_total = np.zeros(24, dtype=int)

    for i in range(len(m_values) - 1):
        h_from = m_hours[i]
        h_to = m_hours[i + 1]
        expected_to = (h_from + 1) % 24
        if h_to != expected_to:
            continue
        m_total[h_from] += 1
        if m_values[i + 1] > m_values[i]:
            m_inc[h_from] += 1
        else:
            m_dec[h_from] += 1

    print(f"\n  【{m}月】")
    print(f"  {'起始小时':<12}{'总次数':<8}{'增%':<10}{'减%':<10}")
    print(f"  {'-' * 38}")
    for h in range(24):
        total = m_total[h]
        if total == 0:
            continue
        h_next = (h + 1) % 24
        print(f"  {h:02d}→{h_next:02d}       {total:<8}"
              f"{m_inc[h]/total*100:<10.1f}{m_dec[h]/total*100:<10.1f}")

# ── 画图 ──
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 图1: 增/减概率柱状图
ax = axes[0]
x = np.arange(24)
width = 0.35
inc_pcts = [increase_count[h] / total_count[h] * 100 if total_count[h] > 0 else 0 for h in range(24)]
dec_pcts = [decrease_count[h] / total_count[h] * 100 if total_count[h] > 0 else 0 for h in range(24)]

bars1 = ax.bar(x - width/2, inc_pcts, width, label="增概率 (%)", color="#e74c3c", alpha=0.8)
bars2 = ax.bar(x + width/2, dec_pcts, width, label="减概率 (%)", color="#3498db", alpha=0.8)

ax.set_xlabel("起始小时")
ax.set_ylabel("概率 (%)")
ax.set_title("每小时到下一小时 流量增/减概率")
ax.set_xticks(x)
ax.set_xticklabels([f"{h:02d}→{(h+1)%24:02d}" for h in range(24)], fontsize=8, rotation=45)
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# 标注数值
for bar, val in zip(bars1, inc_pcts):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=7)

# 图2: 各月份热力图 (增概率)
ax = axes[1]
month_inc_matrix = np.full((len(months), 24), np.nan)

for idx, m in enumerate(months):
    m_mask = pd.DatetimeIndex(flow.index).month == m
    m_flow = flow[m_mask]
    m_hours = m_flow.index.hour.values
    m_values = m_flow.values

    m_inc = np.zeros(24, dtype=int)
    m_total = np.zeros(24, dtype=int)

    for i in range(len(m_values) - 1):
        h_from = m_hours[i]
        h_to = m_hours[i + 1]
        if h_to != (h_from + 1) % 24:
            continue
        m_total[h_from] += 1
        if m_values[i + 1] > m_values[i]:
            m_inc[h_from] += 1

    for h in range(24):
        if m_total[h] > 0:
            month_inc_matrix[idx, h] = m_inc[h] / m_total[h] * 100

im = ax.imshow(month_inc_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}→{(h+1)%24:02d}" for h in range(24)], fontsize=8, rotation=45)
ax.set_yticks(range(len(months)))
ax.set_yticklabels([f"{m}月" for m in months])
ax.set_xlabel("起始小时")
ax.set_ylabel("月份")
ax.set_title("各月份 每小时增概率热力图 (%)")
plt.colorbar(im, ax=ax, label="增概率 (%)")

# 在热力图格子中标注数值
for i in range(len(months)):
    for j in range(24):
        val = month_inc_matrix[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6,
                    color="black" if 30 < val < 70 else "white")

fig.suptitle("军山水厂 流量增/减概率分析", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig_path = os.path.join(os.path.dirname(__file__), "flow_transition_probability.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n图表已保存: {fig_path}")

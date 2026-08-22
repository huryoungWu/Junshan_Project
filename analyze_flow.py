# -*- coding: utf-8 -*-
"""
水厂2025年流量与压力数据分析
1. 分析流量和压力的相关系数
2. 分析24个时间点的不同日期的流量是否随时间变化明显
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 数据加载 ============
df = pd.read_csv(r'D:\Junshan_Project\data\水厂2025年小时级汇总.csv')
df['时间'] = pd.to_datetime(df['时间'])

# 丢弃缺失值
df = df.dropna(subset=['出厂水流量', '出厂水压力']).reset_index(drop=True)
print(f"丢弃缺失值后剩余: {len(df)} 条")

df['小时'] = df['时间'].dt.hour
df['月份'] = df['时间'].dt.month
df['日期'] = df['时间'].dt.date

print("=" * 60)
print("一、数据基本信息")
print("=" * 60)
print(f"数据时间范围: {df['时间'].min()} ~ {df['时间'].max()}")
print(f"数据总量: {len(df)} 条 (约{len(df)/24:.0f}天)")
print(f"\n列名: {df.columns.tolist()}")
print(f"\n统计描述:\n{df[['出厂水流量','出厂水压力']].describe()}")

# ============ 2. 流量与压力相关系数分析 ============
print("\n" + "=" * 60)
print("二、流量与压力相关系数分析")
print("=" * 60)

pearson_r, pearson_p = stats.pearsonr(df['出厂水流量'], df['出厂水压力'])
spearman_r, spearman_p = stats.spearmanr(df['出厂水流量'], df['出厂水压力'])
kendall_r, kendall_p = stats.kendalltau(df['出厂水流量'], df['出厂水压力'])

print(f"\nPearson 相关系数:  r = {pearson_r:.4f}, p = {pearson_p:.2e}")
print(f"Spearman 秩相关:  r = {spearman_r:.4f}, p = {spearman_p:.2e}")
print(f"Kendall 秩相关:   r = {kendall_r:.4f}, p = {kendall_p:.2e}")

corr_matrix = df[['出厂水流量','出厂水压力']].corr()
print(f"\n相关矩阵:\n{corr_matrix}")

# 图1: 散点图 + 回归线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.scatter(df['出厂水流量'], df['出厂水压力'], alpha=0.1, s=5, c='steelblue')
z = np.polyfit(df['出厂水流量'], df['出厂水压力'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['出厂水流量'].min(), df['出厂水流量'].max(), 100)
ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Pearson r={pearson_r:.3f}')
ax1.set_xlabel('出厂水流量 (m³/h)')
ax1.set_ylabel('出厂水压力 (MPa)')
ax1.set_title('出厂水流量 vs 压力 散点图')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: 各月份相关系数
ax2 = axes[1]
month_corrs = []
for m in range(1, 13):
    sub = df[df['月份'] == m]
    if len(sub) > 10:
        r, _ = stats.pearsonr(sub['出厂水流量'], sub['出厂水压力'])
        month_corrs.append((m, r))
months, corrs = zip(*month_corrs)
colors = ['green' if c > 0 else 'red' for c in corrs]
ax2.bar(months, corrs, color=colors, alpha=0.7)
ax2.set_xlabel('月份')
ax2.set_ylabel('Pearson 相关系数')
ax2.set_title('各月份流量-压力相关系数')
ax2.set_xticks(range(1, 13))
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data/flow_pressure_correlation.png', dpi=150, bbox_inches='tight')
print("\n图1已保存: data/flow_pressure_correlation.png")

# ============ 3. 24小时流量模式分析 ============
print("\n" + "=" * 60)
print("三、24个时间点的流量模式分析")
print("=" * 60)

# 计算每个时间点的统计量
hourly_stats = df.groupby('小时')['出厂水流量'].agg(['mean', 'std', 'median', 'min', 'max'])
print(f"\n各小时流量统计:")
print(hourly_stats.round(2))

# 计算各小时的变异系数(CV)
hourly_stats['CV'] = (hourly_stats['std'] / hourly_stats['mean'] * 100)
print(f"\n各小时变异系数(CV%):\n{hourly_stats['CV'].round(2)}")

# ANOVA: 检验不同小时的流量是否有显著差异
groups = [group['出厂水流量'].values for name, group in df.groupby('小时')]
f_stat, p_value = stats.f_oneway(*groups)
print(f"\n单因素ANOVA检验 (不同小时): F = {f_stat:.2f}, p = {p_value:.2e}")
if p_value < 0.05:
    print("=> 不同小时的流量存在显著差异 (p < 0.05)")
else:
    print("=> 不同小时的流量无显著差异 (p >= 0.05)")

# Kruskal-Wallis检验 (非参数)
h_stat, kw_p = stats.kruskal(*groups)
print(f"\nKruskal-Wallis检验: H = {h_stat:.2f}, p = {kw_p:.2e}")

# 图3: 24小时平均流量 + 置信区间
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
hour_means = df.groupby('小时')['出厂水流量'].mean()
hour_stds = df.groupby('小时')['出厂水流量'].std()
ax.fill_between(range(24), hour_means - hour_stds, hour_means + hour_stds, alpha=0.2, color='steelblue')
ax.plot(range(24), hour_means, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.set_xlabel('小时')
ax.set_ylabel('平均流量 (m³/h)')
ax.set_title('24小时平均流量模式 (阴影=±1标准差)')
ax.set_xticks(range(24))
ax.grid(True, alpha=0.3)
peak_hour = hour_means.idxmax()
valley_hour = hour_means.idxmin()
ax.annotate(f'峰值: {peak_hour}:00\n({hour_means[peak_hour]:.0f})',
            xy=(peak_hour, hour_means[peak_hour]),
            xytext=(peak_hour+2, hour_means[peak_hour]+200),
            arrowprops=dict(arrowstyle='->', color='red'), color='red')
ax.annotate(f'谷值: {valley_hour}:00\n({hour_means[valley_hour]:.0f})',
            xy=(valley_hour, hour_means[valley_hour]),
            xytext=(valley_hour+2, hour_means[valley_hour]-200),
            arrowprops=dict(arrowstyle='->', color='blue'), color='blue')

# 图4: 各月份24小时流量热力图
ax = axes[0, 1]
pivot = df.groupby(['月份', '小时'])['出厂水流量'].mean().unstack()
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, cbar_kws={'label': '流量 (m³/h)'})
ax.set_title('各月份24小时平均流量热力图')
ax.set_ylabel('月份')
ax.set_xlabel('小时')

# 图5: 各小时箱线图
ax = axes[1, 0]
df.boxplot(column='出厂水流量', by='小时', ax=ax, grid=False)
ax.set_title('各小时流量分布箱线图')
ax.set_xlabel('小时')
ax.set_ylabel('流量 (m³/h)')
plt.sca(ax)
plt.title('各小时流量分布箱线图')

# 图6: 流量时间序列 + 24小时滚动平均
ax = axes[1, 1]
ax.plot(df['时间'], df['出厂水流量'], alpha=0.3, linewidth=0.5, label='原始数据')
rolling_24h = df['出厂水流量'].rolling(24).mean()
rolling_7d = df['出厂水流量'].rolling(168).mean()
ax.plot(df['时间'], rolling_24h, linewidth=1.5, label='24h滚动平均')
ax.plot(df['时间'], rolling_7d, linewidth=2.5, label='7天滚动平均')
ax.set_xlabel('时间')
ax.set_ylabel('流量 (m³/h)')
ax.set_title('流量时间序列趋势')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/flow_24h_pattern.png', dpi=150, bbox_inches='tight')
print("\n图2已保存: data/flow_24h_pattern.png")

# ============ 4. 不同日期的流量变化分析 ============
print("\n" + "=" * 60)
print("四、不同日期流量随时间变化分析")
print("=" * 60)

# 按日统计
daily_stats = df.groupby('日期')['出厂水流量'].agg(['mean', 'std', 'sum'])
daily_stats['CV'] = daily_stats['std'] / daily_stats['mean'] * 100

print(f"\n日均流量统计:")
print(f"  平均值: {daily_stats['mean'].mean():.2f}")
print(f"  标准差: {daily_stats['mean'].std():.2f}")
print(f"  日均流量CV: {daily_stats['mean'].std()/daily_stats['mean'].mean()*100:.2f}%")

# 不同星期几的流量
df['星期'] = df['时间'].dt.dayofweek
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
weekday_stats = df.groupby('星期')['出厂水流量'].agg(['mean', 'std'])
weekday_stats.index = [weekday_names[i] for i in weekday_stats.index]
print(f"\n各星期流量统计:\n{weekday_stats.round(2)}")

# 周末vs工作日
df['是否周末'] = df['星期'].apply(lambda x: '周末' if x >= 5 else '工作日')
weekend_comp = df.groupby('是否周末')['出厂水流量'].agg(['mean', 'std', 'median'])
print(f"\n工作日 vs 周末:\n{weekend_comp.round(2)}")

# T检验: 工作日 vs 周末
weekday_data = df[df['是否周末'] == '工作日']['出厂水流量']
weekend_data = df[df['是否周末'] == '周末']['出厂水流量']
t_stat, t_p = stats.ttest_ind(weekday_data, weekend_data)
print(f"\n工作日vs周末 T检验: t = {t_stat:.2f}, p = {t_p:.2e}")

# 图7: 各星期24小时流量对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for day in range(7):
    sub = df[df['星期'] == day]
    hourly = sub.groupby('小时')['出厂水流量'].mean()
    ax.plot(range(24), hourly.values, 'o-', linewidth=1.5, markersize=4,
            label=weekday_names[day], alpha=0.8)
ax.set_xlabel('小时')
ax.set_ylabel('平均流量 (m³/h)')
ax.set_title('各星期24小时流量模式对比')
ax.legend(ncol=4, fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(24))

# 图8: 月均流量趋势
ax = axes[1]
monthly = df.groupby('月份')['出厂水流量'].agg(['mean', 'std'])
ax.bar(monthly.index, monthly['mean'], yerr=monthly['std'], capsize=5,
       color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('月份')
ax.set_ylabel('平均流量 (m³/h)')
ax.set_title('2025年各月平均流量')
ax.set_xticks(range(1, 13))
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data/flow_weekly_monthly.png', dpi=150, bbox_inches='tight')
print("\n图3已保存: data/flow_weekly_monthly.png")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)

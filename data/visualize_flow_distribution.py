import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv(r'D:\Junshan_Project\data\水厂2025年小时级汇总.csv')

# 将时间列转换为datetime类型
df['时间'] = pd.to_datetime(df['时间'])

# 提取小时信息
df['小时'] = df['时间'].dt.hour

# 处理NaN值：用该小时的平均值填充
df['出厂水流量'] = df.groupby('小时')['出厂水流量'].transform(lambda x: x.fillna(x.mean()))

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 每小时平均流量及其标准差
hourly_mean = df.groupby('小时')['出厂水流量'].mean()
hourly_std = df.groupby('小时')['出厂水流量'].std()
hourly_min = df.groupby('小时')['出厂水流量'].min()
hourly_max = df.groupby('小时')['出厂水流量'].max()
hourly_p25 = df.groupby('小时')['出厂水流量'].quantile(0.25)
hourly_p75 = df.groupby('小时')['出厂水流量'].quantile(0.75)
hourly_p99 = df.groupby('小时')['出厂水流量'].quantile(0.99)

# 图1: 每小时平均流量和标准差
ax1 = axes[0, 0]
ax1.errorbar(hourly_mean.index, hourly_mean.values, yerr=hourly_std.values,
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
             color='#2196F3', ecolor='#90CAF9', label='平均值 ± 标准差')
ax1.fill_between(hourly_mean.index, hourly_mean.values - hourly_std.values,
                 hourly_mean.values + hourly_std.values, alpha=0.2, color='#2196F3')
ax1.set_xlabel('小时', fontsize=12)
ax1.set_ylabel('出厂水流量', fontsize=12)
ax1.set_title('每小时平均流量及标准差', fontsize=14, fontweight='bold')
ax1.set_xticks(range(0, 24))
ax1.grid(True, alpha=0.3)
ax1.legend()

# 图2: 每小时流量范围 (Min, P25, Median, P75, Max)
ax2 = axes[0, 1]
median = df.groupby('小时')['出厂水流量'].median()
hours = range(0, 24)

ax2.fill_between(hours, hourly_p25.values, hourly_p75.values, alpha=0.3, color='#4CAF50', label='P25-P75')
ax2.plot(hours, hourly_mean.values, 'o-', color='#2196F3', linewidth=2, markersize=6, label='平均值')
ax2.plot(hours, median.values, 's--', color='#FF9800', linewidth=2, markersize=6, label='中位数(P50)')
ax2.plot(hours, hourly_min.values, '^:', color='#F44336', linewidth=1.5, markersize=5, label='最小值')
ax2.plot(hours, hourly_max.values, 'v:', color='#9C27B0', linewidth=1.5, markersize=5, label='最大值')

ax2.set_xlabel('小时', fontsize=12)
ax2.set_ylabel('出厂水流量', fontsize=12)
ax2.set_title('每小时流量分布范围', fontsize=14, fontweight='bold')
ax2.set_xticks(range(0, 24))
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

# 图3: P99 vs 平均值对比
ax3 = axes[1, 0]
x = np.arange(24)
width = 0.35

bars1 = ax3.bar(x - width/2, hourly_mean.values, width, label='平均值', color='#2196F3', alpha=0.8)
bars2 = ax3.bar(x + width/2, hourly_p99.values, width, label='P99', color='#F44336', alpha=0.8)

ax3.set_xlabel('小时', fontsize=12)
ax3.set_ylabel('出厂水流量', fontsize=12)
ax3.set_title('每小时平均值 vs P99', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(range(0, 24))
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend()

# 图4: 热力图 - 每小时流量分布
ax4 = axes[1, 1]

# 创建箱线图数据
data_by_hour = [df[df['小时'] == h]['出厂水流量'].values for h in range(24)]

bp = ax4.boxplot(data_by_hour, patch_artist=True, labels=range(0, 24))

# 为箱线图添加颜色
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 24))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax4.set_xlabel('小时', fontsize=12)
ax4.set_ylabel('出厂水流量', fontsize=12)
ax4.set_title('每小时流量箱线图', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(r'D:\Junshan_Project\data\flow_distribution_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("图表已保存到: D:\\Junshan_Project\\data\\flow_distribution_analysis.png")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ========== 中文字体设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']   # Windows 黑体；macOS 可用 'PingFang SC'，Linux 可用 'WenQuanYi Micro Hei'
plt.rcParams['axes.unicode_minus'] = False

# ================== 读取 Excel ==================
file_path = input("请输入 Excel 文件路径: ").strip()
df = pd.read_excel(file_path)

# 检查列名
if '时间' not in df.columns or '出厂水流量' not in df.columns:
    raise ValueError("Excel 中必须包含 '时间' 和 '出厂水流量' 两列")

# 转换时间列
df['时间'] = pd.to_datetime(df['时间'])
df.set_index('时间', inplace=True)

# 提取日期和小时
df['日期'] = df.index.date
df['小时'] = df.index.hour

# ================== 创建输出目录 ==================
output_dir = "flow_plots"
hourly_dir = os.path.join(output_dir, "hourly")
os.makedirs(hourly_dir, exist_ok=True)   # 自动创建目录，如果存在则忽略

print(f"所有图片将保存至: {os.path.abspath(output_dir)}")

# ================== 绘图1：每日小时流量曲线叠加（保存） ==================
plt.figure(figsize=(12, 6))
for date, group in df.groupby('日期'):
    group = group.sort_values('小时')
    plt.plot(group['小时'], group['出厂水流量'], label=date.strftime('%Y-%m-%d'), alpha=0.7)

plt.xlabel('小时 (0-23)')
plt.ylabel('出厂水流量')
plt.title('每日小时流量曲线叠加图')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "daily_overlay.png"), dpi=150, bbox_inches='tight')
plt.close()   # 关闭图形，释放内存

# ================== 绘图2：每日汇总图（总流量）柱状图（保存） ==================
daily_total = df.groupby('日期')['出厂水流量'].sum()

plt.figure(figsize=(12, 5))
daily_total.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('日期')
plt.ylabel('日总流量')
plt.title('每日出厂水总流量汇总')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "daily_total_bar.png"), dpi=150, bbox_inches='tight')
plt.close()

# ================== 绘图3：全天各小时平均流量（保存） ==================
hourly_avg = df.groupby('小时')['出厂水流量'].mean()

plt.figure(figsize=(12, 5))
hourly_avg.plot(marker='o', linestyle='-', color='red')
plt.xlabel('小时')
plt.ylabel('平均出厂水流量')
plt.title('全天各小时平均流量（汇总）')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "hourly_average.png"), dpi=150, bbox_inches='tight')
plt.close()

# ================== 绘图4：每个小时的流量日变化图（保存到子文件夹） ==================
available_hours = sorted(df['小时'].unique())
print(f"正在生成并保存 {len(available_hours)} 张小时流量日变化图...")

for hour in available_hours:
    subset = df[df['小时'] == hour].sort_values('日期')
    
    plt.figure(figsize=(12, 5))
    plt.plot(subset['日期'], subset['出厂水流量'], marker='o', linestyle='-', color='steelblue', linewidth=1.5)
    
    plt.xlabel('日期')
    plt.ylabel('出厂水流量')
    plt.title(f'{hour:02d}:00 时流量日变化')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 格式化日期刻度
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    save_path = os.path.join(hourly_dir, f'hour_{hour:02d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

print("所有图片已保存完毕！")
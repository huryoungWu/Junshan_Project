import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv(r'D:\Junshan_Project\data\水厂2025年小时级汇总.csv')

# 查看数据基本信息
print("=" * 60)
print("数据基本信息")
print("=" * 60)
print(f"数据形状: {df.shape}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\n前10行数据:\n{df.head(10)}")

# 将时间列转换为datetime类型
df['时间'] = pd.to_datetime(df['时间'])

# 提取小时信息
df['小时'] = df['时间'].dt.hour

# 处理NaN值：用该小时的平均值填充
print("\n" + "=" * 60)
print("处理NaN值")
print("=" * 60)
nan_count_before = df['出厂水流量'].isnull().sum()
print(f"处理前NaN数量: {nan_count_before}")

# 对每个小时分组，用该组的均值填充NaN
df['出厂水流量'] = df.groupby('小时')['出厂水流量'].transform(lambda x: x.fillna(x.mean()))

nan_count_after = df['出厂水流量'].isnull().sum()
print(f"处理后NaN数量: {nan_count_after}")

# 按小时分组统计
print("\n" + "=" * 60)
print("每个时间点(小时)的流量分布统计")
print("=" * 60)

# 计算每个小时的统计量
hourly_stats = df.groupby('小时')['出厂水流量'].agg([
    'count', 'mean', 'std', 'min', 'max',
    lambda x: x.quantile(0.01),  # P1
    lambda x: x.quantile(0.05),  # P5
    lambda x: x.quantile(0.25),  # P25
    lambda x: x.quantile(0.50),  # P50/中位数
    lambda x: x.quantile(0.75),  # P75
    lambda x: x.quantile(0.90),  # P90
    lambda x: x.quantile(0.95),  # P95
    lambda x: x.quantile(0.99),  # P99
]).round(2)

# 重命名列
hourly_stats.columns = ['样本数', '平均值', '标准差', '最小值', '最大值',
                        'P1', 'P5', 'P25', '中位数(P50)', 'P75', 'P90', 'P95', 'P99']

print("\n各小时流量分布统计表:")
print("-" * 120)
print(hourly_stats.to_string())
print("-" * 120)

# 保存到CSV
hourly_stats.to_csv(r'D:\Junshan_Project\data\hourly_flow_distribution_stats.csv', encoding='utf-8-sig')
print("\n统计结果已保存到: D:\\Junshan_Project\\data\\hourly_flow_distribution_stats.csv")

# 整体统计
print("\n" + "=" * 60)
print("整体流量统计")
print("=" * 60)
overall_stats = df['出厂水流量'].describe()
print(overall_stats)

# 找出最高流量和最低流量的小时
print("\n" + "=" * 60)
print("流量特征总结")
print("=" * 60)
max_hour = hourly_stats['平均值'].idxmax()
min_hour = hourly_stats['平均值'].idxmin()
print(f"平均流量最高的小时: {max_hour}:00 (平均值: {hourly_stats.loc[max_hour, '平均值']:.2f})")
print(f"平均流量最低的小时: {min_hour}:00 (平均值: {hourly_stats.loc[min_hour, '平均值']:.2f})")

# 流量波动最大的小时
max_std_hour = hourly_stats['标准差'].idxmax()
print(f"流量波动最大的小时: {max_std_hour}:00 (标准差: {hourly_stats.loc[max_std_hour, '标准差']:.2f})")

# 流量波动最小的小时
min_std_hour = hourly_stats['标准差'].idxmin()
print(f"流量波动最小的小时: {min_std_hour}:00 (标准差: {hourly_stats.loc[min_std_hour, '标准差']:.2f})")

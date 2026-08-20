# -*- coding: utf-8 -*-
"""合并 data 目录下 4 个 Excel 为完整的小时级 CSV（2025 全年 8760 小时）。"""
import pandas as pd

DATA = 'D:/Junshan_Project/data'

# 1) 三个小时级文件（时间戳完全对齐，8755 行）
pressure = pd.read_excel(f'{DATA}/出厂水压力2025.xlsx')
flow     = pd.read_excel(f'{DATA}/出厂水流量（2025-01-01至2026-01-01).xlsx')
freq     = pd.read_excel(f'{DATA}/送水泵运行频率2025.xlsx')

for df in (pressure, flow, freq):
    df['时间'] = pd.to_datetime(df['时间'])

# 2) 完整小时时间轴：2025-01-01 00:00 -> 2025-12-31 23:00 共 8760 个整点
grid = pd.DataFrame({'时间': pd.date_range('2025-01-01', '2025-12-31 23:00', freq='1h')})

# 3) 小时级文件按时间轴对齐（4/14 17:00~21:00 缺失的 5 小时自然补 NaN）
merged = grid.merge(pressure, on='时间', how='left') \
             .merge(flow,     on='时间', how='left') \
             .merge(freq,     on='时间', how='left')

# 4) 送水泵运行状态：各泵独立轮询记录（NaN=该时刻未记录），用 ffill 推断每整点的状态
status = pd.read_excel(f'{DATA}/送水泵运行状态2025.xlsx')
status['时间'] = pd.to_datetime(status['时间'])
status = (status.set_index('时间')
               .sort_index()
               .ffill()
               .reindex(merged['时间'], method='ffill')
               .reset_index())
merged = merged.merge(status, on='时间', how='left')

# 5) 列顺序与命名统一
merged = merged[['时间',
                 '出厂水压力', '出厂水流量',
                 '1#泵运行频率', '6#泵运行频率',
                 '1#送水泵运行', '2#送水泵运行', '6#送水泵运行']]

out = f'{DATA}/水厂2025年小时级汇总.csv'
merged.to_csv(out, index=False, encoding='utf-8-sig')

# 6) 校验报告
print(f'输出: {out}')
print(f'总行数: {len(merged)}  (2025 全年应为 8760)')
print(f'时间戳重复: {merged["时间"].duplicated().sum()}')
print(f'时间范围: {merged["时间"].min()} -> {merged["时间"].max()}')
print('各列缺失值:')
print(merged.isna().sum().to_string())

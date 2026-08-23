"""合并2024年流量Excel与2025年流量CSV，删除不需要的列"""
import pandas as pd

# ---- 1. 读取2024年Excel，只保留时间+出水流量 ----
excel_path = r"D:\Junshan_Project\data\武汉军山 流量_2024-01-01至2025-01-01.xlsx"
df_2024 = pd.read_excel(excel_path)
print(f"2024 Excel 原始列: {list(df_2024.columns)}")
print(f"2024 Excel 行数: {len(df_2024)}")

# 只保留需要的列
df_2024 = df_2024[['时间', '出水流量']].copy()
df_2024 = df_2024.rename(columns={'出水流量': '出厂水流量'})
df_2024['时间'] = pd.to_datetime(df_2024['时间'])

# ---- 2. 读取2025年CSV，只保留时间+出厂水流量 ----
csv_path = r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv"
df_2025 = pd.read_csv(csv_path, encoding='utf-8-sig')
print(f"\n2025 CSV 原始列: {list(df_2025.columns)}")
print(f"2025 CSV 行数: {len(df_2025)}")

df_2025 = df_2025[['时间', '出厂水流量']].copy()
df_2025['时间'] = pd.to_datetime(df_2025['时间'])

# ---- 3. 合并: 2024在前，2025在后 ----
df_all = pd.concat([df_2024, df_2025], ignore_index=True)
df_all = df_all.sort_values('时间').drop_duplicates(subset='时间', keep='first').reset_index(drop=True)

print(f"\n合并后行数: {len(df_all)}")
print(f"时间范围: {df_all['时间'].min()} ~ {df_all['时间'].max()}")
print(f"列: {list(df_all.columns)}")
print(f"NaN数: {df_all.isna().sum().to_dict()}")

# ---- 4. 覆盖保存 ----
df_all.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n已保存: {csv_path}")

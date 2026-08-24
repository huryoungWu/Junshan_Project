"""诊断突变检测逻辑: 展示哪些突变点被捕获, 哪些漏掉了"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, r"D:\Junshan_Project")

# 加载原始数据
file_path = r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")
df["时间"] = pd.to_datetime(df["时间"])
df = df.sort_values("时间").set_index("时间")
flow_cols = [c for c in df.columns if "流量" in c]
df = df.rename(columns={flow_cols[0]: "Total_Flow"})
df = df[~df["Total_Flow"].isna()].copy()

f = df["Total_Flow"]
k = 2.0  # spike_ratio

# === 日内判据: t vs t-1h, t+1h ===
within_steps = 1
day_steps = 24

idx = df.index.to_series()
n1_ok = (idx - idx.shift(within_steps)) == pd.Timedelta(hours=1)
n2_ok = (idx.shift(-within_steps) - idx) == pd.Timedelta(hours=1)
p1_ok = (idx - idx.shift(day_steps)) == pd.Timedelta(hours=24)
p2_ok = (idx.shift(-day_steps) - idx) == pd.Timedelta(hours=24)

n1, n2 = f.shift(within_steps), f.shift(-within_steps)
p1, p2 = f.shift(day_steps), f.shift(-day_steps)

# 突变检测
ref_hi_w = pd.concat([n1, n2], axis=1).max(axis=1)
ref_lo_w = pd.concat([n1, n2], axis=1).min(axis=1)
within_hi = (f > k * ref_hi_w) & (ref_hi_w > 0) & n1_ok & n2_ok & n1.notna() & n2.notna()
within_lo = (f < ref_lo_w / k) & (ref_lo_w > 0) & n1_ok & n2_ok & n1.notna() & n2.notna()
within_anom = within_hi | within_lo

ref_hi_d = pd.concat([p1, p2], axis=1).max(axis=1)
ref_lo_d = pd.concat([p1, p2], axis=1).min(axis=1)
day_hi = (f > k * ref_hi_d) & (ref_hi_d > 0) & p1_ok & p2_ok & p1.notna() & p2.notna()
day_lo = (f < ref_lo_d / k) & (ref_lo_d > 0) & p1_ok & p2_ok & p1.notna() & p2.notna()
day_anom = day_hi | day_lo

# Hampel 检测
window = 48
ham_k = 10.0
med = f.rolling(window, center=False, min_periods=window//2).median()
res_abs = (f - med).abs()
mad = res_abs.rolling(window, center=False, min_periods=window//2).median()
scale = np.maximum(1.4826 * mad, 0.02 * med)
hampel_flag = (res_abs > ham_k * scale).fillna(False)

# === 寻找"看起来像突变但没被捕获"的点 ===
# 用日内邻居偏差定义"明显突变": |当前 - median(前后)| > 某阈值
median_neigh = pd.concat([n1, n2], axis=1).median(axis=1)
deviation = (f - median_neigh).abs()
dev_ratio = f / median_neigh.replace(0, np.nan)

# 明显突变: 偏差 > 500 且 比率 > 1.5 或 < 0.5
obvious_spike = (deviation > 500) & ((dev_ratio > 1.5) | (dev_ratio < 0.667))
obvious_spike = obvious_spike & f.notna() & median_neigh.notna()

caught_spike = obvious_spike & (within_anom | day_anom | hampel_flag)
missed_spike = obvious_spike & ~(within_anom | day_anom | hampel_flag)

print("=" * 90)
print("突变检测诊断报告")
print("=" * 90)
print(f"\n原始数据点: {len(f)}")
print(f"日内突变检测命中 (correct_flow_spikes): {int(within_anom.sum())} 条")
print(f"跨日突变检测命中 (correct_flow_spikes): {int(day_anom.sum())} 条")
print(f"Hampel 离群命中 (detect_outliers): {int(hampel_flag.sum())} 条")
print(f"总清洗: {int((within_anom | day_anom | hampel_flag).sum())} 条")

print(f"\n\"明显突变\" 定义: |当前 - median(前后邻居)| > 500 且 偏离比 > 1.5x 或 < 0.67x")
print(f"明显突变点总数: {int(obvious_spike.sum())}")
print(f"  已捕获: {int(caught_spike.sum())}")
print(f"  未捕获(漏检): {int(missed_spike.sum())}")

# === 漏检原因分析 ===
if missed_spike.sum() > 0:
    print(f"\n{'='*90}")
    print("漏检点详情 (前20个):")
    print(f"{'='*90}")
    print(f"{'时间':<22} {'原始值':>8} {'前后邻居均值':>12} {'偏差':>8} {'比率':>6} "
          f"{'日内判据':>8} {'跨日判据':>8} {'Hampel':>8}")
    print("-" * 90)

    missed_idx = missed_spike[missed_spike].index[:20]
    for t in missed_idx:
        cur = f[t]
        # 日内邻居
        t_idx = f.index.get_loc(t)
        nb1 = f.iloc[t_idx - 1] if t_idx > 0 else np.nan
        nb2 = f.iloc[t_idx + 1] if t_idx < len(f) - 1 else np.nan
        neighbors = [v for v in [nb1, nb2] if not np.isnan(v)]
        med_n = np.median(neighbors) if neighbors else np.nan

        # 判断为什么没捕获
        in_flag = within_anom.get(t, False)
        day_flag = day_anom.get(t, False)
        ham_flag = hampel_flag.get(t, False)

        # 分析漏检原因
        reasons = []
        if not in_flag:
            # 日内为什么没捕获?
            if not n1_ok.get(t, False) or not n2_ok.get(t, False):
                reasons.append("邻居间隔不对")
            elif pd.isna(n1.get(t)) or pd.isna(n2.get(t)):
                reasons.append("邻居NaN")
            else:
                n1v, n2v = n1[t], n2[t]
                ref_hi = max(n1v, n2v)
                ref_lo = min(n1v, n2v)
                if not (cur > k * ref_hi) and not (cur < ref_lo / k):
                    reasons.append(f"邻居值({n1v:.0f},{n2v:.0f})偏差不够k={k}")

        if not day_flag:
            if not p1_ok.get(t, False) or not p2_ok.get(t, False):
                reasons.append("跨日邻居间隔不对")
            elif pd.isna(p1.get(t)) or pd.isna(p2.get(t)):
                reasons.append("跨日邻居NaN")
            else:
                p1v, p2v = p1[t], p2[t]
                ref_hi_d = max(p1v, p2v)
                ref_lo_d = min(p1v, p2v)
                if not (cur > k * ref_hi_d) and not (cur < ref_lo_d / k):
                    reasons.append(f"跨日邻居值({p1v:.0f},{p2v:.0f})偏差不够k={k}")

        if not ham_flag:
            m = med.get(t, np.nan)
            s = scale.get(t, np.nan)
            reasons.append(f"Hampel偏差({abs(cur-m):.0f}) < k*MAD({ham_k*s:.0f})")

        print(f"{str(t):<22} {cur:>8.1f} {med_n:>12.1f} {abs(cur - med_n):>8.0f} "
              f"{cur/med_n if med_n > 0 else 0:>6.2f}x "
              f"{'Y' if in_flag else 'N':>8} {'Y' if day_flag else 'N':>8} "
              f"{'Y' if ham_flag else 'N':>8}  [{', '.join(reasons)}]")

# === 已捕获的点 ===
print(f"\n{'='*90}")
print("已捕获的突变点 (前10个):")
print(f"{'='*90}")
caught_idx = caught_spike[caught_spike].index[:10]
print(f"{'时间':<22} {'原始值':>8} {'前后邻居均值':>12} {'偏差':>8} {'比率':>6} "
      f"{'日内':>4} {'跨日':>4} {'Hampel':>6}")
print("-" * 80)
for t in caught_idx:
    cur = f[t]
    t_idx = f.index.get_loc(t)
    nb1 = f.iloc[t_idx - 1] if t_idx > 0 else np.nan
    nb2 = f.iloc[t_idx + 1] if t_idx < len(f) - 1 else np.nan
    neighbors = [v for v in [nb1, nb2] if not np.isnan(v)]
    med_n = np.median(neighbors) if neighbors else np.nan
    print(f"{str(t):<22} {cur:>8.1f} {med_n:>12.1f} {abs(cur - med_n):>8.0f} "
          f"{cur/med_n if med_n > 0 else 0:>6.2f}x "
          f"{'Y' if within_anom.get(t, False) else 'N':>4} "
          f"{'Y' if day_anom.get(t, False) else 'N':>4} "
          f"{'Y' if hampel_flag.get(t, False) else 'N':>6}")

# === Hampel 漏检分析 ===
print(f"\n{'='*90}")
print("Hampel 为什么没捕获这些突变点:")
print(f"{'='*90}")
missed_by_hampel = obvious_spike & ~hampel_flag
print(f"  明显突变但 Hampel 未捕获: {int(missed_by_hampel.sum())} 条")
if missed_by_hampel.sum() > 0:
    for t in missed_by_hampel[missed_by_hampel].index[:5]:
        cur = f[t]
        m = med.get(t, np.nan)
        s = scale.get(t, np.nan)
        print(f"  {t}  flow={cur:.0f}  median={m:.0f}  MAD*1.48={1.4826*s:.0f}  "
              f"deviation={abs(cur-m):.0f}  threshold={ham_k*s:.0f}  "
              f"deviation/threshold={abs(cur-m)/(ham_k*s) if ham_k*s > 0 else 0:.2f}")

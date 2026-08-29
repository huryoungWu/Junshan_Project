import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import timesfm
import pandas as pd
import numpy as np
from typing import Optional

try:
    from chinese_calendar import is_holiday, is_workday as _cal_is_workday
    _HAS_CHINESE_CALENDAR = True
except ImportError:
    _HAS_CHINESE_CALENDAR = False
    print("⚠ chinese-calendar 未安装, 节假日特征退化为简单周末判断")

# ── 数据清洗参数 (参考 data_processing.py) ──
HAMPEL_WINDOW = 48      # 滚动窗口 (2 天整点)
HAMPEL_K = 10.0         # 离群判据阈值
MAD_FLOOR_RATIO = 0.02  # scale 下限 = 中位数的 2%
FLOW_MIN = 0.0          # 流量物理下限
FLOW_MAX = 10000.0      # 流量物理上限


def detect_outliers(s, window=HAMPEL_WINDOW, k=HAMPEL_K, floor_ratio=MAD_FLOOR_RATIO):
    """Hampel 滤波: 基于滚动中位数 + MAD 的稳健离群检测。

    窗口 center=False: 判定 t 时刻只用 t 及之前的数据, 不引入未来数据。
    """
    med = s.rolling(window, center=False, min_periods=window // 2).median()
    res = (s - med).abs()
    mad = res.rolling(window, center=False, min_periods=window // 2).median()
    scale = np.maximum(1.4826 * mad, floor_ratio * med)
    flag = (res > k * scale).fillna(False)
    return flag


def correct_flow_spikes(df, freq_minutes=60, k_within=1.2, k_cross=2.0):
    """整点流量的突变检测 + 插值修正 (参考 data_processing.py)。

    两种突变判据:
      日内 (k_within): t 时刻与 t±1h 比较
      跨日 (k_cross): t 时刻与 t±24h 比较
    """
    if "出厂水流量" not in df.columns:
        return df

    out = df.copy()
    f = out["出厂水流量"]
    orig = f.copy()

    within_steps = max(1, 60 // freq_minutes)      # 1h 折算步数
    day_steps = (24 * 60) // freq_minutes          # 1 天折算步数

    idx = out.index.to_series()
    n1_ok = (idx - idx.shift(within_steps)) == pd.Timedelta(minutes=freq_minutes * within_steps)
    n2_ok = (idx.shift(-within_steps) - idx) == pd.Timedelta(minutes=freq_minutes * within_steps)
    p1_ok = (idx - idx.shift(day_steps)) == pd.Timedelta(minutes=freq_minutes * day_steps)
    p2_ok = (idx.shift(-day_steps) - idx) == pd.Timedelta(minutes=freq_minutes * day_steps)

    n1, n2 = f.shift(within_steps), f.shift(-within_steps)   # 日内参考: t±1h
    p1, p2 = f.shift(day_steps), f.shift(-day_steps)         # 跨日参考: t±1 天

    def flags(cur, r1, r2, ok1, ok2, k):
        ref_hi = pd.concat([r1, r2], axis=1).max(axis=1)
        ref_lo = pd.concat([r1, r2], axis=1).min(axis=1)
        hi = (cur > k * ref_hi) & (ref_hi > 0)
        lo = (cur < ref_lo / k) & (ref_lo > 0)
        return (hi | lo) & ok1 & ok2 & r1.notna() & r2.notna()

    within_anom = flags(f, n1, n2, n1_ok, n2_ok, k_within)
    day_anom = flags(f, p1, p2, p1_ok, p2_ok, k_cross)
    anomaly = within_anom | day_anom

    if anomaly.any():
        fix = pd.Series(np.nan, index=out.index)
        fix[within_anom] = (n1 + n2)[within_anom] / 2.0
        fix[day_anom & ~within_anom] = (p1 + p2)[day_anom & ~within_anom] / 2.0
        fix = fix[anomaly]
        n_anom = int(anomaly.sum())
        out.loc[anomaly, "出厂水流量"] = fix
        print(f"  突变流量插值修正: {n_anom} 条 ({n_anom / len(out):.3%}) "
              f"[日内 {int(within_anom.sum())} / 跨日 {int((day_anom & ~within_anom).sum())}]")

    return out


def clean_flow_data(df, hampel_window=HAMPEL_WINDOW, hampel_k=HAMPEL_K):
    """完整的数据清洗流程 (参考 data_processing.py 的 clean_and_resample)。

    清洗步骤:
    1. 突变流量插值修正
    2. Hampel 离群清洗
    3. 流量物理界限裁剪
    4. 删除空 bin
    """
    print("─" * 50)
    print("开始数据清洗...")
    print(f"  原始数据: {len(df)} 条")

    out = df.copy()

    # 步骤 1: 突变流量插值修正
    out = correct_flow_spikes(out)

    # 步骤 2: Hampel 离群清洗
    flag = detect_outliers(out["出厂水流量"], window=hampel_window, k=hampel_k)
    n_out = int(flag.sum())
    if n_out > 0:
        out.loc[flag, "出厂水流量"] = np.nan
        print(f"  Hampel 离群点: {n_out} 条 ({n_out / len(out):.3%}) 置 NaN")

    # 步骤 3: 流量物理界限裁剪
    s = out["出厂水流量"].copy()
    invalid = (s < FLOW_MIN) | (s > FLOW_MAX) | s.isna()
    n_invalid = invalid.sum()
    if n_invalid > 0:
        out = out[~invalid].copy()
        print(f"  物理界限裁剪/NaN 删除: {n_invalid} 条")

    print(f"  清洗后数据: {len(out)} 条 (删除 {len(df) - len(out)} 条)")
    print("─" * 50)

    return out


# ── 1. 读取原始数据 ──
df = pd.read_csv(
    r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv",
    encoding="utf-8-sig", parse_dates=["时间"], index_col="时间"
)

# ── 2. 数据清洗 ──
df_clean = clean_flow_data(df)

# 目标序列（出厂水流量）
y = df_clean["出厂水流量"].values.astype(np.float32)

# ── 2. 生成时间外生变量 (参考 data_processing.py 的 CALENDAR_COLS) ──
def add_calendar_features(df_index):
    """由 DatetimeIndex 确定性生成日历特征，训练/推理/未来同源"""
    h = df_index.hour.to_numpy()
    dow = df_index.dayofweek.to_numpy()
    month = df_index.month.to_numpy()
    doy = df_index.dayofyear.to_numpy()

    features = {
        "hour_sin": np.sin(2.0 * np.pi * h / 24.0).astype(np.float32),
        "hour_cos": np.cos(2.0 * np.pi * h / 24.0).astype(np.float32),
        "dow_sin": np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32),
        "dow_cos": np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32),
        "month_sin": np.sin(2.0 * np.pi * month / 12.0).astype(np.float32),
        "month_cos": np.cos(2.0 * np.pi * month / 12.0).astype(np.float32),
        "doy_sin": np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32),
        "doy_cos": np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32),
    }

    # 工作日/节假日特征
    dates = df_index.normalize()
    unique_dates = dates.unique()

    if _HAS_CHINESE_CALENDAR:
        workday_map = {d: float(_cal_is_workday(d.to_pydatetime().date()))
                       for d in unique_dates}
        holiday_map = {}
        for d in unique_dates:
            dt = d.to_pydatetime().date()
            is_special = (not _cal_is_workday(dt)) and d.dayofweek < 5
            holiday_map[d] = float(is_special)

        eve_map = {}
        next_map = {}
        for d in unique_dates:
            prev_d = d - pd.Timedelta(days=1)
            next_d = d + pd.Timedelta(days=1)
            try:
                prev_dt = prev_d.to_pydatetime().date()
                eve_map[d] = float(not _cal_is_workday(prev_dt) and prev_d.dayofweek < 5)
            except ValueError:
                eve_map[d] = 0.0
            try:
                next_dt = next_d.to_pydatetime().date()
                next_map[d] = float(not _cal_is_workday(next_dt) and next_d.dayofweek < 5)
            except ValueError:
                next_map[d] = 0.0

        features["is_workday"] = dates.map(workday_map).astype(np.float32)
        features["is_holiday"] = dates.map(holiday_map).astype(np.float32)
        features["holiday_eve"] = dates.map(eve_map).astype(np.float32)
        features["holiday_next"] = dates.map(next_map).astype(np.float32)
    else:
        is_weekend = (dow >= 5).astype(np.float32)
        features["is_workday"] = 1.0 - is_weekend
        features["is_holiday"] = np.float32(0.0)
        features["holiday_eve"] = np.float32(0.0)
        features["holiday_next"] = np.float32(0.0)

    return features


# 生成外生变量字典 (使用清洗后的索引)
covariates = add_calendar_features(df_clean.index)

# ── 3. 划分训练/测试 ──
test_days = 90
split_point = df_clean.index[-1] - pd.Timedelta(days=test_days)
split_idx = df_clean.index.searchsorted(split_point)

y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"训练集: {len(y_train)} 样本, 测试集: {len(y_test)} 样本")

# ── 4. 加载 TimesFM 模型 ──
CONTEXT_LEN = 336   # 回看 14 天 (336 小时) - 参数搜索最优
HORIZON = 1         # 每次预测 1 步 = 1 小时 - MAPE 最优 5.66%

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    r"D:\Junshan_Project\models\timesfm-2.5-200m-pytorch",
)

config = timesfm.ForecastConfig(
    max_context=CONTEXT_LEN,
    max_horizon=HORIZON,
    use_continuous_quantile_head=True,
    return_backcast=True,  # forecast_with_covariates 需要
)
model.compile(config)

# ── 5. 逐段预测 (每段用真实历史，不用预测值) ──
all_pred = []
n_train = len(y_train)
n_steps = len(y_test) // HORIZON  # 按 HORIZON 切分

for step_idx in range(n_steps):
    # 截取到当前段之前的真实历史
    hist_end = n_train + step_idx * HORIZON
    history = y[:hist_end].tolist()

    # 外生变量需要覆盖 0..hist_end+HORIZON-1
    cov_slice = {k: v[0:hist_end + HORIZON].tolist() for k, v in covariates.items()}

    # 预测 HORIZON 步
    point_outputs, quantile_outputs = model.forecast_with_covariates(
        inputs=[history],
        dynamic_numerical_covariates={k: [v] for k, v in cov_slice.items()},
    )
    # point_outputs[0] shape = (HORIZON,)
    all_pred.append(point_outputs[0].flatten())

y_pred = np.concatenate(all_pred)
y_true = y_test[:len(y_pred)]

# ── 6. 评估 MAPE ──
overall_mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
print(f"Overall MAPE: {overall_mape:.2f}%")

# 按天统计 MAPE
test_index = df_clean.index[split_idx:split_idx + len(y_true)]
df_eval = pd.DataFrame({
    "date": test_index.normalize(),
    "y_true": y_true,
    "y_pred": y_pred,
})

print(f"\n{'日期':<14} {'MAPE%':>8}  {'样本数':>6}")
print("-" * 32)

daily_mapes = []
for date, grp in df_eval.groupby("date"):
    t = grp["y_true"].values
    p = grp["y_pred"].values
    mape_d = np.mean(np.abs((t - p) / (t + 1e-8))) * 100
    daily_mapes.append(mape_d)
    print(f"{str(date.date()):<14} {mape_d:>8.2f}  {len(grp):>6}")

avg_daily_mape = np.mean(daily_mapes)
print("-" * 32)
print(f"{'平均每天 MAPE':<14} {avg_daily_mape:>8.2f}%")

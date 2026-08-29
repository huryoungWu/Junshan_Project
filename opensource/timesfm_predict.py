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

# ── 数据清洗参数 ──
HAMPEL_WINDOW = 48
HAMPEL_K = 10.0
MAD_FLOOR_RATIO = 0.02
FLOW_MIN = 0.0
FLOW_MAX = 10000.0

CONTEXT_LEN = 336   # 14 天
HORIZON = 24         # 预测 24 小时


def detect_outliers(s, window=HAMPEL_WINDOW, k=HAMPEL_K, floor_ratio=MAD_FLOOR_RATIO):
    """Hampel 滤波: 基于滚动中位数 + MAD 的稳健离群检测"""
    med = s.rolling(window, center=False, min_periods=window // 2).median()
    res = (s - med).abs()
    mad = res.rolling(window, center=False, min_periods=window // 2).median()
    scale = np.maximum(1.4826 * mad, floor_ratio * med)
    flag = (res > k * scale).fillna(False)
    return flag


def correct_flow_spikes(df, freq_minutes=60, k_within=1.2, k_cross=2.0):
    """整点流量的突变检测 + 插值修正"""
    if "出厂水流量" not in df.columns:
        return df

    out = df.copy()
    f = out["出厂水流量"]

    within_steps = max(1, 60 // freq_minutes)
    day_steps = (24 * 60) // freq_minutes

    idx = out.index.to_series()
    n1_ok = (idx - idx.shift(within_steps)) == pd.Timedelta(minutes=freq_minutes * within_steps)
    n2_ok = (idx.shift(-within_steps) - idx) == pd.Timedelta(minutes=freq_minutes * within_steps)
    p1_ok = (idx - idx.shift(day_steps)) == pd.Timedelta(minutes=freq_minutes * day_steps)
    p2_ok = (idx.shift(-day_steps) - idx) == pd.Timedelta(minutes=freq_minutes * day_steps)

    n1, n2 = f.shift(within_steps), f.shift(-within_steps)
    p1, p2 = f.shift(day_steps), f.shift(-day_steps)

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
        print(f"  突变流量插值修正: {n_anom} 条")

    return out


def clean_flow_data(df, hampel_window=HAMPEL_WINDOW, hampel_k=HAMPEL_K):
    """完整的数据清洗流程"""
    print("─" * 50)
    print("开始数据清洗...")
    print(f"  原始数据: {len(df)} 条")

    out = df.copy()
    out = correct_flow_spikes(out)

    flag = detect_outliers(out["出厂水流量"], window=hampel_window, k=hampel_k)
    n_out = int(flag.sum())
    if n_out > 0:
        out.loc[flag, "出厂水流量"] = np.nan
        print(f"  Hampel 离群点: {n_out} 条 置 NaN")

    s = out["出厂水流量"].copy()
    invalid = (s < FLOW_MIN) | (s > FLOW_MAX) | s.isna()
    n_invalid = invalid.sum()
    if n_invalid > 0:
        out = out[~invalid].copy()
        print(f"  物理界限裁剪/NaN 删除: {n_invalid} 条")

    print(f"  清洗后数据: {len(out)} 条 (删除 {len(df) - len(out)} 条)")
    print("─" * 50)
    return out


def add_calendar_features(df_index):
    """由 DatetimeIndex 生成日历特征"""
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

    dates = df_index.normalize()
    unique_dates = dates.unique()

    if _HAS_CHINESE_CALENDAR:
        workday_map = {d: float(_cal_is_workday(d.to_pydatetime().date()))
                       for d in unique_dates}
        holiday_map = {}
        eve_map = {}
        next_map = {}
        for d in unique_dates:
            dt = d.to_pydatetime().date()
            is_special = (not _cal_is_workday(dt)) and d.dayofweek < 5
            holiday_map[d] = float(is_special)
            prev_d = d - pd.Timedelta(days=1)
            next_d = d + pd.Timedelta(days=1)
            try:
                eve_map[d] = float(not _cal_is_workday(prev_d.to_pydatetime().date()) and prev_d.dayofweek < 5)
            except ValueError:
                eve_map[d] = 0.0
            try:
                next_map[d] = float(not _cal_is_workday(next_d.to_pydatetime().date()) and next_d.dayofweek < 5)
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


# ══════════════════════════════════════════════════════════════
#  主流程: 读取 30 天数据 → 取最近 14 天 → 预测未来 24 小时
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── 1. 读取 CSV ──
    csv_path = r"D:\Junshan_Project\transformer_pkg\input.csv"
    print(f"读取文件: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["时间"], index_col="时间")
    print(f"  共 {len(df)} 条数据, 时间范围: {df.index[0]} ~ {df.index[-1]}")

    # ── 2. 取最近 14 天 (CONTEXT_LEN = 336 小时) ──
    if len(df) > CONTEXT_LEN:
        df = df.iloc[-CONTEXT_LEN:]
        print(f"  截取最近 {CONTEXT_LEN} 小时 (14 天): {df.index[0]} ~ {df.index[-1]}")

    # ── 3. 数据清洗 ──
    df_clean = clean_flow_data(df)

    # ── 4. 目标序列 ──
    y = df_clean["出厂水流量"].values.astype(np.float32)

    # ── 5. 外生变量 (需要覆盖历史 + 未来 HORIZON) ──
    # 为未来 24 小时生成时间索引
    last_time = df_clean.index[-1]
    future_index = pd.date_range(start=last_time + pd.Timedelta(hours=1),
                                 periods=HORIZON, freq="h")
    full_index = df_clean.index.append(future_index)

    # 历史部分 + 未来部分的外生变量
    hist_cov = add_calendar_features(df_clean.index)
    future_cov = add_calendar_features(future_index)

    # 拼接: 历史 + 未来
    covariates = {}
    for k in hist_cov:
        covariates[k] = np.concatenate([hist_cov[k], future_cov[k]]).tolist()

    # ── 6. 加载模型并预测 ──
    print(f"\n加载 TimesFM 模型...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        r"D:\Junshan_Project\models\timesfm-2.5-200m-pytorch",
    )

    config = timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=HORIZON,
        use_continuous_quantile_head=True,
        return_backcast=True,
    )
    model.compile(config)

    print(f"预测中... (输入 {len(y)} 小时, 输出 {HORIZON} 小时)")
    point_outputs, quantile_outputs = model.forecast_with_covariates(
        inputs=[y.tolist()],
        dynamic_numerical_covariates={k: [v] for k, v in covariates.items()},
    )

    predictions = point_outputs[0].flatten()

    # ── 7. 输出结果 ──
    result_df = pd.DataFrame({
        "时间": future_index,
        "预测流量": predictions,
    })

    print(f"\n{'═' * 40}")
    print(f"预测结果 (未来 {HORIZON} 小时)")
    print(f"{'═' * 40}")
    for _, row in result_df.iterrows():
        print(f"  {row['时间']}  {row['预测流量']:.2f}")
    print(f"{'═' * 40}")

    # 保存到文件
    output_path = os.path.join(os.path.dirname(csv_path), "prediction.csv")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {output_path}")

# -*- coding: utf-8 -*-
"""
水厂出厂水流量预测 —— 使用 Chronos-2 做滚动预测 + 参数搜索
(CONTEXT / HORIZON 网格搜索，按 MAPE 选最优)

与原 TimesFM 版本的关键差异：
1. Chronos-2 没有 ForecastConfig / compile，上下文 = context_df 长度，
   预测步数 = prediction_length。
2. 协变量走 DataFrame 接口 (predict_df)，而非 forecast_with_covariates。
3. Chronos-2 最大上下文上限由 model config 决定 (默认 8192)。
"""

import os
# ★ 镜像站设置必须放在最前面，确保 chronos / huggingface 工具链都走镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
import itertools
import time

# ★ 模型加载：首次会从镜像站自动下载，也可填本地权重文件夹
from chronos import Chronos2Pipeline

try:
    from chinese_calendar import is_holiday, is_workday as _cal_is_workday
    _HAS_CHINESE_CALENDAR = True
except ImportError:
    _HAS_CHINESE_CALENDAR = False
    print("⚠ chinese-calendar 未安装")


# ── 1. 读取原始数据 ──
df = pd.read_csv(
    r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv",
    encoding="utf-8-sig", parse_dates=["时间"], index_col="时间"
)

y = df["出厂水流量"].values.astype(np.float32)
y_series = pd.Series(y, index=df.index)
y = y_series.interpolate(method="linear").bfill().ffill().values.astype(np.float32)

# ── 2. 生成时间外生变量 ──
def add_calendar_features(df_index):
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
            holiday_map[d] = float((not _cal_is_workday(dt)) and d.dayofweek < 5)
            prev_d = d - pd.Timedelta(days=1)
            next_d = d + pd.Timedelta(days=1)
            try:
                eve_map[d] = float(not _cal_is_workday(prev_d.to_pydatetime().date())
                                   and prev_d.dayofweek < 5)
            except ValueError:
                eve_map[d] = 0.0
            try:
                next_map[d] = float(not _cal_is_workday(next_d.to_pydatetime().date())
                                    and next_d.dayofweek < 5)
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


covariates = add_calendar_features(df.index)
# 协变量列名列表（timestamp / target 除外）
COV_COLS = [c for c in covariates.keys()]

# 完整时间戳序列（训练+测试拼接），用于构造每个滚动窗口的 context/future 时间戳
full_index = df.index
n_total = len(y)

# ── 3. 划分训练/测试 ──
test_days = 30
split_point = df.index[-1] - pd.Timedelta(days=test_days)
split_idx = df.index.searchsorted(split_point)

y_train = y[:split_idx]
y_test = y[split_idx:]
print(f"训练集: {len(y_train)} 样本, 测试集: {len(y_test)} 样本")

# ── 4. 加载模型 (只加载一次) ──
model = Chronos2Pipeline.from_pretrained(
    r"D:\Junshan_Project\models\chronos-2",
    # device_map="cuda",   # 有 GPU 取消注释
)

# 读取模型支持的最大上下文长度，超长组合自动跳过
_cfg = getattr(getattr(model, "model", None), "config", None)
_chronos_cfg = getattr(_cfg, "chronos_config", None)
MAX_CONTEXT = getattr(_chronos_cfg, "context_length", 8192) if _chronos_cfg else 8192
print(f"模型最大上下文长度: {MAX_CONTEXT}")


# ── 5. 参数搜索 ──
CONTEXT_OPTIONS = [48, 72, 96, 120, 168, 240, 336]
HORIZON_OPTIONS = [1, 2, 3, 6, 12, 24, 48, 72]

results = []
total_combos = len(CONTEXT_OPTIONS) * len(HORIZON_OPTIONS)
print(f"\n开始参数搜索: {total_combos} 种组合")
print("=" * 60)


def build_dataframes(hist_end, ctx, hor):
    """构造 context_df (历史) + future_df (未来协变量)，返回时间戳对齐的预测值数组。"""
    # 历史窗口起点
    ctx_start = max(0, hist_end - ctx)
    ctx_idx = range(ctx_start, hist_end)          # 长度 <= ctx
    fut_idx = range(hist_end, min(hist_end + hor, n_total))  # 未来 hor 步

    # context: item_id + timestamp + target + 协变量
    ctx_rows = []
    for i in ctx_idx:
        row = {"item_id": "flow", "timestamp": full_index[i], "target": float(y[i])}
        row.update({c: float(covariates[c][i]) for c in COV_COLS})
        ctx_rows.append(row)
    context_df = pd.DataFrame(ctx_rows)

    # future: item_id + timestamp + 协变量 (target 不填，由模型预测)
    fut_rows = []
    for i in fut_idx:
        row = {"item_id": "flow", "timestamp": full_index[i]}
        row.update({c: float(covariates[c][i]) for c in COV_COLS})
        fut_rows.append(row)
    future_df = pd.DataFrame(fut_rows)

    return context_df, future_df, len(fut_idx)


for combo_idx, (ctx, hor) in enumerate(itertools.product(CONTEXT_OPTIONS, HORIZON_OPTIONS), 1):
    print(f"\n[{combo_idx}/{total_combos}] CONTEXT={ctx} ({ctx // 24}天), HORIZON={hor} ({hor}h)")
    print("-" * 40)

    if hor >= ctx:
        print("  跳过: horizon >= context")
        continue
    if ctx > MAX_CONTEXT:
        print(f"  跳过: context {ctx} > 模型上限 {MAX_CONTEXT}")
        continue

    n_train = len(y_train)
    all_pred = []
    start_time = time.time()

    # 滚动预测：按 horizon 步长推进，每次用最近 ctx 长度历史
    n_test = len(y_test)
    for start in range(0, n_test, hor):
        hist_end = n_train + start
        context_df, future_df, n_fut = build_dataframes(hist_end, ctx, hor)

        try:
            pred_df = model.predict_df(
                context_df,
                future_df=future_df,
                prediction_length=n_fut,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            )
        except Exception as e:
            print(f"  预测出错: {e}")
            break

        # 预测列：0.5 分位即均值；若版本返回列名不同，按需调整
        if "predictions" in pred_df.columns:
            preds = pred_df["predictions"].to_numpy().astype(np.float32)
        elif "target" in pred_df.columns:
            preds = pred_df["target"].to_numpy().astype(np.float32)
        else:
            # 兜底：取数值列
            num_cols = [c for c in pred_df.columns if c not in ("timestamp",)]
            preds = pred_df[num_cols[0]].to_numpy().astype(np.float32)

        all_pred.append(preds)

    if not all_pred:
        print("  无有效预测")
        continue

    y_pred = np.concatenate(all_pred)[:n_test]
    y_true = y_test[:len(y_pred)]
    elapsed = time.time() - start_time

    # Overall MAPE
    overall_mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # 按天 (24h) 统计 MAPE
    daily_mapes = []
    for i in range(0, len(y_true), 24):
        chunk_true = y_true[i:i + 24]
        chunk_pred = y_pred[i:i + 24]
        if len(chunk_true) == 24:
            mape_d = np.mean(np.abs((chunk_true - chunk_pred) / (chunk_true + 1e-8))) * 100
            daily_mapes.append(mape_d)
    avg_daily_mape = np.mean(daily_mapes) if daily_mapes else 0

    results.append({
        "context": ctx,
        "horizon": hor,
        "overall_mape": overall_mape,
        "avg_daily_mape": avg_daily_mape,
        "time": elapsed,
        "n_predictions": len(y_pred),
    })
    print(f"  Overall MAPE: {overall_mape:.2f}%")
    print(f"  Avg Daily MAPE: {avg_daily_mape:.2f}%")
    print(f"  耗时: {elapsed:.1f}s")


# ── 6. 汇总结果 ──
print("\n" + "=" * 80)
print("参数搜索结果汇总")
print("=" * 80)

if not results:
    print("没有任何有效结果，请检查模型加载与预测接口。")
else:
    df_results = pd.DataFrame(results).sort_values("overall_mape")
    print(f"\n{'Context':>8} {'Horizon':>8} {'MAPE%':>8} {'Avg Daily%':>12} {'Time(s)':>10}")
    print("-" * 50)
    for _, row in df_results.iterrows():
        print(f"{int(row['context']):>8} {int(row['horizon']):>8} "
              f"{row['overall_mape']:>8.2f} {row['avg_daily_mape']:>12.2f} {row['time']:>10.1f}")

    best = df_results.iloc[0]
    print(f"\n最优参数: CONTEXT={int(best['context'])}, HORIZON={int(best['horizon'])}")
    print(f"  Overall MAPE: {best['overall_mape']:.2f}%")
    print(f"  Avg Daily MAPE: {best['avg_daily_mape']:.2f}%")

    out_csv = r"D:\Junshan_Project\chronos2_grid_results.csv"
    df_results.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"已保存: {out_csv}")

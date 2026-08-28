import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
from darts import TimeSeries
from darts.models import TimesFM2p5Model
import numpy as np
from darts.utils.likelihood_models import QuantileRegression
# ── 1. 读取原始数据 ──
df = pd.read_csv(
    r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv",
    encoding="utf-8-sig", parse_dates=["时间"], index_col="时间"
)

# 目标序列（出厂水流量）
series = TimeSeries.from_series(
    df["出厂水流量"],
    fill_missing_dates=True,
    freq="1h"
)

# ── 2. 划分训练/测试 ──
test_days = 30
split_point = df.index[-1] - pd.Timedelta(days=test_days)
train, test = series.split_after(split_point)
print(f'train:{train},test:{test}')
# ── 3. 模型定义 ──
# TimesFM 2.5 是零样本基础模型, 不支持外生变量
# 参数说明见: darts.models.forecasting.timesfm2p5_model.py
model = TimesFM2p5Model(
    input_chunk_length=168,                          # 回看 7 天 (168 小时)
    output_chunk_length=24,                          # 每次输出 24 步 = 1 天
    hub_model_name="google/timesfm-2.5-200m-pytorch",
    local_dir=r"D:\Junshan_Project\models\timesfm-2.5-200m-pytorch",
    likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
)

# ── 4. 微调训练 (零样本基线可跳过, 直接用预训练权重预测) ──
model.fit(
    series=train,
    epochs=200,
    verbose=True,
)

# ── 5. 逐天预测 (每天用真实历史数据, 不用预测值) ──
all_pred = []
# 完整真实序列 (训练+测试), 用于每天取真实历史
full_series = train.concatenate(test)
n_train = len(train)

for day_idx in range(len(test) // 24):
    # 截取到当天之前的真实历史 (位置索引: 训练集 + 测试集中已过的真实数据)
    hist_len = n_train + day_idx * 24
    history = full_series[:hist_len]
    # 预测当天 24 小时
    pred_day = model.predict(n=24, series=history)
    all_pred.append(pred_day.values().flatten())

y_pred = np.concatenate(all_pred)
y_true = test.values().flatten()[:len(y_pred)]

# ── 6. 评估 MAPE ──
# 整体 MAPE
overall_mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
print(f"Overall MAPE: {overall_mape:.2f}%")

# 按天统计 MAPE
test_index = test.time_index[:len(y_true)]
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

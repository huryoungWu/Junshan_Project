"""滚动窗口评估脚本 — 自回归模型逐日滚动预测 + MAE/RMSE/MAPE 评估。

滚动逻辑:
  用第 1~7 天数据预测第 8 天, 用第 2~8 天预测第 9 天, 以此类推。
  每次滑动 1 天, 预测下一天的 24 小时流量, 与真实值对比计算指标。

用法:
  python eval_rolling_ar.py
  python eval_rolling_ar.py --lookback 7 --result_dir results/xxx
"""

import os
import sys
import pickle
import argparse
import math

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor, needed_csv_columns

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv"
DEFAULT_RESULT_DIR = os.path.join(HERE, "results",
                                  "junshan_L1D_P24H_1h_itransformer_autoregressive_test")


def load_model_and_scaler(result_dir, device):
    """加载训练产物: 模型 + scaler + config。"""
    scaler_path = os.path.join(result_dir, "scaler.pkl")
    model_path = os.path.join(result_dir, "best_seq2seq_model.pth")
    for p in (scaler_path, model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到: {p}")

    with open(scaler_path, "rb") as f:
        saved = pickle.load(f)

    config = saved["config"]
    feature_scaler = saved["feature_scaler"]
    target_scaler = saved["target_scaler"]
    feature_cols = saved["feature_cols"]
    target_cols = saved["target_cols"]
    target_feat_idx = saved.get("target_feat_idx",
                                feature_cols.index(target_cols[0]))

    model_type = config.get("model_type", "transformer")
    lookback_steps = int(config["lookback_days"] * 24)
    model_kwargs = dict(
        input_dim=len(feature_cols),
        output_dim=1,
        horizon=1,
        input_len=lookback_steps,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["transformer_dropout"],
    )
    if model_type == "itransformer":
        model = iTransformer(**model_kwargs, target_idx=target_feat_idx).to(device)
    else:
        model = TimeSeriesTransformer(**model_kwargs).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, config, saved


def autoregressive_predict(model, window_scaled, future_feat_scaled,
                           predict_steps, target_feat_idx, device):
    """单次自回归 rollout, 返回 (predict_steps,) 的预测值 (scaled 域)。"""
    window = window_scaled.clone()
    preds = []
    with torch.no_grad():
        for k in range(predict_steps):
            one = model(window, target_len=1)          # (1, 1, 1)
            val = float(one[0, 0, 0].cpu())
            preds.append(val)

            next_row = future_feat_scaled[k].copy()
            next_row[target_feat_idx] = val
            next_row_t = torch.from_numpy(
                next_row.astype(np.float32)).view(1, 1, -1).to(device)
            window = torch.cat([window[:, 1:, :], next_row_t], dim=1)

    return np.array(preds, dtype=np.float32)


def compute_metrics(y_true, y_pred, floor_ratio=0.1):
    """计算 MAE, RMSE, MAPE (过滤近零流量点)。"""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    n = len(y_true)
    if n == 0:
        return {"mae": 0, "rmse": 0, "mape": 0, "n_total": 0, "n_used": 0}

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))

    thr = floor_ratio * np.abs(y_true).max()
    mask = np.abs(y_true) >= thr
    n_used = int(mask.sum())
    if n_used > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) /
                              (y_true[mask] + 1e-8))) * 100
    else:
        mape = 0.0

    return {"mae": mae, "rmse": rmse, "mape": mape,
            "n_total": n, "n_used": n_used, "threshold": thr}


def main():
    parser = argparse.ArgumentParser(
        description="滚动窗口评估: 逐日滚动预测 + MAE/RMSE/MAPE")
    parser.add_argument("--data", default=DEFAULT_DATA, help="原始数据 CSV")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="训练结果目录")
    parser.add_argument("--lookback", type=int, default=7,
                        help="回看天数 (默认 7)")
    parser.add_argument("--out_dir", default=None,
                        help="输出目录 (默认与 result_dir 同级的 eval_rolling)")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV 编码")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 加载模型 ──
    model, config, saved = load_model_and_scaler(args.result_dir, device)
    feature_scaler = saved["feature_scaler"]
    target_scaler = saved["target_scaler"]
    feature_cols = saved["feature_cols"]
    target_cols = saved["target_cols"]
    target_feat_idx = saved.get("target_feat_idx",
                                feature_cols.index(target_cols[0]))

    resample_freq = config["resample_freq"]
    freq_minutes = int(resample_freq.replace("min", ""))
    points_per_day = (24 * 60) // freq_minutes
    predict_steps = int(config["predict_days"] * points_per_day)  # 24
    lookback = args.lookback
    lookback_steps = lookback * points_per_day  # 168

    print(f"模型: {config.get('model_type', 'transformer')}, "
          f"训练 lookback={config['lookback_days']}d, "
          f"评估 lookback={lookback}d, predict={config['predict_days']}d ({predict_steps}步)")
    print(f"特征: {feature_cols}")

    # ── 加载原始数据 ──
    print(f"\n加载数据: {args.data}")
    processor = DataProcessor(config)
    df_raw = pd.read_csv(args.data, encoding=args.encoding)
    # 时间列处理
    for ts_col in ("时间", "timestamp"):
        if ts_col in df_raw.columns:
            df_raw[ts_col] = pd.to_datetime(df_raw[ts_col])
            df_raw = df_raw.set_index(ts_col)
            break
    df_raw = df_raw.sort_index()
    print(f"原始数据: {df_raw.index.min()} ~ {df_raw.index.max()}, {len(df_raw)} 行")

    # ── 清洗 + 特征 ──
    df_base = processor.build_base_features(df_raw)
    df_clean = processor.clean_and_resample(df_base)
    df_feat = processor.add_calendar_features(df_clean)
    df_feat = df_feat[feature_cols]
    print(f"清洗后: {len(df_feat)} 行, {df_feat.index.min()} ~ {df_feat.index.max()}")

    # ── 缩放 ──
    data_scaled = feature_scaler.transform(df_feat.values.astype(np.float32))

    # ── 滚动评估 ──
    total_len = len(data_scaled)
    min_start = lookback_steps
    max_start = total_len - predict_steps  # 留出 predict_steps 天做真实值

    if max_start <= min_start:
        print(f"数据不足: 清洗后 {total_len} 行, "
              f"需要至少 {lookback_steps + predict_steps} 行")
        return

    n_windows = max_start - min_start
    print(f"\n滚动评估: {n_windows} 个窗口, "
          f"lookback={lookback}d ({lookback_steps}步), "
          f"predict={predict_steps}步")
    print(f"{'='*70}")

    all_preds = []     # (n_windows, predict_steps)
    all_trues = []     # (n_windows, predict_steps)
    window_dates = []  # 每个窗口预测的起始日期

    pbar = tqdm(range(min_start, max_start),
                desc="滚动预测", unit="win",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for start in pbar:
        # 回看窗口
        window = data_scaled[start - lookback_steps:start]
        window_t = torch.from_numpy(window).unsqueeze(0).to(device)

        # 未来特征行 (日历真值 + 目标通道占位)
        future_idx = df_feat.index[start:start + predict_steps]
        if len(future_idx) < predict_steps:
            break
        future_dates = future_idx

        # 更新进度条显示当前预测日期
        current_date = df_feat.index[start].strftime("%Y-%m-%d %H:%M")
        pbar.set_postfix_str(f"当前: {current_date}")

        # 未来特征: 从原始数据取 (已缩放), 目标通道会被预测覆盖
        future_raw = df_feat.iloc[start:start + predict_steps][feature_cols].values.copy()
        future_scaled = feature_scaler.transform(future_raw.astype(np.float32))

        # 自回归预测
        pred_scaled = autoregressive_predict(
            model, window_t, future_scaled, predict_steps,
            target_feat_idx, device)

        # 反归一化
        pred_inv = target_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)).flatten()

        # 真实值
        true_inv = target_scaler.inverse_transform(
            data_scaled[start:start + predict_steps, 0].reshape(-1, 1)).flatten()

        all_preds.append(pred_inv)
        all_trues.append(true_inv)
        window_dates.append(df_feat.index[start])
    pbar.close()

    all_preds = np.array(all_preds)  # (n_windows, predict_steps)
    all_trues = np.array(all_trues)

    # ── 逐窗口指标 ──
    rows = []
    for i in range(len(all_preds)):
        m = compute_metrics(all_trues[i], all_preds[i], floor_ratio=0.1)
        rows.append({
            "window_idx": i,
            "predict_date": window_dates[i].strftime("%Y-%m-%d %H:%M"),
            "mae": m["mae"],
            "rmse": m["rmse"],
            "mape": m["mape"],
            "n_used": m["n_used"],
        })

    df_window = pd.DataFrame(rows)

    # ── 整体指标 ──
    overall = compute_metrics(all_trues.flatten(), all_preds.flatten(), floor_ratio=0.1)

    # ── 按预测步长统计 (第 1 小时 / 第 2 小时 / ... / 第 24 小时) ──
    step_rows = []
    for s in range(predict_steps):
        m = compute_metrics(all_trues[:, s], all_preds[:, s], floor_ratio=0.1)
        step_rows.append({
            "step": s + 1,
            "hour": f"{s:02d}:00",
            "mae": m["mae"],
            "rmse": m["rmse"],
            "mape": m["mape"],
        })
    df_step = pd.DataFrame(step_rows)

    # ── 输出 ──
    out_dir = args.out_dir or os.path.join(HERE, "eval_rolling")
    os.makedirs(out_dir, exist_ok=True)

    df_window.to_csv(os.path.join(out_dir, "window_metrics.csv"),
                     index=False, float_format="%.4f")
    df_step.to_csv(os.path.join(out_dir, "step_metrics.csv"),
                   index=False, float_format="%.4f")

    # 保存所有预测值
    pred_records = []
    for i in range(len(all_preds)):
        for s in range(predict_steps):
            pred_records.append({
                "window_idx": i,
                "predict_date": window_dates[i].strftime("%Y-%m-%d %H:%M"),
                "step": s + 1,
                "timestamp": (window_dates[i] + pd.Timedelta(hours=s + 1)).strftime("%Y-%m-%d %H:%M"),
                "true": float(all_trues[i, s]),
                "pred": float(all_preds[i, s]),
            })
    df_detail = pd.DataFrame(pred_records)
    df_detail.to_csv(os.path.join(out_dir, "predictions.csv"),
                     index=False, float_format="%.4f")

    # ── 每个时间点只保留一个预测值 (取 step 最小 = 最近窗口的预测) ──
    df_detail["timestamp"] = pd.to_datetime(df_detail["timestamp"])
    df_detail = df_detail.sort_values(["timestamp", "step"])
    df_unique = df_detail.drop_duplicates(subset="timestamp", keep="first").copy()
    df_unique["residual"] = df_unique["true"] - df_unique["pred"]
    df_unique = df_unique.sort_values("timestamp").reset_index(drop=True)
    df_unique.to_csv(os.path.join(out_dir, "predictions_unique.csv"),
                     index=False, float_format="%.4f")
    print(f"\n去重后预测点数: {len(df_unique)} (原始 {len(df_detail)} 条)")

    # ── 残差随时间的分布图 ──
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ts = df_unique["timestamp"]
    residual = df_unique["residual"]

    # 子图1: 真实值 vs 预测值
    ax1 = axes[0]
    ax1.plot(ts, df_unique["true"], label="True", linewidth=0.8, alpha=0.8)
    ax1.plot(ts, df_unique["pred"], label="Pred", linewidth=0.8, alpha=0.8)
    ax1.set_ylabel("Flow (m³/h)")
    ax1.set_title("True vs Predicted (unique per timestamp)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 残差时序
    ax2 = axes[1]
    ax2.plot(ts, residual, linewidth=0.6, alpha=0.7, color="steelblue")
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(y=residual.mean(), color="orange", linestyle="--",
                linewidth=0.8, alpha=0.7, label=f"mean={residual.mean():.2f}")
    ax2.set_ylabel("Residual (m³/h)")
    ax2.set_title("Residual over Time (true - pred)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 残差分布直方图
    ax3 = axes[2]
    ax3.hist(residual, bins=80, edgecolor="white", alpha=0.7, color="steelblue")
    ax3.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
    ax3.axvline(x=residual.mean(), color="orange", linestyle="--",
                linewidth=0.8, label=f"mean={residual.mean():.2f}")
    ax3.set_xlabel("Residual (m³/h)")
    ax3.set_ylabel("Count")
    ax3.set_title("Residual Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # x 轴日期格式
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.autofmt_xdate()
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "residual_over_time.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"残差分布图已保存: {plot_path}")

    # ── 打印结果 ──
    print(f"\n{'='*70}")
    print(f" 整体评估结果")
    print(f"{'='*70}")
    print(f"  窗口数:   {len(all_preds)}")
    print(f"  预测点数: {overall['n_total']} (MAPE 使用 {overall['n_used']} 点)")
    print(f"  MAE:      {overall['mae']:.2f} m³/h")
    print(f"  RMSE:     {overall['rmse']:.2f} m³/h")
    print(f"  MAPE:     {overall['mape']:.2f}%")

    print(f"\n{'='*70}")
    print(f" 按预测步长统计 (每小时)")
    print(f"{'='*70}")
    print(f"  {'步长':<8}{'MAE':<12}{'RMSE':<12}{'MAPE%':<10}")
    for _, r in df_step.iterrows():
        print(f"  {int(r['step']):<8}{r['mae']:<12.2f}{r['rmse']:<12.2f}{r['mape']:<10.2f}")

    print(f"\n结果已保存到: {out_dir}")
    print(f"  window_metrics.csv      — 逐窗口指标")
    print(f"  step_metrics.csv        — 按预测步长指标")
    print(f"  predictions.csv         — 所有预测值与真实值")
    print(f"  predictions_unique.csv  — 每时间点一个预测值 (最近窗口)")
    print(f"  residual_over_time.png  — 残差随时间分布图")


if __name__ == "__main__":
    main()

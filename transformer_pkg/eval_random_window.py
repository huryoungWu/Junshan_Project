"""随机窗口评估: 从原始数据中随机取连续 8 天, 前 7 天预测第 8 天, 计算 MAE/RMSE/MAPE。

用法:
  python eval_random_window.py
  python eval_random_window.py --lookback 7 --seed 42 --result_dir results/xxx
"""

import os
import sys
import pickle

# GBK 控制台无法编码 m³/h 等字符 → 统一 UTF-8 输出
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
import math
import random

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv"
DEFAULT_RESULT_DIR = os.path.join(HERE, "results",
                                  "junshan_L1D_P24H_1h_itransformer_autoregressive_test")


def main():
    parser = argparse.ArgumentParser(
        description="随机窗口评估: 前 N 天预测最后 1 天")
    parser.add_argument("--data", default=DEFAULT_DATA, help="原始数据 CSV")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="训练结果目录")
    parser.add_argument("--lookback", type=int, default=7, help="回看天数 (默认 7)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (None=随机)")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV 编码")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 加载模型 ──
    scaler_path = os.path.join(args.result_dir, "scaler.pkl")
    model_path = os.path.join(args.result_dir, "best_seq2seq_model.pth")
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
    model_lookback_steps = int(config["lookback_days"] * 24)
    model_kwargs = dict(
        input_dim=len(feature_cols),
        output_dim=1,
        horizon=1,
        input_len=model_lookback_steps,
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

    freq_minutes = int(config["resample_freq"].replace("min", ""))
    points_per_day = (24 * 60) // freq_minutes
    predict_steps = int(config["predict_days"] * points_per_day)
    lookback = args.lookback

    print(f"模型: {model_type}, 训练 lookback={config['lookback_days']}d, "
          f"评估 lookback={lookback}d, predict={predict_steps}步")

    # ── 加载数据 ──
    processor = DataProcessor(config)
    df_raw = pd.read_csv(args.data, encoding=args.encoding)
    for ts_col in ("时间", "timestamp"):
        if ts_col in df_raw.columns:
            df_raw[ts_col] = pd.to_datetime(df_raw[ts_col])
            df_raw = df_raw.set_index(ts_col)
            break
    df_raw = df_raw.sort_index()

    df_base = processor.build_base_features(df_raw)
    df_clean = processor.clean_and_resample(df_base)
    df_feat = processor.add_calendar_features(df_clean)
    df_feat = df_feat[feature_cols]

    data_scaled = feature_scaler.transform(df_feat.values.astype(np.float32))
    total_len = len(data_scaled)
    window_size = (lookback + 1) * points_per_day  # lookback天 + 1天预测

    print(f"清洗后: {total_len} 行 ({df_feat.index.min()} ~ {df_feat.index.max()})")

    # ── 随机选一个起始点 ──
    max_start = total_len - window_size
    if max_start < 0:
        print(f"数据不足: 需要 {window_size} 行, 只有 {total_len} 行")
        return

    start = random.randint(0, max_start)
    lookback_steps = lookback * points_per_day

    lookback_start_idx = start
    lookback_end_idx = start + lookback_steps
    pred_start_idx = lookback_end_idx
    pred_end_idx = pred_start_idx + predict_steps

    lookback_start_time = df_feat.index[lookback_start_idx]
    lookback_end_time = df_feat.index[lookback_end_idx - 1]
    pred_start_time = df_feat.index[pred_start_idx]
    pred_end_time = df_feat.index[pred_end_idx - 1]

    print(f"\n{'='*70}")
    print(f" 随机窗口 (seed={args.seed})")
    print(f"{'='*70}")
    print(f"  回看: {lookback_start_time} ~ {lookback_end_time} ({lookback}天, {lookback_steps}步)")
    print(f"  预测: {pred_start_time} ~ {pred_end_time} ({predict_steps}步)")

    # ── 构造输入 ──
    window_scaled = data_scaled[lookback_start_idx:lookback_end_idx]
    window_t = torch.from_numpy(window_scaled).unsqueeze(0).to(device)

    future_raw = df_feat.iloc[pred_start_idx:pred_end_idx][feature_cols].values.copy()
    future_scaled = feature_scaler.transform(future_raw.astype(np.float32))

    # ── 自回归预测 ──
    preds_scaled = []
    with torch.no_grad():
        for k in range(predict_steps):
            one = model(window_t, target_len=1)
            val = float(one[0, 0, 0].cpu())
            preds_scaled.append(val)

            next_row = future_scaled[k].copy()
            next_row[target_feat_idx] = val
            next_row_t = torch.from_numpy(
                next_row.astype(np.float32)).view(1, 1, -1).to(device)
            window_t = torch.cat([window_t[:, 1:, :], next_row_t], dim=1)

    # ── 反归一化 ──
    pred_inv = target_scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)).flatten()
    true_inv = target_scaler.inverse_transform(
        data_scaled[pred_start_idx:pred_end_idx, 0].reshape(-1, 1)).flatten()

    # ── 计算指标 ──
    abs_err = np.abs(true_inv - pred_inv)
    mae = np.mean(abs_err)
    rmse = math.sqrt(np.mean((true_inv - pred_inv) ** 2))
    thr = 0.1 * np.abs(true_inv).max()
    mask = np.abs(true_inv) >= thr
    mape = (np.mean(np.abs((true_inv[mask] - pred_inv[mask]) /
                           (true_inv[mask] + 1e-8))) * 100
            if mask.sum() > 0 else 0.0)

    # ── 输出逐小时结果 ──
    timestamps = df_feat.index[pred_start_idx:pred_end_idx]
    print(f"\n{'='*70}")
    print(f" 逐小时预测结果")
    print(f"{'='*70}")
    print(f"  {'时间':<22}{'真实值':<12}{'预测值':<12}{'绝对误差':<12}{'相对误差':<10}")
    print(f"  {'-'*68}")
    for i in range(predict_steps):
        ts = timestamps[i].strftime("%Y-%m-%d %H:%M")
        rel = abs_err[i] / (abs(true_inv[i]) + 1e-8) * 100
        print(f"  {ts:<22}{true_inv[i]:<12.2f}{pred_inv[i]:<12.2f}"
              f"{abs_err[i]:<12.2f}{rel:<10.2f}%")

    print(f"\n{'='*70}")
    print(f" 整体指标")
    print(f"{'='*70}")
    print(f"  MAE:  {mae:.2f} m³/h")
    print(f"  RMSE: {rmse:.2f} m³/h")
    print(f"  MAPE: {mape:.2f}% (过滤 |true| < {thr:.0f} 的点, {mask.sum()}/{len(mask)} 使用)")

    # ── 保存 ──
    ape = np.where(true_inv != 0, abs_err / np.abs(true_inv) * 100, 0.0)
    out_df = pd.DataFrame({
        "timestamp": timestamps,
        "true": true_inv,
        "pred": pred_inv,
        "abs_error": abs_err,
        "ape": ape,
    })
    out_path = os.path.join(HERE, "eval_random_result.csv")
    out_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n结果已保存: {out_path}")

    # ── 画图: 真实流量 vs 预测流量 ──
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 5))
    hours = [t.strftime("%H:%M") for t in timestamps]
    ax.plot(hours, true_inv, "o-", color="#2c3e50", linewidth=1.5, markersize=4, label="True")
    ax.plot(hours, pred_inv, "s--", color="#e74c3c", linewidth=1.5, markersize=4, label="Pred")
    ax.fill_between(hours, true_inv, pred_inv, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Flow (m3/h)")
    ax.set_title(f"True vs Pred  |  MAE={mae:.0f}  RMSE={rmse:.0f}  MAPE={mape:.1f}%  |  "
                 f"{lookback_start_time.strftime('%m/%d')}~{lookback_end_time.strftime('%m/%d')} -> "
                 f"{pred_start_time.strftime('%m/%d')}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(HERE, "eval_random_result.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已保存: {plot_path}")


if __name__ == "__main__":
    main()

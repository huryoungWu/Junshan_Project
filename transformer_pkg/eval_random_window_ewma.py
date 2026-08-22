"""EWMA 残差修正评估: 选择起始日期, 逐日预测并用指数加权移动平均 (EWMA) 修正。

用法:
  python eval_random_window_ewma.py --start_date 2025-10-01
  python eval_random_window_ewma.py --start_date 2025-01-08 --all_days
  python eval_random_window_ewma.py --start_date 2025-01-08 --all_days --ewma_alpha 0.15
  python eval_random_window_ewma.py --start_date 2025-01-08 --all_days --ewma_alpha 0.0
  python eval_random_window_ewma.py --seed 42
"""

import os
import sys
import pickle

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
import math
import random
import warnings
warnings.filterwarnings("ignore")

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


def correct_day_ewma(pred_day, true_day, ewma_state):
    """EWMA 残差修正: r̂_t = α·r_{t-1} + (1-α)·r̂_{t-1}

    Args:
        pred_day:    当天原始预测 (predict_steps,)
        true_day:    当天真实值 (predict_steps,)
        ewma_state:  dict, keys: ewma_val (float), alpha (float)

    Returns:
        corrected:   修正后预测 (predict_steps,)
    """
    ewma_val = ewma_state["ewma_val"]
    alpha = ewma_state["alpha"]
    corrected = np.zeros_like(pred_day)
    for h in range(len(pred_day)):
        corrected[h] = pred_day[h] + ewma_val
        r = true_day[h] - pred_day[h]
        ewma_val = alpha * r + (1 - alpha) * ewma_val
    ewma_state["ewma_val"] = ewma_val
    return corrected


def autoregressive_predict(model, window_scaled, future_feat_scaled,
                           predict_steps, target_feat_idx, device):
    """单次自回归 rollout, 返回 (predict_steps,) 的预测值 (scaled 域)。"""
    window = window_scaled.clone()
    preds = []
    with torch.no_grad():
        for k in range(predict_steps):
            one = model(window, target_len=1)
            val = float(one[0, 0, 0].cpu())
            preds.append(val)
            next_row = future_feat_scaled[k].copy()
            next_row[target_feat_idx] = val
            next_row_t = torch.from_numpy(
                next_row.astype(np.float32)).view(1, 1, -1).to(device)
            window = torch.cat([window[:, 1:, :], next_row_t], dim=1)
    return np.array(preds, dtype=np.float32)


def calc_metrics(y_true, y_pred):
    """计算 MAE, RMSE, MAPE。"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    thr = 0.1 * np.abs(y_true).max()
    mask = np.abs(y_true) >= thr
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) /
                           (y_true[mask] + 1e-8))) * 100
            if mask.sum() > 0 else 0.0)
    return mae, rmse, mape


def main():
    parser = argparse.ArgumentParser(
        description="EWMA 残差修正评估: 逐日预测 + 指数加权移动平均修正")
    parser.add_argument("--data", default=DEFAULT_DATA, help="原始数据 CSV")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="训练结果目录")
    parser.add_argument("--lookback", type=int, default=7, help="回看天数 (默认 7)")
    parser.add_argument("--start_date", default=None,
                        help="评估起始日期 (格式 YYYY-MM-DD, 第一个预测日)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (仅未指定 --start_date 时生效)")
    parser.add_argument("--ewma_alpha", type=float, default=0.2,
                        help="EWMA 衰减因子 (默认 0.2)")
    parser.add_argument("--all_days", action="store_true",
                        help="从 start_date 到数据末尾逐日预测")
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
    lookback_steps = lookback * points_per_day

    print(f"模型: {model_type}, lookback={lookback}d, predict={predict_steps}步")
    print(f"EWMA: alpha={args.ewma_alpha}")

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

    print(f"清洗后: {total_len} 行 ({df_feat.index.min()} ~ {df_feat.index.max()})")

    # ── 确定起始索引 ──
    if args.start_date is not None:
        target_date = pd.Timestamp(args.start_date)
        mask = df_feat.index >= target_date
        if not mask.any():
            print(f"错误: 数据中没有 >= {args.start_date} 的日期")
            return
        first_pred_idx = int(np.where(mask)[0][0])
        start = first_pred_idx - lookback_steps
        if start < 0:
            print(f"错误: {args.start_date} 前没有足够的 {lookback} 天回看数据")
            return
        print(f"起始日期: {args.start_date}, 回看从 {df_feat.index[start].date()}")
    else:
        max_start = total_len - (lookback + 1) * points_per_day
        if max_start < 0:
            print(f"数据不足")
            return
        start = random.randint(0, max_start)
        first_pred_idx = start + lookback_steps
        print(f"随机起始索引: {start} (seed={args.seed})")

    # ── 计算可预测天数 ──
    max_pred_days = (total_len - first_pred_idx) // points_per_day
    if args.all_days:
        n_predict_days = max_pred_days
    else:
        n_predict_days = min(lookback, max_pred_days)
    if n_predict_days < 1:
        print(f"错误: 无可预测天数")
        return

    print(f"预测天数: {n_predict_days} "
          f"({df_feat.index[first_pred_idx].date()} ~ "
          f"{df_feat.index[first_pred_idx + (n_predict_days-1)*points_per_day].date()})")

    # ── 逐日预测 ──
    print(f"\n{'='*70}")
    print(f" 逐日预测 ({n_predict_days} 天)")
    print(f"{'='*70}")

    all_day_preds = []
    all_day_trues = []
    all_day_timestamps = []

    for day in range(n_predict_days):
        pred_start = first_pred_idx + day * points_per_day
        pred_end = pred_start + predict_steps
        if pred_end > total_len:
            break

        lb_start = pred_start - lookback_steps
        lb_end = pred_start

        if day % 20 == 0 or day == n_predict_days - 1:
            print(f"  Day {day+1:>4}/{n_predict_days}: "
                  f"{df_feat.index[pred_start].strftime('%Y-%m-%d')}")

        window_scaled = data_scaled[lb_start:lb_end]
        window_t = torch.from_numpy(window_scaled).unsqueeze(0).to(device)
        future_raw = df_feat.iloc[pred_start:pred_end][feature_cols].values.copy()
        future_scaled = feature_scaler.transform(future_raw.astype(np.float32))

        pred_scaled = autoregressive_predict(
            model, window_t, future_scaled, predict_steps,
            target_feat_idx, device)

        pred_inv = target_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)).flatten()
        true_inv = target_scaler.inverse_transform(
            data_scaled[pred_start:pred_end, 0].reshape(-1, 1)).flatten()

        all_day_preds.append(pred_inv)
        all_day_trues.append(true_inv)
        all_day_timestamps.append(df_feat.index[pred_start:pred_end])

    n_actual = len(all_day_preds)
    print(f"实际预测: {n_actual} 天")

    # ── 逐日 EWMA 修正 ──
    ewma_state = {"ewma_val": 0.0, "alpha": args.ewma_alpha}
    all_day_corrected = []

    for day in range(n_actual):
        corr = correct_day_ewma(all_day_preds[day], all_day_trues[day], ewma_state)
        all_day_corrected.append(corr)

    # ── 每天指标对比 ──
    print(f"\n{'='*70}")
    print(f" 每天指标对比 (EWMA alpha={args.ewma_alpha})")
    print(f"{'='*70}")

    # 分三个子表打印, 避免列拥挤
    for metric_name, metric_key, fmt in [("MAE", "mae", ".1f"), ("RMSE", "rmse", ".1f"), ("MAPE%", "mape", ".2f")]:
        print(f"\n  {metric_name}:")
        print(f"  {'日期':<12}{'原始':<12}{'修正后':<12}{'提升':<12}")
        print(f"  {'-'*48}")
        for day in range(n_actual):
            date_str = all_day_timestamps[day][0].strftime("%Y-%m-%d")
            o_mae, o_rmse, o_mape = calc_metrics(all_day_trues[day], all_day_preds[day])
            c_mae, c_rmse, c_mape = calc_metrics(all_day_trues[day], all_day_corrected[day])
            orig = {"mae": o_mae, "rmse": o_rmse, "mape": o_mape}[metric_key]
            corr = {"mae": c_mae, "rmse": c_rmse, "mape": c_mape}[metric_key]
            imp = (1 - corr / orig) * 100 if orig > 0 else 0
            print(f"  {date_str:<12}{orig:<12{fmt}}{corr:<12{fmt}}{imp:>+10.1f}%")

    day_rows = []
    for day in range(n_actual):
        o_mae, o_rmse, o_mape = calc_metrics(all_day_trues[day], all_day_preds[day])
        c_mae, c_rmse, c_mape = calc_metrics(all_day_trues[day], all_day_corrected[day])
        mae_imp = (1 - c_mae / o_mae) * 100 if o_mae > 0 else 0
        rmse_imp = (1 - c_rmse / o_rmse) * 100 if o_rmse > 0 else 0
        mape_imp = (1 - c_mape / o_mape) * 100 if o_mape > 0 else 0
        day_rows.append({
            "date": all_day_timestamps[day][0].strftime("%Y-%m-%d"),
            "orig_mae": o_mae, "corr_mae": c_mae, "mae_imp": mae_imp,
            "orig_rmse": o_rmse, "corr_rmse": c_rmse, "rmse_imp": rmse_imp,
            "orig_mape": o_mape, "corr_mape": c_mape, "mape_imp": mape_imp,
        })

    # ── 汇总 ──
    all_true = np.concatenate(all_day_trues)
    all_orig = np.concatenate(all_day_preds)
    all_corr = np.concatenate(all_day_corrected)
    o_all = calc_metrics(all_true, all_orig)
    c_all = calc_metrics(all_true, all_corr)

    print(f"  {'-'*110}")
    print(f"  {'ALL':<12}{o_all[0]:<10.2f}{c_all[0]:<10.2f}{(1-c_all[0]/o_all[0])*100:>+10.2f}%"
          f"{o_all[1]:<10.2f}{c_all[1]:<10.2f}{(1-c_all[1]/o_all[1])*100:>+10.2f}%"
          f"{o_all[2]:<10.2f}{c_all[2]:<10.2f}{(1-c_all[2]/o_all[2])*100:>+10.2f}%")

    print(f"\n{'='*70}")
    print(f" 全量指标汇总 ({n_actual} 天)")
    print(f"{'='*70}")
    print(f"  {'方法':<15}{'MAE':<10}{'RMSE':<10}{'MAPE%':<10}"
          f"{'MAE提升':<12}{'RMSE提升':<12}{'MAPE提升':<12}")
    print(f"  {'-'*81}")
    print(f"  {'原始':<15}{o_all[0]:<10.2f}{o_all[1]:<10.2f}{o_all[2]:<10.2f}"
          f"{'---':<12}{'---':<12}{'---':<12}")
    print(f"  {'EWMA(α='+str(args.ewma_alpha)+')':<15}{c_all[0]:<10.2f}{c_all[1]:<10.2f}{c_all[2]:<10.2f}"
          f"{(1-c_all[0]/o_all[0])*100:>+10.2f}%{(1-c_all[1]/o_all[1])*100:>+10.2f}%{(1-c_all[2]/o_all[2])*100:>+10.2f}%")

    # ── 保存 CSV ──
    day_df = pd.DataFrame(day_rows)
    csv_path = os.path.join(HERE, "eval_ewma_day_metrics.csv")
    day_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n每天指标已保存: {csv_path}")

    # ── 保存逐小时预测 ──
    all_records = []
    for day in range(n_actual):
        for h in range(predict_steps):
            all_records.append({
                "date": all_day_timestamps[day][0].strftime("%Y-%m-%d"),
                "timestamp": str(all_day_timestamps[day][h]),
                "true": float(all_day_trues[day][h]),
                "pred": float(all_day_preds[day][h]),
                "pred_corrected": float(all_day_corrected[day][h]),
                "residual": float(all_day_trues[day][h] - all_day_preds[day][h]),
                "residual_corrected": float(all_day_trues[day][h] - all_day_corrected[day][h]),
            })
    all_df = pd.DataFrame(all_records)
    all_csv = os.path.join(HERE, "eval_ewma_all_predictions.csv")
    all_df.to_csv(all_csv, index=False, float_format="%.4f")
    print(f"全部预测已保存: {all_csv}")

    # ── 画图 ──
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    n_days_plot = min(n_actual, 120)
    plot_rows = day_rows[-n_days_plot:]
    dates = [r["date"] for r in plot_rows]
    x = np.arange(len(dates))

    # 三指标对比
    fig, axes = plt.subplots(3, 1, figsize=(max(18, len(dates) * 0.25), 14))
    for ax, metric, title in zip(axes, ["mae", "rmse", "mape"], ["MAE", "RMSE", "MAPE%"]):
        orig_vals = [r[f"orig_{metric}"] for r in plot_rows]
        corr_vals = [r[f"corr_{metric}"] for r in plot_rows]
        ax.plot(x, orig_vals, "-", color="#2c3e50", linewidth=1.2, alpha=0.8, label="原始")
        ax.plot(x, corr_vals, "-", color="#e74c3c", linewidth=1.0, alpha=0.8,
                label=f"EWMA(α={args.ewma_alpha})")
        ax.set_xticks(x[::max(1, len(x)//20)])
        ax.set_xticklabels([d[5:] for d in dates[::max(1, len(x)//20)]], rotation=45)
        ax.set_title(title, fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"EWMA 残差修正 (α={args.ewma_alpha}) — {dates[0]} ~ {dates[-1]}, {n_actual}天",
                 fontsize=14)
    fig.tight_layout()
    fig_path = os.path.join(HERE, "eval_ewma_comparison.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已保存: {fig_path}")

    # MAE 提升图
    fig2, ax2 = plt.subplots(figsize=(max(18, len(dates) * 0.25), 5))
    mae_imps = [r["mae_imp"] for r in plot_rows]
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in mae_imps]
    ax2.bar(x, mae_imps, color=colors, alpha=0.7, width=0.8)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xticks(x[::max(1, len(x)//20)])
    ax2.set_xticklabels([d[5:] for d in dates[::max(1, len(x)//20)]], rotation=45)
    ax2.set_ylabel("MAE 提升 (%)")
    ax2.set_title(f"每天 MAE 提升 (EWMA α={args.ewma_alpha})")
    ax2.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    imp_path = os.path.join(HERE, "eval_ewma_improvement.png")
    fig2.savefig(imp_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"提升图已保存: {imp_path}")


if __name__ == "__main__":
    main()

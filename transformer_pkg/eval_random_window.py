"""指定窗口评估: 选择起始日期, 逐日预测, 输出原始预测指标。

用法:
  # 单天模式
  python eval_random_window.py --start_date 2025-10-01

  # 全天模式
  python eval_random_window.py --start_date 2025-01-08 --all_days

  # 指定日期范围, 逐日画图保存
  python eval_random_window.py --start_date 2025-10-01 --end_date 2025-12-31 --plot
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
                                  "junshan_L1D_P24H_1h_transformer_autoregressive_20260824_234455")


# ==================== 工具函数 ====================

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


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(
        description="指定窗口评估: 逐日预测, 输出原始指标")
    parser.add_argument("--data", default=DEFAULT_DATA, help="原始数据 CSV")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="训练结果目录")
    parser.add_argument("--lookback", type=int, default=7, help="回看天数 (默认 7)")
    parser.add_argument("--start_date", default=None,
                        help="评估起始日期 (格式 YYYY-MM-DD, 第一个预测日)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (仅未指定 --start_date 时生效)")
    parser.add_argument("--all_days", action="store_true",
                        help="从 start_date 到数据末尾逐日预测")
    parser.add_argument("--end_date", default=None,
                        help="评估结束日期 (格式 YYYY-MM-DD, 配合 --start_date 使用)")
    parser.add_argument("--plot", action="store_true",
                        help="逐日画图并保存到本地")
    parser.add_argument("--plot_dir", default=None,
                        help="逐日图片保存目录 (默认: result_dir/daily_plots)")
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
    df_feat = processor.add_data_driven_features(df_feat)
    df_feat = df_feat.dropna(subset=feature_cols)
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
    if args.end_date is not None:
        # 指定结束日期: 计算 start_date 到 end_date 之间的天数
        end_date = pd.Timestamp(args.end_date)
        pred_start_date = df_feat.index[first_pred_idx]
        n_predict_days = (end_date - pred_start_date).days + 1
        n_predict_days = max(1, min(n_predict_days, max_pred_days))
    elif args.all_days:
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

    # 逐日画图: 如果指定了 --plot, 每天保存一张独立的图
    plot_dir = None
    if args.plot:
        plot_dir = args.plot_dir or os.path.join(args.result_dir, "daily_plots")
        os.makedirs(plot_dir, exist_ok=True)
        print(f"  逐日图片将保存到: {plot_dir}")

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

        # 逐日画图并保存
        if plot_dir is not None:
            ts = df_feat.index[pred_start:pred_end]
            d_mae, d_rmse, d_mape = calc_metrics(true_inv, pred_inv)

            plt.rcParams["font.sans-serif"] = ["SimHei"]
            plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(ts, true_inv, "b-o", markersize=2, linewidth=1.2, label="真实值")
            ax.plot(ts, pred_inv, "r-s", markersize=2, linewidth=1.2, label="预测值")
            ax.fill_between(ts, true_inv, pred_inv, alpha=0.15, color="gray")
            ax.set_title(f"{ts[0].strftime('%Y-%m-%d')}  "
                         f"MAE={d_mae:.2f}  RMSE={d_rmse:.2f}  MAPE={d_mape:.2f}%",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("时间", fontsize=10)
            ax.set_ylabel("流量", fontsize=10)
            ax.legend(fontsize=10, loc="upper right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            date_str = ts[0].strftime("%Y-%m-%d")
            fig.savefig(os.path.join(plot_dir, f"{date_str}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    n_actual = len(all_day_preds)
    print(f"实际预测: {n_actual} 天")
    if plot_dir is not None:
        print(f"  逐日图片已保存: {plot_dir} ({n_actual} 张)")

    # ── 原始指标 ──
    all_true = np.concatenate(all_day_trues)
    all_orig = np.concatenate(all_day_preds)
    orig_mae, orig_rmse, orig_mape = calc_metrics(all_true, all_orig)

    print(f"\n{'='*70}")
    print(f" 预测指标 ({n_actual} 天)")
    print(f"{'='*70}")
    print(f"  MAE={orig_mae:.2f}, RMSE={orig_rmse:.2f}, MAPE={orig_mape:.2f}%")

    # ── 逐日指标 ──
    print(f"\n{'='*70}")
    print(f" 逐日指标")
    print(f"{'='*70}")
    print(f"  {'日期':<14}{'MAE':<10}{'RMSE':<10}{'MAPE%':<10}")
    print(f"  {'-'*44}")
    for day in range(n_actual):
        d_mae, d_rmse, d_mape = calc_metrics(all_day_trues[day], all_day_preds[day])
        date_str = all_day_timestamps[day][0].strftime("%Y-%m-%d")
        print(f"  {date_str:<14}{d_mae:<10.2f}{d_rmse:<10.2f}{d_mape:<10.2f}")

    # ── 画预测对比图 ──
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(min(n_actual, 5), 1,
                             figsize=(14, 3 * min(n_actual, 5)),
                             sharex=False)
    if n_actual == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= n_actual:
            break
        ts = all_day_timestamps[i]
        ax.plot(ts, all_day_trues[i], "b-o", markersize=2, label="真实值")
        ax.plot(ts, all_day_preds[i], "r-s", markersize=2, label="预测值")
        ax.set_title(f"Day {i+1}: {ts[0].strftime('%Y-%m-%d')}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"逐日预测对比 ({n_actual}天, MAPE={orig_mape:.2f}%)", fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(HERE, "eval_random_window.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n预测对比图已保存: {fig_path}")


if __name__ == "__main__":
    main()

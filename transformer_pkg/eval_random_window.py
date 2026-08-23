"""指定窗口评估: 选择起始日期, 逐日预测, 调用外部 EWMA 修正模块对比修正前后指标。

用法:
  # 单天模式
  python eval_random_window.py --start_date 2025-10-01

  # 全天模式
  python eval_random_window.py --start_date 2025-01-08 --all_days

  # 指定 EWMA alpha
  python eval_random_window.py --start_date 2025-01-08 --all_days
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

# 导入外部 EWMA 修正模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ewma_correction import run_ewma_correction

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv"
DEFAULT_RESULT_DIR = os.path.join(HERE, "results",
                                  "junshan_L1D_P24H_1h_transformer_autoregressive_20260823_120030")


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
        description="指定窗口评估: 逐日预测 + 外部 EWMA 修正对比")
    parser.add_argument("--data", default=DEFAULT_DATA, help="原始数据 CSV")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="训练结果目录")
    parser.add_argument("--lookback", type=int, default=7, help="回看天数 (默认 7)")
    parser.add_argument("--start_date", default=None,
                        help="评估起始日期 (格式 YYYY-MM-DD, 第一个预测日)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (仅未指定 --start_date 时生效)")
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

    # ── 原始指标 ──
    all_true = np.concatenate(all_day_trues)
    all_orig = np.concatenate(all_day_preds)
    orig_mae, orig_rmse, orig_mape = calc_metrics(all_true, all_orig)

    print(f"\n{'='*70}")
    print(f" 原始预测指标 ({n_actual} 天)")
    print(f"{'='*70}")
    print(f"  MAE={orig_mae:.2f}, RMSE={orig_rmse:.2f}, MAPE={orig_mape:.2f}%")

    # ── 导出 CSV 供外部 EWMA 模块使用 ──
    ewma_work_dir = os.path.join(args.result_dir, "_ewma_work")
    os.makedirs(ewma_work_dir, exist_ok=True)
    pred_csv = os.path.join(ewma_work_dir, "predictions.csv")
    true_csv = os.path.join(ewma_work_dir, "true_values.csv")
    warmup_csv = os.path.join(ewma_work_dir, "warmup_predictions.csv")
    state_csv = os.path.join(ewma_work_dir, "ewma_state.csv")
    corrected_csv = os.path.join(ewma_work_dir, "corrected_predictions.csv")

    SITE_ID = "junshan"

    # 构建 predictions.csv: site_id, date, predicted_value (每小时一行)
    pred_rows = []
    true_rows = []
    for day in range(n_actual):
        for h in range(predict_steps):
            ts = all_day_timestamps[day][h]
            pred_rows.append({
                "site_id": SITE_ID,
                "date": ts,
                "predicted_value": float(all_day_preds[day][h]),
            })
            true_rows.append({
                "site_id": SITE_ID,
                "date": ts,
                "true_value": float(all_day_trues[day][h]),
            })

    # 追加预热期的真实值到 true_rows (供 EWMA 预热阶段匹配)
    warmup_start_idx = first_pred_idx - lookback * points_per_day
    if warmup_start_idx >= 0:
        for wi in range(warmup_start_idx, first_pred_idx):
            ts = df_feat.index[wi]
            true_val = float(target_scaler.inverse_transform(
                data_scaled[wi, 0].reshape(-1, 1)).flatten()[0])
            true_rows.append({
                "site_id": SITE_ID,
                "date": ts,
                "true_value": true_val,
            })

    df_pred_out = pd.DataFrame(pred_rows)
    df_true_out = pd.DataFrame(true_rows)
    df_pred_out.to_csv(pred_csv, index=False)
    df_true_out.to_csv(true_csv, index=False)
    print(f"\n预测数据已导出: {pred_csv} ({len(df_pred_out)} 条)")
    print(f"真实值已导出: {true_csv} ({len(df_true_out)} 条)")

    # ── 生成预热数据: 用模型对预测期之前的历史窗口做预测 ──
    warmup_days = lookback  # 用前 lookback 天作为预热
    warmup_start = first_pred_idx - warmup_days * points_per_day
    if warmup_start >= 0:
        print(f"\n生成预热数据: {warmup_days} 天 ({df_feat.index[warmup_start].date()} ~ "
              f"{df_feat.index[first_pred_idx - 1].date()})")
        warmup_rows = []
        for wday in range(warmup_days):
            w_pred_start = warmup_start + wday * points_per_day
            w_pred_end = w_pred_start + predict_steps
            if w_pred_end > first_pred_idx:  # 不超过预测期起点
                break
            w_lb_start = w_pred_start - lookback_steps
            if w_lb_start < 0:
                continue

            window_scaled = data_scaled[w_lb_start:w_pred_start]
            window_t = torch.from_numpy(window_scaled).unsqueeze(0).to(device)
            future_raw = df_feat.iloc[w_pred_start:w_pred_end][feature_cols].values.copy()
            future_scaled = feature_scaler.transform(future_raw.astype(np.float32))

            w_pred_scaled = autoregressive_predict(
                model, window_t, future_scaled, predict_steps,
                target_feat_idx, device)
            w_pred_inv = target_scaler.inverse_transform(
                w_pred_scaled.reshape(-1, 1)).flatten()

            for h in range(predict_steps):
                ts = df_feat.index[w_pred_start + h]
                warmup_rows.append({
                    "site_id": SITE_ID,
                    "date": ts,
                    "predicted_value": float(w_pred_inv[h]),
                })

        df_warmup = pd.DataFrame(warmup_rows)
        df_warmup.to_csv(warmup_csv, index=False)
        print(f"预热数据已导出: {warmup_csv} ({len(df_warmup)} 条)")
    else:
        warmup_csv = None
        print("  ⚠ 数据不足, 跳过预热")

    # ── Alpha 扫描: 0.0 ~ 1.0 步长 0.1 ──
    alpha_list = [round(i * 0.1, 1) for i in range(11)]  # [0.0, 0.1, ..., 1.0]
    sweep_results = []

    print(f"\n{'='*70}")
    print(f" Alpha 扫描 (0.0 ~ 1.0, 步长 0.1)")
    print(f"{'='*70}")

    for alpha in alpha_list:
        # 每个 alpha 用独立的状态/结果文件
        state_alpha = os.path.join(ewma_work_dir, f"ewma_state_a{alpha}.csv")
        corrected_alpha = os.path.join(ewma_work_dir, f"corrected_a{alpha}.csv")

        for f in (state_alpha, corrected_alpha):
            if os.path.exists(f):
                os.remove(f)

        run_ewma_correction(
            predictions_path=pred_csv,
            true_values_path=true_csv,
            state_path=state_alpha,
            output_path=corrected_alpha,
            alpha_default=alpha,
            target_date=None,
            warmup_path=warmup_csv,
        )

        if not os.path.exists(corrected_alpha):
            print(f"  alpha={alpha}: 修正失败, 跳过")
            continue

        df_corr = pd.read_csv(corrected_alpha)
        df_corr["date"] = pd.to_datetime(df_corr["date"])
        corr_map = dict(zip(df_corr["date"], df_corr["corrected_prediction"]))

        all_corr = np.array([corr_map.get(ts, orig)
                             for ts, orig in zip(
                                 pd.to_datetime(df_pred_out["date"]),
                                 all_orig)])
        c_mae, c_rmse, c_mape = calc_metrics(all_true, all_corr)
        mae_imp = (1 - c_mae / orig_mae) * 100 if orig_mae > 0 else 0
        rmse_imp = (1 - c_rmse / orig_rmse) * 100 if orig_rmse > 0 else 0
        mape_imp = (1 - c_mape / orig_mape) * 100 if orig_mape > 0 else 0

        sweep_results.append({
            "alpha": alpha, "mae": c_mae, "rmse": c_rmse, "mape": c_mape,
            "mae_imp": mae_imp, "rmse_imp": rmse_imp, "mape_imp": mape_imp,
        })
        print(f"  alpha={alpha:.1f}: MAE={c_mae:.2f} ({mae_imp:+.2f}%), "
              f"RMSE={c_rmse:.2f} ({rmse_imp:+.2f}%), MAPE={c_mape:.2f}% ({mape_imp:+.2f}%)")

    # ── 汇总表 ──
    print(f"\n{'='*70}")
    print(f" Alpha 扫描汇总 ({n_actual} 天, {len(all_true)} 个时间点)")
    print(f"{'='*70}")
    print(f"  {'alpha':<8}{'MAE':<12}{'MAE提升':<12}{'RMSE':<12}{'RMSE提升':<12}{'MAPE%':<12}{'MAPE提升':<12}")
    print(f"  {'-'*80}")
    print(f"  {'(无修正)':<8}{orig_mae:<12.2f}{'---':<12}{orig_rmse:<12.2f}{'---':<12}{orig_mape:<12.2f}{'---':<12}")
    for r in sweep_results:
        print(f"  {r['alpha']:<8.1f}{r['mae']:<12.2f}{r['mae_imp']:>+10.2f}%"
              f"{r['rmse']:<12.2f}{r['rmse_imp']:>+10.2f}%"
              f"{r['mape']:<12.2f}{r['mape_imp']:>+10.2f}%")

    # 找最优 alpha
    if sweep_results:
        best = max(sweep_results, key=lambda r: r["mae_imp"])
        print(f"\n  最优 alpha={best['alpha']:.1f}: "
              f"MAE={best['mae']:.2f} ({best['mae_imp']:+.2f}%), "
              f"RMSE={best['rmse']:.2f} ({best['rmse_imp']:+.2f}%), "
              f"MAPE={best['mape']:.2f}% ({best['mape_imp']:+.2f}%)")

    # ── 保存扫描结果 CSV ──
    sweep_df = pd.DataFrame(sweep_results)
    sweep_csv = os.path.join(HERE, "eval_alpha_sweep.csv")
    sweep_df.to_csv(sweep_csv, index=False, float_format="%.4f")
    print(f"\nAlpha 扫描结果已保存: {sweep_csv}")

    # ── 画 Alpha 扫描对比图 ──
    if sweep_results:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        alphas = [r["alpha"] for r in sweep_results]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, metric, label in zip(axes, ["mae", "rmse", "mape"], ["MAE", "RMSE", "MAPE%"]):
            vals = [r[metric] for r in sweep_results]
            orig_val = orig_mae if metric == "mae" else orig_rmse if metric == "rmse" else orig_mape
            ax.plot(alphas, vals, "o-", color="#2c3e50", linewidth=1.5, markersize=6, label="EWMA修正")
            ax.axhline(y=orig_val, color="#e74c3c", linestyle="--", linewidth=1, label=f"原始({orig_val:.2f})")
            ax.set_xlabel("Alpha")
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs Alpha")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xticks(alphas)

        fig.suptitle(f"EWMA Alpha 扫描 ({n_actual}天)", fontsize=14)
        fig.tight_layout()
        sweep_fig_path = os.path.join(HERE, "eval_alpha_sweep.png")
        fig.savefig(sweep_fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Alpha 扫描图已保存: {sweep_fig_path}")

    # 清理临时文件
    print(f"\n临时文件保留在: {ewma_work_dir}")


if __name__ == "__main__":
    main()

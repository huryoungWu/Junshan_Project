"""残差修正方法对比: STL / 小模型(在线学习) / 卡尔曼滤波 / 滑动窗口 / ARIMA

与 baseline (MAPE=6.85%) 对比, 评估数据与 eval_random_window.py 一致。

用法:
  python residual_correction_compare.py --start_date 2025-01-08 --all_days
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import math
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer_pkg"))
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv"
DEFAULT_RESULT_DIR = os.path.join(HERE, "transformer_pkg", "results",
                                  "junshan_L1D_P24H_1h_transformer_autoregressive_20260823_120030")


# ==================== 工具函数 ====================

def autoregressive_predict(model, window_scaled, future_feat_scaled,
                           predict_steps, target_feat_idx, device):
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
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    thr = 0.1 * np.abs(y_true).max()
    mask = np.abs(y_true) >= thr
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) /
                           (y_true[mask] + 1e-8))) * 100
            if mask.sum() > 0 else 0.0)
    return mae, rmse, mape


# ==================== 修正方法 ====================

class SlidingWindowCorrection:
    """滑动窗口均值修正: 用最近 N 个残差的均值修正。"""
    def __init__(self, window=168):
        self.window = window
        self.history = []

    def correct(self, pred_day):
        corrected = np.zeros_like(pred_day)
        for h in range(len(pred_day)):
            w = min(self.window, len(self.history))
            mean_res = np.mean(self.history[-w:]) if w > 0 else 0.0
            corrected[h] = pred_day[h] + mean_res
        return corrected

    def update(self, pred_day, true_day):
        residuals = true_day - pred_day
        self.history.extend(residuals.tolist())
        # 限制历史长度
        max_hist = self.window * 10
        if len(self.history) > max_hist:
            self.history = self.history[-max_hist:]


class EWMACorrection:
    """EWMA 修正: 指数加权移动平均。"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.ewma_val = 0.0

    def correct(self, pred_day):
        corrected = pred_day + self.ewma_val
        return corrected

    def update(self, pred_day, true_day):
        for h in range(len(true_day)):
            r = true_day[h] - pred_day[h]
            self.ewma_val = self.alpha * r + (1 - self.alpha) * self.ewma_val


class STLCorrection:
    """STL 季节性分解修正: 分解残差的趋势+季节性, 用趋势外推+季节模式修正。"""
    def __init__(self, period=24, min_history=48):
        self.period = period
        self.min_history = min_history
        self.history = []

    def correct(self, pred_day):
        from statsmodels.tsa.seasonal import STL

        n_hours = len(pred_day)
        if len(self.history) < self.min_history:
            # 历史不足, 用均值
            mean_r = np.mean(self.history) if self.history else 0.0
            return pred_day + mean_r

        try:
            res_series = pd.Series(self.history)
            stl = STL(res_series, period=self.period, robust=True)
            result = stl.fit()
            seasonal = result.seasonal.values
            trend = result.trend.values

            # 趋势外推
            if len(trend) >= 2:
                trend_slope = trend[-1] - trend[-2]
            else:
                trend_slope = 0
            trend_base = trend[-1]

            # 季节性: 取最后一个完整周期
            last_season = seasonal[-self.period:]
            corrected = np.zeros(n_hours)
            for h in range(n_hours):
                next_idx = (len(self.history) + h) % self.period
                corrected[h] = pred_day[h] + trend_base + trend_slope * (h + 1) + last_season[next_idx]
            return corrected
        except Exception:
            mean_r = np.mean(self.history[-48:]) if len(self.history) >= 48 else np.mean(self.history)
            return pred_day + mean_r

    def update(self, pred_day, true_day):
        residuals = true_day - pred_day
        self.history.extend(residuals.tolist())
        max_hist = self.period * 30  # 保留30个周期
        if len(self.history) > max_hist:
            self.history = self.history[-max_hist:]


class KalmanCorrection:
    """卡尔曼滤波修正: 把残差当作隐状态, 用观测值动态估计。

    状态模型: x_t = x_{t-1} + w_t  (随机游走)
    观测模型: z_t = x_t + v_t       (观测 = 真实残差)
    """
    def __init__(self, process_noise=1.0, measurement_noise=100.0):
        self.Q = process_noise       # 过程噪声方差
        self.R = measurement_noise   # 观测噪声方差
        self.x = 0.0                 # 状态估计
        self.P = 1000.0              # 估计误差协方差 (大初始值表示不确定)
        self.initialized = False

    def correct(self, pred_day):
        return pred_day + self.x

    def update(self, pred_day, true_day):
        for h in range(len(true_day)):
            z = true_day[h] - pred_day[h]  # 观测残差

            if not self.initialized:
                self.x = z
                self.P = self.R
                self.initialized = True
                continue

            # 预测步
            x_pred = self.x
            P_pred = self.P + self.Q

            # 更新步
            K = P_pred / (P_pred + self.R)  # 卡尔曼增益
            self.x = x_pred + K * (z - x_pred)
            self.P = (1 - K) * P_pred


class ARIMACorrection:
    """ARIMA(1,0,0) 残差修正: 每天拟合一次 ARIMA(1,0,0), 预测残差。"""
    def __init__(self, min_history=24):
        self.min_history = min_history
        self.history = []

    def correct(self, pred_day):
        from statsmodels.tsa.arima.model import ARIMA

        n_hours = len(pred_day)
        if len(self.history) >= self.min_history:
            try:
                model = ARIMA(self.history, order=(1, 0, 0))
                fit = model.fit()
                forecasts = fit.forecast(steps=n_hours)
            except Exception:
                forecasts = np.full(n_hours, np.mean(self.history[-48:]))
        else:
            mean_r = np.mean(self.history) if self.history else 0.0
            forecasts = np.full(n_hours, mean_r)

        return pred_day + forecasts

    def update(self, pred_day, true_day):
        residuals = true_day - pred_day
        self.history.extend(residuals.tolist())
        max_hist = 24 * 30
        if len(self.history) > max_hist:
            self.history = self.history[-max_hist:]


class OnlineLinearCorrection:
    """在线线性修正: 用历史残差特征训练轻量线性模型, 预测修正量。

    特征: [hour_sin, hour_cos, dow_sin, dow_cos, 残差均值(24h), 残差均值(168h), 残差std(24h)]
    使用特征归一化 + L2 正则化防止权重爆炸。
    """
    def __init__(self, lr=0.001, n_features=7, weight_decay=0.01):
        self.lr = lr
        self.n_features = n_features
        self.weight_decay = weight_decay
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history = []
        # 在线均值/方差用于归一化
        self.feat_sum = np.zeros(n_features)
        self.feat_sq_sum = np.zeros(n_features)
        self.feat_count = 0
        self.min_history = 48

    def _make_features(self, hour_of_day, dow, history):
        hour_sin = math.sin(2 * math.pi * hour_of_day / 24)
        hour_cos = math.cos(2 * math.pi * hour_of_day / 24)
        dow_sin = math.sin(2 * math.pi * dow / 7)
        dow_cos = math.cos(2 * math.pi * dow / 7)

        if len(history) >= 24:
            mean_24 = np.mean(history[-24:])
            std_24 = np.std(history[-24:])
        else:
            mean_24 = np.mean(history) if history else 0.0
            std_24 = 0.0

        if len(history) >= 168:
            mean_168 = np.mean(history[-168:])
        else:
            mean_168 = mean_24

        return np.array([hour_sin, hour_cos, dow_sin, dow_cos,
                         mean_24, mean_168, std_24])

    def _normalize(self, feat):
        """在线 z-score 归一化。"""
        self.feat_count += 1
        self.feat_sum += feat
        self.feat_sq_sum += feat ** 2
        mean = self.feat_sum / self.feat_count
        var = self.feat_sq_sum / self.feat_count - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8))
        return (feat - mean) / std

    def correct(self, pred_day, timestamps):
        corrected = np.zeros_like(pred_day)
        for h in range(len(pred_day)):
            ts = timestamps[h]
            feat = self._make_features(ts.hour, ts.dayofweek, self.history)
            feat_norm = self._normalize(feat)
            correction = np.dot(self.w, feat_norm) + self.b
            corrected[h] = pred_day[h] + correction
        return corrected

    def update(self, pred_day, true_day, timestamps):
        residuals = true_day - pred_day
        for h in range(len(residuals)):
            ts = timestamps[h]
            feat = self._make_features(ts.hour, ts.dayofweek, self.history)
            feat_norm = self._normalize(feat)
            r = residuals[h]

            pred_val = np.dot(self.w, feat_norm) + self.b
            error = r - pred_val
            # L2 正则化 + 梯度下降
            self.w *= (1 - self.lr * self.weight_decay)
            self.w += self.lr * error * feat_norm
            self.b += self.lr * error

            self.history.append(r)

        max_hist = 24 * 60
        if len(self.history) > max_hist:
            self.history = self.history[-max_hist:]


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="残差修正方法对比")
    parser.add_argument("--data", default=DEFAULT_DATA, help="原始数据 CSV")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="训练结果目录")
    parser.add_argument("--lookback", type=int, default=7, help="回看天数")
    parser.add_argument("--start_date", default=None, help="评估起始日期")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--all_days", action="store_true", help="逐日预测到数据末尾")
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
    target_feat_idx = saved.get("target_feat_idx", feature_cols.index(target_cols[0]))

    model_type = config.get("model_type", "transformer")
    model_lookback_steps = int(config["lookback_days"] * 24)
    model_kwargs = dict(
        input_dim=len(feature_cols), output_dim=1, horizon=1,
        input_len=model_lookback_steps,
        d_model=config["d_model"], nhead=config["nhead"],
        num_layers=config["num_layers"], dim_feedforward=config["dim_feedforward"],
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

    print(f"模型: {model_type}, predict={predict_steps}步, lookback={lookback}d")

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
            print(f"错误: 回看数据不足")
            return
    else:
        max_start = total_len - (lookback + 1) * points_per_day
        start = random.randint(0, max_start)
        first_pred_idx = start + lookback_steps

    max_pred_days = (total_len - first_pred_idx) // points_per_day
    n_predict_days = max_pred_days if args.all_days else min(lookback, max_pred_days)
    if n_predict_days < 1:
        print("错误: 无可预测天数")
        return

    print(f"起始: {df_feat.index[first_pred_idx].date()}, 预测 {n_predict_days} 天")

    # ── 逐日预测 ──
    print(f"\n{'='*70}")
    print(f" 逐日预测 ({n_predict_days} 天)")
    print(f"{'='*70}")

    all_preds = []
    all_trues = []
    all_timestamps = []

    for day in range(n_predict_days):
        pred_start = first_pred_idx + day * points_per_day
        pred_end = pred_start + predict_steps
        if pred_end > total_len:
            break

        if day % 50 == 0 or day == n_predict_days - 1:
            print(f"  Day {day+1:>4}/{n_predict_days}: {df_feat.index[pred_start].strftime('%Y-%m-%d')}")

        lb_start = pred_start - lookback_steps
        window_scaled = data_scaled[lb_start:pred_start]
        window_t = torch.from_numpy(window_scaled).unsqueeze(0).to(device)
        future_raw = df_feat.iloc[pred_start:pred_end][feature_cols].values.copy()
        future_scaled = feature_scaler.transform(future_raw.astype(np.float32))

        pred_scaled = autoregressive_predict(model, window_t, future_scaled,
                                             predict_steps, target_feat_idx, device)
        pred_inv = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        true_inv = target_scaler.inverse_transform(
            data_scaled[pred_start:pred_end, 0].reshape(-1, 1)).flatten()

        all_preds.append(pred_inv)
        all_trues.append(true_inv)
        all_timestamps.append(df_feat.index[pred_start:pred_end])

    n_actual = len(all_preds)
    all_true_arr = np.concatenate(all_trues)
    all_orig_arr = np.concatenate(all_preds)
    baseline_mae, baseline_rmse, baseline_mape = calc_metrics(all_true_arr, all_orig_arr)

    print(f"\n{'='*70}")
    print(f" Baseline: MAE={baseline_mae:.2f}, RMSE={baseline_rmse:.2f}, MAPE={baseline_mape:.2f}%")
    print(f"{'='*70}")

    # ── 初始化各修正方法 ──
    methods = {
        "Baseline":           None,
        "滑动窗口(w=168)":    SlidingWindowCorrection(window=168),
        "EWMA(α=0.3)":        EWMACorrection(alpha=0.3),
        "STL(周期=24)":       STLCorrection(period=24, min_history=48),
        "卡尔曼滤波":          KalmanCorrection(process_noise=1.0, measurement_noise=100.0),
        "ARIMA(1,0,0)":       ARIMACorrection(min_history=24),
        "在线线性修正":         OnlineLinearCorrection(lr=0.01),
    }

    method_corrected = {m: [] for m in methods if m != "Baseline"}

    # ── 逐日修正 (在线模式: 用当天真实值更新状态, 修正下一天) ──
    print(f"\n逐日修正中...")

    for day in range(n_actual):
        pred = all_preds[day]
        true = all_trues[day]
        ts = all_timestamps[day]

        for mname, mobj in methods.items():
            if mname == "Baseline":
                continue
            if mname == "在线线性修正":
                corr = mobj.correct(pred, ts)
            else:
                corr = mobj.correct(pred)
            method_corrected[mname].append(corr)

        # 用当天真实值更新各方法状态 (模拟在线场景)
        for mname, mobj in methods.items():
            if mname == "Baseline":
                continue
            if mname == "在线线性修正":
                mobj.update(pred, true, ts)
            else:
                mobj.update(pred, true)

    # ── 汇总指标 ──
    print(f"\n{'='*70}")
    print(f" 修正结果对比 ({n_actual} 天, {len(all_true_arr)} 个时间点)")
    print(f"{'='*70}")
    print(f"  {'方法':<20}{'MAE':<10}{'MAE提升':<12}{'RMSE':<10}{'RMSE提升':<12}{'MAPE%':<10}{'MAPE提升':<12}")
    print(f"  {'-'*86}")

    results = []
    # Baseline
    print(f"  {'Baseline':<20}{baseline_mae:<10.2f}{'---':<12}{baseline_rmse:<10.2f}{'---':<12}{baseline_mape:<10.2f}{'---':<12}")
    results.append({"method": "Baseline", "mae": baseline_mae, "rmse": baseline_rmse, "mape": baseline_mape})

    best_mape = baseline_mape
    best_method = "Baseline"

    for mname in method_corrected:
        all_corr = np.concatenate(method_corrected[mname])
        c_mae, c_rmse, c_mape = calc_metrics(all_true_arr, all_corr)
        mae_imp = (1 - c_mae / baseline_mae) * 100
        rmse_imp = (1 - c_rmse / baseline_rmse) * 100
        mape_imp = (1 - c_mape / baseline_mape) * 100

        print(f"  {mname:<20}{c_mae:<10.2f}{mae_imp:>+10.2f}%  "
              f"{c_rmse:<10.2f}{rmse_imp:>+10.2f}%  "
              f"{c_mape:<10.2f}{mape_imp:>+10.2f}%")
        results.append({"method": mname, "mae": c_mae, "rmse": c_rmse, "mape": c_mape,
                         "mae_imp": mae_imp, "rmse_imp": rmse_imp, "mape_imp": mape_imp})

        if c_mape < best_mape:
            best_mape = c_mape
            best_method = mname

    print(f"\n  最优方法: {best_method} (MAPE={best_mape:.2f}%)")

    # ── 保存 CSV ──
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(HERE, "residual_correction_results.csv")
    results_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n结果已保存: {csv_path}")

    # ── 画图 ──
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图1: 指标对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    method_names = [r["method"] for r in results]
    colors = ["#95a5a6"] + ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    for ax, metric, title in zip(axes, ["mae", "rmse", "mape"], ["MAE", "RMSE", "MAPE%"]):
        vals = [r[metric] for r in results]
        bars = ax.bar(range(len(vals)), vals, color=colors[:len(vals)], edgecolor="white")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(method_names, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        # 标注数值
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"残差修正方法对比 ({n_actual}天)", fontsize=14)
    fig.tight_layout()
    fig_path = os.path.join(HERE, "residual_correction_comparison.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已保存: {fig_path}")

    # 图2: MAE 提升百分比
    imp_results = [r for r in results if "mae_imp" in r]
    if imp_results:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        imp_names = [r["method"] for r in imp_results]
        imps = [r["mae_imp"] for r in imp_results]
        bar_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in imps]
        ax2.barh(range(len(imp_names)), imps, color=bar_colors, edgecolor="white")
        ax2.set_yticks(range(len(imp_names)))
        ax2.set_yticklabels(imp_names)
        ax2.axvline(x=0, color="black", linewidth=0.8)
        ax2.set_xlabel("MAE 提升 (%)")
        ax2.set_title("各修正方法 vs Baseline")
        ax2.grid(axis="x", alpha=0.3)
        for i, v in enumerate(imps):
            ax2.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.2f}%",
                     va="center", ha="left" if v >= 0 else "right", fontsize=9)
        fig2.tight_layout()
        imp_fig_path = os.path.join(HERE, "residual_correction_improvement.png")
        fig2.savefig(imp_fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"提升图已保存: {imp_fig_path}")


if __name__ == "__main__":
    main()

"""
train_transformer_ensemble.py — TimesFM + AR 自回归集成训练脚本

在 train_transformer_autoregressive.py 基础上，将 TimesFM 预训练模型的预测值
作为额外特征通道输入 Transformer，让模型自适应融合两个预测信号。

核心改动：
1. 预计算 TimesFM 对每个时间步的 24h 预测（按天粒度）
2. 在 feature_cols 末尾追加 "timesfm_pred" 通道（C: 31 → 32）
3. 自回归 rollout 时，TimesFM 值随 future_exog 滑动进入模型视野
"""

import os
import sys
import argparse
import math
import pickle
import random
from copy import deepcopy
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error

from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor

try:
    from chinese_calendar import is_holiday, is_workday as _cal_is_workday
    _HAS_CHINESE_CALENDAR = True
except ImportError:
    _HAS_CHINESE_CALENDAR = False

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 配置
# ============================================================================

BASE_CONFIG = {
    "file_path": r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv",
    "encoding": "utf-8-sig",
    "resample_freq": "60min",
    "stride": 1,
    "hampel_cols": ["Total_Flow"],
    "hampel_window": 48,
    "spike_ratio_within": 1.2,
    "spike_ratio_cross": 1.3,

    "lookback_days": 7,
    "predict_days": 1.0,
    "label": f"junshan_ensemble_{time.strftime('%Y%m%d_%H%M%S')}",

    "test_days": 90,
    "mape_floor_ratio": 0.1,
    "target_transform": None,

    # Transformer 架构
    "d_model": 32,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 256,
    "transformer_dropout": 0.2,
    "model_type": "transformer",  # transformer | itransformer

    # 自回归
    "detach_feedback": True,

    # 峰值样本增强
    "peak_augment_ratio": 0.4,
    "peak_threshold_ratio": 0.5,

    # 训练超参
    "batch_size": 64,
    "epochs": 40,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "patience": 10,
    "min_delta": 1e-4,
    "T_0": 30,
    "T_mult": 2,
    "eval_interval": 2,

    # TimesFM 集成配置
    "use_timesfm": True,
    "timesfm_model_path": r"D:\Junshan_Project\models\timesfm-2.5-200m-pytorch",
    "timesfm_cache_path": os.path.join(HERE, "timesfm_cache.pkl"),

    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "base_result_dir": os.path.join(HERE, "results"),
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def add_calendar_features_standalone(df_index):
    """由 DatetimeIndex 确定性生成日历特征 (独立函数, 供 TimesFM 预计算使用)"""
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


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.stop = False

    def step(self, val):
        if self.best is None:
            self.best = val
            return False
        if val < self.best - self.min_delta:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def weighted_mse_loss(pred, target, flow_weight=1.0):
    flow_loss = ((pred[:, :, 0] - target[:, :, 0]) ** 2).mean()
    return flow_weight * flow_loss


def compute_mape(y_true, y_pred, floor_ratio=0.05):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n_total = len(y_true)
    if n_total == 0:
        return 0.0, 0, 0
    thr = floor_ratio * np.abs(y_true).max()
    mask = np.abs(y_true) >= thr
    n_used = int(mask.sum())
    if n_used == 0:
        return 0.0, n_total, 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-8))) * 100
    return mape, n_total, n_used


# ============================================================================
# TimesFM 预计算
# ============================================================================

def precompute_timesfm_predictions(df, config):
    """预计算 TimesFM 对数据集中每个时间点的 24h 预测。

    按天粒度运行：每天 00:00 取前14天上下文，预测次日24h。
    返回与 df 等长的数组，TimesFM 预测值对齐到对应的时间位置。
    """
    cache_path = config.get("timesfm_cache_path")
    if cache_path and os.path.exists(cache_path):
        print(f"  加载 TimesFM 缓存: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        # 校验缓存是否匹配当前数据
        if (cache.get("index") is not None and
            len(cache["index"]) == len(df) and
            (cache["index"] == df.index).all()):
            print(f"  缓存命中，跳过 TimesFM 预计算")
            return cache["timesfm_array"]
        print(f"  缓存不匹配，重新计算...")

    print("  预计算 TimesFM 预测...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    import timesfm

    model_path = config["timesfm_model_path"]
    context_len = 336  # 14天
    horizon = 24

    tfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path)
    tfm_config = timesfm.ForecastConfig(
        max_context=context_len,
        max_horizon=horizon,
        use_continuous_quantile_head=True,
        return_backcast=True,
    )
    tfm_model.compile(tfm_config)

    flow_col = "Total_Flow"
    timesfm_array = np.full(len(df), np.nan, dtype=np.float32)

    # 按天预计算: 从第 context_len 个点开始，每天预测一次24h
    date_index = df.index.normalize()
    unique_dates = sorted(date_index.unique())
    total_days = len(unique_dates)

    # 跳过前14天（无足够上下文）
    valid_dates = [d for d in unique_dates
                   if df.index.searchsorted(d) >= context_len]

    print(f"  共 {total_days} 天, 可预测 {len(valid_dates)} 天")

    for i, target_date in enumerate(tqdm(valid_dates, desc="TimesFM预计算", unit="day")):
        # 找到目标日期在 df 中的位置
        target_start = df.index.searchsorted(target_date)
        if target_start + horizon > len(df):
            continue

        # 取前 context_len 个点作为上下文
        ctx_start = target_start - context_len
        if ctx_start < 0:
            continue

        context = df[flow_col].iloc[ctx_start:target_start].values.astype(np.float32).tolist()

        # 外生变量: 历史 + 未来
        hist_idx = df.index[ctx_start:target_start]
        future_idx = df.index[target_start:target_start + horizon]
        full_idx = hist_idx.append(future_idx)

        hist_cov = add_calendar_features_standalone(hist_idx)
        future_cov = add_calendar_features_standalone(future_idx)
        covariates = {}
        for k in hist_cov:
            covariates[k] = np.concatenate([hist_cov[k], future_cov[k]]).tolist()

        point_outputs, _ = tfm_model.forecast_with_covariates(
            inputs=[context],
            dynamic_numerical_covariates={k: [v] for k, v in covariates.items()},
        )
        pred = point_outputs[0].flatten()  # (24,)
        timesfm_array[target_start:target_start + horizon] = pred

    # 前 context_len 个点填 0（无预测）
    timesfm_array[:context_len] = 0.0
    # 仍有 NaN 的位置填 0
    nan_mask = np.isnan(timesfm_array)
    if nan_mask.any():
        print(f"  ⚠ {nan_mask.sum()} 个点无 TimesFM 预测，填 0")
        timesfm_array[nan_mask] = 0.0

    print(f"  TimesFM 预计算完成，有效预测点: {(timesfm_array != 0).sum()}")

    # 缓存
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "index": df.index,
                "timesfm_array": timesfm_array,
            }, f)
        print(f"  缓存已保存: {cache_path}")

    return timesfm_array


# ============================================================================
# 自回归核心
# ============================================================================

def autoregressive_rollout(model, src, future_exog, predict_steps,
                           target_feat_idx, detach_feedback=True):
    """单步自回归 rollout。future_exog 最后一列为 TimesFM 预测值。"""
    window = src
    preds = []
    for k in range(predict_steps):
        one = model(window, target_len=1)
        preds.append(one[:, 0:1, :])

        feedback = one[:, 0, 0]
        if detach_feedback:
            feedback = feedback.detach()
        next_row = future_exog[:, k:k + 1, :].clone()
        next_row[:, 0, target_feat_idx] = feedback

        window = torch.cat([window[:, 1:, :], next_row], dim=1)

    return torch.cat(preds, dim=1)


# ============================================================================
# 评估
# ============================================================================

def evaluate(model, loader, device, processor, predict_steps,
             target_feat_idx, detach_feedback):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch_x, batch_y, batch_xf in tqdm(loader, desc="Evaluating",
                                                unit="batch", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_xf = batch_xf.to(device)
            pred = autoregressive_rollout(model, batch_x, batch_xf, predict_steps,
                                          target_feat_idx, detach_feedback)
            loss = weighted_mse_loss(pred, batch_y)
            total_loss += loss.item() * len(batch_x)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())

    if len(all_preds) == 0:
        return {"loss": 0, "flow_mae": 0, "flow_rmse": 0, "flow_mape": 0,
                "y_pred_inv": np.array([]), "y_true_inv": np.array([])}

    avg_loss = total_loss / len(loader.dataset)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)
    y_pred_inv = processor.inverse_transform_targets(y_pred)
    y_true_inv = processor.inverse_transform_targets(y_true)

    y_true_flat = y_true_inv[:, :, 0].reshape(-1)
    y_pred_flat = y_pred_inv[:, :, 0].reshape(-1)
    flow_mae = mean_absolute_error(y_true_flat, y_pred_flat)
    flow_rmse = math.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    flow_mape, mape_n_total, mape_n_used = compute_mape(
        y_true_flat, y_pred_flat, processor.config.get("mape_floor_ratio", 0.05))

    return {
        "loss": avg_loss, "flow_mae": flow_mae, "flow_rmse": flow_rmse,
        "flow_mape": flow_mape, "mape_n_used": mape_n_used, "mape_n_total": mape_n_total,
        "y_pred_inv": y_pred_inv, "y_true_inv": y_true_inv
    }


# ============================================================================
# 画图 / 统计
# ============================================================================

def plot_best_worst_cases(y_true_inv, y_pred_inv, save_dir, num_best=30, num_worst=30, title_prefix="Test"):
    n_samples = len(y_true_inv)
    if n_samples == 0:
        return

    per_sample_mae = np.mean(np.abs(y_true_inv - y_pred_inv), axis=(1, 2))
    sorted_idx = np.argsort(per_sample_mae)
    best_idx = sorted_idx[:min(num_best, n_samples)]
    worst_idx = sorted_idx[-min(num_worst, n_samples):][::-1]

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    ncols = 5

    def draw_grid(indices, nrows, tag, save_filename):
        nrows = max(nrows, 1)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2))
        axes = np.atleast_2d(axes)
        for i, idx in enumerate(indices):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            ax.plot(y_true_inv[idx, :, 0], color='#2c3e50', linewidth=1.2, label='True')
            ax.plot(y_pred_inv[idx, :, 0], color='#e74c3c', linewidth=1.2, linestyle='--', label='Pred')
            ax.set_title(f"#{i+1} MAE={per_sample_mae[idx]:.0f}", fontsize=9)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc='upper right')
        for j in range(len(indices), nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].set_visible(False)
        fig.suptitle(f"{title_prefix} - {tag} ({len(indices)} cases)", fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, save_filename), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"    {tag}图已保存: {os.path.join(save_dir, save_filename)}")

    nrows_best = int(np.ceil(len(best_idx) / ncols))
    nrows_worst = int(np.ceil(len(worst_idx) / ncols))
    draw_grid(best_idx, nrows_best, "Best", f"{title_prefix}_best_cases.png")
    draw_grid(worst_idx, nrows_worst, "Worst", f"{title_prefix}_worst_cases.png")


def plot_error_distribution(y_true_inv, y_pred_inv, save_path, title_prefix="Test"):
    if len(y_true_inv) == 0:
        return
    y_true_flat = y_true_inv[:, :, 0].reshape(-1)
    y_pred_flat = y_pred_inv[:, :, 0].reshape(-1)
    abs_errors = np.abs(y_true_flat - y_pred_flat)
    relative_errors = np.where(y_true_flat != 0, abs_errors / np.abs(y_true_flat), 0)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(abs_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(abs_errors), color='red', linestyle='--', label=f'Mean AE: {np.mean(abs_errors):.2f}')
    axes[0].axvline(np.median(abs_errors), color='green', linestyle='--', label=f'Median AE: {np.median(abs_errors):.2f}')
    axes[0].set_title(f"{title_prefix} Absolute Error Distribution")
    axes[0].set_xlabel("Absolute Error")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    mask = relative_errors < 0.5
    rel_err_display = relative_errors[mask]
    axes[1].hist(rel_err_display * 100, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(rel_err_display) * 100, color='red', linestyle='--', label=f'Mean RE: {np.mean(rel_err_display)*100:.2f}%')
    axes[1].axvline(np.median(rel_err_display) * 100, color='green', linestyle='--', label=f'Median RE: {np.mean(rel_err_display)*100:.2f}%')
    axes[1].set_title(f"{title_prefix} Relative Error Distribution (<50%)")
    axes[1].set_xlabel("Relative Error (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    误差分布图已保存: {save_path}")


def stat_by_start_time(y_true_inv, y_pred_inv, start_times, save_dir, floor_ratio=0.05):
    n = len(y_true_inv)
    if n == 0 or start_times is None or len(start_times) != n:
        print("    ⚠ 样本数与起点时间不匹配, 跳过按起点时刻统计")
        return

    slots = np.array([t.hour * 2 + (1 if t.minute >= 30 else 0) for t in start_times])
    y_true = y_true_inv[:, :, 0]
    y_pred = y_pred_inv[:, :, 0]

    thr = floor_ratio * np.abs(y_true).max()

    rows = []
    for s in range(48):
        mask = slots == s
        n_slot = int(mask.sum())
        if n_slot == 0:
            continue
        tt = y_true[mask].reshape(-1)
        pp = y_pred[mask].reshape(-1)
        mae = mean_absolute_error(tt, pp)
        rmse = math.sqrt(mean_squared_error(tt, pp))
        m_map = np.abs(tt) >= thr
        mape = (np.mean(np.abs((tt[m_map] - pp[m_map]) / (tt[m_map] + 1e-8))) * 100
                if m_map.sum() > 0 else 0.0)
        rows.append({
            "start_time": f"{s // 2:02d}:{30 if s % 2 else 0:02d}",
            "n_samples": n_slot, "mae": mae, "rmse": rmse, "mape": mape,
        })

    df_stat = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "test_start_time_metrics.csv")
    df_stat.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"    按起点时刻精度统计已保存: {csv_path}")
    print(f"    起点时刻  样本数      MAE      RMSE     MAPE%")
    for r in rows:
        print(f"    {r['start_time']}  {r['n_samples']:<6}  "
              f"{r['mae']:<8.2f}  {r['rmse']:<8.2f}  {r['mape']:<8.2f}")


# ============================================================================
# 数据集 & 序列构建
# ============================================================================

class ARSeqDataset(Dataset):
    def __init__(self, X, Y, Xf):
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.Y = torch.from_numpy(np.ascontiguousarray(Y))
        self.Xf = torch.from_numpy(np.ascontiguousarray(Xf))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Xf[idx]


def make_sequences_ar(processor, x_array, y_array, lookback_days, predict_days,
                      timesfm_array=None):
    """生成回看窗口 + 未来目标 + 未来外生特征行。

    若 timesfm_array 不为 None, 则在 X/Xf 的最后一列追加 TimesFM 预测值。
    X 中 lookback 窗口的 TimesFM 列填 0（历史无预测）。
    """
    freq_minutes = int(processor.config["resample_freq"].replace("min", ""))
    points_per_day = (24 * 60) // freq_minutes
    lookback_steps = int(lookback_days * points_per_day)
    horizon_steps = int(predict_days * points_per_day)
    stride = processor.config["stride"]

    total_len = len(x_array)
    n_samples = (total_len - lookback_steps - horizon_steps) // stride + 1
    if n_samples <= 0:
        empty = np.empty((0, lookback_steps, x_array.shape[1]), dtype=np.float32)
        empty_h = np.empty((0, horizon_steps, x_array.shape[1]), dtype=np.float32)
        empty_y = np.empty((0, horizon_steps, y_array.shape[1]), dtype=np.float32)
        return empty, empty_y, empty_h

    use_tfm = timesfm_array is not None
    C = x_array.shape[1] + (1 if use_tfm else 0)

    X = np.empty((n_samples, lookback_steps, C), dtype=np.float32)
    Y = np.empty((n_samples, horizon_steps, y_array.shape[1]), dtype=np.float32)
    Xf = np.empty((n_samples, horizon_steps, C), dtype=np.float32)

    idx = 0
    for i in range(0, total_len - lookback_steps - horizon_steps + 1, stride):
        # 基础特征
        X[idx, :, :x_array.shape[1]] = x_array[i:i + lookback_steps]
        Y[idx] = y_array[i + lookback_steps:i + lookback_steps + horizon_steps]
        Xf[idx, :, :x_array.shape[1]] = x_array[i + lookback_steps:i + lookback_steps + horizon_steps]

        # TimesFM 通道
        if use_tfm:
            # X 中的 TimesFM 列: lookback 窗口填 0（历史无预测）
            X[idx, :, -1] = 0.0
            # Xf 中的 TimesFM 列: 对应时间点的预计算值
            tfm_start = i + lookback_steps
            tfm_end = tfm_start + horizon_steps
            Xf[idx, :, -1] = timesfm_array[tfm_start:tfm_end]

        idx += 1
    return X, Y, Xf


def augment_peak_samples(X, Y, Xf, peak_threshold_ratio=0.7, peak_augment_ratio=0.3):
    """峰值样本增强"""
    if len(X) == 0:
        return X, Y, Xf

    sample_max = Y[:, :, 0].max(axis=1)
    global_max = sample_max.max()
    threshold = peak_threshold_ratio * global_max

    peak_indices = np.where(sample_max > threshold)[0]
    if len(peak_indices) == 0:
        print(f"  ⚠ 未找到峰值样本（阈值={threshold:.2f}），跳过增强")
        return X, Y, Xf

    n_original = len(X)
    n_target_aug = int(n_original * peak_augment_ratio)

    aug_indices = np.random.choice(peak_indices, n_target_aug, replace=True)

    X_aug = np.concatenate([X, X[aug_indices]], axis=0)
    Y_aug = np.concatenate([Y, Y[aug_indices]], axis=0)
    Xf_aug = np.concatenate([Xf, Xf[aug_indices]], axis=0)

    print(f"  ✅ 峰值样本增强完成:")
    print(f"     原始样本数: {n_original}")
    print(f"     峰值样本数: {len(peak_indices)} (阈值={threshold:.2f})")
    print(f"     新增样本数: {len(aug_indices)}")
    print(f"     增强后总样本数: {len(X_aug)}")

    return X_aug, Y_aug, Xf_aug


# ============================================================================
# 单次实验
# ============================================================================

def run_experiment(cfg, x_train_all, y_train_all, x_test_all, y_test_all,
                   processor, device, test_index=None, timesfm_array=None):
    processor.config = cfg
    lookback = cfg["lookback_days"]
    predict = cfg["predict_days"]
    label = cfg["label"]
    detach_feedback = cfg.get("detach_feedback", True)
    use_timesfm = cfg.get("use_timesfm", False) and timesfm_array is not None

    result_dir = os.path.join(cfg["base_result_dir"], label)
    ensure_dir(result_dir)

    print(f"\n{'='*80}")
    print(f" 实验(集成): {label}")
    print(f"  model_type={cfg.get('model_type', 'transformer')}")
    print(f"  lookback={lookback}d | predict={predict}d | freq={cfg['resample_freq']}")
    print(f"  test_days={cfg['test_days']} | d_model={cfg['d_model']}")
    print(f"  TimesFM={'启用' if use_timesfm else '禁用'}")
    print(f"{'='*80}")

    # 构建序列
    X_train, Y_train, Xf_train = make_sequences_ar(
        processor, x_train_all, y_train_all, lookback, predict,
        timesfm_array=timesfm_array if use_timesfm else None)
    X_test, Y_test, Xf_test = make_sequences_ar(
        processor, x_test_all, y_test_all, lookback, predict,
        timesfm_array=timesfm_array if use_timesfm else None)

    # 峰值样本增强
    peak_augment_ratio = cfg.get("peak_augment_ratio", 0.3)
    peak_threshold_ratio = cfg.get("peak_threshold_ratio", 0.7)
    if peak_augment_ratio > 0 and len(X_train) > 0:
        X_train, Y_train, Xf_train = augment_peak_samples(
            X_train, Y_train, Xf_train,
            peak_threshold_ratio=peak_threshold_ratio,
            peak_augment_ratio=peak_augment_ratio
        )

    freq_minutes = int(cfg["resample_freq"].replace("min", ""))
    points_per_day = (24 * 60) // freq_minutes
    predict_steps = int(predict * points_per_day)

    stride = cfg.get("stride", 1)
    lookback_steps = int(lookback * points_per_day)
    if test_index is not None:
        test_starts = test_index[lookback_steps:len(X_test) * stride + lookback_steps:stride]
    else:
        test_starts = None

    target_feat_idx = processor.feature_cols.index(processor.target_cols[0])

    print(f"  输入维度: {X_train.shape[2]} (原 {len(processor.feature_cols)} + "
          f"{'TimesFM 1' if use_timesfm else '无 TimesFM'})")
    print(f"  target_feat_idx={target_feat_idx} ({processor.target_cols[0]})")
    print(f"  X_train={X_train.shape}, Y_train={Y_train.shape}")
    print(f"  X_test={X_test.shape}, Y_test={Y_test.shape}")

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"  ⚠ 样本数为0，跳过此配置")
        return None

    train_loader = DataLoader(ARSeqDataset(X_train, Y_train, Xf_train),
                              batch_size=cfg["batch_size"], shuffle=True,
                              pin_memory=True, num_workers=2)
    test_loader = DataLoader(ARSeqDataset(X_test, Y_test, Xf_test),
                            batch_size=cfg["batch_size"], shuffle=False,
                            pin_memory=True, num_workers=2)

    # 模型工厂
    model_type = cfg.get("model_type", "transformer")
    common_model_kwargs = dict(
        input_dim=X_train.shape[2],
        output_dim=1,
        horizon=1,
        input_len=lookback_steps,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["transformer_dropout"],
    )
    if model_type == "itransformer":
        assert cfg.get("target_transform") is None, \
            "iTransformer 要求 target_transform=None"
        model = iTransformer(
            **common_model_kwargs,
            target_idx=target_feat_idx,
        ).to(device)
    else:
        model = TimeSeriesTransformer(**common_model_kwargs).to(device)

    if hasattr(torch, "compile") and tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 1):
        try:
            model = torch.compile(model)
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile failed ({e}), using eager mode")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg["T_0"], T_mult=cfg["T_mult"])
    early_stopper = EarlyStopping(cfg["patience"], cfg["min_delta"])

    best_state = None
    best_test_mape = float("inf")
    history = []
    test_metrics = {}

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    eval_interval = cfg.get("eval_interval", 1)

    print(f"\n  Training(集成): {label}")
    print(f"  AMP={'ON' if use_amp else 'OFF'}, eval_interval={eval_interval}")

    epoch_pbar = tqdm(range(1, cfg["epochs"] + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss_sum = 0.0

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}",
                          unit="batch", leave=False)
        for batch_x, batch_y, batch_xf in batch_pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_xf = batch_xf.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                pred = autoregressive_rollout(model, batch_x, batch_xf, predict_steps,
                                             target_feat_idx, detach_feedback)
                loss = weighted_mse_loss(pred, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(batch_x)
            batch_pbar.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = train_loss_sum / len(train_loader.dataset)

        do_eval = (epoch % eval_interval == 0) or (epoch == cfg["epochs"])
        if do_eval:
            test_metrics = evaluate(model, test_loader, device, processor, predict_steps,
                                    target_feat_idx, detach_feedback)

        test_loss = test_metrics.get("loss", float("nan"))
        current_mape = test_metrics.get("flow_mape", float("inf"))

        if do_eval and current_mape < best_test_mape:
            best_test_mape = current_mape
            state_to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            best_state = deepcopy(state_to_save)
            torch.save(best_state, os.path.join(result_dir, "best_seq2seq_model.pth"))

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": train_loss, "test_loss": test_loss,
            "flow_mae": test_metrics.get("flow_mae", float("nan")),
            "flow_rmse": test_metrics.get("flow_rmse", float("nan")),
            "flow_mape": current_mape, "lr": lr_now
        })

        eval_tag = "" if do_eval else " [skip eval]"
        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.6f}",
            test_loss=f"{test_loss:.6f}" if not math.isnan(test_loss) else "---",
            best_MAPE=f"{best_test_mape:.2f}%",
            cur_MAPE=f"{current_mape:.2f}%" if current_mape != float("inf") else "---",
            lr=f"{lr_now:.6f}",
            note=eval_tag,
        )

        if early_stopper.step(current_mape):
            print(f"\n  Early stopping at epoch={epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(result_dir, "train_history.csv"), index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    # 保存 scaler
    scaler_path = os.path.join(result_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "config": cfg,
            "feature_scaler": processor.feature_scaler,
            "target_scaler": processor.target_scaler,
            "feature_cols": processor.feature_cols,
            "target_cols": processor.target_cols,
            "autoregressive": True,
            "target_feat_idx": target_feat_idx,
            "detach_feedback": detach_feedback,
            "use_timesfm": use_timesfm,
        }, f)
    print(f"  scaler/配置已保存: {scaler_path}")

    # 最终评估
    train_metrics = evaluate(model, train_loader, device, processor, predict_steps,
                             target_feat_idx, detach_feedback)
    test_metrics = evaluate(model, test_loader, device, processor, predict_steps,
                            target_feat_idx, detach_feedback)

    print(f"\n  最终结果:")
    print(f"  Train: Loss={train_metrics['loss']:.6f}, MAE={train_metrics['flow_mae']:.2f}, "
          f"RMSE={train_metrics['flow_rmse']:.2f}, MAPE={train_metrics['flow_mape']:.2f}%")
    print(f"  Test : Loss={test_metrics['loss']:.6f}, MAE={test_metrics['flow_mae']:.2f}, "
          f"RMSE={test_metrics['flow_rmse']:.2f}, MAPE={test_metrics['flow_mape']:.2f}%")

    with open(os.path.join(result_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"[集成版] TimesFM={'启用' if use_timesfm else '禁用'}\n")
        for name, m in [("Train", train_metrics), ("Test", test_metrics)]:
            f.write(f"{name}: Loss={m['loss']:.6f}, MAE={m['flow_mae']:.2f}, "
                    f"RMSE={m['flow_rmse']:.2f}, MAPE={m['flow_mape']:.2f}%\n")

    # 画图
    y_true_test = test_metrics["y_true_inv"]
    y_pred_test = test_metrics["y_pred_inv"]

    if len(y_true_test) > 0:
        plot_best_worst_cases(y_true_test, y_pred_test, result_dir, num_best=30, num_worst=30, title_prefix="Test")
        plot_error_distribution(y_true_test, y_pred_test,
                                os.path.join(result_dir, "test_error_distribution.png"), title_prefix="Test")

        mae_per_step = np.mean(np.abs(y_true_test - y_pred_test), axis=0)
        step_mae_df = pd.DataFrame({"step": np.arange(len(mae_per_step)), "flow_mae": mae_per_step[:, 0]})
        step_mae_df.to_csv(os.path.join(result_dir, "test_step_mae.csv"), index=False)

        stat_by_start_time(y_true_test, y_pred_test, test_starts, result_dir,
                           floor_ratio=cfg.get("mape_floor_ratio", 0.05))

    # loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["test_loss"], label="Test Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Training Curve (Ensemble) — {label}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "label": label,
        "use_timesfm": use_timesfm,
        "resample_freq": cfg["resample_freq"],
        "test_days": cfg["test_days"],
        "d_model": cfg["d_model"],
        "nhead": cfg["nhead"],
        "num_layers": cfg["num_layers"],
        "learning_rate": cfg["learning_rate"],
        "lookback_days": lookback,
        "predict_days": predict,
        "predict_steps": predict_steps,
        "detach_feedback": detach_feedback,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "input_dim": X_train.shape[2],
        "best_epoch": int(history_df.loc[history_df["flow_mape"].idxmin(), "epoch"]),
        "train_mae": train_metrics["flow_mae"],
        "test_mae": test_metrics["flow_mae"],
        "train_rmse": train_metrics["flow_rmse"],
        "test_rmse": test_metrics["flow_rmse"],
        "train_mape": train_metrics["flow_mape"],
        "test_mape": test_metrics["flow_mape"],
    }


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="训练 TimesFM + AR 集成模型")
    parser.add_argument("--model", choices=["transformer", "itransformer"], default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--lookback", type=float, default=None)
    parser.add_argument("--no-timesfm", dest="use_timesfm", action="store_false", default=True,
                        help="禁用 TimesFM 集成 (纯 AR 基线)")
    parser.add_argument("--detach-feedback", dest="detach_feedback",
                        action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    config = dict(BASE_CONFIG)
    if args.model is not None:
        config["model_type"] = args.model
    if args.label is not None:
        config["label"] = args.label
    if args.lookback is not None:
        config["lookback_days"] = args.lookback
    if args.detach_feedback is not None:
        config["detach_feedback"] = args.detach_feedback
    config["use_timesfm"] = args.use_timesfm

    set_seed(config["seed"])

    device = torch.device(config["device"])
    print(f"Device: {device}")
    print(f"TimesFM 集成: {'启用' if config['use_timesfm'] else '禁用'}")

    # Phase 1: 数据加载
    print("\n" + "=" * 80)
    print(f" [Phase 1] 数据加载 & 清洗 (resample_freq={config['resample_freq']})")
    print("=" * 80)

    processor = DataProcessor(config)
    df_all_feat = processor.build_feature_table()
    print(f" 全量特征表: {df_all_feat.shape}")
    print(f" 时间范围: {df_all_feat.index.min()} ~ {df_all_feat.index.max()}")

    # Phase 2: 划分训练/测试集
    print("\n" + "=" * 80)
    print(" [Phase 2] 按时间划分训练/测试集")
    print("=" * 80)

    df_train_feat, df_test_feat = processor.split_by_time(df_all_feat)
    print(f" 训练集: {df_train_feat.shape} | {df_train_feat.index.min()} ~ {df_train_feat.index.max()}")
    print(f" 测试集: {df_test_feat.shape} | {df_test_feat.index.min()} ~ {df_test_feat.index.max()}")

    processor.fit_scalers(df_train_feat)
    x_train_all, y_train_all = processor.transform_df(df_train_feat)
    x_test_all, y_test_all = processor.transform_df(df_test_feat)

    # Phase 3: 预计算 TimesFM 预测 (全量数据)
    timesfm_array = None
    if config["use_timesfm"]:
        print("\n" + "=" * 80)
        print(" [Phase 3] 预计算 TimesFM 预测")
        print("=" * 80)
        # 需要用原始数据（非标准化）来计算 TimesFM
        # 但 TimesFM 只需要流量列，从 df_all_feat 中提取
        timesfm_array = precompute_timesfm_predictions(df_all_feat, config)
        print(f"  TimesFM 数组 shape: {timesfm_array.shape}")

    # Phase 4: 训练
    print("\n" + "=" * 80)
    print(" [Phase 4] 训练集成模型")
    print("=" * 80)

    result = run_experiment(config, x_train_all, y_train_all,
                            x_test_all, y_test_all,
                            processor, device,
                            test_index=df_test_feat.index,
                            timesfm_array=timesfm_array)
    if result is not None:
        print(f"\n 训练完成! 结果保存在: {os.path.join(config['base_result_dir'], config['label'])}")
    else:
        print(f"\n ⚠ 样本数为 0, 未产生结果")


if __name__ == "__main__":
    main()

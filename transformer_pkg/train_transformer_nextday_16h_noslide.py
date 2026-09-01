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

# GBK 控制台/重定向文件无法编码 ⚠ 等非 GBK 字符 → 统一改用 UTF-8 输出
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

# ── 共享模块 (与 train_transformer_autoregressive.py 同目录, 不改动原文件) ──
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor

HERE = os.path.dirname(os.path.abspath(__file__))

# 自回归 rollout 最大步数: 样本真实步数 = 缺口 (24-H) + 目标天 24, H=1 时最长
# = 24-1+24 = 47 步 < 48; 统一 padding 到 48 步, 超出的填充步由掩码排除。
PREDICT_STEPS_MAX = 48

# ============================================================================
# train_transformer_nextday_16h_noslide.py — 每天 16 点预测第二天全天的单步
# 自回归训练脚本 (滑动截止改造前的原版快照, 2026-09-01)
#
# ⚠ 本文件 = train_transformer_nextday_16h.py 引入 slide_cutoff (每天 17 个
#   截止位置滑动) 之前的版本: 每天只在 15:00 (H=16) 采 1 个样本, 窗口
#   [D-7 0:00, D 15:00] = 184 步, 目标 D+1 全天。保留用于与滑动版对比/回退。
#   与当前版共享 data_processing.py / transformer_model.py, 特征与清洗口径一致。
#
# (以 train_transformer_autoregressive.py 为模板, 实现改动 1b/1c/3/4/5)
#
# 业务任务:
#   每天 16:00 用"前 7 天全天 (7*24=168h) + 当天 16 点前的 16 个小时
#   (0:00~15:00)"= 184 小时, 预测**第二天 (D+1) 全天 0:00~23:00 的 24 小时**。
#
# 窗口与目标对齐 (本文件的核心语义, 与初版不同 — 初版 Y 误取为窗口末尾起的
# 连续 24 步 = D 16:00~D+1 15:00, 不是"第二天"):
#   每个样本: 截止时刻 e = 回看窗口的最后一小时 (主任务 e 时刻 = 当天 15:00,
#   窗口 [e-183, e] = [D-7 0:00, D 15:00], 恰好 184 步)。
#   目标天 D+1 = 截止时刻所在天的下一天 (0:00~23:00)。
#   截止时刻到 D+1 0:00 之间是缺口 (D 16:00~23:00, 共 24-H 小时):
#     自回归 rollout 从窗口末尾起滚动, 先滚过缺口再滚目标天, 真实步数
#     = 缺口 (24-H) + 目标天 24 (H=16 时 = 32 步); 缺口步在回测中已知,
#     一并监督学习 (部署时也是模型自己滚出来, 口径一致)。
#
# 与 train_transformer_autoregressive.py 的其余区别:
#   1. 多截止点任务 (改动 1b, 现默认关闭): use_multicutoff=True 时每天 24 个
#      小时都可作截止时刻 (H=1..24), 窗口一律取"最近 184 小时", 样本量 ×24;
#      False (默认) 只保留 H=16 原任务 (每天一个样本, 训练快 ~24 倍)。
#      两种模式下 H=16 样本真实 rollout 步数 = 缺口 8 + 目标天 24 = 32,
#      统一 padding 到 PREDICT_STEPS_MAX=48, 损失/指标用 per-sample 掩码
#      只算真实步数, 目标天部分单独报告。2026-09 起默认 False (训练提速)。
#   2. 节假日目标日上采样 (改动 1c): 目标日为法定节假日的样本复制增强。
#   3. 数据清洗改进 (data_processing.py, 改动 3/4/5): 突变修正绝对幅值下限
#      spike_abs_floor / 中值平滑 smooth_median_window — 均为 config 开关,
#      本文件已启用。
#   4. 特征清理 (2026-09): 删除全部 rollout 泄漏特征 (含 flow.shift(1)/
#      滚动窗/前一天日内统计 —— 未来行含截止时刻之后的真值, 训练/推理口径
#      不一致, 训练测试集指标虚好), 见 data_processing.py DATA_DRIVEN_COLS
#      注释; 改动后必须重新训练 (旧权重与旧 scaler.pkl 不兼容)。
#
# 其余 (模型架构 / 自回归 rollout / 评估管线 / 画图) 与原版一致。
# ============================================================================

BASE_CONFIG = {
    "file_path": r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv",
    "encoding": "utf-8-sig",
    "resample_freq": "60min",
    "stride": 24,               # 非多截止点模式下的采样步长 (每天一个样本)
    "hampel_cols": ["Total_Flow"],   # 仅清洗流量 (压力/泵量测不再读取)
    "hampel_window": 48,
    "spike_ratio_within": 1.2,   # 日内 t±1h 突变阈值 (曲线平滑, 阈值可低)
    "spike_ratio_cross": 1.3,    # 跨日 t±24h 突变阈值 (日间差异大, 需更高阈值)

    "lookback_days": 7,          # 前 7 天全天
    "lookback_extra_hours": 12,  # + 当天 0:00~15:00 共 16 个小时 (16 点前)
                                 # 总回看 = 7*24 + 16 = 184 小时
    "predict_days": 1.0,         # 第二天全天 24 小时
    "label": f"junshan_L1D_P24H_1h_transformer_nextday16h_mc_{time.strftime('%Y%m%d_%H%M%S')}",

    "test_days": 90,

    "mape_floor_ratio": 0.1,
    "target_transform": None,

    # ── 数据清洗改进 (data_processing.py 对应开关, 改动 3/4/5) ──
    "spike_abs_floor": 100.0,     # 突变修正绝对幅值下限 (m³/h): 夜间低流量防误修
    "smooth_median_window": 3,    # 3h 滚动中位数去仪表噪声 (None=关闭; 会轻微削峰)

    # ── 多截止点任务 (改动 1b) ──
    "use_multicutoff": False,     # False (默认): 只保留 H=16 原任务 (每天一个样本,
                                  #   训练快 ~24 倍); True: 每天 24 个截止时刻都训练
                                  #   (样本量 ×24, 缓解数据饥饿, 但训练显著变慢)
    "multicutoff_hours": None,    # 使用的截止小时列表 (1~24), None=全部24个;
                                  # 提速可设子集 (16 点主任务始终自动保留),
                                  # 如 [16,4,8,12,20,24] 样本量减半

    # ── Transformer 架构超参 (head 改为单步: horizon=1 在工厂里固定) ──
    "d_model": 32,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 256,
    "transformer_dropout": 0.2,

    "model_type": "transformer",  # transformer | itransformer

    # 自回归专用: 回灌的预测值是否 detach (True=稳定省显存, False=完整 BPTT)
    "detach_feedback": True,

    # 峰值样本增强配置
    "peak_augment_ratio": 0.4,      # 增强比例：增强后的峰值样本占总样本数的比例
    "peak_threshold_ratio": 0.5,    # 峰值判定阈值：相对于训练集最大值的比例

    # 节假日目标日上采样 (改动 1c)
    "holiday_augment_factor": 4.0,      # 节假日目标日样本复制后的总倍数
    "holiday_augment_max_ratio": 0.25,  # 节假日样本占总样本数的上限比例

    # 训练超参 (多截止点后样本量 ×24, 每 epoch 显著变长, 可用 --epochs 调低)
    "batch_size": 16,
    "epochs": 40,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "patience": 10,
    "min_delta": 1e-4,

    "T_0": 30,
    "T_mult": 2,

    "eval_interval": 2,  # 每隔N个epoch评估一次测试集 (加速训练)

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


# ==================== 训练工具 ====================

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


# ==================== 工具函数 ====================

def masked_flow_loss(pred, target, H, totals=None, predict_steps_max=PREDICT_STEPS_MAX,
                     day_steps=24):
    """掩码 MSE: 每个样本真实 rollout 步数 total = 缺口 (24-H) + 目标天 24;
    超出真实步数的填充步 (输入为 0, 预测无意义) 不参与损失。
    缺口步与目标天步都监督 (回测中缺口已知; 部署时缺口也是模型自己滚出来)。

    totals: 每样本真实 rollout 步数 (make_sequences_ar 返回; 数据空洞/Hampel
            剔除时 < 48-H)。None = 退化为 48-H。
    """
    B, T, _ = pred.shape
    if target.shape[1] != T:                              # 分桶后 pred 只滚真实步数
        target = target[:, :T]
    if totals is None:
        totals = (predict_steps_max - H).view(B, 1)      # 真实步数 (long)
    else:
        totals = totals.view(B, 1)
    idx = torch.arange(T, device=pred.device).unsqueeze(0).expand(B, T)
    mask = (idx < totals).unsqueeze(-1).float()          # (B, T, 1)
    loss = (((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1)
    return loss


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


# ==================== 自回归核心: 单步滚动 rollout ====================

def autoregressive_rollout(model, src, future_exog, predict_steps,
                           target_feat_idx, detach_feedback=True):
    """单步自回归 rollout: 把上一轮预测回灌为输入, 滚动预测 predict_steps 个点。

    输入:
      src          : (B, L, C) ground-truth 回看窗口 (feature 域)
      future_exog  : (B, H, C) 未来 H 步的 ground-truth 外生特征行; 其中目标通道
                     (target_feat_idx) 会被本轮预测值覆盖, 其余通道 (日历特征等)
                     用真值 (外生 teacher forcing); 超出样本真实步数的行为 0
                     (填充, 预测无意义, 由外层掩码排除)
      target_feat_idx: 目标通道在 C 维的下标 (Total_Flow 在 feature_cols 中的位置)
      detach_feedback: True → 回灌预测值 detach, 梯度只训练当前步 (省显存, 稳定);
                       False → 完整 BPTT, 梯度穿过整条 rollout 链
    输出:
      preds : (B, H, 1) 预测序列 (target 域, 与 target_scaler 一致)
    """
    window = src                       # (B, L, C), 滑窗前移
    preds = []
    for k in range(predict_steps):
        one = model(window, target_len=1)              # (B, 1, 1) 单步预测
        preds.append(one[:, 0:1, :])                   # (B, 1, 1)

        # 下一行: 取未来外生特征行, 目标通道替换成本轮预测 (回灌)
        feedback = one[:, 0, 0]                        # (B,)
        if detach_feedback:
            feedback = feedback.detach()
        next_row = future_exog[:, k:k + 1, :].clone()  # (B, 1, C)
        next_row[:, 0, target_feat_idx] = feedback      # 覆盖目标通道

        # 滑窗: 丢最旧一行, 末尾拼上 next_row
        window = torch.cat([window[:, 1:, :], next_row], dim=1)

    return torch.cat(preds, dim=1)                     # (B, H, 1)


# ==================== 评估 ====================

def slice_day(arr, totals, day_steps=24):
    """取每个样本的"目标天"部分: 真实 rollout 步数中最后 day_steps 步
    → (B, day_steps, C)。totals 为每个样本的真实步数数组。
    """
    B = arr.shape[0]
    out = np.empty((B, day_steps, arr.shape[2]), dtype=arr.dtype)
    for i in range(B):
        out[i] = arr[i, totals[i] - day_steps:totals[i]]
    return out


def _agg_metrics(y_true_day, y_pred_day, floor_ratio):
    """目标天部分 (B,24,1) 的 MAE/RMSE/MAPE 聚合。"""
    yt = y_true_day[:, :, 0].reshape(-1)
    yp = y_pred_day[:, :, 0].reshape(-1)
    mae = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mean_squared_error(yt, yp))
    mape, n_total, n_used = compute_mape(yt, yp, floor_ratio)
    return mae, rmse, mape, n_total, n_used


def evaluate(model, loader, device, processor, predict_steps_max,
             target_feat_idx, detach_feedback, day_steps=24):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues, all_Hs, all_totals = [], [], [], []

    with torch.no_grad():
        for batch_x, batch_y, batch_xf, batch_H, batch_totals in tqdm(loader, desc="Evaluating",
                                                                      unit="batch", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_xf = batch_xf.to(device)
            batch_H = batch_H.to(device)
            batch_totals = batch_totals.to(device)
            pred = autoregressive_rollout(model, batch_x, batch_xf, predict_steps_max,
                                          target_feat_idx, detach_feedback)
            loss = masked_flow_loss(pred, batch_y, batch_H, batch_totals,
                                    predict_steps_max, day_steps)
            total_loss += loss.item() * len(batch_x)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
            all_Hs.append(batch_H.cpu().numpy())
            all_totals.append(batch_totals.cpu().numpy())

    empty = {"loss": 0, "flow_mae": 0, "flow_rmse": 0, "flow_mape": 0,
             "h16_mae": 0, "h16_rmse": 0, "h16_mape": 0,
             "y_pred_inv": np.array([]), "y_true_inv": np.array([]),
             "n_samples": 0, "n_h16": 0}
    if len(all_preds) == 0:
        return empty

    avg_loss = total_loss / len(loader.dataset)
    y_pred = np.concatenate(all_preds, axis=0)   # (B, 48, 1)
    y_true = np.concatenate(all_trues, axis=0)
    Hs = np.concatenate(all_Hs, axis=0)
    totals = np.concatenate(all_totals, axis=0) # 每个样本真实 rollout 步数
    floor_ratio = processor.config.get("mape_floor_ratio", 0.05)

    y_pred_inv = processor.inverse_transform_targets(y_pred)
    y_true_inv = processor.inverse_transform_targets(y_true)

    # 目标天部分 (第二天 0:00~23:00, 24 步) 单独取出评估
    y_pred_day = slice_day(y_pred_inv, totals, day_steps)
    y_true_day = slice_day(y_true_inv, totals, day_steps)

    mae, rmse, mape, n_total, n_used = _agg_metrics(y_true_day, y_pred_day, floor_ratio)

    # H=16 子集 (原始 16 点任务) 单独报告
    h16_mask = Hs == 16
    n_h16 = int(h16_mask.sum())
    if n_h16 > 0:
        h16_mae, h16_rmse, h16_mape, _, _ = _agg_metrics(
            y_true_day[h16_mask], y_pred_day[h16_mask], floor_ratio)
    else:
        h16_mae, h16_rmse, h16_mape = float("nan"), float("nan"), float("nan")

    return {
        "loss": avg_loss,
        "flow_mae": mae, "flow_rmse": rmse, "flow_mape": mape,
        "mape_n_used": n_used, "mape_n_total": n_total,
        "h16_mae": h16_mae, "h16_rmse": h16_rmse, "h16_mape": h16_mape,
        "n_samples": len(Hs), "n_h16": n_h16,
        "y_pred_inv": y_pred_day, "y_true_inv": y_true_day,   # 目标天部分 (B,24,1)
    }


# ==================== 画图 / 统计 (与原版一致) ====================

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

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "test_start_time_metrics.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"    按起点时刻精度统计已保存: {csv_path}")
    print(f"    (MAPE 过滤阈值 {thr:.2f}, 排除 |true| < 阈值 的点; 起点时刻 = 截止时刻)")
    print("    起点时刻  样本数      MAE      RMSE     MAPE%")
    for r in rows:
        print(f"    {r['start_time']}  {r['n_samples']:<6}  "
              f"{r['mae']:<8.2f}  {r['rmse']:<8.2f}  {r['mape']:<8.2f}")


# ==================== 数据序列 (带未来外生特征行) ====================

class ARSeqDataset(Dataset):
    """每个样本返回 (回看窗口 X, 未来目标 Y, 未来外生特征行 X_future,
    截止小时 H, 真实 rollout 步数 totals)。

    X_future 用于自回归 rollout 时补齐每一步的外生通道 (目标通道由预测覆盖)。
    H = 窗口内当天的小时数 (1~24); totals = make_sequences_ar 的真实步数
    (数据空洞时 < 48-H), loss 掩码用。
    """
    def __init__(self, X, Y, Xf, H, totals):
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.Y = torch.from_numpy(np.ascontiguousarray(Y))
        self.Xf = torch.from_numpy(np.ascontiguousarray(Xf))
        self.H = torch.from_numpy(np.ascontiguousarray(H))
        self.totals = torch.from_numpy(np.ascontiguousarray(totals))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx], self.Xf[idx],
                self.H[idx], self.totals[idx])


class CutoffBucketSampler:
    """按截止小时 H 分桶的 batch sampler (训练加速用)。

    同一 batch 内所有样本 H 相同 → 该 batch 的 rollout 只需滚 48-H 步,
    消除"统一 padding 到 48 步"的填充前向浪费 (多截止点平均省 ~35%)。
    桶内/桶间都打乱; 只用于训练 loader (评估 loader 保持原始顺序,
    与 test_starts 逐样本对应)。
    """
    def __init__(self, H, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.batches = []                       # [(indices, H)]
        for h in np.unique(H):
            idx = np.where(H == h)[0]
            if shuffle:
                np.random.shuffle(idx)
            for s in range(0, len(idx), batch_size):
                self.batches.append((idx[s:s + batch_size].tolist(), int(h)))
        if shuffle:
            random.shuffle(self.batches)
        self._n = len(self.batches)

    def __len__(self):
        return self._n

    def __iter__(self):
        for idxs, _h in self.batches:
            yield idxs


def make_sequences_ar(processor, index, x_array, y_array, lookback_days, predict_days,
                      lookback_extra_hours=0, use_multicutoff=True, stride=24):
    """生成回看窗口 + 未来目标 + 未来外生特征行 (供自回归 rollout) + 样本元信息。

    任务语义 (本文件):
      - 截止时刻 e = 回看窗口的最后一小时; 窗口 = 最近 lookback_steps 小时
        [e-lookback_steps+1, e] (lookback_steps = 7*24+16 = 184; 对 16 点任务
        窗口恰好 = [D-7 0:00, D 15:00], 即前 7 天全天 + 当天 16 小时)。
      - 主任务 (16 点): 截止时刻的实际小时 == 每天 15:00
        (H = 16, 窗口内含当天 0:00~15:00 共 16 小时)。
      - 多截止点 (use_multicutoff=True): 每天 24 个小时都作截止时刻 (H=1..24),
        H=16 的样本即主任务。
      - 目标 = 第二天 D+1 全天 24 小时 (截止时刻所在天 D 的下一天 0:00~23:00)。
        截止时刻到 D+1 0:00 之间为缺口 (24-H 小时), 由自回归 rollout 顺带预测
        (回测中已知, 一并监督; 部署时也是模型自己滚出来)。
      - 真实 rollout 步数 total = 缺口步 + 24 ≤ 47; Y/Xf 统一 padding 到
        PREDICT_STEPS_MAX=48 步, 超出的填充步填 0, 由 per-sample total 掩码排除。

    注意: 截止小时/目标天一律按 index (真实时间戳) 定位, 不用位置取模 ——
    特征表开头可能因特征 NaN 被截断 (如从 12:00 开始), 位置与小时不对齐。

    参数:
      index: 与 x_array 行对应的 DatetimeIndex
    返回:
      X      : (n, lookback_steps, C)
      Y      : (n, PREDICT_STEPS_MAX, 1)   目标值 (真实步数内为 y, 其余填 0)
      Xf     : (n, PREDICT_STEPS_MAX, C)   未来外生行 (真实步数内为 x, 其余填 0)
      Hs     : (n,) int   截止时刻的当天小时数 H = (实际小时)+1
      totals: (n,) int    真实 rollout 步数 = 缺口步 + 24
      e_idx  : (n,) int    截止时刻 e 在 x_array 中的下标 (用于还原时间戳/统计)
    """
    freq_minutes = int(processor.config["resample_freq"].replace("min", ""))
    points_per_day = (24 * 60) // freq_minutes
    lookback_steps = int(lookback_days * points_per_day) + int(lookback_extra_hours)
    day_steps = int(predict_days * points_per_day)       # 24
    predict_steps_max = PREDICT_STEPS_MAX                # 48

    total_len = len(x_array)
    hours = index.hour.to_numpy()
    main_cutoff_hour = (lookback_extra_hours - 1) % 24   # 主任务截止时刻 (16 点前 → 15:00)

    # 候选截止时刻 e (按真实小时筛选, 而非位置取模):
    #   多截止点 = 每个小时; 否则每天固定一个 (主任务, 按天步进)
    if use_multicutoff:
        e_candidates = np.arange(lookback_steps - 1, total_len - 1)
    else:
        mask = (hours == main_cutoff_hour) & (np.arange(total_len) >= lookback_steps - 1)
        cand = np.where(mask)[0]
        step_days = max(1, stride // points_per_day)
        e_candidates = cand[::step_days]

    # 截止小时过滤 (multicutoff_hours): 提速用子集, 主任务 H 始终保留
    hours_used = processor.config.get("multicutoff_hours")
    if hours_used is not None:
        hours_used = set(int(h) for h in hours_used)
        if main_cutoff_hour + 1 not in hours_used:
            hours_used.add(main_cutoff_hour + 1)         # 主任务 (H=16) 始终保留

    samples = []   # (e, H, total, pos)  pos = 目标天 0:00 的位置
    for e in e_candidates:
        H = int(hours[e]) + 1                            # 当天 0:00~e 共 H 小时
        if hours_used is not None and H not in hours_used:
            continue
        # 目标天起点: 截止时刻所在天的次日 0:00 (按时间戳定位)
        target_date = index[e].normalize() + pd.Timedelta(days=1)
        pos = index.searchsorted(target_date)
        if pos >= total_len or index[pos] != target_date:
            continue                      # 次日 0:00 缺失 (数据空洞)
        if pos + day_steps > total_len:
            continue                      # 目标天 24 个点超出数据末尾
        if index[pos + day_steps - 1].normalize() != target_date:
            continue                      # 目标天内含空洞 → 跳过该样本
        total = (pos - (e + 1)) + day_steps               # 缺口步 + 目标天
        if total > predict_steps_max:
            continue
        samples.append((e, H, total, pos))

    n = len(samples)
    if n == 0:
        empty_X = np.empty((0, lookback_steps, x_array.shape[1]), dtype=np.float32)
        empty_Y = np.empty((0, predict_steps_max, y_array.shape[1]), dtype=np.float32)
        empty_Xf = np.empty((0, predict_steps_max, x_array.shape[1]), dtype=np.float32)
        empty_meta = (np.empty(0, dtype=np.int64),) * 3
        return (empty_X, empty_Y, empty_Xf) + empty_meta

    # 填充步必须显式填 0, 不能用 np.empty: 未初始化内存可能是 NaN/inf 位模式,
    # 训练时 (pred - target)**2 * mask 里 NaN*0=NaN 会毒化整批 loss
    # (2026-09 修复: 空洞样本真实步数 < 48-H, 其填充步曾读到垃圾值)。
    X = np.zeros((n, lookback_steps, x_array.shape[1]), dtype=np.float32)
    Y = np.zeros((n, predict_steps_max, y_array.shape[1]), dtype=np.float32)
    Xf = np.zeros((n, predict_steps_max, x_array.shape[1]), dtype=np.float32)
    Hs = np.empty(n, dtype=np.int64)
    totals = np.empty(n, dtype=np.int64)
    e_idx = np.empty(n, dtype=np.int64)

    for idx, (e, H, total, pos) in enumerate(samples):
        X[idx] = x_array[e - lookback_steps + 1:e + 1]
        Xf[idx, :total] = x_array[e + 1:e + 1 + total]
        Y[idx, :total, 0] = y_array[e + 1:e + 1 + total, 0]   # 目标通道
        Hs[idx] = H
        totals[idx] = total
        e_idx[idx] = e
    return X, Y, Xf, Hs, totals, e_idx


def augment_peak_samples(X, Y, Xf, Hs, totals, e_idx, peak_threshold_ratio=0.7,
                         peak_augment_ratio=0.3):
    """峰值样本增强：对包含高峰时段的训练样本进行上采样。

    原理：
    - 检测每个样本的未来目标 Y 中是否存在峰值（最大值超过阈值）
    - 峰值样本会被复制并添加到训练集中，提升模型对高峰时段的学习权重
    - 避免模型倾向于预测"均值"而低估峰值

    参数：
    - X, Y, Xf, Hs, totals, e_idx: 原始训练序列 + 样本元信息 (随样本同步复制)
    - peak_threshold_ratio: 峰值判定阈值（相对于整个训练集的最大值），默认0.7
    - peak_augment_ratio: 增强比例（增强后的峰值样本数占总样本数的比例），默认0.3

    返回：
    - 增强后的序列 (与输入同结构)
    """
    if len(X) == 0:
        return X, Y, Xf, Hs, totals, e_idx

    # 计算每个样本的最大值 (真实步数内; 填充步为 0, 不影响峰值判定)
    sample_max = Y[:, :, 0].max(axis=1)  # 每个样本的最大流量值

    # 确定峰值阈值
    global_max = sample_max.max()
    threshold = peak_threshold_ratio * global_max

    # 找出峰值样本的索引
    peak_indices = np.where(sample_max > threshold)[0]

    if len(peak_indices) == 0:
        print(f"  ⚠ 未找到峰值样本（阈值={threshold:.2f}），跳过增强")
        return X, Y, Xf, Hs, totals, e_idx

    # 计算需要增强的样本数
    n_original = len(X)
    n_target_aug = int(n_original * peak_augment_ratio)
    aug_indices = np.random.choice(peak_indices, n_target_aug, replace=True)

    # 拼接原始数据和增强数据
    X = np.concatenate([X, X[aug_indices]], axis=0)
    Y = np.concatenate([Y, Y[aug_indices]], axis=0)
    Xf = np.concatenate([Xf, Xf[aug_indices]], axis=0)
    Hs = np.concatenate([Hs, Hs[aug_indices]], axis=0)
    totals = np.concatenate([totals, totals[aug_indices]], axis=0)
    e_idx = np.concatenate([e_idx, e_idx[aug_indices]], axis=0)

    print(f"  ✅ 峰值样本增强完成:")
    print(f"     原始样本数: {n_original}")
    print(f"     峰值样本数: {len(peak_indices)} (阈值={threshold:.2f})")
    print(f"     新增样本数: {len(aug_indices)}")
    print(f"     增强后总样本数: {len(X)}")
    print(f"     峰值占比: {len(peak_indices)/n_original*100:.1f}% -> {min(1.0, (len(peak_indices)+len(aug_indices))/len(X))*100:.1f}%")

    return X, Y, Xf, Hs, totals, e_idx


def build_holiday_map(hol_series):
    """把按小时的 is_holiday 序列聚合成 {日期(midnight): 0/1} 字典。"""
    if hol_series is None:
        return {}
    by_day = hol_series.groupby(hol_series.index.normalize()).first()
    return by_day.to_dict()


def augment_holiday_samples(X, Y, Xf, Hs, totals, e_idx, cutoff_times, holiday_map,
                            factor=4.0, max_ratio=0.25):
    """节假日目标日上采样 (改动 1c): 目标日为法定节假日的样本复制增强。

    节假日天数少 (一年 ~10 天), 目标日样本占比低, 模型容易把节假日模式
    平均掉; 复制这些样本提高学习权重。目标日 = 截止时刻所在天 + 1 天。

    参数:
    - cutoff_times: 每个样本截止时刻的时间戳 (长度 = n, 与 e_idx 对应)
    - holiday_map: build_holiday_map 的返回值
    - factor: 节假日样本复制后的总倍数 (默认 4)
    - max_ratio: 节假日样本占总样本数的上限比例 (默认 0.25)
    """
    n = len(X)
    if n == 0 or cutoff_times is None or len(cutoff_times) != n or not holiday_map:
        return X, Y, Xf, Hs, totals, e_idx

    target_dates = [pd.Timestamp(ct).normalize() + pd.Timedelta(days=1)
                    for ct in cutoff_times]
    is_hol = np.array([holiday_map.get(d, 0.0) > 0.5 for d in target_dates], dtype=bool)
    n_hol = int(is_hol.sum())
    if n_hol == 0:
        print("  ⚠ 训练集无节假日目标日样本, 跳过节假日上采样")
        return X, Y, Xf, Hs, totals, e_idx

    n_cap = int(n * max_ratio)
    n_add = min(int(n_hol * (factor - 1)), max(0, n_cap - n_hol))
    if n_add <= 0:
        return X, Y, Xf, Hs, totals, e_idx

    add_idx = np.random.choice(np.where(is_hol)[0], n_add, replace=True)
    X = np.concatenate([X, X[add_idx]], axis=0)
    Y = np.concatenate([Y, Y[add_idx]], axis=0)
    Xf = np.concatenate([Xf, Xf[add_idx]], axis=0)
    Hs = np.concatenate([Hs, Hs[add_idx]], axis=0)
    totals = np.concatenate([totals, totals[add_idx]], axis=0)
    e_idx = np.concatenate([e_idx, e_idx[add_idx]], axis=0)

    print(f"  ✅ 节假日目标日上采样: 原始 {n_hol} 个 → 新增 {n_add} 个 "
          f"(总样本 {n} → {len(X)}, 节假日占比 "
          f"{n_hol/n*100:.1f}% -> {min(1.0, (n_hol+n_add)/len(X))*100:.1f}%)")
    return X, Y, Xf, Hs, totals, e_idx


# ==================== 单次实验运行 ====================

def run_experiment(cfg, x_train_all, y_train_all, x_test_all, y_test_all, processor,
                   device, train_index=None, test_index=None,
                   train_holiday=None, test_holiday=None):
    processor.config = cfg
    lookback = cfg["lookback_days"]
    lookback_extra = cfg.get("lookback_extra_hours", 0)
    predict = cfg["predict_days"]
    label = cfg["label"]
    detach_feedback = cfg.get("detach_feedback", True)
    use_multicutoff = cfg.get("use_multicutoff", True)
    predict_steps_max = PREDICT_STEPS_MAX
    day_steps = 24

    result_dir = os.path.join(cfg["base_result_dir"], label)
    ensure_dir(result_dir)

    freq_minutes = int(cfg["resample_freq"].replace("min", ""))
    points_per_day = (24 * 60) // freq_minutes
    lookback_steps = int(lookback * points_per_day) + int(lookback_extra)

    print(f"\n{'='*80}")
    print(f" 实验(自回归单步, 每天16点预测次日全天): {label}")
    print(f"   model_type={cfg.get('model_type', 'transformer')}"
          f"  |  lookback={lookback}d+{lookback_extra}h={lookback_steps}步"
          f"  |  目标=次日全天{day_steps}步  |  freq={cfg['resample_freq']}"
          f"  |  use_multicutoff={use_multicutoff}  |  test_days={cfg['test_days']}")
    print(f"   d_model={cfg['d_model']}  |  nhead={cfg['nhead']}"
          f"  |  layers={cfg['num_layers']}  |  lr={cfg['learning_rate']}"
          f"  |  detach_feedback={detach_feedback}")
    print(f" 清洗改进: spike_abs_floor={cfg.get('spike_abs_floor')}"
          f"  |  smooth_median_window={cfg.get('smooth_median_window')}")
    print(f" 结果目录: {result_dir}")
    print(f"{'='*80}")

    # 构建序列 (返回样本元信息: H / 真实步数 / 截止时刻下标)
    X_train, Y_train, Xf_train, H_train, tot_train, e_train = make_sequences_ar(
        processor, train_index, x_train_all, y_train_all, lookback, predict,
        lookback_extra, use_multicutoff, cfg.get("stride", 24))
    X_test, Y_test, Xf_test, H_test, tot_test, e_test = make_sequences_ar(
        processor, test_index, x_test_all, y_test_all, lookback, predict,
        lookback_extra, use_multicutoff, cfg.get("stride", 24))

    # 峰值样本增强（仅对训练集）
    peak_augment_ratio = cfg.get("peak_augment_ratio", 0.3)
    peak_threshold_ratio = cfg.get("peak_threshold_ratio", 0.7)
    if peak_augment_ratio > 0 and len(X_train) > 0:
        X_train, Y_train, Xf_train, H_train, tot_train, e_train = augment_peak_samples(
            X_train, Y_train, Xf_train, H_train, tot_train, e_train,
            peak_threshold_ratio=peak_threshold_ratio,
            peak_augment_ratio=peak_augment_ratio)

    # 节假日目标日上采样 (改动 1c, 仅训练集)
    if train_index is not None and len(X_train) > 0:
        cutoff_times_train = train_index[e_train]
        holiday_map = build_holiday_map(train_holiday)
        X_train, Y_train, Xf_train, H_train, tot_train, e_train = augment_holiday_samples(
            X_train, Y_train, Xf_train, H_train, tot_train, e_train,
            cutoff_times_train, holiday_map,
            factor=cfg.get("holiday_augment_factor", 4.0),
            max_ratio=cfg.get("holiday_augment_max_ratio", 0.25))

    # 截止时刻时间戳 (统计/画图用)
    if test_index is not None and len(e_test) == len(X_test):
        test_starts = test_index[e_test]
    else:
        test_starts = None

    # 目标通道在 feature_cols 中的下标 (Total_Flow); 回灌时覆盖该通道
    target_feat_idx = processor.feature_cols.index(processor.target_cols[0])

    print(f"  窗口={lookback_steps}步 (截止时刻=当天15:00/16点前, H=16 即原任务),"
          f" 目标=次日全天 {day_steps}步")
    print(f"  rollout: 每样本真实步数 = 缺口 (24-H) + 目标天 {day_steps},"
          f" 统一 padding 到 {predict_steps_max} 步 (掩码排除填充步)")
    print(f"  target_feat_idx={target_feat_idx} ({processor.target_cols[0]})")
    print(f"  X_train={X_train.shape}, Y_train={Y_train.shape}, Xf_train={Xf_train.shape}")
    print(f"  X_test={X_test.shape}, Y_test={Y_test.shape}, Xf_test={Xf_test.shape}")
    print(f"  H_train 分布: {np.bincount(H_train)[1:] if len(H_train) else 0} (H=1..24)")

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"  ⚠ 样本数为0，跳过此配置")
        return None

    # 训练 loader 按 H 分桶 (桶内 H 相同 → rollout 只滚 48-H 步, 省填充开销);
    # 测试 loader 保持原始顺序 (与 test_starts 逐样本对应)
    train_loader = DataLoader(ARSeqDataset(X_train, Y_train, Xf_train, H_train, tot_train),
                              batch_sampler=CutoffBucketSampler(H_train, cfg["batch_size"], shuffle=True),
                              pin_memory=True, num_workers=2)
    test_loader = DataLoader(ARSeqDataset(X_test, Y_test, Xf_test, H_test, tot_test),
                             batch_size=cfg["batch_size"], shuffle=False,
                             pin_memory=True, num_workers=2)

    # ── 模型工厂: horizon=1 (单步头), 自回归 rollout 负责滚出 horizon ──
    model_type = cfg.get("model_type", "transformer")
    common_model_kwargs = dict(
        input_dim=X_train.shape[2],
        output_dim=1,
        horizon=1,                       # ← 单步头: 只预测下一个点
        input_len=lookback_steps,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["transformer_dropout"],
    )
    if model_type == "itransformer":
        assert cfg.get("target_transform") is None, \
            "iTransformer 要求 target_transform=None (RevIN 反归一化与 log1p 目标域不兼容)"
        model = iTransformer(
            **common_model_kwargs,
            target_idx=target_feat_idx,
        ).to(device)
    else:
        model = TimeSeriesTransformer(**common_model_kwargs).to(device)

    # torch.compile: JIT编译加速 (需要 PyTorch >= 2.1)
    if hasattr(torch, "compile") and tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 1):
        try:
            model = torch.compile(model)
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile failed ({e}), using eager mode")
    else:
        print(f"  torch.compile skipped (PyTorch {torch.__version__} < 2.1)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg["T_0"], T_mult=cfg["T_mult"])
    early_stopper = EarlyStopping(cfg["patience"], cfg["min_delta"])

    best_state = None
    best_test_mape = float("inf")
    history = []
    test_metrics = {}  # 防止首次跳过评估时未定义

    # AMP: 混合精度训练加速
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    eval_interval = cfg.get("eval_interval", 1)

    print(f"\n  Training(自回归): {label}")
    print(f"  AMP={'ON' if use_amp else 'OFF'}, eval_interval={eval_interval}")

    epoch_pbar = tqdm(range(1, cfg["epochs"] + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss_sum = 0.0

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}",
                          unit="batch", leave=False)
        for batch_x, batch_y, batch_xf, batch_H, batch_totals in batch_pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_xf = batch_xf.to(device, non_blocking=True)
            batch_H = batch_H.to(device, non_blocking=True)
            batch_totals = batch_totals.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # AMP: forward 用 float16, backward 用 float16 + 缩放
            with autocast(enabled=use_amp):
                # 桶内 H 相同 → 本 batch 只需滚 48-H 步 (无填充前向浪费)
                batch_steps = int(predict_steps_max - batch_H[0].item())
                pred = autoregressive_rollout(model, batch_x, batch_xf, batch_steps,
                                             target_feat_idx, detach_feedback)
                loss = masked_flow_loss(pred, batch_y, batch_H, batch_totals,
                                        predict_steps_max, day_steps)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(batch_x)
            batch_pbar.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = train_loss_sum / len(train_loader.dataset)

        # 按 eval_interval 频率评估测试集
        do_eval = (epoch % eval_interval == 0) or (epoch == cfg["epochs"])
        if do_eval:
            test_metrics = evaluate(model, test_loader, device, processor, predict_steps_max,
                                    target_feat_idx, detach_feedback, day_steps)

        test_loss = test_metrics.get("loss", float("nan"))
        current_mape = test_metrics.get("flow_mape", float("inf"))

        if do_eval and current_mape < best_test_mape:
            best_test_mape = current_mape
            # torch.compile 后需要 .module 获取原始 state_dict
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

    # 保存 scaler / 特征列 / 配置 (供推理脚本加载; 自回归推理需配套 rollout)
    scaler_path = os.path.join(result_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "config": cfg,
            "feature_scaler": processor.feature_scaler,
            "target_scaler": processor.target_scaler,
            "feature_cols": processor.feature_cols,
            "target_cols": processor.target_cols,
            # 自回归推理需要: 单步头 (horizon=1) + 与训练一致的 rollout
            "autoregressive": True,
            "target_feat_idx": target_feat_idx,
            "detach_feedback": detach_feedback,
            "lookback_steps": lookback_steps,
            "predict_steps_max": predict_steps_max,   # 推理时 16 点任务滚 32 步
            "use_multicutoff": use_multicutoff,
        }, f)
    print(f"  推理用 scaler/特征配置已保存: {scaler_path}")
    print(f"  ⚠ 注意: 本模型为单步自回归, 推理须用配套的 autoregressive_rollout "
          f"(见 train_transformer_nextday_16h.py), 不能用原直接多步推理脚本; "
          f"16 点任务需滚过缺口 (16:00~23:00) 再取次日 24 小时")

    # 最终评估 (指标均为"目标天"部分; 另有 H=16 原任务子集)
    train_metrics = evaluate(model, train_loader, device, processor, predict_steps_max,
                             target_feat_idx, detach_feedback, day_steps)
    test_metrics = evaluate(model, test_loader, device, processor, predict_steps_max,
                            target_feat_idx, detach_feedback, day_steps)

    def fmt_line(name, m):
        h16 = f", H16(16点任务): MAE={m['h16_mae']:.2f}, RMSE={m['h16_rmse']:.2f}, MAPE={m['h16_mape']:.2f}%" \
              if m.get("n_h16", 0) > 0 else ""
        return (f"  {name}: Loss={m['loss']:.6f}, MAE={m['flow_mae']:.2f}, "
                f"RMSE={m['flow_rmse']:.2f}, MAPE={m['flow_mape']:.2f}%"
                f" (n={m.get('n_samples', 0)}){h16}")

    print(f"\n  最终结果 (指标=目标天 D+1 全天 24h):")
    print(fmt_line("Train", train_metrics))
    print(fmt_line("Test ", test_metrics))
    mape_floor = cfg.get("mape_floor_ratio", 0.05)
    print(f"  (MAPE 已过滤 |true| < {mape_floor:.0%}*max 的近零流量点: "
          f"Train 保留 {train_metrics['mape_n_used']}/{train_metrics['mape_n_total']}, "
          f"Test 保留 {test_metrics['mape_n_used']}/{test_metrics['mape_n_total']})")

    with open(os.path.join(result_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"[自回归单步, 每天16点预测次日全天] use_multicutoff={use_multicutoff}, "
                f"lookback={lookback}d+{lookback_extra}h, detach_feedback={detach_feedback}, "
                f"predict_steps_max={predict_steps_max}\n")
        f.write(f"清洗: spike_abs_floor={cfg.get('spike_abs_floor')}, "
                f"smooth_median_window={cfg.get('smooth_median_window')}\n")
        f.write("指标=目标天 D+1 全天 24h (缺口步参与训练损失, 不计入指标):\n")
        for name, m in [("Train", train_metrics), ("Test", test_metrics)]:
            f.write(f"{name}: Loss={m['loss']:.6f}, MAE={m['flow_mae']:.2f}, "
                    f"RMSE={m['flow_rmse']:.2f}, MAPE={m['flow_mape']:.2f}% (n={m.get('n_samples', 0)})\n")
            if m.get("n_h16", 0) > 0:
                f.write(f"{name} H16(16点任务子集): MAE={m['h16_mae']:.2f}, "
                        f"RMSE={m['h16_rmse']:.2f}, MAPE={m['h16_mape']:.2f}% (n={m['n_h16']})\n")
        f.write(f"MAPE 过滤: 排除 |true| < {mape_floor:.0%} * max|true| 的点 "
                f"(Train 保留 {train_metrics['mape_n_used']}/{train_metrics['mape_n_total']} 点, "
                f"Test 保留 {test_metrics['mape_n_used']}/{test_metrics['mape_n_total']} 点)\n")

    # 画图 (仅测试集, 目标天部分)
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
    plt.title(f"Training Curve (Autoregressive, 16h→NextDay, multicutoff={use_multicutoff}) — {label}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "label": label,
        "resample_freq": cfg["resample_freq"],
        "test_days": cfg["test_days"],
        "d_model": cfg["d_model"],
        "nhead": cfg["nhead"],
        "num_layers": cfg["num_layers"],
        "learning_rate": cfg["learning_rate"],
        "lookback_days": lookback,
        "lookback_extra_hours": lookback_extra,
        "lookback_steps": lookback_steps,
        "predict_days": predict,
        "predict_steps_max": predict_steps_max,
        "use_multicutoff": use_multicutoff,
        "detach_feedback": detach_feedback,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "best_epoch": int(history_df.loc[history_df["flow_mape"].idxmin(), "epoch"]),
        "train_loss": train_metrics["loss"],
        "test_loss": test_metrics["loss"],
        "train_mae": train_metrics["flow_mae"],
        "test_mae": test_metrics["flow_mae"],
        "train_rmse": train_metrics["flow_rmse"],
        "test_rmse": test_metrics["flow_rmse"],
        "train_mape": train_metrics["flow_mape"],
        "test_mape": test_metrics["flow_mape"],
        "test_h16_mape": test_metrics.get("h16_mape", float("nan")),
        "test_h16_mae": test_metrics.get("h16_mae", float("nan")),
    }


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="训练每天16点预测次日全天的单步自回归 Transformer / iTransformer 流量预测模型")
    parser.add_argument("--model", choices=["transformer", "itransformer"], default=None,
                        help="模型类型 (默认 None = 用 BASE_CONFIG['model_type'])")
    parser.add_argument("--label", default=None,
                        help="结果子目录名 (默认 None = 用 BASE_CONFIG['label'])")
    parser.add_argument("--lookback", type=float, default=None,
                        help="前 N 整天回看 (默认 None = 用 BASE_CONFIG['lookback_days']=7)")
    parser.add_argument("--lookback-extra-hours", type=int, default=None,
                        help="当天 16 点前额外追加的小时数 (默认 None = 用 BASE_CONFIG['lookback_extra_hours']=16)")
    parser.add_argument("--stride", type=int, default=None,
                        help="非多截止点模式的采样步长, 24 = 每天一个样本 (默认 None = 用 BASE_CONFIG['stride'])")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数 (多截止点后每轮较长, 建议 15~20; 默认 None = 用 BASE_CONFIG)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="batch size (4GB 显存建议 16, 越小越省显存; 默认 None = 用 BASE_CONFIG)")
    # --use-multicutoff / --no-use-multicutoff: 多截止点任务训练
    #   默认 True: 每天 24 个截止时刻都训练 (样本量 ×24, H=16 即原任务)
    #   --no-use-multicutoff: 只保留 16 点任务样本
    parser.add_argument("--multicutoff-hours", default=None,
                        help="多截止点使用的小时列表, 逗号分隔, 如 '16,8,20,4,12,24'"
                             " (16 点主任务始终自动保留; 默认 None = 全部 24 小时;"
                             " 子集可大幅提速)")
    parser.add_argument("--use-multicutoff", dest="use_multicutoff",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="多截止点任务训练 (默认 True; --no-use-multicutoff 只保留 16 点任务)")
    # --detach-feedback / --no-detach-feedback: 切换回灌预测值是否 detach
    parser.add_argument("--detach-feedback", dest="detach_feedback",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="回灌预测是否 detach (默认 True; --no-detach-feedback=完整BPTT)")
    args = parser.parse_args()

    config = dict(BASE_CONFIG)
    if args.model is not None:
        config["model_type"] = args.model
    if args.label is not None:
        config["label"] = args.label
    if args.lookback is not None:
        config["lookback_days"] = args.lookback
    if args.lookback_extra_hours is not None:
        config["lookback_extra_hours"] = args.lookback_extra_hours
    if args.stride is not None:
        config["stride"] = args.stride
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.multicutoff_hours is not None:
        config["multicutoff_hours"] = [int(x) for x in args.multicutoff_hours.split(",")]
    if args.use_multicutoff is not None:
        config["use_multicutoff"] = args.use_multicutoff
    if args.detach_feedback is not None:
        config["detach_feedback"] = args.detach_feedback
    set_seed(config["seed"])

    device = torch.device(config["device"])
    print(f"Device: {device}")
    print(f"结果目录: {os.path.join(config['base_result_dir'], config['label'])}")
    print(f"模式: 单步自回归 (horizon=1 head + 滚动 rollout, detach_feedback={config.get('detach_feedback', True)})")
    print(f"输入: 前{config['lookback_days']}天全天 + 当天16点前{config.get('lookback_extra_hours', 16)}小时"
          f" = {int(config['lookback_days'] * 24) + int(config.get('lookback_extra_hours', 16))}小时"
          f" → 预测第二天全天 24 小时"
          f" (多截止点={config.get('use_multicutoff', True)})")
    print(f"清洗改进: spike_abs_floor={config.get('spike_abs_floor')}, "
          f"smooth_median_window={config.get('smooth_median_window')}")

    # ============ 第一步: 数据加载 & 清洗 ============
    print("\n" + "=" * 80)
    print(f" [Phase 1] 数据加载 & 清洗 (resample_freq={config['resample_freq']}, 无特征工程)")
    print("=" * 80)

    processor = DataProcessor(config)
    print(" 正在加载并处理数据...")
    df_all_feat = processor.build_feature_table()
    print(f" 全量特征表: {df_all_feat.shape}")
    print(f" 时间范围: {df_all_feat.index.min()} ~ {df_all_feat.index.max()}")

    # ============ 第二步: 按时间划分训练/测试集 + 归一化 ============
    print("\n" + "=" * 80)
    print(" [Phase 2] 按时间划分训练/测试集")
    print("=" * 80)

    df_train_feat, df_test_feat = processor.split_by_time(df_all_feat)
    print(f" 训练集: {df_train_feat.shape} | {df_train_feat.index.min()} ~ {df_train_feat.index.max()}")
    print(f" 测试集: {df_test_feat.shape} | {df_test_feat.index.min()} ~ {df_test_feat.index.max()}")

    processor.fit_scalers(df_train_feat)
    x_train_all, y_train_all = processor.transform_df(df_train_feat)
    x_test_all, y_test_all = processor.transform_df(df_test_feat)

    # ============ 第三步: 训练 & 评估 ============
    result = run_experiment(config, x_train_all, y_train_all,
                            x_test_all, y_test_all,
                            processor, device,
                            train_index=df_train_feat.index, test_index=df_test_feat.index,
                            train_holiday=df_train_feat["is_holiday"],
                            test_holiday=df_test_feat["is_holiday"])
    if result is not None:
        print(f"\n 训练完成! 结果保存在: {os.path.join(config['base_result_dir'], config['label'])}")
    else:
        print(f"\n ⚠ 样本数为 0, 未产生结果 (请检查 test_days 是否 ≥ 回看+预测天数, "
              f"即 ≥ {int(config['lookback_days']) + 1 + int(config['predict_days'])} 天)")


if __name__ == "__main__":
    main()

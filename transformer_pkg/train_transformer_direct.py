import os
import sys
import argparse
import math
import pickle
import random
import time
from copy import deepcopy
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── 共享模块 (与 train_transformer_autoregressive.py 同目录, 不改动原文件) ──
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor, SeqDataset

HERE = os.path.dirname(os.path.abspath(__file__))
# ============================================================================
# train_transformer_direct.py — 直接多步 (非自回归) 版训练脚本 (新文件)
#
# 与 train_transformer_autoregressive.py 的区别:
#   自回归版              : 模型 head 只输出 1 步 (horizon=1), 把上一轮预测回灌
#                          为输入窗口最新行, 滚动 rollout predict_steps 次; 训练
#                          时每个 batch 做 predict_steps 次 forward。
#   本文件 (直接多步)    : 模型 head 直接一次输出整个 horizon (horizon=predict_steps),
#                          一次 forward 出 24 步, 无自回归 / 无回灌 / 无 rollout;
#                          训练并行, 推理一次出 24 步。
#
# 数据口径与自回归版完全一致 (清洗 / 日历特征 / 数据驱动特征 / scaler / 划分),
# 仅替换"前向输出方式":
#   - 不需要未来外生特征行 Xf (直接多步不滚动, 无需补齐外生通道)
#   - 用 DataProcessor.make_sequences (X, Y 两元组) 而非 make_sequences_ar (X, Y, Xf)
#   - 模型工厂里 horizon=predict_steps (整段 horizon 一次输出)
#
# 其余 (评估管线 / 画图 / 按起点时刻统计 / 峰值增强) 与自回归版完全一致。
# ============================================================================

BASE_CONFIG = {
    "file_path": r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv",
    "encoding": "utf-8-sig",
    "resample_freq": "60min",
    "stride": 1,
    "hampel_cols": ["Total_Flow"],   # 仅清洗流量 (压力/泵量测不再读取)
    "hampel_window": 48,
    "spike_ratio_within": 1.2,   # 日内 t±1h 突变阈值 (曲线平滑, 阈值可低)
    "spike_ratio_cross": 1.3,    # 跨日 t±24h 突变阈值 (日间差异大, 需更高阈值)

    "lookback_days": 7,
    "predict_days": 1.0,
    "label": f"junshan_L1D_P24H_1h_transformer_direct_{time.strftime('%Y%m%d_%H%M%S')}",

    "test_days": 90,

    "mape_floor_ratio": 0.1,
    "target_transform": None,

    # ── Transformer 架构超参 (head 直接多步: horizon=predict_steps 在工厂里给定) ──
    "d_model": 32,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 256,
    "transformer_dropout": 0.2,

    "model_type": "itransformer",  # transformer | itransformer

    # 峰值样本增强配置
    "peak_augment_ratio": 0.0,      # 增强比例：增强后的峰值样本占总样本数的比例
    "peak_threshold_ratio": 0.7,    # 峰值判定阈值：相对于训练集最大值的比例

    # 训练超参 (直接多步一次 forward 出 horizon 步, 比 rollout 快 horizon 倍;
    # batch 可放大省显存无虞)
    "batch_size": 64,
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


# ==================== 评估 ====================

def evaluate(model, loader, device, processor, predict_steps):
    """直接多步评估: 一次 forward 出 predict_steps 步, 无 rollout。"""
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Evaluating",
                                     unit="batch", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x, target_len=predict_steps)   # (B, H, 1) 一次出全段
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
    print(f"    (MAPE 过滤阈值 {thr:.2f}, 排除 |true| < 阈值 的点)")
    print("    起点时刻  样本数      MAE      RMSE     MAPE%")
    for r in rows:
        print(f"    {r['start_time']}  {r['n_samples']:<6}  "
              f"{r['mae']:<8.2f}  {r['rmse']:<8.2f}  {r['mape']:<8.2f}")


# ==================== 峰值样本增强 (与自回归版一致, 去掉 Xf) ====================

def augment_peak_samples(X, Y, peak_threshold_ratio=0.7, peak_augment_ratio=0.3):
    """峰值样本增强：对包含高峰时段的训练样本进行上采样。

    原理：
    - 检测每个样本的未来目标 Y 中是否存在峰值（最大值超过阈值）
    - 峰值样本会被复制并添加到训练集中，提升模型对高峰时段的学习权重
    - 避免模型倾向于预测"均值"而低估峰值

    参数：
    - X, Y: 原始训练序列 (直接多步无需 Xf)
    - peak_threshold_ratio: 峰值判定阈值（相对于整个训练集的最大值），默认0.7
    - peak_augment_ratio: 增强比例（增强后的峰值样本数占总样本数的比例），默认0.3

    返回：
    - X_aug, Y_aug: 增强后的序列
    """
    if len(X) == 0:
        return X, Y

    sample_max = Y[:, :, 0].max(axis=1)  # 每个样本的最大流量值

    global_max = sample_max.max()
    threshold = peak_threshold_ratio * global_max

    peak_indices = np.where(sample_max > threshold)[0]

    if len(peak_indices) == 0:
        print(f"  ⚠ 未找到峰值样本（阈值={threshold:.2f}），跳过增强")
        return X, Y

    n_original = len(X)
    n_target_aug = int(n_original * peak_augment_ratio)

    aug_indices = np.random.choice(peak_indices, n_target_aug, replace=True)

    X_aug = np.concatenate([X, X[aug_indices]], axis=0)
    Y_aug = np.concatenate([Y, Y[aug_indices]], axis=0)

    print(f"  ✅ 峰值样本增强完成:")
    print(f"     原始样本数: {n_original}")
    print(f"     峰值样本数: {len(peak_indices)} (阈值={threshold:.2f})")
    print(f"     新增样本数: {len(aug_indices)}")
    print(f"     增强后总样本数: {len(X_aug)}")
    print(f"     峰值占比: {len(peak_indices)/n_original*100:.1f}% -> {min(1.0, (len(peak_indices)+len(aug_indices))/len(X_aug))*100:.1f}%")

    return X_aug, Y_aug


# ==================== 单次实验运行 ====================

def run_experiment(cfg, x_train_all, y_train_all, x_test_all, y_test_all, processor, device, test_index=None):
    processor.config = cfg
    lookback = cfg["lookback_days"]
    predict = cfg["predict_days"]
    label = cfg["label"]

    result_dir = os.path.join(cfg["base_result_dir"], label)
    ensure_dir(result_dir)

    print(f"\n{'='*80}")
    print(f" 实验(直接多步): {label}  |  model_type={cfg.get('model_type', 'transformer')}"
          f"  |  lookback={lookback}d  |  predict={predict}d"
          f"  |  freq={cfg['resample_freq']}  |  test_days={cfg['test_days']}"
          f"  |  d_model={cfg['d_model']}  |  nhead={cfg['nhead']}"
          f"  |  layers={cfg['num_layers']}  |  lr={cfg['learning_rate']}")
    print(f" 结果目录: {result_dir}")
    print(f"{'='*80}")

    # 构建序列 (直接多步: 仅 X, Y 两元组, 无需未来外生特征行)
    X_train, Y_train = processor.make_sequences(x_train_all, y_train_all, lookback, predict)
    X_test, Y_test = processor.make_sequences(x_test_all, y_test_all, lookback, predict)

    # 峰值样本增强（仅对训练集）
    peak_augment_ratio = cfg.get("peak_augment_ratio", 0.3)
    peak_threshold_ratio = cfg.get("peak_threshold_ratio", 0.7)
    if peak_augment_ratio > 0 and len(X_train) > 0:
        X_train, Y_train = augment_peak_samples(
            X_train, Y_train,
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

    print(f"  {cfg['resample_freq']}频率: lookback={lookback_steps}步, predict={predict_steps}步"
          f" (直接多步, 一次 forward 出 {predict_steps} 步)")
    print(f"  X_train={X_train.shape}, Y_train={Y_train.shape}")
    print(f"  X_test={X_test.shape}, Y_test={Y_test.shape}")

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"  ⚠ 样本数为0，跳过此配置")
        return None

    train_loader = DataLoader(SeqDataset(X_train, Y_train),
                              batch_size=cfg["batch_size"], shuffle=True,
                              pin_memory=True, num_workers=2)
    test_loader = DataLoader(SeqDataset(X_test, Y_test),
                             batch_size=cfg["batch_size"], shuffle=False,
                             pin_memory=True, num_workers=2)

    # 目标通道在 feature_cols 中的下标 (iTransformer 需要)
    target_feat_idx = processor.feature_cols.index(processor.target_cols[0])

    # ── 模型工厂: horizon=predict_steps (直接多步头, 一次出整段 horizon) ──
    model_type = cfg.get("model_type", "transformer")
    common_model_kwargs = dict(
        input_dim=X_train.shape[2],
        output_dim=1,
        horizon=predict_steps,           # ← 直接多步头: 一次输出整个 horizon
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

    print(f"\n  Training(直接多步): {label}")
    print(f"  AMP={'ON' if use_amp else 'OFF'}, eval_interval={eval_interval}")

    epoch_pbar = tqdm(range(1, cfg["epochs"] + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        train_loss_sum = 0.0

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}",
                          unit="batch", leave=False)
        for batch_x, batch_y in batch_pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # AMP: forward 用 float16, backward 用 float16 + 缩放
            with autocast(enabled=use_amp):
                pred = model(batch_x, target_len=predict_steps)   # 一次出全段
                loss = weighted_mse_loss(pred, batch_y)

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
            test_metrics = evaluate(model, test_loader, device, processor, predict_steps)

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

    # 保存 scaler / 特征列 / 配置 (供推理脚本加载; 直接多步推理)
    scaler_path = os.path.join(result_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "config": cfg,
            "feature_scaler": processor.feature_scaler,
            "target_scaler": processor.target_scaler,
            "feature_cols": processor.feature_cols,
            "target_cols": processor.target_cols,
            # 直接多步推理: horizon=predict_steps 一次输出, 无需 rollout
            "autoregressive": False,
            "target_feat_idx": target_feat_idx,
        }, f)
    print(f"  推理用 scaler/特征配置已保存: {scaler_path}")
    print(f"  ⚠ 注意: 本模型为直接多步, 推理用 model(x, target_len=predict_steps) "
          f"一次出整段, 不能用自回归 rollout 推理脚本")

    # 最终评估
    train_metrics = evaluate(model, train_loader, device, processor, predict_steps)
    test_metrics = evaluate(model, test_loader, device, processor, predict_steps)

    print(f"\n  最终结果:")
    print(f"  Train: Loss={train_metrics['loss']:.6f}, MAE={train_metrics['flow_mae']:.2f}, "
          f"RMSE={train_metrics['flow_rmse']:.2f}, MAPE={train_metrics['flow_mape']:.2f}%")
    print(f"  Test : Loss={test_metrics['loss']:.6f}, MAE={test_metrics['flow_mae']:.2f}, "
          f"RMSE={test_metrics['flow_rmse']:.2f}, MAPE={test_metrics['flow_mape']:.2f}%")
    mape_floor = cfg.get("mape_floor_ratio", 0.05)
    print(f"  (MAPE 已过滤 |true| < {mape_floor:.0%}*max 的近零流量点: "
          f"Train 保留 {train_metrics['mape_n_used']}/{train_metrics['mape_n_total']}, "
          f"Test 保留 {test_metrics['mape_n_used']}/{test_metrics['mape_n_total']})")

    with open(os.path.join(result_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"[直接多步版] predict_steps={predict_steps}\n")
        for name, m in [("Train", train_metrics), ("Test", test_metrics)]:
            f.write(f"{name}: Loss={m['loss']:.6f}, MAE={m['flow_mae']:.2f}, "
                    f"RMSE={m['flow_rmse']:.2f}, MAPE={m['flow_mape']:.2f}%\n")
        f.write(f"MAPE 过滤: 排除 |true| < {mape_floor:.0%} * max|true| 的点 "
                f"(Train 保留 {train_metrics['mape_n_used']}/{train_metrics['mape_n_total']} 点, "
                f"Test 保留 {test_metrics['mape_n_used']}/{test_metrics['mape_n_total']} 点)\n")

    # 画图 (仅测试集)
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
    plt.title(f"Training Curve (Direct Multi-step) — {label}")
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
        "predict_days": predict,
        "predict_steps": predict_steps,
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
    }


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="训练直接多步 (非自回归) Transformer / iTransformer 流量预测模型")
    parser.add_argument("--model", choices=["transformer", "itransformer"], default=None,
                        help="模型类型 (默认 None = 用 BASE_CONFIG['model_type'])")
    parser.add_argument("--label", default=None,
                        help="结果子目录名 (默认 None = 用 BASE_CONFIG['label'])")
    parser.add_argument("--lookback", type=float, default=None,
                        help="回看天数 (默认 None = 用 BASE_CONFIG['lookback_days'])")
    args = parser.parse_args()

    config = dict(BASE_CONFIG)
    if args.model is not None:
        config["model_type"] = args.model
    if args.label is not None:
        config["label"] = args.label
    if args.lookback is not None:
        config["lookback_days"] = args.lookback
    set_seed(config["seed"])

    device = torch.device(config["device"])
    print(f"Device: {device}")
    print(f"结果目录: {os.path.join(config['base_result_dir'], config['label'])}")
    print(f"模式: 直接多步 (horizon=predict_steps head, 一次 forward 出全段, 无 rollout)")

    # ============ 第一步: 数据加载 & 清洗 (特征工程已删除) ============
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
                            processor, device, test_index=df_test_feat.index)
    if result is not None:
        print(f"\n 训练完成! 结果保存在: {os.path.join(config['base_result_dir'], config['label'])}")
    else:
        print(f"\n ⚠ 样本数为 0, 未产生结果 (请检查 test_days 是否 ≥ 回看+预测天数)")


if __name__ == "__main__":
    main()

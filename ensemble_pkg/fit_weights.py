# -*- coding: utf-8 -*-
"""融合权重训练 — 用样本外回测训练 α, 并给出无偏的效果评估。

    融合预测 = α · Transformer + (1-α) · TimesFM

── 回测流程 ──
对每一天 day ∈ fit_days + valid_days:
  1. 截取 day 之前的数据 (Transformer 需 184 步回看, TimesFM 需 168h context)
  2. Transformer.predict_transformer(truncated_df, target_date=day)
     → 24 点小时预测 (自回归 rollout)
  3. TimesFM: 同一条清洗管线的 Total_Flow 小时序列, 取最后 168h → forecast → 24 点
  4. 真值: DataProcessor 清洗后的 Total_Flow 当天 24 小时
  5. 记录 [y_true, pred_transformer, pred_timesfm]

── 权重搜索 ──
fit 窗口 (前半): 网格搜索 α, 最小化 MAE
valid 窗口 (后半): 独立验证, 完全不参与选择

用法:
  cd D:\Junshan_Project\ensemble_pkg
  python fit_weights.py
  python fit_weights.py --fit_days 12 --valid_days 12 --metric mae

产物:
  weights.json / results/backtest_predictions.csv / results/ensemble_metrics.csv
"""
import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# GBK 控制台兼容
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from _paths import ensure_import_paths, PROJECT_ROOT                    # noqa: E402
TRANSFORMER_PKG_DIR = ensure_import_paths(verbose=True)

from data_processing import DataProcessor                                # type: ignore # noqa: E402
from ensemble_predictor import (                                         # noqa: E402
    JunshanEnsemblePredictor, fuse, model_fingerprint,
    fuse_quantiles, anchor_median, enforce_monotone, qf_columns, day_interval,
    QUANTILE_LEVELS, MEDIAN_INDEX,
    DEFAULT_RESULT_DIR, DEFAULT_TIMESFM_MODEL, DEFAULT_WEIGHTS_PATH,
    DEFAULT_RAW_DATA, DEFAULT_CALIB_PATH, DAY_STEPS, CONTEXT_HOURS,
)

DEFAULT_OUT_DIR = os.path.join(_HERE, "results")
METRIC_CHOICES = ("mae", "rmse", "mape")

# 回测上下文天数: Transformer 需 184 步 lookback + 168 步 lag_168h warmup
# = 352h ≈ 15 天; 取 25 天留足余量
BACKTEST_CONTEXT_DAYS = 25


# ==================== 指标与权重搜索 ====================

def compute_metrics(y_true, y_pred, mape_min_actual=0.0):
    """MAE / RMSE / MAPE。MAPE 只对真值 > mape_min_actual 的点计算。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        return {"n": 0, "n_mape": 0, "mae": float("nan"),
                "rmse": float("nan"), "mape": float("nan")}
    err = np.abs(y_pred - y_true)
    m = y_true > mape_min_actual
    return {
        "n": int(len(y_true)), "n_mape": int(m.sum()),
        "mae": float(err.mean()),
        "rmse": float(math.sqrt(float((err ** 2).mean()))),
        "mape": float((err[m] / y_true[m]).mean() * 100.0) if m.sum() > 0 else float("nan"),
    }


def grid_search_alpha(y_true, pred_t, pred_f, metric="mae", grid=None):
    """在 [0,1] 上网格搜索使 metric 最小的 α (全向量化)。"""
    if grid is None:
        grid = np.arange(0.0, 1.0 + 1e-9, 0.005)
    grid = np.asarray(grid, dtype=np.float64)

    y_true = np.asarray(y_true, dtype=np.float64)
    pred_t = np.asarray(pred_t, dtype=np.float64)
    pred_f = np.asarray(pred_f, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(pred_t) & np.isfinite(pred_f)
    y_true, pred_t, pred_f = y_true[ok], pred_t[ok], pred_f[ok]
    if len(y_true) == 0:
        return 0.0, []

    D = pred_t - pred_f
    C = pred_f - y_true
    abs_err = np.abs(np.outer(grid, D) + C)
    sq_err = (np.outer(grid, D) + C) ** 2

    mae = abs_err.mean(axis=1)
    rmse = np.sqrt(sq_err.mean(axis=1))
    mask = y_true > 0
    mape = (abs_err[:, mask] / y_true[mask]).mean(axis=1) * 100.0 if mask.sum() else \
        np.full(len(grid), np.nan)

    curves = {"mae": mae, "rmse": rmse, "mape": mape}
    vals = curves[metric]
    best_i = int(np.nanargmin(vals))

    curve = [(float(grid[i]), {"mae": float(mae[i]), "rmse": float(rmse[i]),
                               "mape": float(mape[i])}) for i in range(len(grid))]
    return float(grid[best_i]), curve


def closed_form_mse_alpha(pred_t, pred_f, y_true):
    """MSE 最优权的闭式解 (无约束), 仅作参考。"""
    d = pred_t - pred_f
    denom = float((d ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float(((y_true - pred_f) * d).sum() / denom)


def err_correlation(pred_t, pred_f, y_true):
    """两模型误差的相关系数 — 越低越适合融合。"""
    et, ef = pred_t - y_true, pred_f - y_true
    if len(et) < 2 or np.std(et) == 0 or np.std(ef) == 0:
        return float("nan")
    return float(np.corrcoef(et, ef)[0, 1])


# ==================== 区间 (概率预测) 指标 ====================

def interval_scores(y_true, q, levels=QUANTILE_LEVELS):
    """区间质量指标。

    q: (N, L) 分位数矩阵, q[:, j] 对应 levels[j]。

    返回 {"crps", "pinball", "levels": [{nominal, picp, mpiw, winkler, pinball}]}
      PICP    经验覆盖率 (应贴近 nominal)
      MPIW    平均区间宽度 (同覆盖率下越窄越好)
      Winkler 区间得分 (主指标, 越小越好):
              IS = (u-l) + (2/a)(l-y)·1{y<l} + (2/a)(y-u)·1{y>u}
              其中 a = 误覆盖率 = 1 - 名义覆盖率 (Gneiting & Raftery 2007)。
              80% 区间 -> a = 0.20 -> 惩罚系数 2/0.20 = 10。
      pinball 分位数损失; 对 levels 取平均再 ×2 ≈ CRPS
    """
    y = np.asarray(y_true, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    lv = np.asarray(levels, dtype=np.float64)
    L = len(lv)
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    y, q = y[ok], q[ok]
    if len(y) == 0:
        return {"crps": float("nan"), "pinball": float("nan"), "levels": []}

    pins = np.stack([
        np.where(y >= q[:, j], lv[j] * (y - q[:, j]), (1.0 - lv[j]) * (q[:, j] - y))
        for j in range(L)], axis=1)                       # (N, L)

    rows = []
    for j in range(L):
        if lv[j] >= 0.5:
            continue
        k = L - 1 - j                                     # 对称的另一侧
        lo, hi = q[:, j], q[:, k]
        nominal = float(lv[k] - lv[j])                # 名义覆盖率 (如 0.80)
        alpha_mis = 1.0 - nominal                     # 误覆盖率 (如 0.20)
        IS = (hi - lo) + (2.0 / alpha_mis) * (lo - y) * (y < lo) + \
            (2.0 / alpha_mis) * (y - hi) * (y > hi)
        rows.append({
            "nominal": round(nominal, 2),
            "picp": float(np.mean((y >= lo) & (y <= hi))),
            "mpiw": float(np.mean(hi - lo)),
            "winkler": float(np.mean(IS)),
            "pinball": float((pins[:, j] + pins[:, k]).mean() / 2.0),
        })

    return {"crps": float(2.0 * pins.mean()),
            "pinball": float(pins.mean()), "levels": rows}


def calibrate_residual_quantiles(pred_cols, y_true, hours, is_fit,
                                 levels=QUANTILE_LEVELS, min_count=40,
                                 smooth=True):
    """用样本外回测残差标定逐小时经验分位数 (中位数中心化)。

    ĝ(τ, hour) = quantile_τ(resid[hour]) - quantile_0.5(resid[hour])

    中心化保证 q(0.5) = pred + 0 = pred, 即点预测不被概率预测改动。
    桶内样本不足 min_count 时回退到全局池化分位数。

    pred_cols: {"transformer": 数组, "ensemble": 数组} — 各自标定一份
    hours: 每个回测点的小时 (0..23)
    is_fit: 用于标定的布尔掩码 (必须与 α 拟合窗口独立或至少不在 valid 上)

    返回 (hourly_dict, meta_dict)
    """
    y = np.asarray(y_true, dtype=np.float64)
    hours = np.asarray(hours)
    is_fit = np.asarray(is_fit, dtype=bool)
    L = len(levels)

    hourly, meta = {}, {}
    for key, pred in pred_cols.items():
        resid = np.asarray(pred, dtype=np.float64) - y
        # 丢弃非有限点, 否则 np.quantile 会把整桶污染成 NaN
        calib_ok = is_fit & np.isfinite(resid)
        if not calib_ok.any():
            raise ValueError(f"{key} 的校准残差全为非有限值, 无法校准")
        pool = np.quantile(resid[calib_ok], levels)
        pool = pool - pool[MEDIAN_INDEX]

        tab = np.zeros((DAY_STEPS, L), dtype=np.float64)
        counts = np.zeros(DAY_STEPS, dtype=int)
        n_fallback = 0
        for h in range(DAY_STEPS):
            r = resid[calib_ok & (hours == h)]
            counts[h] = len(r)
            if len(r) >= min_count:
                qq = np.quantile(r, levels)
                tab[h] = qq - qq[MEDIAN_INDEX]
            else:
                tab[h] = pool
                n_fallback += 1

        if smooth:
            # 时刻的 0 点和 23 点是相邻的, 用循环 3 点均值降噪 (保序, 中位数仍为 0)
            tab = (np.roll(tab, 1, axis=0) + tab + np.roll(tab, -1, axis=0)) / 3.0
            tab = enforce_monotone(tab)

        hourly[key] = tab
        meta[key] = {"pooled": pool.tolist(),
                     "n_per_hour": counts.tolist(),
                     "n_fallback_hours": int(n_fallback),
                     "n_points": int(is_fit.sum())}
    return hourly, meta


def residual_quantile_table(calib, key, pred_point, hours):
    """把校准表套到点预测上: q = pred + ĝ(τ, hour)。"""
    tab = np.asarray(calib["hourly"][key], dtype=np.float64)
    pred_point = np.asarray(pred_point, dtype=np.float64)
    hours = np.asarray(hours)
    return pred_point[:, None] + tab[hours]


# ==================== 回测 ====================

def run_backtest(predictor, raw_df, test_days,
                 context_days=BACKTEST_CONTEXT_DAYS, decision_hour=15):
    """逐日滚动回测: 截断到决策时刻, 两个模型分别预测当天。

    生产口径是"每天 16 点做次日全天预测", 此刻手上只有前一天 15:00 为止的
    数据 (decision_hour=15)。回测必须用同一个截止点, 否则会给模型喂进生产
    拿不到的数据 (前一天 16:00~23:00 共 8 小时), 指标偏乐观。

    predictor: JunshanEnsemblePredictor (已初始化)
    raw_df: 原始 DataFrame (时间索引 + 出厂水流量)
    test_days: 回测天数列表或 int
    context_days: 每个回测窗口的上下文天数
    decision_hour: 决策时刻可用数据的最后小时 (默认 15; 23 表示可用整个前一天)

    返回: (backtest_df, per_day_df)
    """
    # 完整预处理 (全量数据, 供 TimesFM 用)
    print(f"\n[回测] 预处理全量数据...")
    full_feat, full_clean = predictor.preprocess(raw_df)
    hourly_full = full_clean["Total_Flow"].resample("h").mean().dropna()

    # 确定回测天列表
    if isinstance(test_days, int):
        all_dates = sorted(set(full_feat.index.normalize()))
        cutoff = all_dates[-1] - pd.Timedelta(days=test_days)
        days = [d for d in all_dates if d > cutoff]
    else:
        days = [pd.Timestamp(d).normalize() for d in test_days]

    print(f"[回测] {len(days)} 天: {days[0].date()} ~ {days[-1].date()}")

    frames, per_day = [], []
    for i, day in enumerate(days):
        print(f"\n[回测 {i+1}/{len(days)}] {day.date()}", end="")

        # ① 截断数据: 到决策时刻 (前一天 decision_hour 点) 为止的 context_days 天
        decision_ts = day.normalize() - pd.Timedelta(days=1) + \
            pd.Timedelta(hours=decision_hour)
        mask = (raw_df.index > decision_ts - pd.Timedelta(days=context_days)) & \
               (raw_df.index <= decision_ts)
        win_raw = raw_df.loc[mask]
        if len(win_raw) == 0:
            print("  [跳过] 无数据")
            continue

        # ② Transformer 预测 (自己滚过日内缺口, 与生产一致)
        try:
            pred_t = predictor.predict_transformer(win_raw, target_date=day)
        except Exception as e:
            print(f"  [失败] Transformer: {e}")
            continue

        # ③ TimesFM 预测 (与生产同一条代码路径, 保证对齐口径一致)
        try:
            pred_f, q_f = predictor.predict_timesfm_quantiles(win_raw, target_date=day)
        except Exception as e:
            print(f"  [失败] TimesFM: {e}")
            continue

        # ④ 安全网: TimesFM 异常值裁剪 (点预测 + 分位数, 裁剪后恢复单调)
        ref_ctx = hourly_full[hourly_full.index <= decision_ts].iloc[-CONTEXT_HOURS:]
        ref = float(np.median(ref_ctx.to_numpy()))
        lo, hi = 0.3 * ref, 3.0 * ref
        n_clipped = int(((pred_f < lo) | (pred_f > hi)).sum())
        if n_clipped:
            pred_f = np.clip(pred_f, lo, hi)
        if q_f.shape == (DAY_STEPS, len(QUANTILE_LEVELS)):
            # 裁剪到护栏区间, 再把中位数钉回 pred_f (anchor_median 会同时
            # 投影左右两侧, 避免裁剪+钉中位数把单调性弄坏)
            q_f = anchor_median(np.clip(q_f, lo, hi), pred_f)
        else:
            q_f = None

        # ⑤ 真值: 清洗后的当天24小时
        day_mask = hourly_full.index.normalize() == day
        actual_day = hourly_full.loc[day_mask]
        if len(actual_day) < 20:
            print(f"  [跳过] 真值仅 {len(actual_day)} 点")
            continue
        y_true = actual_day.iloc[:DAY_STEPS].to_numpy(dtype=np.float64)
        n_true = len(y_true)
        n_gap = max(0, 23 - decision_hour)      # 15 -> 8 (前一天 16:00~23:00)

        print(f"  T:{len(pred_t)} | F:{len(pred_f)} | 真值:{n_true}"
              f" | 日内缺口:{n_gap}h"
              f"{' | 裁剪:'+str(n_clipped) if n_clipped else ''}")

        # 逐点记录 (qf_10..qf_90 = TimesFM 原生分位数, 供区间校准与出图复用)
        times = pd.date_range(start=day, periods=DAY_STEPS, freq="h")
        row = {
            "date": str(day.date()),
            "y_true": y_true, "pred_transformer": pred_t, "pred_timesfm": pred_f,
        }
        if q_f is not None:
            for j, lvl in enumerate(QUANTILE_LEVELS):
                row[f"qf_{int(round(lvl * 100)):02d}"] = q_f[:, j]
        frames.append(pd.DataFrame(row, index=times))
        per_day.append({"date": str(day.date()), "n_points": DAY_STEPS,
                        "n_true": n_true, "n_timesfm_clipped": n_clipped,
                        "n_gap_steps": n_gap, "decision_hour": decision_hour})

    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out.index.name = "timestamp"
    return out, pd.DataFrame(per_day)


# ==================== 画图 ====================

def _setup_cn_font():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def plot_weight_curve(curve, best_alpha, metric, save_path):
    """指标随 α 变化的曲线。"""
    _setup_cn_font()
    import matplotlib.pyplot as plt

    alphas = np.array([a for a, _ in curve])
    vals = np.array([m[metric] for _, m in curve])
    finite = np.isfinite(vals)
    unit = " (%)" if metric == "mape" else " (m\u00b3/h)"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(alphas[finite], vals[finite], color="#2c3e50", linewidth=1.8, label=metric.upper())
    ax.axvline(best_alpha, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"\u03b1 = {best_alpha:.3f}")
    ax.scatter([best_alpha], [vals[finite].min()], color="#e74c3c", zorder=5, s=45)
    ax.axvline(0.0, color="#95a5a6", linestyle=":", linewidth=1, label="\u03b1=0 (仅TimesFM)")
    ax.axvline(1.0, color="#95a5a6", linestyle="-.", linewidth=1, label="\u03b1=1 (仅Transformer)")
    ax.set_xlabel("\u03b1  (1-\u03b1 为 TimesFM 权重)", fontsize=12)
    ax.set_ylabel(metric.upper() + unit, fontsize=12)
    ax.set_title(f"融合权重搜索 \u2014 拟合窗口 {metric.upper()} 随 \u03b1 变化", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 权重曲线已保存: {save_path}")


def plot_comparison(bt, alpha, valid_start, save_path):
    """真值 + Transformer + TimesFM + 融合 四条曲线。"""
    _setup_cn_font()
    import matplotlib.pyplot as plt

    y = bt["y_true"].to_numpy(dtype=np.float64)
    pt = bt["pred_transformer"].to_numpy(dtype=np.float64)
    pf = bt["pred_timesfm"].to_numpy(dtype=np.float64)
    pe = fuse(alpha, pt, pf)

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(bt.index, y, color="#2c3e50", linewidth=1.3, label="真实值")
    ax.plot(bt.index, pt, color="#3498db", linewidth=1.0, alpha=0.85, label="Transformer")
    ax.plot(bt.index, pf, color="#27ae60", linewidth=1.0, alpha=0.85, label="TimesFM")
    ax.plot(bt.index, pe, color="#e74c3c", linewidth=1.6, alpha=0.9,
            label=f"融合 (\u03b1={alpha:.3f})")
    if valid_start is not None:
        ax.axvline(pd.Timestamp(valid_start), color="#8e44ad", linestyle="--",
                   linewidth=1.6, label="验证窗口起点")
    ax.set_title("两个模型独立预测 vs 加权融合 (样本外回测)", fontsize=14)
    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("出厂水流量 (m\u00b3/h)", fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 对比图已保存: {save_path}")


def plot_daily_figures(bt, save_dir, alpha, valid_start=None, per_day_y=False,
                       mape_min_actual=0.0, calib=None,
                       interval_method="vincentization", with_band=True):
    """逐日对比图: 每天一张 (+ 可选 80% 区间带)。"""
    _setup_cn_font()
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    n_band = 0
    dates = sorted(bt["date"].unique())
    for day_str in dates:
        day_data = bt[bt["date"] == day_str]
        if len(day_data) == 0:
            continue

        y = day_data["y_true"].to_numpy()
        pt = day_data["pred_transformer"].to_numpy()
        pf = day_data["pred_timesfm"].to_numpy()
        pe = fuse(alpha, pt, pf)
        hours = np.arange(len(y))

        q_e = day_interval(day_data, alpha, calib, interval_method) \
            if with_band else None
        if q_e is not None:
            n_band += 1

        mae_t = np.mean(np.abs(pt - y))
        mae_f = np.mean(np.abs(pf - y))
        mae_e = np.mean(np.abs(pe - y))

        fig, ax = plt.subplots(figsize=(12, 5))
        if q_e is not None:
            ax.fill_between(hours, q_e[:, 0], q_e[:, -1], color="#e74c3c",
                            alpha=0.15, linewidth=0, label="80% 区间")
        ax.plot(hours, y, color="#2c3e50", linewidth=1.8, marker="o", ms=3, label="真实值")
        ax.plot(hours, pt, color="#3498db", linewidth=1.2, linestyle="--",
                marker="s", ms=2, label=f"Transformer (MAE={mae_t:.0f})")
        ax.plot(hours, pf, color="#27ae60", linewidth=1.2, linestyle="--",
                marker="^", ms=2, label=f"TimesFM (MAE={mae_f:.0f})")
        ax.plot(hours, pe, color="#e74c3c", linewidth=1.8,
                label=f"融合 (MAE={mae_e:.0f})")
        ax.set_xticks(hours[::2])
        ax.set_xticklabels([f"{h:02d}:00" for h in hours[::2]])
        ax.set_xlabel("时刻")
        ax.set_ylabel("出厂水流量 (m\u00b3/h)")

        tag = ""
        if valid_start and pd.Timestamp(day_str) >= pd.Timestamp(valid_start):
            tag = " [验证]"
        ax.set_title(f"{day_str} 融合预测 vs 实际{tag}   \u03b1={alpha:.3f}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"daily_{day_str.replace('-', '')}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[plot] 逐日图已保存: {save_dir} ({len(dates)} 张)"
          + (f" (其中 {n_band} 张含 80% 区间带)" if n_band else ""))


def plot_coverage_curve(curves, save_path, tag=""):
    """名义覆盖率 vs 经验覆盖率 —— 越贴近对角线越可信。

    curves: {"方法名": [{"nominal","picp"}, ...], ...}
    """
    _setup_cn_font()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot([0, 1], [0, 1], color="#95a5a6", linestyle="--", linewidth=1.2,
            label="完美校准", zorder=1)
    colors = ["#3498db", "#27ae60", "#e74c3c", "#8e44ad", "#f39c12"]
    for (name, rows), c in zip(curves.items(), colors):
        if not rows:
            continue
        x = [r["nominal"] for r in rows]
        y = [r["picp"] for r in rows]
        ax.plot(x, y, marker="o", ms=6, linewidth=1.8, color=c, label=name)
    ax.set_xlabel("名义覆盖率", fontsize=12)
    ax.set_ylabel("经验覆盖率", fontsize=12)
    ax.set_title(f"区间校准曲线 {tag}", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 校准曲线已保存: {save_path}")


def plot_fan_chart(bt, q_lo, q_hi, q_lo2, q_hi2, save_path, valid_start=None,
                   title="融合预测区间 (样本外回测)",
                   outer="80% 区间", inner="60% 区间"):
    """扇形图: 真值 + 中位数 + 两层区间带。"""
    _setup_cn_font()
    import matplotlib.pyplot as plt

    idx = bt.index
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.fill_between(idx, q_lo, q_hi, color="#e74c3c", alpha=0.16,
                    label=outer, linewidth=0)
    ax.fill_between(idx, q_lo2, q_hi2, color="#e74c3c", alpha=0.22,
                    label=inner, linewidth=0)
    ax.plot(idx, bt["y_true"].to_numpy(), color="#2c3e50", linewidth=1.4,
            label="真实值")
    ax.plot(idx, bt["pred_ensemble"].to_numpy(), color="#e74c3c", linewidth=1.5,
            label="融合中位数")
    if valid_start is not None:
        ax.axvline(pd.Timestamp(valid_start), color="#8e44ad", linestyle="--",
                   linewidth=1.6, label="验证窗口起点")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("出厂水流量 (m³/h)", fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 扇形图已保存: {save_path}")


# ==================== 概率预测: 校准 + 评估 + 产物 ====================

def run_interval_calibration(args, bt, alpha, method, fit_days, valid_days,
                             is_fit, is_valid):
    """区间校准 → 多方案对比 → 落盘产物。返回 summary dict。"""
    y = bt["y_true"].to_numpy(dtype=np.float64)
    pt = bt["pred_transformer"].to_numpy(dtype=np.float64)
    pe = bt["pred_ensemble"].to_numpy(dtype=np.float64)
    q_f = bt[qf_columns()].to_numpy(dtype=np.float64)
    hours = bt.index.hour.to_numpy()

    # ── fit / calib 窗口切分 (calib 从 fit 尾部切出, valid 保持完全独立) ──
    if 0 < args.calib_days < len(fit_days):
        calib_days = fit_days[-args.calib_days:]
        alpha_days = fit_days[:-args.calib_days]
    else:
        calib_days = fit_days
        alpha_days = fit_days
    is_calib = bt["date"].isin({str(d.date()) for d in calib_days}).to_numpy()

    print(f"\n{'=' * 78}\n 概率预测 (区间) 校准与评估\n{'=' * 78}")
    print(f"  校准窗口: {calib_days[0].date()} ~ {calib_days[-1].date()} "
          f"({len(calib_days)} 天, {int(is_calib.sum())} 点)")
    if args.calib_days > 0:
        print(f"  α 拟合窗口: {alpha_days[0].date()} ~ {alpha_days[-1].date()} "
              f"({len(alpha_days)} 天)")
    print(f"  评估窗口: valid {valid_days[0].date()} ~ {valid_days[-1].date()} "
          f"({int(is_valid.sum())} 点, 完全不参与校准)")
    print(f"  融合方法: {method}   逐小时最少样本: {args.min_calib_count}")

    # ── 逐小时残差分位数校准 ──
    hourly, calib_meta = calibrate_residual_quantiles(
        {"transformer": pt, "ensemble": pe}, y, hours, is_calib,
        min_count=args.min_calib_count, smooth=not args.no_calib_smooth)
    n_fb = calib_meta["transformer"]["n_fallback_hours"]
    counts = calib_meta["transformer"]["n_per_hour"]
    print(f"  逐小时样本数: min={min(counts)} max={max(counts)} "
          f"(阈值 {args.min_calib_count})")
    if n_fb:
        print(f"  [提示] Transformer 有 {n_fb}/24 个小时样本不足 "
              f"{args.min_calib_count}, 已回退全局池化分位数; "
              f"可用更长的校准窗口或调低 --min_calib_count")

    calib = {
        "schema_version": 1,
        "levels": list(QUANTILE_LEVELS),
        "median_index": MEDIAN_INDEX,
        "day_steps": DAY_STEPS,
        "context_hours": CONTEXT_HOURS,
        "alpha": float(alpha),
        "interval_method": method,
        "hourly": {k: v.tolist() for k, v in hourly.items()},
        "pooled": {k: m["pooled"] for k, m in calib_meta.items()},
        "n_per_hour": calib_meta["transformer"]["n_per_hour"],
        "n_fallback_hours": n_fb,
        "n_days": len(calib_days),
        "decision_hour": args.decision_hour,
        "calib_window": f"{calib_days[0].date()} ~ {calib_days[-1].date()} "
                        f"({len(calib_days)} 天)",
        "smooth_hourly": not args.no_calib_smooth,
        "transformer_result_dir": os.path.abspath(args.result_dir),
        "timesfm_model_path": os.path.abspath(args.timesfm_model),
        "lora_path": os.path.abspath(args.lora_path) if args.lora_path else None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    # ── 四个候选区间 ──
    q_T = residual_quantile_table(calib, "transformer", pt, hours)
    q_A = fuse_quantiles(alpha, q_T, q_f, method)
    q_B = residual_quantile_table(calib, "ensemble", pe, hours)
    candidates = [
        ("TimesFM 原生区间", q_f),
        ("Transformer 残差校准", q_T),
        (f"方案A {method}", q_A),
        ("方案B 直接校准", q_B),
    ]
    if method != "mixture":
        # 线性池也放进来做对照: 它天然不保 "中位数 == 点预测", 这里按推理时的
        # 做法 (anchor_median) 锚定后再比, 保证比的是实际会用到的区间。
        q_mix = anchor_median(fuse_quantiles(alpha, q_T, q_f, "mixture"), pe)
        mix_gap = float(np.max(np.abs(
            fuse_quantiles(alpha, q_T, q_f, "mixture")[:, MEDIAN_INDEX] - pe)))
        print(f"  对照: mixture 原始中位数偏离点预测 max = {mix_gap:.1f} m³/h "
              f"(已锚定; 该偏离量说明两模型分歧很大)")
        candidates.append(("方案A' mixture", q_mix))
    for name, qq in candidates:
        bad = np.argwhere(np.diff(qq, axis=-1) < -1e-9)
        assert len(bad) == 0, (f"{name} 区间非单调: {len(bad)} 处, "
                               f"首个在 行{bad[0][0]} 列{bad[0][1]}")
    gap = float(np.max(np.abs(q_A[:, MEDIAN_INDEX] - pe)))
    print(f"  不变量校验: |q_A(0.5) - 融合点预测| max = {gap:.2e}  (应恒为 0)")

    # ── 评估 ──
    rows, summary = [], {}
    for tag, mask in (("valid", is_valid), ("fit", is_fit), ("全部", is_fit | is_valid)):
        if mask.sum() == 0:
            continue
        for name, qq in candidates:
            sc = interval_scores(y[mask], qq[mask])
            summary.setdefault(tag, {})[name] = {
                "crps": sc["crps"], "pinball": sc["pinball"],
                "levels": sc["levels"]}
            for r in sc["levels"]:
                rows.append({"窗口": tag, "方法": name, "名义覆盖": r["nominal"],
                             "经验覆盖": r["picp"], "覆盖偏差": r["picp"] - r["nominal"],
                             "平均宽度": r["mpiw"], "Winkler": r["winkler"],
                             "CRPS": sc["crps"]})
    df_interval = pd.DataFrame(rows)

    for tag in ("valid", "fit"):
        if tag not in summary or not (is_valid if tag == "valid" else is_fit).sum():
            continue
        n_pt = int((is_valid if tag == "valid" else is_fit).sum())
        print(f"\n  ── {tag} 窗口 ({n_pt} 点) ──")
        print(f"  {'方法':<24}{'名义':>6}{'经验':>8}{'偏差':>8}"
              f"{'宽度':>9}{'Winkler':>9}{'CRPS':>8}")
        for name, _ in candidates:
            blk = summary[tag][name]
            for i, r in enumerate(blk["levels"]):
                nm = name if i == 0 else ""
                crps = f"{blk['crps']:8.1f}" if i == 0 else ""
                print(f"  {nm:<24}{r['nominal']:>6.2f}{r['picp']:>8.3f}"
                      f"{r['picp'] - r['nominal']:>+8.3f}{r['mpiw']:>9.1f}"
                      f"{r['winkler']:>9.1f}{crps}")

    # ── 结论: 主指标是 Winkler (同时惩罚覆盖不足与区间过宽) ──
    if "valid" in summary:
        v = summary["valid"]
        best = min(v.items(), key=lambda kv: kv[1]["crps"])
        w = min(v.items(), key=lambda kv: kv[1]["levels"][0]["winkler"])
        print(f"\n  valid 窗口最优 (CRPS): {best[0]}   CRPS={best[1]['crps']:.1f}")
        print(f"  valid 窗口最优 (80% Winkler): {w[0]}   "
              f"Winkler={w[1]['levels'][0]['winkler']:.1f}")
        print(f"  建议: 默认使用 '{method}' 区间融合方法")

    # ── 产物 ──
    with open(args.calib_out, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"\n[区间] 校准表已保存: {args.calib_out}")
    df_interval.to_csv(os.path.join(args.out_dir, "interval_metrics.csv"),
                       index=False, encoding="utf-8-sig", float_format="%.4f")

    if "valid" in summary:
        plot_coverage_curve(
            {name: summary["valid"][name]["levels"] for name, _ in candidates},
            os.path.join(args.out_dir, "interval_coverage.png"),
            tag=f"(valid 窗口, α={alpha:.3f})")

    plot_fan_chart(
        bt, q_A[:, 0], q_A[:, -1], q_A[:, 1], q_A[:, -2],
        os.path.join(args.out_dir, "interval_fan.png"),
        valid_start=pd.Timestamp(valid_days[0]).normalize(),
        title=f"融合预测区间 (样本外回测, {method}, α={alpha:.3f})")

    return {
        "method": method,
        "calib_path": os.path.abspath(args.calib_out),
        "calib_window": calib["calib_window"],
        "n_fallback_hours": n_fb,
        "median_gap": gap,
        "valid": summary.get("valid", {}),
    }


# ==================== 主流程 ====================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Transformer × TimesFM 融合权重训练 (军山水厂)")
    p.add_argument("--data", default=DEFAULT_RAW_DATA, help="原始数据 CSV")
    p.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="Transformer 结果目录")
    p.add_argument("--timesfm_model", default=DEFAULT_TIMESFM_MODEL, help="TimesFM 模型目录")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="产物输出目录")
    p.add_argument("--weights_out", default=DEFAULT_WEIGHTS_PATH, help="weights.json 输出路径")
    p.add_argument("--test_days", type=int, default=90, help="样本外回测天数 (默认 90)")
    p.add_argument("--fit_days", type=int, default=45, help="拟合 α 的天数 (默认 = test_days/2)")
    p.add_argument("--valid_days", type=int, default=45, help="独立验证天数 (默认 = test_days/2)")
    p.add_argument("--context_days", type=int, default=BACKTEST_CONTEXT_DAYS,
                   help="每个回测窗口的上下文天数 (默认 25)")
    p.add_argument("--decision_hour", type=int, default=15,
                   help="决策时刻可用数据的最后小时 (默认 15 = 16 点决策口径, "
                        "与生产输入 CSV 一致; 23 = 可用整个前一天, 偏乐观)")
    p.add_argument("--metric", choices=METRIC_CHOICES, default="mae", help="优化目标")
    p.add_argument("--mape_min_actual", type=float, default=0.0,
                   help="MAPE 过滤阈值 (m\u00b3/h, 默认 0)")
    p.add_argument("--alpha_grid", type=float, default=0.005, help="α 网格步长")
    p.add_argument("--lora_path", default=None, help="TimesFM LoRA 微调权重路径 (lora_weights.pth)")
    p.add_argument("--no_lora", action="store_true", help="不使用 LoRA 微调权重 (使用原始 TimesFM)")
    p.add_argument("--device", default=None, help="推理设备")
    p.add_argument("--no_daily_plots", action="store_true", help="跳过逐日图")
    p.add_argument("--per_day_y", action="store_true", help="逐日图各自缩放 y 轴")
    # ─ 概率预测 (区间) ──
    p.add_argument("--calib_out", default=DEFAULT_CALIB_PATH,
                   help="区间校准表输出路径 (quantile_calibration.json)")
    p.add_argument("--no_interval", action="store_true",
                   help="跳过区间校准与评估, 只做点预测权重")
    p.add_argument("--calib_days", type=int, default=0,
                   help="从 fit 窗口尾部切出用于区间校准的天数 (默认 0 = 用整个 fit 窗口)")
    p.add_argument("--interval_method", default="vincentization",
                   choices=["vincentization", "mixture"],
                   help="区间融合方法 (默认 vincentization)")
    p.add_argument("--min_calib_count", type=int, default=40,
                   help="逐小时校准的最少样本数, 不足则回退全局池化 (默认 40; "
                        "45 天校准窗口下每小时约 45 个样本)")
    p.add_argument("--no_calib_smooth", action="store_true",
                   help="关闭逐小时分位数的循环 3 点平滑")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 加载模型 ──
    # 训练权重时 weights.json 尚不存在, 用 alpha=0.5 占位 (不影响回测)
    print(f"[fit_weights] 加载模型...")
    if args.no_lora:
        lora_path = None
        print("[fit_weights] --no_lora: 不使用 LoRA, 使用原始 TimesFM")
    else:
        lora_path = args.lora_path or os.path.join(_HERE, "timesfm_lora", "lora_weights.pth")
        if not os.path.exists(lora_path):
            lora_path = None
            print("[fit_weights] 未找到 LoRA 权重, 使用原始 TimesFM")
        else:
            print(f"[fit_weights] 使用 LoRA 微调 TimesFM: {lora_path}")
    predictor = JunshanEnsemblePredictor(
        result_dir=args.result_dir, timesfm_model_path=args.timesfm_model,
        weights_path=args.weights_out, alpha=0.5, lora_path=lora_path,
        device=args.device, verbose=False)

    # ── 加载数据 ──
    raw_df = pd.read_csv(args.data, encoding="utf-8-sig")
    for ts_col in ("时间", "timestamp"):
        if ts_col in raw_df.columns:
            raw_df[ts_col] = pd.to_datetime(raw_df[ts_col])
            raw_df = raw_df.set_index(ts_col)
            break
    raw_df = raw_df.sort_index()
    print(f"[fit_weights] 原始数据: {raw_df.index.min()} ~ {raw_df.index.max()}, "
          f"{len(raw_df)} 行")

    # ── 确定样本外天数 ──
    # 用 Transformer 的 test_days 确定样本外段
    train_test_days = int(predictor.config.get("test_days", 90))
    full_feat, _ = predictor.preprocess(raw_df)
    all_dates = sorted(set(full_feat.index.normalize()))
    oos_cutoff = all_dates[-1] - pd.Timedelta(days=train_test_days)
    oos_days = [d for d in all_dates if d > oos_cutoff]

    n_need = args.fit_days + args.valid_days
    if len(oos_days) < n_need:
        print(f"\n[警告] 样本外只有 {len(oos_days)} 天, "
              f"不足 fit({args.fit_days}) + valid({args.valid_days}) = {n_need} 天")
        print(f"  请用更大的 test_days 重训: "
              f"python train_transformer_nextday_16h.py --test_days {n_need * 2}")
        # 降级: 用所有可用天
        args.fit_days = len(oos_days) // 2
        args.valid_days = len(oos_days) - args.fit_days
        print(f"  降级: fit={args.fit_days}, valid={args.valid_days}")

    fit_days = oos_days[:args.fit_days]
    valid_days = oos_days[args.fit_days:args.fit_days + args.valid_days]
    backtest_days = fit_days + valid_days

    print(f"\n[fit_weights] 样本外 {len(oos_days)} 天: "
          f"{oos_days[0].date()} ~ {oos_days[-1].date()}")
    print(f"  拟合窗口 (fit)  : {fit_days[0].date()} ~ {fit_days[-1].date()}  "
          f"({len(fit_days)} 天)")
    print(f"  验证窗口 (valid): {valid_days[0].date()} ~ {valid_days[-1].date()}  "
          f"({len(valid_days)} 天)")

    # ── 回测 ──
    print(f"\n{'=' * 78}\n 滚动回测 ({len(backtest_days)} 天)\n{'=' * 78}")
    bt, per_day = run_backtest(predictor, raw_df, backtest_days,
                               context_days=args.context_days,
                               decision_hour=args.decision_hour)
    if bt.empty:
        print("\n[警告] 回测未产生任何数据点")
        return 1

    fit_set = {str(d.date()) for d in fit_days}
    val_set = {str(d.date()) for d in valid_days}
    is_fit = bt["date"].isin(fit_set).to_numpy()
    is_valid = bt["date"].isin(val_set).to_numpy()

    y = bt["y_true"].to_numpy(dtype=np.float64)
    pt = bt["pred_transformer"].to_numpy(dtype=np.float64)
    pf = bt["pred_timesfm"].to_numpy(dtype=np.float64)

    # ── 训练 α (只用 fit 窗口) ──
    grid = np.arange(0.0, 1.0 + 1e-9, args.alpha_grid)
    alpha, curve = grid_search_alpha(y[is_fit], pt[is_fit], pf[is_fit],
                                     args.metric, grid)
    alpha_mse = closed_form_mse_alpha(pt[is_fit], pf[is_fit], y[is_fit])
    alpha_valid, _ = grid_search_alpha(y[is_valid], pt[is_valid], pf[is_valid],
                                       args.metric, grid)
    rho_fit = err_correlation(pt[is_fit], pf[is_fit], y[is_fit])

    print(f"\n{'=' * 78}\n α 训练结果 (目标 {args.metric.upper()})\n{'=' * 78}")
    print(f"  fit 窗口最优 α   = {alpha:.3f}   "
          f"(Transformer {alpha:.1%} + TimesFM {1-alpha:.1%})")
    print(f"  valid 窗口最优 α = {alpha_valid:.3f}   ← 独立估计")
    print(f"  MSE 闭式解 α     = {alpha_mse:.3f}   (仅参考)")
    print(f"  fit/valid α 之差 = {abs(alpha - alpha_valid):.3f}")
    print(f"  两模型误差相关系数 ρ = {rho_fit:.3f}   (越低越互补)")

    # ── 指标对比 ──
    def eval_block(mask, tag):
        rows = []
        candidates = [
            ("Transformer (α=1)", pt[mask]),
            ("TimesFM (α=0)", pf[mask]),
            (f"融合 (α={alpha:.3f})", fuse(alpha, pt[mask], pf[mask])),
        ]
        for name, pred in candidates:
            m = compute_metrics(y[mask], pred, args.mape_min_actual)
            rows.append({"窗口": tag, "模型": name, "有效点": m["n"],
                         "MAE": m["mae"], "RMSE": m["rmse"], "MAPE(%)": m["mape"]})
        return rows

    metric_rows = eval_block(is_fit, f"fit ({len(fit_days)}天)") + \
        eval_block(is_valid, f"valid ({len(valid_days)}天)") + \
        eval_block(is_fit | is_valid, "全部")
    df_metrics = pd.DataFrame(metric_rows)

    print(f"\n{'=' * 78}\n 指标对比 (MAE / RMSE 单位 m\u00b3/h)\n{'=' * 78}")
    print(df_metrics.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # ── 结论: 融合是否优于最好单模型 (valid 窗口) ──
    v = df_metrics[df_metrics["窗口"].str.startswith("valid")].set_index("模型")
    ens_row = [i for i in v.index if i.startswith("融合")][0]
    best_single = min(v.loc["Transformer (α=1)", "MAE"],
                      v.loc["TimesFM (α=0)", "MAE"])
    ens_mae = v.loc[ens_row, "MAE"]
    improved = ens_mae < best_single
    print(f"\n  验证窗口: 融合 MAE = {ens_mae:.2f} vs 最好单模型 = {best_single:.2f}")
    print(f"    -> {'[OK] 融合更优' if improved else '[失败] 融合未超过最好单模型'}, "
          f"改善 {100 * (best_single - ens_mae) / best_single:+.2f}%")

    # ── 产物 ──
    bt_out = bt.copy()
    bt_out["pred_ensemble"] = fuse(alpha, pt, pf)
    bt_csv = os.path.join(args.out_dir, "backtest_predictions.csv")
    bt_out.to_csv(bt_csv, index=True, encoding="utf-8-sig")

    # ── 概率预测 (区间): 校准 + 评估 + 落盘 ──
    interval_summary = None
    if args.no_interval:
        print("\n[区间] --no_interval: 跳过概率预测")
    elif not all(c in bt_out.columns for c in qf_columns()):
        print("\n[区间] [警告] 回测缺少 TimesFM 分位数列, 跳过概率预测")
    else:
        interval_summary = run_interval_calibration(
            args, bt_out, alpha, args.interval_method, fit_days, valid_days,
            is_fit, is_valid)

    df_metrics.to_csv(os.path.join(args.out_dir, "ensemble_metrics.csv"),
                      index=False, encoding="utf-8-sig", float_format="%.4f")
    if not per_day.empty:
        per_day.to_csv(os.path.join(args.out_dir, "backtest_per_day.csv"),
                       index=False, encoding="utf-8-sig")

    n_clipped_total = int(per_day["n_timesfm_clipped"].sum()) if not per_day.empty else 0
    with open(os.path.join(args.out_dir, "ensemble_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"融合权重训练结果 ({datetime.now():%Y-%m-%d %H:%M:%S})\n")
        f.write(f"Transformer : {os.path.abspath(args.result_dir)}\n")
        f.write(f"TimesFM     : {os.path.abspath(args.timesfm_model)}\n")
        f.write(f"LoRA        : {os.path.abspath(lora_path) if lora_path else '无'}\n")
        f.write(f"拟合窗口    : {fit_days[0].date()} ~ {fit_days[-1].date()} "
                f"({len(fit_days)} 天)\n")
        f.write(f"验证窗口    : {valid_days[0].date()} ~ {valid_days[-1].date()} "
                f"({len(valid_days)} 天)\n")
        f.write(f"优化目标    : {args.metric}\n")
        f.write(f"决策口径    : 前一天 {args.decision_hour}:00 截止 "
                f"(日内缺口 {max(0, 23 - args.decision_hour)}h)\n")
        f.write(f"误差相关系数 ρ = {rho_fit:.4f}\n")
        f.write(f"TimesFM 护栏裁剪 = {n_clipped_total}\n\n")
        f.write(f"alpha (fit)   = {alpha:.4f}\n")
        f.write(f"alpha (valid) = {alpha_valid:.4f}\n")
        f.write(f"alpha (MSE)   = {alpha_mse:.4f}\n\n")
        f.write(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        f.write("\n")
        if interval_summary:
            f.write(f"\n── 概率预测 (区间) ──\n")
            f.write(f"区间融合方法  = {interval_summary['method']}\n")
            f.write(f"区间校准窗口  = {interval_summary['calib_window']}\n")
            f.write(f"校准表        = {interval_summary['calib_path']}\n")
            f.write(f"点预测不变量  |q(0.5)-pred|max = "
                    f"{interval_summary['median_gap']:.2e}\n")
            for name, blk in interval_summary["valid"].items():
                l0 = blk["levels"][0]
                f.write(f"  [valid] {name:<24} CRPS={blk['crps']:8.2f}  "
                        f"80%覆盖={l0['picp']:.3f} 宽度={l0['mpiw']:.1f} "
                        f"Winkler={l0['winkler']:.1f}\n")
            f.write("\n")

    # ── weights.json ──
    weights = {
        "schema_version": 1,
        "alpha": alpha,
        "alpha_transformer": alpha,
        "alpha_timesfm": 1.0 - alpha,
        "fit_metric": args.metric,
        "fit_resolution": "1h",
        "decision_hour": args.decision_hour,
        "fit_window": f"{fit_days[0].date()} ~ {fit_days[-1].date()} ({len(fit_days)} 天)",
        "valid_window": f"{valid_days[0].date()} ~ {valid_days[-1].date()} "
                        f"({len(valid_days)} 天)",
        "alpha_valid_estimate": alpha_valid,
        "alpha_mse_closed_form": alpha_mse,
        "error_correlation": rho_fit,
        "n_timesfm_guard_clipped": n_clipped_total,
        "valid_metrics": {
            r["模型"]: {"mae": r["MAE"], "rmse": r["RMSE"], "mape": r["MAPE(%)"]}
            for r in metric_rows if r["窗口"].startswith("valid")
        },
        "transformer_result_dir": os.path.abspath(args.result_dir),
        "timesfm_model_path": os.path.abspath(args.timesfm_model),
        "lora_path": os.path.abspath(lora_path) if lora_path else None,
        "interval": ({
            "method": interval_summary["method"],
            "calibration_path": interval_summary["calib_path"],
            "calibration_window": interval_summary["calib_window"],
            "median_gap": interval_summary["median_gap"],
            "valid_metrics": interval_summary["valid"],
        } if interval_summary else None),
        "n_fit_points": int(is_fit.sum()),
        "n_valid_points": int(is_valid.sum()),
        "mape_min_actual": args.mape_min_actual,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(args.weights_out, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    print(f"\n[weights] 已保存: {args.weights_out}")

    # ── 画图 ──
    plot_weight_curve(curve, alpha, args.metric,
                      os.path.join(args.out_dir, "weight_curve.png"))
    plot_comparison(bt_out, alpha, pd.Timestamp(valid_days[0]).normalize(),
                    os.path.join(args.out_dir, "ensemble_compare.png"))

    if not args.no_daily_plots:
        daily_dir = os.path.join(args.out_dir, "daily")
        band_calib = None if args.no_interval else \
            JunshanEnsemblePredictor._load_calibration(args.calib_out)
        plot_daily_figures(bt_out, daily_dir, alpha,
                           valid_start=pd.Timestamp(valid_days[0]).normalize(),
                           per_day_y=args.per_day_y, calib=band_calib,
                           interval_method=args.interval_method)

    print(f"\n{'=' * 78}")
    print(f" weights.json : {args.weights_out}")
    print(f" 回测明细     : {bt_csv}")
    print(f" 指标对比     : {os.path.join(args.out_dir, 'ensemble_metrics.csv')}")
    print(f"{'=' * 78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

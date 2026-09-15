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
import torch

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
    DEFAULT_RESULT_DIR, DEFAULT_TIMESFM_MODEL, DEFAULT_WEIGHTS_PATH,
    DEFAULT_RAW_DATA, DAY_STEPS, CONTEXT_HOURS,
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


# ==================== 回测 ====================

def run_backtest(predictor, raw_df, test_days,
                 context_days=BACKTEST_CONTEXT_DAYS):
    """逐日滚动回测: 截断到目标日前一天, 两个模型分别预测当天。

    predictor: JunshanEnsemblePredictor (已初始化)
    raw_df: 原始 DataFrame (时间索引 + 出厂水流量)
    test_days: 回测天数列表或 int
    context_days: 每个回测窗口的上下文天数

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

        # ① 截断数据: day 之前 context_days 天
        cutoff_ts = day
        mask = (raw_df.index >= cutoff_ts - pd.Timedelta(days=context_days)) & \
               (raw_df.index < cutoff_ts)
        win_raw = raw_df.loc[mask]
        if len(win_raw) == 0:
            print("  [跳过] 无数据")
            continue

        # ② Transformer 预测
        try:
            pred_t = predictor.predict_transformer(win_raw, target_date=day)
        except Exception as e:
            print(f"  [失败] Transformer: {e}")
            continue

        # ③ TimesFM 预测 (用全量清洗数据的 context, 不截断)
        try:
            ctx = hourly_full[hourly_full.index < cutoff_ts].iloc[-CONTEXT_HOURS:]
            if len(ctx) < 48:
                raise ValueError(f"context 仅 {len(ctx)} 小时")
            point_forecast, _ = predictor.tfm.forecast(
                horizon=DAY_STEPS, inputs=[ctx.to_numpy(dtype=np.float64)])
            pred_f = np.asarray(point_forecast[0], dtype=np.float64)[:DAY_STEPS]
        except Exception as e:
            print(f"  [失败] TimesFM: {e}")
            continue

        # ④ 安全网: TimesFM 异常值裁剪
        ref = float(np.median(ctx.to_numpy()))
        lo, hi = 0.3 * ref, 3.0 * ref
        n_clipped = int(((pred_f < lo) | (pred_f > hi)).sum())
        if n_clipped:
            pred_f = np.clip(pred_f, lo, hi)

        # ⑤ 真值: 清洗后的当天24小时
        day_mask = hourly_full.index.normalize() == day
        actual_day = hourly_full.loc[day_mask]
        if len(actual_day) < 20:
            print(f"  [跳过] 真值仅 {len(actual_day)} 点")
            continue
        y_true = actual_day.iloc[:DAY_STEPS].to_numpy(dtype=np.float64)
        n_true = len(y_true)

        print(f"  T:{len(pred_t)} | F:{len(pred_f)} | 真值:{n_true}"
              f"{' | 裁剪:'+str(n_clipped) if n_clipped else ''}")

        # 逐点记录
        times = pd.date_range(start=day, periods=DAY_STEPS, freq="h")
        frames.append(pd.DataFrame({
            "date": str(day.date()),
            "y_true": y_true, "pred_transformer": pred_t, "pred_timesfm": pred_f,
        }, index=times))
        per_day.append({"date": str(day.date()), "n_points": DAY_STEPS,
                        "n_true": n_true, "n_timesfm_clipped": n_clipped})

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
                       mape_min_actual=0.0):
    """逐日对比图: 每天一张。"""
    _setup_cn_font()
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

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

        mae_t = np.mean(np.abs(pt - y))
        mae_f = np.mean(np.abs(pf - y))
        mae_e = np.mean(np.abs(pe - y))

        fig, ax = plt.subplots(figsize=(12, 5))
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

    print(f"[plot] 逐日图已保存: {save_dir} ({len(dates)} 张)")


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
    p.add_argument("--metric", choices=METRIC_CHOICES, default="mae", help="优化目标")
    p.add_argument("--mape_min_actual", type=float, default=0.0,
                   help="MAPE 过滤阈值 (m\u00b3/h, 默认 0)")
    p.add_argument("--alpha_grid", type=float, default=0.005, help="α 网格步长")
    p.add_argument("--device", default=None, help="推理设备")
    p.add_argument("--no_daily_plots", action="store_true", help="跳过逐日图")
    p.add_argument("--per_day_y", action="store_true", help="逐日图各自缩放 y 轴")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 加载模型 ──
    # 训练权重时 weights.json 尚不存在, 用 alpha=0.5 占位 (不影响回测)
    print(f"[fit_weights] 加载模型...")
    predictor = JunshanEnsemblePredictor(
        result_dir=args.result_dir, timesfm_model_path=args.timesfm_model,
        weights_path=args.weights_out, alpha=0.5, device=args.device, verbose=False)

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
                               context_days=args.context_days)
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
        f.write(f"拟合窗口    : {fit_days[0].date()} ~ {fit_days[-1].date()} "
                f"({len(fit_days)} 天)\n")
        f.write(f"验证窗口    : {valid_days[0].date()} ~ {valid_days[-1].date()} "
                f"({len(valid_days)} 天)\n")
        f.write(f"优化目标    : {args.metric}\n")
        f.write(f"误差相关系数 ρ = {rho_fit:.4f}\n")
        f.write(f"TimesFM 护栏裁剪 = {n_clipped_total}\n\n")
        f.write(f"alpha (fit)   = {alpha:.4f}\n")
        f.write(f"alpha (valid) = {alpha_valid:.4f}\n")
        f.write(f"alpha (MSE)   = {alpha_mse:.4f}\n\n")
        f.write(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        f.write("\n")

    # ── weights.json ──
    weights = {
        "schema_version": 1,
        "alpha": alpha,
        "alpha_transformer": alpha,
        "alpha_timesfm": 1.0 - alpha,
        "fit_metric": args.metric,
        "fit_resolution": "1h",
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
        plot_daily_figures(bt_out, daily_dir, alpha,
                           valid_start=pd.Timestamp(valid_days[0]).normalize(),
                           per_day_y=args.per_day_y)

    print(f"\n{'=' * 78}")
    print(f" weights.json : {args.weights_out}")
    print(f" 回测明细     : {bt_csv}")
    print(f" 指标对比     : {os.path.join(args.out_dir, 'ensemble_metrics.csv')}")
    print(f"{'=' * 78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

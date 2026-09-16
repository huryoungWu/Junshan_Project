# -*- coding: utf-8 -*-
"""逐日对比图 — 每天一张, 显示真实值 / Transformer / TimesFM / 融合 四条曲线。

可独立运行 (从 backtest_predictions.csv 重新出图, 不重跑回测):
  python plot_daily.py
  python plot_daily.py --per_day_y
  python plot_daily.py --alpha 0.6
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ensemble_predictor import (                            # noqa: E402
    JunshanEnsemblePredictor, fuse, day_interval,
    DEFAULT_CALIB_PATH,
)

DEFAULT_BT_CSV = os.path.join(_HERE, "results", "backtest_predictions.csv")
DEFAULT_OUT_DIR = os.path.join(_HERE, "results", "daily")
DEFAULT_ALPHA = 0.5


def _setup_cn_font():
    rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def plot_daily_figures(bt, save_dir, alpha, valid_start=None,
                       per_day_y=False, mape_min_actual=0.0,
                       calib=None, interval_method="vincentization",
                       with_band=True):
    """逐日对比图: 每天一张, 真实值 + Transformer + TimesFM + 融合 (+区间带)。"""
    _setup_cn_font()
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
            if (with_band and calib is not None) else None
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
        ax.set_ylabel("出厂水流量 (m³/h)")

        tag = ""
        if valid_start and pd.Timestamp(day_str) >= pd.Timestamp(valid_start):
            tag = " [验证]"
        ax.set_title(f"{day_str} 融合预测 vs 实际{tag}   α={alpha:.3f}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        fig.savefig(os.path.join(save_dir, f"daily_{day_str.replace('-', '')}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[plot_daily] {len(dates)} 张逐日图已保存: {save_dir}"
          + (f" (其中 {n_band} 张含 80% 区间带)" if n_band else ""))


def main():
    parser = argparse.ArgumentParser(description="逐日对比图 (从回测 CSV 重新出图)")
    parser.add_argument("--csv", default=DEFAULT_BT_CSV,
                        help="backtest_predictions.csv 路径")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--alpha", type=float, default=None,
                        help="融合权重 (默认: 从 CSV 的 pred_ensemble 列反推, 或用 0.5)")
    parser.add_argument("--per_day_y", action="store_true", help="每张图独立 y 轴")
    parser.add_argument("--valid_start", default=None, help="验证窗口起始日期")
    parser.add_argument("--calib", default=DEFAULT_CALIB_PATH,
                        help="区间校准表 (quantile_calibration.json)")
    parser.add_argument("--interval_method", default="vincentization",
                        choices=["vincentization", "mixture"],
                        help="区间融合方法 (默认 vincentization)")
    parser.add_argument("--no_band", action="store_true",
                        help="不画 80% 区间带")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[plot_daily] 找不到回测 CSV: {args.csv}")
        print(f"  请先运行 fit_weights.py 生成回测数据")
        return 1

    bt = pd.read_csv(args.csv, encoding="utf-8-sig")
    if "timestamp" in bt.columns:
        bt = bt.set_index(pd.to_datetime(bt["timestamp"])).drop(columns=["timestamp"])
        bt.index.name = "timestamp"

    # α: 显式指定 > 从 pred_ensemble 列反推 > 默认 0.5
    alpha = args.alpha
    if alpha is None and "pred_ensemble" in bt.columns:
        # 反推: pred_ensemble = alpha * pred_t + (1-alpha) * pred_f
        # 用最小二乘拟合
        pt = bt["pred_transformer"].to_numpy()
        pf = bt["pred_timesfm"].to_numpy()
        pe = bt["pred_ensemble"].to_numpy()
        d = pt - pf
        denom = float((d ** 2).sum())
        if denom > 0:
            alpha = float(((pe - pf) * d).sum() / denom)
            alpha = max(0.0, min(1.0, alpha))
            print(f"[plot_daily] 从 CSV 反推 α = {alpha:.3f}")
    if alpha is None:
        alpha = DEFAULT_ALPHA
        print(f"[plot_daily] 使用默认 α = {alpha}")

    valid_start = args.valid_start
    calib = None if args.no_band else \
        JunshanEnsemblePredictor._load_calibration(args.calib)
    if not args.no_band and calib is None:
        print(f"[plot_daily] 未找到区间校准表, 不画区间带: {args.calib}")
    plot_daily_figures(bt, args.out_dir, alpha,
                       valid_start=valid_start, per_day_y=args.per_day_y,
                       calib=calib, interval_method=args.interval_method,
                       with_band=not args.no_band)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

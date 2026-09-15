# -*- coding: utf-8 -*-
"""融合推理入口 — Transformer + TimesFM 加权融合预测次日全天24小时流量。

用法:
  # 自动选最后一个可用日, 预测次日
  python predict_ensemble.py

  # 指定输入 CSV
  python predict_ensemble.py --data data/input_nextday16h_20251225_35d.csv

  # 指定 α (不读 weights.json)
  python predict_ensemble.py --alpha 0.6

  # 从原始数据切片 → 预测
  python predict_ensemble.py --raw data/水厂2025年小时级汇总.csv --days 35

  # 输出 JSON 文件
  python predict_ensemble.py --out pred_ensemble.json

库调用:
  from predict_ensemble import predict_ensemble
  result = predict_ensemble("data/input_nextday16h_20251225_35d.csv")
"""
import argparse
import json
import os
import sys

# GBK 控制台兼容
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from _paths import ensure_import_paths, PROJECT_ROOT                  # noqa: E402
TRANSFORMER_PKG_DIR = ensure_import_paths(verbose=False)

from ensemble_predictor import (                                      # noqa: E402
    JunshanEnsemblePredictor, fuse,
    DEFAULT_RESULT_DIR, DEFAULT_TIMESFM_MODEL, DEFAULT_WEIGHTS_PATH,
    DEFAULT_RAW_DATA, DAY_STEPS,
)
from inference_nextday_16h import prepare_input_csv, DEFAULT_SLICE_DAYS  # type: ignore # noqa: E402

DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data",
                            "input_nextday16h_20250820_35d.csv")


def predict_ensemble(csv_path, result_dir=DEFAULT_RESULT_DIR,
                     timesfm_model_path=DEFAULT_TIMESFM_MODEL,
                     weights_path=DEFAULT_WEIGHTS_PATH, alpha=None,
                     device=None):
    """融合预测 (库调用接口)。

    Returns: dict {date, provider, unit, interval_minutes, horizon, values}
    """
    p = JunshanEnsemblePredictor(
        result_dir=result_dir, timesfm_model_path=timesfm_model_path,
        weights_path=weights_path, alpha=alpha, device=device)
    return p.predict(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="军山融合预测 (Transformer + TimesFM)")
    parser.add_argument("--data", default=None, help="输入 CSV 文件路径")
    parser.add_argument("--raw", default=DEFAULT_RAW_DATA,
                        help="原始小时级数据 (与 --days 配合使用)")
    parser.add_argument("--days", type=int, default=DEFAULT_SLICE_DAYS,
                        help="从原始数据切出的天数 (默认 35)")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR,
                        help="Transformer 结果目录")
    parser.add_argument("--timesfm_model", default=DEFAULT_TIMESFM_MODEL,
                        help="TimesFM 模型目录")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH,
                        help="融合权重文件 (weights.json)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="显式指定 α (不读 weights.json)")
    parser.add_argument("--out", default=None, help="JSON 输出文件路径")
    parser.add_argument("--no-compare", action="store_true",
                        help="跳过预测 vs 实际对比")
    args = parser.parse_args()

    # 确定输入 CSV
    if args.data:
        csv_path = args.data
    elif os.path.exists(DEFAULT_DATA):
        csv_path = DEFAULT_DATA
        print(f"[main] 使用默认输入: {csv_path}")
    else:
        print(f"[main] 默认输入不存在, 从原始数据切片...")
        csv_path = prepare_input_csv(args.raw, days=args.days)

    # 融合预测
    predictor = JunshanEnsemblePredictor(
        result_dir=args.result_dir, timesfm_model_path=args.timesfm_model,
        weights_path=args.weights, alpha=args.alpha)
    result = predictor.predict(csv_path)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print("融合预测结果:")
    print(text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已保存: {args.out}")

    # 预测 vs 实际对比 (复用 inference_nextday_16h 的对比逻辑)
    if not args.no_compare:
        _compare_with_actual(predictor, result, args.raw, csv_path)


def _compare_with_actual(predictor, result, raw_csv, fallback_csv):
    """融合预测 vs 真实值对比 (简化版)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_date = pd.Timestamp(result["date"]).normalize()
    y_pred = np.asarray(result["values"], dtype=float)

    # 读取真实值
    day_clean = None
    for src in [str(raw_csv), str(fallback_csv)]:
        if not src or not os.path.exists(src):
            continue
        df = pd.read_csv(src, encoding="utf-8-sig")
        ts_col = next((c for c in ("时间", "timestamp") if c in df.columns), None)
        if ts_col is None:
            continue
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        try:
            df_base = predictor.processor.build_base_features(df)
            df_clean_src = predictor.processor.clean_and_resample(df_base)
        except Exception:
            continue
        mask = df_clean_src.index.normalize() == target_date
        if mask.sum() > 0:
            hourly = df_clean_src["Total_Flow"].resample("h").mean().dropna()
            day_mask = hourly.index.normalize() == target_date
            day_clean = hourly.loc[day_mask]
            break

    if day_clean is None or len(day_clean) == 0:
        print(f"\n[对比] 未找到 {target_date.date()} 的实际数据, 跳过")
        return

    n = min(len(day_clean), DAY_STEPS)
    y_true = day_clean.iloc[:n].to_numpy(dtype=float)
    y_pred_n = y_pred[:n]

    mae = float(np.mean(np.abs(y_true - y_pred_n)))
    mask_mape = np.abs(y_true) > 0
    mape = float(np.mean(np.abs((y_true[mask_mape] - y_pred_n[mask_mape]) /
                                y_true[mask_mape]) * 100)) if mask_mape.sum() > 0 else float("nan")

    print(f"\n[对比] {target_date.date()} 融合预测 vs 实际 ({n} 小时):")
    print(f"       MAE = {mae:.2f} m³/h")
    print(f"       MAPE = {mape:.2f}%")

    # 画对比图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    hours = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(hours, y_true, color="#2c3e50", linewidth=1.8, marker="o", ms=4, label="实际流量")
    ax.plot(hours, y_pred_n, color="#e74c3c", linewidth=1.8, linestyle="--",
            marker="s", ms=4, label="融合预测")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours])
    ax.set_xlabel(f"时刻 (目标天 {target_date.date()})")
    ax.set_ylabel("出厂水流量 (m³/h)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_title(f"{target_date.date()} 融合预测 vs 实际   "
                 f"MAE={mae:.1f} m³/h  MAPE={mape:.1f}%")
    fig.tight_layout()
    save_dir = os.path.join(_HERE, "results")
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"compare_{target_date:%Y%m%d}.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[对比] 曲线图: {plot_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""生产口径端到端验证: 把 data/input_nextday16h_*.csv 逐个喂给 predict()。

这些就是生产环境真正会用的输入文件 (都停在 15:00), 所以最能反映线上表现。
只读, 不写任何产物。

    cd D:\\Junshan_Project\\ensemble_pkg
    python eval_production.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ensemble_predictor import (                            # noqa: E402
    JunshanEnsemblePredictor, MEDIAN_INDEX, DEFAULT_RAW_DATA)

raw = pd.read_csv(DEFAULT_RAW_DATA, encoding="utf-8-sig")
raw["时间"] = pd.to_datetime(raw["时间"])
raw = raw.set_index("时间").sort_index()

p = JunshanEnsemblePredictor(verbose=False)
print(f"α = {p.alpha:.3f}   区间方法 = {p.interval_method}")

files = sorted(glob.glob(os.path.join(os.path.dirname(DEFAULT_RAW_DATA),
                                      "input_nextday16h_*_35d.csv")))
rows, skipped = [], 0
for f in files:
    try:
        r = p.predict(f, with_interval=True)
    except Exception as e:                                  # 数据不足等
        skipped += 1
        continue
    tgt = pd.Timestamp(r["date"])
    truth = raw[raw.index.normalize() == tgt]["出厂水流量"].iloc[:24].to_numpy()
    if len(truth) < 24:
        skipped += 1
        continue

    pred = np.asarray(r["values"], dtype=float)
    q = np.asarray(r["quantiles"]["values"], dtype=float)
    lo, hi = q[0], q[-1]
    rows.append({
        "date": r["date"],
        "mae": float(np.abs(pred - truth).mean()),
        "mape": float((np.abs(pred - truth) / truth).mean() * 100),
        "in80": float(((truth >= lo) & (truth <= hi)).mean()),
        "width80": float((hi - lo).mean()),
        "median_gap": float(np.abs(q[MEDIAN_INDEX] - pred).max()),
    })

df = pd.DataFrame(rows)
print(f"\n{'=' * 72}")
print(f"生产输入文件 {len(files)} 个, 有效评估 {len(df)} 天, 跳过 {skipped} 个")
print(f"{'=' * 72}")
print(f"  MAE    : 均值 {df.mae.mean():7.1f}   中位 {df.mae.median():7.1f}   "
      f"最差 {df.mae.max():7.1f} ({df.loc[df.mae.idxmax(), 'date']})")
print(f"  MAPE   : 均值 {df.mape.mean():6.2f}%  中位 {df.mape.median():6.2f}%  "
      f"最差 {df.mape.max():6.2f}% ({df.loc[df.mape.idxmax(), 'date']})")
print(f"  80%覆盖: 均值 {df.in80.mean():6.3f}  (名义 0.800)")
print(f"  80%宽度: 均值 {df.width80.mean():7.1f} m³/h")
print(f"  不变量 |q(0.5)-values| 最大 = {df.median_gap.max():.2e}")
print(f"{'=' * 72}")
print(df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print(f"{'=' * 72}")
print(f"  结论: MAPE 中位 {df.mape.median():.2f}%, 80% 实测覆盖率 "
      f"{df.in80.mean():.3f} (名义 0.800)")
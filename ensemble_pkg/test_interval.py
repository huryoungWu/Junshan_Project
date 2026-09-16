# -*- coding: utf-8 -*-
"""概率预测 (区间) 的不变量与纯函数测试。

不需要加载模型, 秒级跑完:
    cd D:\\Junshan_Project\\ensemble_pkg
    python test_interval.py
"""
import os
import sys

import numpy as np
import pandas as pd

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ensemble_predictor import (                            # noqa: E402
    JunshanEnsemblePredictor, fuse, fuse_quantiles, anchor_median,
    enforce_monotone, qf_columns, day_interval,
    QUANTILE_LEVELS, MEDIAN_INDEX, INTERVAL_METHODS,
)
from fit_weights import (                                   # noqa: E402
    interval_scores, calibrate_residual_quantiles, residual_quantile_table,
)

RNG = np.random.default_rng(20260916)
NH = 24          # 每天小时数
ND = 45          # 造 45 天回测数据
FAILED = []


def check(name, cond, detail=""):
    mark = "OK  " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def make_case(shift=0.0):
    """造一组 (point_t, point_f, q_t, q_f), 形状与真实推理一致。"""
    lv = np.array(QUANTILE_LEVELS)
    hours = np.tile(np.arange(NH), ND)
    pt = 2500 + 600 * np.sin(2 * np.pi * hours / 24) + RNG.normal(0, 60, len(hours))
    pf = pt + shift + RNG.normal(0, 80, len(hours))
    # Transformer 残差校准: 中位数为 0 的逐小时分位表
    g_t = (lv - 0.5)[None, :] * 420 * \
        (1 + 0.3 * np.sin(2 * np.pi * np.arange(NH) / 24))[:, None]
    q_t = pt[:, None] + g_t[hours]
    # TimesFM 原生分位数
    q_f = pf[:, None] + (lv - 0.5)[None, :] * 480
    return pt, pf, q_t, q_f, hours


def main():
    alpha = 0.32
    pt, pf, q_t, q_f, hours = make_case(shift=40.0)
    y = fuse(alpha, pt, pf)

    print("\n【1】区间融合方法")
    for method in INTERVAL_METHODS:
        q_e = fuse_quantiles(alpha, q_t, q_f, method)
        check(f"{method}: 形状保持", q_e.shape == q_t.shape, str(q_e.shape))
        check(f"{method}: 分位轴单调", np.all(np.diff(q_e, axis=-1) >= 0))
        gap = float(np.max(np.abs(q_e[:, MEDIAN_INDEX] - y)))
        if method == "vincentization":
            check(f"{method}: 中位数 == 点预测 (精确)", gap == 0.0, f"gap={gap}")
        else:
            check(f"{method}: 中位数不等于点预测 (线性池的固有性质)",
                  gap > 0, f"gap={gap:.2f} -> 需要 anchor_median")

    print("\n【2】anchor_median 同时保住单调性与中位数")
    for method in INTERVAL_METHODS:
        q_e = np.maximum(fuse_quantiles(alpha, q_t, q_f, method), 0.0)
        qa = anchor_median(q_e, np.maximum(y, 0.0))
        check(f"{method}: 锚定后中位数 == 点预测",
              np.allclose(qa[:, MEDIAN_INDEX], np.maximum(y, 0.0)))
        check(f"{method}: 锚定后仍单调", np.all(np.diff(qa, axis=-1) >= 0))
        left = np.abs(qa[:, :MEDIAN_INDEX] - q_e[:, :MEDIAN_INDEX]).max()
        right = np.abs(qa[:, MEDIAN_INDEX + 1:] - q_e[:, MEDIAN_INDEX + 1:]).max()
        check(f"{method}: 锚定扰动仅在需要处", left >= 0 and right >= 0,
              f"左 max={left:.2f} 右 max={right:.2f}")

    print("\n【3】相同分布 -> 融合结果不变")
    for method in INTERVAL_METHODS:
        r = fuse_quantiles(alpha, q_t, q_t, method)
        check(f"{method}: q_T == q_F 时还原自身",
              np.allclose(r, q_t, atol=1e-9))

    print("\n【4】enforce_monotone")
    bad = np.array([[5.0, 1.0, 3.0, 2.0, 4.0]])
    check("乱序输入被修成单调", np.all(np.diff(enforce_monotone(bad), axis=-1) >= 0))

    print("\n【5】残差校准: 中位数中心化 + 回退")
    pred = pt.copy()
    y_true = y.copy()
    hourly, meta = calibrate_residual_quantiles(
        {"transformer": pred}, y_true, hours, np.ones(len(pt), bool),
        min_count=40, smooth=True)
    tab = hourly["transformer"]
    check("校准表形状 (24, 9)", tab.shape == (NH, len(QUANTILE_LEVELS)), str(tab.shape))
    check("中位数列恒为 0 (点预测不被改动)", np.allclose(tab[:, MEDIAN_INDEX], 0.0))
    check("校准表分位轴单调", np.all(np.diff(tab, axis=-1) >= 0))
    q_back = residual_quantile_table({"hourly": hourly}, "transformer", pred, hours)
    check("套回后中位数 == 点预测", np.allclose(q_back[:, MEDIAN_INDEX], pred))

    _, meta_lo = calibrate_residual_quantiles(
        {"transformer": pred}, y_true, hours, np.ones(len(pt), bool),
        min_count=10 ** 6, smooth=False)
    check("样本不足时全部回退全局池化",
          meta_lo["transformer"]["n_fallback_hours"] == NH,
          f"fallback={meta_lo['transformer']['n_fallback_hours']}/{NH}")

    print("\n【6】区间指标")
    # 分位轴拉得极开 (跨度 ±4e4), 远大于 N(0,100) 的散布 -> 每个 level 都应 100% 覆盖
    lv = np.array(QUANTILE_LEVELS)
    lo_cov = np.tile((lv - 0.5)[None, :] * 100000, (200, 1))
    s = interval_scores(RNG.normal(0, 100, 200), lo_cov)
    check("过宽区间 -> 覆盖率 100%",
          all(r["picp"] == 1.0 for r in s["levels"]),
          str([r["picp"] for r in s["levels"]]))
    tight = np.tile(np.zeros(9)[None, :], (200, 1))
    s2 = interval_scores(RNG.normal(0, 100, 200), tight)
    check("零宽区间 -> 覆盖率 0%", all(r["picp"] == 0.0 for r in s2["levels"]))
    check("CRPS 有限且为正", np.isfinite(s["crps"]) and s["crps"] > 0,
          f"crps={s['crps']:.1f}")

    print("\n【7】回测 CSV 列契约 + day_interval 重建")
    cols = qf_columns()
    check("qf_ 列名与 levels 对应", cols[0] == "qf_10" and cols[-1] == "qf_90",
          str(cols))
    day = pd.DataFrame({
        "pred_transformer": pt[:NH], "pred_timesfm": pf[:NH],
        **{c: q_f[:NH, j] for j, c in enumerate(cols)},
    }, index=pd.date_range("2025-11-20", periods=NH, freq="h"))
    calib = {"hourly": {"transformer": tab.tolist()}}
    q_day = day_interval(day, alpha, calib)
    check("day_interval 形状 (24, 9)", q_day.shape == (NH, 9), str(q_day.shape))
    check("day_interval 单调", np.all(np.diff(q_day, axis=-1) >= 0))
    check("day_interval 中位数 == 当天点预测",
          np.allclose(q_day[:, MEDIAN_INDEX], fuse(alpha, pt[:NH], pf[:NH])))
    check("缺校准表时返回 None", day_interval(day, alpha, None) is None)

    print("\n【8】TimesFM 输出对齐 (历史 bug: 固定取 [0:24], 15:00 截止时错位 8 小时)")
    off = JunshanEnsemblePredictor.timesfm_day_offset
    for ctx_hour, expect in ((23, 0), (22, 1), (15, 8), (16, 7), (0, 23)):
        got = off(ctx_hour)
        check(f"上下文末尾 {ctx_hour:02d}:00 -> 取第 {expect} 步起",
              got == expect, f"got={got}")
    check("生产口径 15:00 截止 == 缺口 8 小时",
          off(15) == 8 and max(0, 23 - 15) == 8)

    print("\n" + "=" * 62)
    if FAILED:
        print(f" 失败 {len(FAILED)} 项: {FAILED}")
        return 1
    print(" 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
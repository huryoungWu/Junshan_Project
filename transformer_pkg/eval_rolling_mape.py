#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滚动窗口预测评估 — 用前 W 天预测次日, 与真实值比较 MAPE / MAE / RMSE。

示例 (W=3, 默认): 一个 5 天的 CSV →
  窗口 1: 第 1/2/3 天 → 预测第 4 天, 与真实第 4 天比较
  窗口 2: 第 2/3/4 天 → 预测第 5 天, 与真实第 5 天比较
  共 len(天数) - W 个窗口, 以此类推。

说明:
  - 复用 inference_transformer.FlowPredictor 的完整推理链路 (清洗/特征/模型完全一致);
    每个窗口把输入截断到目标日前一天 23:59, 交给 predict() → 输出恰为目标日
    全部整点 (按训练 resample_freq)。
  - 真实值取与训练/推理完全相同的清洗管线 (DataProcessor.clean_and_resample) 产出的
    Total_Flow (出厂水流量, 整点均值), 保证预测与实际口径一致。
  - 只比较 Total_Flow: predict_pressure 是固定分时压力时段表 + 随机误差, 不是模型预测, 无评估意义。
  - MAPE 口径与项目训练一致: 仅对真实值 > --mape_min_actual (默认 10 m3/h) 的样本计算,
    避免除以小真值导致爆炸。

用法:
  python eval_rolling_mape.py --data input_5days.csv
  python eval_rolling_mape.py --data input.csv --window_days 3 --mape_min_actual 10 \
      --out_prefix eval_rolling_mape --result_dir results/junshan_L1D_P24H_1h_itransformer_raw

输出:
  <out_prefix>_results.csv   — 每窗口一行: 输入日期区间 / 预测日期 / 点数 / MAPE / MAE / RMSE
  <out_prefix>_detail.csv    — 每窗口逐点: 时刻 / 真实值 / 预测值 / 绝对误差 / 相对误差
  <out_prefix>_byslot.csv    — 跨窗口聚合: 每个时间点 (HH:MM 槽位) 的 MAPE 分布
  <out_prefix>_day_slot.csv  — 每天全部整点 (槽位) 的 APE / AE 统计
      (n_valid / ae / ape / actual_mean / pred_mean)
  <plot_dir>/day_<日期>_metrics.png — 每天一张图: 横轴全部整点槽位,
      上图左轴 AE (m3/h) 柱状 + 右轴 APE (%) 折线 + 5% 阈值虚线 (针对 APE);
      中图为 AE 柱状 + 真实值均值折线 (判断大误差是否来自大流量);
      下图为预测值 vs 真实值折线对比。
  <plot_dir>/daily_mape.png — 一张汇总图: 横轴每天预测日期, 纵轴每日 MAPE (%)
      折线 + 5% 阈值虚线。
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")        # 无显示环境也能直接保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 保证能从本目录导入共享模块 (模型/数据管线, 与 inference_transformer 一致)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_transformer import FlowPredictor, DEFAULT_RESULT_DIR
from data_processing import needed_csv_columns

# 与推理脚本一致的时间列候选
TIME_COLUMNS = ("时间", "timestamp")


def make_slot_labels(freq_minutes):
    """按训练频率生成一天内所有槽位标签 (60min → 24 个整点, 30min → 48 个)。"""
    return [f"{h:02d}:{m:02d}"
            for h in range(24) for m in range(0, 60, freq_minutes)]


def load_raw(csv_path, encoding="utf-8-sig"):
    """读取原始 CSV → 以 DatetimeIndex 索引、按时间升序的 DataFrame。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"输入 CSV 不存在: {csv_path}")
    # 方案B: 只读管线必需列 (与训练 load_raw 口径一致), 大 CSV 内存峰值骤降
    header = pd.read_csv(csv_path, encoding=encoding, nrows=0).columns
    df = pd.read_csv(csv_path, encoding=encoding,
                     usecols=needed_csv_columns(header))
    if not isinstance(df.index, pd.DatetimeIndex):
        for ts_col in TIME_COLUMNS:
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col])
                df = df.set_index(ts_col)
                break
        else:
            raise ValueError(
                f"输入数据必须包含时间列 ({' / '.join(TIME_COLUMNS)}) 或 DatetimeIndex 索引")
    return df.sort_index()


def get_actual_series(predictor, raw_df):
    """真实 Total_Flow (整点级): 走与训练/推理完全相同的清洗 + 重采样管线。

    clean_and_resample 内部: 出厂水流量 Hampel 清洗 → 物理裁剪 → 整点均值;
    流量为 NaN 的 bin 保持 NaN (不虚构)。
    """
    df_base = predictor.processor.build_base_features(raw_df)
    df_clean = predictor.processor.clean_and_resample(df_base)
    return df_clean["Total_Flow"]


def compute_metrics(actual, pred, min_actual=10.0):
    """对齐真实/预测序列, 计算 MAPE/MAE/RMSE (仅对真实值 > min_actual 的样本)。

    返回 (n, mape_pct, mae, rmse, mask): mask 用于逐点明细表的相对误差列。
    """
    df = pd.concat([actual.rename("actual"), pred.rename("pred")], axis=1).dropna()
    if df.empty:
        return 0, float("nan"), float("nan"), float("nan"), df.index
    mask = df["actual"] > min_actual
    df_valid = df[mask]
    n = len(df_valid)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan"), mask
    err = (df_valid["pred"] - df_valid["actual"]).abs()
    mape = float((err / df_valid["actual"]).mean() * 100.0)
    mae = float(err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    return n, mape, mae, rmse, mask


def plot_day_metrics(predict_date, slot_df, plot_dir):
    """绘制单日 48 个时间点的 APE / AE 图, 保存到 <plot_dir>/day_<日期>_metrics.png。

    上图: 横轴 48 个时间点 (00:00~23:30), 左轴 AE (m3/h) 柱状,
          右轴 APE (%) 折线 + 5% 阈值虚线 (阈值针对 APE);
    中图: 同一时间点的 AE 柱状 + 真实均值折线, 判断大误差是否来自大流量;
    下图: 预测值 vs 真实值折线对比。
    slot_df: 长表 [slot, n_valid, ae, ape, actual_mean, pred_mean], 按 48 槽顺序。
    返回保存的 PNG 路径。
    """
    os.makedirs(plot_dir, exist_ok=True)
    slots = slot_df["slot"].astype(str).tolist()
    ae = slot_df["ae"].to_numpy()
    ape = slot_df["ape"].to_numpy()
    actual = slot_df["actual_mean"].to_numpy()
    pred = slot_df["pred_mean"].to_numpy()

    fig, (ax1, ax3, ax5) = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f"Day Metrics {predict_date}", fontsize=14, y=0.99)

    # ── 上图: AE 柱 + APE 折线 + 5% 阈值线 ──
    ax1.bar(slots, ae, color="#6baed6", label="AE (m3/h)")
    ax1.set_ylabel("AE (m3/h)")
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax1.grid(axis="y", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(slots, ape, color="#d62728", marker="o", markersize=4,
             linewidth=1.5, label="APE (%)")
    ax2.axhline(5.0, color="black", linestyle="--", linewidth=1.2,
                label="Threshold 5%")
    ax2.set_ylabel("APE (%)")
    ax2.set_ylim(bottom=0)
    ax1.set_title("MAPE / AE per slot", fontsize=11)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # ── 下图: AE 柱 + 真实均值折线 ──
    ax3.bar(slots, ae, color="#6baed6", alpha=0.6, label="AE (m3/h)")
    ax3.set_ylabel("AE (m3/h)")
    ax3.set_ylim(bottom=0)
    ax3.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax3.grid(axis="y", alpha=0.3)
    ax4 = ax3.twinx()
    ax4.plot(slots, actual, color="#fd8d3c", marker="s", markersize=4,
             linewidth=1.5, label="Actual (m3/h)")
    ax4.set_ylabel("Actual (m3/h)")
    ax4.set_ylim(bottom=0)
    ax3.set_title("AE vs Actual", fontsize=11)
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="upper left", fontsize=9)

    # ── 下图: 预测值 vs 真实值折线对比 ──
    ax5.plot(slots, actual, color="#fd8d3c", marker="o", markersize=4,
             linewidth=1.5, label="Actual (m3/h)")
    ax5.plot(slots, pred, color="#3182bd", marker="s", markersize=4,
             linewidth=1.5, label="Pred (m3/h)")
    ax5.set_ylabel("Flow (m3/h)")
    ax5.set_ylim(bottom=0)
    ax5.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax5.grid(axis="y", alpha=0.3)
    ax5.set_title("Pred vs Actual", fontsize=11)
    ax5.legend(loc="upper left", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png_path = os.path.join(plot_dir, f"day_{predict_date}_metrics.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def plot_daily_mape(df_results, plot_dir):
    """绘制所有窗口的每日 MAPE 时序图, 保存到 <plot_dir>/daily_mape.png。

    横轴: 每天预测日期; 纵轴: 当日 MAPE (%); 附 5% 阈值虚线 (针对 MAPE)。
    无效窗口 (无有效样本, MAPE 为 NaN) 自动跳过。返回保存的 PNG 路径。
    """
    os.makedirs(plot_dir, exist_ok=True)
    d = df_results[df_results["mape_pct"].notna()].copy()
    dates = pd.to_datetime(d["predict_date"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, d["mape_pct"], color="#3182bd", marker="o", markersize=5,
            linewidth=1.5, label="Daily MAPE (%)")
    ax.axhline(5.0, color="black", linestyle="--", linewidth=1.2,
               label="Threshold 5%")
    ax.set_xlabel("Predict Date")
    ax.set_ylabel("MAPE (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Daily MAPE over Rolling Windows", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    for x, y in zip(dates, d["mape_pct"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)

    fig.tight_layout()
    png_path = os.path.join(plot_dir, "daily_mape.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="滚动窗口预测评估: 前 W 天预测次日, 与真实值比较 MAPE/MAE/RMSE")
    parser.add_argument("--data", required=True,
                        help="输入原始数据 CSV 路径 (与训练数据格式一致, 需 ≥ window_days+1 天)")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR,
                        help="训练结果目录 (含 best_seq2seq_model.pth 与 scaler.pkl)")
    parser.add_argument("--window_days", type=int, default=3,
                        help="每个窗口使用的输入天数 W (默认 3, 即 123 预测 4, 234 预测 5, ...)")
    parser.add_argument("--mape_min_actual", type=float, default=10.0,
                        help="MAPE 仅对真实值 > 该阈值 (m3/h) 的样本计算 (默认 10.0)")
    parser.add_argument("--out_prefix", default="eval_rolling_mape",
                        help="输出文件前缀 (默认 eval_rolling_mape → *_results.csv / *_detail.csv)")
    parser.add_argument("--plot_dir", default="day_metric_plots",
                        help="每日 APE/AE 图保存目录 (默认 day_metric_plots, 每天一张 day_<日期>_metrics.png)")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV 编码 (默认 utf-8-sig)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.window_days < 1:
        raise ValueError(f"--window_days 必须 ≥ 1, 实际 {args.window_days}")

    predictor = FlowPredictor(args.result_dir)
    slot_labels = make_slot_labels(predictor.freq_minutes)   # 槽位横轴: 按训练频率
    raw = load_raw(args.data, args.encoding)
    print(f"[eval] 输入数据: {raw.index.min()} ~ {raw.index.max()}, {len(raw)} 行")

    actual = get_actual_series(predictor, raw)
    days = sorted(set(actual.index.date))
    n_windows = len(days) - args.window_days
    if n_windows < 1:
        print(f"[eval] 数据共 {len(days)} 个自然日, 不足 window_days+1 = {args.window_days + 1} 天, "
              f"无可滚动窗口 (建议给 ≥5~7 天数据)")
        return 1
    print(f"[eval] 共 {len(days)} 个自然日 → {n_windows} 个滚动窗口 (输入 {args.window_days} 天 → 预测次日)")

    results_rows, detail_rows, day_slot_rows = [], [], []
    for k in range(n_windows):
        target_date = days[k + args.window_days]
        cutoff = pd.Timestamp(target_date)              # 目标日 00:00
        train_raw = raw[raw.index < cutoff]             # 截断: 只用目标日之前的数据
        predict_date = str(target_date)
        input_dates = f"{days[k]} ~ {days[k + args.window_days - 1]}"
        note = ""

        print(f"\n[窗口 {k + 1}/{n_windows}] 输入 {input_dates} → 预测 {predict_date}")
        try:
            pred = predictor.predict(train_raw)         # DataFrame: DatetimeIndex + Total_Flow
        except Exception as exc:
            note = f"失败: {exc}"
            print(f"[窗口 {k + 1}] 预测失败: {exc}")
            results_rows.append({
                "window": k + 1, "input_dates": input_dates, "predict_date": predict_date,
                "n_points": 0, "mape_pct": float("nan"), "mae": float("nan"),
                "rmse": float("nan"), "note": note,
            })
            continue

        target_actual = actual.loc[str(target_date)]
        n, mape, mae, rmse, mask = compute_metrics(target_actual, pred["Total_Flow"],
                                                   args.mape_min_actual)
        print(f"  真实 {target_actual.index.min()} ~ {target_actual.index.max()} "
              f"({len(target_actual.dropna())} 点), "
              f"预测 {pred.index.min()} ~ {pred.index.max()} ({len(pred)} 点)")
        print(f"  有效样本 {n} 点 (真实值 > {args.mape_min_actual:g}): "
              f"MAPE = {mape:.2f}%  MAE = {mae:.2f} m3/h  RMSE = {rmse:.2f} m3/h")
        results_rows.append({
            "window": k + 1, "input_dates": input_dates, "predict_date": predict_date,
            "n_points": n, "mape_pct": mape, "mae": mae, "rmse": rmse, "note": note,
        })

        aligned = pd.concat(
            [target_actual.rename("actual"), pred["Total_Flow"].rename("pred")],
            axis=1)
        err = (aligned["pred"] - aligned["actual"]).abs()
        pct = err / aligned["actual"] * 100.0
        pct = pct.where(mask.reindex(aligned.index).fillna(False))   # 相对误差仅对有效样本
        for ts, row in aligned.iterrows():
            detail_rows.append({
                "window": k + 1, "predict_date": predict_date,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "actual": row["actual"], "pred": row["pred"],
                "abs_err": err.loc[ts], "pct_err": pct.loc[ts],
            })

        # ── 单日全部整点槽位的 APE / AE 统计 + 每日一图 ──
        # 与 MAPE 口径一致: 仅真实值 > --mape_min_actual 的有效样本参与 APE / AE
        daily = aligned.assign(abs_err=err, pct_err=pct)
        day_valid = daily[daily["pct_err"].notna()]
        if day_valid.empty:
            print(f"  [单日统计] 无有效样本 (真实值 > {args.mape_min_actual:g}), 跳过绘图")
        else:
            day_slots = day_valid.groupby(day_valid.index.strftime("%H:%M")).agg(
                n_valid=("pct_err", "size"),
                ae=("abs_err", "mean"),
                ape=("pct_err", "mean"),
                actual_mean=("actual", "mean"),
                pred_mean=("pred", "mean"),
            ).reindex(slot_labels)
            day_slots["slot"] = slot_labels
            day_slots["n_valid"] = day_slots["n_valid"].fillna(0).astype(int)
            for _, r in day_slots.iterrows():
                day_slot_rows.append({
                    "date": predict_date, "slot": r["slot"], "n_valid": r["n_valid"],
                    "ae": r["ae"], "ape": r["ape"],
                    "actual_mean": r["actual_mean"], "pred_mean": r["pred_mean"],
                })
            ape_ok = day_slots["ape"].dropna()
            ae_ok = day_slots["ae"].dropna()
            png = plot_day_metrics(predict_date, day_slots, args.plot_dir)
            print(f"  单日 {len(slot_labels)} 时间点: APE 均值 {ape_ok.mean():.2f}%, "
                  f"最大 {ape_ok.max():.2f}% ({day_slots.loc[ape_ok.idxmax(), 'slot']}), "
                  f"AE 均值 {ae_ok.mean():.2f} m3/h → {png}")

    # ── 汇总 ──
    df_results = pd.DataFrame(results_rows)
    df_detail = pd.DataFrame(detail_rows)
    df_results.to_csv(f"{args.out_prefix}_results.csv", index=False, encoding="utf-8-sig")
    df_detail.to_csv(f"{args.out_prefix}_detail.csv", index=False, encoding="utf-8-sig")
    if day_slot_rows:
        pd.DataFrame(day_slot_rows).to_csv(f"{args.out_prefix}_day_slot.csv",
                                           index=False, encoding="utf-8-sig")

    # ── 按一天中的时刻 (HH:MM 槽位) 聚合: 每个时间点的 MAPE 分布 (跨窗口) ──
    if not df_detail.empty:
        df_detail["slot"] = pd.to_datetime(df_detail["timestamp"]).dt.strftime("%H:%M")
        slot_valid = df_detail[df_detail["pct_err"].notna()].copy()
        if not slot_valid.empty:
            byslot = slot_valid.groupby("slot").agg(
                n=("pct_err", "size"),
                mape_mean=("pct_err", "mean"),
                mape_median=("pct_err", "median"),
                mape_min=("pct_err", "min"),
                mape_max=("pct_err", "max"),
                mae_mean=("abs_err", "mean"),
            ).reindex(sorted(slot_valid["slot"].unique()))
            byslot.to_csv(f"{args.out_prefix}_byslot.csv", encoding="utf-8-sig")
            print("\n" + "=" * 72)
            print(f"按时间点 (HH:MM 槽位) 的 MAPE 分布 (跨 {len(slot_valid['window'].unique())} 个窗口):")
            print(byslot.round(2).to_string())
            worst = byslot.sort_values("mape_mean", ascending=False).head(5)
            print("-" * 72)
            print("MAPE 最高的 5 个时间点(最难预测):")
            print(worst.round(2).to_string())

    # ── 每日 MAPE 汇总图: 横轴预测日期, 纵轴当日 MAPE (%) + 5% 阈值线 ──
    if not df_results.empty:
        png = plot_daily_mape(df_results, args.plot_dir)
        print(f"每日 MAPE 汇总图已保存: {png}")

    ok = df_results[df_results["n_points"] > 0]
    print("\n" + "=" * 72)
    print(f"滚动评估汇总 (窗口数 {len(df_results)}):")
    print(df_results[["window", "input_dates", "predict_date", "n_points",
                      "mape_pct", "mae", "rmse"]].to_string(index=False))
    if len(ok) > 0:
        # 总 MAPE = 全部有效点混合计算; 窗口均值 = 各窗口 MAPE 的算术平均
        pooled_n = int(df_detail["window"].isin(ok["window"]).sum())
        pooled_mae = float(df_detail.loc[df_detail["pct_err"].notna(), "abs_err"].mean())
        pooled_mape = float(df_detail.loc[df_detail["pct_err"].notna(), "pct_err"].mean())
        print("-" * 72)
        print(f"总体 (全部窗口混合):  有效点 {pooled_n}  MAPE = {pooled_mape:.2f}%  "
              f"MAE = {pooled_mae:.2f} m3/h")
        print(f"窗口平均:              MAPE = {ok['mape_pct'].mean():.2f}%  "
              f"MAE = {ok['mae'].mean():.2f} m3/h  RMSE = {ok['rmse'].mean():.2f} m3/h")
    print("=" * 72)
    print(f"明细已保存: {args.out_prefix}_results.csv / {args.out_prefix}_detail.csv")
    if day_slot_rows:
        print(f"单日时间点统计已保存: {args.out_prefix}_day_slot.csv")
        print(f"每日一图已保存: {args.plot_dir}/day_<日期>_metrics.png "
              f"(共 {len(set(r['date'] for r in day_slot_rows))} 天)")
        print(f"每日 MAPE 汇总图已保存: {args.plot_dir}/daily_mape.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EWMA alpha 参数搜索: 在已有预测结果上遍历多个 alpha, 找到最优值。

前置条件: 先运行 eval_random_window_ewma.py 生成 eval_ewma_all_predictions.csv

用法:
  python tune_ewma_alpha.py
  python tune_ewma_alpha.py --start_date 2025-01-08
  python tune_ewma_alpha.py --alpha_min 0.01 --alpha_max 0.5 --alpha_step 0.01
"""

import os
import sys
import argparse
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def calc_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    thr = 0.1 * np.abs(y_true).max()
    mask = np.abs(y_true) >= thr
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) /
                           (y_true[mask] + 1e-8))) * 100
            if mask.sum() > 0 else 0.0)
    return mae, rmse, mape


def ewma_correct_day(pred_day, true_day, ewma_val, alpha):
    """对一天逐小时 EWMA 修正, 返回 (corrected, new_ewma_val)。"""
    corrected = np.zeros_like(pred_day)
    for h in range(len(pred_day)):
        corrected[h] = pred_day[h] + ewma_val
        r = true_day[h] - pred_day[h]
        ewma_val = alpha * r + (1 - alpha) * ewma_val
    return corrected, ewma_val


def main():
    parser = argparse.ArgumentParser(description="EWMA alpha 参数搜索")
    parser.add_argument("--predictions", default=None,
                        help="预测 CSV 路径 (默认 eval_ewma_all_predictions.csv)")
    parser.add_argument("--alpha_min", type=float, default=0.01, help="alpha 最小值")
    parser.add_argument("--alpha_max", type=float, default=0.50, help="alpha 最大值")
    parser.add_argument("--alpha_step", type=float, default=0.01, help="alpha 步长")
    args = parser.parse_args()

    pred_path = args.predictions or os.path.join(HERE, "eval_ewma_all_predictions.csv")
    if not os.path.exists(pred_path):
        print(f"错误: 找不到 {pred_path}")
        print(f"请先运行: python eval_random_window_ewma.py --start_date 2025-01-08 --all_days")
        return

    print(f"加载预测: {pred_path}")
    df = pd.read_csv(pred_path)
    print(f"数据量: {len(df)} 行, {df['date'].nunique()} 天")

    # 按天分组
    days = sorted(df["date"].unique())
    n_days = len(days)
    predict_steps = len(df[df["date"] == days[0]])

    # 预提取每天的 pred 和 true
    day_preds = []
    day_trues = []
    for d in days:
        day_df = df[df["date"] == d]
        day_preds.append(day_df["pred"].values)
        day_trues.append(day_df["true"].values)

    # 原始指标
    all_true = np.concatenate(day_trues)
    all_orig = np.concatenate(day_preds)
    o_mae, o_rmse, o_mape = calc_metrics(all_true, all_orig)

    print(f"\n原始: MAE={o_mae:.2f}  RMSE={o_rmse:.2f}  MAPE={o_mape:.2f}%")
    print(f"\n{'='*70}")
    print(f" 搜索 alpha [{args.alpha_min:.2f} ~ {args.alpha_max:.2f}, step={args.alpha_step:.2f}]")
    print(f"{'='*70}")

    # 遍历 alpha
    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step / 2, args.alpha_step)
    results = []

    for alpha in alphas:
        ewma_val = 0.0
        all_corrected = []
        for day in range(n_days):
            corr, ewma_val = ewma_correct_day(day_preds[day], day_trues[day], ewma_val, alpha)
            all_corrected.append(corr)

        all_corr = np.concatenate(all_corrected)
        c_mae, c_rmse, c_mape = calc_metrics(all_true, all_corr)
        mae_imp = (1 - c_mae / o_mae) * 100
        rmse_imp = (1 - c_rmse / o_rmse) * 100
        mape_imp = (1 - c_mape / o_mape) * 100

        results.append({
            "alpha": alpha,
            "mae": c_mae, "rmse": c_rmse, "mape": c_mape,
            "mae_imp": mae_imp, "rmse_imp": rmse_imp, "mape_imp": mape_imp,
        })

    # 打印结果
    print(f"\n  {'alpha':<8}{'MAE':<10}{'MAE提升':<12}{'RMSE':<10}{'RMSE提升':<12}{'MAPE%':<10}{'MAPE提升':<12}")
    print(f"  {'-'*74}")
    for r in results:
        print(f"  {r['alpha']:<8.2f}{r['mae']:<10.2f}{r['mae_imp']:>+10.2f}%"
              f"{r['rmse']:<10.2f}{r['rmse_imp']:>+10.2f}%"
              f"{r['mape']:<10.2f}{r['mape_imp']:>+10.2f}%")

    # 找最优
    best_mae = min(results, key=lambda r: r["mae"])
    best_rmse = min(results, key=lambda r: r["rmse"])
    best_mape = min(results, key=lambda r: r["mape"])

    print(f"\n{'='*70}")
    print(f" 最优 alpha")
    print(f"{'='*70}")
    print(f"  MAE  最优: alpha={best_mae['alpha']:.2f}  "
          f"MAE={best_mae['mae']:.2f}  提升={best_mae['mae_imp']:+.2f}%")
    print(f"  RMSE 最优: alpha={best_rmse['alpha']:.2f}  "
          f"RMSE={best_rmse['rmse']:.2f}  提升={best_rmse['rmse_imp']:+.2f}%")
    print(f"  MAPE 最优: alpha={best_mape['alpha']:.2f}  "
          f"MAPE={best_mape['mape']:.2f}%  提升={best_mape['mape_imp']:+.2f}%")

    # 综合最优 (MAE+RMSE+MAPE 排名之和最小)
    for r in results:
        r["rank_mae"] = sorted(results, key=lambda x: x["mae"]).index(r) + 1
        r["rank_rmse"] = sorted(results, key=lambda x: x["rmse"]).index(r) + 1
        r["rank_mape"] = sorted(results, key=lambda x: x["mape"]).index(r) + 1
        r["rank_sum"] = r["rank_mae"] + r["rank_rmse"] + r["rank_mape"]

    best_overall = min(results, key=lambda r: r["rank_sum"])
    print(f"\n  综合最优: alpha={best_overall['alpha']:.2f}  "
          f"(排名和={best_overall['rank_sum']}, "
          f"MAE排名={best_overall['rank_mae']}, "
          f"RMSE排名={best_overall['rank_rmse']}, "
          f"MAPE排名={best_overall['rank_mape']})")
    print(f"    MAE={best_overall['mae']:.2f} ({best_overall['mae_imp']:+.2f}%)  "
          f"RMSE={best_overall['rmse']:.2f} ({best_overall['rmse_imp']:+.2f}%)  "
          f"MAPE={best_overall['mape']:.2f}% ({best_overall['mape_imp']:+.2f}%)")

    # 保存
    result_df = pd.DataFrame(results)
    out_path = os.path.join(HERE, "ewma_alpha_search.csv")
    result_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n搜索结果已保存: {out_path}")

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, metric, title in zip(axes, ["mae", "rmse", "mape"], ["MAE", "RMSE", "MAPE%"]):
            vals = [r[metric] for r in results]
            imps = [r[f"{metric}_imp"] for r in results]
            ax2 = ax.twinx()
            ax.plot(alphas, vals, "o-", color="#2c3e50", markersize=3, linewidth=1.5, label=title)
            ax2.bar(alphas, imps, width=args.alpha_step * 0.8, alpha=0.3,
                    color=["#27ae60" if v >= 0 else "#e74c3c" for v in imps], label="提升%")
            ax2.axhline(y=0, color="black", linewidth=0.5)
            ax.set_xlabel("alpha")
            ax.set_ylabel(title)
            ax2.set_ylabel("提升 (%)")
            ax.set_title(title)
            # 标记最优点
            best_r = min(results, key=lambda r: r[metric])
            ax.axvline(x=best_r["alpha"], color="red", linestyle="--", alpha=0.5)
            ax.plot(best_r["alpha"], best_r[metric], "r*", markersize=12)

        fig.suptitle(f"EWMA Alpha 搜索 ({n_days} 天)", fontsize=14)
        fig.tight_layout()
        fig_path = os.path.join(HERE, "ewma_alpha_search.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"搜索图已保存: {fig_path}")
    except Exception as e:
        print(f"画图跳过: {e}")


if __name__ == "__main__":
    main()

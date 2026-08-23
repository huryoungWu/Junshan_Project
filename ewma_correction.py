"""
离线 EWMA 残差修正脚本

纯离线批处理：读取模型预测值 + 历史真实值，用 EWMA 对残差做修正。
不依赖 PyTorch / 模型加载，仅 pandas + numpy。

用法示例:
    python ewma_correction.py --predictions predictions.csv --true_values true_values.csv
    python ewma_correction.py --predictions predictions.csv --true_values true_values.csv --alpha 0.3 --date 2025-10-01
"""

import os
import sys
import argparse
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# GBK 控制台兼容
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ==================== 核心修正逻辑 ====================

def ewma_update(ewma_val, residual, alpha):
    """更新 EWMA 状态: ewma_val = alpha * residual + (1 - alpha) * ewma_val"""
    return alpha * residual + (1 - alpha) * ewma_val


def clip_residual(residual, hist_std, clip_factor=3.0):
    """异常值裁剪: 残差绝对值超过 clip_factor * std 时截断"""
    threshold = clip_factor * hist_std
    if abs(residual) > threshold:
        return math.copysign(threshold, residual)
    return residual


# ==================== 主处理流程 ====================

def _update_ewma_for_pair(site_id, true_val, pred_val, site_state):
    """对单个 (true, pred) 对更新 EWMA 状态, 返回 new_ewma。"""
    residual = true_val - pred_val
    old_ewma = site_state["ewma_val"]

    # 异常值裁剪
    if site_state["residual_count"] >= 2:
        residual = clip_residual(residual, site_state["residual_std"])

    # 更新 EWMA
    new_ewma = ewma_update(old_ewma, residual, site_state["alpha"])

    # 在线更新残差标准差
    n = site_state["residual_count"]
    old_std = site_state["residual_std"]
    if n == 0:
        new_std = 0.0
    else:
        new_std = math.sqrt(((n - 1) * old_std ** 2 + (residual - old_ewma) ** 2) / n) if n > 1 else abs(residual)

    site_state["ewma_val"] = new_ewma
    site_state["residual_count"] = n + 1
    site_state["residual_std"] = new_std
    return new_ewma


def run_ewma_correction(predictions_path, true_values_path, state_path, output_path,
                        alpha_default, target_date, warmup_path=None):
    print("=" * 60)
    print("离线 EWMA 残差修正 - 手动运行")
    print("=" * 60)

    # ── 1. 读取 EWMA 状态 ──
    if os.path.exists(state_path):
        df_state = pd.read_csv(state_path)
        print(f"读取状态文件: {state_path} ({len(df_state)} 个站点)")
    else:
        df_state = pd.DataFrame(columns=["site_id", "ewma_val", "alpha", "residual_count", "residual_std"])
        print(f"状态文件不存在，将初始化: {state_path}")

    # 确保列存在（兼容旧格式）
    for col, default in [("ewma_val", 0.0), ("alpha", alpha_default),
                         ("residual_count", 0), ("residual_std", 0.0)]:
        if col not in df_state.columns:
            df_state[col] = default

    state_dict = {}
    for _, row in df_state.iterrows():
        state_dict[row["site_id"]] = {
            "ewma_val": float(row["ewma_val"]),
            "alpha": float(row["alpha"]),
            "residual_count": int(row["residual_count"]),
            "residual_std": float(row["residual_std"]),
        }

    # ── 2. 读取预测值 ──
    df_pred = pd.read_csv(predictions_path)
    required_pred_cols = {"site_id", "date", "predicted_value"}
    if not required_pred_cols.issubset(df_pred.columns):
        print(f"错误: predictions.csv 缺少必要字段 {required_pred_cols}, 实际字段: {list(df_pred.columns)}")
        sys.exit(1)

    df_pred["date"] = pd.to_datetime(df_pred["date"])

    if target_date is not None:
        target_date = pd.to_datetime(target_date)
        df_pred = df_pred[df_pred["date"] == target_date]
        if len(df_pred) == 0:
            print(f"错误: predictions.csv 中没有 date={target_date.date()} 的数据")
            sys.exit(1)

    target_dates = sorted(df_pred["date"].unique())
    print(f"读取预测文件: {predictions_path} (日期范围: {target_dates[0].date()} ~ {target_dates[-1].date()}, {len(df_pred)} 条)")

    # ── 3. 读取真实值 ──
    df_true = pd.read_csv(true_values_path)
    required_true_cols = {"site_id", "date", "true_value"}
    if not required_true_cols.issubset(df_true.columns):
        print(f"错误: true_values.csv 缺少必要字段 {required_true_cols}, 实际字段: {list(df_true.columns)}")
        sys.exit(1)

    df_true["date"] = pd.to_datetime(df_true["date"])
    print(f"读取真实值文件: {true_values_path} ({len(df_true)} 条)")

    # ── 4. 预热阶段: 用历史残差初始化 EWMA 状态 ──
    if warmup_path is not None and os.path.exists(warmup_path):
        df_warmup = pd.read_csv(warmup_path)
        required_warmup_cols = {"site_id", "date", "predicted_value"}
        if required_warmup_cols.issubset(df_warmup.columns):
            df_warmup["date"] = pd.to_datetime(df_warmup["date"])
            warmup_dates = sorted(df_warmup["date"].unique())
            print(f"\n预热阶段: {len(warmup_dates)} 天 ({warmup_dates[0].date()} ~ {warmup_dates[-1].date()})")

            for w_date in warmup_dates:
                day_warmup = df_warmup[df_warmup["date"] == w_date]
                day_true_w = df_true[pd.to_datetime(df_true["date"]) == w_date]
                true_map_w = dict(zip(day_true_w["site_id"], day_true_w["true_value"]))

                for _, w_row in day_warmup.iterrows():
                    site_id = w_row["site_id"]
                    pred_val = float(w_row["predicted_value"])

                    if site_id not in state_dict:
                        state_dict[site_id] = {
                            "ewma_val": 0.0, "alpha": alpha_default,
                            "residual_count": 0, "residual_std": 0.0,
                        }

                    if site_id in true_map_w:
                        true_val = true_map_w[site_id]
                        new_ewma = _update_ewma_for_pair(
                            site_id, true_val, pred_val, state_dict[site_id])

            for sid, s in state_dict.items():
                print(f"  站点 {sid}: 预热完成, ewma_val={s['ewma_val']:.4f}, "
                      f"residual_count={s['residual_count']}, std={s['residual_std']:.4f}")
        else:
            print(f"  ⚠ 预热文件缺少必要字段 {required_warmup_cols}, 跳过预热")
    else:
        print("\n  无预热数据, 从零开始 (前几天修正效果可能较差)")

    # ── 5. 逐日逐站点修正 ──
    all_results = []
    corrections_detail = []

    for process_date in target_dates:
        day_pred = df_pred[df_pred["date"] == process_date]
        # 取 process_date 的前一天作为"昨天"
        yesterday = process_date - pd.Timedelta(days=1)
        day_true = df_true[df_true["date"] == yesterday]

        true_map = dict(zip(day_true["site_id"], day_true["true_value"]))

        for _, pred_row in day_pred.iterrows():
            site_id = pred_row["site_id"]
            pred_val = float(pred_row["predicted_value"])

            # 初始化站点状态
            if site_id not in state_dict:
                state_dict[site_id] = {
                    "ewma_val": 0.0, "alpha": alpha_default,
                    "residual_count": 0, "residual_std": 0.0,
                }

            site_state = state_dict[site_id]

            # 无昨天真实值 → 跳过修正
            if site_id not in true_map:
                all_results.append({
                    "site_id": site_id, "date": process_date,
                    "original_prediction": pred_val,
                    "corrected_prediction": pred_val,
                    "ewma_val_used": site_state["ewma_val"],
                })
                continue

            true_val = float(true_map[site_id])
            old_ewma = site_state["ewma_val"]
            new_ewma = _update_ewma_for_pair(site_id, true_val, pred_val, site_state)

            # 修正预测
            corrected_val = pred_val + new_ewma

            corrections_detail.append({
                "site_id": site_id, "residual": true_val - pred_val,
                "old_ewma": old_ewma, "new_ewma": new_ewma,
                "original": pred_val, "corrected": corrected_val,
            })

            all_results.append({
                "site_id": site_id, "date": process_date,
                "original_prediction": pred_val,
                "corrected_prediction": corrected_val,
                "ewma_val_used": new_ewma,
            })

    # ── 5. 保存修正结果 (追加模式) ──
    df_result = pd.DataFrame(all_results)
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        df_existing["date"] = pd.to_datetime(df_existing["date"])
        # 去重: 新数据覆盖旧数据 (相同 site_id + date)
        df_combined = pd.concat([df_existing, df_result], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["site_id", "date"], keep="last")
        df_combined = df_combined.sort_values(["site_id", "date"]).reset_index(drop=True)
        df_combined.to_csv(output_path, index=False)
        print(f"\n修正结果已追加保存: {output_path} (合并后共 {len(df_combined)} 条)")
    else:
        df_result.to_csv(output_path, index=False)
        print(f"\n修正结果已保存: {output_path} ({len(df_result)} 条)")

    # ── 6. 写回状态 ──
    state_rows = []
    for site_id, s in state_dict.items():
        state_rows.append({
            "site_id": site_id,
            "ewma_val": s["ewma_val"],
            "alpha": s["alpha"],
            "residual_count": s["residual_count"],
            "residual_std": s["residual_std"],
        })
    df_state_out = pd.DataFrame(state_rows)
    df_state_out.to_csv(state_path, index=False)
    print(f"状态已保存: {state_path}")

    # ── 7. 摘要 ──
    print(f"\n{'='*60}")
    print("修正摘要:")
    print(f"{'='*60}")
    print(f"  处理站点数: {len(state_dict)}")
    if corrections_detail:
        avg_correction = np.mean([c["new_ewma"] for c in corrections_detail])
        print(f"  平均修正量: {avg_correction:+.4f}")

        # 对比 MAE
        orig_preds = np.array([c["original"] for c in corrections_detail])
        corr_preds = np.array([c["corrected"] for c in corrections_detail])
        # 需要真实值来计算 MAE —— 从最后一批中提取
        last_date = target_dates[-1]
        last_true = df_true[df_true["date"] == (last_date - pd.Timedelta(days=1))]
        true_map_last = dict(zip(last_true["site_id"], last_true["true_value"]))

        orig_errors, corr_errors = [], []
        for c in corrections_detail:
            if c["site_id"] in true_map_last:
                t = true_map_last[c["site_id"]]
                orig_errors.append(abs(t - c["original"]))
                corr_errors.append(abs(t - c["corrected"]))

        if orig_errors:
            mae_orig = np.mean(orig_errors)
            mae_corr = np.mean(corr_errors)
            improvement = (1 - mae_corr / mae_orig) * 100 if mae_orig > 0 else 0
            print(f"  修正前 MAE: {mae_orig:.4f}")
            print(f"  修正后 MAE: {mae_corr:.4f}")
            print(f"  提升: {improvement:+.1f}%")
    else:
        print("  无可修正数据")

    print(f"\n完成!")


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description="离线 EWMA 残差修正脚本")
    parser.add_argument("--predictions", required=True,
                        help="预测结果 CSV 路径 (必填, 字段: site_id, date, predicted_value)")
    parser.add_argument("--true_values", required=True,
                        help="真实值 CSV 路径 (必填, 字段: site_id, date, true_value)")
    parser.add_argument("--state", default="ewma_state.csv",
                        help="EWMA 状态 CSV 路径 (默认: ewma_state.csv)")
    parser.add_argument("--output", default="corrected_predictions.csv",
                        help="修正结果输出路径 (默认: corrected_predictions.csv)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="EWMA 平滑因子，仅首次初始化时使用 (默认: 0.3)")
    parser.add_argument("--date", default=None,
                        help="要处理的日期 (格式: YYYY-MM-DD, 默认: predictions.csv 中最新日期)")
    parser.add_argument("--warmup", default=None,
                        help="预热数据 CSV 路径 (字段: site_id, date, predicted_value; 用历史残差预热 EWMA 状态)")
    args = parser.parse_args()

    run_ewma_correction(
        predictions_path=args.predictions,
        true_values_path=args.true_values,
        state_path=args.state,
        output_path=args.output,
        alpha_default=args.alpha,
        target_date=args.date,
        warmup_path=args.warmup,
    )


if __name__ == "__main__":
    main()

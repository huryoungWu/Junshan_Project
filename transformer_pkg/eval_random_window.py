"""指定窗口评估 (与训练同口径): 指标与训练测试集完全一致。

本脚本与 train_transformer_autoregressive.py 的评估口径保持一致:

  - 数据管线: 与训练完全相同, 走 DataProcessor.build_feature_table()
    (内部先按训练/测试段分别清洗再拼接, 清洗不跨段; 日历/数据驱动特征、
    NaN 删除与训练同一份代码), 再用训练保存的 scaler 做归一化。绝不在本
    脚本里手工重建特征表 —— 旧版手工重建会导致测试段开头几天的 Hampel /
    突变修正口径与训练不同 (分段清洗 vs 全量清洗), 时间行集错位, 结果不一致。
  - 评估协议: stride=1 滑窗 (每个时刻起点一个 24h 预测窗口, 与训练
    evaluate() 逐样本一致), 单步自回归 rollout 直接复用训练脚本里的
    autoregressive_rollout / compute_mape (单一实现, 不可能漂移)。
    因此当所选范围覆盖整个测试集 (最后 test_days 天) 时, 汇总指标
    (MAE / RMSE / MAPE) 与训练结果目录 metrics.txt 完全一致。
  - 逐日视图: 为可读性保留"每天 00:00 起点窗口"的逐日指标表与画图
    (旧版逐日预测), 但它只是每天一次的独立预测, 不参与汇总指标
    (汇总指标 = stride=1 全样本, 与训练一致)。

用法:
  # 默认 (无 --start_date): 随机起点 + 7 天
  python eval_random_window.py

  # 指定起始日期, 默认 7 天
  python eval_random_window.py --start_date 2025-10-03

  # 全天模式: 覆盖从起始日到数据末尾的全部 stride=1 样本
  #   → 起始日取测试集起点时, 汇总指标 == 训练 metrics.txt
  python eval_random_window.py --start_date 2025-10-03 --all_days

  # 指定结束日期 + 逐日画图保存
  python eval_random_window.py --start_date 2025-10-03 --end_date 2025-12-31 --plot
"""

import os
import sys
import pickle
import argparse
import math
import random
import re
import warnings
warnings.filterwarnings("ignore")

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor
# 评估与训练共用同一套 rollout / MAPE 实现 (见 train_transformer_autoregressive.py
# 头注释: 自回归推理须用配套的 autoregressive_rollout, 不能另写一份)
from train_transformer_autoregressive import autoregressive_rollout, compute_mape

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv"
DEFAULT_RESULT_DIR = os.path.join(HERE, "results",
                                  "junshan_L1D_P24H_1h_transformer_autoregressive_20260828_143855")


def calc_metrics(y_true, y_pred, floor_ratio=0.1):
    """与训练 evaluate() 同口径: sklearn MAE/RMSE + compute_mape (floor 过滤)。"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape, n_total, n_used = compute_mape(y_true, y_pred, floor_ratio)
    return mae, rmse, mape, n_used, n_total


def main():
    parser = argparse.ArgumentParser(
        description="指定窗口评估 (与训练同口径): 覆盖测试集时指标与训练 metrics.txt 一致")
    parser.add_argument("--data", default=DEFAULT_DATA, help="原始数据 CSV (默认用训练 config 的 file_path)")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="训练结果目录")
    parser.add_argument("--start_date", default=None,
                        help="评估起始日期 (格式 YYYY-MM-DD, 第一个预测时刻)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (仅未指定 --start_date 时生效)")
    parser.add_argument("--all_days", action="store_true",
                        help="从 start_date 到数据末尾, 全部 stride=1 样本 (汇总指标与训练一致)")
    parser.add_argument("--end_date", default=None,
                        help="评估结束日期 (格式 YYYY-MM-DD, 最后一个预测日, 配合 --start_date 使用)")
    parser.add_argument("--plot", action="store_true",
                        help="逐日画图并保存到本地")
    parser.add_argument("--plot_dir", default=None,
                        help="逐日图片保存目录 (默认: result_dir/daily_plots)")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV 编码")
    parser.add_argument("--lookback", type=float, default=None,
                        help="回看天数 (默认: 训练配置; 模型 input_len 固定, 传值必须等于训练配置)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 加载训练保存的 scaler / 配置 / 模型 ──
    scaler_path = os.path.join(args.result_dir, "scaler.pkl")
    model_path = os.path.join(args.result_dir, "best_seq2seq_model.pth")
    for p in (scaler_path, model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到: {p}")

    with open(scaler_path, "rb") as f:
        saved = pickle.load(f)

    config = saved["config"]
    feature_scaler = saved["feature_scaler"]
    target_scaler = saved["target_scaler"]
    feature_cols = saved["feature_cols"]
    target_cols = saved["target_cols"]
    target_feat_idx = saved.get("target_feat_idx",
                                feature_cols.index(target_cols[0]))
    detach_feedback = saved.get("detach_feedback", True)

    # --data 只允许换数据文件, 清洗/特征/切分管线仍与训练同口径
    if args.data is not None and os.path.abspath(args.data) != os.path.abspath(config["file_path"]):
        config = dict(config)
        config["file_path"] = args.data
        print(f"数据文件: {args.data} (覆盖训练配置, 其余管线不变)")

    if args.lookback is not None and args.lookback != config["lookback_days"]:
        raise SystemExit(f"错误: --lookback {args.lookback} != 训练配置 lookback_days="
                         f"{config['lookback_days']} (模型 input_len 固定为训练值, "
                         f"评估必须用训练的回看天数)")

    freq_minutes = int(config["resample_freq"].replace("min", ""))
    points_per_day = (24 * 60) // freq_minutes
    predict_steps = int(config["predict_days"] * points_per_day)
    lookback_steps = int(config["lookback_days"] * points_per_day)

    # ── 数据: 完全复用训练管线 (分段清洗 + 特征 + 保存的 scaler) ──
    print("\n" + "=" * 70)
    print(" [数据] 重建特征表 (与训练 build_feature_table 同口径)")
    print("=" * 70)
    processor = DataProcessor(config)
    df_all_feat = processor.build_feature_table()

    # 校验: 训练保存的 feature_cols 必须能在当前管线表中找到 (防旧 scaler 配新代码)
    missing = [c for c in feature_cols if c not in df_all_feat.columns]
    if missing:
        raise SystemExit(f"错误: 训练保存的 feature_cols 在当前数据管线中缺失 {missing} —— "
                         f"本结果目录的模型基于旧特征集, 需用当前管线重新训练")
    if len(feature_cols) != len(df_all_feat.columns):
        print(f"  ⚠ 当前管线特征数 {len(df_all_feat.columns)} != 训练保存 {len(feature_cols)}, "
              f"按训练保存的 feature_cols 取列")

    # 用训练保存的 scaler 做归一化 (与训练 transform_df 完全一致)
    processor.feature_scaler = feature_scaler
    processor.target_scaler = target_scaler
    processor.feature_cols = feature_cols
    processor.target_cols = target_cols
    X_all, Y_all = processor.transform_df(df_all_feat)
    # 与训练 make_sequences_ar 一致: 序列张量用 float32
    X_all = np.ascontiguousarray(X_all, dtype=np.float32)
    Y_all = np.ascontiguousarray(Y_all, dtype=np.float32)
    ts_all = df_all_feat.index.to_numpy()          # datetime64 数组, 与行一一对应
    total_len = len(X_all)

    # 训练/测试段边界 (与训练 split_by_time 同一口径): 评估样本不跨段
    df_train_feat, _ = processor.split_by_time(df_all_feat)   # 返回 (train, test)
    test_start_row = len(df_train_feat)

    print(f"特征表: {total_len} 行 × {len(feature_cols)} 列 "
          f"({pd.Timestamp(ts_all[0]).date()} ~ {pd.Timestamp(ts_all[-1]).date()})")
    print(f"模型: 回看 {config['lookback_days']}d ({lookback_steps}步) → 预测 "
          f"{predict_steps}步 (单步自回归, detach_feedback={detach_feedback})")

    # ── 加载模型 ──
    model_type = config.get("model_type", "transformer")
    model_kwargs = dict(
        input_dim=len(feature_cols),
        output_dim=1,
        horizon=1,
        input_len=lookback_steps,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["transformer_dropout"],
    )
    if model_type == "itransformer":
        model = iTransformer(**model_kwargs, target_idx=target_feat_idx).to(device)
    else:
        model = TimeSeriesTransformer(**model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── 确定预测起点 (第一个预测时刻) ──
    if args.start_date is not None:
        target_date = pd.Timestamp(args.start_date)
        mask = df_all_feat.index >= target_date
        if not mask.any():
            print(f"错误: 数据中没有 >= {args.start_date} 的日期")
            return
        first_pred_idx = int(np.where(mask)[0][0])
        print(f"起始日期: {args.start_date}")
    else:
        if total_len <= predict_steps:
            print("数据不足")
            return
        first_pred_idx = random.randint(0, total_len - predict_steps)
        print(f"随机起始索引: {first_pred_idx} (seed={args.seed})")

    # ── 预测范围: stride=1 滑窗 (与训练 evaluate 同一评估协议) ──
    if args.end_date is not None:
        end_mask = df_all_feat.index >= pd.Timestamp(args.end_date)
        if not end_mask.any():
            print(f"错误: 数据中没有 >= {args.end_date} 的日期")
            return
        last_start = int(np.where(end_mask)[0][0])
    elif args.all_days:
        last_start = total_len - predict_steps
    else:
        # 默认 7 天 (每天 24 个起点)
        last_start = first_pred_idx + 7 * points_per_day - 1
    last_start = min(last_start, total_len - predict_steps)

    # ── 与训练 evaluate 同协议: 样本窗口/预测均不跨训练/测试段边界 ──
    # 训练侧: 训练段样本预测止于段尾 (起点 ≤ 段尾 - predict),
    #         测试段样本起点 ≥ 段首 + lookback (段首 lookback 天只作回看)。
    # 请求范围若跨段, 按段拆成若干连续子区间, 边界交叉样本不评估。
    chunks = []
    train_chunk = (max(first_pred_idx, lookback_steps),
                   min(last_start, test_start_row - predict_steps))
    test_chunk = (max(first_pred_idx, test_start_row + lookback_steps),
                  min(last_start, total_len - predict_steps))
    for lo, hi in (train_chunk, test_chunk):
        if lo <= hi:
            chunks.append((lo, hi))

    if not chunks:
        print(f"错误: 无可预测样本 (起点 {first_pred_idx} 后数据不够回看 "
              f"{config['lookback_days']} 天 + 预测 {predict_steps} 步, 且样本不跨段边界)")
        return

    sample_starts = [i for lo, hi in chunks for i in range(lo, hi + 1)]
    n_samples = len(sample_starts)
    first_eff = sample_starts[0]
    last_eff = sample_starts[-1]
    if args.start_date is not None and first_eff != first_pred_idx:
        print(f"  ⚠ 起点前移: {pd.Timestamp(ts_all[first_pred_idx]).date()} 距所在段段首不足 "
              f"lookback {config['lookback_days']} 天 (与训练同协议: 段首回看天不参与预测), "
              f"预测实际从 {pd.Timestamp(ts_all[first_eff]).date()} 开始")
    n_days = (pd.Timestamp(ts_all[last_eff]).normalize() -
              pd.Timestamp(ts_all[first_eff]).normalize()).days + 1
    print(f"\n{'='*70}")
    print(f" 逐样本预测 ({n_samples} 个 stride=1 样本, 跨 {n_days} 天)")
    print(f" 回看窗口: {pd.Timestamp(ts_all[first_eff - lookback_steps])} ~ "
          f"{pd.Timestamp(ts_all[first_eff - 1])}")
    print(f" 预测输出范围: {pd.Timestamp(ts_all[first_eff])} ~ "
          f"{pd.Timestamp(ts_all[last_eff + predict_steps - 1])}")
    print(f" 样本不跨训练/测试段边界 (与训练 evaluate 同协议)")
    print(f"{'='*70}")

    # ── 逐样本自回归 rollout (与训练 evaluate 完全一致) ──
    all_preds = []
    all_trues = []
    sample_starts_used = []     # 每个样本起点在 ts_all 中的下标
    for i in tqdm(sample_starts, desc="Autoregressive rollout", unit="sample"):
        window_t = torch.from_numpy(X_all[i - lookback_steps:i]).unsqueeze(0).to(device)
        future_t = torch.from_numpy(X_all[i:i + predict_steps]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = autoregressive_rollout(model, window_t, future_t, predict_steps,
                                          target_feat_idx, detach_feedback)
        pred_inv = processor.inverse_transform_targets(pred.cpu().numpy())[0, :, 0]
        true_inv = processor.inverse_transform_targets(Y_all[i:i + predict_steps])[:, 0]
        all_preds.append(pred_inv)
        all_trues.append(true_inv)
        sample_starts_used.append(i)

    # ── 汇总指标 (与训练 evaluate 同口径: 全部样本点拼一起算) ──
    y_true_all = np.concatenate(all_trues)
    y_pred_all = np.concatenate(all_preds)
    floor = config.get("mape_floor_ratio", 0.1)
    agg_mae, agg_rmse, agg_mape, n_used, n_total = calc_metrics(y_true_all, y_pred_all, floor)

    print(f"\n{'='*70}")
    print(f" 汇总指标 ({n_samples} 样本 × {predict_steps} 步, stride=1, 与训练同口径)")
    print(f"{'='*70}")
    print(f"  MAE={agg_mae:.2f}, RMSE={agg_rmse:.2f}, MAPE={agg_mape:.2f}%")
    print(f"  (MAPE 过滤: 排除 |true| < {floor:.0%} * max|true| 的点, "
          f"保留 {n_used}/{n_total} 点)")

    # 范围覆盖整个测试集时, 直接与训练 metrics.txt 对比
    if args.all_days:
        mtxt_path = os.path.join(args.result_dir, "metrics.txt")
        if os.path.exists(mtxt_path):
            with open(mtxt_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Test"):
                        m = re.search(r"MAE=([\d.]+), RMSE=([\d.]+), MAPE=([\d.]+)%", line)
                        if m:
                            tr_mae, tr_rmse, tr_mape = map(float, m.groups())
                            ok = abs(agg_mape - tr_mape) < 0.02 and abs(agg_mae - tr_mae) < 0.5
                            print(f"  训练 metrics.txt: MAE={tr_mae:.2f}, RMSE={tr_rmse:.2f}, "
                                  f"MAPE={tr_mape:.2f}%"
                                  f"   → {'✅ 与训练一致' if ok else '❌ 不一致 (请检查范围是否覆盖整个测试集)'}")
                        break

    # ── 流量区间 MAPE 分析 ──
    # ts_all[sample_starts_used] 是每个样本的起始时刻 (1969 个),
    # 但 y_true_all/y_pred_all 是展平的 (1969*24=47256 个点),
    # 需要把时间戳展开: 每个起始时刻 → 后续 predict_steps 个时间点
    expanded_ts = np.concatenate([
        ts_all[i:i + predict_steps] for i in sample_starts_used
    ])
    save_dir_flow = os.path.join(HERE, "flow_range_analysis")
    analyze_mape_by_flow_range(
        y_true_all, y_pred_all, expanded_ts,
        flow_min=1000, flow_max=5000, flow_step=100,
        floor_ratio=floor, save_dir=save_dir_flow,
    )

    # ── 逐日视图: 每天 00:00 起点的 24h 预测 (可读性用, 不参与汇总) ──
    daily = []      # (date, pred_inv, true_inv, ts, 样本起点下标)
    for idx, i in enumerate(sample_starts_used):
        ts_s = pd.Timestamp(ts_all[i])
        if ts_s.hour == 0:
            daily.append((ts_s, all_preds[idx], all_trues[idx],
                          ts_all[i:i + predict_steps], i))

    if len(daily) > 0:
        print(f"\n{'='*70}")
        print(f" 逐日指标 (每天 00:00 起点窗口的独立预测, 供参考)")
        print(f"{'='*70}")
        print(f"  {'日期':<14}{'MAE':<10}{'RMSE':<10}{'MAPE%':<10}")
        print(f"  {'-'*44}")
        for date, pred, true, ts, _ in daily:
            d_mae, d_rmse, d_mape, _, _ = calc_metrics(true, pred, floor)
            # 注意: 日期必须先 strftime 成字符串再对齐 (datetime 的 __format__
            # 会把格式串当 strftime 处理, f"{date.date():<14}" 会输出字面量 "<14")
            print(f"  {date.strftime('%Y-%m-%d'):<14}{d_mae:<10.2f}"
                  f"{d_rmse:<10.2f}{d_mape:<10.2f}")

    # ── 画图 ──
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 逐日画图: 每天一张 (00:00 起点窗口)
    plot_dir = None
    if args.plot and len(daily) > 0:
        plot_dir = args.plot_dir or os.path.join(args.result_dir, "daily_plots")
        os.makedirs(plot_dir, exist_ok=True)
        print(f"\n  逐日图片将保存到: {plot_dir} ({len(daily)} 张)")
        for date, pred, true, ts, _ in daily:
            d_mae, d_rmse, d_mape, _, _ = calc_metrics(true, pred, floor)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(ts, true, "b-o", markersize=2, linewidth=1.2, label="真实值")
            ax.plot(ts, pred, "r-s", markersize=2, linewidth=1.2, label="预测值")
            ax.fill_between(ts, true, pred, alpha=0.15, color="gray")
            ax.set_title(f"{date.strftime('%Y-%m-%d')} (00:00 起点窗口)  "
                         f"MAE={d_mae:.2f}  RMSE={d_rmse:.2f}  MAPE={d_mape:.2f}%",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("时间", fontsize=10)
            ax.set_ylabel("流量", fontsize=10)
            ax.legend(fontsize=10, loc="upper right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"{date.strftime('%Y-%m-%d')}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"  逐日图片已保存: {plot_dir} ({len(daily)} 张)")

    # 总对比图 (最多前 5 天)
    if len(daily) > 0:
        n_show = min(len(daily), 5)
        fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), sharex=False)
        if n_show == 1:
            axes = [axes]
        for i in range(n_show):
            date, pred, true, ts, _ = daily[i]
            axes[i].plot(ts, true, "b-o", markersize=2, label="真实值")
            axes[i].plot(ts, pred, "r-s", markersize=2, label="预测值")
            axes[i].set_title(f"{date.strftime('%Y-%m-%d')} (00:00 起点窗口)")
            axes[i].legend(fontsize=8)
            axes[i].grid(alpha=0.3)
        fig.suptitle(f"逐日预测对比 (前 {n_show}/{len(daily)} 天, 汇总 MAPE={agg_mape:.2f}% "
                     f"@ {n_samples} 个 stride=1 样本)", fontsize=13)
        fig.tight_layout()
        fig_path = os.path.join(HERE, "eval_random_window.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n预测对比图已保存: {fig_path}")


def analyze_mape_by_flow_range(y_true_all, y_pred_all, timestamps,
                                flow_min=1000, flow_max=5000, flow_step=100,
                                floor_ratio=0.1, save_dir=None):
    """按真实流量区间分析 MAPE 分布 + 各区间误差最大的时刻。

    将真实流量按 flow_step 步长分箱 (默认 1000~5000, 步长 100),
    统计每个区间的 MAPE / MAE / RMSE / 样本数, 并分析每个区间内
    误差最大的时刻分布。

    Args:
        y_true_all: (N,) 真实流量数组 (已反归一化, m³/h)
        y_pred_all: (N,) 预测流量数组 (已反归一化, m³/h)
        timestamps: (N,) 时间戳数组 (datetime64 或 pd.DatetimeIndex)
        flow_min:   流量区间下界 (默认 1000)
        flow_max:   流量区间上界 (默认 5000)
        flow_step:  步长 (默认 100)
        floor_ratio: MAPE 过滤阈值 (与训练一致)
        save_dir:   图片保存目录 (None 则不画图)
    """
    y_true = np.asarray(y_true_all).reshape(-1)
    y_pred = np.asarray(y_pred_all).reshape(-1)
    ts = pd.DatetimeIndex(timestamps) if not isinstance(timestamps, pd.DatetimeIndex) else timestamps
    n_total = len(y_true)
    if n_total == 0:
        print("  ⚠ 无样本, 跳过流量区间分析")
        return None
    if len(ts) != n_total:
        print(f"  ⚠ 时间戳长度 ({len(ts)}) != 样本数 ({n_total}), 跳过流量区间分析")
        return None

    # ── 构造流量区间 ──
    bins = np.arange(flow_min, flow_max + flow_step, flow_step)
    labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins) - 1)]

    # ── 分箱: 每个样本分配到对应区间 ──
    bin_indices = np.digitize(y_true, bins) - 1   # 0-indexed
    # 超出范围的样本归到两端
    bin_indices = np.clip(bin_indices, 0, len(labels) - 1)

    # ── 逐区间统计 ──
    rows = []
    all_hour_errors = {}   # {bin_label: {hour: [abs_errors]}}
    worst_samples = {}     # {bin_label: [(abs_err, hour, true_val, pred_val, ts)]}

    thr = floor_ratio * np.abs(y_true).max()

    for b in range(len(labels)):
        mask = bin_indices == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            rows.append({
                "flow_range": labels[b], "n_samples": 0,
                "mape": np.nan, "mae": np.nan, "rmse": np.nan,
                "mean_true": np.nan, "max_abs_err": np.nan,
                "worst_hour": np.nan, "worst_ts": np.nan,
            })
            continue

        t = y_true[mask]
        p = y_pred[mask]
        abs_err = np.abs(t - p)
        # MAPE: 仅对 |true| >= thr 的点
        mape_mask = np.abs(t) >= thr
        if mape_mask.sum() > 0:
            mape = np.mean(np.abs((t[mape_mask] - p[mape_mask]) / (t[mape_mask] + 1e-8))) * 100
        else:
            mape = np.nan
        mae = float(abs_err.mean())
        rmse = float(np.sqrt((abs_err ** 2).mean()))

        # 每个小时的误差分布
        bin_ts = ts[mask]
        hours = bin_ts.hour
        for h in range(24):
            h_mask = hours == h
            if h_mask.sum() > 0:
                key = (labels[b], h)
                if labels[b] not in all_hour_errors:
                    all_hour_errors[labels[b]] = {}
                all_hour_errors[labels[b]].setdefault(h, []).extend(abs_err[h_mask].tolist())

        # 误差最大的样本
        worst_idx_local = np.argsort(abs_err)[-5:][::-1]
        worst_list = []
        for wi in worst_idx_local:
            orig_idx = np.where(mask)[0][wi]
            worst_list.append({
                "abs_err": float(abs_err[wi]),
                "hour": int(hours[wi]),
                "true": float(t[wi]),
                "pred": float(p[wi]),
                "timestamp": str(ts[orig_idx]),
            })
        worst_samples[labels[b]] = worst_list

        rows.append({
            "flow_range": labels[b], "n_samples": n_bin,
            "mape": float(mape), "mae": mae, "rmse": rmse,
            "mean_true": float(t.mean()),
            "max_abs_err": float(abs_err.max()),
            "worst_hour": int(hours[np.argmax(abs_err)]),
            "worst_ts": str(bin_ts[np.argmax(abs_err)]),
        })

    df_range = pd.DataFrame(rows)

    # ── 打印结果 ──
    print(f"\n{'=' * 80}")
    print(f" 流量区间 MAPE 分析 (范围 {flow_min}~{flow_max}, 步长 {flow_step})")
    print(f"{'=' * 80}")
    valid = df_range[df_range["n_samples"] > 0].copy()
    print(f"  {'流量区间':<18}{'样本数':<8}{'MAPE%':<10}{'MAE':<10}{'RMSE':<10}"
          f"{'均值流量':<12}{'最大误差':<10}{'误差最大时'}")
    print(f"  {'-' * 86}")
    for _, r in valid.iterrows():
        mape_str = f"{r['mape']:.2f}" if not np.isnan(r['mape']) else "N/A"
        n_samp = int(r['n_samples'])
        worst_h = int(r['worst_hour']) if not np.isnan(r['worst_hour']) else 0
        print(f"  {r['flow_range']:<18}{n_samp:<8}{mape_str:<10}"
              f"{r['mae']:<10.2f}{r['rmse']:<10.2f}"
              f"{r['mean_true']:<12.1f}{r['max_abs_err']:<10.2f}"
              f"{worst_h:02d}:00")

    # ── 找出误差最大的区间 ──
    if len(valid) > 0:
        best_mape_row = valid.loc[valid["mape"].idxmin()]
        worst_mape_row = valid.loc[valid["mape"].idxmax()]
        print(f"\n  MAPE 最低区间: {best_mape_row['flow_range']}  MAPE={best_mape_row['mape']:.2f}%")
        print(f"  MAPE 最高区间: {worst_mape_row['flow_range']}  MAPE={worst_mape_row['mape']:.2f}%")

    # ── 分析各区间误差最大的时刻分布 ──
    print(f"\n{'=' * 80}")
    print(f" 各流量区间误差最大的 5 个样本")
    print(f"{'=' * 80}")
    for b_label, worst_list in worst_samples.items():
        if not worst_list:
            continue
        print(f"\n  [{b_label}] (前 5 个最大误差样本):")
        for i, w in enumerate(worst_list):
            print(f"    #{i+1}  AE={w['abs_err']:.2f}  hour={w['hour']:02d}:00  "
                  f"true={w['true']:.1f}  pred={w['pred']:.1f}  ts={w['timestamp']}")

    # ── 按流量区间 × 小时的误差热力图数据 ──
    print(f"\n{'=' * 80}")
    print(f" 流量区间 × 小时 平均绝对误差 (AE) 热力图")
    print(f"{'=' * 80}")

    heatmap_data = np.full((len(labels), 24), np.nan)
    for b_label, hour_dict in all_hour_errors.items():
        b_idx = labels.index(b_label)
        for h, errs in hour_dict.items():
            heatmap_data[b_idx, h] = np.mean(errs)

    # 只打印有数据的行
    has_data_rows = ~np.all(np.isnan(heatmap_data), axis=1)
    if has_data_rows.any():
        # 表头: 小时
        hour_header = "  流量区间\\小时  " + "".join(f"{h:>6}" for h in range(24))
        print(hour_header)
        print(f"  {'-' * (14 + 24 * 6)}")
        for b_idx in range(len(labels)):
            if not has_data_rows[b_idx]:
                continue
            row_str = f"  {labels[b_idx]:<14}"
            for h in range(24):
                val = heatmap_data[b_idx, h]
                if np.isnan(val):
                    row_str += f"{'—':>6}"
                else:
                    row_str += f"{val:>6.0f}"
            print(row_str)

    # ── 综合: 每个小时整体误差 ──
    print(f"\n{'=' * 80}")
    print(f" 按小时的整体误差分布 (跨所有流量区间)")
    print(f"{'=' * 80}")
    hours_all = ts.hour
    hour_rows = []
    for h in range(24):
        h_mask = hours_all == h
        n_h = int(h_mask.sum())
        if n_h == 0:
            continue
        h_true = y_true[h_mask]
        h_pred = y_pred[h_mask]
        h_abs_err = np.abs(h_true - h_pred)
        h_mape_mask = np.abs(h_true) >= thr
        h_mape = (np.mean(np.abs((h_true[h_mape_mask] - h_pred[h_mape_mask]) /
                                  (h_true[h_mape_mask] + 1e-8))) * 100
                  if h_mape_mask.sum() > 0 else np.nan)
        hour_rows.append({
            "hour": h, "n_samples": n_h,
            "mae": float(h_abs_err.mean()),
            "rmse": float(np.sqrt((h_abs_err ** 2).mean())),
            "mape": float(h_mape),
            "max_ae": float(h_abs_err.max()),
        })

    df_hour = pd.DataFrame(hour_rows)
    if len(df_hour) > 0:
        print(f"  {'小时':<8}{'样本数':<8}{'MAE':<10}{'RMSE':<10}{'MAPE%':<10}{'最大AE':<10}")
        print(f"  {'-' * 54}")
        for _, r in df_hour.iterrows():
            mape_str = f"{r['mape']:.2f}" if not np.isnan(r['mape']) else "N/A"
            print(f"  {int(r['hour']):02d}:00   {int(r['n_samples']):<8}"
                  f"{r['mae']:<10.2f}{r['rmse']:<10.2f}{mape_str:<10}{r['max_ae']:<10.2f}")

        # 找出误差最大的小时
        worst_hour_row = df_hour.loc[df_hour["mape"].idxmax()]
        best_hour_row = df_hour.loc[df_hour["mape"].idxmin()]
        print(f"\n  MAPE 最高小时: {int(worst_hour_row['hour']):02d}:00  "
              f"MAPE={worst_hour_row['mape']:.2f}%  MAE={worst_hour_row['mae']:.2f}")
        print(f"  MAPE 最低小时: {int(best_hour_row['hour']):02d}:00  "
              f"MAPE={best_hour_row['mape']:.2f}%  MAE={best_hour_row['mae']:.2f}")

    # ── 画图 ──
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        # 图1: 各流量区间 MAPE 柱状图
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))

        ax = axes[0, 0]
        plot_valid = valid[valid["mape"].notna()].copy()
        if len(plot_valid) > 0:
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(plot_valid)))
            bars = ax.bar(range(len(plot_valid)), plot_valid["mape"], color=colors,
                          edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(plot_valid)))
            ax.set_xticklabels(plot_valid["flow_range"], rotation=45, ha="right", fontsize=7)
            ax.set_xlabel("真实流量区间 (m³/h)")
            ax.set_ylabel("MAPE (%)")
            ax.set_title("各流量区间 MAPE 分布")
            ax.grid(axis="y", alpha=0.3)
            # 标注数值
            for bar, val in zip(bars, plot_valid["mape"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=6)
        ax.axhline(y=5.0, color="red", linestyle="--", linewidth=1.2, label="5% 阈值")
        ax.legend(fontsize=9)

        # 图2: 各流量区间样本数
        ax = axes[0, 1]
        ax.bar(range(len(valid)), valid["n_samples"], color="#3498db",
               edgecolor="white", alpha=0.8)
        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels(valid["flow_range"], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("真实流量区间 (m³/h)")
        ax.set_ylabel("样本数")
        ax.set_title("各流量区间样本数分布")
        ax.grid(axis="y", alpha=0.3)

        # 图3: 各流量区间 MAE
        ax = axes[1, 0]
        ax.bar(range(len(valid)), valid["mae"], color="#e74c3c",
               edgecolor="white", alpha=0.8)
        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels(valid["flow_range"], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("真实流量区间 (m³/h)")
        ax.set_ylabel("MAE (m³/h)")
        ax.set_title("各流量区间 MAE 分布")
        ax.grid(axis="y", alpha=0.3)

        # 图4: 热力图 — 流量区间 × 小时 平均绝对误差
        ax = axes[1, 1]
        # 只画有数据的行
        plot_heatmap = heatmap_data[has_data_rows]
        plot_labels = [labels[i] for i in range(len(labels)) if has_data_rows[i]]
        if plot_heatmap.size > 0:
            im = ax.imshow(plot_heatmap, aspect="auto", cmap="YlOrRd",
                           interpolation="nearest")
            ax.set_xticks(range(24))
            ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
            ax.set_yticks(range(len(plot_labels)))
            ax.set_yticklabels(plot_labels, fontsize=7)
            ax.set_xlabel("小时")
            ax.set_ylabel("真实流量区间 (m³/h)")
            ax.set_title("流量区间 × 小时 平均绝对误差 (AE) 热力图")
            plt.colorbar(im, ax=ax, label="AE (m³/h)")

        fig.suptitle("流量区间 MAPE 分析", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig_path = os.path.join(save_dir, "flow_range_mape_analysis.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  流量区间分析图已保存: {fig_path}")

        # 图5: 小时维度 MAPE / MAE 双轴图
        if len(df_hour) > 0:
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            ax1.bar(df_hour["hour"], df_hour["mape"], color="#e74c3c", alpha=0.7, label="MAPE (%)")
            ax1.set_xlabel("小时")
            ax1.set_ylabel("MAPE (%)")
            ax1.set_title("各小时 MAPE 分布 (跨所有流量区间)")
            ax1.set_xticks(range(24))
            ax1.set_xticklabels([f"{h:02d}:00" for h in range(24)], fontsize=8)
            ax1.axhline(y=5.0, color="black", linestyle="--", linewidth=1, label="5% 阈值")
            ax1.legend(fontsize=9)
            ax1.grid(axis="y", alpha=0.3)

            ax2.bar(df_hour["hour"], df_hour["mae"], color="#3498db", alpha=0.7, label="MAE (m³/h)")
            ax2_twin = ax2.twinx()
            ax2_twin.plot(df_hour["hour"], df_hour["max_ae"], "r-o", markersize=4,
                          linewidth=1.5, label="最大 AE")
            ax2.set_xlabel("小时")
            ax2.set_ylabel("MAE (m³/h)")
            ax2_twin.set_ylabel("最大 AE (m³/h)")
            ax2.set_title("各小时 MAE / 最大 AE 分布")
            ax2.set_xticks(range(24))
            ax2.set_xticklabels([f"{h:02d}:00" for h in range(24)], fontsize=8)
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
            ax2.grid(axis="y", alpha=0.3)

            fig2.suptitle("小时维度误差分析", fontsize=14, fontweight="bold")
            fig2.tight_layout(rect=(0, 0, 1, 0.96))
            fig2_path = os.path.join(save_dir, "hourly_error_analysis.png")
            fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
            plt.close(fig2)
            print(f"  小时维度误差分析图已保存: {fig2_path}")

        # 图6: 散点图 — 真实流量 vs 绝对误差
        fig3, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(y_true, np.abs(y_true - y_pred), c=ts.hour, cmap="hsv",
                             alpha=0.3, s=5, vmin=0, vmax=23)
        ax.set_xlabel("真实流量 (m³/h)")
        ax.set_ylabel("绝对误差 (m³/h)")
        ax.set_title("真实流量 vs 绝对误差 (颜色=小时)")
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="小时 (0-23)")
        fig3_path = os.path.join(save_dir, "flow_vs_error_scatter.png")
        fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
        plt.close(fig3)
        print(f"  流量-误差散点图已保存: {fig3_path}")

    return df_range, df_hour


if __name__ == "__main__":
    main()

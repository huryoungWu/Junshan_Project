# -*- coding: utf-8 -*-
"""junshan_inference.py — 军山水厂流量预测 (简化版)

输入: CSV 文件 (时间列 + 出厂水流量列, 最少 ~16 天数据)
输出: JSON 格式的预测结果

用法:
  python junshan_inference.py
  python junshan_inference.py --data path/to/data.csv
  python junshan_inference.py --data path/to/data.csv --output pred.json
"""

import os
import sys
import pickle
import argparse
import json

import numpy as np
import pandas as pd
import torch

# UTF-8 输出
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor

HERE = os.path.dirname(os.path.abspath(__file__))

# 默认输入 CSV (绝对路径)
DEFAULT_DATA = os.path.join(HERE, "..", "data", "input_nextday16h_20250820_35d.csv")
# 默认模型目录 (绝对路径)
DEFAULT_RESULT_DIR = os.path.join(
    HERE, "results", "junshan_L1D_P24H_1h_transformer_nextday16h_mc_20260901_142931")

DAY_STEPS = 24
UNIT = "m3/h"


def predict(csv_path, result_dir=DEFAULT_RESULT_DIR, provider=None, device=None):
    """读取 CSV → 预测 → 返回 JSON 格式结果

    Args:
        csv_path: 输入 CSV 文件路径
        result_dir: 训练结果目录 (含 scaler.pkl + best_seq2seq_model.pth)
        provider: 接口里的 provider 字段
        device: 推理设备 (None = 自动)

    Returns:
        dict: {date, provider, unit, interval_minutes, horizon, values}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── 加载模型 ──
    print(f"[1/4] 加载模型: {result_dir}")
    scaler_path = os.path.join(result_dir, "scaler.pkl")
    model_path = os.path.join(result_dir, "best_seq2seq_model.pth")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"未找到 scaler.pkl: {scaler_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型权重: {model_path}")

    with open(scaler_path, "rb") as f:
        saved = pickle.load(f)

    config = saved["config"]
    feature_scaler = saved["feature_scaler"]
    target_scaler = saved["target_scaler"]
    feature_cols = saved["feature_cols"]
    target_cols = saved["target_cols"]
    target_feat_idx = saved.get("target_feat_idx", feature_cols.index(target_cols[0]))
    target_col = target_cols[0]

    # 推理参数
    lookback_steps = saved.get("lookback_steps")
    if lookback_steps is None:
        lookback_days = config["lookback_days"]
        lookback_extra = config.get("lookback_extra_hours", 0)
        freq_minutes = int(config["resample_freq"].replace("min", ""))
        points_per_day = (24 * 60) // freq_minutes
        lookback_steps = int(lookback_days * points_per_day) + int(lookback_extra)
    resample_freq = config["resample_freq"]
    freq_minutes = int(resample_freq.replace("min", ""))

    # 加载模型
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

    # 初始化处理器
    processor = DataProcessor(config)
    processor.feature_scaler = feature_scaler
    processor.target_scaler = target_scaler
    processor.feature_cols = feature_cols

    prov = provider or f"junshan_{model_type}_nextday16h"
    print(f"      模型: {model_type}, device={device}")

    # ── 读取并处理 CSV ──
    print(f"[2/4] 读取 CSV: {csv_path}")
    df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")

    if not isinstance(df_raw.index, pd.DatetimeIndex):
        for ts_col in ("时间", "timestamp"):
            if ts_col in df_raw.columns:
                df_raw[ts_col] = pd.to_datetime(df_raw[ts_col])
                df_raw = df_raw.set_index(ts_col)
                break
        else:
            raise ValueError("CSV 必须包含 时间 / timestamp 列")

    print("[3/4] 数据清洗 + 特征构建")
    df_base = processor.build_base_features(df_raw)
    df_clean = processor.clean_and_resample(df_base)
    df_feat = processor.add_calendar_features(df_clean)
    df_feat = processor.add_data_driven_features(df_feat)

    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"特征列缺失: {missing}")

    df_feat = df_feat[feature_cols].dropna()
    if len(df_feat) < lookback_steps:
        raise ValueError(f"数据不足: 仅 {len(df_feat)} 行, 需要 {lookback_steps} 步")

    # ── 回看窗口 ──
    window_feat = df_feat.iloc[-lookback_steps:]
    last_ts = window_feat.index[-1]
    H = int(last_ts.hour) + 1
    total_steps = (DAY_STEPS - H) + DAY_STEPS
    target_date = last_ts.normalize() + pd.Timedelta(days=1)

    print(f"      截止时刻: {last_ts} (H={H}), 目标天: {target_date.date()}")

    # ── 自回归 rollout ──
    print("[4/4] 自回归推理")
    X = feature_scaler.transform(window_feat.values.astype(np.float32))
    window = torch.from_numpy(X).unsqueeze(0).to(device)
    last_hist_row = window_feat.iloc[-1]

    future_idx = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=freq_minutes),
        periods=total_steps, freq=resample_freq)
    n_hist_ext = min(len(df_clean), 800)
    hist_ext = processor.add_calendar_features(
        df_clean[[target_col]].iloc[-n_hist_ext:].copy())
    fut_ext = processor.add_calendar_features(
        pd.DataFrame({target_col: np.nan}, index=future_idx))
    ext = pd.concat([hist_ext, fut_ext])

    preds_scaled = []
    with torch.no_grad():
        for k in range(total_steps):
            feat_ext = processor.add_data_driven_features(ext)
            row = feat_ext.loc[future_idx[k], feature_cols].astype(np.float32)
            if row.isna().any():
                row = row.fillna(last_hist_row)
            row_scaled = feature_scaler.transform(row.values.reshape(1, -1))[0].astype(np.float32)

            one = model(window, target_len=1)
            pred_val = float(one[0, 0, 0].cpu())
            preds_scaled.append(pred_val)

            pred_orig = float(target_scaler.inverse_transform(
                np.array([[pred_val]], dtype=np.float64))[0, 0])
            ext.loc[future_idx[k], target_col] = pred_orig

            next_row = row_scaled.copy()
            next_row[target_feat_idx] = pred_val
            next_row_t = torch.from_numpy(
                next_row.astype(np.float32)).view(1, 1, -1).to(device)
            window = torch.cat([window[:, 1:, :], next_row_t], dim=1)

    # ── 反归一化, 取目标天 24 小时 ──
    preds_arr = np.array(preds_scaled, dtype=np.float32).reshape(1, total_steps, 1)
    y_inv = processor.inverse_transform_targets(preds_arr)[0]
    day_vals = y_inv[-DAY_STEPS:, 0]
    values = [round(max(0.0, float(v)), 1) for v in day_vals]

    # ── 返回 JSON 格式 ──
    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "provider": prov,
        "unit": UNIT,
        "interval_minutes": freq_minutes,
        "horizon": DAY_STEPS,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description="军山水厂流量预测 (CSV → JSON)")
    parser.add_argument("--data", default=DEFAULT_DATA, help="输入 CSV 文件路径")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="训练结果目录")
    parser.add_argument("--provider", default=None, help="provider 字段")
    parser.add_argument("--output", "-o", default=None, help="输出 JSON 文件路径 (可选)")
    args = parser.parse_args()

    resp = predict(args.data, args.result_dir, args.provider)

    text = json.dumps(resp, ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print("预测结果:")
    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(resp, f, ensure_ascii=False, indent=2)
        print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()

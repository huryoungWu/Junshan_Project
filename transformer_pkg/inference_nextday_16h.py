# -*- coding: utf-8 -*-
"""inference_nextday_16h.py — 每天 16 点预测次日全天 (nextday-16h 任务) 的推理脚本。

与 train_transformer_nextday_16h.py 完全配套 (同一窗口语义 / 同一 rollout 口径):

  输入: CSV (时间列 + 出厂水流量列, 训练/推理同一清洗管线 DataProcessor)
        —— 数据截止时刻即"16 点任务"的截止点: 若最后一条为 D 日 15:00,
        窗口 = 最近 184 小时 (前 7 天全天 + 当天 0:00~15:00), H=16 (主任务)。

  数据需求 (推理需要多少天):
    最少 ~16 天 (363 小时) = 窗口 184h + 30d 滚动特征 min_periods=180 的
    warmup 180h; 低于此清洗+dropna 后不足 184 行, 直接报错。
    31 天 = 30d 滚动窗满窗 (特征与全量数据计算口径完全一致)。
    建议 ≥ 35 天 (满窗 + 清洗剔除余量)。

  输入 CSV 的生成 (不再引用整体原始数据文件):
    默认流程: 从原始小时级数据 (--raw, 可多个用 ; 分隔) 读取 → 自动切出最近
    --days 天 → 生成新的小 CSV (input_nextday16h_日期_天数d.csv) → 对该新
    CSV 推理。也可用 --data 直接指定已备好的小 CSV (跳过切片)。
  输出: 接口 (dict / JSON):
        {
          "date": "2025-08-21",          # 目标天 D+1 (次日)
          "provider": "your_model",
          "unit": "m3/h",
          "interval_minutes": 60,
          "horizon": 24,
          "values": [2093.0, 1860.0, ..., 24 个, 单位 m3/h]
        }

  与 train_transformer_autoregressive.py 系 (inference_ar.py) 的区别:
    - 回看 = 7*24 + 16 = 184 步 (而非 168), 见 scaler.pkl 里 saved['lookback_steps']
    - rollout 真实步数 total = 缺口 (24-H) + 目标天 24 (主任务 H=16 → 32 步):
      先滚过 D 日 16:00~23:00 缺口, 再滚 D+1 全天, 取最后 24 步输出
    - 未来外生特征行 (日历已知 + 数据驱动特征依赖"已生成的预测") 每步在线重算,
      与训练 make_sequences_ar 的 Xf 口径一致 (部署时缺口步也是模型自己滚出来)

用法:
  # 默认: 直接读取已备好的切片 CSV (DEFAULT_DATA) → 预测次日 (打印接口 JSON);
  #       若原始数据已含目标天的真实值, 自动追加 预测 vs 实际 对比
  #       (MAE/MAPE + 对比曲线图 compare_YYYYMMDD.png + 明细 CSV)
  python inference_nextday_16h.py

  # 跳过对比
  python inference_nextday_16h.py --no-compare

  # 新数据到达: 从原始数据重新切片 → 生成新 CSV → 预测 (不指定 --data 时自动)
  python inference_nextday_16h.py --raw 水厂2025年小时级汇总.csv \
      --days 35 --provider junshan_transformer --out pred.json

  # 回测历史日 (如预测 2025-12-30, 用 2025 文件里的真实值对比):
  python inference_nextday_16h.py --raw 水厂2025年小时级汇总.csv \
      --cutoff "2025-12-29 15:00" --compare_raw 水厂2025年小时级汇总.csv

  # 直接指定目标日期 (程序自动计算截止时刻并从原始数据获取所需数据):
  python inference_nextday_16h.py --raw 水厂2025年小时级汇总.csv --target-date 2025-12-29

  # 批量预测日期范围 (生成每天的对比图和 MAE/MAPE, 以及汇总统计):
  python inference_nextday_16h.py --raw 水厂2025年小时级汇总.csv --target-date 2025-12-01:2025-12-31

  # 直接指定其他小 CSV
  python inference_nextday_16h.py --data input_nextday16h_20250820_35d.csv

  # 以 HTTP 接口服务 (POST /prepare 生成输入 CSV, POST /predict 预测)
  python inference_nextday_16h.py --serve --port 8000

  # 库调用
  from inference_nextday_16h import NextDayPredictor, prepare_input_csv
  csv_path = prepare_input_csv(r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv", days=35)
  p = NextDayPredictor("results/junshan_L1D_P24H_1h_transformer_nextday16h_mc_20260901_002524")
  resp = p.predict(csv_path)            # dict, 即上方接口 JSON
"""

import os
import sys
import pickle
import argparse
import json

import numpy as np
import pandas as pd
import torch

# GBK 控制台无法编码 ⚠ 等非 GBK 字符 → 统一改用 UTF-8 输出 (与训练脚本一致)
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor

HERE = os.path.dirname(os.path.abspath(__file__))

# 默认输入: 已备好的切片 CSV (由 prepare_input_csv 从 2024-2025 原始数据生成,
# 最近 35 天, 止于 2025-08-20 15:00 = 标准 16 点截止 → 预测 2025-08-21,
# 次日真实值在原始数据中已有, 可直接对比)。原始数据更新后重新生成新切片文件
# (python inference_nextday_16h.py --raw <新原始数据>) 并更新这里的路径。
DEFAULT_DATA = r"D:\Junshan_Project\data\input_nextday16h_20250820_35d.csv"
# 原始数据 (备用): 当 DEFAULT_DATA 不存在时, 从它切出最近 DEFAULT_SLICE_DAYS 天自动生成
DEFAULT_RAW = r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv"
DEFAULT_SLICE_DAYS = 35     # 默认切多少天: 35 = 30d 滚动窗满窗 + 清洗剔除余量
# 数据需求下限 (精确按小时): 窗口 184h + 30d 滚动特征 min_periods=180 warmup 180h
# = 363h ≈ 15.2 天; 低于此清洗+dropna 后不足 184 行
MIN_DATA_HOURS = 363
MIN_DATA_DAYS = 16          # 消息用: ~16 天
# 默认模型: 最新一次 nextday16h 多截止点训练
DEFAULT_RESULT_DIR = os.path.join(
    HERE, "results", "junshan_L1D_P24H_1h_transformer_nextday16h_mc_20260901_002524")

DAY_STEPS = 24              # 目标天小时数
UNIT = "m3/h"


class NextDayPredictor:
    """每天 16 点预测次日全天的推理接口 (与 train_transformer_nextday_16h.py 配套)。

    输入: CSV 路径
    输出: dict (接口 JSON):
          {date, provider, unit, interval_minutes, horizon, values}
    """

    def __init__(self, result_dir=DEFAULT_RESULT_DIR, device=None, provider=None):
        self.result_dir = result_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── 加载 scaler.pkl (训练时保存的 config / scaler / 特征列 / 自回归参数) ──
        scaler_path = os.path.join(result_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"未找到 scaler.pkl: {scaler_path}")
        with open(scaler_path, "rb") as f:
            saved = pickle.load(f)

        self.config = saved["config"]
        self.feature_scaler = saved["feature_scaler"]
        self.target_scaler = saved["target_scaler"]
        self.feature_cols = saved["feature_cols"]
        self.target_cols = saved["target_cols"]
        self.target_feat_idx = saved.get("target_feat_idx",
                                         self.feature_cols.index(self.target_cols[0]))
        if not saved.get("autoregressive", False):
            raise ValueError("scaler.pkl 里 autoregressive != True, "
                             "请用 train_transformer_nextday_16h.py 的训练结果")

        # ── 推理参数 (从 saved 里取, 不用重新推算) ──
        self.lookback_steps = saved.get("lookback_steps")
        if self.lookback_steps is None:          # 老 scaler 兜底
            lookback_days = self.config["lookback_days"]
            lookback_extra = self.config.get("lookback_extra_hours", 0)
            freq_minutes = int(self.config["resample_freq"].replace("min", ""))
            points_per_day = (24 * 60) // freq_minutes
            self.lookback_steps = int(lookback_days * points_per_day) + int(lookback_extra)
        self.resample_freq = self.config["resample_freq"]
        self.freq_minutes = int(self.resample_freq.replace("min", ""))
        self.predict_steps_max = saved.get("predict_steps_max", 48)
        self.target_col = self.target_cols[0]

        # ── 模型工厂 (与训练完全一致: 单步头 horizon=1, 由 rollout 滚出全天) ──
        model_type = self.config.get("model_type", "transformer")
        model_kwargs = dict(
            input_dim=len(self.feature_cols),
            output_dim=1,
            horizon=1,                       # ← 单步头: 只预测下一个点
            input_len=self.lookback_steps,
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["transformer_dropout"],
        )
        if model_type == "itransformer":
            self.model = iTransformer(**model_kwargs,
                                      target_idx=self.target_feat_idx).to(self.device)
        else:
            self.model = TimeSeriesTransformer(**model_kwargs).to(self.device)

        model_path = os.path.join(result_dir, "best_seq2seq_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型权重: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # ── 复用训练的 DataProcessor (清洗/特征口径一致, 含 spike_abs_floor 等开关) ──
        self.processor = DataProcessor(self.config)
        self.processor.feature_scaler = self.feature_scaler
        self.processor.target_scaler = self.target_scaler
        self.processor.feature_cols = self.feature_cols

        # provider 默认 = 模型标识
        self.provider = provider or f"junshan_{model_type}_nextday16h"

        print(f"[NextDayPredictor] 模型已加载: {os.path.basename(model_path)}")
        print(f"[NextDayPredictor] model_type={model_type}, lookback={self.lookback_steps}步"
              f" (7天+16h), 特征数={len(self.feature_cols)}, device={self.device}")
        print(f"[NextDayPredictor] 任务: 截止时刻 H=16 时滚 8h 缺口 + 次日 24h = 32 步,"
              f" 取最后 {DAY_STEPS} 步为输出")

    # ── 核心推理: CSV → 接口 JSON ──

    def predict(self, csv_path, encoding="utf-8-sig"):
        """读 CSV → 清洗 → 特征 → 自回归 rollout → 返回接口 JSON (dict)。"""
        print(f"\n[1/5] 读取 CSV: {csv_path}")
        df_raw = pd.read_csv(csv_path, encoding=encoding)

        # 解析时间列 → DatetimeIndex (DataProcessor 的清洗/特征都依赖时间索引)
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            for ts_col in ("时间", "timestamp"):
                if ts_col in df_raw.columns:
                    df_raw[ts_col] = pd.to_datetime(df_raw[ts_col])
                    df_raw = df_raw.set_index(ts_col)
                    break
            else:
                raise ValueError("输入 CSV 必须包含 时间 / timestamp 列")

        print("[2/5] 数据清洗 + 特征构建 (与训练 build_feature_table 口径一致)")
        df_base = self.processor.build_base_features(df_raw)
        df_clean = self.processor.clean_and_resample(df_base)
        df_clean = self._fill_day_gaps(df_clean)
        df_feat = self.processor.add_calendar_features(df_clean)
        df_feat = self.processor.add_data_driven_features(df_feat)

        # 特征列校验
        missing = [c for c in self.feature_cols if c not in df_feat.columns]
        if missing:
            raise ValueError(f"特征列缺失 (训练与推理不匹配): {missing}")

        df_feat = df_feat[self.feature_cols].dropna()
        if len(df_feat) < self.lookback_steps:
            raise ValueError(
                f"清洗+dropna 后仅 {len(df_feat)} 行, 不足 lookback={self.lookback_steps} 步 "
                f"(最少需 ~363 小时 ≈ {MIN_DATA_DAYS} 天: 窗口 184h + 30d 滚动特征 "
                f"min_periods=180 的 warmup 180h; 建议用 prepare_input_csv 切出 "
                f"≥ {DEFAULT_SLICE_DAYS} 天)")
        print(f"      清洗+dropna 后: {len(df_feat)} 行, 时间 {df_feat.index.min()} ~ {df_feat.index.max()}")

        # ── 回看窗口: 最后 lookback_steps 行, 校验小时连续 (训练要求窗口无空洞) ──
        window_feat = df_feat.iloc[-self.lookback_steps:]
        self._check_window_continuous(window_feat)

        # ── 截止时刻语义: 窗口末尾 = 截止点; H = 当天小时数+1 ──
        last_ts = window_feat.index[-1]
        H = int(last_ts.hour) + 1
        total_steps = (DAY_STEPS - H) + DAY_STEPS          # 缺口 (24-H) + 目标天 24
        if H != 16:
            print(f"      ⚠ 截止时刻 {last_ts} (H={H}) ≠ 主任务 16 点 (H=16); "
                  f"模型为多截止点训练, 仍可推理, 缺口滚 {DAY_STEPS - H} 步")
        target_date = last_ts.normalize() + pd.Timedelta(days=1)   # 目标天 = D+1
        print(f"      截止时刻: {last_ts} (H={H}), 缺口={DAY_STEPS - H}h, "
              f"目标天={target_date.date()}, rollout 共 {total_steps} 步")

        # ── 窗口缩放 → (1, L, C) ──
        print("[3/5] 自回归 rollout (先滚缺口再滚目标天)")
        X = self.feature_scaler.transform(window_feat.values.astype(np.float32))
        window = torch.from_numpy(X).unsqueeze(0).to(self.device)   # (1, L, C)
        last_hist_row = window_feat.iloc[-1]                        # NaN 兜底行

        # ── 未来时间轴 + 在线特征重建的扩展表 (历史flow + 未来占位) ──
        future_idx = pd.date_range(
            start=last_ts + pd.Timedelta(minutes=self.freq_minutes),
            periods=total_steps, freq=self.resample_freq)
        n_hist_ext = min(len(df_clean), 800)         # 覆盖 30d 滚动窗 720 + 余量
        hist_ext = self.processor.add_calendar_features(
            df_clean[[self.target_col]].iloc[-n_hist_ext:].copy())
        fut_ext = self.processor.add_calendar_features(
            pd.DataFrame({self.target_col: np.nan}, index=future_idx))
        ext = pd.concat([hist_ext, fut_ext])         # 未来 flow=NaN; 每步回灌预测

        # ── rollout: 数据驱动特征依赖"已生成的预测", 每步回灌原始流量重算
        #    与训练 autoregressive_rollout 一致: 目标通道用预测覆盖, 其余用真值/重算 ──
        preds_scaled = []
        with torch.no_grad():
            for k in range(total_steps):
                feat_ext = self.processor.add_data_driven_features(ext)
                row = feat_ext.loc[future_idx[k], self.feature_cols].astype(np.float32)
                if row.isna().any():                 # 兜底: 边缘 min_periods 缺值
                    row = row.fillna(last_hist_row)
                row_scaled = self.feature_scaler.transform(
                    row.values.reshape(1, -1))[0].astype(np.float32)

                one = self.model(window, target_len=1)     # (1, 1, 1) 单步预测
                pred_val = float(one[0, 0, 0].cpu())
                preds_scaled.append(pred_val)

                # 预测回灌 ext (转原始流量域, 因 data-driven 特征在原始域计算;
                #   feature_scaler 对 Total_Flow 的缩放与 target_scaler 一致)
                pred_orig = float(self.processor.target_scaler.inverse_transform(
                    np.array([[pred_val]], dtype=np.float64))[0, 0])
                ext.loc[future_idx[k], self.target_col] = pred_orig

                # 滑窗: 末尾拼 future 行, 目标通道用 scaled 预测覆盖 (与训练一致)
                next_row = row_scaled.copy()
                next_row[self.target_feat_idx] = pred_val
                next_row_t = torch.from_numpy(
                    next_row.astype(np.float32)).view(1, 1, -1).to(self.device)
                window = torch.cat([window[:, 1:, :], next_row_t], dim=1)

        # ── 反归一化, 取最后 24 步 = 目标天 D+1 0:00~23:00 ──
        print("[4/5] 反归一化 + 提取目标天 24 小时")
        preds_arr = np.array(preds_scaled, dtype=np.float32).reshape(1, total_steps, 1)
        y_inv = self.processor.inverse_transform_targets(preds_arr)[0]   # (total, 1)
        day_vals = y_inv[-DAY_STEPS:, 0]                                 # 目标天 24 个
        values = [round(max(0.0, float(v)), 1) for v in day_vals]        # 流量≥0, 保留1位

        print(f"      输出: {target_date.date()} 0:00~23:00, 24 个点, "
              f"min={min(values):.1f}, max={max(values):.1f}")

        # ── 接口 JSON ──
        print("[5/5] 生成接口结果")
        return {
            "date": target_date.strftime("%Y-%m-%d"),
            "provider": self.provider,
            "unit": UNIT,
            "interval_minutes": self.freq_minutes,
            "horizon": DAY_STEPS,
            "values": values,
        }

    # ── 批量预测: 日期范围 ──

    def predict_date_range(self, start_date, end_date, raw_csv, days=DEFAULT_SLICE_DAYS,
                           encoding="utf-8-sig", save_dir=None):
        """批量预测日期范围内每一天, 生成每天的对比图和 MAE/MAPE, 最后给出汇总统计。

        Args:
            start_date: 起始日期 (如 "2025-12-01")
            end_date: 结束日期 (如 "2025-12-31")
            raw_csv: 原始小时级数据文件路径
            days: 从原始数据切出的天数 (默认 35)
            encoding: CSV 编码
            save_dir: 保存目录 (默认 = 训练结果目录)
        Returns:
            dict: {results: [每天的结果], summary: {avg_mae, avg_mape, ...}}
        """
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        date_range = pd.date_range(start, end, freq='D')

        print(f"\n{'='*60}")
        print(f"批量预测: {start_date} ~ {end_date} (共 {len(date_range)} 天)")
        print(f"{'='*60}\n")

        results = []
        save_dir = save_dir or self.result_dir
        os.makedirs(save_dir, exist_ok=True)

        for i, target_date in enumerate(date_range):
            print(f"\n[{i+1}/{len(date_range)}] 预测 {target_date.date()}")
            print("-" * 40)

            try:
                # 准备输入 CSV (截止时刻 = 目标日期前一天 15:00)
                csv_path = prepare_input_csv(raw_csv, days=days, target_date=target_date,
                                            encoding=encoding)

                # 预测
                resp = self.predict(csv_path, encoding=encoding)

                # 对比
                cmp_result = self.compare_with_actual(resp, raw_csv=raw_csv,
                                                     save_dir=save_dir, fallback_csv=csv_path)

                if cmp_result:
                    results.append(cmp_result)
                else:
                    print(f"[跳过] {target_date.date()} 无法获取实际数据进行对比")
            except Exception as e:
                print(f"[错误] {target_date.date()} 预测失败: {e}")
                continue

        # ── 汇总统计 ──
        if not results:
            print("\n没有成功预测的日期, 无法生成汇总统计")
            return {"results": [], "summary": None}

        mae_values = [r["mae"] for r in results if not np.isnan(r["mae"])]
        mape_values = [r["mape"] for r in results if not np.isnan(r["mape"])]
        mape_raw_values = [r["mape_raw"] for r in results if not np.isnan(r["mape_raw"])]

        # 数据点级别的 MAPE (与训练代码一致: 把所有天的所有小时展平后一起计算)
        all_y_true = []
        all_y_pred = []
        for r in results:
            # 从明细 CSV 读取每天的数据点
            csv_path = r.get("csv_path")
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                all_y_true.extend(df["actual"].values)
                all_y_pred.extend(df["predict"].values)

        if all_y_true:
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            floor_ratio = self.config.get("mape_floor_ratio", 0.05)
            thr = floor_ratio * np.abs(all_y_true).max()
            keep = np.abs(all_y_true) >= thr
            n_total = len(all_y_true)
            n_used = int(keep.sum())
            if n_used > 0:
                mape_pointwise = float(np.mean(
                    np.abs((all_y_true[keep] - all_y_pred[keep]) / (all_y_true[keep] + 1e-8)) * 100))
            else:
                mape_pointwise = float("nan")
            mape_pointwise_raw = float(np.mean(
                np.abs((all_y_true - all_y_pred) / (all_y_true + 1e-8)) * 100))
            # 调试信息
            print(f"\n[调试] floor_ratio={floor_ratio}, thr={thr:.2f}")
            print(f"[调试] y_true 范围: {np.min(all_y_true):.2f} ~ {np.max(all_y_true):.2f}")
            print(f"[调试] 过滤前点数: {n_total}, 过滤后点数: {n_used}")
        else:
            mape_pointwise = float("nan")
            mape_pointwise_raw = float("nan")
            n_total = 0
            n_used = 0

        summary = {
            "start_date": start_date,
            "end_date": end_date,
            "n_days": len(results),
            "avg_mae": np.mean(mae_values) if mae_values else float("nan"),
            "std_mae": np.std(mae_values) if mae_values else float("nan"),
            "min_mae": np.min(mae_values) if mae_values else float("nan"),
            "max_mae": np.max(mae_values) if mae_values else float("nan"),
            "avg_mape": np.mean(mape_values) if mape_values else float("nan"),
            "std_mape": np.std(mape_values) if mape_values else float("nan"),
            "min_mape": np.min(mape_values) if mape_values else float("nan"),
            "max_mape": np.max(mape_values) if mape_values else float("nan"),
            "avg_mape_raw": np.mean(mape_raw_values) if mape_raw_values else float("nan"),
            "mape_pointwise": mape_pointwise,
            "mape_pointwise_raw": mape_pointwise_raw,
            "mape_n_total": n_total,
            "mape_n_used": n_used,
        }

        # 打印汇总
        print(f"\n{'='*60}")
        print(f"汇总统计 ({start_date} ~ {end_date})")
        print(f"{'='*60}")
        print(f"成功预测天数: {summary['n_days']}")
        print(f"\n【日级别统计】(每天计算MAPE后取平均)")
        print(f"MAE:  平均={summary['avg_mae']:.2f} m³/h, "
              f"标准差={summary['std_mae']:.2f}, "
              f"最小={summary['min_mae']:.2f}, 最大={summary['max_mae']:.2f}")
        print(f"MAPE: 平均={summary['avg_mape']:.2f}%, "
              f"标准差={summary['std_mape']:.2f}%, "
              f"最小={summary['min_mape']:.2f}%, 最大={summary['max_mape']:.2f}%")
        print(f"MAPE (不过滤近零点): 平均={summary['avg_mape_raw']:.2f}%")
        print(f"\n【数据点级别统计】(与训练代码一致: 所有天所有小时展平后计算)")
        print(f"MAPE (过滤近零点): {summary['mape_pointwise']:.2f}%"
              f" (保留 {summary['mape_n_used']}/{summary['mape_n_total']} 个点)")
        print(f"MAPE (不过滤近零点): {summary['mape_pointwise_raw']:.2f}%")

        # 保存汇总到 CSV
        summary_df = pd.DataFrame(results)
        summary_csv_path = os.path.join(save_dir,
                                        f"summary_{start:%Y%m%d}_{end:%Y%m%d}.csv")
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
        print(f"\n汇总明细已保存: {summary_csv_path}")

        # 保存汇总统计到 JSON
        summary_json_path = os.path.join(save_dir,
                                         f"summary_{start:%Y%m%d}_{end:%Y%m%d}.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"汇总统计已保存: {summary_json_path}")

        return {"results": results, "summary": summary}

    # ── 预测 vs 实际对比 ──

    def compare_with_actual(self, resp, raw_csv=DEFAULT_RAW, encoding="utf-8-sig",
                            save_dir=None, fallback_csv=None):
        """预测结果 vs 真实值对比: 从原始数据读取目标天的实际流量, 计算 MAE/MAPE,
        画对比曲线图 + 保存逐小时明细 CSV。

        实际值经与训练完全相同的清洗管线 (突变修正 + 中值平滑 + Hampel,
        由 scaler.pkl 的 config 决定), 口径与训练 y_true 一致 —— 模型学的是
        清洗后的流量, 对比清洗值才是模型真实误差 (含仪表噪声会虚高)。
        MAPE 与训练口径一致: 按 config['mape_floor_ratio'] 过滤近零点 (另给
        不过滤的参考值)。实际数据不足 24 小时时按已有小时对比并明确提示。

        Args:
            resp: predict() 的返回 dict (需含 date / values)
            raw_csv: 原始小时级数据 (可多个用 ; 分隔), 目标天实际值从这里读取
            save_dir: 对比图/明细 CSV 保存目录 (默认 = 训练结果目录)
            fallback_csv: 兜底数据源 (如输入切片 CSV 已含目标天但原始文件未更新)
        Returns:
            dict {date, n_hours, mae, mape, mape_raw, mape_used, plot_path, csv_path}
            原始数据中无目标天实际值时返回 None (提示跳过)。
        """
        target_date = pd.Timestamp(resp["date"]).normalize()
        y_pred = np.asarray(resp["values"], dtype=float)

        # ── 读取实际值: 优先原始数据, 未找到时兜底用输入 CSV ──
        day_clean = None
        src_ranges = []        # 各数据源覆盖范围 (用于"找不到"时的诊断提示)
        for src in (str(raw_csv).split(";") + ([str(fallback_csv)] if fallback_csv else [])):
            src = src.strip()
            if not src or not os.path.exists(src):
                continue
            df = pd.read_csv(src, encoding=encoding)
            ts_col = next((c for c in ("时间", "timestamp") if c in df.columns), None)
            if ts_col is None:
                continue
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.set_index(ts_col).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            try:
                df_clean = self.processor.clean_and_resample(
                    self.processor.build_base_features(df))
            except Exception:
                continue
            src_ranges.append((src, df_clean.index.min(), df_clean.index.max()))
            mask = df_clean.index.normalize() == target_date
            if mask.sum() > 0:
                day_clean = df_clean.loc[mask, self.target_col]
                break

        if day_clean is None:
            print(f"[对比] 未找到 {target_date.date()} 的实际数据, 跳过:")
            for src, t0, t1 in src_ranges:
                print(f"       - {src}: {t0} ~ {t1}")
            if src_ranges and src_ranges[-1][2] < target_date:
                print(f"       → 目标天 {target_date.date()} 在数据源截止时间之后: "
                      f"该天真实数据入库后重跑即可对比; "
                      f"如需回测历史日, 用 --cutoff 指定截止时刻")
            return None

        n_avail = len(day_clean)
        n = min(n_avail, 24)
        y_true = day_clean.iloc[:n].to_numpy(dtype=float)
        y_pred = y_pred[:n]
        if n < 24:
            print(f"[对比] ⚠ 实际数据仅 {n}/24 小时 (目标天数据不完整), 按已有小时对比")

        # ── 指标: MAE / MAPE (训练口径: 按 mape_floor_ratio 过滤近零点) ──
        floor_ratio = self.config.get("mape_floor_ratio", 0.05)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        thr = floor_ratio * np.abs(y_true).max()
        keep = np.abs(y_true) >= thr
        n_used = int(keep.sum())
        n_total = len(y_true)
        if n_used > 0:
            mape = float(np.mean(
                np.abs((y_true[keep] - y_pred[keep]) / (y_true[keep] + 1e-8)) * 100))
        else:
            mape = float("nan")
        mape_raw = float(np.mean(
            np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100))     # 不过滤参考值

        print(f"\n[对比] {target_date.date()} 预测 vs 实际 ({n} 小时):")
        print(f"       MAE = {mae:.2f} m³/h")
        print(f"       MAPE = {mape:.2f}%"
              + (f" (过滤 |实际| < {thr:.1f} 的近零点, 用 {n_used}/{n} 点; "
                 f"不过滤参考: {mape_raw:.2f}%)" if n_used < n
                 else f" (不过滤: {mape_raw:.2f}%)"))

        # ── 对比曲线图 (与训练画图风格一致: 实际=深蓝实线, 预测=红虚线) ──
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        hours = np.arange(n)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(hours, y_true, color="#2c3e50", linewidth=1.8,
                marker="o", ms=4, label="实际流量")
        ax.plot(hours, y_pred, color="#e74c3c", linewidth=1.8,
                linestyle="--", marker="s", ms=4, label="预测流量")
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h:02d}:00" for h in hours])
        ax.set_xlabel(f"时刻 (目标天 {target_date.date()})")
        ax.set_ylabel("出厂水流量 (m³/h)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_title(f"{target_date.date()} 流量预测 vs 实际   "
                     f"MAE={mae:.1f} m³/h  MAPE={mape:.1f}%")
        fig.tight_layout()

        save_dir = save_dir or self.result_dir
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"compare_{target_date:%Y%m%d}.png")
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ── 逐小时明细 CSV ──
        cmp_df = pd.DataFrame({
            "actual": y_true, "predict": y_pred, "abs_err": np.abs(y_true - y_pred),
        }, index=day_clean.index[:n])
        cmp_df.index.name = "timestamp"
        csv_path = os.path.join(save_dir, f"compare_{target_date:%Y%m%d}.csv")
        cmp_df.to_csv(csv_path, encoding="utf-8-sig")

        print(f"[对比] 曲线图: {plot_path}")
        print(f"[对比] 逐小时明细: {csv_path}")
        return {"date": str(target_date.date()), "n_hours": n, "mae": mae,
                "mape": mape, "mape_raw": mape_raw, "mape_used": n_used,
                "plot_path": plot_path, "csv_path": csv_path}

    # ── 工具 ──

    def _check_window_continuous(self, window_feat):
        """窗口内时间必须逐小时连续 (训练 make_sequences_ar 对空洞样本直接跳过,
        推理时窗口有空洞 → 特征错位, 直接报错提示补数据)。"""
        if len(window_feat) < self.lookback_steps:
            raise ValueError(f"窗口不足 {self.lookback_steps} 步: {len(window_feat)}")
        step_ns = int(pd.Timedelta(minutes=self.freq_minutes).total_seconds()) * 10**9
        deltas = np.diff(window_feat.index.asi8)
        if not np.all(deltas == step_ns):
            bad = np.where(deltas != step_ns)[0][0]
            raise ValueError(
                f"回看窗口存在时间空洞: {window_feat.index[bad]} → "
                f"{window_feat.index[bad + 1]} (应逐{self.freq_minutes}连续)。"
                f"请补全缺失小时或缩短截止时间后再推理")

    def _fill_day_gaps(self, df_clean):
        """日级缺口填充: 缺失 bin 用最近一天同时段数据补齐 (与 inference_ar.py 一致)。"""
        if df_clean is None or df_clean.empty:
            return df_clean

        grid_start = df_clean.index.min().normalize()
        # 网格只到"最后一条真实数据": 尾部缺失 bin 是截止点之后的未来 (如 CSV
        # 天然止于 15:00), 不是空洞, 不能用昨天同时段虚构 —— 否则截止时刻被
        # 平移到 23:00, H=16 主任务语义被破坏。只补"两侧都有真实数据"的内部空洞。
        grid_end = df_clean.index.max()
        expected = pd.date_range(grid_start, grid_end, freq=self.resample_freq)
        if len(expected) == len(df_clean):
            return df_clean

        present = set(df_clean.index)
        missing = expected.difference(df_clean.index)

        filled = {}
        max_dist = (expected[-1].normalize() - expected[0].normalize()).days
        for s in missing:
            d, t = s.normalize(), s - s.normalize()
            for dist in range(1, max_dist + 1):
                for cand in (d - pd.Timedelta(days=dist), d + pd.Timedelta(days=dist)):
                    src = cand + t
                    if src in present:
                        filled[s] = df_clean.loc[src]
                        break
                if s in filled:
                    break

        if filled:
            df_clean = pd.concat([df_clean, pd.DataFrame(filled).T]).sort_index()
            print(f"      [fill_day_gaps] 补齐 {len(filled)} 个缺失 bin "
                  f"(网格 {len(expected)} 槽位)")
        return df_clean


# ── 输入 CSV 生成: 原始数据 → 切片 → 新 CSV (不引用整体原始文件) ──

def prepare_input_csv(raw_csv, days=DEFAULT_SLICE_DAYS, out_path=None,
                      encoding="utf-8-sig", cutoff=None, target_date=None):
    """从原始小时级数据读取 → 切出最近 days 天 → 生成新的输入 CSV (格式与原文件一致)。

    数据需求 (针对 nextday-16h 模型):
      - 最少 ~16 天 (363h): 窗口 184h + 30d 滚动特征 min_periods=180 的 warmup
        180h; 低于此清洗+dropna 后不足 184 行, 推理直接报错
      - 31 天: 30d 滚动窗满窗 (特征与全量数据计算口径完全一致)
      - 建议 ≥ 35 天 (满窗 + 清洗剔除余量)

    Args:
        raw_csv: 原始小时级数据路径; 可传多个, 用 ";" 分隔
        days: 从截止时刻往前切多少天
        cutoff: 切片截止时刻 (如 "2025-12-29 15:00"), 用于回测历史日的预测
                (None = 数据最后一条, 即当前日期); 截止日 = 切片最后一天,
                预测目标 = 截止日次日, 可与原始数据里的真实值对比
        target_date: 目标预测日期 (如 "2025-12-29"), 程序自动计算截止时刻为
                     该日期前一天的 15:00 (与主任务 H=16 一致)
        out_path: 新 CSV 输出路径 (None = 自动: 原始数据同目录
                  input_nextday16h_日期_天数d.csv)
    返回: 生成的 CSV 路径 (传给 NextDayPredictor.predict)
    """
    files = [f.strip() for f in str(raw_csv).split(";") if f.strip()]
    frames = []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"原始数据文件不存在: {f}")
        df = pd.read_csv(f, encoding=encoding)
        ts_col = next((c for c in ("时间", "timestamp") if c in df.columns), None)
        if ts_col is None:
            raise ValueError(f"原始数据必须包含 时间 / timestamp 列: {f}")
        df[ts_col] = pd.to_datetime(df[ts_col])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates().sort_values(ts_col)

    last_ts = df[ts_col].iloc[-1]
    if target_date is not None:
        # 目标日期模式: 用户指定想要预测的日期, 程序自动计算截止时刻
        target_ts = pd.Timestamp(target_date)
        # 截止时刻 = 目标日期前一天的 15:00 (与主任务 H=16 一致)
        cutoff_ts = (target_ts - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=15)
        if not (df[ts_col].iloc[0] <= cutoff_ts <= last_ts):
            raise ValueError(
                f"目标日期 {target_date} 对应的截止时刻 {cutoff_ts} 超出原始数据范围 "
                f"({df[ts_col].iloc[0]} ~ {last_ts})。请确保原始数据包含 {cutoff_ts} 之前的数据")
        last_ts = cutoff_ts
        print(f"[prepare] 目标日期模式: 预测目标 = {target_ts.date()}, "
              f"截止时刻 = {last_ts} (H=16)")
    elif cutoff is not None:
        cutoff_ts = pd.Timestamp(cutoff)
        if not (df[ts_col].iloc[0] <= cutoff_ts <= last_ts):
            raise ValueError(f"cutoff={cutoff} 超出原始数据范围 "
                             f"({df[ts_col].iloc[0]} ~ {last_ts})")
        last_ts = cutoff_ts
        print(f"[prepare] 回测模式: 截止时刻 = {last_ts} (预测目标 = 次日)")
    start = (last_ts - pd.Timedelta(days=days - 1)).normalize()   # 整日对齐
    ts_arr = df[ts_col].to_numpy()
    cut = int(np.searchsorted(ts_arr, np.datetime64(start)))
    cut_end = int(np.searchsorted(ts_arr, np.datetime64(last_ts), side="right"))
    df = df.iloc[cut:cut_end]

    n_hours = len(df)
    n_days = n_hours / 24.0
    if n_hours < MIN_DATA_HOURS:
        raise ValueError(
            f"切片后仅 {n_hours} 小时 ({n_days:.1f} 天), 少于最少 {MIN_DATA_HOURS} 小时 "
            f"≈ {MIN_DATA_DAYS} 天 (窗口 184h + 30d 滚动特征 warmup 180h = 363h)。"
            f"请提供更多历史数据")

    if out_path is None:
        out_path = os.path.join(os.path.dirname(files[0]) or ".",
                                f"input_nextday16h_{last_ts:%Y%m%d}_{n_days:.0f}d.csv")
    df.to_csv(out_path, index=False, encoding=encoding)

    print(f"[prepare] 原始数据: {'; '.join(files)}")
    print(f"[prepare] 切片: 最近 {n_days:.1f} 天 ({n_hours} 小时), 止于 {last_ts}")
    print(f"[prepare] 数据需求: 最少 {MIN_DATA_DAYS} 天 (363h) / 31 天满 30d 滚动窗 / "
          f"建议 ≥ {DEFAULT_SLICE_DAYS} 天")
    print(f"[prepare] 已生成新输入 CSV: {out_path}")
    return out_path


# ── HTTP 接口 (可选, 依赖 fastapi + uvicorn) ──

def serve(result_dir=DEFAULT_RESULT_DIR, host="0.0.0.0", port=8000, provider=None):
    """以 HTTP 服务方式提供预测接口。

    POST /predict   body: {"csv_path": "/path/to/data.csv"}
                    返回: {date, provider, unit, interval_minutes, horizon, values}
    """
    try:
        from typing import Optional
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
    except ImportError as e:
        raise ImportError("--serve 需要 fastapi + uvicorn (pip install fastapi uvicorn)") from e

    predictor = NextDayPredictor(result_dir, provider=provider)

    app = FastAPI(title="军山出厂水流量预测 API (nextday-16h)",
                  description="每天16点预测次日全天24小时流量")

    class PredictRequest(BaseModel):
        csv_path: str
        result_dir: Optional[str] = None
        provider: Optional[str] = None

    class PrepareRequest(BaseModel):
        raw_path: str
        days: int = DEFAULT_SLICE_DAYS
        out_path: Optional[str] = None

    @app.get("/")
    def info():
        return {"service": "junshan-flow-predictor", "task": "nextday-16h",
                "unit": UNIT, "interval_minutes": predictor.freq_minutes,
                "horizon": DAY_STEPS, "model": os.path.basename(predictor.result_dir)}

    @app.post("/prepare")
    def prepare(req: PrepareRequest):
        """原始数据 → 切片 → 生成新输入 CSV (不引用整体原始文件)。"""
        csv_path = prepare_input_csv(req.raw_path, days=req.days, out_path=req.out_path)
        return {"csv_path": csv_path, "message": "已生成推理输入 CSV, 可传给 /predict"}

    @app.post("/predict")
    def predict(req: PredictRequest):
        p = predictor
        if req.result_dir or req.provider:           # 每请求独立实例 (可不同模型)
            p = NextDayPredictor(req.result_dir or predictor.result_dir,
                                 provider=req.provider)
        return p.predict(req.csv_path)

    print(f"* FastAPI 服务启动: http://{host}:{port}  "
          f"(POST /prepare 生成输入, POST /predict 预测)")
    uvicorn.run(app, host=host, port=port)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="每天16点预测次日全天: 原始数据切片 → 新 CSV → 接口 JSON")
    parser.add_argument("--data", default=None,
                        help="指定输入 CSV (已备好的小切片文件); "
                             "不指定时默认读取 DEFAULT_DATA, 若不存在则从 --raw 切片生成")
    parser.add_argument("--raw", default=DEFAULT_RAW,
                        help="原始小时级数据文件 (可多个用 ; 分隔)")
    parser.add_argument("--days", type=int, default=DEFAULT_SLICE_DAYS,
                        help="从原始数据切出的天数 (最少 16; 31=30d 滚动窗满窗; "
                             "建议 35)")
    parser.add_argument("--cutoff", default=None,
                        help="切片截止时刻 (如 '2025-12-29 15:00'): 回测历史日的预测 —— "
                             "预测截止日次日, 再用原始数据里的真实值对比 "
                             "(默认 = 数据最后一条)")
    parser.add_argument("--target-date", default=None,
                        help="目标预测日期 (如 '2025-12-29'); 或日期范围 (如 '2025-12-01:2025-12-31')"
                             "进行批量预测, 程序自动计算截止时刻为该日期前一天的 15:00, "
                             "并从原始数据获取所需数据进行预测")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR,
                        help="训练结果目录 (含 scaler.pkl + best_seq2seq_model.pth)")
    parser.add_argument("--provider", default=None, help="接口里的 provider 字段")
    parser.add_argument("--out", default=None, help="JSON 输出文件路径 (默认只打印)")
    parser.add_argument("--no-compare", action="store_true",
                        help="跳过 预测 vs 实际 对比 (默认: 原始数据含目标天真实值时自动对比)")
    parser.add_argument("--compare_raw", default=None,
                        help="对比用原始数据文件 (默认 DEFAULT_RAW; 可多个用 ; 分隔)")
    parser.add_argument("--serve", action="store_true", help="以 HTTP 接口服务方式运行")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.serve:
        serve(args.result_dir, args.host, args.port, args.provider)
        return

    predictor = NextDayPredictor(args.result_dir, provider=args.provider)
    if args.data:
        csv_path = args.data
        print(f"[main] 使用指定输入 CSV: {csv_path}")
    elif args.target_date:
        # 检查是否是日期范围 (格式: 2025-12-01:2025-12-31 或 2025-12-01~2025-12-31)
        if ':' in args.target_date or '~' in args.target_date:
            separator = ':' if ':' in args.target_date else '~'
            start_date, end_date = args.target_date.split(separator, 1)
            result = predictor.predict_date_range(
                start_date.strip(), end_date.strip(), args.raw,
                days=args.days, save_dir=args.result_dir)
            # 保存汇总结果
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"已保存: {args.out}")
            return
        else:
            csv_path = prepare_input_csv(args.raw, days=args.days, target_date=args.target_date)
    elif args.cutoff:
        csv_path = prepare_input_csv(args.raw, days=args.days, cutoff=args.cutoff)
    elif os.path.exists(DEFAULT_DATA):
        csv_path = DEFAULT_DATA
        print(f"[main] 使用默认输入 CSV: {csv_path}")
    else:
        print(f"[main] 默认输入不存在 ({DEFAULT_DATA}), 从原始数据切片生成…")
        csv_path = prepare_input_csv(args.raw, days=args.days)
    resp = predictor.predict(csv_path)

    text = json.dumps(resp, ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print("预测接口输出:")
    print(text)

    # 预测 vs 实际对比 (默认开启; 原始数据未更新到目标天时自动跳过)
    if not args.no_compare:
        predictor.compare_with_actual(resp, raw_csv=args.compare_raw or DEFAULT_RAW,
                                      fallback_csv=csv_path)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(resp, f, ensure_ascii=False, indent=2)
        print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()

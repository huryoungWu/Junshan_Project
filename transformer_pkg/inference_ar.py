"""单步自回归推理脚本 — 与 train_transformer_autoregressive.py 完全配套。

用法:
  # 默认: 加载最新训练结果, 对默认 CSV 推理
  python inference_ar.py

  # 指定结果目录和输入数据
  python inference_ar.py --result_dir results/xxx --data input.csv

  # 输出压力预测
  python inference_ar.py --with_pressure

  # 命令行直接调用
  from inference_ar import FlowPredictor
  p = FlowPredictor("results/junshan_L1D_P24H_1h_itransformer_autoregressive_test")
  pred = p.predict("data.csv")               # DataFrame
  pred = p.predict("data.csv", as_list=True)  # list[dict]
"""

import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = r"D:\Junshan_Project\transformer_pkg\input.csv"
DEFAULT_RESULT_DIR = os.path.join(HERE, "results",
                                  "junshan_L1D_P24H_1h_transformer_autoregressive_20260828_141342")

# 分时压力默认参数
DEFAULT_PRESSURE_SCHEDULE = [
    (0, 5, 0.30),
    (5, 12, 0.33),
    (12, 16, 0.33),
    (16, 23, 0.33),
    (23, 24, 0.30),
]


class FlowPredictor:
    """自回归流量预测推理接口。

    与 train_transformer_autoregressive.py 完全配套:
      - 复用 DataProcessor 做数据清洗 + 日历特征 (口径一致)
      - horizon=1 单步头 + autoregressive rollout 滚出完整预测
      - scaler.pkl 含 autoregressive=True / target_feat_idx / config

    输入: CSV 路径 / DataFrame / list[dict]
    输出: DataFrame (index=timestamp, col=Total_Flow) 或 list[dict]
    """

    def __init__(self, result_dir=DEFAULT_RESULT_DIR, device=None):
        self.result_dir = result_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 scaler.pkl
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
            raise ValueError("scaler.pkl 里 autoregressive != True, 请用配套的自回归训练结果")

        # 推理参数
        self.lookback_days = self.config["lookback_days"]
        self.predict_days = self.config["predict_days"]
        self.resample_freq = self.config["resample_freq"]
        self.freq_minutes = int(self.resample_freq.replace("min", ""))
        self.points_per_day = (24 * 60) // self.freq_minutes
        self.lookback_steps = int(self.lookback_days * self.points_per_day)
        self.predict_steps = int(self.predict_days * self.points_per_day)

        # 构建模型 (与训练完全一致)
        model_type = self.config.get("model_type", "transformer")
        model_kwargs = dict(
            input_dim=len(self.feature_cols),
            output_dim=1,
            horizon=1,
            input_len=self.lookback_steps,
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["transformer_dropout"],
        )
        if model_type == "itransformer":
            self.model = iTransformer(**model_kwargs, target_idx=self.target_feat_idx).to(self.device)
        else:
            self.model = TimeSeriesTransformer(**model_kwargs).to(self.device)

        model_path = os.path.join(result_dir, "best_seq2seq_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型权重: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 复用训练的 DataProcessor (清洗口径一致)
        self.processor = DataProcessor(self.config)
        self.processor.feature_scaler = self.feature_scaler
        self.processor.target_scaler = self.target_scaler
        self.processor.feature_cols = self.feature_cols

        print(f"[FlowPredictor] 模型已加载: {os.path.basename(model_path)}")
        print(f"[FlowPredictor] model_type={model_type}, lookback={self.lookback_days}d "
              f"({self.lookback_steps}步), predict={self.predict_days}d "
              f"({self.predict_steps}步), 特征数={len(self.feature_cols)}, "
              f"device={self.device}")

    # ── 核心推理 ──

    def predict(self, data, columns=None, encoding="utf-8-sig", as_list=False):
        """统一推理接口。

        Parameters
        ----------
        data : CSV 路径 / DataFrame / list[dict]
        columns : list[str]  (仅 list[list]/ndarray 输入需要)
        encoding : str
        as_list : bool  (True 返回 list[dict])

        Returns
        -------
        pd.DataFrame 或 list[dict]
        """
        if isinstance(data, pd.DataFrame):
            result = self._predict_df(data)
        elif isinstance(data, (str, os.PathLike)):
            result = self._predict_from_csv(data, encoding)
        elif isinstance(data, np.ndarray):
            result = self._predict_from_array(data, columns)
        elif isinstance(data, (list, tuple)):
            if len(data) > 0 and isinstance(data[0], dict):
                result = self._predict_df(pd.DataFrame(list(data)))
            else:
                result = self._predict_from_array(np.array(data), columns)
        else:
            raise TypeError(f"不支持的数据类型: {type(data).__name__}")

        return self._to_list(result) if as_list else result

    def predict_pressure(self, pred, schedule=None, as_list=False):
        """基于 predict() 结果生成分时压力预测 (不覆盖原 pred)。"""
        schedule = schedule or DEFAULT_PRESSURE_SCHEDULE
        if isinstance(pred, pd.DataFrame):
            result = pred.copy()
        else:
            result = pd.DataFrame(list(pred))
            result["timestamp"] = pd.to_datetime(result["timestamp"])
            result = result.set_index("timestamp")
        result = result.sort_index()

        pressures = []
        for ts in result.index:
            target = next(t for s, e, t in schedule if s <= ts.hour < e)
            pressures.append(round(target, 3))
        result["Pressure"] = pressures

        if as_list:
            return [{"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                     "Total_Flow": float(row["Total_Flow"]),
                     "Pressure": float(row["Pressure"])}
                    for ts, row in result.iterrows()]
        return result

    # ── 内部实现 ──

    def _predict_from_csv(self, csv_path, encoding):
        df = pd.read_csv(csv_path, encoding=encoding)
        print(f"[predict] 从 CSV 读取: {csv_path} ({len(df)} 行)")
        return self._predict_df(df)

    def _predict_from_array(self, data, columns):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if columns is None:
            raise ValueError("数组输入必须提供 columns 参数")
        return self._predict_df(pd.DataFrame(data, columns=list(columns)))

    def _predict_df(self, raw_df):
        """核心推理: 清洗 → 特征 → 缩放 → 自回归 rollout → 反归一化。"""
        df = raw_df.copy()

        # 解析时间列 → DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            for ts_col in ("时间", "timestamp"):
                if ts_col in df.columns:
                    df[ts_col] = pd.to_datetime(df[ts_col])
                    df = df.set_index(ts_col)
                    break
            else:
                raise ValueError("输入数据必须包含 时间 / timestamp 列或 DatetimeIndex")
        df = df.sort_index()
        print(f"[predict] 输入: {df.index.min()} ~ {df.index.max()}, {len(df)} 行")

        # ① 清洗 + 日历 + 数据驱动特征 (与训练 build_feature_table 完全一致)
        df_base = self.processor.build_base_features(df)
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
                f"(数据驱动特征需 ≥22 天历史; 推荐 ≥30 天)")
        print(f"[predict] 清洗+dropna 后: {len(df_feat)} 行, 特征数={len(self.feature_cols)}")

        # ② 取最后 lookback_steps 行, 缩放
        window_feat = df_feat.iloc[-self.lookback_steps:]
        X = self.feature_scaler.transform(window_feat.values.astype(np.float32))
        window = torch.from_numpy(X).unsqueeze(0).to(self.device)  # (1, L, C)
        last_hist_row = window_feat.iloc[-1]   # 未来行特征兜底 (理论不触发)

        # ③ 未来时间轴 + 在线特征重建的扩展表 (历史flow + 未来占位)
        # 数据驱动特征已无泄露 (基准=flow[t-1]), 未来行的 lag/rolling/zscore 依赖
        # "已生成的预测", 故每步把预测回灌成*原始流量*重算, 与训练 Xf 口径一致。
        last_ts = df_clean.index[-1]
        future_idx = pd.date_range(
            start=last_ts + pd.Timedelta(minutes=self.freq_minutes),
            periods=self.predict_steps, freq=self.resample_freq)
        target_col = self.target_cols[0]
        # 扩展表: 取最近 800 行历史 (覆盖 30 天滚动窗 720 + 余量) + 未来行 (flow=NaN)
        n_hist_ext = min(len(df_clean), 800)
        hist_ext = self.processor.add_calendar_features(
            df_clean[[target_col]].iloc[-n_hist_ext:].copy())
        fut_ext = self.processor.add_calendar_features(
            pd.DataFrame({target_col: np.nan}, index=future_idx))
        ext = pd.concat([hist_ext, fut_ext])   # 列: target_col + 日历; 未来 flow=NaN

        # ④ 自回归 rollout (与训练 autoregressive_rollout 一致: 目标通道用预测覆盖)
        preds_scaled = []
        with torch.no_grad():
            for k in range(self.predict_steps):
                # 重算数据驱动特征 (ext 里已填入 t<k 的预测 → lag/rolling 用到它们)
                feat_ext = self.processor.add_data_driven_features(ext)
                row = feat_ext.loc[future_idx[k], self.feature_cols].astype(np.float32)
                if row.isna().any():                       # 兜底: 边缘 min_periods 缺值
                    row = row.fillna(last_hist_row)
                row_scaled = self.feature_scaler.transform(
                    row.values.reshape(1, -1))[0].astype(np.float32)

                one = self.model(window, target_len=1)     # (1, 1, 1)
                pred_val = float(one[0, 0, 0].cpu())
                preds_scaled.append(pred_val)

                # 预测回灌 ext (转原始流量域, 因 data-driven 在原始域计算;
                #   feature_scaler 对 Total_Flow 的缩放与 target_scaler 一致)
                pred_orig = float(self.processor.target_scaler.inverse_transform(
                    np.array([[pred_val]], dtype=np.float64))[0, 0])
                ext.loc[future_idx[k], target_col] = pred_orig

                # 滑窗: 末尾拼 future 行, 目标通道用 scaled 预测覆盖 (与训练一致)
                next_row = row_scaled.copy()
                next_row[self.target_feat_idx] = pred_val
                next_row_t = torch.from_numpy(
                    next_row.astype(np.float32)).view(1, 1, -1).to(self.device)
                window = torch.cat([window[:, 1:, :], next_row_t], dim=1)

        # ⑤ 反归一化
        preds_arr = np.array(preds_scaled, dtype=np.float32).reshape(
            1, self.predict_steps, 1)
        y_inv = self.processor.inverse_transform_targets(preds_arr)[0]

        result = pd.DataFrame({"Total_Flow": y_inv[:, 0]}, index=future_idx)
        result.index.name = "timestamp"
        print(f"[predict] 输出: {result.index[0]} ~ {result.index[-1]}, "
              f"{len(result)} 行 ({self.resample_freq})")
        return result

    def _fill_day_gaps(self, df_clean):
        """日级缺口填充: 缺失 bin 用最近一天同时段数据补齐。"""
        if df_clean is None or df_clean.empty:
            return df_clean

        grid_start = df_clean.index.min().normalize()
        grid_end = (df_clean.index.max().normalize()
                    + pd.Timedelta(days=1) - pd.Timedelta(minutes=self.freq_minutes))
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
            print(f"[fill_day_gaps] 补齐 {len(filled)} 个缺失 bin "
                  f"(网格 {len(expected)} 槽位)")
        return df_clean

    @staticmethod
    def _to_list(result):
        return [{"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                 "Total_Flow": float(v)}
                for ts, v in result["Total_Flow"].items()]


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="自回归 Transformer 流量预测推理")
    parser.add_argument("--data", default=DEFAULT_DATA, help="输入 CSV 路径")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="训练结果目录")
    parser.add_argument("--out", default="prediction_ar.csv", help="输出 CSV 路径")
    parser.add_argument("--with_pressure", action="store_true", help="同时输出压力预测")
    args = parser.parse_args()

    predictor = FlowPredictor(args.result_dir)
    pred = predictor.predict(args.data)

    # 输出流量预测
    pred.to_csv(args.out, index=True, encoding="utf-8-sig")
    print(f"\n预测结果已保存: {args.out}")
    print(pred.head(10))
    print(f"... (共 {len(pred)} 行)")

    # 可选: 输出压力预测
    if args.with_pressure:
        pred_p = predictor.predict_pressure(pred)
        out_p = args.out.replace(".csv", "_pressure.csv")
        pred_p.to_csv(out_p, index=True, encoding="utf-8-sig")
        print(f"\n压力预测已保存: {out_p}")


if __name__ == "__main__":
    main()

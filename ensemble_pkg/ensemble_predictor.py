# -*- coding: utf-8 -*-
"""融合预测器 — Transformer(有监督) 与 TimesFM 2.5(零样本) 加权平均。

    pred = α · Transformer + (1-α) · TimesFM        (α 由 fit_weights.py 训练)

军山项目特点 (与武汉项目对比):
  - 两个模型输出分辨率相同: 都是 1h × 24 点, 无需升采样对齐
  - Transformer 输入: 184 步回看 (7天+16h) → 自回归 rollout → 24 点
  - TimesFM 输入: 同一条清洗管线的 168h 小时序列 → forecast → 24 点
  - 口径统一: 两个模型都看 DataProcessor 清洗后的 Total_Flow

对外接口与 junshan_inference.py 一致:
  predict(csv_path) -> dict {date, provider, unit, interval_minutes, horizon, values}
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# GBK 控制台兼容
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from _paths import ensure_import_paths, PROJECT_ROOT          # noqa: E402
TRANSFORMER_PKG_DIR = ensure_import_paths(verbose=False)

from transformer_model import TimeSeriesTransformer           # type: ignore  # noqa: E402
from itransformer_model import iTransformer                   # type: ignore  # noqa: E402
from data_processing import DataProcessor                     # type: ignore  # noqa: E402

# ── 默认路径 ──
DEFAULT_RESULT_DIR = os.path.join(
    PROJECT_ROOT, "transformer_pkg", "results",
    "junshan_L1D_P24H_1h_transformer_nextday16h_mc_20260917_181339")
DEFAULT_TIMESFM_MODEL = os.path.join(PROJECT_ROOT, "timesfm_model_transformers")
DEFAULT_WEIGHTS_PATH = os.path.join(_HERE, "weights.json")
DEFAULT_RAW_DATA = os.path.join(PROJECT_ROOT, "data", "水厂2025年小时级汇总.csv")

DAY_STEPS = 24
UNIT = "m3/h"
CONTEXT_HOURS = 168 * 3                 # TimesFM context = 7 天


# ── 工具函数 ──

def model_fingerprint(result_dir):
    """模型权重文件的 MD5, 用于一致性校验。"""
    path = os.path.join(result_dir, "best_seq2seq_model.pth")
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fuse(alpha, pred_transformer, pred_timesfm):
    """加权平均: α·Transformer + (1-α)·TimesFM。"""
    return alpha * np.asarray(pred_transformer, dtype=np.float64) + \
        (1.0 - alpha) * np.asarray(pred_timesfm, dtype=np.float64)


# ── 融合预测器 ──

class JunshanEnsemblePredictor:
    """Transformer × TimesFM 融合预测器。

    用法:
        p = JunshanEnsemblePredictor()
        result = p.predict("data/input_nextday16h_20251225_35d.csv")
        # result = {"date": "2025-12-26", "values": [...24个...], ...}
    """

    def __init__(self, result_dir=DEFAULT_RESULT_DIR,
                 timesfm_model_path=DEFAULT_TIMESFM_MODEL,
                 weights_path=DEFAULT_WEIGHTS_PATH, alpha=None,
                 lora_path=None,
                 device=None, verbose=True):
        self.result_dir = result_dir
        self.timesfm_model_path = timesfm_model_path
        self.weights_path = weights_path
        self.lora_path = lora_path
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── 加载 Transformer ──
        scaler_path = os.path.join(result_dir, "scaler.pkl")
        model_path = os.path.join(result_dir, "best_seq2seq_model.pth")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"未找到 scaler.pkl: {scaler_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型权重: {model_path}")

        with open(scaler_path, "rb") as f:
            saved = pickle.load(f)

        self.config = saved["config"]
        self.feature_scaler = saved["feature_scaler"]
        self.target_scaler = saved["target_scaler"]
        self.feature_cols = saved["feature_cols"]
        self.target_cols = saved["target_cols"]
        self.target_col = self.target_cols[0]
        self.target_feat_idx = saved.get("target_feat_idx",
                                         self.feature_cols.index(self.target_col))
        if not saved.get("autoregressive", False):
            raise ValueError("scaler.pkl 里 autoregressive != True, "
                             "请用 train_transformer_nextday_16h.py 的训练结果")

        self.lookback_steps = saved.get("lookback_steps")
        if self.lookback_steps is None:
            lb = self.config["lookback_days"]
            lbe = self.config.get("lookback_extra_hours", 0)
            freq = int(self.config["resample_freq"].replace("min", ""))
            self.lookback_steps = int(lb * (1440 // freq)) + int(lbe)
        self.resample_freq = self.config["resample_freq"]
        self.freq_minutes = int(self.resample_freq.replace("min", ""))
        self.predict_steps_max = saved.get("predict_steps_max", 48)

        # 模型工厂 (与训练完全一致: 单步头 horizon=1)
        model_type = self.config.get("model_type", "transformer")
        model_kwargs = dict(
            input_dim=len(self.feature_cols), output_dim=1, horizon=1,
            input_len=self.lookback_steps, d_model=self.config["d_model"],
            nhead=self.config["nhead"], num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["transformer_dropout"])
        if model_type == "itransformer":
            self.model = iTransformer(**model_kwargs, target_idx=self.target_feat_idx)
        else:
            self.model = TimeSeriesTransformer(**model_kwargs)
        state_dict = torch.load(model_path, map_location=self.device)
        # 训练若启用了 torch.compile, 权重会带上 TorchDynamo 的 "_orig_mod." 前缀
        # (OptimizedModule 把原始模块挂在 ._orig_mod 上), 而这里加载的是未编译的
        # 模型, 需要把前缀剥掉。兼容已存盘的旧权重。
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        # DataProcessor (与训练同口径)
        self.processor = DataProcessor(self.config)
        self.processor.feature_scaler = self.feature_scaler
        self.processor.target_scaler = self.target_scaler
        self.processor.feature_cols = self.feature_cols

        # ── TimesFM (懒加载, 首次 predict 时才初始化) ──
        self._tfm = None

        # ── 融合权重 ──
        if alpha is not None:
            self.alpha = float(alpha)
            self._weights_meta = {"alpha": self.alpha, "source": "explicit"}
        else:
            self._weights_meta = self._load_weights(weights_path)
            self.alpha = float(self._weights_meta["alpha"])

        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"α 必须在 [0,1], 实际 {self.alpha}")

        # 一致性校验: MD5 比对
        fitted_dir = self._weights_meta.get("transformer_result_dir")
        if fitted_dir:
            cur_fp = model_fingerprint(result_dir)
            old_fp = model_fingerprint(fitted_dir)
            if old_fp is not None and cur_fp is not None and old_fp != cur_fp:
                print(f"[Ensemble] [警告] α 是在另一个 Transformer 权重上拟合的")
                print(f"[Ensemble]   拟合时: {fitted_dir}")
                print(f"[Ensemble]   现在用: {result_dir}")

        if verbose:
            print(f"[Ensemble] Transformer + TimesFM 融合预测器")
            print(f"[Ensemble]   α={self.alpha:.4f} "
                  f"(Transformer {self.alpha:.1%} + TimesFM {1-self.alpha:.1%})")

    @property
    def tfm(self):
        """TimesFM 懒加载 (首次访问才初始化, 避免不必要的显存占用)。"""
        if self._tfm is None:
            from transformers import TimesFm2_5ModelForPrediction
            print(f"[Ensemble] 加载 TimesFM 模型 (transformers): {self.timesfm_model_path}")
            self._tfm = TimesFm2_5ModelForPrediction.from_pretrained(
                self.timesfm_model_path).to(torch.float32).eval()

            # 加载 LoRA 权重 (如果提供)
            if self.lora_path and os.path.exists(self.lora_path):
                from finetune_timesfm_lora import apply_lora
                # 读取 LoRA 配置 (rank, alpha)
                lora_dir = os.path.dirname(self.lora_path)
                config_path = os.path.join(lora_dir, "lora_config.json")
                lora_rank, lora_alpha = 8, 16.0
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                    lora_rank = cfg.get("lora_rank", lora_rank)
                    lora_alpha = cfg.get("lora_alpha", lora_alpha)
                # 先替换注意力层为 LoRALinear, 再加载权重
                self._tfm, _ = apply_lora(self._tfm, rank=lora_rank, alpha=lora_alpha)
                lora_state = torch.load(self.lora_path, map_location=self.device,
                                        weights_only=True)
                self._tfm.load_state_dict(lora_state, strict=False)
                print(f"[Ensemble] LoRA 权重已加载: {self.lora_path} "
                      f"(rank={lora_rank}, alpha={lora_alpha})")
                self._tfm.eval()

            print("[Ensemble] TimesFM 加载完成")
        return self._tfm

    @staticmethod
    def _load_weights(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"未找到融合权重文件: {path}\n"
                f"请先运行 fit_weights.py 训练权重, 或显式传 alpha=...")
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "alpha" not in meta:
            raise KeyError(f"权重文件缺少 'alpha' 字段: {path}")
        return meta

    # ── 数据预处理 (共用, Transformer 和 TimesFM 都从这里走) ──

    def preprocess(self, raw_df):
        """原始 DataFrame → 清洗 + 特征工程, 返回 (df_feat, df_clean)。

        df_feat: 完整特征表 (含 calendar + data-driven, 已 dropna)
        df_clean: 清洗后的 Total_Flow (DatetimeIndex, 小时级)
        """
        df_base = self.processor.build_base_features(raw_df.copy())
        df_clean = self.processor.clean_and_resample(df_base)
        df_feat = self.processor.add_calendar_features(df_clean)
        df_feat = self.processor.add_data_driven_features(df_feat)
        df_feat = df_feat[self.feature_cols].dropna()
        return df_feat, df_clean

    # ── Transformer 预测 ──

    def predict_transformer(self, raw_df, target_date=None):
        """Transformer 自回归推理, 返回24小时预测值数组 (原始流量域)。

        流程: 清洗 → 特征工程 → 取回看窗口 → 自回归 rollout → 反归一化。
        """
        df_feat, _ = self.preprocess(raw_df)

        if len(df_feat) < self.lookback_steps:
            raise ValueError(f"清洗+dropna 后仅 {len(df_feat)} 行, "
                             f"不足 lookback={self.lookback_steps} 步")

        if target_date is not None:
            td = pd.Timestamp(target_date).normalize()
            cand = df_feat.index[df_feat.index.normalize() == td - pd.Timedelta(days=1)]
            if len(cand) == 0:
                raise ValueError(f"截止日 {td - pd.Timedelta(days=1)} 不在特征表中")
            e = df_feat.index.get_loc(cand[-1])
        else:
            e = len(df_feat) - 1

        if e + 1 < self.lookback_steps:
            raise ValueError(f"截止位置 {e} 不足 lookback={self.lookback_steps}")

        window_feat = df_feat.iloc[e - self.lookback_steps + 1:e + 1]
        target_date_ts = window_feat.index[-1].normalize() + pd.Timedelta(days=1)
        last_ts = window_feat.index[-1]
        H = int(last_ts.hour) + 1
        total_steps = (DAY_STEPS - H) + DAY_STEPS

        # 时间连续性校验
        step_ns = int(pd.Timedelta(minutes=self.freq_minutes).total_seconds()) * 10**9
        deltas = np.diff(window_feat.index.asi8)
        if not np.all(deltas == step_ns):
            bad = np.where(deltas != step_ns)[0][0]
            raise ValueError(f"回看窗口有时间空洞: {window_feat.index[bad]}")

        # 窗口缩放 → 自回归 rollout
        X = self.feature_scaler.transform(window_feat.values.astype(np.float32))
        window = torch.from_numpy(X).unsqueeze(0).to(self.device)
        last_hist_row = window_feat.iloc[-1]

        _, df_clean = self.preprocess(raw_df)
        future_idx = pd.date_range(start=last_ts + pd.Timedelta(minutes=self.freq_minutes),
                                   periods=total_steps, freq=self.resample_freq)
        n_hist_ext = min(len(df_clean), 800)
        hist_ext = self.processor.add_calendar_features(
            df_clean[[self.target_col]].iloc[-n_hist_ext:].copy())
        fut_ext = self.processor.add_calendar_features(
            pd.DataFrame({self.target_col: np.nan}, index=future_idx))
        ext = pd.concat([hist_ext, fut_ext])

        preds_scaled = []
        with torch.no_grad():
            for k in range(total_steps):
                feat_ext = self.processor.add_data_driven_features(ext)
                row = feat_ext.loc[future_idx[k], self.feature_cols].astype(np.float32)
                if row.isna().any():
                    row = row.fillna(last_hist_row)
                row_scaled = self.feature_scaler.transform(
                    row.values.reshape(1, -1))[0].astype(np.float32)

                one = self.model(window, target_len=1)
                pred_val = float(one[0, 0, 0].cpu())
                preds_scaled.append(pred_val)

                pred_orig = float(self.target_scaler.inverse_transform(
                    np.array([[pred_val]], dtype=np.float64))[0, 0])
                ext.loc[future_idx[k], self.target_col] = pred_orig

                next_row = row_scaled.copy()
                next_row[self.target_feat_idx] = pred_val
                next_row_t = torch.from_numpy(
                    next_row.astype(np.float32)).view(1, 1, -1).to(self.device)
                window = torch.cat([window[:, 1:, :], next_row_t], dim=1)

        preds_arr = np.array(preds_scaled, dtype=np.float32).reshape(1, total_steps, 1)
        y_inv = self.processor.inverse_transform_targets(preds_arr)[0]
        return y_inv[-DAY_STEPS:, 0]

    # ── TimesFM 预测 ──

    def predict_timesfm(self, raw_df, target_date=None):
        """TimesFM 零样本预测, 返回24小时预测值数组 (原始流量域)。

        流程: 同一条清洗管线 → Total_Flow 小时级 → 取168h context → forecast。
        """
        _, df_clean = self.preprocess(raw_df)
        hourly = df_clean["Total_Flow"].resample("h").mean().dropna()

        if target_date is not None:
            td = pd.Timestamp(target_date).normalize()
            cutoff = td
            ctx = hourly[hourly.index < cutoff].iloc[-CONTEXT_HOURS:]
        else:
            ctx = hourly.iloc[-CONTEXT_HOURS:]

        if len(ctx) < 48:
            raise ValueError(f"TimesFM context 仅 {len(ctx)} 小时, 不足 48")

        ctx_tensor = torch.tensor(ctx.to_numpy(dtype=np.float64), dtype=torch.float32)
        with torch.no_grad():
            outputs = self.tfm(
                past_values=[ctx_tensor],
                forecast_context_len=16256)
        return outputs.mean_predictions[0, :DAY_STEPS].cpu().numpy().astype(np.float64)

    # ── 融合预测 (主接口) ──

    def predict(self, csv_path, encoding="utf-8-sig"):
        """融合预测: 读 CSV → 双模型推理 → 加权平均 → 返回接口 JSON。

        Returns
        -------
        dict : {date, provider, unit, interval_minutes, horizon, values}
        """
        raw_df = pd.read_csv(csv_path, encoding=encoding)
        for ts_col in ("时间", "timestamp"):
            if ts_col in raw_df.columns:
                raw_df[ts_col] = pd.to_datetime(raw_df[ts_col])
                raw_df = raw_df.set_index(ts_col)
                break

        # 目标天 = 数据最后一条的次日
        df_feat, _ = self.preprocess(raw_df)
        last_ts = df_feat.index[-1]
        target_date = last_ts.normalize() + pd.Timedelta(days=1)

        # ① Transformer
        pred_t = self.predict_transformer(raw_df)

        # ② TimesFM
        pred_f = self.predict_timesfm(raw_df)

        # ③ 融合
        y = fuse(self.alpha, pred_t, pred_f)
        values = [round(max(0.0, float(v)), 1) for v in y]

        if self.verbose:
            print(f"[Ensemble] 目标天 {target_date.date()}, "
                  f"α={self.alpha:.3f}, min={min(values):.1f}, max={max(values):.1f}")

        return {
            "date": target_date.strftime("%Y-%m-%d"),
            "provider": "junshan_ensemble",
            "unit": UNIT,
            "interval_minutes": self.freq_minutes,
            "horizon": DAY_STEPS,
            "values": values,
        }


# 延迟导入 (模块顶部会导致循环引用风险)
import pickle                       # noqa: E402


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="军山融合预测 (Transformer + TimesFM)")
    parser.add_argument("--data", default=os.path.join(
        PROJECT_ROOT, "data", "input_nextday16h_20250820_35d.csv"))
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--timesfm_model", default=DEFAULT_TIMESFM_MODEL)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lora_path", default=None, help="LoRA 微调权重路径")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    p = JunshanEnsemblePredictor(
        result_dir=args.result_dir, timesfm_model_path=args.timesfm_model,
        weights_path=args.weights, alpha=args.alpha, lora_path=args.lora_path)
    result = p.predict(args.data)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print("融合预测结果:")
    print(text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已保存: {args.out}")

# -*- coding: utf-8 -*-
"""融合预测器 — Transformer(有监督) 与 TimesFM 2.5(零样本) 加权平均。

    pred = α · Transformer + (1-α) · TimesFM        (α 由 fit_weights.py 训练)

军山项目特点 (与武汉项目对比):
  - 两个模型输出分辨率相同: 都是 1h × 24 点, 无需升采样对齐
  - Transformer 输入: 184 步回看 (7天+16h) → 自回归 rollout → 24 点
  - TimesFM 输入: 同一条清洗管线的 504h 小时序列 → forecast → 24 点
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
    "junshan_L1D_P24H_1h_transformer_nextday16h_mc_20260901_155317")
DEFAULT_TIMESFM_MODEL = os.path.join(PROJECT_ROOT, "timesfm_model_transformers")
DEFAULT_WEIGHTS_PATH = os.path.join(_HERE, "weights.json")
DEFAULT_RAW_DATA = os.path.join(PROJECT_ROOT, "data", "水厂2025年小时级汇总.csv")

DAY_STEPS = 24
UNIT = "m3/h"
CONTEXT_HOURS = 168 * 3                 # TimesFM context = 504h (21 天)


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


# ── 概率预测 (分位数) ──
#
# TimesFM 2.5 在同一次 forward 里就同时吐出了点预测和 9 个分位数
# (输出张量 full_predictions, 形状 [batch, horizon, 10]):
#
#     索引  0      1     2     3     4     5     6     7     8     9
#     含义  point  q0.1  q0.2  q0.3  q0.4  q0.5  q0.6  q0.7  q0.8  q0.9
#
# 索引 0 是独立的 point 头, 可能跑到区间外面, 所以只取 1..9。
# config.decode_index = 5, 即 mean_predictions ≡ full_predictions[:, :, 5],
# 也就是说 TimesFM 现在用的"点预测"本来就是它的中位数。
#
# Transformer 是 MSE 训的单点模型, 没有分位数头, 它的区间来自回测残差
# 校准 (fit_weights.py 产出 quantile_calibration.json)。

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_INDEX = 4                      # QUANTILE_LEVELS 里 0.5 的位置
INTERVAL_METHODS = ("vincentization", "mixture")
DEFAULT_CALIB_PATH = os.path.join(_HERE, "results", "quantile_calibration.json")


def enforce_monotone(q):
    """沿最后一维 (分位数轴) 强制单调不减。"""
    return np.maximum.accumulate(np.asarray(q, dtype=np.float64), axis=-1)


def clip_quantiles(q, lo, hi):
    """裁剪到 [lo, hi] 并恢复单调性 (裁剪可能破坏分位轴单调)。"""
    return enforce_monotone(np.clip(np.asarray(q, dtype=np.float64), lo, hi))


def anchor_median(q, median):
    """把中位数列钉到 median, 同时分别恢复左右两侧的单调性。

    mixture (线性池) 的中位数一般不等于两个中位数的加权平均, 所以需要这一步
    才能保证 "区间中位数 == 点预测"。左右两侧分别投影, 不会把 median 顶走。
    """
    q = np.array(q, dtype=np.float64, copy=True)
    m = MEDIAN_INDEX
    med = np.asarray(median, dtype=np.float64).reshape(-1, 1)
    if q.ndim != 2 or q.shape[1] <= m:
        raise ValueError(f"anchor_median 需要 (N, >{m}) 的数组, 实际 {q.shape}")

    q[:, m] = med[:, 0]
    q[:, :m] = np.maximum.accumulate(np.minimum(q[:, :m], med), axis=-1)
    q[:, m + 1:] = np.maximum.accumulate(np.maximum(q[:, m + 1:], med), axis=-1)
    return q


def qf_columns(levels=QUANTILE_LEVELS):
    """回测 CSV 里 TimesFM 原生分位数的列名 (校准表与 CSV 的列契约)。"""
    return [f"qf_{int(round(lv * 100)):02d}" for lv in levels]


def day_interval(day_data, alpha, calib, method="vincentization"):
    """从回测行 (一天) 重建当天的融合分位数 (24, 9)。

    回测 CSV 里只存了 TimesFM 的原生分位数; Transformer 那半从校准表套回,
    再按 method 融合。缺列或缺校准表时返回 None。
    小时数优先取 DatetimeIndex, 否则取 timestamp 列。
    """
    if calib is None:
        return None
    cols = qf_columns()
    if not all(c in day_data.columns for c in cols):
        return None

    if isinstance(day_data.index, pd.DatetimeIndex):
        hours = day_data.index.hour.to_numpy()
    elif "timestamp" in day_data.columns:
        hours = pd.to_datetime(day_data["timestamp"]).dt.hour.to_numpy()
    else:
        raise ValueError("day_interval 需要 DatetimeIndex 或 timestamp 列")

    q_f = day_data[cols].to_numpy(dtype=np.float64)
    pred_t = day_data["pred_transformer"].to_numpy(dtype=np.float64)
    g = np.asarray(calib["hourly"]["transformer"], dtype=np.float64)
    q_t = pred_t[:, None] + g[hours]
    return fuse_quantiles(alpha, q_t, q_f, method)


def fuse_quantiles(alpha, q_transformer, q_timesfm,
                   method="vincentization", levels=QUANTILE_LEVELS):
    """融合两个模型的分位数, 返回同形状数组。

    vincentization: q_E(τ) = α·q_T(τ) + (1-α)·q_F(τ)  逐分位数加权平均,
        是现有点融合 fuse() 的直接推广, 自动保持单调性。
    mixture: 线性池 F_E = α·F_T + (1-α)·F_F 之后对 level 数值求逆。
        区间通常更宽 (把"不知道哪个模型对"也算进不确定性)。

    两种方法都满足 q_E(0.5) = α·q_T(0.5) + (1-α)·q_F(0.5),
    因此只要两边中位数等于各自点预测, 融合中位数就等于融合点预测。
    """
    qt = enforce_monotone(q_transformer)
    qf = enforce_monotone(q_timesfm)
    if method == "vincentization":
        return alpha * qt + (1.0 - alpha) * qf
    if method != "mixture":
        raise ValueError(f"未知的区间融合方法: {method} (可选 {INTERVAL_METHODS})")

    levels = np.asarray(levels, dtype=np.float64)
    if qt.ndim == 1:
        return _mixture_invert(alpha, qt, qf, levels)
    return np.stack([_mixture_invert(alpha, a, b, levels) for a, b in zip(qt, qf)])


def _mixture_invert(alpha, qt_row, qf_row, levels, n_grid=2001):
    """线性池分位数: 先叠加两条 CDF, 再在 levels 上求逆。"""
    lo = float(min(qt_row[0], qf_row[0]))
    hi = float(max(qt_row[-1], qf_row[-1]))
    pad = 0.05 * max(hi - lo, 1e-6)
    x = np.linspace(lo - pad, hi + pad, n_grid)

    tau_pad = np.concatenate([[0.0], levels, [1.0]])

    def cdf(q):
        # 在 τ∈[0,1] 上线性插值, 两端按最外斜率外推
        xq = np.concatenate([[q[0] - (q[1] - q[0])], q,
                             [q[-1] + (q[-1] - q[-2])]])
        return np.interp(x, xq, tau_pad, left=0.0, right=1.0)

    F = alpha * cdf(qt_row) + (1.0 - alpha) * cdf(qf_row)
    F = np.maximum.accumulate(F)
    F = F + np.linspace(0.0, 1e-9, len(F))       # 防平台化导致求逆歧义
    return np.interp(levels, F, x)


UNSET = object()


# ── 融合预测器 ───

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
                 lora_path=None, calib_path=DEFAULT_CALIB_PATH,
                 interval_method="vincentization",
                 device=None, verbose=True):
        self.result_dir = result_dir
        self.timesfm_model_path = timesfm_model_path
        self.weights_path = weights_path
        self.lora_path = lora_path
        self.calib_path = calib_path
        if interval_method not in INTERVAL_METHODS:
            raise ValueError(f"interval_method 必须是 {INTERVAL_METHODS}, "
                             f"实际 {interval_method!r}")
        self.interval_method = interval_method
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
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        # DataProcessor (与训练同口径)
        self.processor = DataProcessor(self.config)
        self.processor.feature_scaler = self.feature_scaler
        self.processor.target_scaler = self.target_scaler
        self.processor.feature_cols = self.feature_cols

        # ── TimesFM (懒加载, 首次 predict 时才初始化) ──
        self._tfm = None

        # ── 区间校准表 (懒加载; 缺失时优雅降级) ──
        self._calib = UNSET

        # ─ 融合权重 ─
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

    @property
    def calibration(self):
        """区间校准表 (懒加载)。找不到返回 None, 由调用方优雅降级。"""
        if self._calib is UNSET:
            self._calib = self._load_calibration(self.calib_path)
            if self._calib is None:
                print(f"[Ensemble] [警告] 未找到区间校准表: {self.calib_path}")
                print(f"[Ensemble]   概率预测将退化为 'TimesFM 区间形状 + "
                      f"融合点预测'。请先运行 fit_weights.py 生成校准表。")
            elif self.verbose:
                print(f"[Ensemble] 区间校准表已加载: {self.calib_path} "
                      f"({self._calib.get('n_days', '?')} 天)")
        return self._calib

    @property
    def interval_levels(self):
        calib = self.calibration
        if calib and calib.get("levels"):
            return tuple(float(x) for x in calib["levels"])
        return QUANTILE_LEVELS

    @staticmethod
    def _load_calibration(path):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                calib = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[Ensemble] [警告] 校准表读取失败 ({e}), 忽略")
            return None
        if "hourly" not in calib or "transformer" not in calib.get("hourly", {}):
            print(f"[Ensemble] [警告] 校准表缺少 hourly.transformer, 忽略")
            return None
        return calib

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

    def predict_transformer_quantiles(self, raw_df, target_date=None):
        """Transformer 点预测 + 残差校准分位数, 返回 (point(24,), q(24, 9)|None)。

        Transformer 没有分位数头, 区间来自 fit_weights.py 用样本外回测残差
        标定出的经验分位数 ĝ(τ, hour)。ĝ 已按中位数中心化, 所以
        q[:, 0.5] ≡ point, 点预测不会被概率预测改动。

        校准表缺失时 q 返回 None, 由 _build_interval 优雅降级。
        """
        point = self.predict_transformer(raw_df, target_date)
        calib = self.calibration
        if calib is None:
            return point, None

        n_levels = len(self.interval_levels)
        g = calib.get("hourly", {}).get("transformer")
        if g is None:
            g = calib.get("pooled", {}).get("transformer")
            if g is None:
                return point, None
            g = np.tile(np.asarray(g, dtype=np.float64), (DAY_STEPS, 1))
        g = np.asarray(g, dtype=np.float64)
        if g.shape != (DAY_STEPS, n_levels):
            raise ValueError(f"校准表 transformer 分位数形状 {g.shape} 与预期 "
                             f"({DAY_STEPS}, {n_levels}) 不符")
        return point, point[:, None] + g

    # ── TimesFM 预测 ─

    def predict_timesfm(self, raw_df, target_date=None):
        """TimesFM 零样本预测, 返回24小时预测值数组 (原始流量域)。

        流程: 同一条清洗管线 → Total_Flow 小时级 → 取 context → forecast。
        """
        return self.predict_timesfm_quantiles(raw_df, target_date)[0]

    @staticmethod
    def timesfm_day_offset(last_ctx_hour):
        """上下文末尾在 H 点 → 目标天 00:00 落在 TimesFM 的第几步。

        TimesFM 一次 forward 输出的第 0 步 = 上下文结束后的第一个小时,
        所以默认(上下文停在 23:00)时目标天 00:00 正好是第 0 步, 偏移为 0。
        但生产口径是"16 点决策", 上下文停在 15:00, 此时第 0 步对应
        前一天 16:00 —— 目标天 00:00 要往后数 8 步。
        历史 bug: 无论上下文停在哪都取 [0:24], 导致输出整体错位 8 小时。
        """
        return (23 - int(last_ctx_hour)) % 24

    def predict_timesfm_quantiles(self, raw_df, target_date=None):
        """TimesFM 点预测 + 分位数, 返回 (point(24,), quantiles(24, 9))。

        分位数直接来自同一次 forward 的 full_predictions, 零额外推理成本。
        full_predictions 的索引 0 是独立 point 头 (可能跑到区间外), 取 [:, 1:]。
        取哪 24 步由 timesfm_day_offset 决定, 见那里的说明。
        """
        _, df_clean = self.preprocess(raw_df)
        hourly = df_clean["Total_Flow"].resample("h").mean().dropna()

        if target_date is not None:
            td = pd.Timestamp(target_date).normalize()
            ctx = hourly[hourly.index < td].iloc[-CONTEXT_HOURS:]
        else:
            ctx = hourly.iloc[-CONTEXT_HOURS:]

        if len(ctx) < 48:
            raise ValueError(f"TimesFM context 仅 {len(ctx)} 小时, 不足 48")

        ctx_tensor = torch.tensor(ctx.to_numpy(dtype=np.float64), dtype=torch.float32)
        with torch.no_grad():
            outputs = self.tfm(
                past_values=[ctx_tensor],
                forecast_context_len=16256)

        # 对齐: 目标天 00:00 之前的"缺口"小时数 = 23 - 上下文末尾小时
        offset = self.timesfm_day_offset(ctx.index[-1].hour)
        stop = offset + DAY_STEPS
        horizon = outputs.mean_predictions.shape[1]
        if stop > horizon:
            raise ValueError(
                f"TimesFM horizon={horizon} 不够取 [{offset}:{stop}]; "
                f"上下文末尾 {ctx.index[-1]} (hour={ctx.index[-1].hour})")

        point = outputs.mean_predictions[0, offset:stop].cpu().numpy() \
            .astype(np.float64)
        q = outputs.full_predictions[0, offset:stop, 1:].cpu().numpy() \
            .astype(np.float64)

        if q.shape != (DAY_STEPS, len(QUANTILE_LEVELS)):
            raise ValueError(
                f"TimesFM 分位数形状 {q.shape} 与预期 "
                f"({DAY_STEPS}, {len(QUANTILE_LEVELS)}) 不符; "
                f"模型 config.quantiles = {getattr(self.tfm.config, 'quantiles', None)}")

        # 分位轴理论上单调; 万一不单调才修 (anchor_median 保证修完
        # 中位数仍精确等于点预测, 且左右两侧各自单调)
        if not np.all(np.diff(q, axis=-1) >= 0):
            q = anchor_median(enforce_monotone(q), point)
        return point, q

    # ── 融合预测 (主接口) ──

    def predict(self, csv_path, encoding="utf-8-sig", with_interval=True):
        """融合预测: 读 CSV → 双模型推理 → 加权平均 → 返回接口 JSON。

        Parameters
        ----------
        with_interval : bool
            True 时额外返回 quantiles / interval / uncertainty 三个字段。
            False 时返回值与加入概率预测之前逐字段完全一致。

        Returns
        -------
        dict : {date, provider, unit, interval_minutes, horizon, values
                [, quantiles, interval, uncertainty]}
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

        # 决策口径一致性: 校准表是在某个截止小时上标定的, 推理时截止小时不同
        # 会让逐小时残差分位数对不上 (Transformer 的日内缺口步数变了)
        calib = self.calibration if with_interval else None
        if calib is not None:
            fit_hour = calib.get("decision_hour")
            if fit_hour is not None and int(fit_hour) != int(last_ts.hour):
                print(f"[Ensemble] [警告] 校准表标定于 {int(fit_hour)}:00 截止, "
                      f"本次推理是 {int(last_ts.hour)}:00 截止 "
                      f"(日内缺口 {abs(23 - int(fit_hour))}h vs "
                      f"{abs(23 - int(last_ts.hour))}h); "
                      f"区间可能偏窄/偏宽, 建议用同口径重跑 fit_weights.py")

        # ① Transformer  ② TimesFM
        if with_interval:
            pred_t, q_t = self.predict_transformer_quantiles(raw_df)
            pred_f, q_f = self.predict_timesfm_quantiles(raw_df)
        else:
            pred_t = self.predict_transformer(raw_df)
            pred_f = self.predict_timesfm(raw_df)
            q_t = q_f = None

        # ③ 融合 (点预测: 与之前完全一致)
        y = fuse(self.alpha, pred_t, pred_f)
        values = [round(max(0.0, float(v)), 1) for v in y]

        if self.verbose:
            print(f"[Ensemble] 目标天 {target_date.date()}, "
                  f"α={self.alpha:.3f}, min={min(values):.1f}, max={max(values):.1f}")

        result = {
            "date": target_date.strftime("%Y-%m-%d"),
            "provider": "junshan_ensemble",
            "unit": UNIT,
            "interval_minutes": self.freq_minutes,
            "horizon": DAY_STEPS,
            "values": values,
        }

        if with_interval:
            result.update(self._build_interval(y, pred_f, q_t, q_f))
            if self.verbose:
                lo = result["interval"]["lower"]
                hi = result["interval"]["upper"]
                w = float(np.mean(np.asarray(hi) - np.asarray(lo)))
                print(f"[Ensemble] {result['interval']['coverage']:.0%} 区间 "
                      f"平均宽度 {w:.1f} m³/h, 方法={self.interval_method}")
        return result

    def _build_interval(self, y, pred_f, q_t, q_f):
        """由两模型的分位数构造融合区间。

        不变量: 融合区间的中位数 ≡ 点预测 values (median_gap 应恒为 0)。
        """
        y = np.asarray(y, dtype=np.float64)
        pred_f = np.asarray(pred_f, dtype=np.float64)
        if q_f is None:
            raise ValueError("TimesFM 分位数缺失, 无法构造区间")
        q_f = enforce_monotone(q_f)

        if q_t is None:
            # 降级: 只有 TimesFM 的不确定性形状, 平移到融合点预测上
            q_e = y[:, None] + (q_f - pred_f[:, None])
            source_t = "unavailable(TimesFM 形状平移)"
        else:
            q_e = fuse_quantiles(self.alpha, q_t, q_f, self.interval_method)
            source_t = "residual_calibration"

        median_gap = float(np.max(np.abs(q_e[:, MEDIAN_INDEX] - y)))
        if self.interval_method == "mixture":
            rel = median_gap / max(float(np.abs(y).mean()), 1e-9)
            if rel > 0.01:
                print(f"[Ensemble] [提示] mixture (线性池) 的原始中位数偏离点预测 "
                      f"{median_gap:.1f} m³/h ({rel:.1%}); 已锚定到点预测, "
                      f"但该方法的区间与点预测并不自洽, 建议改用 vincentization")

        # 流量物理下限 0 (与 values 的 max(0,·) 同口径), 再把中位数钉到点预测。
        # maximum 是保序运算; anchor_median 保证 mixture 下也能同时满足
        # 单调性与 "区间中位数 == 点预测"。
        q_e = np.maximum(q_e, 0.0)
        q_e = anchor_median(q_e, np.maximum(y, 0.0))

        levels = self.interval_levels
        q_out = [[round(float(v), 1) for v in row] for row in q_e.T]   # (9, 24)
        return {
            "quantiles": {"levels": list(levels), "values": q_out},
            "interval": {
                "coverage": round(float(levels[-1] - levels[0]), 4),
                "lower": q_out[0],
                "upper": q_out[-1],
            },
            "uncertainty": {
                "method": self.interval_method,
                "median_gap": median_gap,
                "components": {"transformer": source_t, "timesfm": "native"},
            },
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
    parser.add_argument("--calib", default=DEFAULT_CALIB_PATH,
                        help="区间校准表 (quantile_calibration.json)")
    parser.add_argument("--interval_method", default="vincentization",
                        choices=list(INTERVAL_METHODS),
                        help="区间融合方法 (默认 vincentization)")
    parser.add_argument("--no-interval", action="store_true",
                        help="关闭概率预测, 只输出点预测")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    p = JunshanEnsemblePredictor(
        result_dir=args.result_dir, timesfm_model_path=args.timesfm_model,
        weights_path=args.weights, alpha=args.alpha, lora_path=args.lora_path,
        calib_path=args.calib, interval_method=args.interval_method)
    result = p.predict(args.data, with_interval=not args.no_interval)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print("融合预测结果:")
    print(text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已保存: {args.out}")

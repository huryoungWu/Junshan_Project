import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

# ============================================================================
# data_processing.py — 军山水厂数据清洗 / 特征工程 / 训练测试集划分 (共用管线)
#
# 数据源: D:\Junshan_Project\data\水厂2025年小时级汇总.csv
#   (由 data/merge_data.py 将 4 个 Excel 合并为整点小时级: 出厂水流量/压力为整点值,
#    泵状态为分钟级轮询 ffill 推断到整点 0/1, 泵频率为实测 Hz; 4/14 缺口 5 小时
#    保持 NaN 不虚构)
#
# 清洗/特征流程 (训练 train_transformer.py / train_transformer_raw.py 与推理
# inference_transformer.py 共用, 保证训练与推理特征口径完全一致):
#   0. 出厂水流量列改名 Total_Flow (目标通道), 剔除流量全 NaN 行 (缺口段)
#   1. 突变流量插值修正: 日内 (t±1h) / 跨日 (t±1 天) 双判据, 泵切换与空洞保护
#      (在 Hampel 之前执行, 命中整点以插值值保留; 泵台数变化阶跃受保护不修正)
#   2. Hampel 离群清洗 (窗口 2 天整点, 泵切换 ±1 整点保护) + 物理界限裁剪
#   3. 重采样 (60min, 已是整点 → 恒等) + 删除空 bin (空洞不虚构)
#   4. 时间特征 / 滞后滚动特征 (可选: 原始量测版不调用)
#   5. 按时间划分训练/测试集 (split_by_time) 与归一化 (scaler 由训练端拟合并保存)
#   6. 序列窗口生成 (make_sequences) 与 SeqDataset
#
# 军山数据特点说明:
#   - 停机后频率残值 (0.25Hz 起) 使"状态=0 且 频率>0"占全年 ~26%, 是传感器特征,
#     不做状态/频率一致性剔除 (剔除会砍掉 1/4 训练数据)
#   - 泵状态列 "1#送水泵运行" 与频率列 "1#泵运行频率" 都含 "泵运行" 子串,
#     状态列筛选必须排除含 "频率" 的列
#   - 小时级数据不再插值: 已是整点聚合结果, 15 分钟级插值上限不适用, 空洞保持 NaN
#
# 注意: 本文件任何修改都会同时改变训练与推理的特征口径 —— 改动后必须重新训练,
# 否则旧模型权重与新特征不匹配 (scaler.pkl 里的 feature_cols 校验只能兜底一部分)。
# ============================================================================

# ==================== 异常值清洗参数 (Hampel 滤波) ====================
HAMPEL_WINDOW     = 48      # 滚动窗口 (整点, 2 天; 小时级数据, 可在 config 覆盖)
HAMPEL_K          = 10.0    # 阈值: |x - 局部中位数| > k * scale 判为异常
MAD_FLOOR_RATIO   = 0.02    # scale 下限 = 局部中位数的 2% (防 MAD≈0 误报)
PUMP_GUARD_STEPS  = 1       # 泵切换前后 ±1 个整点不做 Hampel 判定 (真实阶跃保护)


def needed_csv_columns(header):
    """按 build_base_features 的实际口径筛出 CSV 必需列 (训练/推理/滚动评估共用)。

    军山已合并小时级 CSV 共 8 列: 时间 + 出厂水流量 + 出厂水压力 + 泵频率 2 列
    + 泵状态 3 列, 全部显式收录。
    """
    ts_col = next((c for c in ("时间", "timestamp") if c in header), None)
    used = [c for c in header
            if c in {"出厂水流量", "出厂水压力", "1#泵运行频率", "6#泵运行频率",
                     "1#送水泵运行", "2#送水泵运行", "6#送水泵运行"}]
    if ts_col is not None:
        used.insert(0, ts_col)
    return used


def detect_outliers(s, window=HAMPEL_WINDOW, k=HAMPEL_K, floor_ratio=MAD_FLOOR_RATIO, guard=None):
    """Hampel 滤波: 基于滚动中位数 + MAD 的稳健离群检测。

    窗口 center=False: 判定 t 时刻只用 t 及之前的数据, 不引入未来数据
    (避免训练/测试边界泄漏)。scale = max(1.4826*MAD, floor_ratio*中位数),
    底部分数保证信号极稳定 (MAD≈0) 时仍不会把正常波动误判为异常。

    guard: 布尔 Series (True = 受保护不判异常), 用于泵切换等真实阶跃时刻。
    """
    med = s.rolling(window, center=False, min_periods=window // 2).median()
    res = (s - med).abs()
    mad = res.rolling(window, center=False, min_periods=window // 2).median()
    scale = np.maximum(1.4826 * mad, floor_ratio * med)
    flag = (res > k * scale).fillna(False)
    if guard is not None:
        flag = flag & ~guard
    return flag


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.target_cols = ["Total_Flow"]
        self.feature_cols = None

    def load_raw(self):
        file_path = self.config["file_path"]
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            encoding = self.config.get("encoding", "utf-8-sig")
            header = pd.read_csv(file_path, encoding=encoding, nrows=0).columns
            df = pd.read_csv(file_path, encoding=encoding,
                             usecols=needed_csv_columns(header))
        else:
            df = pd.read_excel(file_path)

        if "timestamp" not in df.columns:
            for ts_col in ("时间", "timestamp"):
                if ts_col in df.columns:
                    df = df.rename(columns={ts_col: "timestamp"})
                    break
            else:
                raise ValueError("数据中必须包含 时间 / timestamp 列")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df = df.set_index("timestamp")
        return df

    @staticmethod
    def _pump_status_cols(df):
        """泵运行状态列: 含"泵运行"且不含"频率"。

        "1#泵运行频率" 同时含 "泵运行" 子串, 若按 "泵运行" 裸匹配会把频率列
        当成状态列 (0~50Hz 会被加进运行泵数量)。
        """
        return [c for c in df.columns if "泵运行" in c and "频率" not in c]

    @staticmethod
    def _pump_freq_cols(df):
        return [c for c in df.columns if "运行频率" in c]

    def build_base_features(self, df):
        """列挑选/类型转换: 出厂水流量 → Total_Flow, 压力 → Target_Pressure,
        泵状态求和 → 运行泵数量, 泵状态/频率列原样透传。
        """
        # ── 流量列: 军山出厂水流量单列, 重命名为 Total_Flow (目标通道) ──
        flow_cols = [c for c in df.columns if "流量" in c and "频率" not in c]
        if len(flow_cols) != 1:
            raise ValueError(f"军山数据流量列异常: {flow_cols} (应为单个出厂水流量列)")
        df = df.rename(columns={flow_cols[0]: "Total_Flow"})
        flow_cols = ["Total_Flow"]

        # 剔除流量全 NaN 的行 (2025-04-14 17:00~21:00 缺口 5 小时, 保持 NaN 不虚构)
        all_nan = df[flow_cols].isna().all(axis=1)
        if all_nan.sum() > 0:
            df = df[~all_nan].copy()
            print(f"  剔除流量全 NaN 行: {all_nan.sum()} 条 (剩 {len(df)})")

        df = df.copy()
        df["Total_Flow"] = df["Total_Flow"].astype(np.float32)

        pressure_candidates = [c for c in df.columns if "压力" in c]
        if len(pressure_candidates) == 0:
            raise ValueError("未找到压力列 (出厂水压力)")
        df["Target_Pressure"] = df[pressure_candidates[0]].astype(np.float32)

        # 泵状态列 (0/1) → 运行泵数量 (台数, 1~3); 全部缺失 → NaN (缺失 ≠ 停泵)
        pump_run_cols = self._pump_status_cols(df)
        if len(pump_run_cols) > 0:
            pump_sum = df[pump_run_cols].sum(axis=1)
            pump_valid = df[pump_run_cols].notna().sum(axis=1)
            df["运行泵数量"] = pump_sum.where(pump_valid > 0, np.nan)
        else:
            df["运行泵数量"] = 0.0

        # 泵状态列与频率列原样透传, 由 clean_and_resample 重采样 (状态 last / 频率 mean)
        out = df[flow_cols + ["Target_Pressure", "运行泵数量"]
                 + pump_run_cols + self._pump_freq_cols(df)].copy()
        return out

    def clean_and_resample(self, df):
        """Hampel 清洗 + 物理界限裁剪 → 重采样 (60min, 已是整点 → 恒等) → 删空 bin。

        泵状态/频率列透传: 状态取 bin 末值 (last), 频率取均值 (mean)。
        小时级数据不做插值: 已是整点聚合结果, 缺口保持 NaN 由 dropna 删 bin,
        不被线性插值"平滑斜坡"伪造数据。
        """
        out = df.copy()
        flow_cols = ["Total_Flow"]          # 已在 build_base_features 改名
        status_cols = self._pump_status_cols(out)
        freq_cols = self._pump_freq_cols(out)

        # 泵切换保护掩码: 运行泵数量变化的 ±PUMP_GUARD_STEPS 内不判异常
        # (泵切换导致流量真实阶跃, 不应被 Hampel/突变修正抹平)
        pump_guard = pd.Series(False, index=out.index)
        if out["运行泵数量"].notna().any():
            switch = out["运行泵数量"].diff().fillna(0).ne(0)
            guard_steps = self.config.get("pump_guard_steps", PUMP_GUARD_STEPS)
            pump_guard = switch.rolling(2 * guard_steps + 1, center=True,
                                        min_periods=1).max().astype(bool)

        # ── 突变流量插值修正: 在 Hampel 之前执行, 保证规则命中的突变整点
        # 以插值修正值保留参与训练 (若放在 Hampel 之后, 极端突变已被置 NaN 删除) ──
        self.correct_flow_spikes(out)

        # 展开 hampel_cols: "Total_Flow" → 单列原样; 其余 (如 Target_Pressure) 原样
        hampel_targets = []
        for c in self.config.get("hampel_cols", []):
            if c == "Total_Flow":
                hampel_targets += flow_cols
            elif c in out.columns:
                hampel_targets.append(c)
        hampel_targets = list(dict.fromkeys(hampel_targets))  # 去重保序

        # ── Hampel 清洗: 检出后置 NaN, 与物理界限越界值一起在重采样时剔除 ──
        for col in hampel_targets:
            if col not in out.columns:
                continue
            flag = detect_outliers(out[col], guard=pump_guard,
                                   window=self.config.get("hampel_window", HAMPEL_WINDOW))
            n_out = int(flag.sum())
            if n_out > 0:
                out.loc[flag, col] = np.nan
                print(f"  {col} Hampel 离群点 {n_out} 条 ({n_out / len(out):.3%}) 置 NaN")

        # ── 物理界限裁剪 (压力 0.1~0.5 MPa 实测 0.21~0.38, 界限仅作兜底) ──
        BOUNDS = {
            "Target_Pressure": (0.1, 0.5),     # MPa, 越界 → NaN → 该 bin 剔除
        }
        for col, (lo, hi) in BOUNDS.items():
            if col not in out.columns:
                continue
            s = out[col].copy()
            s[(s < lo) | (s > hi)] = np.nan
            out[col] = s
        for col in flow_cols:
            s = out[col].copy()
            s[(s < 0.0) | (s > 10000.0)] = np.nan
            out[col] = s

        freq = self.config["resample_freq"]
        res = pd.DataFrame(index=out.resample(freq).mean().index)

        res["Total_Flow"] = out["Total_Flow"].resample(freq).mean()
        res["Target_Pressure"] = out["Target_Pressure"].resample(freq).mean()
        res["运行泵数量"] = out["运行泵数量"].resample(freq).last()   # 状态量取 bin 末值
        for c in status_cols:
            res[c] = out[c].resample(freq).last()       # 单泵状态 0/1
        for c in freq_cols:
            res[c] = out[c].resample(freq).mean()       # 泵运行频率 Hz

        # 删除无数据的空 bin: 时间轴留下空洞是刻意的 (滚动特征可能跨洞计算),
        # 但优于虚构数据。
        res = res.dropna()
        return res

    def correct_flow_spikes(self, res):
        """整点流量的突变检测 + 插值修正 (仅 Total_Flow, 在原位修改传入的 df)。

        在 Hampel / 物理裁剪之前执行: 规则命中的突变整点直接插值修正, 保留该
        小时参与训练; 修正后残余的极端离群仍由 Hampel 置 NaN 删除, 两者互补。

        两种突变判据 (任一命中即修正, k = config['spike_ratio'], 默认 2.0):
          日内: t 时刻与 t±1h 比较 —— 8 点流量比 7 点、9 点都突然高 (或低) → 异常
          跨日: t 时刻与昨天/明天同一整点比较 —— 3/31 8 点比 3/30、4/1 8 点都
                突然低 (或高) → 异常
        "突然"的定义: 流量 > k × max(两参考点) (高突变) 或 < min(两参考点) / k
        (低突变), 即比两个参考点都显著偏离才判异常 —— 正常日周期/泵切换的
        平滑阶跃不满足条件。

        修正: 两个参考点线性插值 (等间隔 → 均值); 同时命中两种判据时优先日内
        参考 (更局部)。保护规则:
          - 泵切换发生的整点本身不判异常 (spike_guard_steps=0, 过渡读数兜底;
            不 ±1h 保护: 本厂每日 7:00/23:00 切换泵, ±1h 会罩住全部早高峰时段)
          - 参考点与 t 的间隔必须恰为 1h / 24h: 跨数据空洞 (如 4/14 17~21 时缺失)
            的伪邻居不参与判据
          - 参考点为 NaN 时该判据不参与 (与 Hampel 置 NaN 的值互为退化)
          - 单遍处理, 参考点一律取原始值, 不做级联修正
        """
        if "Total_Flow" not in res.columns:
            return res
        freq_minutes = int(self.config["resample_freq"].replace("min", ""))
        k = self.config.get("spike_ratio", 2.0)
        within_steps = max(1, 60 // freq_minutes)      # 1h 折算步数
        day_steps = (24 * 60) // freq_minutes          # 1 天折算步数
        within_delta = pd.Timedelta(minutes=freq_minutes * within_steps)
        day_delta = pd.Timedelta(minutes=freq_minutes * day_steps)

        f = res["Total_Flow"]
        orig = f.copy()    # 原始值快照: res.loc 赋值会原地修改 f, 打印需用修正前的值

        # 泵切换保护掩码: 仅保护切换发生的整点本身 (spike_guard_steps=0 默认)。
        # 不能像 Hampel 那样 ±1h 保护: 本厂几乎每天 7:00/23:00 切换泵, ±1h 会把
        # 6:00~8:00 全部罩住, 用户场景"8 点流量比 7 点 9 点突然高"将永不触发;
        # 且"比两参考点都显著偏离"的判据本身已免疫真实阶跃 (阶跃后至少一个
        # 参考点在新水平上, 不满足"都偏离"), 保护只兜底切换瞬间的过渡读数。
        guard = pd.Series(False, index=res.index)
        if "运行泵数量" in res.columns and res["运行泵数量"].notna().any():
            switch = res["运行泵数量"].diff().fillna(0).ne(0)
            g = 2 * self.config.get("spike_guard_steps", 0) + 1
            guard = switch.rolling(g, center=True, min_periods=1).max().astype(bool)

        # 邻居时间间隔校验: 只认"恰为 1h / 24h"的真邻居 (跨空洞的伪邻居不参与)
        # 用索引相减而非 diff: diff 只给相邻行的间隔, 跨 24 步必须 t - t[-24]
        idx = res.index.to_series()
        n1_ok = (idx - idx.shift(within_steps)) == within_delta        # t 的前 1h 参考
        n2_ok = (idx.shift(-within_steps) - idx) == within_delta       # t 的后 1h 参考
        p1_ok = (idx - idx.shift(day_steps)) == day_delta              # t 的前 1 天参考
        p2_ok = (idx.shift(-day_steps) - idx) == day_delta             # t 的后 1 天参考

        n1, n2 = f.shift(within_steps), f.shift(-within_steps)   # 日内参考: t±1h
        p1, p2 = f.shift(day_steps), f.shift(-day_steps)         # 跨日参考: t±1 天

        def flags(cur, r1, r2, ok1, ok2):
            ref_hi = pd.concat([r1, r2], axis=1).max(axis=1)
            ref_lo = pd.concat([r1, r2], axis=1).min(axis=1)
            hi = (cur > k * ref_hi) & (ref_hi > 0)
            lo = (cur < ref_lo / k) & (ref_lo > 0)
            # ok 含时间间隔校验; 参考点 NaN (被 Hampel 等置空) 时 ref 为 NaN → 判据不命中
            return (hi | lo) & ok1 & ok2 & ~guard & r1.notna() & r2.notna()

        within_anom = flags(f, n1, n2, n1_ok, n2_ok)       # 判据1: 日内 t±1h
        day_anom = flags(f, p1, p2, p1_ok, p2_ok)          # 判据2: 跨日 t±1 天
        anomaly = within_anom | day_anom
        if not anomaly.any():
            return res

        # 修正值: 参考点线性插值 (等间隔 → 均值); 同时命中时优先日内参考 (更局部)
        fix = pd.Series(np.nan, index=res.index)
        fix[within_anom] = (n1 + n2)[within_anom] / 2.0
        fix[day_anom & ~within_anom] = (p1 + p2)[day_anom & ~within_anom] / 2.0
        fix = fix[anomaly]
        n_anom = int(anomaly.sum())
        res.loc[anomaly, "Total_Flow"] = fix

        print(f"  突变流量插值修正: {n_anom} 条 ({n_anom / len(res):.3%}) "
              f"[日内 {int(within_anom.sum())} / 跨日 {int((day_anom & ~within_anom).sum())}]")
        for t in anomaly[anomaly].index[:5]:
            print(f"    {t}  {orig[t]:9.1f} → {fix[t]:9.1f}")
        return res

    def add_time_features(self, df):
        out = df.copy()
        out["hour"] = out.index.hour
        out["dayofweek"] = out.index.dayofweek
        out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
        out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
        out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
        return out

    def add_lag_rolling_features(self, df):
        out = df.copy()

        base_vars = ["Total_Flow", "Target_Pressure"]

        # 物理时间窗口 (分钟) → 按当前 resample_freq 折算步数, 换频率不改变特征
        # 物理含义; 窗口下限为 2 步: rolling(1).std() 恒为 NaN (单样本无标准差)
        # 会把 dropna 后整表删空 (60min 下 90min 窗口折为 2 步)。
        freq_minutes = int(self.config["resample_freq"].replace("min", ""))
        ppd = (24 * 60) // freq_minutes                     # 每24小时点数 (24 @60min)
        lags = sorted({max(1, m // freq_minutes) for m in [15, 90, 180, 360, 720, 1440]})
        rolls = sorted({max(2, m // freq_minutes) for m in [90, 180, 360, 720, 1440]})
        trend_mins = {"trend_90m": 90, "trend_6h": 360}

        for col in base_vars:
            if col not in out.columns:
                continue
            for lag in lags:
                out[f"{col}_lag_{lag}"] = out[col].shift(lag)
            for d in range(1, 7):
                out[f"{col}_diff_{d}"] = out[col].diff(d)
            out[f"{col}_acc"] = out[col].diff(1).diff(1)
            for w in rolls:
                out[f"{col}_roll_mean_{w}"] = out[col].rolling(w).mean()
                out[f"{col}_roll_std_{w}"] = out[col].rolling(w).std()
                out[f"{col}_roll_min_{w}"] = out[col].rolling(w).min()
                out[f"{col}_roll_max_{w}"] = out[col].rolling(w).max()
            for name, win_min in trend_mins.items():
                win = max(2, win_min // freq_minutes)
                out[f"{col}_{name}"] = out[col] - out[col].rolling(win).mean()

        if "Total_Flow" in out.columns:
            out["flow_lag_1day"] = out["Total_Flow"].shift(ppd)
            out["flow_lag_2day"] = out["Total_Flow"].shift(2 * ppd)
        if "Target_Pressure" in out.columns:
            out["pressure_lag_1day"] = out["Target_Pressure"].shift(ppd)
            out["pressure_lag_2day"] = out["Target_Pressure"].shift(2 * ppd)
        if "Total_Flow" in out.columns and "Target_Pressure" in out.columns:
            out["flow_pressure_ratio"] = out["Total_Flow"] / (out["Target_Pressure"] + 1e-3)
            out["flow_pressure_diff"] = out["Total_Flow"] - out["Target_Pressure"]
        if "运行泵数量" in out.columns:
            for m in [15, 90]:
                out[f"运行泵数量_lag_{m}min"] = out["运行泵数量"].shift(max(1, m // freq_minutes))
            out["pump_change"] = out["运行泵数量"].diff(1)
            if "Total_Flow" in out.columns:
                out["flow_per_pump"] = out["Total_Flow"] / (out["运行泵数量"] + 1e-3)
                out["flow_mul_pump"] = out["Total_Flow"] * out["运行泵数量"]
        if "Total_Flow" in out.columns:
            out["flow_volatility_90m"] = out["Total_Flow"].rolling(max(2, 90 // freq_minutes)).std()
            out["flow_volatility_6h"] = out["Total_Flow"].rolling(max(2, 360 // freq_minutes)).std()
        if "hour" in out.columns:
            out["is_morning_peak"] = out["hour"].isin([6, 7, 8, 9]).astype(int)
            out["is_evening_peak"] = out["hour"].isin([17, 18, 19, 20]).astype(int)
            out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4]).astype(int)

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna()
        return out

    def build_feature_table(self):
        df_raw = self.load_raw()
        df_base = self.build_base_features(df_raw)

        # 先按时间切分, 再分别清洗: 清洗不跨越训练/测试边界。
        # 滞后/滚动特征仍在拼接后的全量表上计算 (因果特征, 无泄漏)。
        test_days = self.config.get("test_days", 15)
        test_start = df_base.index[-1] - pd.Timedelta(days=test_days)
        df_base_train = df_base.loc[df_base.index <= test_start].copy()
        df_base_test = df_base.loc[df_base.index > test_start].copy()
        print(f"  清洗分段: 训练段 {df_base_train.index.min()} ~ {df_base_train.index.max()} "
              f"({len(df_base_train)} 行), 测试段 {df_base_test.index.min()} ~ {df_base_test.index.max()} "
              f"({len(df_base_test)} 行), 分段清洗互不跨界")

        df_clean = pd.concat([self.clean_and_resample(df_base_train),
                              self.clean_and_resample(df_base_test)])

        df_time = self.add_time_features(df_clean)
        df_feat = self.add_lag_rolling_features(df_time)
        self.feature_cols = df_feat.columns.tolist()
        return df_feat

    def split_by_time(self, df):
        """按时间划分训练/测试集: 测试集 = 最后 test_days 天。

        按天数切分保证测试段长度确定、所有实验都能生成测试序列。
        """
        test_days = self.config.get("test_days", 7)
        test_start = df.index[-1] - pd.Timedelta(days=test_days)
        df_test = df.loc[df.index > test_start].copy()
        df_train = df.loc[df.index <= test_start].copy()
        return df_train, df_test

    def fit_scalers(self, df_train):
        self.feature_scaler.fit(df_train[self.feature_cols].values)
        target_raw = df_train[self.target_cols].values.astype(np.float64)
        if self.config.get("target_transform") == "log1p":
            target_raw = np.log1p(target_raw)
        self.target_scaler.fit(target_raw)

    def transform_df(self, df):
        X = self.feature_scaler.transform(df[self.feature_cols].values)
        target_raw = df[self.target_cols].values.astype(np.float64)
        if self.config.get("target_transform") == "log1p":
            target_raw = np.log1p(target_raw)
        Y = self.target_scaler.transform(target_raw)
        return X, Y

    def inverse_transform_targets(self, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return arr
        if arr.ndim == 2:
            inv = self.target_scaler.inverse_transform(arr)
        elif arr.ndim == 3:
            shape = arr.shape
            flat = arr.reshape(-1, shape[-1])
            inv = self.target_scaler.inverse_transform(flat).reshape(shape)
        else:
            raise ValueError("只支持2维或3维数组反归一化")
        if self.config.get("target_transform") == "log1p":
            inv = np.expm1(inv)   # 反变换回原始流量单位 (评估指标口径与 baseline 一致)
        return inv

    def make_sequences(self, x_array, y_array, lookback_days, predict_days):
        """支持自定义 lookback/predict 天数的序列生成"""
        freq_minutes = int(self.config["resample_freq"].replace("min", ""))
        points_per_day = (24 * 60) // freq_minutes

        lookback_steps = int(lookback_days * points_per_day)
        horizon_steps = int(predict_days * points_per_day)
        stride = self.config["stride"]

        total_len = len(x_array)
        n_samples = (total_len - lookback_steps - horizon_steps) // stride + 1
        if n_samples <= 0:
            return np.empty((0, lookback_steps, x_array.shape[1]), dtype=np.float32), \
                   np.empty((0, horizon_steps, y_array.shape[1]), dtype=np.float32)

        # 预分配 + 直填 (from_numpy 共享内存, 不做双份拷贝)
        X = np.empty((n_samples, lookback_steps, x_array.shape[1]), dtype=np.float32)
        Y = np.empty((n_samples, horizon_steps, y_array.shape[1]), dtype=np.float32)

        idx = 0
        for i in range(0, total_len - lookback_steps - horizon_steps + 1, stride):
            X[idx] = x_array[i:i + lookback_steps]
            Y[idx] = y_array[i + lookback_steps:i + lookback_steps + horizon_steps]
            idx += 1

        return X, Y


class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.Y = torch.from_numpy(np.ascontiguousarray(Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

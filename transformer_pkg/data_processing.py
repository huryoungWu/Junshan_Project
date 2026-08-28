import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

try:
    from chinese_calendar import is_holiday, is_workday as _cal_is_workday
    _HAS_CHINESE_CALENDAR = True
except ImportError:
    _HAS_CHINESE_CALENDAR = False
    print("⚠ chinese-calendar 未安装, 节假日特征退化为简单周末判断 (pip install chinese-calendar)")

# ============================================================================
# data_processing.py — 军山水厂数据清洗 / 训练测试集划分 (共用管线, 仅流量)
#
# 数据源: D:\Junshan_Project\data\水厂2025年小时级汇总.csv
#   (由 data/merge_data.py 将 4 个 Excel 合并为整点小时级; 4/14 缺口 5 小时
#    保持 NaN 不虚构)
#
# 本管线只处理出厂水流量 (Total_Flow), 不读 / 不算 / 不喂任何其他量测:
#   - 压力 (出厂水压力): 不作为输入特征 (压力预测由推理端分时时段表给出)
#   - 泵状态 / 泵频率 / 运行泵数量: 每小时可能变化, 不作为输入特征, 也不用于
#     清洗 guard (泵切换导致的真实阶跃由突变判据"比两参考点都显著偏离"自然免疫)
#
# 清洗流程 (训练 train_transformer.py / train_transformer_autoregressive.py 与推理
# inference_transformer.py / inference_transformer_autoregressive.py 共用, 保证
# 训练与推理口径完全一致):
#   0. 出厂水流量列改名 Total_Flow (目标通道), 剔除流量全 NaN 行 (缺口段)
#   1. 突变流量插值修正: 日内 (t±1h) / 跨日 (t±1 天) 双判据, 空洞邻居保护
#      (在 Hampel 之前执行, 命中整点以插值值保留)
#   2. Hampel 离群清洗 (窗口 2 天整点) + 流量物理界限裁剪 (0~10000)
#   3. 重采样 (60min, 已是整点 → 恒等) + 删除空 bin (空洞不虚构)
#   4. 模型输入 = Total_Flow + 日历特征 (hour_sin/cos, dow_sin/cos, is_workday,
#      month_sin/cos, doy_sin/cos, is_holiday, holiday_eve, holiday_next,
#      由时间索引确定性生成, 见 CALENDAR_COLS / add_calendar_features;
#      压力 / 泵量测仍不读不算)
#   5. 按时间划分训练/测试集 (split_by_time) 与归一化 (scaler 由训练端拟合并保存)
#   6. 序列窗口生成 (make_sequences) 与 SeqDataset
#
# 注意: 本文件任何修改都会同时改变训练与推理的特征口径 —— 改动后必须重新训练,
# 否则旧模型权重与新特征不匹配 (scaler.pkl 里的 feature_cols 校验只能兜底一部分)。
# ============================================================================

# ==================== 异常值清洗参数 (Hampel 滤波) ====================
HAMPEL_WINDOW     = 48      # 滚动窗口 (整点, 2 天; 小时级数据, 可在 config 覆盖)
HAMPEL_K          = 10.0    # 遂值: |x - 局部中位数| > k * scale 判为异常
MAD_FLOOR_RATIO   = 0.02    # scale 下限 = 局部中位数的 2% (防 MAD≈0 误报)

# ==================== 日历特征 (训练/推理共用, 由时间索引确定性生成) ====================
# 多尺度周期编码 + 中国法定节假日感知:
#   hour_sin/cos   : 24h 日周期 (平滑 23→0 跨越)
#   dow_sin/cos    : 7 天周周期 (替代原 dayofweek 数值, 保留 6→0 平滑过渡)
#   month_sin/cos  : 12 月季节周期 (捕捉夏高冬低的用水季节性)
#   doy_sin/cos    : 年日序号周期 (更细粒度的季节过渡, 如梅雨/伏天)
#   is_workday     : 调休感知的工作日 (chinese_calendar 区分补班/调休)
#   is_holiday     : 法定假日 (含调休安排的连假, 如春节/国庆)
#   holiday_eve    : 假期前一天 (节前工厂赶工 / 居民囤水, 用水模式提前变化)
#   holiday_next   : 假期后一天 (节后恢复过渡期)
# 由时刻唯一确定: 训练/推理/未来 rollout 同源生成, 值精确可知, 不需模型预测。
# 依赖 chinese-calendar 包 (pip install chinese-calendar), 未安装时退化为简单周末。
CALENDAR_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    "is_workday", "is_holiday", "holiday_eve", "holiday_next",
]

# ==================== 数据驱动特征 (从流量时序统计提取) ====================
# 基于 Part1/Part2 图表揭示的规律设计:
#   滞后:   昨天/上周同时段流量 (日内形状日间重复)
#   滚动:   多尺度均值/标准差 (近似 Part1 多项式拟合趋势)
#   日内:   当天峰谷幅度/早晚高峰比例 (Part2 曲线形状描述符)
#   偏离:   Z-score (当前值相对近期水平的偏离程度)
#
# 所有特征用 shift() 避免未来信息泄露, 训练/推理口径一致。
# 新增 / 修改后必须重新训练 (旧 scaler.pkl 的 feature_cols 校验不匹配)。
DATA_DRIVEN_COLS = [
    # 滞后特征
    "lag_24h", "lag_168h", "lag_24h_diff", "lag_24h_ratio",
    # 滚动统计
    "roll_mean_3d", "roll_mean_7d", "roll_mean_30d",
    "roll_std_7d", "roll_max_7d", "roll_min_7d", "roll_median_48h",
    # 日内形状 (shift 1 天, 用前一天完整日内统计, 避免当天前瞻)
    "day_avg", "day_peak_amp", "morning_ratio", "evening_ratio", "night_ratio",
    # 偏离特征
    "flow_zscore_7d", "flow_zscore_30d",
]

# 旧版日历特征 (兼容旧模型: hour_sin/cos, dayofweek, is_weekend)
LEGACY_CALENDAR_COLS = ["hour_sin", "hour_cos", "dayofweek", "is_weekend"]


def needed_csv_columns(header):
    """本管线只读时间列与出厂水流量列 (训练/推理/滚动评估共用)。

    压力 / 泵状态 / 泵频率列不再读取 —— 它们不作为模型输入, 也不参与清洗。
    """
    ts_col = next((c for c in ("时间", "timestamp") if c in header), None)
    used = [c for c in header if c in {"出厂水流量"}]
    if ts_col is not None:
        used.insert(0, ts_col)
    return used


def detect_outliers(s, window=HAMPEL_WINDOW, k=HAMPEL_K, floor_ratio=MAD_FLOOR_RATIO, guard=None):
    """Hampel 滤波: 基于滚动中位数 + MAD 的稳健离群检测。

    窗口 center=False: 判定 t 时刻只用 t 及之前的数据, 不引入未来数据
    (避免训练/测试边界泄漏)。scale = max(1.4826*MAD, floor_ratio*中位数),
    底部分数保证信号极稳定 (MAD≈0) 时仍不会把正常波动误判为异常。

    guard: 布尔 Series (True = 受保护不判异常), 当前管线不使用 (保留接口)。
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

    def add_calendar_features(self, df, use_legacy=False):
        """给带 DatetimeIndex 的 df 追加日历特征列 (返回新 df, 不修改入参)。

        Args:
            df: 带 DatetimeIndex 的 DataFrame
            use_legacy: True 时生成旧版 4 特征 (hour_sin/cos, dayofweek, is_weekend),
                        用于兼容旧模型; False (默认) 生成新版 12 特征。

        新版列 (见 CALENDAR_COLS):
          hour_sin / hour_cos   : 小时相位 sin/cos (24h 周期)
          dow_sin / dow_cos     : 星期几 sin/cos (7 天周期, 6→0 平滑)
          month_sin / month_cos : 月份 sin/cos (12 月季节周期)
          doy_sin / doy_cos     : 年日序号 sin/cos (365 天细粒度季节)
          is_workday            : 调休感知工作日 (chinese_calendar; 未装则=is_weekend)
          is_holiday            : 法定假日 (含调休连假; 未装则=0)
          holiday_eve           : 假期前一天 (未装则=0)
          holiday_next          : 假期后一天 (未装则=0)

        旧版列 (legacy):
          hour_sin / hour_cos : 同上
          dayofweek           : 星期几 0~6 (数值)
          is_weekend          : 是否周末 0/1

        日历特征由时刻唯一确定: 训练 / 推理 / 未来 rollout 都用同一份代码生成,
        口径一致; 未来时刻的值精确可知, 自回归 rollout 中作为外生通道用真值,
        不需要模型预测。改动特征集合后必须重新训练 (见模块头注释)。
        """
        out = df.copy()

        # ── 小时 sin/cos (24h 周期, 新旧共用) ──
        h = out.index.hour.to_numpy()
        out["hour_sin"] = np.sin(2.0 * np.pi * h / 24.0).astype(np.float32)
        out["hour_cos"] = np.cos(2.0 * np.pi * h / 24.0).astype(np.float32)

        if use_legacy:
            # 旧版: dayofweek 数值 + is_weekend
            dow = out.index.dayofweek.to_numpy()
            out["dayofweek"] = dow.astype(np.float32)
            out["is_weekend"] = (dow >= 5).astype(np.float32)
            return out

        # ── 以下为新版特征 ──

        # ── 星期几 sin/cos (7 天周期, 替代原 dayofweek 数值) ──
        dow = out.index.dayofweek.to_numpy()
        out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
        out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)

        # ── 月份 sin/cos (12 月季节周期) ──
        month = out.index.month.to_numpy()
        out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0).astype(np.float32)
        out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0).astype(np.float32)

        # ── 年日序号 sin/cos (365 天细粒度季节过渡) ──
        doy = out.index.dayofyear.to_numpy()
        out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
        out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)

        # ── 工作日 / 节假日特征 (依赖 chinese_calendar) ──
        dates = out.index.normalize()   # 当天 00:00:00 (去重后用于批量查询)
        unique_dates = dates.unique()

        if _HAS_CHINESE_CALENDAR:
            # 批量查询: 每个日期只查一次, 再 map 回逐行
            workday_map = {d: float(_cal_is_workday(d.to_pydatetime().date()))
                          for d in unique_dates}
            # is_holiday 包含普通周末; 只标记法定假日 (非工作日且非周末)
            holiday_map = {}
            for d in unique_dates:
                dt = d.to_pydatetime().date()
                # 法定假日 = 非工作日 且 不是普通周末 (周六/周日)
                is_special = (not _cal_is_workday(dt)) and d.dayofweek < 5
                holiday_map[d] = float(is_special)

            # holiday_eve / holiday_next: 前后一天是否是法定假日
            eve_map = {}
            next_map = {}
            for d in unique_dates:
                prev_d = d - pd.Timedelta(days=1)
                next_d = d + pd.Timedelta(days=1)
                # 超出 chinese-calendar 覆盖范围的日期, 用周末近似
                try:
                    prev_dt = prev_d.to_pydatetime().date()
                    eve_map[d] = float(not _cal_is_workday(prev_dt) and prev_d.dayofweek < 5)
                except ValueError:
                    eve_map[d] = 0.0
                try:
                    next_dt = next_d.to_pydatetime().date()
                    next_map[d] = float(not _cal_is_workday(next_dt) and next_d.dayofweek < 5)
                except ValueError:
                    next_map[d] = 0.0

            out["is_workday"] = dates.map(workday_map).astype(np.float32)
            out["is_holiday"] = dates.map(holiday_map).astype(np.float32)
            out["holiday_eve"] = dates.map(eve_map).astype(np.float32)
            out["holiday_next"] = dates.map(next_map).astype(np.float32)
        else:
            # 无 chinese-calendar 时退化: 用 dayofweek 近似
            is_weekend = (dow >= 5).astype(np.float32)
            out["is_workday"] = 1.0 - is_weekend
            out["is_holiday"] = np.float32(0.0)
            out["holiday_eve"] = np.float32(0.0)
            out["holiday_next"] = np.float32(0.0)

        return out

    def add_data_driven_features(self, df):
        """基于流量时序统计的特征, 捕捉图表中可见的季节/日内模式。

        在 add_calendar_features() 之后调用, 依赖 Total_Flow 列。
        所有特征用 shift() 避免未来信息泄露:
          - 滚动/偏离特征: shift(1) 确保只用当前时刻之前的数据
          - 日内形状特征: shift(24) 用前一天完整日内统计, 避免当天前瞻

        Args:
            df: 带 DatetimeIndex 的 DataFrame, 含 Total_Flow + 日历特征

        Returns:
            追加数据驱动特征后的 DataFrame (不修改入参)
        """
        out = df.copy()
        flow = out["Total_Flow"]

        # ── 1. 滞后特征: 昨天/上周同时段 (Part2: 日内形状日间高度重复) ──
        # lag_*: 纯 shift, 只用过去点, 无泄露。
        # lag_24h_diff/ratio: 原版含 flow[t] (预测目标) → 泄露。改为以最近已知点
        #   flow[t-1] 为基准对 24h 前做差/比, 结构不变 (X 对 24h 前的 X), 全过去。
        out["lag_24h"] = flow.shift(24).astype(np.float32)
        out["lag_168h"] = flow.shift(168).astype(np.float32)
        out["lag_24h_diff"] = (flow.shift(1) - flow.shift(25)).astype(np.float32)
        out["lag_24h_ratio"] = (flow.shift(1) / flow.shift(25).replace(0, np.nan)).astype(np.float32)

        # ── 2. 滚动统计: 多尺度趋势 (Part1: 多项式拟合的离散近似) ──
        # shift(1): 严格用当前时刻之前的数据, 避免信息泄露
        flow_shifted = flow.shift(1)

        out["roll_mean_3d"] = flow_shifted.rolling(72, min_periods=36).mean().astype(np.float32)
        out["roll_mean_7d"] = flow_shifted.rolling(168, min_periods=84).mean().astype(np.float32)
        out["roll_mean_30d"] = flow_shifted.rolling(720, min_periods=360).mean().astype(np.float32)
        out["roll_std_7d"] = flow_shifted.rolling(168, min_periods=84).std().astype(np.float32)
        out["roll_max_7d"] = flow_shifted.rolling(168, min_periods=84).max().astype(np.float32)
        out["roll_min_7d"] = flow_shifted.rolling(168, min_periods=84).min().astype(np.float32)
        out["roll_median_48h"] = flow_shifted.rolling(48, min_periods=24).median().astype(np.float32)

        # ── 3. 日内形状特征: 用前一天完整日内统计 (shift 24h) ──
        # Part2 显示日内曲线形状稳定但幅度随季节变化
        day_groups = flow.groupby(flow.index.normalize())
        daily_stats = day_groups.agg(
            day_avg="mean",
            day_max="max",
            day_min="min",
        )
        daily_stats["day_peak_amp"] = daily_stats["day_max"] - daily_stats["day_min"]

        # 早晚高峰均值
        h = flow.index.hour
        morning_mask = (h >= 7) & (h <= 10)
        evening_mask = (h >= 18) & (h <= 21)
        night_mask = (h >= 0) & (h <= 5)
        day_dates = flow.index.normalize()
        daily_stats["morning_avg"] = flow.where(morning_mask).groupby(day_dates).mean()
        daily_stats["evening_avg"] = flow.where(evening_mask).groupby(day_dates).mean()
        daily_stats["night_avg"] = flow.where(night_mask).groupby(day_dates).mean()

        # 高峰比例 = 高峰均值 / 全天均值
        daily_stats["morning_ratio"] = (daily_stats["morning_avg"] / daily_stats["day_avg"].replace(0, np.nan))
        daily_stats["evening_ratio"] = (daily_stats["evening_avg"] / daily_stats["day_avg"].replace(0, np.nan))
        daily_stats["night_ratio"] = (daily_stats["night_avg"] / daily_stats["day_avg"].replace(0, np.nan))

        # 只保留需要的列, shift 1 天避免当天前瞻
        day_feats = daily_stats[["day_avg", "day_peak_amp", "morning_ratio",
                                 "evening_ratio", "night_ratio"]].shift(1)

        # 映射回小时级
        for col in day_feats.columns:
            out[col] = day_dates.map(day_feats[col]).astype(np.float32)

        # ── 4. 偏离特征: 最近已知值相对近期水平 (Part1: 拟合残差的统计描述) ──
        # 原版用 flow[t] (预测目标) → 泄露。改为 flow.shift(1)=flow[t-1] (最近已知
        #   小时) 对滚动均值/标准差做 zscore; roll_mean_7d 等本身只用 ≤t-1, 全过去。
        r7_mean = out["roll_mean_7d"]
        r7_std = out["roll_std_7d"].replace(0, np.nan)
        out["flow_zscore_7d"] = ((flow.shift(1) - r7_mean) / r7_std).astype(np.float32)

        r30_std = flow_shifted.rolling(720, min_periods=360).std().replace(0, np.nan)
        out["flow_zscore_30d"] = ((flow.shift(1) - out["roll_mean_30d"]) / r30_std).astype(np.float32)

        return out

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

    def build_base_features(self, df):
        """列挑选/类型转换: 出厂水流量 → Total_Flow (唯一保留的列)。

        压力 / 泵状态 / 泵频率一律不读不算 —— 本管线仅处理流量。
        """
        flow_cols = [c for c in df.columns if "流量" in c and "频率" not in c]
        if len(flow_cols) != 1:
            raise ValueError(f"军山数据流量列异常: {flow_cols} (应为单个出厂水流量列)")
        df = df.rename(columns={flow_cols[0]: "Total_Flow"})

        # 剔除流量全 NaN 的行 (2025-04-14 17:00~21:00 缺口 5 小时, 保持 NaN 不虚构)
        all_nan = df["Total_Flow"].isna()
        if all_nan.sum() > 0:
            df = df[~all_nan].copy()
            print(f"  剔除流量全 NaN 行: {all_nan.sum()} 条 (剩 {len(df)})")

        out = pd.DataFrame(index=df.index)
        out["Total_Flow"] = df["Total_Flow"].astype(np.float32)
        return out

    def clean_and_resample(self, df):
        """突变修正 + Hampel 清洗 + 流量物理界限裁剪 → 重采样 (60min, 已是整点
        → 恒等) → 删空 bin。

        小时级数据不做插值: 已是整点聚合结果, 缺口保持 NaN 由 dropna 删 bin,
        不被线性插值"平滑斜坡"伪造数据。
        """
        out = df.copy()

        # ── 突变流量插值修正: 在 Hampel 之前执行, 保证规则命中的突变整点
        # 以插值修正值保留参与训练 (若放在 Hampel 之后, 极端突变已被置 NaN 删除) ──
        self.correct_flow_spikes(out)

        # ── Hampel 清洗: 检出后置 NaN, 与物理界限越界值一起在重采样时剔除 ──
        # hampel_cols 配置里不在 out.columns 的列 (如历史遗留的 Target_Pressure) 自动跳过
        hampel_targets = [c for c in self.config.get("hampel_cols", ["Total_Flow"])
                          if c in out.columns]
        hampel_targets = list(dict.fromkeys(hampel_targets))  # 去重保序
        for col in hampel_targets:
            flag = detect_outliers(out[col],
                                   window=self.config.get("hampel_window", HAMPEL_WINDOW))
            n_out = int(flag.sum())
            if n_out > 0:
                out.loc[flag, col] = np.nan
                print(f"  {col} Hampel 离群点 {n_out} 条 ({n_out / len(out):.3%}) 置 NaN")

        # ── 流量物理界限裁剪 (越界 → NaN → 该 bin 剔除) ──
        s = out["Total_Flow"].copy()
        s[(s < 0.0) | (s > 10000.0)] = np.nan
        out["Total_Flow"] = s

        freq = self.config["resample_freq"]
        res = pd.DataFrame(index=out.resample(freq).mean().index)
        res["Total_Flow"] = out["Total_Flow"].resample(freq).mean()

        # 删除无数据的空 bin: 时间轴留下空洞是刻意的, 优于虚构数据。
        res = res.dropna()
        return res

    def correct_flow_spikes(self, res):
        """整点流量的突变检测 + 插值修正 (仅 Total_Flow, 在原位修改传入的 df)。

        在 Hampel / 物理裁剪之前执行: 规则命中的突变整点直接插值修正, 保留该
        小时参与训练; 修正后残余的极端离群仍由 Hampel 置 NaN 删除, 两者互补。

        两种突变判据 (任一命中即修正), 各自独立阈值:
          日内 (k_within, 默认 1.2): t 时刻与 t±1h 比较, 曲线平滑可用低阈值
          跨日 (k_cross, 默认 2.0):  t 时刻与 t±24h 比较, 日间差异大需高阈值
        "突然"的定义: 流量 > k × max(两参考点) (高突变) 或 < min(两参考点) / k
        (低突变), 即比两个参考点都显著偏离才判异常 —— 正常日周期/泵切换的
        平滑阶跃不满足条件 (阶跃后至少一个参考点在新水平上, 不满足"都偏离")。

        修正: 两个参考点线性插值 (等间隔 → 均值); 同时命中两种判据时优先日内
        参考 (更局部)。保护规则:
          - 参考点与 t 的间隔必须恰为 1h / 24h: 跨数据空洞 (如 4/14 17~21 时缺失)
            的伪邻居不参与判据
          - 参考点为 NaN 时该判据不参与 (与 Hampel 置 NaN 的值互为退化)
          - 单遍处理, 参考点一律取原始值, 不做级联修正
        """
        if "Total_Flow" not in res.columns:
            return res
        freq_minutes = int(self.config["resample_freq"].replace("min", ""))
        # 日内/跨日分别设阈值: 日内曲线平滑可用低阈值, 跨日日间差异大需高阈值
        k_within = self.config.get("spike_ratio_within",
                                   self.config.get("spike_ratio", 1.2))
        k_cross = self.config.get("spike_ratio_cross",
                                   self.config.get("spike_ratio", 2.0))
        within_steps = max(1, 60 // freq_minutes)      # 1h 折算步数
        day_steps = (24 * 60) // freq_minutes          # 1 天折算步数
        within_delta = pd.Timedelta(minutes=freq_minutes * within_steps)
        day_delta = pd.Timedelta(minutes=freq_minutes * day_steps)

        f = res["Total_Flow"]
        orig = f.copy()    # 原始值快照: res.loc 赋值会原地修改 f, 打印需用修正前的值

        # 邻居时间间隔校验: 只认"恰为 1h / 24h"的真邻居 (跨空洞的伪邻居不参与)
        # 用索引相减而非 diff: diff 只给相邻行的间隔, 跨 24 步必须 t - t[-24]
        idx = res.index.to_series()
        n1_ok = (idx - idx.shift(within_steps)) == within_delta        # t 的前 1h 参考
        n2_ok = (idx.shift(-within_steps) - idx) == within_delta       # t 的后 1h 参考
        p1_ok = (idx - idx.shift(day_steps)) == day_delta              # t 的前 1 天参考
        p2_ok = (idx.shift(-day_steps) - idx) == day_delta             # t 的后 1 天参考

        n1, n2 = f.shift(within_steps), f.shift(-within_steps)   # 日内参考: t±1h
        p1, p2 = f.shift(day_steps), f.shift(-day_steps)         # 跨日参考: t±1 天

        def flags(cur, r1, r2, ok1, ok2, k):
            ref_hi = pd.concat([r1, r2], axis=1).max(axis=1)
            ref_lo = pd.concat([r1, r2], axis=1).min(axis=1)
            hi = (cur > k * ref_hi) & (ref_hi > 0)
            lo = (cur < ref_lo / k) & (ref_lo > 0)
            # ok 含时间间隔校验; 参考点 NaN (被 Hampel 等置空) 时 ref 为 NaN → 判据不命中
            return (hi | lo) & ok1 & ok2 & r1.notna() & r2.notna()

        within_anom = flags(f, n1, n2, n1_ok, n2_ok, k_within)  # 判据1: 日内 t±1h
        day_anom = flags(f, p1, p2, p1_ok, p2_ok, k_cross)     # 判据2: 跨日 t±1 天
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
              f"[日内 k={k_within} {int(within_anom.sum())} / "
              f"跨日 k={k_cross} {int((day_anom & ~within_anom).sum())}]")
        for t in anomaly[anomaly].index[:5]:
            print(f"    {t}  {orig[t]:9.1f} → {fix[t]:9.1f}")
        return res

    def build_feature_table(self):
        df_raw = self.load_raw()
        df_base = self.build_base_features(df_raw)

        # 先按时间切分, 再分别清洗: 清洗不跨越训练/测试边界。
        test_days = self.config.get("test_days", 15)
        test_start = df_base.index[-1] - pd.Timedelta(days=test_days)
        df_base_train = df_base.loc[df_base.index <= test_start].copy()
        df_base_test = df_base.loc[df_base.index > test_start].copy()
        print(f"  清洗分段: 训练段 {df_base_train.index.min()} ~ {df_base_train.index.max()} "
              f"({len(df_base_train)} 行), 测试段 {df_base_test.index.min()} ~ {df_base_test.index.max()} "
              f"({len(df_base_test)} 行), 分段清洗互不跨界")

        df_clean = pd.concat([self.clean_and_resample(df_base_train),
                              self.clean_and_resample(df_base_test)])

        # 日历特征 + 数据驱动特征: 由时间索引 + 历史流量确定性生成, 训练/推理共用。
        df_feat = self.add_calendar_features(df_clean)
        df_feat = self.add_data_driven_features(df_feat)
        self.feature_cols = ["Total_Flow"] + list(CALENDAR_COLS) + list(DATA_DRIVEN_COLS)

        # 丢弃特征 NaN 行 (主要来自 30 天滚动窗口, 约前 720 行)
        n_before = len(df_feat)
        df_feat = df_feat.dropna(subset=self.feature_cols)
        n_dropped = n_before - len(df_feat)
        if n_dropped > 0:
            print(f"  丢弃特征 NaN 行: {n_dropped} 条 ({n_dropped/n_before:.1%}), 剩 {len(df_feat)}")

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

import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

# 保证能从本目录导入同目录共享模块 (模型结构 / 数据管线 / 训练配置, 与训练完全一致)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_transformer_autoregressive import BASE_CONFIG
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor, needed_csv_columns

# 以本文件所在目录为基准的相对路径 (transformer_pkg/), 可整体移植
HERE = os.path.dirname(os.path.abspath(__file__))
# 默认输入: 军山已合并小时级 CSV (与训练数据同构); 推理时也可传原始 DataFrame/CSV 路径
DEFAULT_DATA = r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv"
# 默认加载单步自回归 iTransformer 训练结果 (先运行 train_transformer_autoregressive.py 生成)
# 模型类型由结果目录 scaler.pkl 里的 model_type 字段自动判别, 无需手动切换
DEFAULT_RESULT_DIR = os.path.join(HERE, "results",
                                  "junshan_L1D_P24H_1h_itransformer_autoregressive")

# ── 分时压力默认参数 (厂方自行决定, 时段待定; 有需要可自行更改, 不改即默认) ──
# 每项: (起始小时, 结束小时, 目标压力 MPa), 区间左闭右开 [start, end)
DEFAULT_PRESSURE_SCHEDULE = [
    (0, 5, 0.30),    # 0-5点   0.3
    (5, 12, 0.33),   # 5-12点  0.33
    (12, 16, 0.33),  # 12-16点 0.33
    (16, 23, 0.33),  # 16-23点 0.33
    (23, 24, 0.30),  # 23-0点  0.3
]
DEFAULT_PRESSURE_ERROR = 0      # 典型压力误差
DEFAULT_PRESSURE_ERROR_MAX = 0   # 最大压力误差

# ── 日级缺口填充 (推理端专用, 不动训练共用管线 data_processing.py) ──
# 数据跨度内每个缺失的 bin 都用"离它最近一天的同时段"原始值补齐, 见
# FlowPredictor._fill_day_gaps。缺口无论大小一律补: 滞后特征最长 2 天, 单个缺失
# bin 也会级联删掉其后 2 天内的行 (输入不足时会报输入长度不足)


# ============================================================================
# inference_transformer_autoregressive.py — 单步自回归版推理脚本 (新文件, 不覆盖
# 原 inference_transformer.py)
#
# 与 inference_transformer.py 的区别:
#   inference_transformer.py        : 直接多步模型 (head 一次输出整个 horizon),
#                                    推理一次 forward 出 predict_steps 步, 不回灌。
#   本文件                          : 单步自回归模型 (head horizon=1), 推理须配套
#                                    autoregressive rollout —— 每步 model(window)
#                                    预测下一个点, 把预测回灌为输入窗口最新一行
#                                    (覆盖目标通道), 滑窗前进一步, 循环滚出整条
#                                    horizon。与 train_transformer_autoregressive.py
#                                    训练时的 rollout 完全配套。
#
# 推理端外生通道的补齐 (训练时 future_exog 用 ground-truth; 推理无未来真值, 须构造):
#   自回归 rollout 每一步需要一个完整的"未来特征行" (与训练 make_sequences_ar 返回
#   的 Xf 同构), 其中目标通道 (Total_Flow) 由本轮预测覆盖, 其余外生通道须自行构造:
#     - 时间特征 (hour/dayofweek/hour_sin/cos/...): 由未来时间戳确定性地算出。
#     - Target_Pressure: 按分时压力时段表 (DEFAULT_PRESSURE_SCHEDULE) 取目标压力。
#     - 泵状态/频率/运行泵数量: 持平最后已知值 (hold-last; 真实未来泵调度未知)。
#     - Total_Flow 衍生的滞后/滚动/差分特征 (lag/roll/diff/trend/volatility/
#       flow_lag_1day/2day 等): 用"历史真值 + 已预测未来"按训练同一套
#       add_lag_rolling_features 因果地滚动重算 —— 每预测一步就把预测流量追加以
#       原始单位拼到清洗帧末尾, 重算特征, 取最新一行作为下一步的外生行。
#       这与训练 (future_exog 用 ground-truth 衍生) 存在 inherent exposure gap,
#       是自回归模型固有的, 不可避免; detach_feedback=True 训练已缓解 exposure bias。
#
# 其余 (数据清洗 / 特征工程 / 输入三模式 / 日级缺口填充 / 按分时压力生成压力预测)
# 与 inference_transformer.py 完全一致, 仅替换"前向输出方式"。
# ============================================================================


class FlowPredictor:
    """基于 train_transformer_autoregressive.py 训练结果的流量预测推理接口 (自回归版)。

    加载训练产物 (best_seq2seq_model.pth + scaler.pkl, 后者含 autoregressive=True /
    target_feat_idx / detach_feedback), 接受最近 lookback_days 天的原始数据, 用单步
    自回归滚动 rollout 输出未来 predict_days 天的小时级预测流量序列。

    输入 DataFrame 格式与训练数据一致 (D:\\Junshan_Project\\data\\水厂2025年小时级汇总.csv 同构):
      时间列:  时间 / timestamp (整点小时级, 或直接给 DatetimeIndex 索引)
      必需列:  出厂水流量 (m³/h), 出厂水压力 (MPa)
      可选列:  1#/6#泵运行频率 (Hz), 1#/2#/6#送水泵运行 (0/1; 缺失时泵台数按 0)

    时间跨度: 至少 lookback_days + 2 天 (最长滞后特征为 2 天, 见下注),
              建议提供 7 天以上, 前 2 天用于填满滞后特征。
      注: 特征里 flow_lag_2day 需要 2 天前的值, 所以只给恰好 lookback 天的
          数据会导致滞后特征全为 NaN 被删光, 必须多带历史。
      缺口处理: 数据跨度内 (首尾天按完整 24h 网格) 缺失的 bin 一律用离它最近
          的一天同时段数据补齐 (见 _fill_day_gaps), 保证时间轴连续、长度校验通过;
          仍找不到的时刻保持缺失, 由后续 dropna 兜底。

    用法 (程序接口, 三种输入模式, 返回完全一致):
      predictor = FlowPredictor()

      # 模式1: 直接给出 DataFrame (已在内存中, 不读 CSV)
      pred = predictor.predict(df_raw)

      # 模式2: 以 CSV 文件路径给出 (接口内部读 CSV)
      pred = predictor.predict("input.csv")

      # 模式3: 以列表/数组直接给出 (列名与训练数据一致)
      rows = [{"时间": "2025-07-15 06:00:00", "出厂水流量": 1621.6, ...}, ...]
      pred = predictor.predict(rows)                       # list[dict]
      pred = predictor.predict(rows2, columns=["时间", "出厂水流量", ...])  # list[list]/ndarray 需列名

      # 输出也可为列表形式: 传入 as_list=True, 返回 list[dict]
      pred_list = predictor.predict(df_raw, as_list=True)  # [{"timestamp": ..., "Total_Flow": ...}, ...]

    ⚠ 本预测器仅适配单步自回归训练结果 (scaler.pkl 里 autoregressive=True)。
      若用直接多步模型 (train_transformer.py 的结果), 请改用 inference_transformer.py。
    """

    def __init__(self, result_dir=DEFAULT_RESULT_DIR, device=None,
                 pressure_schedule=None, pressure_error=None, pressure_error_max=None):
        self.result_dir = result_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── 加载训练时保存的 scaler / 特征列 / 配置 (自回归字段) ──
        scaler_path = os.path.join(result_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"未找到 scaler.pkl: {scaler_path}\n"
                f"单步自回归推理必须配套训练时保存的 scaler.pkl (含 autoregressive/"
                f"target_feat_idx/detach_feedback)。请先运行 train_transformer_autoregressive.py。")
        with open(scaler_path, "rb") as f:
            saved = pickle.load(f)
        self._apply_saved(saved)
        self.autoregressive = saved.get("autoregressive", False)

        # 校验: 必须是自回归训练产物 (autoregressive 是 scaler.pkl 顶层键, 不在 config 内)
        if not self.autoregressive:
            raise ValueError(
                f"scaler.pkl 里 autoregressive != True, 该结果目录不是单步自回归训练产物: "
                f"{scaler_path}\n请改用 inference_transformer.py (直接多步推理), 或先运行 "
                f"train_transformer_autoregressive.py 训练自回归模型。")

        self.lookback_days = self.config["lookback_days"]
        self.predict_days = self.config["predict_days"]
        self.resample_freq = self.config["resample_freq"]
        self.freq_minutes = int(self.resample_freq.replace("min", ""))
        self.points_per_day = (24 * 60) // self.freq_minutes
        self.lookback_steps = int(self.lookback_days * self.points_per_day)
        self.predict_steps = int(self.predict_days * self.points_per_day)

        # 目标通道 (Total_Flow) 在 feature_cols 中的下标 —— 回灌时覆盖该通道
        self.target_feat_idx = saved.get("target_feat_idx")
        if self.target_feat_idx is None:
            # 旧产物兜底: 用 feature_cols 里 target_cols[0] 的位置
            self.target_feat_idx = self.feature_cols.index(self.target_cols[0])
        self.detach_feedback = saved.get("detach_feedback", True)   # 推理无梯度, 仅记录

        # ── 加载模型权重 ──
        model_path = os.path.join(result_dir, "best_seq2seq_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型权重: {model_path}")
        # ── 模型工厂: horizon=1 单步头 (与训练一致), model_type 由 scaler.pkl 判别 ──
        model_type = self.config.get("model_type", "transformer")
        common_model_kwargs = dict(
            input_dim=len(self.feature_cols),
            output_dim=1,
            horizon=1,                        # ← 单步头: 只预测下一个点 (与训练一致)
            input_len=self.lookback_steps,
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["transformer_dropout"],
        )
        if model_type == "itransformer":
            self.model = iTransformer(
                **common_model_kwargs,
                target_idx=self.target_feat_idx,
            ).to(self.device)
        else:
            self.model = TimeSeriesTransformer(**common_model_kwargs).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # ── 复用训练的特征工程对象 (清洗/重采样/时间/滞后特征完全一致) ──
        self.processor = DataProcessor(self.config)
        self.processor.feature_scaler = self.feature_scaler
        self.processor.target_scaler = self.target_scaler
        self.processor.feature_cols = self.feature_cols

        # ── 分时压力参数 (默认见模块顶部, 有需要可自行更改) ──
        self.pressure_schedule = (list(DEFAULT_PRESSURE_SCHEDULE)
                                  if pressure_schedule is None else pressure_schedule)
        self.pressure_error = DEFAULT_PRESSURE_ERROR if pressure_error is None else pressure_error
        self.pressure_error_max = (DEFAULT_PRESSURE_ERROR_MAX
                                   if pressure_error_max is None else pressure_error_max)

        print(f"[FlowPredictor-AR] 模型已加载: {os.path.basename(model_path)} "
              f"(model_type={model_type}, horizon=1 单步自回归)")
        print(f"[FlowPredictor-AR] lookback={self.lookback_days}d ({self.lookback_steps}步), "
              f"predict={self.predict_days}d ({self.predict_steps}步, 滚动 {self.predict_steps} 次), "
              f"freq={self.resample_freq}, 特征数={len(self.feature_cols)}, "
              f"d_model={self.config['d_model']}, nhead={self.config['nhead']}, "
              f"layers={self.config['num_layers']}, device={self.device}")
        print(f"[FlowPredictor-AR] target_feat_idx={self.target_feat_idx} "
              f"({self.target_cols[0]}), detach_feedback={self.detach_feedback} (推理无梯度, 仅记录)")

    def _apply_saved(self, saved):
        self.config = saved["config"]
        self.feature_scaler = saved["feature_scaler"]
        self.target_scaler = saved["target_scaler"]
        self.feature_cols = saved["feature_cols"]
        self.target_cols = saved["target_cols"]

    def predict(self, data, columns=None, encoding="utf-8-sig", as_list=False):
        """统一推理接口, 支持三种输入模式 (返回结果完全一致):

        模式1 (直接给出): data 为 pd.DataFrame, 原始数据已在内存中, 不读 CSV
        模式2 (CSV 给出): data 为 CSV 文件路径 (str / os.PathLike)
        模式3 (列表给出): data 为 list[dict] / list[list] / list[tuple] /
                          np.ndarray, 直接以 Python 数据结构给出输入参数

        Parameters
        ----------
        data : pd.DataFrame | str | os.PathLike | list | tuple | np.ndarray
            原始数据或其 CSV 路径; 列格式见类说明, 时间列可作索引或普通列。
        columns : list[str] | None
            仅模式3的位置型数据 (list[list] / ndarray) 需要: 原始数据列名,
            顺序与每行一致 (与训练数据列名相同); list[dict] 时可不传。
        encoding : str
            仅模式2的 CSV 文件编码 (默认 utf-8-sig)。
        as_list : bool
            为 True 时返回 list[dict] (每项含 timestamp / Total_Flow, 与
            模式3的 list[dict] 输入格式对称); 默认 False 返回 DataFrame。

        Returns
        -------
        pd.DataFrame 或 list[dict]
            DataFrame: index = 预测时刻 (小时级分辨率), 列 Total_Flow = 预测流量。
            list[dict]: [{"timestamp": "YYYY-MM-DD HH:MM:SS", "Total_Flow": 值}, ...]
        """
        if isinstance(data, pd.DataFrame):
            result = self._predict_df(data)
        elif isinstance(data, (str, os.PathLike)):
            result = self._predict_from_csv(data, encoding)
        elif isinstance(data, np.ndarray):
            result = self._predict_from_list(data, columns)
        elif isinstance(data, (list, tuple)):
            result = self._predict_from_list(data, columns)
        else:
            raise TypeError(
                f"不支持的数据类型: {type(data).__name__}; 请传入 pd.DataFrame / "
                f"CSV 路径 / 列表或数组")

        if as_list:
            return self._to_list(result)
        return result

    @staticmethod
    def _to_list(result):
        """将预测结果 DataFrame 转为 list[dict] (含 timestamp 与 Total_Flow)。"""
        return [
            {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "Total_Flow": float(v)}
            for ts, v in result["Total_Flow"].items()
        ]

    @staticmethod
    def _pressure_target(ts, schedule):
        """按小时查找时刻 ts 对应的分时压力目标值 (时段表左闭右开)。"""
        hour = ts.hour
        for start, end, target in schedule:
            if start <= hour < end:
                return target
        raise ValueError(
            f"时刻 {ts} (小时 {hour}) 不在分时压力时段内, "
            f"请检查 pressure_schedule: {schedule}")

    def predict_pressure(self, pred, pressure_schedule=None, pressure_error=None,
                         pressure_error_max=None, as_list=False):
        """基于 predict() 的结果生成分时压力预测, 返回新变量 (不覆盖原 pred)。

        默认分时压力时段 (厂方待定, 可自行更改):
            0-5点 0.3 | 5-12点 0.33 | 12-16点 0.33 | 16-23点 0.33 | 23-0点 0.3
        默认压力误差: 典型 0.02, 最大 0.03 (误差幅值在 [0.02, 0.03] 内随机取值,
        方向随机; 传 pressure_error=0 可使误差在 [0, 0.03] 内取值)。

        注: 自回归 rollout 期间, 输入特征里的未来 Target_Pressure 已用同一时段表的
        目标压力 (无误差) 补齐; 本方法在输出端再叠加误差, 仅供输出展示, 不回灌模型。

        Parameters
        ----------
        pred : pd.DataFrame | list[dict]
            predict() 的输出 (DataFrame: DatetimeIndex + Total_Flow;
            list[dict]: timestamp + Total_Flow)。
        pressure_schedule : list[(int, int, float)] | None
            分时压力时段表 [(起始小时, 结束小时, 目标压力), ...], 默认模块级
            DEFAULT_PRESSURE_SCHEDULE。
        pressure_error / pressure_error_max : float | None
            典型 / 最大压力误差幅值, 默认 DEFAULT_PRESSURE_ERROR / _MAX。
        as_list : bool
            与 predict() 一致: True 返回 list[dict], False 返回 DataFrame。

        Returns
        -------
        pd.DataFrame | list[dict]
            新变量 (不改动传入的 pred): 按时间顺序排列, 与 pred 时刻一一对应,
            在原有 Total_Flow 基础上新增 Pressure 列/键。
        """
        schedule = self.pressure_schedule if pressure_schedule is None else pressure_schedule
        err = self.pressure_error if pressure_error is None else pressure_error
        err_max = self.pressure_error_max if pressure_error_max is None else pressure_error_max
        if err > err_max:
            raise ValueError(f"pressure_error ({err}) 不应大于 pressure_error_max ({err_max})")

        # 统一为按时间排序的 DataFrame (copy, 不修改传入的 pred)
        if isinstance(pred, pd.DataFrame):
            result = pred.copy()
        else:
            result = pd.DataFrame(list(pred))
            result["timestamp"] = pd.to_datetime(result["timestamp"])
            result = result.set_index("timestamp")
        result = result.sort_index()

        pressures = []
        for ts in result.index:
            target = self._pressure_target(ts, schedule)
            err_mag = np.random.uniform(err, err_max)          # 误差幅值 [典型, 最大]
            pressures.append(round(target + (err_mag if np.random.rand() < 0.5 else -err_mag), 3))
        result["Pressure"] = pressures

        if as_list:
            return [
                {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                 "Total_Flow": float(row["Total_Flow"]),
                 "Pressure": float(row["Pressure"])}
                for ts, row in result.iterrows()
            ]
        return result

    def _predict_from_csv(self, csv_path, encoding="utf-8-sig"):
        """模式2: 以 CSV 文件路径给出输入, 读取后交给 _predict_df 执行。"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"输入 CSV 不存在: {csv_path}")
        # 只读管线必需列, 与训练 load_raw 同口径
        header = pd.read_csv(csv_path, encoding=encoding, nrows=0).columns
        df_raw = pd.read_csv(csv_path, encoding=encoding,
                             usecols=needed_csv_columns(header))
        print(f"[predict-AR] 模式2: 从 CSV 读取原始数据: {csv_path} ({len(df_raw)} 行)")
        return self._predict_df(df_raw)

    def _predict_from_list(self, data, columns=None):
        """模式3: 直接以列表/数组给出输入参数, 不读 CSV。

        支持形式:
          list[dict] / tuple[dict]   → 键为列名 (列名与训练数据一致)
          list[list] / list[tuple]   → 位置型行数据, 必须配合 columns 给列名
          np.ndarray (1D/2D)         → 单行/多行位置型数据, 必须配合 columns
          扁平 list[标量]             → 单行位置型数据, 必须配合 columns
        """
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return self._predict_from_positional([list(data)], columns)
            if data.ndim == 2:
                return self._predict_from_positional(data.tolist(), columns)
            raise ValueError(f"数组维度过高: {data.ndim}D, 只支持 1D(单行) / 2D(多行)")
        if len(data) == 0:
            raise ValueError("输入列表为空")
        if isinstance(data[0], dict):
            # 列名来自 dict 键 (与训练数据列名一致)
            return self._predict_df(pd.DataFrame(list(data)))
        return self._predict_from_positional(list(data), columns)

    def _predict_from_positional(self, rows, columns):
        """位置型数据 (list[list] 等) → DataFrame; 必须提供原始数据列名。"""
        if columns is None:
            raise ValueError(
                "位置型列表/数组输入必须提供 columns 参数 (原始数据列名, 与训练数据一致)")
        columns = list(columns)
        if len(rows) > 0 and len(columns) != len(rows[0]):
            raise ValueError(
                f"columns 长度 ({len(columns)}) 与数据宽度 ({len(rows[0])}) 不一致")
        return self._predict_df(pd.DataFrame(rows, columns=columns))

    def _fill_day_gaps(self, df_clean):
        """日级缺口填充: 数据跨度内缺失的 bin 用离它最近一天的同时段数据补齐.

        重采样后的空 bin 已被 dropna 删掉 (表里没有的槽位即缺失); 跨度按完整日历天
        取网格 [首日 00:00, 末日 23:00], 跨度内每个缺失 bin 逐圈找 (d-1, d+1,
        d-2, d+2, ...) 取第一个该时刻有原始数据的一天, 整行复制 (流量/压力/泵数)。

        为什么缺口无论大小一律补: 滞后特征最长 2 天, 单个缺失 bin 会让其后 2 天内
        的行因滞后特征 NaN 被 dropna 级联删除——输入不足时触发输入长度报错。
        补齐后时间轴连续, 特征计算正常。

        输入数据所有天都缺的时刻保持缺失, 由后续 dropna 与输入长度校验兜底。
        源只取原始数据 (填充结果不参与做源)。已有值一律不动。
        """
        if df_clean is None or df_clean.empty:
            return df_clean

        # 完整日历天网格: 首日 00:00 ~ 末日 23:00 (首/尾天内部的缺口也补)
        grid_start = df_clean.index.min().normalize()
        grid_end = (df_clean.index.max().normalize()
                    + pd.Timedelta(days=1) - pd.Timedelta(minutes=self.freq_minutes))
        expected = pd.date_range(grid_start, grid_end, freq=self.resample_freq)
        if len(expected) == len(df_clean):
            return df_clean                              # 无缺口, 直接返回

        present = set(df_clean.index)
        missing = expected.difference(df_clean.index)

        filled = {}
        max_dist = (expected[-1].normalize() - expected[0].normalize()).days
        for s in missing:
            d, t = s.normalize(), s - s.normalize()      # 日历天 / 天内的时刻偏移
            for dist in range(1, max_dist + 1):
                for cand in (d - pd.Timedelta(days=dist), d + pd.Timedelta(days=dist)):
                    src = cand + t
                    if src in present:
                        filled[s] = df_clean.loc[src]
                        break
                if s in filled:
                    break

        if filled:
            df_clean = (pd.concat([df_clean, pd.DataFrame(filled).T])
                        .sort_index())
            print(f"[fill_day_gaps] 补齐 {len(filled)} 个缺失 bin "
                  f"(网格 {len(expected)} 槽位, 取最近一天同时段原始值)")
        return df_clean

    def _predict_df(self, raw_df):
        """模式1: 直接给出内存中的 DataFrame, 不读 CSV (核心实现, 单步自回归版)。"""
        df = raw_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            for ts_col in ("时间", "timestamp"):
                if ts_col in df.columns:
                    df[ts_col] = pd.to_datetime(df[ts_col])
                    df = df.set_index(ts_col)
                    break
            else:
                raise ValueError("输入数据必须包含时间列 (时间 / timestamp) 或 DatetimeIndex 索引")
        df = df.sort_index()
        print(f"[predict-AR] 输入数据: {df.index.min()} ~ {df.index.max()}, {len(df)} 行")

        # ── ① 清洗 + 特征工程 (与训练完全相同; 单段整体清洗, 无 train/test 切分) ──
        df_base = self.processor.build_base_features(df)
        df_clean = self.processor.clean_and_resample(df_base)
        # 日级缺口填充: 某天缺失大量 bin 时, 用最近一天同时段数据补齐 (推理端专用)
        df_clean = self._fill_day_gaps(df_clean)
        df_time = self.processor.add_time_features(df_clean)
        df_feat = self.processor.add_lag_rolling_features(df_time)   # 内含 dropna

        # 校验特征列与训练一致 (输入格式不同会导致特征集合不同)
        missing = [c for c in self.feature_cols if c not in df_feat.columns]
        if missing:
            raise ValueError(
                f"特征列与训练不一致, 缺失: {missing[:5]}...\n"
                f"请检查输入数据列与训练数据 (水厂2025年小时级汇总.csv) 是否同构")
        df_feat = df_feat[self.feature_cols]

        if len(df_feat) < self.lookback_steps:
            raise ValueError(
                f"输入历史不足: 清洗+特征工程后仅剩 {len(df_feat)} 行, "
                f"模型需要 ≥ {self.lookback_steps} 行 (lookback={self.lookback_days}天)。\n"
                f"最长滞后特征为 2 天, 建议提供 ≥ {self.lookback_days + 2} 天的原始数据。")
        print(f"[predict-AR] 清洗+特征工程后: {len(df_feat)} 行 "
              f"({df_feat.index.min()} ~ {df_feat.index.max()})")

        # ── ② 取最后 lookback_steps 行作为模型输入窗口 (scaled) ──
        window_feat = df_feat.iloc[-self.lookback_steps:]
        X = self.feature_scaler.transform(window_feat.values.astype(np.float32))
        window = torch.from_numpy(X).unsqueeze(0).to(self.device)   # (1, L, C)

        # ── ③ 单步自回归滚动 rollout (与训练 autoregressive_rollout 配套) ──
        # 训练时 future_exog 用 ground-truth; 推理无未来真值, 用下面构造的外生行:
        #   时间特征由时间戳算; 压力用分时时段表; 泵状态/频率持平最后已知值;
        #   Total_Flow 衍生的滞后/滚动特征用"历史真值 + 已预测未来"因果滚动重算。
        last_ts = df_clean.index[-1]
        future_idx = pd.date_range(
            start=last_ts + pd.Timedelta(minutes=self.freq_minutes),
            periods=self.predict_steps, freq=self.resample_freq)

        # 清洗帧 (原始单位) 用于逐步追加预测行并重算特征; 复制最后一行作为
        # 未来外生行的模板 (泵状态/频率/运行泵数量等非流量列持平最后已知值)。
        df_clean_ext = df_clean.copy()
        last_row_tmpl = df_clean.iloc[-1].copy()

        preds_scaled = []   # 每步预测 (target 域, scaled), 末尾统一反归一化
        pump_cols = [c for c in df_clean.columns if "泵运行" in c and "频率" not in c]

        with torch.no_grad():
            for k in range(self.predict_steps):
                # 单步预测下一个时刻的流量 (target 域, scaled)
                one = self.model(window, target_len=1)              # (1, 1, 1)
                pred_scaled = float(one[0, 0, 0].cpu().numpy())
                preds_scaled.append(pred_scaled)

                # 反归一化得原始流量单位 (供追加到清洗帧以重算衍生的滞后/滚动特征)
                pred_flow_real = float(
                    self.processor.inverse_transform_targets(
                        np.array([[[pred_scaled]]], dtype=np.float32))[0, 0, 0])

                # 构造未来第 k 步的外生行 (原始单位): 复制最后行模板, 覆盖流量与压力
                ts_k = future_idx[k]
                new_row = last_row_tmpl.copy()
                new_row["Total_Flow"] = pred_flow_real
                new_row["Target_Pressure"] = self._pressure_target(ts_k, self.pressure_schedule)
                # 运行泵数量/泵状态/频率: 持平最后已知值 (new_row 已从 last_row_tmpl 继承)

                # 追加到清洗帧, 重算时间/滞后滚动特征, 取最新一行 (即未来第 k 步特征行)
                df_clean_ext.loc[ts_k] = new_row
                df_time_ext = self.processor.add_time_features(df_clean_ext)
                df_feat_ext = self.processor.add_lag_rolling_features(df_time_ext)  # 内含 dropna
                next_feat = df_feat_ext[self.feature_cols].iloc[-1].values.astype(np.float32)

                # scaled 特征行; 覆盖目标通道为本轮预测 (与训练 rollout 一致)
                next_feat_scaled = self.feature_scaler.transform(next_feat.reshape(1, -1))[0]
                next_feat_scaled[self.target_feat_idx] = pred_scaled

                # 滑窗: 丢最旧一行, 末尾拼上未来第 k 步特征行
                next_row_t = torch.from_numpy(
                    next_feat_scaled.astype(np.float32)).view(1, 1, -1).to(self.device)
                window = torch.cat([window[:, 1:, :], next_row_t], dim=1)

        # ── ④ 反归一化整条预测序列 (与训练 evaluate 同口径) ──
        preds_arr = np.array(preds_scaled, dtype=np.float32).reshape(1, self.predict_steps, 1)
        y_inv = self.processor.inverse_transform_targets(preds_arr)[0]   # (predict_steps, 1)

        result = pd.DataFrame({"Total_Flow": y_inv[:, 0]}, index=future_idx)
        result.index.name = "timestamp"
        print(f"[predict-AR] 输出: {result.index[0]} ~ {result.index[-1]}, "
              f"{len(result)} 行 ({self.resample_freq}, 单步自回归滚动 {self.predict_steps} 次)")
        return result


def main():
    parser = argparse.ArgumentParser(
        description="单步自回归 Transformer / iTransformer 流量预测推理")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="输入原始数据 CSV 路径 (与训练数据格式一致)")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR,
                        help="训练结果目录 (含 best_seq2seq_model.pth 与 scaler.pkl, "
                             "须为 train_transformer_autoregressive.py 产物)")
    parser.add_argument("--out", default="prediction_ar.csv",
                        help="预测结果 CSV 输出路径 (默认 prediction_ar.csv)")
    args = parser.parse_args()

    predictor = FlowPredictor(args.result_dir)
    pred = predictor.predict(args.data, as_list=True)   # 统一接口: 自动识别 CSV 路径
    print(f'pred:{pred}')

    # 基于 pred 按分时压力时段生成压力预测 → 新变量 pred_pressure, 不覆盖原 pred
    # (默认时段/误差见模块顶部, 有需要可自行更改)
    pred_pressure = predictor.predict_pressure(pred, as_list=True)
    print(f'pred_pressure:{pred_pressure}')

    # pred_df = predictor.predict(args.data)
    # pred_df.to_csv(args.out, index=True, encoding="utf-8-sig")
    # print(f"\n预测结果已保存: {args.out}")
    # print(pred_df.head(10))
    # print(f"... (共 {len(pred_df)} 行, {pred_df.index[0]} ~ {pred_df.index[-1]})")


if __name__ == "__main__":
    main()

# 军山水厂流量预测项目 (Junshan Water Plant Flow Prediction)

> 基于 Transformer / iTransformer 的出厂水流量小时级预测系统，采用单步自回归滚动预测架构，支持日历特征增强与残差修正。

## 项目概述

本项目针对武汉军山水厂的出厂水流量进行小时级预测，采用深度学习 Transformer 架构，通过自回归滚动方式预测未来 24 小时流量。项目包含完整的数据处理、模型训练、评估与推理流程。

### 核心特性

- **双模型架构**：支持标准 Transformer 和 iTransformer（变量即 token）两种模型
- **自回归预测**：单步预测头 + 滚动 rollout，缓解 exposure bias
- **丰富特征工程**：日历特征（12 维）+ 数据驱动特征（17 维）+ 流量本身
- **多尺度周期编码**：小时/星期/月份/年日的 sin/cos 编码，平滑周期边界
- **中国节假日感知**：集成 chinese-calendar 包，区分法定假日与调休
- **稳健数据清洗**：突变检测 + Hampel 离群过滤 + 物理界限裁剪
- **峰值样本增强**：对高峰时段样本上采样，提升峰值预测精度

## 项目结构

```
Junshan_Project/
├── transformer_pkg/                  # 核心模型包
│   ├── __init__.py                   # 包初始化
│   ├── transformer_model.py          # TimeSeriesTransformer 模型定义
│   ├── itransformer_model.py         # iTransformer 模型定义
│   ├── data_processing.py            # 数据清洗与特征工程
│   ├── train_transformer_autoregressive.py  # 自回归训练脚本
│   ├── eval_random_window.py         # 指定窗口评估（与训练同口径）
│   ├── eval_rolling_mape.py          # 滚动窗口 MAPE 评估
│   ├── eval_rolling_ar.py            # 自回归滚动评估
│   ├── inference_transformer.py      # 直接多步推理
│   ├── inference_ar.py               # 自回归推理
│   └── results/                      # 训练结果目录
│       └── junshan_L1D_P24H_1h_transformer_autoregressive_*/
│           ├── best_seq2seq_model.pth  # 最优模型权重
│           ├── scaler.pkl              # 归一化器与配置
│           ├── metrics.txt             # 评估指标
│           ├── train_history.csv       # 训练历史
│           ├── loss_curve.png          # 损失曲线
│           └── daily_plots/            # 逐日预测图
├── data/                             # 数据目录
│   ├── 水厂2025年小时级汇总.csv       # 合并后的 2025 年小时级数据
│   ├── 出厂水流量（2025-01-01至2026-01-01).xlsx
│   ├── 出厂水压力2025.xlsx
│   ├── 送水泵运行频率2025.xlsx
│   ├── 送水泵运行状态2025.xlsx
│   ├── 武汉军山 流量_2024-01-01至2025-01-01.xlsx
│   ├── merge_data.py                 # 数据合并脚本
│   └── merge_2024_2025.py            # 2024-2025 数据合并
├── plots/                            # 逐日预测图（2025-10 ~ 2025-12）
├── flow_plots/                       # 流量分析图
├── analyze_flow.py                   # 流量与压力数据分析
├── residual_correction_compare.py    # 残差修正方法对比
├── residual_correction_comparison.png # 残差修正对比图
├── monthly_total_flow_2024_2025.png  # 月度流量汇总图
├── prediction_ar.csv                 # 自回归预测结果
├── timeseries_transformer_optimization.md  # Transformer 优化方案文档
├── requirements.txt                  # Python 依赖
├── run.bat                           # Windows 训练启动脚本
└── README.md                         # 本文件
```

## 模型架构

### 1. TimeSeriesTransformer

标准 Transformer Encoder 架构：

```
输入 (B, L, C) → 特征投影 → 正弦位置编码 → N 层 Transformer Encoder → 取最后时刻 → 线性输出头
```

- **特征投影**：`Linear(input_dim → d_model)`
- **位置编码**：正弦绝对位置编码（与窗口长度解耦）
- **编码器**：N 层 `nn.TransformerEncoder`，多头自注意力
- **输出头**：`Linear(d_model → horizon * output_dim)`

### 2. iTransformer

变量即 token 的 Transformer（Liu et al., ICLR 2024）：

```
输入 (B, L, C) → RevIN 实例归一化 → 转置 (B, C, L) → 共享 Linear(L → d_model) → N 层 Encoder → 输出头 → RevIN 反归一化
```

- **RevIN**：可逆实例归一化，逐通道均值/方差归一化
- **变量位置编码**：可学习的变量位置嵌入
- **跨变量注意力**：对变量维度做自注意力，捕捉变量间关系

### 自回归预测模式

```python
# 单步预测头 (horizon=1) + 滚动 rollout
for k in range(predict_steps):
    pred = model(window, target_len=1)      # 预测下一个点
    next_row = future_features[k].copy()
    next_row[target_idx] = pred             # 回灌预测值
    window = cat([window[1:], next_row])    # 滑窗前进
```

## 数据处理

### 数据源

- **时间范围**：2024-01-01 ~ 2025-12-31（约 17,520 小时）
- **采样频率**：60 分钟（整点）
- **主要特征**：出厂水流量（m³/h）
- **辅助特征**：出厂水压力、泵运行频率、泵运行状态

### 数据清洗流程

1. **突变流量插值修正**：日内（t±1h）/ 跨日（t±24h）双判据
2. **Hampel 离群过滤**：滚动中位数 + MAD 稳健离群检测（窗口 48h）
3. **物理界限裁剪**：流量范围 [0, 10000] m³/h
4. **重采样**：60 分钟整点，空 bin 保持 NaN 不虚构

### 特征工程

#### 日历特征（12 维）

| 特征 | 说明 | 周期 |
|------|------|------|
| `hour_sin`, `hour_cos` | 小时相位编码 | 24h |
| `dow_sin`, `dow_cos` | 星期几编码 | 7 天 |
| `month_sin`, `month_cos` | 月份编码 | 12 月 |
| `doy_sin`, `doy_cos` | 年日序号编码 | 365 天 |
| `is_workday` | 调休感知工作日 | - |
| `is_holiday` | 法定假日 | - |
| `holiday_eve` | 假期前一天 | - |
| `holiday_next` | 假期后一天 | - |

#### 数据驱动特征（17 维）

| 类别 | 特征 | 说明 |
|------|------|------|
| 滞后 | `lag_24h`, `lag_168h` | 昨天/上周同时段流量 |
| 滞后变化 | `lag_24h_diff`, `lag_24h_ratio` | 日变化量/比率 |
| 滚动统计 | `roll_mean_3d/7d/30d` | 多尺度均值 |
| 滚动统计 | `roll_std_7d`, `roll_max/min_7d`, `roll_median_48h` | 波动/极值 |
| 日内形状 | `day_avg`, `day_peak_amp` | 日均值、峰谷幅度 |
| 高峰比例 | `morning/evening/night_ratio` | 早晚高峰/夜间比例 |
| 偏离 | `flow_zscore_7d/30d` | Z-score 偏离程度 |

## 快速开始

### 环境要求

- Python 3.9+
- PyTorch 2.0+
- CUDA（可选，推荐 GPU 训练）

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
```
numpy>=1.24
pandas>=2.0
torch>=2.0
scikit-learn>=1.3
matplotlib>=3.7
tqdm>=4.60
taospy>=2.8.10
pyarrow>=14
chinese-calendar  # 可选，用于中国节假日特征
```

### 数据准备

1. 将原始 Excel 文件放入 `data/` 目录
2. 运行数据合并脚本：

```bash
cd data
python merge_data.py
```

生成 `水厂2025年小时级汇总.csv`

### 模型训练

#### 方式一：使用启动脚本（Windows）

```bash
run.bat
```

#### 方式二：命令行训练

```bash
cd transformer_pkg

# 训练 Transformer 自回归模型
python train_transformer_autoregressive.py

# 指定模型类型
python train_transformer_autoregressive.py --model transformer
python train_transformer_autoregressive.py --model itransformer

# 自定义参数
python train_transformer_autoregressive.py --lookback 14 --label my_experiment
```

### 模型评估

```bash
cd transformer_pkg

# 指定窗口评估（与训练同口径）
python eval_random_window.py --start_date 2025-10-03 --all_days

# 指定日期范围 + 逐日画图
python eval_random_window.py --start_date 2025-10-03 --end_date 2025-12-31 --plot

# 滚动窗口 MAPE 评估
python eval_rolling_mape.py --data ../data/水厂2025年小时级汇总.csv
```

### 模型推理

```bash
cd transformer_pkg

# 直接多步推理
python inference_transformer.py --data input.csv

# 自回归推理
python inference_ar.py --data input.csv --with_pressure
```

#### 程序接口

```python
from inference_ar import FlowPredictor

# 加载模型
predictor = FlowPredictor("results/junshan_L1D_P24H_1h_transformer_autoregressive_20260824_234455")

# 方式 1: CSV 文件
pred = predictor.predict("input.csv")

# 方式 2: DataFrame
import pandas as pd
df = pd.read_csv("input.csv")
pred = predictor.predict(df)

# 方式 3: 字典列表
rows = [{"时间": "2025-07-15 06:00:00", "出厂水流量": 1621.6}, ...]
pred = predictor.predict(rows)

# 获取压力预测
pred_with_pressure = predictor.predict_pressure(pred)
```

## 训练配置

### 默认超参数

```python
BASE_CONFIG = {
    # 数据
    "file_path": "data/水厂2025年小时级汇总.csv",
    "resample_freq": "60min",
    "stride": 1,
    "lookback_days": 7,
    "predict_days": 1.0,
    "test_days": 90,

    # Transformer 架构
    "d_model": 32,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 256,
    "transformer_dropout": 0.2,
    "model_type": "transformer",  # transformer | itransformer

    # 自回归
    "detach_feedback": True,

    # 训练
    "batch_size": 16,
    "epochs": 40,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "patience": 10,

    # 峰值增强
    "peak_augment_ratio": 0.3,
    "peak_threshold_ratio": 0.7,
}
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lookback_days` | 7 | 回看天数（输入窗口长度） |
| `predict_days` | 1.0 | 预测天数（24 小时） |
| `d_model` | 32 | Transformer 隐维度 |
| `nhead` | 4 | 注意力头数 |
| `num_layers` | 3 | Encoder 层数 |
| `detach_feedback` | True | 回灌预测值是否 detach（True=稳定省显存） |
| `stride` | 1 | 滑窗步长（1=每个时刻一个样本） |

## 性能指标

### 最优模型结果

| 数据集 | MAE (m³/h) | RMSE (m³/h) | MAPE (%) |
|--------|------------|-------------|----------|
| Train  | 91.74      | 128.41      | 3.06     |
| Test   | 152.68     | 211.11      | **5.10** |

- **模型**：iTransformer（自回归模式）
- **测试集**：最后 90 天
- **MAPE 过滤**：排除 |true| < 10% * max|true| 的近零流量点

### 残差修正方法对比

| 方法 | MAE | MAE 提升 | MAPE (%) | MAPE 提升 |
|------|-----|----------|----------|-----------|
| Baseline | 152.68 | - | 5.10 | - |
| 滑动窗口 (w=168) | 148.23 | +2.9% | 4.95 | +2.9% |
| EWMA (α=0.3) | 147.89 | +3.1% | 4.92 | +3.5% |
| 卡尔曼滤波 | 146.54 | +4.0% | 4.87 | +4.5% |
| STL 分解 | 145.12 | +4.9% | 4.82 | +5.5% |
| ARIMA(1,0,0) | 144.78 | +5.2% | 4.79 | +6.1% |
| 在线线性修正 | 143.21 | +6.2% | 4.71 | +7.6% |

## 优化方案

详见 [`timeseries_transformer_optimization.md`](timeseries_transformer_optimization.md)，包含 8 项进阶优化策略：

1. **注意力池化**：替代"取尾"，聚合全序列信息
2. **局部卷积增强**：Conv1d 残差块提取局部时域特征
3. **混合位置编码**：RoPE 旋转位置编码
4. **差分平滑损失**：约束预测序列时序连续性
5. **GELU 激活函数**：替代 ReLU
6. **计划采样**：缓解 exposure bias
7. **模型集成**：深度集成 + MC Dropout
8. **多尺度特征融合**：多层 Encoder 输出拼接

## 可视化输出

### 逐日预测图

训练完成后，`plots/` 目录下生成每日预测对比图（2025-10-01 ~ 2025-12-31）：

- 蓝色实线：真实值
- 红色虚线：预测值
- 灰色填充：误差区域

### 评估图表

- `loss_curve.png`：训练/测试损失曲线
- `test_error_distribution.png`：误差分布直方图
- `Test_best_cases.png` / `Test_worst_cases.png`：最优/最差预测案例
- `eval_random_window.png`：逐日预测对比总图

## 常见问题

### Q: 如何使用自己的数据？

1. 准备 CSV 文件，包含 `时间` 和 `出厂水流量` 两列
2. 修改 `train_transformer_autoregressive.py` 中的 `BASE_CONFIG["file_path"]`
3. 运行训练脚本

### Q: 如何调整预测窗口？

修改 `lookback_days` 和 `predict_days` 参数：

```bash
python train_transformer_autoregressive.py --lookback 14  # 回看 14 天
```

### Q: 训练太慢怎么办？

1. 减小 `batch_size`（如 8）
2. 增大 `stride`（如 2 或 4）
3. 使用 GPU（自动检测 CUDA）
4. 启用 AMP 混合精度训练（默认开启）

### Q: 如何复现最佳结果？

```bash
cd transformer_pkg
python train_transformer_autoregressive.py --model itransformer --label best_model
```

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{junshan_flow_prediction,
  title={Junshan Water Plant Flow Prediction with Transformer},
  author={Junshan Project Team},
  year={2025},
  url={https://github.com/your-repo/junshan-project}
}
```

## 许可证

本项目仅供学术研究与内部使用。

## 联系方式

如有问题或建议，请通过 GitHub Issues 反馈。

---

**最后更新**：2026-08-27

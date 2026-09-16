# ensemble_pkg — 军山水厂流量融合预测系统

## 1. 系统概述

军山水厂出厂水流量融合预测系统，将两个独立的时序预测模型——**Transformer（有监督）** 和 **TimesFM 2.5（零样本基础模型）**——通过加权平均融合，得到比任何单一模型更好的预测结果。

**核心融合公式：**

$$\hat{y} = \alpha \cdot \hat{y}_{\text{Transformer}} + (1-\alpha) \cdot \hat{y}_{\text{TimesFM}}$$

**项目特点：**

- 两个模型输出分辨率相同：都是 1h × 24 点，无需升采样对齐
- Transformer 输入：184 步回看（7天+16h）→ 自回归 rollout → 24 点
- TimesFM 输入：同一条清洗管线的 504h 小时序列 → forecast → 24 点
- 口径统一：两个模型都看 DataProcessor 清洗后的 Total_Flow

---

## 2. 目录结构

```
D:\Junshan_Project\
├── transformer_pkg/              # Transformer 模型训练 + 推理
├── ensemble_pkg/                 # ★ 本融合包
│   ├── __init__.py               # 空文件，使目录成为 Python 包
│   ├── _paths.py                 # 路径发现模块
│   ├── ensemble_predictor.py     # 核心融合预测器
│   ├── fit_weights.py            # 融合权重训练 + 区间校准
│   ├── predict_ensemble.py       # 融合预测入口脚本
│   ├── plot_daily.py             # 逐日对比图
│   ├── eval_production.py        # 生产口径端到端验证
│   ├── finetune_timesfm_lora.py  # TimesFM LoRA 微调
│   ├── test_interval.py          # 概率预测单元测试
│   ├── weights.json              # 融合权重（运行 fit_weights.py 生成）
│   ├── results/
│   │   ├── backtest_predictions.csv
│   │   ├── ensemble_metrics.csv
│   │   ├── quantile_calibration.json
│   │   ├── interval_metrics.csv
│   │   ├── weight_curve.png
│   │   ├── ensemble_compare.png
│   │   ├── interval_coverage.png
│   │   ├── interval_fan.png
│   │   └── daily/                # 逐日对比图
│   └── timesfm_lora/             # LoRA 微调产物
│       ├── lora_weights.pth
│       ├── lora_config.json
│       ├── train_history.csv
│       └── loss_curve.png
└── data/
    └── 水厂2025年小时级汇总.csv
```

---

## 3. 各模块详解

### 3.1 `_paths.py` — 路径发现模块

自动定位同级目录下的 `transformer_pkg`，并将它加入 `sys.path`，使融合包能导入 Transformer 模型的训练代码（`data_processing.py`, `transformer_model.py` 等）。

检查候选目录是否包含 5 个必需文件，避免硬编码路径。

### 3.2 `ensemble_predictor.py` — 核心融合预测器

定义 `JunshanEnsemblePredictor` 类，封装双模型推理 + 加权融合 + 概率区间预测的完整流程。

#### 3.2.1 点预测融合

$$\hat{y} = \alpha \cdot \hat{y}_T + (1-\alpha) \cdot \hat{y}_F$$

#### 3.2.2 概率区间融合

**Vincentization（逐分位数加权平均）：**

$$q_E(\tau) = \alpha \cdot q_T(\tau) + (1-\alpha) \cdot q_F(\tau)$$

直接对分位数做加权平均，自动保持单调性。

**Mixture（线性池）：**

$$F_E(x) = \alpha \cdot F_T(x) + (1-\alpha) \cdot F_F(x)$$

先对两条 CDF 做加权叠加，再在分位数水平上数值求逆。区间通常更宽，因为额外计入了"不知道哪个模型更准"的不确定性。

#### 3.2.3 `anchor_median` 锚定中位数

由于 mixture 的中位数不等于两个中位数的加权平均，需要把融合区间的中位数列钉回点预测值，同时分别恢复左右两侧的单调性。

#### 3.2.4 Transformer 分位数校准

Transformer 是 MSE 训练的单点模型，没有分位数头。它的区间来自回测残差校准：

$$q_{T}(\tau, \text{hour}) = \hat{y}_T + \hat{g}(\tau, \text{hour})$$

其中 $\hat{g}$ 是逐小时经验残差分位数，经中位数中心化处理，保证 $q(0.5) = \hat{y}_T$。

#### 3.2.5 TimesFM 分位数

TimesFM 2.5 在一次 forward 中同时输出点预测和 9 个分位数（q10~q90），直接从 `full_predictions[:, :, 1:]` 取出。

#### 3.2.6 `timesfm_day_offset` — 时间对齐修正

生产口径是"16点决策"，上下文停在 15:00，此时 TimesFM 第 0 步对应前一天 16:00，目标天 00:00 需要往后数 8 步。

### 3.3 `fit_weights.py` — 融合权重训练

用样本外回测拟合最优 α，并生成区间校准表。

#### 3.3.1 回测流程

对每一天：

1. 截取数据到决策时刻（前一天 15:00）
2. Transformer 自回归推理 → 24 点预测
3. TimesFM 零样本推理 → 24 点预测
4. TimesFM 异常值裁剪（0.3~3.0 倍中位数）
5. 记录真值和两个预测值

#### 3.3.2 α 拟合

**网格搜索：** 在 [0, 1] 上以步长 0.005 遍历所有候选 α，对每天的回测数据计算 MAE/RMSE/MAPE，选使指标最小的 α。

向量化计算技巧：

$$\text{error}(\alpha) = \alpha \cdot (p_T - p_F) + (p_F - y) = \alpha \cdot D + C$$

用外积一次性算出所有 α 下的误差矩阵。

**MSE 闭式解（参考值）：**

$$\alpha^* = \frac{\sum (y - p_F)(p_T - p_F)}{\sum (p_T - p_F)^2}$$

#### 3.3.3 误差相关系数

$$\rho = \text{corr}(e_T, e_F)$$

两模型误差相关性越低，融合收益越大。

#### 3.3.4 逐小时残差分位数校准

$$\hat{g}(\tau, h) = \text{quantile}_\tau(\text{resid}[h]) - \text{quantile}_{0.5}(\text{resid}[h])$$

中位数中心化确保点预测不被改动。桶内样本不足时回退全局池化分位数。用循环 3 点均值降噪。

#### 3.3.5 区间质量指标

| 指标 | 说明 |
|------|------|
| **PICP** | Prediction Interval Coverage Probability，经验覆盖率 |
| **MPIW** | Mean Prediction Interval Width，平均区间宽度 |
| **Winkler** | 区间得分（主指标），同时惩罚覆盖不足和区间过宽 |
| **CRPS** | Continuous Ranked Probability Score，通过 pinball 损失估计 |

**Winkler 区间得分：**

$$IS = (u-l) + \frac{2}{\alpha_{mis}}(l-y)\mathbf{1}_{y<l} + \frac{2}{\alpha_{mis}}(y-u)\mathbf{1}_{y>u}$$

其中 $\alpha_{mis} = 1 - \text{名义覆盖率}$（80% 区间 → $\alpha_{mis} = 0.20$，惩罚系数 $2/0.20 = 10$）。

### 3.4 `predict_ensemble.py` — 融合预测入口脚本

命令行入口，提供 `predict_ensemble()` 库函数和 CLI 接口。读 CSV → 调用融合预测器 → 输出 JSON → 对比真实值画图。

### 3.5 `plot_daily.py` — 逐日对比图

从 `backtest_predictions.csv` 读取回测结果，每天画一张图，显示真实值、Transformer、TimesFM、融合四条曲线 + 可选 80% 区间带。

### 3.6 `eval_production.py` — 生产口径端到端验证

把 `data/input_nextday16h_*.csv`（生产环境真实输入）逐个喂给 `predict()`，评估 MAE、MAPE、80%覆盖率、80%宽度、中位数偏差不变量。只读不写。

### 3.7 `finetune_timesfm_lora.py` — TimesFM LoRA 微调

用 LoRA（Low-Rank Adaptation）微调 TimesFM 2.5 的注意力层，使其更好地适配水厂流量数据。

#### LoRA 原理

在冻结的权重矩阵 $W_0$ 旁边插入低秩分解 $A \cdot B$：

$$W = W_0 + \Delta W = W_0 + B \cdot A \cdot \frac{\alpha}{r}$$

其中 $A \in \mathbb{R}^{d \times r}$，$B \in \mathbb{R}^{r \times d}$，$r \ll d$（默认 $r=8$）。只训练 $A, B$，冻结原始权重。

**目标层：** 每层 Transformer 的 `q_proj`, `k_proj`, `v_proj`, `o_proj`。

**训练策略：** MSE loss（z-score 归一化空间）、AdamW + CosineAnnealing、早停、时间序列前后切分（杜绝数据泄露）。

### 3.8 `test_interval.py` — 概率预测单元测试

不需要加载模型的单元测试，秒级完成。验证：

1. 融合方法的形状保持和单调性
2. anchor_median 同时保住单调性与中位数
3. 相同分布融合不变性
4. enforce_monotone 正确性
5. 残差校准的中位数中心化和回退
6. 区间指标的边界情况
7. 回测 CSV 列契约
8. TimesFM 时间偏移计算

---

## 4. 概率区间预测详解

### 4.1 两个模型各自的分位数来源

#### Transformer — 残差校准（事后统计）

Transformer 是用 MSE 训练的纯点预测模型，本身没有分位数头。它的区间来自回测残差的经验分位数。

**本质：** 不是"预测"出来的，而是用历史残差分布标定出来的。

#### TimesFM — 原生输出（模型自带）

TimesFM 2.5 的一次 forward 直接输出 `full_predictions`，形状 `[batch, horizon, 10]`：

```
索引  0      1     2     3     4     5     6     7     8     9
含义  point  q0.1  q0.2  q0.3  q0.4  q0.5  q0.6  q0.7  q0.8  q0.9
```

取 `[:, :, 1:]` 就得到 9 个分位数，零额外推理成本。

### 4.2 融合发生在哪一层

两个模型的分位数也被融合了：

```
Transformer 点预测 ──┐
                     ├──→ α·ŷ_T + (1-α)·ŷ_F ──→ 融合点预测
TimesFM 点预测 ──────┘

Transformer 残差分位数 ──┐
                         ├──→ vincentization/mixture ──→ 融合区间
TimesFM 原生分位数 ──────┘
```

融合后的区间再经过 `anchor_median` 锚定，确保中位数精确等于融合点预测。

### 4.3 TimesFM 2.5 分位数的内部实现

TimesFM 是 Google 的 200M 参数时序基础模型，基于 PatchTST 架构（Transformer encoder-only）：

1. **输入分 patch**：输入时序被切成长度为 32 的 patch，每个 patch 经过线性投影映射到 Transformer 的 hidden dim
2. **Transformer encoder 处理**：所有 patch 经过多层 Transformer encoder，捕获时序依赖关系
3. **双头输出**：
   - **Point head**：输出单一值，用于 MSE/MAE 训练
   - **Quantile head**：输出 10 个值，对应 10 个分位数水平
4. **分位数训练**：使用 Pinball Loss 训练 quantile head：

$$L_q = \frac{1}{L} \sum_{j=1}^{L} \frac{1}{T} \sum_{t=1}^{T} \rho_{\tau_j}(y_t - \hat{q}_t(\tau_j))$$

其中 Pinball Loss：

$$\rho_\tau(e) = \begin{cases} \tau \cdot e & \text{if } e \geq 0 \\ (\tau - 1) \cdot e & \text{if } e < 0 \end{cases}$$

### 4.4 降级情况

如果 Transformer 的校准表缺失（`q_t is None`），则只用 TimesFM 的不确定性形状，平移到融合点预测上：

```python
q_e = y[:, None] + (q_f - pred_f[:, None])
```

相当于只保留 TimesFM 区间的"形状"（分位数间距），但中心点用融合点预测替换。

### 4.5 两种融合方法对比

| 特征 | Vincentization | Mixture |
|------|----------------|---------|
| 公式 | $q_E(\tau) = \alpha q_T(\tau) + (1-\alpha) q_F(\tau)$ | $F_E = \alpha F_T + (1-\alpha) F_F$，数值求逆 |
| 单调性 | 自动保持 | 需要 anchor_median 修正 |
| 中位数 | 精确等于点预测 | 偏离点预测，需要锚定 |
| 区间宽度 | 较窄 | 较宽（额外计入模型选择不确定性） |

---

## 5. 数学原理总结

| 概念 | 公式/方法 |
|------|----------|
| 点预测融合 | $\hat{y} = \alpha \cdot \hat{y}_T + (1-\alpha) \cdot \hat{y}_F$ |
| α 拟合 | 网格搜索最小化 MAE/RMSE/MAPE |
| MSE 闭式解 | $\alpha^* = \frac{\sum (y - p_F)(p_T - p_F)}{\sum (p_T - p_F)^2}$ |
| Vincentization | $q_E(\tau) = \alpha \cdot q_T(\tau) + (1-\alpha) \cdot q_F(\tau)$ |
| Mixture（线性池） | $F_E = \alpha \cdot F_T + (1-\alpha) \cdot F_F$，数值求逆 |
| Transformer 区间 | 残差校准：$q = \hat{y} + \hat{g}(\tau, h)$，中位数中心化 |
| TimesFM 区间 | 原生分位数头输出 q10~q90，Pinball Loss 训练 |
| LoRA 微调 | $W = W_0 + BA \cdot (\alpha/r)$，冻结 $W_0$，只训 $A,B$ |
| 区间评估 | PICP、MPIW、Winkler score、CRPS |
| 误差相关性 | $\rho = \text{corr}(e_T, e_F)$，越低融合收益越大 |
| 中位数锚定 | `anchor_median` 确保区间中位数 ≡ 点预测 |

---

## 6. 使用方法

### 6.1 训练融合权重

```bash
cd D:\Junshan_Project\ensemble_pkg
python fit_weights.py
python fit_weights.py --fit_days 12 --valid_days 12 --metric mae
```

**产物：** `weights.json` / `results/backtest_predictions.csv` / `results/ensemble_metrics.csv`

### 6.2 运行融合预测

```bash
python predict_ensemble.py
python predict_ensemble.py --alpha 0.6
python predict_ensemble.py --data data/input_nextday16h_20251225_35d.csv
python predict_ensemble.py --raw data/水厂2025年小时级汇总.csv --days 35
```

**库调用：**

```python
from predict_ensemble import predict_ensemble
result = predict_ensemble("data/input_nextday16h_20251225_35d.csv")
```

### 6.3 生成逐日对比图

```bash
python plot_daily.py
python plot_daily.py --per_day_y
python plot_daily.py --alpha 0.6
```

### 6.4 生产口径验证

```bash
python eval_production.py
```

### 6.5 TimesFM LoRA 微调

```bash
python finetune_timesfm_lora.py
python finetune_timesfm_lora.py --epochs 20 --lora_rank 16 --lr 3e-4
```

### 6.6 运行单元测试

```bash
python test_interval.py
```

---

## 7. 接口格式

融合预测返回的 JSON 格式：

```json
{
  "date": "2025-12-26",
  "provider": "junshan_ensemble",
  "unit": "m3/h",
  "interval_minutes": 60,
  "horizon": 24,
  "values": [...],
  "quantiles": {
    "levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "values": [[...], [...], ..., [...]]
  },
  "interval": {
    "coverage": 0.8,
    "lower": [...],
    "upper": [...]
  },
  "uncertainty": {
    "method": "vincentization",
    "median_gap": 0.0,
    "components": {
      "transformer": "residual_calibration",
      "timesfm": "native"
    }
  }
}
```

# 时间序列预测中的残差修正方法

> 适用场景：一年长度的日频时间序列数据（约 365 个样本点）。针对这类数据，直接用马尔科夫链做残差修正效果有限（样本量太小，转移矩阵估计不稳定），本文给出几种更实用的残差修正策略，按推荐程度排序。

---

## 核心思路

设第 $t$ 天的真实值为 $y_t$，模型预测值为 $\hat{y}_t$，则残差定义为：

$$r_t = y_t - \hat{y}_t$$

残差修正的本质：利用历史残差中蕴含的信息（如近期偏差的惯性、周期性、自相关结构），估计一个修正量 $\Delta_t$，对原始预测进行加性修正：

$$\hat{y}_t^{\text{修正}} = \hat{y}_t + \Delta_t$$

---

## 方案一：滑动窗口残差修正（推荐）

最简单有效的方法，不需要额外建模，直接利用近期残差的平均值来修正当前预测。

### 数学表达

设窗口大小为 $w$（例如取最近 7 天或 14 天），第 $t$ 天的修正预测值为：

$$\hat{y}_t^{\text{修正}} = \hat{y}_t + \frac{1}{w} \sum_{i=1}^{w} (y_{t-i} - \hat{y}_{t-i})$$

即用最近 $w$ 天残差的均值作为当前预测的偏差修正量。

### Python 示例

```python
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# 模拟一年日频数据
np.random.seed(42)
dates = pd.date_range('2025-01-01', periods=365, freq='D')
trend = np.linspace(100, 150, 365)
seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 30)
noise = np.random.normal(0, 5, 365)
y_true = trend + seasonality + noise

# 用简单模型做基准预测（这里用滞后1天作为朴素预测）
y_pred_base = np.roll(y_true, 1)  # 昨天预测今天
y_pred_base[0] = y_true[0]

# 滑动窗口残差修正
def sliding_window_correction(y_true, y_pred, window=7):
    residuals = y_true - y_pred
    corrected = y_pred.copy()
    for t in range(window, len(y_pred)):
        mean_residual = np.mean(residuals[t-window:t])
        corrected[t] = y_pred[t] + mean_residual
    return corrected

window = 14
y_pred_corrected = sliding_window_correction(y_true, y_pred_base, window)

mae_before = mean_absolute_error(y_true[window:], y_pred_base[window:])
mae_after = mean_absolute_error(y_true[window:], y_pred_corrected[window:])

print(f"修正前 MAE: {mae_before:.3f}")
print(f"修正后 MAE: {mae_after:.3f}")
```

- **优点**：简单、可解释、对周期性残差有效
- **缺点**：对突变不敏感，窗口大小需要调参

---

## 方案二：ARIMA 残差建模

如果残差有明显的自相关结构（而非纯白噪声），可以用 ARIMA 对残差序列单独建模，然后将 ARIMA 的预测值作为修正量。

### 步骤

1. 用 XGBoost/LightGBM 得到初始预测值和残差序列
2. 检查残差的 ACF（自相关函数）和 PACF（偏自相关函数）图
3. 如果残差存在显著自相关，对残差拟合 ARIMA(p,d,q) 模型
4. 用 ARIMA 预测下一期的残差，加到原始预测值上

### Python 示例

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# 假设已有 y_true 和 y_pred
residuals = y_true - y_pred_base

# 检查残差自相关
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals, lags=30, ax=axes[0])
plot_pacf(residuals, lags=30, ax=axes[1])
plt.show()

# 如果 ACF 拖尾、PACF 截尾，尝试 AR(1) 或 ARIMA(1,0,0)
arima_model = ARIMA(residuals, order=(1, 0, 0))  # p=1, d=0, q=0
arima_fit = arima_model.fit()

# 预测下一期残差
residual_forecast = arima_fit.forecast(steps=1)[0]

# 修正预测
y_pred_final = y_pred_base[-1] + residual_forecast
```

- **优点**：能捕捉复杂的残差动态结构
- **缺点**：需要残差满足平稳性假设，一年数据勉强够用

---

## 方案三：季节性分解 + 残差修正

一年数据通常包含明显的周/月季节性。先用 STL（Seasonal-Trend decomposition using LOESS）将序列分解为趋势、季节、残差三部分，然后对残差部分做修正。

### 数学表达

$$y_t = T_t + S_t + R_t$$

其中 $T_t$ 是趋势，$S_t$ 是季节成分，$R_t$ 是剩余残差。若 XGBoost 已学到趋势和季节，则 $R_t$ 应接近白噪声；若不是，说明模型漏掉了某些模式，可对 $R_t$ 进一步建模。

### Python 示例

```python
from statsmodels.tsa.seasonal import STL

# 对真实值做 STL 分解
stl = STL(y_true, period=7)  # 周周期
result = stl.fit()

trend = result.trend
seasonal = result.seasonal
residual = result.resid

# 检查 XGBoost 预测是否捕捉到了季节成分
# 如果 XGBoost 没学到季节，可以对 seasonal 做预测并叠加
```

- **优点**：直观，能分离不同时间尺度的影响
- **缺点**：STL 需要至少两个完整周期；一年数据对周周期（52 周）够用，对月周期（12 个月）较边缘

---

## 方案四：指数加权移动平均（EWMA）修正

给近期残差更高的权重，适合残差缓慢漂移的场景。

### 数学表达

$$\hat{r}_t = \alpha \cdot r_{t-1} + (1-\alpha) \cdot \hat{r}_{t-1}$$

其中 $\alpha \in (0,1)$ 是衰减因子，通常取 0.1–0.3。修正后的预测：

$$\hat{y}_t^{\text{修正}} = \hat{y}_t + \hat{r}_t$$

### Python 示例

```python
def ewma_correction(y_true, y_pred, alpha=0.2):
    residuals = y_true - y_pred
    smoothed_residual = np.zeros_like(residuals)
    smoothed_residual[0] = residuals[0]
    
    for t in range(1, len(residuals)):
        smoothed_residual[t] = alpha * residuals[t-1] + (1-alpha) * smoothed_residual[t-1]
    
    return y_pred + smoothed_residual

y_pred_ewma = ewma_correction(y_true, y_pred_base, alpha=0.15)
```

- **优点**：参数少、计算快、自适应
- **缺点**：对突变反应慢

---

## 综合建议：针对一年数据的实操流程

1. **先做基准预测**：用 XGBoost/LightGBM，加入时间特征（星期几、月份、节假日、滞后特征）
2. **检查残差自相关**：画 ACF/PACF 图，运行 Durbin-Watson 检验
3. **若残差为白噪声**（DW 检验 p 值 > 0.05）：模型已充分捕捉时序结构，**不建议再修正**
4. **若残差存在短期自相关**（滞后 1–7 天显著）：优先用**滑动窗口修正**（窗口 = 7 或 14）
5. **若残差存在长期自相关**（滞后 > 7 天显著）：尝试 **ARIMA 残差建模**
6. **若数据有明显周/月季节模式且模型没学好**：用 **STL 分解**提取季节成分，单独建模

> 对于一年日频数据，**滑动窗口修正（窗口 = 7）**通常是性价比最高的选择——简单、稳定、不容易过拟合。

---

## 附录：为什么马尔科夫链不适合直接做残差修正

| 维度 | 马尔科夫链 | 梯度提升（残差修正） |
|---|---|---|
| 核心目的 | 描述状态间转移概率 | 逐步逼近目标函数最优解 |
| 数据类型 | 序列/时序数据 | 独立同分布或结构化数据 |
| 修正对象 | 无"修正"概念，只有状态演化 | 每轮拟合上一轮残差 |
| 数学基础 | 概率图模型、随机过程 | 梯度下降、加法模型 |

- 残差不是"状态"，马尔科夫链建模的是状态间概率关系，而非误差的缩小过程
- 缺乏迭代机制：梯度提升依赖多轮迭代、逐轮依赖前轮结果；马尔科夫链是一步转移，无"逐步逼近"概念
- 目标不同：残差修正最小化损失函数；马尔科夫链最大化序列似然或匹配转移模式

**结论**：马尔科夫链不能替代梯度提升做残差修正；仅在残差存在时序依赖或状态切换模式时，可作为后处理工具辅助建模残差结构。对于一年日频数据，样本量太小，马尔科夫链状态转移矩阵估计不稳定，更推荐上述四种基于统计的修正方法。

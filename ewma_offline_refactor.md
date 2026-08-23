# 离线 EWMA 残差修正 — 脚本改造需求（AI 代码助手用）

> 以下是一段面向 AI 代码助手（Claude / GPT 等）的完整改造需求说明。可直接复制发给 AI，由其据此生成/修改代码。

---

## 背景

现有评估脚本加载训练好的 Transformer / iTransformer 模型，对时间序列数据做逐日预测，并使用 EWMA 对残差进行**在线修正**（预测时实时更新 EWMA 状态），输出每天及总体的 MAE / RMSE / MAPE 对比指标。

**改造目标**：将该评估脚本改造为一个**纯离线的批处理脚本**，采用"离线后处理 + 手动运行 + CSV 文件持久化"模式，**不依赖定时任务、数据库或常驻服务**。

---

## 输入文件

| 文件 | 说明 | 字段 |
|---|---|---|
| `ewma_state.csv` | 每个站点（预测单元）的 EWMA 状态；不存在时用默认值初始化（`alpha=0.3`、`ewma_val=0.0`） | `site_id`, `ewma_val`, `alpha` |
| `predictions.csv` | 主模型每天输出的预测结果 | `site_id`, `date`, `predicted_value` |
| `true_values.csv` | 历史真实值（可从原始数据 CSV 或单独文件读取） | `site_id`, `date`, `true_value` |

---

## 处理逻辑（每次手动运行执行一次）

1. 读取 `ewma_state.csv`，获取每个站点的当前 `ewma_val` 与 `alpha`。
2. 读取 `predictions.csv`，获取当天预测值。
3. 从历史真实值中取**昨天（或最近一个有真实值的日期）**的真实值，计算残差 `r = true_value - predicted_value`。
4. 更新 EWMA 状态：`ewma_val = alpha * r + (1 - alpha) * ewma_val`。
5. 修正当天预测：`corrected_value = predicted_value + ewma_val`。
6. 将修正结果**追加**写入 `corrected_predictions.csv`（不覆盖历史记录）。
7. 将更新后的 `ewma_val` **写回** `ewma_state.csv`（覆盖旧状态）。

---

## 输出

### 文件输出

`corrected_predictions.csv`（追加模式），字段：

`site_id`, `date`, `original_prediction`, `corrected_prediction`, `ewma_val_used`

### 控制台摘要（示例）

```
========================================
离线 EWMA 残差修正 - 手动运行
========================================
读取状态文件: ewma_state.csv (3 个站点)
读取预测文件: predictions.csv (2025-10-01, 3 条)
读取真实值文件: true_values.csv (2025-09-30 有 3 条真实值)

处理站点 site_001: 残差=+2.34, ewma_val 从 0.50 -> 1.05, 修正后预测=102.05
处理站点 site_002: 残差=-1.20, ewma_val 从 0.30 -> -0.06, 修正后预测=98.94
处理站点 site_003: 残差=+0.50, ewma_val 从 0.00 -> 0.35, 修正后预测=100.85

修正摘要:
  处理站点数: 3
  平均修正量: +0.65
  修正前 MAE: 1.82
  修正后 MAE: 1.21
  提升: +33.5%

状态已保存: ewma_state.csv
修正结果已保存: corrected_predictions.csv
```

---

## 工程要求

- **移除**所有与模型加载、推理、绘图相关的代码（不再加载 PyTorch 模型）。
- **移除**所有与定时任务、数据库相关的依赖。
- 使用 `pandas` 读写 CSV，保持代码简洁。
- **异常值裁剪（clipping）**：若残差绝对值超过 3 倍历史残差标准差，截断到该阈值。
- **冷启动处理**：首次运行若无历史残差，跳过修正（仅初始化状态）。
- 添加详细日志打印，方便手动运行时查看进度。

---

## 命令行参数

| 参数 | 必填 | 说明 | 默认值 |
|---|---|---|---|
| `--predictions` | 是 | 预测结果 CSV 路径 | — |
| `--true_values` | 是 | 真实值 CSV 路径 | — |
| `--state` | 否 | EWMA 状态 CSV 路径 | `ewma_state.csv` |
| `--output` | 否 | 修正结果输出路径 | `corrected_predictions.csv` |
| `--alpha` | 否 | EWMA 平滑因子（仅首次初始化时使用） | `0.3` |
| `--date` | 否 | 要处理的日期 | 取 `predictions.csv` 中最新日期 |

---

## 交付要求

请 AI 代码助手据此输出**修改后的完整 Python 脚本代码**（单文件），可直接运行，满足上述全部输入/输出/工程/命令行要求，并包含必要的日志与异常处理。

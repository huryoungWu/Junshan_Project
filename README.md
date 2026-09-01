# 军山水厂流量预测 (简化版)

基于 Transformer 模型的出厂水流量预测，每天 16 点预测次日全天 24 小时流量。

## 快速开始

```bash
# 使用默认 CSV 文件预测
python junshan_inference.py

# 指定输入 CSV 文件
python junshan_inference.py --data path/to/your_data.csv

# 指定输出 JSON 文件
python junshan_inference.py --data path/to/your_data.csv --output pred.json
```

## 输入要求

### CSV 文件格式

CSV 文件必须包含以下两列：

| 列名 | 说明 | 示例 |
|------|------|------|
| `时间` | 时间戳，格式 `YYYY-MM-DD HH:MM:SS` | `2025-08-20 15:00:00` |
| `出厂水流量` | 流量值，单位 m³/h | `3843.86` |

**CSV 文件示例：**
```csv
时间,出厂水流量
2025-07-17 00:00:00,2262.91
2025-07-17 01:00:00,1823.1
2025-07-17 02:00:00,1832.14
2025-07-17 03:00:00,1857.82
...
2025-08-20 15:00:00,3843.86
```

### 数据时间范围

- **数据粒度**：小时级（每小时一条记录）
- **最少数据量**：约 16 天（363 小时）
  - 回看窗口：184 小时（7 天 + 16 小时）
  - 滚动特征预热：180 小时（30 天滚动窗口）
- **推荐数据量**：≥ 35 天（满窗 + 清洗余量）
- **截止时刻**：最后一条数据应为某天 15:00（标准 16 点截止）
  - 预测目标 = 截止日次日（0:00~23:00，共 24 小时）

**参考文件：**
- 输入示例：`data/input_nextday16h_20250820_35d.csv`
  - 时间范围：2025-07-17 00:00 ~ 2025-08-20 15:00
  - 数据量：832 条（约 34.7 天）
  - 预测目标：2025-08-21（次日全天）

## 输出格式

### JSON 输出结构

```json
{
  "date": "2025-08-21",
  "provider": "junshan_transformer_nextday16h",
  "unit": "m3/h",
  "interval_minutes": 60,
  "horizon": 24,
  "values": [2093.0, 1860.0, 1665.0, 1580.0, 1520.0, 1480.0, 1450.0, 1520.0, 1680.0, 1950.0, 2280.0, 2520.0, 2680.0, 2750.0, 2820.0, 2900.0, 3050.0, 3180.0, 3250.0, 3100.0, 2850.0, 2520.0, 2280.0, 2150.0]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | string | 预测目标日期，格式 `YYYY-MM-DD` |
| `provider` | string | 模型标识，如 `junshan_transformer_nextday16h` |
| `unit` | string | 流量单位，固定为 `m3/h`（立方米/小时） |
| `interval_minutes` | int | 预测间隔，固定为 `60`（每小时） |
| `horizon` | int | 预测时长，固定为 `24`（全天 24 小时） |
| `values` | array | 24 个浮点数，分别对应 0:00~23:00 的预测流量 |

### values 数组索引对应

| 索引 | 时刻 | 说明 |
|------|------|------|
| 0 | 00:00 | 凌晨 |
| 1 | 01:00 | |
| ... | ... | |
| 11 | 11:00 | 上午 |
| 12 | 12:00 | 中午 |
| ... | ... | |
| 15 | 15:00 | 下午（截止时刻） |
| ... | ... | |
| 23 | 23:00 | 晚上 |

## 命令行参数

```bash
python junshan_inference.py [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `data/input_nextday16h_20250820_35d.csv` | 输入 CSV 文件路径 |
| `--result_dir` | `results/junshan_L1D_P24H_1h_transformer_nextday16h_mc_20260901_142931` | 训练结果目录 |
| `--provider` | `junshan_{model_type}_nextday16h` | 接口中的 provider 字段 |
| `--output, -o` | 无（仅打印） | 输出 JSON 文件路径 |

## 使用示例

### 1. 基本预测

```bash
# 使用默认数据预测
python junshan_inference.py

# 输出示例：
# [1/4] 加载模型: results/junshan_L1D_P24H_1h_transformer_nextday16h_mc_20260901_142931
# [2/4] 读取 CSV: D:\Junshan_Project\data\input_nextday16h_20250820_35d.csv
# [3/4] 数据清洗 + 特征构建
# [4/4] 自回归推理
# 
# ============================================================
# 预测结果:
# {
#   "date": "2025-08-21",
#   "provider": "junshan_transformer_nextday16h",
#   "unit": "m3/h",
#   "interval_minutes": 60,
#   "horizon": 24,
#   "values": [2093.0, 1860.0, ...]
# }
```

### 2. 指定输入文件

```bash
python junshan_inference.py --data D:\Junshan_Project\data\my_data.csv
```

### 3. 保存结果到文件

```bash
python junshan_inference.py --output prediction_result.json
```

### 4. 指定其他模型

```bash
python junshan_inference.py --result_dir results/other_model --provider my_model
```

### 5. 库调用

```python
from junshan_inference import predict

# 构建绝对路径
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "data", "input_nextday16h_20250820_35d.csv")
RESULT_DIR = os.path.join(HERE, "results", "junshan_L1D_P24H_1h_transformer_nextday16h_mc_20260901_142931")

# 调用预测函数
result = predict(csv_path=CSV_PATH, result_dir=RESULT_DIR)

# 打印结果
print("\n" + "=" * 60)
print("预测结果:")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

## 数据处理流程

1. **读取 CSV**：解析时间列，设置为索引
2. **数据清洗**：
   - 突变流量插值修正（日内/跨日双判据）
   - Hampel 离群检测 + 物理界限裁剪（0~10000 m³/h）
   - 重采样（60min，已是整点则恒等）
3. **特征构建**：
   - 日历特征：hour/dow/month/doy sin/cos + 工作日/节假日（12 维）
   - 数据驱动特征：lag_24h, lag_168h（2 维）
4. **自回归推理**：
   - 取最后 184 小时作为回看窗口
   - 滚动预测：先滚缺口（24-H 小时），再滚目标天 24 小时
   - 每步回灌预测值，重算数据驱动特征
5. **输出结果**：反归一化，提取目标天 24 小时预测值

## 模型文件

推理需要以下文件（位于 `result_dir` 目录）：

| 文件 | 说明 |
|------|------|
| `scaler.pkl` | 训练配置、特征缩放器、目标缩放器、特征列 |
| `best_seq2seq_model.pth` | 训练好的模型权重 |

默认模型目录：
```
results/junshan_L1D_P24H_1h_transformer_nextday16h_mc_20260901_142931/
```

## 常见问题

### Q: 报错 "数据不足: 仅 XXX 行, 需要 184 步"

A: 输入数据量不够，至少需要约 16 天（363 小时）的数据。建议使用 35 天以上的数据。

### Q: 报错 "CSV 必须包含 时间 / timestamp 列"

A: CSV 文件必须有时间列，列名可以是 `时间` 或 `timestamp`。

### Q: 报错 "特征列缺失"

A: 输入数据格式不正确，确保 CSV 包含 `时间` 和 `出厂水流量` 两列。

### Q: 预测结果中 values 数组长度不是 24

A: 正常情况下应为 24 个值（0:00~23:00）。如果数据截止时刻不是 15:00，可能会有调整。

## 依赖库

```
numpy
pandas
torch
chinese-calendar (可选，用于节假日特征)
```

安装依赖：
```bash
pip install numpy pandas torch chinese-calendar
```

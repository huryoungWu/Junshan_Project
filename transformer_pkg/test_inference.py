# -*- coding: utf-8 -*-
"""test_inference.py — 调用 junshan_inference.py 的示例脚本"""

import os
import json
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

# 提取关键信息
print(f"\n预测日期: {result['date']}")
print(f"预测时刻: 0:00 ~ 23:00 (共 {result['horizon']} 小时)")
print(f"流量范围: {min(result['values']):.1f} ~ {max(result['values']):.1f} m³/h")

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
import torch

# 将 Time-MoE 仓库路径添加到 sys.path
import sys
TIME_MOE_PATH = r"D:\Junshan_Project\Time-MoE"
if TIME_MOE_PATH not in sys.path:
    sys.path.insert(0, TIME_MOE_PATH)

# 动态导入 TimeMoeForPrediction
try:
    import importlib
    modeling_module = importlib.import_module("time_moe.models.modeling_time_moe")
    TimeMoeForPrediction = modeling_module.TimeMoeForPrediction
    print("✅ 成功导入 Time-MoE 仓库的 TimeMoeForPrediction")
except Exception as e:
    print(f"⚠ 导入失败: {e}")
    print("  使用 transformers 的 AutoModelForCausalLM 替代")
    from transformers import AutoModelForCausalLM

    class TimeMoeForPrediction:
        def __init__(self, model):
            self.model = model

        @classmethod
        def from_pretrained(cls, model_path, device_map="cpu", torch_dtype=torch.float32,
                           attn_implementation="eager", **kwargs):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
            return cls(model)

        def eval(self):
            self.model.eval()
            return self

        def parameters(self):
            return self.model.parameters()

        def generate(self, inputs, max_new_tokens, **kwargs):
            return self.model.generate(inputs, max_new_tokens=max_new_tokens, **kwargs)
# 尝试引入中国节假日库
try:
    from chinese_calendar import is_holiday, is_workday as _cal_is_workday
    _HAS_CHINESE_CALENDAR = True
except ImportError:
    _HAS_CHINESE_CALENDAR = False
    print("⚠ chinese-calendar 未安装, 节假日特征退化为简单周末判断")

# ── 1. 读取原始数据 ──
df = pd.read_csv(
    r"D:\Junshan_Project\data\水厂2025年小时级汇总.csv",
    encoding="utf-8-sig", parse_dates=["时间"], index_col="时间"
)

# 目标序列（出厂水流量）
y = df["出厂水流量"].values.astype(np.float32)

# 填充 NaN (线性插值)
y_series = pd.Series(y)
y = y_series.interpolate(method="linear").bfill().ffill().values.astype(np.float32)

# ── 2. 生成时间外生变量 ──
def add_calendar_features(df_index):
    """由 DatetimeIndex 确定性生成日历特征"""
    h = df_index.hour.to_numpy()
    dow = df_index.dayofweek.to_numpy()
    month = df_index.month.to_numpy()
    doy = df_index.dayofyear.to_numpy()

    features = {
        "hour_sin": np.sin(2.0 * np.pi * h / 24.0).astype(np.float32),
        "hour_cos": np.cos(2.0 * np.pi * h / 24.0).astype(np.float32),
        "dow_sin": np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32),
        "dow_cos": np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32),
        "month_sin": np.sin(2.0 * np.pi * month / 12.0).astype(np.float32),
        "month_cos": np.cos(2.0 * np.pi * month / 12.0).astype(np.float32),
        "doy_sin": np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32),
        "doy_cos": np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32),
    }

    # 工作日/节假日特征
    dates = df_index.normalize()
    unique_dates = dates.unique()

    if _HAS_CHINESE_CALENDAR:
        workday_map = {d: float(_cal_is_workday(d.to_pydatetime().date()))
                       for d in unique_dates}
        holiday_map = {}
        for d in unique_dates:
            dt = d.to_pydatetime().date()
            is_special = (not _cal_is_workday(dt)) and d.dayofweek < 5
            holiday_map[d] = float(is_special)

        eve_map = {}
        next_map = {}
        for d in unique_dates:
            prev_d = d - pd.Timedelta(days=1)
            next_d = d + pd.Timedelta(days=1)
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

        features["is_workday"] = dates.map(workday_map).astype(np.float32)
        features["is_holiday"] = dates.map(holiday_map).astype(np.float32)
        features["holiday_eve"] = dates.map(eve_map).astype(np.float32)
        features["holiday_next"] = dates.map(next_map).astype(np.float32)
    else:
        is_weekend = (dow >= 5).astype(np.float32)
        features["is_workday"] = 1.0 - is_weekend
        features["is_holiday"] = np.float32(0.0)
        features["holiday_eve"] = np.float32(0.0)
        features["holiday_next"] = np.float32(0.0)

    return features

# 生成外生变量字典
covariates = add_calendar_features(df.index)
covariate_names = list(covariates.keys())
covariate_matrix = np.stack([covariates[k] for k in covariate_names], axis=-1)  # (n, n_cov)

print(f"✅ 生成 {len(covariate_names)} 维协变量: {covariate_names}")

# ── 3. 划分训练/测试 ──
test_days = 30
split_point = df.index[-1] - pd.Timedelta(days=test_days)
split_idx = df.index.searchsorted(split_point)

y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"训练集: {len(y_train)} 样本, 测试集: {len(y_test)} 样本")

# ── 4. 加载 Time-MoE 模型 ──
CONTEXT_LEN = 336   # 回看 14 天 (336 小时)
HORIZON = 24         # 每次预测 1 步 = 1 小时

# 从 HuggingFace Hub 下载模型（首次运行会自动下载）
MODEL_PATH = r"D:\Junshan_Project\models\time-moe-200m"

print(f"⏳ 加载 Time-MoE 模型从: {MODEL_PATH}")
print("   (首次运行会自动下载模型，请耐心等待)")

# ✅ 使用 TimeMoeForPrediction 加载，指定 attn_implementation="eager" 避免 flash_attn
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TimeMoeForPrediction.from_pretrained(
    MODEL_PATH,
    device_map=device,
    torch_dtype=torch.float32,
    attn_implementation="eager",  # 使用标准 attention，不需要 flash_attn
)

model.eval()
print(f"✅ 模型已加载，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"✅ 使用设备: {device}")

# ── 5. 逐段预测 (每段用真实历史，不用预测值) ──
all_pred = []
n_train = len(y_train)
n_steps = len(y_test) // HORIZON  # 按 HORIZON 切分

print(f"\n🚀 开始逐时滚动预测，共 {n_steps} 次迭代")
print("=" * 60)

for step_idx in range(n_steps):
    # 截取到当前段之前的真实历史
    hist_end = n_train + step_idx * HORIZON
    
    # Time-MoE 输入: [1, context_len] 的 numpy 数组
    # 注意: 只取最近的 CONTEXT_LEN 个点 (Time-MoE 支持任意长度)
    history_start = max(0, hist_end - CONTEXT_LEN)
    history = y[history_start:hist_end].reshape(1, -1)  # (1, context_len)
    
    # 构造协变量 (静态嵌入, 只取对应历史段的)
    # 将 12 维协变量拼接到序列特征后
    # 注意: Time-MoE 不支持外生变量, 这里简单拼接后仍作为单变量输入
    cov = covariate_matrix[history_start:hist_end]  # (context_len, n_cov)
    
    # 预测 HORIZON 步
    try:
        with torch.no_grad():
            inputs = torch.from_numpy(history).float().to(device)

            # ✅ Time-MoE 要求归一化输入
            mean = inputs.mean(dim=-1, keepdim=True)
            std = inputs.std(dim=-1, keepdim=True) + 1e-8
            normed_inputs = (inputs - mean) / std

            # ✅ 使用 forward 方法预测（避免 generate 的兼容性问题）
            out = model(input_ids=normed_inputs)
            # logits shape: [1, seq_len, 1]，取最后一个位置作为下一步预测
            normed_pred = out.logits[:, -HORIZON:, :]

            # ✅ 反归一化
            pred = normed_pred * std + mean

            pred_np = pred.squeeze().detach().cpu().numpy().flatten()
            all_pred.append(pred_np)

            if (step_idx + 1) % 100 == 0:
                print(f"  已完成 {step_idx+1}/{n_steps} 步预测")
    except Exception as e:
        print(f"  预测出错 (step {step_idx}): {e}")
        # 如果出错, 用最后值填充
        all_pred.append(np.full(HORIZON, history[-1, -1]))
        break

print("-" * 60)

y_pred = np.concatenate(all_pred)
y_true = y_test[:len(y_pred)]

# ── 6. 评估 MAPE ──
overall_mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
print(f"\n📊 Overall MAPE: {overall_mape:.2f}%")

# 按天统计 MAPE
test_index = df.index[split_idx:split_idx + len(y_true)]
df_eval = pd.DataFrame({
    "date": test_index.normalize(),
    "y_true": y_true,
    "y_pred": y_pred,
})

print(f"\n{'日期':<14} {'MAPE%':>8}  {'样本数':>6}")
print("-" * 32)

daily_mapes = []
for date, grp in df_eval.groupby("date"):
    t = grp["y_true"].values
    p = grp["y_pred"].values
    mape_d = np.mean(np.abs((t - p) / (t + 1e-8))) * 100
    daily_mapes.append(mape_d)
    print(f"{str(date.date()):<14} {mape_d:>8.2f}  {len(grp):>6}")

avg_daily_mape = np.mean(daily_mapes)
print("-" * 32)
print(f"{'平均每天 MAPE':<14} {avg_daily_mape:>8.2f}%")
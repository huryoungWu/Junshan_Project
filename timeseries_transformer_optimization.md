# 时间序列 Transformer 模型进阶优化方案（②~⑨）

> 本文档基于对 `TimeSeriesTransformer` 模块的缺陷分析，系统梳理 8 项增强策略（编号 ②~⑨），覆盖输出聚合、局部卷积、位置编码、损失函数、激活函数、训练策略、集成学习与多尺度特征融合。内容按"问题 → 方案 → 实现 → 预期收益"统一结构组织，并给出可复现的 PyTorch 代码片段，便于直接集成到现有训练流程。

## 适用范围与前提

- 基础模型：`TimeSeriesTransformer`，由输入投影（`input_proj`）、位置编码（`pos`）、`nn.TransformerEncoder` 堆叠与前向头（`head`）组成，典型调用为 `h = input_proj(src) + pos; h = encoder(h); out = head(h[:, -1]).view(...)`。
- 任务假设：以窗口化历史序列预测未来若干时刻的连续值（如流量、负荷）；若采用自回归滚动预测（`horizon=1`），则第 ⑦ 项计划采样适用，否则可跳过。
- 术语约定：`d_model` 为模型隐维度，`num_heads` 为注意力头数，`num_layers` 为编码器层数，`B/L/d` 分别表示 batch、序列长度、隐维度。

## 目录

1. [② 输出聚合改进：注意力池化](#②-输出聚合改进注意力池化)
2. [③ 局部时序卷积增强](#③-局部时序卷积增强)
3. [④ 混合位置编码（绝对 + 相对 / RoPE）](#④-混合位置编码绝对--相对--rope)
4. [⑤ 损失函数增强：差分平滑惩罚](#⑤-损失函数增强差分平滑惩罚)
5. [⑥ 激活函数升级：ReLU → GELU](#⑥-激活函数升级relu--gelu)
6. [⑦ 计划采样（Scheduled Sampling）](#⑦-计划采样scheduled-sampling)
7. [⑧ 模型集成与不确定性估计](#⑧-模型集成与不确定性估计)
8. [⑨ 多尺度特征融合](#⑨-多尺度特征融合)
9. [实施路线图与实验建议](#实施路线图与实验建议)

---

## ② 输出聚合改进：注意力池化

### 问题
原模型仅取最后一个时间步隐状态 `h[:, -1]` 作为输出特征，隐含"序列末尾包含全部预测所需信息"的强假设。当关键事件（突变、尖峰、周期拐点）出现在窗口早期或中段时，尾部 pooling 会丢失这些信息，限制长程预测精度。

### 方案
用**多头自注意力池化**替代"取尾"。让模型对全序列隐状态做自适应加权求和，聚焦于与预测目标最相关的时刻；池化后可取均值或保留加权和。

### 实现（PyTorch）
```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, num_heads=1, num_layers=3, ...):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 注意力池化：用 query=key=value 的自注意力得到上下文加权序列
        self.attn_pool = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, src):
        # src: (B, L, in_channels)
        h = self.input_proj(src) + self.pos(src)
        h = self.encoder(h)                 # (B, L, d)
        h_pool, _ = self.attn_pool(h, h, h) # (B, L, d)
        h_pool = h_pool.mean(dim=1)         # (B, d)；也可改为加权求和
        out = self.head(h_pool).view(...)   # (B, ...)
        return out
```

### 预期收益
- 捕捉更丰富的时间依赖，尤其提升长程/含早期关键事件场景的预测精度。
- 计算量相对整体 Transformer 可忽略，参数量仅增加一个单头注意力。

---

## ③ 局部时序卷积增强

### 问题
Transformer 的全局自注意力对局部短期模式（如 3~6 步内的趋势、尖峰形状）不敏感，容易忽略近期波动，而这些模式在水泵流量、负荷等时序中往往至关重要。

### 方案
在输入投影之后、进入 Encoder 之前，插入一维卷积残差块，强制提取局部时域特征，再与 Transformer 的全局特征残差融合。

### 实现（PyTorch）
```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, num_heads=1, num_layers=3, ...):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        # 局部卷积残差块（kernel=3，padding=1 保持长度）
        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, src):
        h = self.input_proj(src) + self.pos(src)   # (B, L, d)
        h_conv = self.conv_block(h.transpose(1, 2)).transpose(1, 2)  # Conv1d 需 (B,d,L)
        h = h + h_conv                              # 残差连接
        h = self.encoder(h)
        out = self.head(h[:, -1]).view(...)
        return out
```

### 变体
- 多尺度并行卷积：同时使用 `kernel_size=3/7/15` 三组卷积，输出拼接或逐元素相加，再投影回 `d_model`，可覆盖不同长度的局部模式。
- 可将卷积块置于每个 Encoder 层之前，构成 Conv-Transformer 交替结构。

### 预期收益
- 提升对尖峰、突变事件的响应能力，与全局注意力互补，泛化性更强。

---

## ④ 混合位置编码（绝对 + 相对 / RoPE）

### 问题
原正弦绝对位置编码仅编码绝对顺序，无法显式表达周期性相对位置（如隔日同一时刻的相似性），也不具备平移不变性，对外推到更长推理窗口不利。

### 方案
保留正弦绝对编码作为基础，同时引入**相对位置偏置**或采用 **RoPE（旋转位置编码）**。RoPE 通过旋转矩阵将相对位置信息嵌入 query/key，无需额外参数，且在注意力计算中原生支持相对距离，适合小时/日/周周期数据。

### 实现建议
- 推荐直接使用 `transformers` 库或 `rotary_embedding` 相关实现中的 RoPE 模块，避免重复造轮子。
- 若坚持修改 `nn.TransformerEncoderLayer` 的自注意力部分以支持相对偏置（Transformer-XL 风格），实现复杂度较高，建议优先采用 RoPE。

### 预期收益
- 增强对时序周期性的理解，尤其适合具有日/周周期的水泵流量等数据。
- 提升外推能力：推理时窗口长度可与训练时不同。

---

## ⑤ 损失函数增强：差分平滑惩罚

### 问题
纯 MSE 对异常值敏感，且不约束预测序列的时序连续性，易导致预测曲线震荡、出现不合理跳变。

### 方案
在 MSE 基础上增加**一阶差分平滑惩罚**，鼓励相邻预测点的变化平缓，使预测曲线更符合实际物理量的平滑变化规律。

### 实现（训练循环中使用）
```python
def weighted_mse_loss(pred, target, smooth_lambda=0.05):
    """
    pred, target: (B, T) 或 (B,)，T 为预测序列长度
    smooth_lambda: 平滑项权重，建议在验证集上于 0.01~0.1 调优
    """
    mse = ((pred - target) ** 2).mean()
    diff_pred = pred[:, 1:] - pred[:, :-1]   # 一阶差分
    smooth_loss = (diff_pred ** 2).mean()
    return mse + smooth_lambda * smooth_loss
```

### 注意事项
- `smooth_lambda` 需在验证集上调优，典型区间 `0.01~0.1`；数据本身噪声大时可适当增大。
- 若任务允许小幅跳变（如开关机事件），平滑强度不宜过大，避免抹除真实突变。

### 预期收益
- 预测曲线更平滑，减少不合理跳变，通常可改善 MAE/MAPE 指标。

---

## ⑥ 激活函数升级：ReLU → GELU

### 问题
`nn.TransformerEncoderLayer` 默认 `activation="relu"`。GELU（Gaussian Error Linear Unit）在多数 Transformer 场景下收敛更快、精度更高，且与 Transformer 架构天然契合。

### 方案
在构造编码器层时显式指定 `activation="gelu"`。

### 实现（PyTorch）
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=num_heads,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    batch_first=True,
    activation="gelu",   # 默认 "relu"
)
```

### 预期收益
- 通常可轻微提升准确率，基本无额外计算负担；改动成本极低，建议默认启用。

---

## ⑦ 计划采样（Scheduled Sampling）

### 适用场景
**仅当模型用于自回归训练**（即 `horizon=1`、滚动预测）时有效。若为直接多步预测（teacher forcing 一次性输出整段），则无需此策略。

### 问题
训练时完全以真实值作为下一时刻输入（teacher forcing），推理时却只能使用自身预测，导致 **exposure bias**（训练-推理分布不一致），长期滚动预测易累积误差。

### 方案
训练过程中，以概率 `p` 使用真实值、以 `1-p` 使用模型预测值作为下一步输入；`p` 随训练进程从 1.0 逐步衰减至 0.0（如按 epoch 线性或反 sigmoid 衰减）。

### 实现（自回归训练循环片段）
```python
import random

# scheduled_sampling_rate 随 epoch 衰减，例如 linear: max(0.0, 1.0 - epoch/decay_epochs)
for k in range(rollout_steps):
    # 以 scheduled_sampling_rate 取真实值，否则取模型预测
    if random.random() < scheduled_sampling_rate:
        next_target = future_exog[:, k, target_idx]   # 真实值
    else:
        next_target = pred_value.detach()              # 模型预测（断梯度避免 BP 穿过）
    # 将 next_target 组装进下一步的输入行 next_row
    ...
```

### 预期收益
- 显著缓解 exposure bias，提升长期滚动预测的稳定性与鲁棒性。

---

## ⑧ 模型集成与不确定性估计

### 问题
单一模型存在方差，预测结果可能不够稳健，且无法量化预测不确定性，不利于关键决策。

### 方案
- **深度集成**：训练 3~5 个相同架构但不同随机种子（或不同数据采样）的模型，推理时取输出均值作为最终预测。
- **蒙特卡洛 Dropout**：推理时保持 Dropout 开启并多次前向采样，以预测均值和方差估计不确定性（置信区间）。

### 实现（深度集成，PyTorch）
```python
models = [load_model(seed=i) for i in range(5)]   # 加载不同种子的模型
preds = [model(x) for model in models]             # 各模型输出 (B, ...)
pred_ensemble = torch.stack(preds).mean(dim=0)    # 集成均值 (B, ...)
pred_var = torch.stack(preds).var(dim=0)          # 不确定性估计 (B, ...)
```

### 预期收益
- 降低过拟合、提升泛化；深度集成通常带来可观精度增益。
- MC Dropout 可提供预测置信区间，辅助运维决策。

---

## ⑨ 多尺度特征融合

### 问题
仅使用 Encoder 最后一层输出，可能丢失不同抽象层级的特征（低层局部形态与高层语义）。

### 方案
将多个 Encoder 层的输出拼接（或部分层输出拼接），经线性层融合回 `d_model`，再送入输出头/池化层，综合低层与高层语义。

### 实现（PyTorch）
```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, num_heads=1, num_layers=3, ...):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fusion = nn.Linear(num_layers * d_model, d_model)  # 多层拼接后融合
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, src):
        h = self.input_proj(src) + self.pos(src)   # (B, L, d)
        outputs = []
        for layer in self.encoder_layers:
            h = layer(h)
            outputs.append(h)                       # 收集各层输出 (B, L, d)
        h_fused = torch.cat(outputs, dim=-1)        # (B, L, num_layers*d)
        h_fused = self.fusion(h_fused)               # (B, L, d)
        h_pool = h_fused.mean(dim=1)                 # 可替换为注意力池化（见②）
        out = self.head(h_pool).view(...)
        return out
```

### 预期收益
- 综合低层与高层抽象特征，提升特征表达能力；可与②注意力池化组合使用。

---

## 实施路线图与实验建议

| 优先级 | 方法 | 实施难度 | 预期精度提升 |
|---|---|---|---|
| 高 | ② 注意力池化 | 低 | 2~5% |
| 高 | ③ 局部卷积 | 中 | 3~8% |
| 中 | ⑤ 差分平滑损失 | 低 | 1~3% |
| 中 | ⑥ GELU 激活 | 低 | <1% |
| 中 | ④ 混合位置编码（RoPE） | 中 | 2~4% |
| 低 | ⑦ 计划采样（仅自回归） | 中 | 3~6% |
| 低 | ⑧ 模型集成 | 中高 | 5~10% |
| 低 | ⑨ 多尺度融合 | 中 | 2~4% |

### 建议集成顺序
1. 先实施 **②③⑥**：改动小、收益明确、风险低。
2. 根据验证集表现，选择性加入 **④⑤** 以进一步提升周期建模与曲线平滑度。
3. 若模型用于生产且对可靠性要求高，再考虑 **⑦⑧⑨**（尤其⑧深度集成对精度与不确定性均有价值）。

### 实验纪律
- 所有改进均须在**固定验证集**上做消融实验（一次仅变动一项），确认增益真实有效后再组合。
- 组合叠加存在过拟合风险，建议配合**早停**与**权重衰减/Dropout 正则化**。
- 记录每项改动对应的验证集 MAE / MAPE / 平滑度指标，作为取舍依据。

---

> 备注：以上 8 项优化可独立或组合使用；其中 ② 注意力池化与 ③ 局部卷积改动最直观、收益最稳定，建议作为基线增强的首选。所有代码片段为示意性质，实际集成时需与现有 `TimeSeriesTransformer` 的输入维度、`out_dim`、`PositionalEncoding` 实现及训练循环对齐。

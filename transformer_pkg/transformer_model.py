import math

import torch
import torch.nn as nn

# ============================================================================
# transformer_model.py — Transformer 流量预测模型结构 (训练/推理共用)
#
# 2026-08-08 从 train_transformer.py 原样抽取; train_transformer.py 与
# inference_transformer.py 均从此文件导入模型定义, 保证训练与推理用的是
# 同一个模型结构 (同一份代码)。
#
# 结构: 特征投影 + 正弦位置编码 + N 层 Transformer Encoder + 线性输出头,
# 直接多步输出 (无自回归 / 无 teacher forcing), 训练可并行。
#
# 前向接口与 LSTM 版一致 (src, target_len, tgt, teacher_forcing_ratio),
# 以便共用 evaluate / 训练循环; tgt 与 teacher_forcing 不使用。
#
# 【增强】v2.0 — 加入周期对齐的时间特征 (方案一)
#   - 输入特征自动附加小时的正余弦编码 (hour_sin, hour_cos)
#   - 帮助注意力机制关注每日相同时刻的历史模式
#   - 零侵入: 不改模型结构, 只改输入特征
# ============================================================================


def build_sinusoid_pe(seq_len, d_model):
    """正弦位置编码 (Vaswani et al. 2017): 偶维 sin / 奇维 cos, 频率按 10000^(-2k/d) 递减。

    与窗口长度解耦: 任意 seq_len 即时生成, 换 lookback 窗口长度无需重训
    (可学习位置编码的长度与训练窗口绑定, 做不到这一点)。
    """
    assert d_model % 2 == 0, "d_model 需为偶数 (正弦编码按偶/奇维拆分)"
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)   # (L, 1)
    k = torch.arange(d_model // 2, dtype=torch.float32)             # (d/2,)
    freq = torch.exp(k * (-math.log(10000.0) / (d_model / 2)))      # 10000^(-2k/d)
    ang = pos * freq                                                # (L, d/2)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


def add_periodic_features(src, period=24):
    """
    为输入序列添加周期对齐的时间特征 (方案一)
    
    参数:
        src: 原始输入, shape (batch, seq_len, input_dim)
             假设序列是按小时连续采样的, 第一个时间步对应某个小时的起点
        period: 周期长度, 默认为24 (日周期)
    
    返回:
        enhanced_src: 增强后的输入, shape (batch, seq_len, input_dim + 2)
                      新增的两个维度是 hour_sin 和 hour_cos
    
    原理:
        - hour_sin 和 hour_cos 共同编码了一天中的小时位置 (0-23)
        - 使用正余弦编码避免了 0点 和 23点 之间的跳变问题
        - 模型可以通过注意力机制, 自动关注历史窗口中"同相位"的时间步
        - 例如预测1月8日12点, 模型会更关注前7天每天的12点
    """
    batch, seq_len, input_dim = src.shape
    
    # 生成每个时间步在周期中的位置 (0, 1, 2, ..., period-1)
    # 假设序列的第一个时间步对应周期起点 (例如午夜0点)
    # 在实际应用中, 可以根据数据的实际起始时间进行调整
    positions = torch.arange(seq_len, device=src.device) % period  # (seq_len,)
    positions = positions.float()  # 转为浮点数用于三角函数
    
    # 计算正余弦编码
    # 归一化到 [0, 2π) 区间
    angle = 2 * math.pi * positions / period
    
    hour_sin = torch.sin(angle).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
    hour_cos = torch.cos(angle).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
    
    # 扩展到 batch 维度
    hour_sin = hour_sin.expand(batch, -1, -1)  # (batch, seq_len, 1)
    hour_cos = hour_cos.expand(batch, -1, -1)  # (batch, seq_len, 1)
    
    # 拼接到原始特征后面
    enhanced_src = torch.cat([src, hour_sin, hour_cos], dim=-1)  # (batch, seq_len, input_dim + 2)
    
    return enhanced_src


class TimeSeriesTransformer(nn.Module):
    """Transformer Encoder 多步预测模型: 输入投影 + 正弦位置编码 + 多头自注意力 + 线性输出头。

    结构:
      Linear(input_dim → d_model) 特征投影
      + 正弦位置编码 (与窗口长度解耦, 换 lookback 无需重训)
      + N 层 Transformer Encoder (自注意力聚合全窗口信息, 长程依赖)
      + 最后时刻表示 → Linear(d_model → horizon * output_dim) 直接多步输出

    增强功能:
      - 自动添加周期对齐的时间特征 (hour_sin, hour_cos)
      - 可通过 use_periodic_features 开关控制
    """
    def __init__(self, input_dim, output_dim, horizon, input_len,
                 d_model=64, nhead=4, num_layers=3, dim_feedforward=256, 
                 dropout=0.1, use_periodic_features=True, period=24):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        
        self.use_periodic_features = use_periodic_features
        self.period = period
        
        # 如果启用周期特征, 输入维度增加2 (hour_sin, hour_cos)
        effective_input_dim = input_dim + (2 if use_periodic_features else 0)
        
        self.input_proj = nn.Linear(effective_input_dim, d_model)
        # input_len 仅用于显式声明支持的最大输入长度, 实际位置编码按输入长度即时生成

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_dim = output_dim
        self.horizon = horizon
        self.head = nn.Linear(d_model, horizon * output_dim)

    def forward(self, src, target_len, tgt=None, teacher_forcing_ratio=0.0):
        # Step 1: 可选地添加周期对齐的时间特征
        if self.use_periodic_features:
            src = add_periodic_features(src, period=self.period)
        
        # Step 2: 生成位置编码并叠加
        pos = build_sinusoid_pe(src.size(1), self.input_proj.out_features).to(src.device)
        h = self.input_proj(src) + pos
        
        # Step 3: Transformer Encoder 编码
        h = self.encoder(h)
        
        # Step 4: 取最后时刻的隐状态 (自注意力已聚合全窗口信息)
        h = h[:, -1]  # (batch, d_model)
        
        # Step 5: 线性映射并reshape为多步输出
        out = self.head(h).view(h.size(0), -1, self.output_dim)  # (batch, horizon, output_dim)
        
        return out[:, :target_len]
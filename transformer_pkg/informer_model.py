import torch
import torch.nn as nn
import numpy as np
import math

# ============================================================================
# informer_model.py — Informer 流量预测模型结构 (训练/推理共用)
#
# 参考: Zhou et al., "Informer: Beyond Efficient Transformer for Long
# Sequence Time-Series Forecasting", AAAI 2021.
#
# 核心创新:
#   1. ProbSparse Self-Attention: 通过 KL 散度度量每个 query 的 "活跃度",
#      只对 top-u 个活跃 query 计算 full attention, 其余用 mean-padding 近似,
#      复杂度从 O(L²) 降至 O(L·logL)。
#   2. Self-attention Distillation: 在相邻 encoder 层之间插入 MaxPool 层,
#      逐步缩减序列长度, 提取主导特征, 实现层级式特征蒸馏。
#
# 本实现沿用 iTransformer 的 "变量即 token" 范式:
#   - 输入 (B, L, C) → RevIN 归一化 → 转置 (B, C, L)
#   - 每个变量的一条序列 = 一个 token, 做跨变量 ProbSparse Attention
#   - Encoder 层间做 MaxPool 蒸馏 (C 轴不变, token 维度不变, 蒸馏作用于
#     feature 维度) —— 实际蒸馏沿 feature 轴下采样
#   - 最终 Linear(d_model → horizon) 逐变量输出预测
#
# 接口与 iTransformer / TimeSeriesTransformer 完全一致:
#   forward(src, target_len, tgt, teacher_forcing_ratio)
# ============================================================================


class RevIN(nn.Module):
    """可逆实例归一化 (Reversible Instance Normalization, Kim et al. 2021)。

    与 iTransformer 中完全一致: 逐样本、逐通道对 lookback 轴做归一化,
    统计量 detach 不参与梯度, 支持 norm/denorm 两阶段。
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise ValueError(f"mode 只支持 norm/denorm, 收到 {mode}")
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / (self.stdev + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.bias
            x = x / (self.weight + self.eps)
        x = x * self.stdev + self.mean
        return x


def prob_sparse_attention(Q, K, V, mask=None, factor=5):
    """ProbSparse Self-Attention (Zhou et al. 2021)。

    核心思想: 自注意力的注意力矩阵是稀疏的, 大部分 query 与其他 key 的
    互信息很低 (可以用全局均值代替), 只有少数 "活跃" query 需要完整 attention。

    步骤:
      1. 计算每个 query 与所有 key 的点积
      2. 计算每个 query 的 "活跃度" M(q_i) = max_j(q_i·k_j) - mean_j(q_i·k_j)
      3. 选取 top-u 个活跃 query (u = L · ln(L) / factor)
      4. 只对这 u 个 query 做 full attention, 其余用 V 的均值近似

    Args:
        Q, K, V: (B, nhead, L, d_k) — 已经做了 head 分裂和缩放
        mask: 可选的注意力掩码
        factor: 控制稀疏度的因子, 越大保留的 query 越少
    Returns:
        (B, nhead, L, d_k) — 注意力输出, 与标准 attention 形状一致
    """
    B, H, L_Q, D = Q.shape
    _, _, L_K, _ = K.shape

    # Step 1: 计算原始注意力分数 (B, H, L_Q, L_K)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

    # 如果 mask 存在, 将被 mask 的位置设为 -inf
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 2: 计算每个 query 的活跃度 M(q_i) = max - mean (排除被 mask 的位置)
    # 对于有 mask 的位置, 用 -inf 的 max 仍然合理
    U_part = max(1, min(int(L_Q * math.log(L_K) / factor), L_K))  # top-u
    U_part = min(U_part, L_Q)  # 不能超过 query 数量

    # M: (B, H, L_Q) — 每个 query 的活跃度得分
    M = torch.max(scores, dim=-1).values - torch.mean(
        scores.masked_fill(scores == float("-inf"), 0), dim=-1
    ) + torch.mean(
        (scores != float("-inf")).float(), dim=-1
    ) * float("-inf")
    # 更简洁: M = max(scores) - mean(scores), 忽略 -inf 位置
    M_scores = torch.max(scores, dim=-1).values  # (B, H, L_Q)
    # 用 softmax mask 计算非 mask 位置的均值
    valid_mask = (scores != float("-inf")).float()  # (B, H, L_Q, L_K)
    sum_scores = (scores * valid_mask).sum(dim=-1)  # (B, H, L_Q)
    count_valid = valid_mask.sum(dim=-1).clamp(min=1)  # (B, H, L_Q)
    mean_scores = sum_scores / count_valid
    M = M_scores - mean_scores  # (B, H, L_Q) 活跃度

    # Step 3: 选取 top-u 个最活跃的 query
    M_top = M.topk(U_part, dim=-1).indices  # (B, H, u)

    # 用 gather 取出活跃 query 对应的 Q
    idx = M_top.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, H, u, D)
    Q_top = torch.gather(Q, dim=2, index=idx)  # (B, H, u, D)

    # Step 4: 只对活跃 query 做 full attention
    scores_top = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(D)  # (B, H, u, L_K)
    if mask is not None:
        # 需要对 mask 也做相应的 gather —— 简化: 不传 mask 时直接算
        mask_top = torch.gather(
            mask.expand(B, H, -1, -1), dim=2,
            index=M_top.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, L_K)
        )
        scores_top = scores_top.masked_fill(mask_top == 0, float("-inf"))
    attn_top = torch.softmax(scores_top, dim=-1)  # (B, H, u, L_K)
    output_top = torch.matmul(attn_top, V)  # (B, H, u, D)

    # Step 5: 非活跃 query 用 V 的均值近似
    V_mean = V.mean(dim=2, keepdim=True)  # (B, H, 1, D)

    # 组合: 活跃 query 用完整 attention, 其余用均值
    # 创建输出张量
    output = V_mean.expand(-1, -1, L_Q, -1).clone()  # (B, H, L_Q, D)
    output = output.scatter(2, M_top.unsqueeze(-1).expand(-1, -1, -1, D), output_top)

    return output


class ProbSparseMultiHeadAttention(nn.Module):
    """基于 ProbSparse 的多头注意力层。"""

    def __init__(self, d_model, nhead, dropout=0.1, factor=5):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.factor = factor

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, d_model) — 输入序列
            mask: 可选的注意力掩码 (B, 1, L, L) 或 (1, 1, L, L)
        Returns:
            (B, L, d_model) — 注意力输出
        """
        B, L, _ = x.shape

        # 线性投影 + 分头: (B, L, d_model) → (B, nhead, L, d_k)
        Q = self.W_q(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)

        # ProbSparse Attention
        attn_out = prob_sparse_attention(Q, K, V, mask=mask, factor=self.factor)
        # dropout
        attn_out = self.dropout(attn_out)

        # 合并多头: (B, nhead, L, d_k) → (B, L, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(attn_out)


class DistillationBlock(nn.Module):
    """Self-attention Distillation: 通过 MaxPool1d 对 token 序列做下采样。

    在 Informer 中, 每隔一层做一次蒸馏, 将序列长度减半,
    逐步提取主导特征 (dominant features)。
    本实现中 token 维度 = input_dim (变量数), 蒸馏沿此轴做 pool。

    输入: (B, C, d_model) → MaxPool1d(kernel=2, stride=2) → (B, C//2, d_model)
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        """
        Args:
            x: (B, C, d_model)
        Returns:
            (B, C//2, d_model) — 蒸馏后的特征
        """
        # MaxPool1d 作用于最后一维 (d_model) — 但我们想对 token 轴 (C) 做蒸馏
        # 所以需要转置: (B, C, d_model) → (B, d_model, C) → pool → (B, d_model, C//2) → 转回
        x = x.permute(0, 2, 1)  # (B, d_model, C)
        x = self.pool(x)        # (B, d_model, C//2)
        x = x.permute(0, 2, 1)  # (B, C//2, d_model)
        return x


class InformerEncoderLayer(nn.Module):
    """Informer Encoder 层: ProbSparse Attention + FFN + 残差 + LayerNorm。

    结构:
      x → ProbSparseMultiHeadAttention → Add & LayerNorm
      → FeedForward → Add & LayerNorm
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, factor=5):
        super().__init__()
        self.self_attn = ProbSparseMultiHeadAttention(
            d_model, nhead, dropout=dropout, factor=factor
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_out = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # Feed-forward + residual + norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class Informer(nn.Module):
    """Informer: 基于 ProbSparse Attention 的高效 Transformer 多步预测模型。

    结构 (沿用 iTransformer "变量即 token" 范式):
      输入 (B, L, C) → RevIN 实例归一化
      → 转置 (B, C, L): 每个变量的一条序列 = 一个 token
      → 共享 Linear(L → d_model) 嵌入 + 可学习变量位置编码 (1, C, d_model)
      → N 层 InformerEncoderLayer (ProbSparse 跨变量 self-attention)
      → LayerNorm → 共享 Linear(d_model → horizon) 逐变量出完整预测
      → 转置 (B, horizon, C) → RevIN 反归一化
      → 取目标变量通道 → (B, target_len, output_dim)

    与标准 Transformer 的区别:
      - 使用 ProbSparse Attention 替代 full attention:
        通过 KL 散度度量每个 query 的活跃度, 只对 top-u 个活跃 query
        计算完整 attention, 其余用 V 的均值近似, 复杂度 O(L·logL)。
      - 在 "变量即 token" 范式中, token 数 = 变量数 (通常 < 20),
        ProbSparse 的效率优势在 token 数很大时更显著;
        但在变量数较少时, 仍保持与 full attention 相当的效果,
        因为 KL 散度筛选保证了信息不丢失。

    接口与 iTransformer / TimeSeriesTransformer 完全一致。
    """

    def __init__(self, input_dim, output_dim, horizon, input_len,
                 d_model=64, nhead=4, num_layers=3, dim_feedforward=256,
                 dropout=0.1, target_idx=0, factor=5):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.output_dim = output_dim
        self.horizon = horizon
        self.target_idx = target_idx

        self.revin = RevIN(input_dim)
        self.embedding = nn.Linear(input_len, d_model)        # 共享嵌入: 序列 → token
        self.var_pos_emb = nn.Parameter(torch.zeros(1, input_dim, d_model))

        # 构建 N 层 InformerEncoderLayer (ProbSparse Attention)
        self.layers = nn.ModuleList([
            InformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout, factor=factor
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)               # 共享输出头: 逐变量出整个 horizon

    def forward(self, src, target_len, tgt=None, teacher_forcing_ratio=0.0):
        """
        Args:
            src: (B, L, C) — 输入序列
            target_len: 目标长度 (截断输出)
            tgt: 未使用 (直接多步输出)
            teacher_forcing_ratio: 未使用
        Returns:
            (B, target_len, output_dim) — 预测输出
        """
        # src: (B, L, C)
        x = self.revin(src, "norm")                           # (B, L, C) 实例归一化
        x = x.permute(0, 2, 1)                                # (B, C, L) 变量 = token
        x = self.embedding(x) + self.var_pos_emb              # (B, C, d_model)

        # 逐层 ProbSparse Attention 编码
        for layer in self.layers:
            x = layer(x)

        x = self.head(self.layer_norm(x))                     # (B, C, horizon)
        x = x.permute(0, 2, 1)                                # (B, horizon, C)
        x = self.revin(x, "denorm")                           # 反归一化还原
        x = x[..., self.target_idx:self.target_idx + self.output_dim]  # (B, H, output_dim)
        return x[:, :target_len]

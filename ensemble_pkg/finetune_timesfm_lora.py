# -*- coding: utf-8 -*-
"""TimesFM 2.5 LoRA 微调 (transformers 版) — 用水厂流量数据微调 TimesFM 的注意力层。

原理:
  TimesFM 2.5 是200M参数的预训练时序基础模型。LoRA (Low-Rank Adaptation) 在每层
  Transformer 的注意力投影矩阵旁边插入低秩矩阵 A·B, 只训练 A·B (参数量极少),
  冻结原始权重, 既保留预训练能力又适配新数据。

LoRA 目标层:
  每层 Transformer 的 self_attn.q_proj, k_proj, v_proj, o_proj

用法:
  python finetune_timesfm_lora.py
  python finetune_timesfm_lora.py --epochs 20 --lora_rank 16 --lr 3e-4

产物:
  timesfm_lora/ 目录: LoRA 权重 + 训练配置 + loss 曲线
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from _paths import ensure_import_paths, PROJECT_ROOT              # noqa: E402
ensure_import_paths(verbose=False)

from data_processing import DataProcessor                          # noqa: E402

# ── 默认路径 ──
DEFAULT_TIMESFM_MODEL = os.path.join(PROJECT_ROOT, "timesfm_model_transformers")
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "水厂2025年小时级汇总.csv")
DEFAULT_OUT_DIR = os.path.join(_HERE, "timesfm_lora")

CONTEXT_LEN = 168 * 3          # 7天 × 24h = 168 小时
HORIZON_LEN = 24           # 预测24小时
PATCH_LEN = 32             # TimesFM 内部 patch 长度


# ==================== LoRA 层 ====================

class LoRALinear(nn.Module):
    """低秩线性层: W = W_frozen + B @ A, 其中 A, B 是可训练的低秩矩阵。

    frozen_weight: 原始权重 (冻结, 不更新)
    in_features: 输入维度
    out_features: 输出维度
    rank: LoRA 秩 (默认 8)
    alpha: LoRA 缩放因子 (默认 16)
    """
    def __init__(self, frozen_linear, rank=8, alpha=16.0):
        super().__init__()
        self.frozen = frozen_linear
        in_features = frozen_linear.in_features
        out_features = frozen_linear.out_features

        # 冻结原始权重
        for p in self.frozen.parameters():
            p.requires_grad = False

        # LoRA 矩阵 (与 frozen 层同设备)
        device = next(frozen_linear.parameters()).device
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device))
        self.scaling = alpha / rank

    def forward(self, x):
        # 原始输出 + LoRA 增量
        frozen_out = self.frozen(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return frozen_out + lora_out


def apply_lora(model, rank=8, alpha=16.0, target_modules=None):
    """给 Transformers 版 TimesFM 的注意力层添加 LoRA。

    Args:
        model: TimesFm2_5ModelForPrediction 模型
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        target_modules: 要替换的模块名列表, 默认 ["q_proj", "k_proj", "v_proj", "o_proj"]

    Returns:
        (model, n_trainable): 添加 LoRA 后的模型, 可训练参数数
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    n_replaced = 0
    # 遍历所有 Transformer 层 (model.model.layers)
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for mod_name in target_modules:
            original = getattr(attn, mod_name)
            if isinstance(original, nn.Linear) and not isinstance(original, LoRALinear):
                lora = LoRALinear(original, rank=rank, alpha=alpha)
                setattr(attn, mod_name, lora)
                n_replaced += 1

    # 统计可训练参数
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] 替换 {n_replaced} 个线性层")
    print(f"[LoRA] 可训练参数: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.2f}%)")
    return model, n_trainable


# ==================== 数据 ====================

def load_flow_series(raw_csv, config_override=None):
    """加载原始数据 → 清洗 → 返回小时级 Total_Flow Series。"""
    config = {
        "file_path": raw_csv,
        "encoding": "utf-8-sig",
        "resample_freq": "60min",
        "hampel_cols": ["Total_Flow"],
        "hampel_window": 48,
        "spike_ratio_within": 1.2,
        "spike_ratio_cross": 1.3,
        "spike_abs_floor": 100.0,
        "smooth_median_window": 3,
    }
    if config_override:
        config.update(config_override)

    proc = DataProcessor(config)
    df_raw = proc.load_raw()
    df_base = proc.build_base_features(df_raw)
    df_clean = proc.clean_and_resample(df_base)

    # 小时级均值
    hourly = df_clean["Total_Flow"].resample("h").mean().dropna()
    print(f"[数据] 小时级序列: {len(hourly)} 点, "
          f"{hourly.index.min()} ~ {hourly.index.max()}")
    return hourly


def _sliding_window(vals, context_len, horizon_len, stride):
    """从一维数组生成滑动窗口 (context, target) 对。"""
    contexts, targets = [], []
    for i in range(0, len(vals) - context_len - horizon_len + 1, stride):
        ctx = vals[i:i + context_len]
        tgt = vals[i + context_len:i + context_len + horizon_len]
        if np.any(np.isnan(ctx)) or np.any(np.isnan(tgt)):
            continue
        contexts.append(ctx)
        targets.append(tgt)
    if not contexts:
        return np.empty((0, context_len)), np.empty((0, horizon_len))
    return np.array(contexts, dtype=np.float64), np.array(targets, dtype=np.float64)


def make_lora_training_samples(hourly, context_len=CONTEXT_LEN,
                                horizon_len=HORIZON_LEN, stride=1,
                                val_ratio=0.2):
    """按时间先切分, 再分别生成训练/验证样本 — 杜绝数据泄露。

    划分方式: 时间序列前 80% → 训练样本, 后 20% → 验证样本。
    训练集的任何样本都不会和验证集共享同一小时的原始数据。

    Returns:
        (train_ctx, train_tgt, val_ctx, val_tgt)
    """
    vals = hourly.values.astype(np.float64)
    n = len(vals)
    split = int(n * (1 - val_ratio))

    # 先按时间切开, 再各自生成样本
    train_vals = vals[:split]
    val_vals = vals[split - context_len:]  # 留出 context 重叠, 保证 val 第一个样本有完整回看

    train_ctx, train_tgt = _sliding_window(train_vals, context_len, horizon_len, stride)
    val_ctx, val_tgt = _sliding_window(val_vals, context_len, horizon_len, stride)

    print(f"[数据] 时间序列 {n} 点, 切分点={split} (前 {1-val_ratio:.0%} 训练 / 后 {val_ratio:.0%} 验证)")
    print(f"[数据] 训练样本: {len(train_ctx)} 个  验证样本: {len(val_ctx)} 个  "
          f"(context={context_len}, horizon={horizon_len}, stride={stride})")
    return train_ctx, train_tgt, val_ctx, val_tgt


# ==================== 训练 ====================

class LoRATimeSeriesDataset(torch.utils.data.Dataset):
    """LoRA 微调数据集: context + target。"""
    def __init__(self, contexts, targets):
        self.contexts = torch.from_numpy(contexts).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


def train_lora(model, train_ctx, train_tgt, val_ctx, val_tgt, args):
    """LoRA 微调主循环。

    训练策略:
      - 冻结所有非 LoRA 参数, 只训练 LoRA 矩阵
      - TimesFM 的输入是 (1, context_len) 的浮点数组
      - 用 MSE loss 对齐预测值和目标值
      - 早停: 连续 patience 个 epoch 验证 loss 不降则停止
    """
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()  # 部分层有 dropout, eval 模式更稳定

    train_dataset = LoRATimeSeriesDataset(train_ctx, train_tgt)
    val_dataset = LoRATimeSeriesDataset(val_ctx, val_tgt)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        pin_memory=True, num_workers=0)

    # 只训练 LoRA 参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[训练] 可训练参数: {sum(p.numel() for p in trainable_params):,}")
    print(f"[训练] 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    # 总 batch 数 (用于进度条总量)
    n_train_batches_total = len(train_loader)
    n_val_batches_total = len(val_loader)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ── 训练 ──
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch:3d}/{args.epochs} [Train]",
            total=n_train_batches_total, leave=False, ncols=100,
            bar_format="{l_bar}{bar:30}{r_bar}")
        for ctx_batch, tgt_batch in train_bar:
            ctx_batch = ctx_batch.to(device)    # (B, 168)
            tgt_batch = tgt_batch.to(device)    # (B, 24)

            optimizer.zero_grad(set_to_none=True)

            # TimesFM forward: 逐样本处理, 每个样本用自身 context 做 z-score 归一化
            batch_preds = []
            batch_tgt_norm = []
            for i in range(len(ctx_batch)):
                ctx_np = ctx_batch[i].cpu().numpy().astype(np.float64)
                # z-score: 用 context 的均值/标准差归一化, 让 loss 量级稳定在 ~1
                mu = ctx_np.mean()
                sigma = ctx_np.std() + 1e-8
                ctx_norm = (ctx_np - mu) / sigma
                ctx_tensor = torch.from_numpy(ctx_norm.astype(np.float32)).to(device)
                outputs = model(
                    past_values=[ctx_tensor],
                    forecast_context_len=16256)
                pred = outputs.mean_predictions[0, :HORIZON_LEN]  # 取前24步
                batch_preds.append(pred)
                # target 用同样的 mu/sigma 归一化
                tgt_norm = (tgt_batch[i] - mu) / sigma
                batch_tgt_norm.append(tgt_norm)

            pred_all = torch.stack(batch_preds)       # (B, 24)  归一化空间
            tgt_norm_all = torch.stack(batch_tgt_norm) # (B, 24)  归一化空间

            # MSE loss (归一化空间, loss ~1 量级)
            loss = nn.functional.mse_loss(pred_all, tgt_norm_all)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1
            train_bar.set_postfix(loss=f"{loss.item():.6f}",
                                  avg=f"{train_loss_sum / n_train_batches:.6f}")
        train_bar.close()

        train_loss = train_loss_sum / max(n_train_batches, 1)
        scheduler.step()

        # ── 验证 ──
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        val_bar = tqdm(
            val_loader, desc=f"Epoch {epoch:3d}/{args.epochs} [Val]  ",
            total=n_val_batches_total, leave=False, ncols=100,
            bar_format="{l_bar}{bar:30}{r_bar}")
        with torch.no_grad():
            for ctx_batch, tgt_batch in val_bar:
                ctx_batch = ctx_batch.to(device)
                tgt_batch = tgt_batch.to(device)

                batch_preds = []
                batch_tgt_norm = []
                for i in range(len(ctx_batch)):
                    ctx_np = ctx_batch[i].cpu().numpy().astype(np.float64)
                    mu = ctx_np.mean()
                    sigma = ctx_np.std() + 1e-8
                    ctx_norm = (ctx_np - mu) / sigma
                    ctx_tensor = torch.from_numpy(ctx_norm.astype(np.float32)).to(device)
                    outputs = model(
                        past_values=[ctx_tensor],
                        forecast_context_len=16256)
                    pred = outputs.mean_predictions[0, :HORIZON_LEN]
                    batch_preds.append(pred)
                    tgt_norm = (tgt_batch[i] - mu) / sigma
                    batch_tgt_norm.append(tgt_norm)

                pred_all = torch.stack(batch_preds)
                tgt_norm_all = torch.stack(batch_tgt_norm)
                val_loss = nn.functional.mse_loss(pred_all, tgt_norm_all)
                val_loss_sum += val_loss.item()
                n_val_batches += 1
                val_bar.set_postfix(loss=f"{val_loss.item():.6f}")
        val_bar.close()

        val_loss = val_loss_sum / max(n_val_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "lr": lr_now})

        best_mark = "  ★ best" if val_loss < best_val_loss else ""
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"lr={lr_now:.6f}  time={epoch_time:.1f}s"
              f"{best_mark}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    return best_state, history, best_val_loss


def save_lora_weights(model, out_dir, args, history, best_val_loss):
    """保存 LoRA 权重 (只保存 LoRA 矩阵, 不保存冻结权重)。"""
    os.makedirs(out_dir, exist_ok=True)

    # 提取 LoRA 参数
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu()

    # 保存
    torch.save(lora_state, os.path.join(out_dir, "lora_weights.pth"))

    config = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "context_len": CONTEXT_LEN,
        "horizon_len": HORIZON_LEN,
        "base_model_path": os.path.abspath(args.timesfm_model),
        "train_data": os.path.abspath(args.data),
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history),
    }
    with open(os.path.join(out_dir, "lora_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 保存训练历史
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "train_history.csv"),
                                 index=False)

    # 画 loss 曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    ax.plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"TimesFM LoRA 微调 (rank={args.lora_rank})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

    print(f"\n[保存] LoRA 权重: {os.path.join(out_dir, 'lora_weights.pth')}")
    print(f"[保存] 配置: {os.path.join(out_dir, 'lora_config.json')}")
    print(f"[保存] 训练历史: {os.path.join(out_dir, 'train_history.csv')}")
    print(f"[保存] Loss 曲线: {os.path.join(out_dir, 'loss_curve.png')}")


# ==================== 主入口 ====================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TimesFM 2.5 LoRA 微调 (transformers 版)")
    p.add_argument("--data", default=DEFAULT_DATA, help="原始数据 CSV")
    p.add_argument("--timesfm_model", default=DEFAULT_TIMESFM_MODEL,
                   help="TimesFM 预训练模型目录 (transformers 格式)")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="输出目录")
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA 秩 (默认 8)")
    p.add_argument("--lora_alpha", type=float, default=16.0,
                   help="LoRA 缩放因子 (默认 16)")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率 (默认 3e-4)")
    p.add_argument("--epochs", type=int, default=30, help="最大训练轮数")
    p.add_argument("--batch_size", type=int, default=4, help="批大小")
    p.add_argument("--patience", type=int, default=5, help="早停耐心值")
    p.add_argument("--stride", type=int, default=24, help="滑动窗口步长 (默认 24 = 一天)")
    p.add_argument("--device", default=None, help="推理设备")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # 随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── 加载数据 ──
    print(f"\n{'='*70}\n TimesFM LoRA 微调 (transformers 版)\n{'='*70}")
    hourly = load_flow_series(args.data)
    train_ctx, train_tgt, val_ctx, val_tgt = make_lora_training_samples(
        hourly, stride=args.stride)

    # ── 加载 TimesFM (transformers) ──
    print(f"\n[模型] 加载 TimesFM: {args.timesfm_model}")
    from transformers import TimesFm2_5ModelForPrediction
    model = TimesFm2_5ModelForPrediction.from_pretrained(args.timesfm_model)
    model = model.to(torch.float32)
    print("[模型] TimesFM 加载完成 (transformers)")

    # ── 应用 LoRA ──
    print(f"\n[LoRA] rank={args.lora_rank}, alpha={args.lora_alpha}")
    model, n_trainable = apply_lora(
        model, rank=args.lora_rank, alpha=args.lora_alpha)

    # ── 训练 ──
    print(f"\n[训练] 开始 LoRA 微调...")
    t0 = time.time()
    best_state, history, best_val_loss = train_lora(
        model, train_ctx, train_tgt, val_ctx, val_tgt, args)
    t1 = time.time()
    print(f"\n[训练] 完成, 耗时 {t1-t0:.1f}s, 最佳验证 loss = {best_val_loss:.6f}")

    # ── 保存 ──
    if best_state is not None:
        model.load_state_dict(best_state)
    save_lora_weights(model, args.out_dir, args, history, best_val_loss)

    print(f"\n{'='*70}")
    print(f" LoRA 微调完成!")
    print(f"  rank={args.lora_rank}, alpha={args.lora_alpha}, lr={args.lr}")
    print(f"  best_val_loss={best_val_loss:.6f}")
    print(f"  可训练参数: {n_trainable:,}")
    print(f"  输出目录: {args.out_dir}")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

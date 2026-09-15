# -*- coding: utf-8 -*-
"""TimesFM 2.5 LoRA 微调 — 用水厂流量数据微调 TimesFM 的注意力层。

原理:
  TimesFM 2.5 是200M参数的预训练时序基础模型。LoRA (Low-Rank Adaptation) 在每层
  Transformer 的注意力投影矩阵旁边插入低秩矩阵 A·B, 只训练 A·B (参数量极少),
  冻结原始权重, 既保留预训练能力又适配新数据。

LoRA 目标层:
  每层 Transformer 的 attn.qkv_proj (Q/K/V合并投影) 和 attn.out (输出投影)

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
DEFAULT_TIMESFM_MODEL = os.path.join(PROJECT_ROOT, "timesfm_model")
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "水厂2025年小时级汇总.csv")
DEFAULT_OUT_DIR = os.path.join(_HERE, "timesfm_lora")

CONTEXT_LEN = 168          # 7天 × 24h = 168 小时
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


def patch_decode_no_grad(model):
    """Monkey-patch: 让 decode() 和 compiled_decode() 中的梯度能回传。

    两步:
    1. decode() 中的 torch.no_grad → no-op
    2. compiled_decode() 闭包中 output[0].cpu().numpy() → 保留 tensor
    """
    import contextlib, types

    class _NoOpNoGrad:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __call__(self, func): return func

    original_decode = model.model.decode
    _real_no_grad = torch.no_grad
    original_compiled_decode = model.compiled_decode

    def patched_decode(horizon, inputs, masks):
        torch.no_grad = _NoOpNoGrad
        try:
            return original_decode(horizon, inputs, masks)
        finally:
            torch.no_grad = _real_no_grad

    model.model.decode = patched_decode

    # Patch compiled_decode: 保留 tensor 输出 (不转 numpy)
    import timesfm.torch.util as tfm_util

    def patched_compiled_decode(horizon, inputs, masks):
        fc = model.forecast_config
        if horizon > fc.max_horizon:
            raise ValueError(f"Horizon {horizon} > max {fc.max_horizon}")

        inputs_t = torch.from_numpy(np.array(inputs)).to(model.model.device).float()
        masks_t = torch.from_numpy(np.array(masks)).to(model.model.device).bool()
        batch_size = inputs_t.shape[0]

        if fc.infer_is_positive:
            is_positive = torch.all(inputs_t >= 0, dim=-1, keepdim=True)
        else:
            is_positive = None

        if fc.normalize_inputs:
            mu = torch.mean(inputs_t, dim=-1, keepdim=True)
            sigma = torch.std(inputs_t, dim=-1, keepdim=True)
            inputs_t = tfm_util.revin(inputs_t, mu, sigma, reverse=False)
        else:
            mu, sigma = None, None

        torch.no_grad = _NoOpNoGrad
        try:
            pf_outputs, quantile_spreads, ar_outputs = model.model.decode(
                fc.max_horizon, inputs_t, masks_t)
        finally:
            torch.no_grad = _real_no_grad

        to_cat = [pf_outputs[:, -1, ...]]
        if ar_outputs is not None:
            to_cat.append(ar_outputs.reshape(batch_size, -1, model.model.q))
        full_forecast = torch.cat(to_cat, dim=1)

        # 保留 tensor 输出 (不转 numpy), 后续通过 .cpu().numpy() 再转换
        def flip_quantile_fn(x):
            return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1)

        if fc.normalize_inputs and mu is not None:
            full_forecast = tfm_util.revin(full_forecast, mu, sigma, reverse=True)
            if is_positive is not None:
                full_forecast = full_forecast * is_positive.float()
            if fc.fix_quantile_crossing:
                full_forecast = torch.cat([
                    full_forecast[..., :5],
                    flip_quantile_fn(full_forecast[..., 5:])
                ], dim=-1)

        return full_forecast[..., :5], full_forecast

    model.compiled_decode = patched_compiled_decode

    def restore():
        torch.no_grad = _real_no_grad
        model.model.decode = original_decode
        model.compiled_decode = original_compiled_decode
    return restore


def apply_lora(model, rank=8, alpha=16.0, target_modules=None):
    """给 TimesFM 的指定层添加 LoRA。

    Args:
        model: TimesFM 模型
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        target_modules: 要替换的模块名列表, 默认 ["qkv_proj", "out"]

    Returns:
        (lora_model, n_trainable): 添加 LoRA 后的模型, 可训练参数数
    """
    if target_modules is None:
        target_modules = ["qkv_proj", "out"]

    n_replaced = 0
    # 遍历所有 Transformer 层
    for layer_idx, transformer in enumerate(model.model.stacked_xf):
        attn = transformer.attn
        for mod_name in target_modules:
            original = getattr(attn, mod_name)
            if isinstance(original, nn.Linear) and not isinstance(original, LoRALinear):
                lora = LoRALinear(original, rank=rank, alpha=alpha)
                setattr(attn, mod_name, lora)
                n_replaced += 1

    # 统计可训练参数 (注意: TimesFM wrapper 的参数在 .model 子模块上)
    net = model.model
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in net.parameters())
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


def make_lora_training_samples(hourly, context_len=CONTEXT_LEN,
                                horizon_len=HORIZON_LEN, stride=1):
    """滑动窗口生成 LoRA 训练样本。

    每个样本: (context[0:context_len], target[context_len:context_len+horizon_len])
    输入 TimesFM 时 context 是归一化后的时间序列 (单变量)。

    Args:
        hourly: 小时级流量 Series
        context_len: 上下文长度 (默认 168)
        horizon_len: 预测长度 (默认 24)
        stride: 滑动步长 (默认1=密集采样, 加快用 >1)

    Returns:
        contexts: (N, context_len) numpy
        targets:  (N, horizon_len) numpy
    """
    vals = hourly.values.astype(np.float64)
    n = len(vals)
    contexts, targets = [], []

    for i in range(0, n - context_len - horizon_len + 1, stride):
        ctx = vals[i:i + context_len]
        tgt = vals[i + context_len:i + context_len + horizon_len]
        if np.any(np.isnan(ctx)) or np.any(np.isnan(tgt)):
            continue
        contexts.append(ctx)
        targets.append(tgt)

    contexts = np.array(contexts, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    print(f"[数据] 训练样本: {len(contexts)} 个 "
          f"(context={context_len}, horizon={horizon_len}, stride={stride})")
    return contexts, targets


def normalize_per_sample(ctx):
    """逐样本归一化: (x - mean) / std, 用于 TimesFM 输入。"""
    mu = ctx.mean()
    sigma = ctx.std()
    if sigma < 1e-8:
        sigma = 1.0
    return (ctx - mu) / sigma, mu, sigma


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


def train_lora(model, contexts, targets, args):
    """LoRA 微调主循环。

    训练策略:
      - 冻结所有非 LoRA 参数, 只训练 LoRA 矩阵
      - TimesFM 的输入是 (1, context_len) 的浮点数组
      - 用 MSE loss 对齐预测值和目标值
      - 早停: 连续 patience 个 epoch 验证 loss 不降则停止
    """
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.model = model.model.to(device)
    model.model.eval()  # 部分层有 dropout, eval 模式更稳定

    # 数据划分: 80% 训练, 20% 验证
    n = len(contexts)
    n_train = int(0.8 * n)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_dataset = LoRATimeSeriesDataset(contexts[train_idx], targets[train_idx])
    val_dataset = LoRATimeSeriesDataset(contexts[val_idx], targets[val_idx])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        pin_memory=True, num_workers=0)

    # 只训练 LoRA 参数
    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
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

    for epoch in range(1, args.epochs + 1):
        # ── 训练 ──
        model.model.train()
        train_loss_sum = 0.0
        n_train_batches = 0

        for ctx_batch, tgt_batch in train_loader:
            ctx_batch = ctx_batch.to(device)    # (B, 168)
            tgt_batch = tgt_batch.to(device)    # (B, 24)

            optimizer.zero_grad(set_to_none=True)

            # TimesFM forecast: 输入 (B, 168), 输出 (B, 24)
            # forecast 方法期望 list of arrays, 逐样本处理
            # 为加速, 用 batch 方式: 对每个样本调用 forecast, 收集输出
            batch_preds = []
            for i in range(len(ctx_batch)):
                ctx_np = ctx_batch[i].cpu().numpy().astype(np.float64)
                point_forecast, _ = model.forecast(
                    horizon=HORIZON_LEN, inputs=[ctx_np])
                pred = torch.tensor(point_forecast[0], dtype=torch.float32,
                                    device=device)
                batch_preds.append(pred)

            pred_all = torch.stack(batch_preds)  # (B, 24)

            # MSE loss
            loss = nn.functional.mse_loss(pred_all, tgt_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1

        train_loss = train_loss_sum / max(n_train_batches, 1)
        scheduler.step()

        # ── 验证 ──
        model.model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for ctx_batch, tgt_batch in val_loader:
                ctx_batch = ctx_batch.to(device)
                tgt_batch = tgt_batch.to(device)

                batch_preds = []
                for i in range(len(ctx_batch)):
                    ctx_np = ctx_batch[i].cpu().numpy().astype(np.float64)
                    point_forecast, _ = model.forecast(
                        horizon=HORIZON_LEN, inputs=[ctx_np])
                    pred = torch.tensor(point_forecast[0], dtype=torch.float32,
                                        device=device)
                    batch_preds.append(pred)

                pred_all = torch.stack(batch_preds)
                val_loss = nn.functional.mse_loss(pred_all, tgt_batch)
                val_loss_sum += val_loss.item()
                n_val_batches += 1

        val_loss = val_loss_sum / max(n_val_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "lr": lr_now})

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"lr={lr_now:.6f}"
              f"{'  ← best' if val_loss < best_val_loss else ''}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.model.state_dict())
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
            # 去掉前缀 "model." 以匹配原始模型路径
            clean_name = name.replace("model.model.", "model.")
            lora_state[clean_name] = param.data.cpu()

    # 保存
    torch.save(lora_state, os.path.join(out_dir, "lora_weights.pth"))

    config = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["qkv_proj", "out"],
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
    p = argparse.ArgumentParser(description="TimesFM 2.5 LoRA 微调 (水厂流量)")
    p.add_argument("--data", default=DEFAULT_DATA, help="原始数据 CSV")
    p.add_argument("--timesfm_model", default=DEFAULT_TIMESFM_MODEL,
                   help="TimesFM 预训练模型目录")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="输出目录")
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA 秩 (默认 8)")
    p.add_argument("--lora_alpha", type=float, default=16.0,
                   help="LoRA 缩放因子 (默认 16)")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率 (默认 3e-4)")
    p.add_argument("--epochs", type=int, default=30, help="最大训练轮数")
    p.add_argument("--batch_size", type=int, default=4, help="批大小")
    p.add_argument("--patience", type=int, default=5, help="早停耐心值")
    p.add_argument("--stride", type=int, default=1, help="滑动窗口步长")
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
    print(f"\n{'='*70}\n TimesFM LoRA 微调\n{'='*70}")
    hourly = load_flow_series(args.data)
    contexts, targets = make_lora_training_samples(hourly, stride=args.stride)

    # ── 加载 TimesFM ──
    print(f"\n[模型] 加载 TimesFM: {args.timesfm_model}")
    import timesfm
    model_obj = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        args.timesfm_model, torch_compile=False)  # torch_compile 会打断梯度, fine-tuning 时关闭
    model_obj.compile(timesfm.ForecastConfig(
        max_context=16256, max_horizon=128, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True,
        infer_is_positive=True, fix_quantile_crossing=True))
    print("[模型] TimesFM 加载完成 (torch_compile=OFF)")

    # ── 应用 LoRA ──
    print(f"\n[LoRA] rank={args.lora_rank}, alpha={args.lora_alpha}")
    model_obj, n_trainable = apply_lora(
        model_obj, rank=args.lora_rank, alpha=args.lora_alpha)

    # ── 训练 ──
    # decode() 内部有 torch.no_grad(), 会阻止梯度回传到 LoRA 层
    # 训练期间重写 decode 移除 no_grad, 训练后恢复
    restore_no_grad = patch_decode_no_grad(model_obj)
    print(f"\n[训练] 开始 LoRA 微调 (decode torch.no_grad 已移除)...")
    t0 = time.time()
    best_state, history, best_val_loss = train_lora(
        model_obj, contexts, targets, args)
    t1 = time.time()
    restore_no_grad()  # 恢复 torch.no_grad
    print(f"\n[训练] 完成, 耗时 {t1-t0:.1f}s, 最佳验证 loss = {best_val_loss:.6f}")

    # ── 保存 ──
    if best_state is not None:
        model_obj.model.load_state_dict(best_state)
    save_lora_weights(model_obj, args.out_dir, args, history, best_val_loss)

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

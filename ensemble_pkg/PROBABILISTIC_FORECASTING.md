# 给 TimesFM 集成预测器加概率预测（区间）

> 方法整理，用于移植到其他项目。
> 本项目（军山）的具体数字仅作参考，**可复用的部分是设计原则、代码骨架和踩坑清单**。

---

## 0. 一句话方法

**在不动点预测的前提下，给融合结果加一层分位数：**

```
TimesFM 侧：原生分位数（同一次 forward 就有，零成本）
确定性模型侧：回测残差校准出的经验分位数（中位数中心化）
融合：q_E(τ) = α·q_T(τ) + (1-α)·q_F(τ)        ← 分位数加权平均（Vincentization）
不变量：q_E(0.5) ≡ 原点预测，逐值精确相等
```

---

## 1. 先搞清楚 TimesFM 2.5 的分位数输出

这是整个方案的基础，**也是当年藏了一个大 bug 的地方**。

`TimesFm2_5ModelForPrediction` 在一次 forward 里同时输出 `mean_predictions` 和
`full_predictions`，形状 `[batch, horizon, num_quantiles]`：

| 索引 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 含义 | **point 头** | q0.1 | q0.2 | q0.3 | q0.4 | **q0.5** | q0.6 | q0.7 | q0.8 | q0.9 |

`num_quantiles = len(config.quantiles) + 1`，`config.quantiles` 见模型目录的 `config.json`。

四个必须知道的事实：

1. **索引 0 是独立的 point 头，可能跑到区间外面**，所以取分位数要用 `[:, 1:]`。
   实测过：`[2885.8, 2576.7, 2692.6, ...]` —— 索引 0 比 q0.1 还大。用了它就会破坏单调性。
2. **`config.decode_index = 5`，所以 `mean_predictions ≡ full_predictions[:, :, 5]`**。
   也就是说 TimesFM "原本的点预测"本来就是它的**中位数**，不是均值。
   这一点很关键：它让 "中位数 = 点预测" 这个不变量在 TimesFM 侧天然成立。
3. 索引 1..9 **严格单调递增**（实测每小时都成立），但不要假设，仍要校验。
4. 取分位数**不需要额外推理**。别为了拿区间再跑一次模型。

自检代码：

```python
assert torch.allclose(out.mean_predictions, out.full_predictions[:, :, 5])
q = out.full_predictions[:, :, 1:]          # 只要 1..9
assert np.all(np.diff(q, axis=-1) >= 0)
```

---

## 2. ⚠️ 最大的坑：输出对齐（offset）

**TimesFM 输出的第 0 步 = 上下文结束后的第一个小时。**

如果你的上下文不是停在 23:00，那么 `[0:24]` 拿到的**不是目标天的 00:00~23:00**，
而是从"上下文末尾 +1h"开始的 24 小时。

### 事故现场

生产口径是"每天 16 点做次日全天预测"，输入数据停在**前一天 15:00**。此时：

- TimesFM 第 0 步 = 前一天 **16:00**
- 目标天 00:00 落在第 **8** 步
- 而代码取了 `[0:24]` → **整体错位 8 小时**

实测后果（同一目标天）：

| 取法 | 对应时刻 | MAE | MAPE |
|---|---|---|---|
| `[0:24]`（错） | 前一天 16:00 起 | 1500.9 | **49.5%** |
| `[8:32]`（对） | 目标天 00:00 起 | 246.6 | **5.8%** |

融合后的 MAPE 从 **33% 掉到 6.3%**（26 天平均）。而且因为错位数据看起来很"平滑"，
**它不会报错，只会静默地把预测质量打烂**，非常难发现。

### 通用公式

```python
def day_offset(last_ctx_hour):
    """上下文末尾在 H 点 → 目标天 00:00 落在 TimesFM 的第几步。"""
    return (23 - int(last_ctx_hour)) % 24
```

| 上下文末尾 | 0 | 15 | 16 | 22 | 23 |
|---|---|---|---|---|---|
| offset | 23 | **8** | 7 | 1 | 0 |

```python
offset = day_offset(ctx.index[-1].hour)
stop = offset + DAY_STEPS
assert stop <= outputs.mean_predictions.shape[1], "horizon 不够"
point = outputs.mean_predictions[0, offset:stop]
q     = outputs.full_predictions[0, offset:stop, 1:]
```

**要点：点和分位数必须用同一个 offset 切，否则区间和点预测对不上。**

### 双重保险

1. **回测和推理走同一条代码路径**。本项目原先回测里内联了一份 TimesFM 调用、
   推理里又有一份，两者对齐口径不同 —— 这正是 bug 能长期潜伏的原因。
   修完后回测直接调 `predictor.predict_timesfm_quantiles(window, target_date=day)`，
   对齐逻辑只有一份实现。
2. **回测的截止点必须等于生产的决策时刻**。本项目回测原先给模型喂到前一天
   23:00（比生产多 8 小时），指标虚高。加了 `decision_hour` 参数统一口径后：
   α 从 0.32 → 0.36，融合 MAE 153.97 → 156.57，**这才是诚实的数字**。

---

## 3. 确定性模型（Transformer 等）的区间从哪来

点预测模型没有分位数头。两种做法，**强烈推荐后者**：

| | MC Dropout | **回测残差校准（推荐）** |
|---|---|---|
| 抓的是什么 | 参数不确定性 | 真实的预测误差分布 |
| 成本 | 每次 rollout × N 次采样 | 0（回测时一次性算好） |
| 效果 | **区间严重偏窄**：误差主要来自噪声和模型误设，不是参数不确定性 | 按构造即校准 |

### 残差校准

```text
ĝ(τ, hour) = quantile_τ(resid[hour]) - quantile_0.5(resid[hour])
q_T(τ, hour) = pred_T(hour) + ĝ(τ, hour)
```

三个关键点：

1. **按 hour-of-day 分桶**。流量的误差有明显的昼夜异方差（本项目 80% 半宽夜间
   ±200、白天 ±170）。分桶才有这个信息。
2. **必须减中位数做中心化**。否则残差的系统性偏差会把中位数推离原点预测，
   "不改变点预测" 的要求就破了。
3. **桶内样本不足要回退全局池化**。45 天校准窗口每个小时只有 45 个样本，
   阈值设 60 会导致 24/24 小时全部回退，白分桶了。本项目最终用 `min_count=40`，
   并对相邻小时做循环 3 点平滑降噪（保序，中位数仍为 0）。

参考实现见 [`fit_weights.py` 的 `calibrate_residual_quantiles`](fit_weights.py)。

---

## 4. 融合方法

### Vincentization（推荐默认）

```
q_E(τ) = α · q_T(τ) + (1-α) · q_F(τ)
```

- 是现有点融合 `α·T + (1-α)·F` 的逐分位数直接推广
- **自动保持单调性**
- **中位数恒等于点预测**（两边中位数都等于各自点预测）
- 实测校准最贴名义覆盖率

### 线性池 / mixture（不推荐）

```
F_E = α·F_T + (1-α)·F_F    再对 level 数值求逆
```

看起来更"统计正确"（把"不知道哪个模型对"也算进不确定性），但有个致命问题：

**线性池的中位数 ≠ 两个中位数的加权平均。** 实测原始偏离可达 55~650 m³/h。
强制定回点预测后就**不自洽了**（区间的中位数不是这个分布的中位数）。
而且区间会宽到 2 倍以上。

本项目实测（valid 45 天）：

| 方法 | 80% 覆盖 | 80% 宽度 | CRPS↓ |
|---|---|---|---|
| 名义值 | 0.800 | — | — |
| TimesFM 原生 | 0.773 | 519.8 | 132.3 |
| Transformer 残差校准 | 0.802 | 585.6 | 135.9 |
| **Vincentization** | **0.813** | 543.5 | **126.4** |
| 直接校准融合值 | 0.823 | 590.9 | 128.1 |
| mixture | 0.814 | 544.9 | 126.6 |

**结论：两种都实现，让数据选。** 但先把 linear pool 的中位数问题想清楚再用。

---

## 5. 不变量：`q(0.5) ≡ 点预测`

这是"不改点预测"这个需求的**可测试化形式**。

### 问题

维护单调性时用 `np.maximum.accumulate`，它**会把中位数列往上抬**；
如果你随后再把中位数钉回点预测，就可能**破坏单调性**。
本项目就是被自己埋的断言抓到的：

```
AssertionError: TimesFM 原生区间 区间非单调
```

护栏裁剪 `np.clip` 也会破坏单调性，叠加"钉中位数"后问题更明显。

### 解法：`anchor_median` —— 左右两侧分别投影

```python
def anchor_median(q, median):
    """把中位数列钉到 median, 同时分别恢复左右两侧的单调性。"""
    q = np.array(q, dtype=np.float64, copy=True)
    m = MEDIAN_INDEX
    med = np.asarray(median, dtype=np.float64).reshape(-1, 1)

    q[:, m] = med[:, 0]
    q[:, :m]  = np.maximum.accumulate(np.minimum(q[:, :m], med), axis=-1)   # 左半 ≤ med 且单调
    q[:, m+1:] = np.maximum.accumulate(np.maximum(q[:, m+1:], med), axis=-1) # 右半 ≥ med 且单调
    return q
```

对 Vincentization 是**零扰动**（实测左/右 max 变化 = 0.00），对 mixture 才起作用。
用它替代"先 accumulate 再钉中位数"的写法。

### 其他会破坏单调性的地方

- `np.clip` 护栏裁剪 → 用 `clip_quantiles`（裁剪 + 恢复单调）
- `np.maximum(q, 0)` 物理下限 → **这个保序，安全**
- 所有区间在返回前做一次断言

---

## 6. 接口设计：新字段，不动老字段

```json
{
  "date": "2025-12-26",
  "provider": "junshan_ensemble",
  "unit": "m3/h",
  "interval_minutes": 60,
  "horizon": 24,
  "values": [...],                          // ← 点预测，逐字节不变

  "quantiles": {                            // ← 新增
    "levels": [0.1, 0.2, ..., 0.9],
    "values": [[24 个], ... 9 行 ...]        // values[level_idx][hour]
  },
  "interval": {                             // ← 新增，消费方最常要的
    "coverage": 0.8,
    "lower": [...],
    "upper": [...]
  },
  "uncertainty": {                          // ← 新增，诊断
    "method": "vincentization",
    "median_gap": 0.0,                      // 自检位：应恒为 0
    "components": {"transformer": "residual_calibration", "timesfm": "native"}
  }
}
```

**用一个开关控制**，`with_interval=False` 时返回的 dict 与加功能前**逐字段一致**，
老调用方零感知。这是能在生产环境安全上线的关键。

**验证方式**：从 git 取出改动前的实现，跑同样的输入，断言 `values` 列表逐值相等
（本项目对比 3 个不同时段的输入，全部 bit-identical）。

---

## 7. 评估指标（不做这个等于没做）

只说"我加了区间"没有意义，必须能回答"这个区间可信吗"。

```python
def interval_scores(y_true, q, levels):
    """返回 {"crps", "pinball", "levels": [{nominal, picp, mpiw, winkler, pinball}]}"""
```

| 指标 | 含义 | 目标 |
|---|---|---|
| **PICP** | 经验覆盖率 | 贴近名义覆盖率（贴对角线） |
| **MPIW** | 平均区间宽度 | 同覆盖率下越窄越好 |
| **Winkler / Interval Score**（主指标） | `(u-l) + (2/a)(l-y)·1{y<l} + (2/a)(y-u)·1{y>u}` | 越小越好，同时惩罚覆盖不足与区间过宽 |
| **Pinball loss** | 分位数损失，对 levels 平均 ×2 ≈ CRPS | 越小越好 |

### ⚠️ Winkler 的 α 是"误覆盖率"，不是名义覆盖率

这个坑真实发生过（两个项目都写过一遍，导致过一次错误结论）：

```python
# 错: a = 名义覆盖率 (0.80) -> 系数 2.5, 违反被低估 4 倍
IS = (hi - lo) + (2.0 / nominal) * ...
# 对: a = 误覆盖率 = 1 - 名义覆盖率 (0.20) -> 系数 10
IS = (hi - lo) + (2.0 / (1.0 - nominal)) * ...
```

为什么必须是 `2/误覆盖率`：这样 Winkler 才**精确等于**两个分位数损失之和的缩放

```
IS_a(l,u,y) = (2/a) · [ ρ_{a/2}(y,l) + ρ_{1-a/2}(y,u) ]      a = 误覆盖率
```

（已用 6 组算例校验，误差到机器精度。）用错常数会破坏这个恒等式，
系统性偏袒**更窄**的区间 —— 而"窄"往往正是错的那个方向。

> 实战教训：修正后，方案A 与方案B 的排名在两个项目里都翻转了。
> 打分规则写错不会报错，只会静默地给出反的结论 —— 一定要用恒等式验一遍。

**CRPS 的近似**：`CRPS = 2∫₀¹ ρ_τ(y, F⁻¹(τ)) dτ`，用 9 个 level 的算术平均 ×2 近似。
实测对 N(0,1) 偏高约 5%（尾部未采样），但各方法一致，用于横向比较没问题。

必出的两张图：

1. **校准曲线**：名义覆盖率 vs 经验覆盖率，加一条 `y=x` 对角线
2. **扇形图**：真值 + 中位数 + 多层区间带（如 80% / 60%）
3. 逐日图叠加区间带 —— 最直观，一眼能看出哪天预测不可靠

> 实测的一个正面例子：某天模型把昼夜相位预测反了，80% 区间只覆盖 6/24 点。
> **区间在这种时候会自己"报警"**，这正是它的价值。

---

## 8. 上手清单（移植到新项目）

- [ ] **确认版本**：TimesFM 2.x 的 `full_predictions` 才有分位数；1.0 的 API 不同
- [ ] **打印 `config.quantiles` 和 `config.decode_index`**，确认索引布局
- [ ] **验证 `mean_predictions == full_predictions[:, :, decode_index]`**
- [ ] **确定生产决策时刻**（数据停在几点？）→ 算出 `day_offset`
- [ ] **回测截止点改成同一个决策时刻**，重新拟合融合权重
- [ ] 加 `enforce_monotone` / `anchor_median` / `clip_quantiles`
- [ ] 实现 `fuse_quantiles`（先只做 Vincentization）
- [ ] 回测里同时存下带分位数模型的 `qf_*` 列
- [ ] 实现 `calibrate_residual_quantiles` 并产出 `quantile_calibration.json`
- [ ] 在**独立验证窗口**上评估 PICP / Winkler / CRPS，确认融合优于单模型
- [ ] 接口用 `with_interval` 开关，验证关闭时与改动前逐字段一致
- [ ] 把"不变量断言"写成测试，进 CI

---

## 9. 踩坑清单（按严重程度）

| # | 坑 | 后果 | 对策 |
|---|---|---|---|
| 1 | **输出对齐 offset** | 静默错位 8 小时，MAPE 33% | `day_offset(last_ctx_hour)`；回测与推理共用一条代码路径 |
| 2 | **回测口径比生产宽松** | 指标虚高、α 偏错 | `decision_hour` 参数统一口径 |
| 3 | **用 `full_predictions[:,:,0]`** | 破坏单调性 | 只用 `[:, 1:]` |
| 4 | **残差没做中位数中心化** | 中位数 ≠ 点预测 | 减去桶内中位数 |
| 5 | **单调性维护压掉中位数** | 中位数 ≠ 点预测 | 用 `anchor_median` 分侧投影 |
| 6 | 护栏裁剪破坏单调 | 断言失败 | `clip_quantiles` |
| 7 | **分桶阈值设太高** | 全部回退池化，丢失昼夜异方差 | 按校准窗口天数算：45 天 → `min_count=40` |
| 8 | mixture 的 median 问题 | 区间与点预测不自洽 | 默认 Vincentization；或先想清楚语义 |
| 9 | **LoRA 会移动分位数** | 校准表失效 | LoRA 改的是注意力层，会连带影响分位数头 → **重跑校准** |
| 10 | 校准口径与推理口径不一致 | 区间偏窄/偏宽 | 校准表里记 `decision_hour`，推理时比对并告警 |
| 11 | 校准残差含 NaN | 整桶被污染成 NaN | `np.isfinite` 过滤 |
| 12 | 上下文长度影响区间宽度 | 校准与推理不符 | 校准表记 `context_hours`；同一验证窗口实测 168h 上下文下 80% 宽度 493.9 / 覆盖 0.760，504h 时 519.8 / 0.773 |

---

## 10. 可复用代码骨架

直接抄这些函数（都是纯函数，无外部依赖，只用到 numpy）：

| 函数 | 位置 | 作用 |
|---|---|---|
| `enforce_monotone(q)` | `ensemble_predictor.py` | 分位轴单调化 |
| `clip_quantiles(q, lo, hi)` | 同上 | 裁剪 + 恢复单调 |
| `anchor_median(q, median)` | 同上 | 钉中位数 + 分侧投影（**核心**） |
| `fuse_quantiles(alpha, q_t, q_f, method)` | 同上 | Vincentization / 线性池 |
| `_mixture_invert(...)` | 同上 | 线性池的数值求逆 |
| `timesfm_day_offset(h)` | 同上 | **对齐 offset（最关键）** |
| `calibrate_residual_quantiles(...)` | `fit_weights.py` | 逐小时残差分位数校准 |
| `residual_quantile_table(...)` | 同上 | 把校准表套到点预测上 |
| `interval_scores(y_true, q, levels)` | 同上 | PICP / MPIW / Winkler / CRPS |
| `day_interval(day_data, alpha, calib)` | `ensemble_predictor.py` | 从回测行重建当日区间 |

最小依赖：`numpy`, `pandas`, `torch`, `transformers`（含 timesfm2_5）。

---

## 11. 参考：本项目的实测结果

**验证窗口 45 天 / 1080 点**（校准窗口完全不参与）：

| 方法 | 80% 覆盖 | 60% | 40% | 20% | 80% 宽度 | CRPS↓ |
|---|---|---|---|---|---|---|
| 名义值 | 0.800 | 0.600 | 0.400 | 0.200 | — | — |
| TimesFM 原生 | 0.773 | 0.581 | 0.410 | 0.217 | 519.8 | 132.3 |
| Transformer 残差校准 | 0.802 | 0.635 | 0.434 | 0.216 | 585.6 | 135.9 |
| **Vincentization** | **0.813** | 0.646 | 0.450 | 0.231 | 543.5 | **126.4** |
| 直接校准融合值 | 0.823 | 0.663 | 0.465 | 0.255 | 590.9 | 128.1 |

**生产口径端到端**（92 个真实输入文件，全部停在 15:00）：

```
MAPE    : 均值 6.36%   中位 6.04%   最差 11.70%
MAE     : 均值 193.1   中位 183.4   最差 383.4
80% 覆盖: 均值 0.783   (名义 0.800)
不变量 |q(0.5) - values| 最大 = 0.00e+00   (92/92)
```

覆盖率略低于名义值（0.783 vs 0.800）是因为校准表拟合在 10–11 月，
而这 92 天横跨 8–12 月 —— **跨 regime 使用时要重新校准**。

---

## 12. 重新校准的触发条件

出现以下任一情况，`quantile_calibration.json` 必须重跑：

- 换了融合权重 α（α 变了，融合残差分布就变了）
- 换了任何一个基模型（重训 Transformer / 启用或更新 LoRA）
- 改了 `CONTEXT_HOURS`
- 改了决策时刻（`decision_hour`）
- 数据分布发生明显漂移（实测覆盖率持续偏离名义值）

命令：

```bash
python fit_weights.py                    # 默认 decision_hour=15，生产口径
python fit_weights.py --decision_hour 23 # 若生产可用整个前一天
python test_interval.py                  # 不变量回归测试（秒级）
python eval_production.py                # 生产输入文件端到端体检（只读）
```
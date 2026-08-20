# -*- coding: utf-8 -*-
"""
每日泵开泵策略 — 策略库版 (Transformer 流量预测 + close7 策略库匹配 + 双向 DP)
=================================================

以 schedule_daily_transformer_pso.py 为基准, 把 ② 逐点 PSO 寻优替换为策略库查询:

  ① Transformer 流量预测 (inference_transformer.FlowPredictor)
       → 未来 48 个时间点 (30min × 24h) 的 总管流量 Total_Flow + 分时压力 Pressure
  ② 逐点策略库匹配 (代替 PSO / 暴力枚举)
       → 每个时间点按 (预测流量, 预测压力) 两个条件查询策略库
         strategy/close7_<日期>.parquet:
            流量 ∈ [Flow_Lower, Flow_Upper)
            压力 ∈ [Header_Press_Lower, Header_Press_Upper)
         (压力不做液位修正, 液位不参与筛选 — 液位是多少无所谓)
         策略库中同一 Pump_Group 在每个液位档各有一行 (id/频率/液位区间不同),
         匹配后按 Pump_Group 去重 (液位变体全部挂在该策略下), 每点只取前
         --topk 个泵组 (默认 50) 作为 DP 候选。
         每个 (时间点, Time_Level) 尽量覆盖全部液位档: 该泵组缺的档从同时间点
         全部匹配行 (放宽后, 含其他泵组) 中借该档的真实行 (泵组可与候选不同,
         保留库中真实 id); 池中也没有该档 → 跳过 (实在找不到就算, 不复制伪行)。
         无精确匹配时按总放宽距离 (压力/流量档偏移组合, 距离 1~3) 逐级放宽,
         并在控制台告警。
  ③ 全局策略选择 (双向动态规划)
       → 前向 DP + 后向 DP 求每个候选的"必经总代价" (最优前缀 + 自身节点
         + 最优后缀); 同一时间点各候选按必经代价升序排名 → Time_Level
         = 该时间点该策略是第几优解 (1 = 全局最优路径所用策略, 2 = 次优解, ...)
       → 节点代价 = −效率×w_eff + 运行大泵×w_large
                   + |预测流量 − 目标流量|×w_flow (目标流量 = 策略流量区间中点;
                     偏差越大惩罚越大, 即使该策略可保持泵组不变也不会被选)
         切换代价 = 泵状态翻转×w_state + 频率变化×w_freq (大泵 ×large_switch)
       → 前向 DP 首层计入初始泵组状态 (pump_run_hours.csv 中 run_hours>0 = 开,
         =0 = 关) 到首个时间点候选的切换代价, 只统计泵状态翻转 (初始频率未知,
         不统计频率差); 全局最优因此从"当前正在运行的泵组"出发
  ④ 输出
       - daily_pump_schedule_strategy_lib.csv            : DP 全局最优路径 (逐点)
       - daily_pump_schedule_strategy_lib_candidates.csv : 每点候选按液位展开 —
            每个 (策略, Time_Level) 一行一个液位变体 (含 Level_Lower/Upper),
            缺失液位档借入同时间点匹配池中其他泵组的真实行 (泵组可与候选不同,
            保留库中真实 id), 池中无该档则跳过, 故某 Time_Level 的行数
            ≤ 时间点 × 液位档数 (有匹配的时间点)
       - daily_pump_schedule_strategy_lib_blocks.csv     : 相同 (泵组,频率) 连续时段合并
       - strategy/close7_<日期>_scheduled.parquet        : 策略库副本 — 被选中的策略
            填上 Suggest_Runtime_Start / Suggest_Runtime_End / Time_Level:
            一行 = 一个 (策略, 时间点, 第N优解) 使用记录, 同一策略用于多个时间点
            时复制策略行到下一行 (每行只含一个建议开启/结束时间与一个最优解序号);
            原行 = 该策略 Time_Level 最小的那次使用, 其余使用依次复制到其下一行;
            未被选中的策略 Time_Level=0 → 1000 (标记为未使用);
            缺档借入的行 (其他泵组真实 id) 与选中策略同样回写, 使每个液位档
            都有对应可用策略行; 池中无该档时该 (时间点, Time_Level) 缺档
       - strategy/close7_<日期>_scheduled.csv            : 已用策略库 —
            只含 Time_Level < 1000 的策略行 (被选中的策略及其使用记录)

时段约定 (同基准脚本): 预测时刻 t 在窗口 [t−15min, t+15min) 内生效,
Suggest_Runtime_Start/End = 该窗口的 HHMM 整数 (如 00:30 点 → 0015/0045)。

连续运行时长约束 (同基准脚本, 策略库版简化): 读 pump_run_hours.csv → 其 run_hours
>0 即当前处于开启状态的泵, 作为 DP 初始状态计入首点切换代价 → 逐点模拟各泵连续
运行小时; 达 --max_run_hours 上限的点强制停该泵 (从该点候选里剔除含此泵的方案) 后
重解; 策略库本身已含替代方案, 无需现场合成; 该点所有候选都含此泵时放行并告警。
日末把各泵连续运行时长写回 pump_run_hours.csv, 下次运行自动累加。

注意:
  - 策略库候选均满足其条件网格内的流量/压力, 本身即为可行解, 无流量超差
    (偏差 ≤ 50 m³/h < 容差 100), 故 DP 无 w_viol 项 (--w_viol 仅保留接口兼容)
  - 效率为策略库 Eff 列 (小数), 节点代价按 ×100 折成百分数, 与基准脚本口径一致
  - 节点代价含流量偏差惩罚: |预测流量 − 目标流量| × w_flow (目标流量 = 策略
    流量区间中点); 精确匹配偏差 ≤ 半档 25 m³/h, 放宽匹配 (流量档偏移) 偏差更大
    → 惩罚更大, 即使该策略可保持泵组不变也不会被选
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 保证能从本目录导入待融合模块 (inference_transformer 在 transformer_pkg)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_pkg.inference_transformer import FlowPredictor, DEFAULT_DATA, DEFAULT_RESULT_DIR

# ============================================================================
# 默认参数
# ============================================================================
# STRATEGY_LIB_DEFAULT = "../docker/mswaterai/app/wswater/data/close7_20260810.parquet"
STRATEGY_LIB_DEFAULT = "close7_20260810.parquet"
DEFAULT_STRATEGY_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    STRATEGY_LIB_DEFAULT)
FREQ_MINUTES = 30               # 与 Transformer 训练一致的重采样频率
W_FLOW_DEFAULT = 0.3           # DP 节点代价: 流量偏差惩罚 (每 m³/h 偏差)
TOP_K_DEFAULT = 50             # 每时间点 DP 备选策略上限 (按 放宽距离/效率 排序取前 k 条)

N_PUMPS = 7                     # 泵总数 (P1~P7, 数组下标 0~6)
LARGE_PUMPS = (0, 2, 3, 5)      # 大泵: P1/P3/P4/P6
MAX_CONSECUTIVE_HOURS = 4 * 24.0  # 单泵连续运行上限 (h) = 4 天 × 24 h
RUN_HOURS_FILE = "pump_run_hours.csv"

# 策略库条件网格: 流量 50 m³/h 一档, 压力 0.01 MPa 一档 (液位不参与匹配)
FLOW_STEP = 50.0
PRESS_STEP = 0.01


# ============================================================================
# 泵连续运行时长追踪 (跨天累加, 中途停泵归 0) — 同基准脚本
# ============================================================================

def load_run_hours(path):
    """读取各泵连续运行时长记录 (小时) 与当前泵组状态.

    返回 (hours, states):
      hours  — (7,) 各泵已连续运行小时
      states — (7,) 初始泵组状态: run_hours > 0 → 1 (开), == 0 → 0 (关)
    无文件/格式错误 → 全 0 (状态全关)。
    """
    hours = np.zeros(N_PUMPS, dtype=float)
    if not os.path.exists(path):
        return hours, np.zeros(N_PUMPS, dtype=np.int64)
    try:
        df = pd.read_csv(path)
        for i in range(N_PUMPS):
            row = df[df["pump"] == f"P{i + 1}"]
            if len(row) > 0:
                hours[i] = max(float(row.iloc[0]["run_hours"]), 0.0)
    except Exception as e:
        print(f"[!] 读取连续运行时长文件失败 {path}: {e} → 按 0 处理")
    return hours, (hours > 0).astype(np.int64)


def save_run_hours(path, hours, as_of):
    """写入各泵连续运行时长 (小时), 供下次运行读取."""
    df = pd.DataFrame({
        "pump": [f"P{i + 1}" for i in range(N_PUMPS)],
        "run_hours": np.round(hours, 2),
        "as_of": as_of,
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")


def simulate_run_hours(states_seq, start_hours):
    """按调度逐点推进各泵连续运行时长: 运行 +30min, 停止归 0.

    返回 (end_hours, per_point_hours): end_hours — (7,) 日末各泵连续运行时长;
    per_point_hours — (T,7) 各时间点末的连续运行时长。
    """
    step = FREQ_MINUTES / 60.0
    hours = np.array(start_hours, dtype=float)
    per_point = []
    for states in states_seq:
        on = np.asarray(states) > 0
        hours = np.where(on, hours + step, 0.0)
        per_point.append(hours.copy())
    return hours, (np.array(per_point) if per_point else np.zeros((0, N_PUMPS)))


# ============================================================================
# ② 策略库加载 / 条件索引 / 逐点匹配
# ============================================================================

def _flow_key(flow):
    return int(flow // FLOW_STEP)


def _press_key(press):
    return int(press * 100.0 + 1e-6)


# 放宽匹配的 (压力, 流量) 相对档位组合, 按总放宽距离 0~3 分组 (液位不参与匹配)
_RELAX_COMBOS = [[] for _ in range(4)]
for _dkp in range(-3, 4):
    for _dkf in range(-3, 4):
        _d = abs(_dkp) + abs(_dkf)
        if _d < 4:
            _RELAX_COMBOS[_d].append((_dkp, _dkf))


def load_strategy_library(path):
    """读取策略库 parquet, 返回 (lib, index):
      lib   — 原始 DataFrame (保留全部列)
      index — {(流量档, 压力档): 行号 ndarray} 条件网格索引
    """
    lib = pd.read_parquet(path)
    kf = np.round(lib["Flow_Lower"].to_numpy() / FLOW_STEP).astype(np.int64)
    kp = np.round(lib["Header_Press_Lower"].to_numpy() * 100.0).astype(np.int64)
    index = {}
    for i, (a, b) in enumerate(zip(kf, kp)):
        index.setdefault((int(a), int(b)), []).append(i)
    index = {k: np.asarray(v, dtype=np.int64) for k, v in index.items()}
    return lib, index


def lookup_candidates(lib, index, flow, press):
    """在策略库中匹配 (流量, 压力) 的备选策略, 按 (放宽距离, Eff 降序)
    排序, 匹配到多少返回多少 (不设上限). 无精确匹配时按总放宽距离逐级放宽
    (压力/流量档偏移组合, 距离 1~3)。

    返回 (rows, relax): rows — 匹配到的全部策略 DataFrame (可能为空);
    relax — 0=精确匹配, >0=放宽的档位距离 (无匹配时为 None)。
    """
    kf0, kp0 = _flow_key(flow), _press_key(press)
    found = {}          # 行号 → 放宽距离
    relax = None
    for dist in range(4):
        for dkp, dkf in _RELAX_COMBOS[dist]:
            key = (kf0 + dkf, kp0 + dkp)
            for i in index.get(key, ()):
                found.setdefault(int(i), dist)
        if found:
            relax = dist
            break
    if not found:
        return None, None
    pos = list(found.keys())
    rows = lib.iloc[pos].copy()
    rows["_dist"] = [found[p] for p in pos]
    rows = rows.sort_values(["_dist", "Eff"], ascending=[True, False])
    relax = int(rows["_dist"].iloc[0]) if len(rows) > 0 else None
    rows = rows.drop(columns="_dist")
    return rows, relax


def _pad_level_vars(grp, all_levels, pool=None):
    """把候选泵组的液位变体行扩展到全部液位档 (缺档从池中借其他泵组真实行).

    业务约定: 液位不参与筛选, 输出每个 (时间点, Time_Level) 尽量覆盖全部液位档;
    该泵组缺的档从 pool (同时间点全部匹配行, 含其他泵组) 中借该档的真实行
    (pool 已按 放宽距离/Eff 排序, 取该档第一条), 泵组可与候选不同, 保留库中
    真实 id; pool 中也没有该档 → 跳过 (实在找不到就算), 不生成复制伪行。
    """
    if pool is None:
        pool = grp
    have = set(float(v) for v in grp["Liquid_Level_1_Lower"].to_numpy())
    missing = [float(L) for L in all_levels if float(L) not in have]
    if not missing:
        return grp
    pads = []
    for L in missing:
        seg = pool[pool["Liquid_Level_1_Lower"].astype(float) == L]
        if len(seg) == 0:
            continue
        pads.append(seg.iloc[0].copy())      # 借该档最优真实行 (保留真实 id)
    if not pads:
        return grp
    out = pd.concat([grp, pd.DataFrame(pads)], ignore_index=True)
    return out.sort_values("Liquid_Level_1_Lower")


def rows_to_cands(rows, flow, all_levels):
    """策略库行 → DP 候选 dict, 按 Pump_Group 去重 (液位不参与筛选).

    rows 已按 (放宽距离, Eff 降序) 排序; 同一 Pump_Group 在多个液位档各有一行
    (id/液位区间不同), 去重后每泵组只留一行作代表 (该组排序后第一行), 液位
    变体 DataFrame 挂在 "level_vars" 上 (缺失档经 _pad_level_vars 从 rows 池借
    其他泵组真实行, 池中无该档则跳过), 输出时按 (策略, Time_Level) 展开。

    flow — 该时间点预测总管流量; 与目标流量 (流量区间中点) 一起用于节点代价的
    流量偏差惩罚。
    all_levels — 策略库全部液位档 (np.ndarray, 用于判断缺失档)。
    """
    cands = []
    pool = rows                  # 该时间点全部匹配行: 缺档借行来源 (池中也没有就算了)
    for pg, grp in rows.groupby("Pump_Group", sort=False):
        r = grp.iloc[0]        # 组内第一行 = (放宽距离最小, 效率最高) 的代表行
        states = np.array([int(ch) for ch in r["Pump_Group"]], dtype=np.int64)
        freqs = np.array([float(r[f"P{i}_Freq"]) for i in range(1, 8)], dtype=float)
        cands.append({
            "id": int(r["id"]),
            "Pump_Group": str(r["Pump_Group"]),
            "states": states,
            "freqs": freqs,
            "Eff": float(r["Eff"]),
            "PCTW": float(r["PCTW"]),
            "flow_lower": float(r["Flow_Lower"]), "flow_upper": float(r["Flow_Upper"]),
            "press_lower": float(r["Header_Press_Lower"]),
            "press_upper": float(r["Header_Press_Upper"]),
            "target_flow": float((r["Flow_Lower"] + r["Flow_Upper"]) / 2.0),
            "flow": float(flow),
            "own_ids": set(int(x) for x in grp["id"].to_numpy()),   # 组内真实行 id
            "level_vars": _pad_level_vars(grp, all_levels, pool),   # 缺档从池借真实行
        })
    return cands


# ============================================================================
# ③ 全局策略选择 (双向动态规划): 全局最优路径 + 每点候选 Time_Level
# ============================================================================

def transition_cost(prev_cand, curr_cand, w_state, w_freq, size_w=None):
    """相邻两点候选之间的切换代价 (同基准脚本):
    - 泵状态切换: 每台泵 开/关 翻转计 w_state × size_w[i]
    - 频率切换:   仅统计前后两点都在运行的泵, 频率差值 (Hz) 计 w_freq × size_w[i]
    """
    if size_w is None:
        size_w = np.ones(N_PUMPS)
    s1, s2 = prev_cand["states"], curr_cand["states"]
    f1, f2 = prev_cand["freqs"], curr_cand["freqs"]
    state_cost = w_state * float(np.sum((s1 != s2) * size_w))
    both_on = (s1 == 1) & (s2 == 1)
    freq_cost = w_freq * float(np.sum(np.abs(f1 - f2)[both_on] * size_w[both_on]))
    return state_cost + freq_cost


def _node_cost(c, w_eff, w_large, w_flow=0.0):
    """候选节点代价: −效率(%/100 折算)×w_eff + 运行大泵台数×w_large
    + 流量偏差惩罚 |预测流量 − 目标流量|×w_flow (目标流量 = 流量区间中点).
    精确匹配偏差 ≤ 半档 25 m³/h; 放宽匹配 (流量档偏移) 偏差更大 → 惩罚更大,
    即使该策略泵组不需切换也不会被选。
    """
    eff_pct = float(c["Eff"]) * 100.0
    n_large = int(np.sum(c["states"][list(LARGE_PUMPS)]))
    dev = abs(float(c["flow"]) - float(c["target_flow"]))
    return -w_eff * eff_pct + w_large * n_large + w_flow * dev


def dp_rank_and_path(cands_per_point, active, w_state, w_freq, w_eff, w_large,
                     size_w, forbidden, init_state=None, w_flow=0.0):
    """双向 DP 求解 (在 given 禁开约束下):

      - 前向 DP: 每个候选的最优前缀代价 (含自身节点; 首层另加初始状态切换代价)
      - 后向 DP: 每个候选的最优后缀代价 (含自身节点)
      - 必经总代价 = 前缀 + 后缀 − 自身节点 (避免重复计一次节点代价)

    返回 (ranks, chosen, order, total_cost):
      ranks — 每点 ndarray: 候选下标 → Time_Level (1=全局最优路径所用, 2=次优解…);
              被禁开约束剔除的候选 → 0 (不参与排名, 也不回写策略库)
      chosen — 每点全局最优路径选中的候选下标 (原始下标; 无候选的点为 None)
      order — 每点候选下标按 Time_Level 升序排列 (用于输出次优解路径展示)
      total_cost — 全局最优路径总代价
    forbidden: {(t, pump)} — 时间点 t 禁止开启泵 pump (连续运行上限的强制停泵)。
    某点候选被全部过滤 → 忽略该点的禁开约束, 避免无解。
    init_state: (7,) 调度开始前各泵状态 (1=开, 0=关); 非 None 时前向 DP 首层
    计入从该状态到首点候选的切换代价。初始频率未知, 只统计泵状态翻转 (w_state),
    不统计频率差 (w_freq)。
    w_flow: 节点代价的流量偏差惩罚权重 (每 m³/h), 见 _node_cost。
    """
    T = len(cands_per_point)
    sub, orig = [], []          # 只对 active 时间点求解
    for t in active:
        lst = cands_per_point[t]
        idx = list(range(len(lst)))
        f_here = [i for i in range(N_PUMPS) if (t, i) in forbidden]
        if f_here:
            keep = [j for j in idx
                    if not any(lst[j]["states"][i] == 1 for i in f_here)]
            if keep:
                lst = [lst[j] for j in keep]
                idx = keep
        sub.append(lst)
        orig.append(idx)

    m = len(sub)
    if m == 0:
        return ([None] * T, [None] * T, [None] * T, float("inf"))

    node = [[_node_cost(c, w_eff, w_large, w_flow) for c in lst] for lst in sub]

    # ── 前向 DP ──
    F, bk = [None] * m, [None] * m
    for i in range(m):
        n = len(sub[i])
        if i == 0:
            F[i] = np.array(node[i], dtype=float)
            if init_state is not None:
                init = (np.asarray(init_state) > 0).astype(np.int64)
                # 初始状态 → 首点候选: 只计泵状态翻转, 初始频率未知不计频率差
                init_sw = np.array([np.sum((init != c["states"]) * size_w)
                                    for c in sub[i]], dtype=float)
                F[i] += w_state * init_sw
            bk[i] = np.full(n, -1, dtype=int)
            continue
        cur = np.full(n, np.inf)
        ptr = np.zeros(n, dtype=int)
        for j in range(n):
            best, best_k = np.inf, 0
            for k in range(len(sub[i - 1])):
                cost = (F[i - 1][k] + node[i][j]
                        + transition_cost(sub[i - 1][k], sub[i][j],
                                          w_state, w_freq, size_w))
                if cost < best:
                    best, best_k = cost, k
            cur[j], ptr[j] = best, best_k
        F[i], bk[i] = cur, ptr

    # ── 后向 DP ──
    B = [None] * m
    for i in range(m - 1, -1, -1):
        n = len(sub[i])
        if i == m - 1:
            B[i] = np.array(node[i], dtype=float)
            continue
        cur = np.full(n, np.inf)
        for j in range(n):
            best = np.inf
            for k in range(len(sub[i + 1])):
                cost = (node[i][j]
                        + transition_cost(sub[i][j], sub[i + 1][k],
                                          w_state, w_freq, size_w)
                        + B[i + 1][k])
                if cost < best:
                    best = cost
            cur[j] = best
        B[i] = cur

    # ── 必经总代价 → 每点 Time_Level 排名 ──
    ranks = [None] * T
    order = [None] * T
    chosen = [None] * T
    total_cost = float("inf")
    for i, t in enumerate(active):
        through = F[i] + B[i] - np.array(node[i])
        sorted_idx = np.argsort(through, kind="stable")
        rk = np.zeros(len(through), dtype=int)
        rk[sorted_idx] = np.arange(1, len(through) + 1)
        full = np.zeros(len(cands_per_point[t]), dtype=int)
        full[orig[i]] = rk            # 禁开被剔除的候选 → 0
        ranks[t] = full
        order[t] = [orig[i][j] for j in sorted_idx]

    # ── 全局最优路径回溯 ──
    last = int(np.argmin(F[m - 1]))
    total_cost = float(F[m - 1][last])
    j = last
    for i in range(m - 1, -1, -1):
        t = active[i]
        chosen[t] = orig[i][j]
        j = bk[i][j]
    return ranks, chosen, order, total_cost


def solve_schedule(cands_per_point, start_hours, max_hours,
                   w_state, w_freq, w_eff, w_large, large_switch,
                   init_state=None, w_flow=0.0):
    """连续运行时长上限 (默认 96h = 4×24h) 的迭代求解 + 双向 DP 排名.

    ① DP 选出全局路径 (首点计入 init_state 初始状态的切换代价) → 逐点模拟各泵
    连续时长 → 超限点禁开该泵 (剔除候选) 重解;
    ② 某点候选全部包含超限泵 → 放行 (relaxed) 告警, 不再对该泵禁开。
    返回 (ranks, chosen, enforced, relaxed, n_iter):
      ranks/chosen — 同 dp_rank_and_path
      enforced — 实际强制停泵的 [(t, pump)]
      relaxed — 放行的 [(t, pump)]
    init_state — (7,) 初始泵组状态 (1=开, 0=关), 透传给 dp_rank_and_path
    w_flow — 流量偏差惩罚权重 (每 m³/h), 透传给 dp_rank_and_path
    """
    size_w = np.ones(N_PUMPS, dtype=float)
    size_w[list(LARGE_PUMPS)] = large_switch
    active = [t for t in range(len(cands_per_point)) if cands_per_point[t]]

    forbidden = set()
    relaxed = set()
    enforced = []
    n_iter = 0
    ranks = chosen = None
    while True:
        n_iter += 1
        ranks, chosen, order, cost = dp_rank_and_path(
            cands_per_point, active, w_state, w_freq, w_eff, w_large,
            size_w, forbidden, init_state=init_state, w_flow=w_flow)
        if max_hours <= 0:
            break
        states_seq = [cands_per_point[t][chosen[t]]["states"] for t in active]
        _, per_point = simulate_run_hours(states_seq, start_hours)
        viol = None
        for i, t in enumerate(active):
            st = cands_per_point[t][chosen[t]]["states"]
            for p in range(N_PUMPS):
                if st[p] == 1 and per_point[i][p] >= max_hours - 1e-9:
                    viol = (t, p)
                    break
            if viol is not None:
                break
        if viol is None:
            break
        t, p = viol
        if (t, p) in relaxed:
            break   # 已确认无法规避 → 接受超限
        st = cands_per_point[t][chosen[t]]["states"]
        for p2 in range(N_PUMPS):
            if st[p2] == 1 and per_point[i][p2] >= max_hours - 1e-9:
                if any(c["states"][p2] == 0 for c in cands_per_point[t]):
                    forbidden.add((t, p2))
                    enforced.append((t, p2))
                else:
                    relaxed.add((t, p2))

    return ranks, chosen, enforced, list(relaxed), n_iter


def path_switch_metrics(cands_per_point, chosen, active=None):
    """统计选中路径: (泵切换总次数, 频率变化总量 Hz, 平均效率 %)."""
    if active is None:
        active = list(range(len(cands_per_point)))
    n_toggle, freq_delta, effs = 0, 0.0, []
    for i, t in enumerate(active):
        c = cands_per_point[t][chosen[t]]
        effs.append(float(c["Eff"]) * 100.0)
        if i > 0:
            p = cands_per_point[active[i - 1]][chosen[active[i - 1]]]
            n_toggle += int(np.sum(p["states"] != c["states"]))
            both = (p["states"] == 1) & (c["states"] == 1)
            freq_delta += float(np.sum(np.abs(p["freqs"] - c["freqs"])[both]))
    return n_toggle, freq_delta, float(np.mean(effs)) if effs else 0.0


def to_hhmm(ts):
    """时间戳 → HHMM 整数 (如 00:15 → 15, 23:45 → 2345)."""
    return int(ts.strftime("%H%M"))


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="每日泵开泵策略: Transformer 流量预测 + 策略库匹配 + 双向 DP")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="原始数据 CSV 路径 (与训练数据格式一致)")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR,
                        help="Transformer 训练结果目录 (含 best_seq2seq_model.pth 与 scaler.pkl)")
    parser.add_argument("--strategy_lib", default=DEFAULT_STRATEGY_LIB,
                        help="策略库 parquet 路径 (条件网格寻优结果)")
    parser.add_argument("--topk", type=int, default=TOP_K_DEFAULT,
                        help=f"每时间点 DP 备选泵组上限 (去重后按 放宽距离/效率 排序取前 k 个, "
                             f"默认 {TOP_K_DEFAULT}; ≤0 不限制)")
    parser.add_argument("--w_state", type=float, default=1.0,
                        help="DP: 泵状态翻转代价 (每台泵, 默认 1.0)")
    parser.add_argument("--w_freq", type=float, default=0.01,
                        help="DP: 频率变化代价 (每 Hz, 仅统计连续运行的泵, 默认 0.01)")
    parser.add_argument("--w_eff", type=float, default=1.0,
                        help="DP: 效率权重 (每 %% 效率, 默认 1.0; 策略库候选无流量超差, w_viol 无效)")
    parser.add_argument("--w_viol", type=float, default=1.0,
                        help="(保留接口兼容) 策略库候选均满足流量条件, 无超差, 恒为 0")
    parser.add_argument("--w_large", type=float, default=0.5,
                        help="DP: 运行大泵偏好惩罚 (每台大泵/每点, 默认 0.5)")
    parser.add_argument("--w_flow", type=float, default=W_FLOW_DEFAULT,
                        help=f"DP: 流量偏差惩罚 (每 m³/h, 默认 {W_FLOW_DEFAULT}; "
                             "目标流量 = 策略流量区间中点, 偏差越大越不选)")
    parser.add_argument("--large_switch", type=float, default=2.0,
                        help="DP: 大泵状态翻转/频率变化代价倍数 (相对小泵, 默认 2.0)")
    parser.add_argument("--max_run_hours", type=float, default=MAX_CONSECUTIVE_HOURS,
                        help=f"单泵连续运行上限 (h, 默认 {MAX_CONSECUTIVE_HOURS:.0f} = 4×24h; "
                             "≤0 不限制; 记录文件 pump_run_hours.csv 自动读写)")
    parser.add_argument("--out", default="daily_pump_schedule_strategy_lib.csv",
                        help="逐点策略 CSV 输出路径")
    parser.add_argument("--lib_out", default=None,
                        help="策略库回写 parquet 输出路径 (默认 输入文件名 _scheduled.parquet)")
    args = parser.parse_args()

    # ── ① Transformer 流量预测 ──
    print("\n" + "=" * 72)
    print("① Transformer 流量预测 (inference_transformer.FlowPredictor)")
    print("=" * 72)
    predictor = FlowPredictor(args.result_dir)
    pred = predictor.predict(args.data)          # DataFrame: Total_Flow (48 行)
    pred = predictor.predict_pressure(pred)      # + Pressure 列 (分时压力目标)
    n_points = len(pred)
    print(f"   预测点数: {n_points}  ({pred.index[0]} ~ {pred.index[-1]})")

    # ── ② 策略库加载 + 逐点匹配 ──
    print("\n" + "=" * 72)
    print(f"② 策略库匹配 (条件网格查询, 按泵组去重后每点取前 {args.topk} 个, "
          "仅限流量/压力, 液位不参与筛选)")
    print("=" * 72)
    lib, index = load_strategy_library(args.strategy_lib)
    levels_all = np.sort(lib["Liquid_Level_1_Lower"].dropna().unique())
    print(f"   策略库: {args.strategy_lib}")
    print(f"   行数 {len(lib)}, 条件组合 "
          f"{len(index)} 个 (流量 {FLOW_STEP:.0f} 档 × 压力 {PRESS_STEP:.2f} 档)")
    print(f"   液位档 {len(levels_all)} 个 ({levels_all[0]:.1f}~{levels_all[-1]:.1f} m, "
          f"缺失档自动补齐)")

    cands_per_point = []          # 每点: [{id, states, freqs, Eff, PCTW, ...}, ...]
    match_info = []               # (匹配数, 放宽距离, 告警文本)
    for t, (ts, r) in enumerate(pred.iterrows()):
        flow = float(r["Total_Flow"])
        press = float(r["Pressure"])
        rows, relax = lookup_candidates(lib, index, flow, press)
        if rows is None or len(rows) == 0:
            cands_per_point.append([])
            match_info.append((0, None,
                               f"[{t + 1:3d}/{n_points}] {ts.strftime('%H:%M')}  "
                               f"流量 {flow:7.0f}  压力 {press:.3f}  "
                               f"无匹配策略 (放宽±3档仍无解)"))
            continue
        n_rows = len(rows)                     # 匹配到的行数 (含全部液位变体, 打印用)
        cands = rows_to_cands(rows, flow, levels_all)   # 按 Pump_Group 去重 + 液位补齐
        n_strat = len(cands)
        if args.topk > 0 and n_strat > args.topk:
            cands = cands[:args.topk]
            capped = f" (取前{args.topk}个泵组)"
        else:
            capped = ""
        cands_per_point.append(cands)
        best = cands[0]
        states_str = "".join(str(int(s)) for s in best["states"])
        freqs_str = " ".join(f"{f:g}" for f in best["freqs"])   # 保留小数 (库中频率精确到 0.01Hz)
        rel = "" if relax == 0 else f"  放宽{relax}档"
        match_info.append((n_rows, relax,
                           f"[{t + 1:3d}/{n_points}] {ts.strftime('%H:%M')}  "
                           f"流量 {flow:7.0f}  压力 {press:.3f}  "
                           f"匹配 {n_strat} 泵组 ({n_rows} 行){capped}  "
                           f"最优 {states_str} {freqs_str}  "
                           f"效率 {best['Eff'] * 100:.1f}%  "
                           f"目标差 {best['target_flow'] - flow:+.0f}{rel}"))
    for line in [m for _, _, m in match_info]:
        print("   " + line)
    n_match_pts = sum(1 for n, _, _ in match_info if n > 0)
    n_relax_pts = sum(1 for _, rel, _ in match_info if rel is not None and rel > 0)
    n_no_pts = n_points - n_match_pts
    print(f"   匹配 {n_match_pts}/{n_points} 点"
          + (f" (放宽 {n_relax_pts} 点)" if n_relax_pts else "")
          + (f", 无匹配 {n_no_pts} 点" if n_no_pts else ""))

    # ── ③ DP 全局选择 (含连续运行时长约束迭代) ──
    print("\n" + "=" * 72)
    print(f"③ 全局策略选择 (双向 DP): 泵状态翻转 {args.w_state}/台 + 频率变化 {args.w_freq}/Hz "
          f"- 效率 {args.w_eff}/% + 大泵偏好 {args.w_large}/台·点 "
          f"+ 流量偏差 {args.w_flow}/m³·h⁻¹ (目标流量=区间中点) "
          f"(大泵切换代价 ×{args.large_switch})")
    print("=" * 72)

    run_hours_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  RUN_HOURS_FILE)
    start_hours, init_state = load_run_hours(run_hours_path)
    print(f"   各泵已连续运行 (读取 {RUN_HOURS_FILE}): "
          + "  ".join(f"P{i + 1}={h:.1f}h" for i, h in enumerate(start_hours)))
    print("   初始泵组状态 (run_hours>0 → 开): "
          + "  ".join(f"P{i + 1}={('开' if s else '关')}"
                      for i, s in enumerate(init_state))
          + "   ← DP 首点计入从该状态的切换代价")
    print(f"   单泵连续运行上限: {args.max_run_hours:.0f} h"
          + (" (本次不限制)" if args.max_run_hours <= 0 else ""))

    ranks, chosen, enforced, relaxed, n_iter = solve_schedule(
        cands_per_point, start_hours, args.max_run_hours,
        args.w_state, args.w_freq, args.w_eff, args.w_large, args.large_switch,
        init_state=init_state, w_flow=args.w_flow)

    active = [t for t in range(n_points) if cands_per_point[t]]
    if n_iter > 1:
        print(f"   连续运行约束: 重解 {n_iter - 1} 次, 强制停泵 {len(enforced)} 处"
              + (f", 无法规避 {len(relaxed)} 处" if relaxed else ""))
    for t, p in enforced:
        print(f"   [限] P{p + 1} 在 {pred.index[t].strftime('%H:%M')} 强制停泵 "
              f"(连续运行将达 {args.max_run_hours:.0f}h 上限)")
    for t, p in relaxed:
        print(f"   [!] P{p + 1} 在 {pred.index[t].strftime('%H:%M')} 无法规避超限: "
              f"该点所有候选都包含此泵")

    # 对照基线: 逐点只选效率最高的候选 (不关心切换)
    baseline = [int(np.argmax([c["Eff"] for c in cands])) for cands in cands_per_point
                if len(cands) > 0]
    b_toggle, b_freq, b_eff = path_switch_metrics(cands_per_point, baseline, active)
    s_toggle, s_freq, s_eff = path_switch_metrics(cands_per_point, chosen, active)
    print(f"   逐点效率最高 (基线): 泵切换 {b_toggle} 次, 频率变化 {b_freq:.0f} Hz, "
          f"平均效率 {b_eff:.1f}%")
    print(f"   DP 全局最优        : 泵切换 {s_toggle} 次, 频率变化 {s_freq:.0f} Hz, "
          f"平均效率 {s_eff:.1f}%")
    print(f"   节省: 泵切换 {b_toggle - s_toggle} 次, 频率变化 {b_freq - s_freq:.0f} Hz")

    # 次优解展示: 每点取 Time_Level=r 的候选组成的路径
    print("   各时间点候选的 Time_Level 排名 (第 N 优解, 由必经总代价排序):")
    for r in (1, 2, 3):
        if not any(len(cands) >= 1 for cands in cands_per_point):
            break
        path_r = []
        for t in active:
            lst = ranks[t]
            # ranks[t] 为原始候选下标 → Time_Level 映射
            cands_t = cands_per_point[t]
            j = next((j for j in range(len(cands_t)) if lst[j] == r), None)
            path_r.append(j if j is not None else 0)
        r_toggle, r_freq, r_eff = path_switch_metrics(cands_per_point, path_r, active)
        print(f"   第 {r} 优解: 泵切换 {r_toggle} 次, 频率变化 {r_freq:.0f} Hz, "
              f"平均效率 {r_eff:.1f}%")

    # ── ④ 逐点结果 (DP 全局最优路径) + 全部候选 (含 Time_Level) ──
    half = FREQ_MINUTES // 2
    rows_out, cand_rows = [], []
    for t, (ts, r) in enumerate(pred.iterrows()):
        flow = float(r["Total_Flow"])
        press = float(r["Pressure"])
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        cands = cands_per_point[t]
        if not cands:
            rows_out.append({
                "timestamp": ts_str, "Total_Flow": round(flow, 1),
                "Pressure": round(press, 3), "strategy_id": np.nan,
                "states": "", "freqs": "", "efficiency": np.nan, "pctw": np.nan,
                "Time_Level": np.nan, "num_candidates": 0,
            })
            continue
        for j, c in enumerate(cands):
            states_str = "".join(str(int(s)) for s in c["states"])
            rep_freqs_str = " ".join(f"{f:g}" for f in c["freqs"])
            lv = int(ranks[t][j])
            # 液位不参与筛选: 每个 (策略, Time_Level) 展开全部液位变体行
            # (缺档已借入其他泵组真实行, 故泵组/频率/条件区间取该行自身值)
            for _, vr in c["level_vars"].iterrows():
                states_v = "".join(str(int(ch)) for ch in str(vr["Pump_Group"]))
                freqs_str = " ".join(f"{float(vr[f'P{i}_Freq']):g}" for i in range(1, 8))
                tgt = float(vr["Flow_Lower"] + vr["Flow_Upper"]) / 2.0
                cand_rows.append({
                    "timestamp": ts_str,
                    "strategy_id": int(vr["id"]),
                    "Pump_Group": str(vr["Pump_Group"]),
                    "Level_Lower": float(vr["Liquid_Level_1_Lower"]),
                    "Level_Upper": float(vr["Liquid_Level_1_Upper"]),
                    "rank_eff": j + 1,                 # 按效率降序的原始排名
                    "Time_Level": lv,                  # 双向 DP 第 N 优解 (0=被禁开剔除)
                    "selected": int(j == chosen[t]),
                    "states": states_v,
                    "freqs": freqs_str,
                    "efficiency": round(float(vr["Eff"]) * 100, 2),
                    "pctw": round(float(vr["PCTW"]), 2),
                    "flow_lower": float(vr["Flow_Lower"]),
                    "flow_upper": float(vr["Flow_Upper"]),
                    "target_flow": round(tgt, 1),
                    "flow_dev": round(float(flow) - tgt, 1),
                    "press_lower": float(vr["Header_Press_Lower"]),
                    "press_upper": float(vr["Header_Press_Upper"]),
                })
            if j == chosen[t]:
                rows_out.append({
                    "timestamp": ts_str,
                    "Total_Flow": round(flow, 1),
                    "Pressure": round(press, 3),
                    "strategy_id": c["id"],
                    "states": states_str,
                    "freqs": rep_freqs_str,
                    "efficiency": round(c["Eff"] * 100, 2),
                    "pctw": round(c["PCTW"], 2),
                    "Time_Level": lv,
                    "num_candidates": len(cands),
                })

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 逐点策略 (DP 全局最优) 已保存: {args.out} ({len(df_out)} 行)")

    cand_path = args.out.replace(".csv", "_candidates.csv")
    pd.DataFrame(cand_rows).to_csv(cand_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 全部候选 + Time_Level 已保存: {cand_path} ({len(cand_rows)} 行)")

    # ── ⑤ 连续时段合并 (相同泵组+频率, 同基准脚本) ──
    blocks = []
    has_sched = (len(df_out) > 0
                 and df_out["states"].astype(str).str.len().gt(0).any())
    if has_sched:
        sched_rows = df_out[df_out["states"].astype(str).str.len().gt(0)]
        first = sched_rows.iloc[0]
        cur_states, cur_freqs = first["states"], first["freqs"]
        cur_start = (pd.to_datetime(first["timestamp"])
                     - pd.Timedelta(minutes=half)).strftime("%Y-%m-%d %H:%M:%S")
        effs, pctws = [], []
        flows, pressures = [], []
        prev_ts = pd.to_datetime(first["timestamp"])

        def _close():
            blocks.append({
                "start": cur_start,
                "end": (prev_ts + pd.Timedelta(minutes=half)).strftime("%Y-%m-%d %H:%M:%S"),
                "num_points": len(effs),
                "flow_mean": round(float(np.mean(flows)), 1),
                "pressure_mean": round(float(np.mean(pressures)), 3),
                "states": cur_states, "freqs": cur_freqs,
                "efficiency_mean": round(float(np.mean(effs)), 2),
                "pctw_mean": round(float(np.mean(pctws)), 2),
                "all_feasible": True,
            })

        for _, row in sched_rows.iterrows():
            if row["states"] == cur_states and row["freqs"] == cur_freqs:
                effs.append(row["efficiency"]); pctws.append(row["pctw"])
                flows.append(row["Total_Flow"]); pressures.append(row["Pressure"])
                prev_ts = pd.to_datetime(row["timestamp"])
                continue
            _close()
            cur_states, cur_freqs = row["states"], row["freqs"]
            cur_start = (pd.to_datetime(row["timestamp"])
                         - pd.Timedelta(minutes=half)).strftime("%Y-%m-%d %H:%M:%S")
            effs, pctws = [row["efficiency"]], [row["pctw"]]
            flows, pressures = [row["Total_Flow"]], [row["Pressure"]]
            prev_ts = pd.to_datetime(row["timestamp"])
        if len(effs) > 0:
            _close()

    df_blocks = pd.DataFrame(blocks)
    if len(df_blocks) > 0:
        blocks_path = args.out.replace(".csv", "_blocks.csv")
        df_blocks.to_csv(blocks_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 连续时段合并已保存: {blocks_path} ({len(df_blocks)} 段)")

        print("\n" + "=" * 72)
        print("每日泵开泵策略 — 汇总 (按相同泵组合合并)")
        print("=" * 72)
        print(f"{'时段':<26}{'时长':>4}  {'流量':>8}{'压力':>6}  "
              f"{'泵状态':<10}{'频率':<22}{'效率':>6}{'千吨电耗':>8}")
        for b in df_blocks.itertuples(index=False):
            hh1 = b.start[11:16]; hh2 = b.end[11:16]
            dur = b.num_points * FREQ_MINUTES
            print(f"{hh1}-{hh2:<20}{dur:>4}min  {b.flow_mean:>8.1f}{b.pressure_mean:>6.2f}  "
                  f"{b.states:<10}{b.freqs:<22}"
                  f"{b.efficiency_mean:>6.1f}{b.pctw_mean:>8.1f}")

    # ── ⑥ 策略库回写: 选中策略填 Suggest_Runtime_Start/End + Time_Level ──
    lib_out = args.lib_out or os.path.join(
        os.path.dirname(args.strategy_lib),
        os.path.basename(args.strategy_lib).replace(".parquet", "_scheduled.parquet"))
    usage_records = []            # {id, t_ord, start, end, level} 库中真实行 (含借行)
    n_borrow = 0                  # 借入其他泵组真实行的使用次数 (补缺失液位档)
    for t in range(n_points):
        if not cands_per_point[t] or ranks[t] is None:
            continue
        for j, c in enumerate(cands_per_point[t]):
            lv = int(ranks[t][j])
            if lv < 1:
                continue         # 被连续运行约束剔除 → 不处理
            start_ts = pred.index[t] - pd.Timedelta(minutes=half)
            end_ts = pred.index[t] + pd.Timedelta(minutes=half)
            # 液位不参与筛选: 该 (策略, Time_Level) 的每个液位变体都回写
            # (缺档已借入其他泵组真实行, 池中无该档则跳过), 故 Time_Level=1 的
            # 行数 = 时间点数 × 覆盖到的液位档数
            for _, vr in c["level_vars"].iterrows():
                vid = int(vr["id"])
                usage_records.append({"id": vid, "t_ord": t,
                                      "start": to_hhmm(start_ts), "end": to_hhmm(end_ts),
                                      "level": lv})
                if vid not in c["own_ids"]:
                    n_borrow += 1

    if not usage_records:
        df_sched = lib.copy()
        df_sched["Time_Level"] = df_sched["Time_Level"].astype("float32")
        print(f"\n[!] 无被选中的策略, 策略库原样输出: {lib_out}")
    else:
        usages = pd.DataFrame(usage_records)
        # 原行 = Time_Level 最小的那次使用, 其余使用复制到下一行
        usages = usages.sort_values(["id", "level", "t_ord"]).reset_index(drop=True)
        is_first = ~usages["id"].duplicated(keep="first")
        first_use = usages[is_first]
        extra_use = usages[~is_first]

        pos_by_id = pd.Series(np.arange(len(lib)), index=lib["id"].to_numpy())
        first_pos = pos_by_id.loc[first_use["id"].to_numpy()].to_numpy()

        df_sched = lib.copy()
        col_s, col_e, col_l = (df_sched.columns.get_loc("Suggest_Runtime_Start"),
                               df_sched.columns.get_loc("Suggest_Runtime_End"),
                               df_sched.columns.get_loc("Time_Level"))
        df_sched.iloc[first_pos, col_s] = first_use["start"].to_numpy().astype("int32")
        df_sched.iloc[first_pos, col_e] = first_use["end"].to_numpy().astype("int32")
        df_sched.iloc[first_pos, col_l] = first_use["level"].to_numpy().astype("float32")

        n_copies = 0
        if len(extra_use) > 0:
            extra_pos = pos_by_id.loc[extra_use["id"].to_numpy()].to_numpy()
            copies = lib.iloc[extra_pos].copy()
            copies["Suggest_Runtime_Start"] = extra_use["start"].to_numpy().astype("int32")
            copies["Suggest_Runtime_End"] = extra_use["end"].to_numpy().astype("int32")
            copies["Time_Level"] = extra_use["level"].to_numpy().astype("float32")
            # 插入键: 原行位置 + 0.01/0.02/…, 使副本紧跟在原行下一行
            copies["_sort_key"] = extra_pos + (np.arange(len(extra_use)) + 1) / 100.0
            df_sched["_sort_key"] = np.arange(len(df_sched))
            df_sched = pd.concat([df_sched, copies])
            df_sched = (df_sched.sort_values("_sort_key")
                        .drop(columns="_sort_key").reset_index(drop=True))
            n_copies = len(extra_use)

        df_sched["Suggest_Runtime_Start"] = df_sched["Suggest_Runtime_Start"].astype("int32")
        df_sched["Suggest_Runtime_End"] = df_sched["Suggest_Runtime_End"].astype("int32")
        df_sched["Time_Level"] = df_sched["Time_Level"].astype("float32")

        n_sel_rows = len(first_use)
        print(f"     选中策略 {n_sel_rows} 条 (原行填写首次使用), "
              f"复制 {n_copies} 行 (同一策略多时间点/多优解使用), "
              f"总使用记录 {len(usages)} 条")

    if n_borrow > 0:
        print(f"     借入其他泵组真实行 {n_borrow} 次 补缺失液位档 "
              f"(池中无该档的液位档已跳过, 不生成伪行)")

    # ── ⑥b 回写输出: Time_Level=0 (未被选中) → 1000 存 parquet;
    #       Time_Level < 1000 的已用策略另存 CSV ──
    n_unused = int((df_sched["Time_Level"] == 0).sum())
    df_sched.loc[df_sched["Time_Level"] == 0, "Time_Level"] = 1000.0
    df_sched["Time_Level"] = df_sched["Time_Level"].astype("float32")
    df_sched.to_parquet(lib_out, index=False)
    print(f"\n[OK] 策略库已回写: {lib_out}")
    print(f"     Time_Level=0 的未选中策略 → 1000 ({n_unused} 条)")

    lib_used_path = lib_out.replace(".parquet", ".csv")
    used = df_sched[df_sched["Time_Level"] < 1000.0].copy()
    used.to_csv(lib_used_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已用策略库 (Time_Level < 1000) 已保存: {lib_used_path} ({len(used)} 行)")

    # ── ⑦ 各泵连续运行时长: 按最终调度逐点模拟 → 写入文件 ──
    states_seq = [cands_per_point[t][chosen[t]]["states"] for t in active]
    end_hours, _ = simulate_run_hours(states_seq, start_hours)
    save_run_hours(run_hours_path, end_hours,
                   pred.index[-1].strftime("%Y-%m-%d %H:%M"))
    print("=" * 72)
    print("各泵连续运行时长 (今日调度模拟, 停泵即归 0):")
    for i in range(N_PUMPS):
        flag = ""
        if args.max_run_hours > 0 and end_hours[i] >= args.max_run_hours - 1e-9:
            flag = "  [达到上限!]"
        print(f"  P{i + 1}: {start_hours[i]:6.1f} h → {end_hours[i]:6.1f} h{flag}")
    print(f"[OK] 已写入 {RUN_HOURS_FILE}, 下次运行自动读取")

    # ── ⑧ 汇总 ──
    n_sched = int(df_out["strategy_id"].notna().sum())
    print("\n" + "=" * 72)
    print(f"汇总: {n_points} 点, 有策略 {n_sched} 点"
          + (f", 无策略 {n_points - n_sched} 点" if n_points - n_sched else ""))
    n_r1 = sum(1 for t in range(n_points)
               if ranks[t] is not None and np.any(ranks[t] == 1))
    n_r2 = sum(1 for t in range(n_points)
               if ranks[t] is not None and np.any(ranks[t] == 2))
    n_r3 = sum(1 for t in range(n_points)
               if ranks[t] is not None and np.any(ranks[t] == 3))
    print(f"Time_Level 分布 (全部候选): 1 优解 {n_r1} 点, 2 优解 {n_r2} 点, "
          f"3 优解 {n_r3} 点")
    print("=" * 72)


if __name__ == "__main__":
    main()

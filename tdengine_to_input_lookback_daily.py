# -*- coding: utf-8 -*-
"""从 TDengine 数据库读取 PLC 点位历史数据, 生成 input_lookback.csv 同构的宽表 CSV。

与 tdengine_to_input_lookback.py 输出完全一致, 区别是内存优化版:
  - 每天一个查询窗口 (00:00~次日 00:00), 提取完当天立即追加写入完整 CSV 并释放内存
  - 泵状态列 (秒级) 单独累计, 窗口结束后统一统计连续运行时长
  - 只保留最近 LOOKBACK_OUT_DAYS 天的宽表, 用于截取输出
  峰值内存约降到整体版的 1/2 ~ 1/3。

数据库表为长表, 列结构与 yoyo/*.csv 导出一致:
    ts, data_poi_name, data_poi_value, dynamics_message, recv_time, entry_time, insert_time
本脚本只读取前三列 (ts, data_poi_name, data_poi_value)。

按 data_poi_name 语义映射到 input_lookback.csv 的列:
    total_pressure_set  -> 170:总管压力
    1~6_泵运行          -> 170:1~6_泵运行
    7_泵运行            -> 70:7_泵运行
不输出: 吸水井液位, flowtotal (在查询时已过滤)
三个分管瞬时流量列 (170:1_瞬时流量 / 170:2_瞬时流量 / 70:3_瞬时流量)
按当前名称直接映射; 库里暂时没有的数据列自动空值占位。

依赖: pip install taosrest pandas numpy
用法: 填好下方 CONFIG 后运行  python tdengine_to_input_lookback_daily.py
时间范围: start_time / end_time 留空时自动取当前时间之前的四个完整自然日
(今天 19 号 -> 15、16、17、18 号 00:00~00:00); 手动填写则覆盖。

输出:
  <输出名>_full.csv       — 完整导出 (逐日追加写入, 不占内存)
  <输出名>.csv            — 从前 4 天数据中截取后 3 天 (Transformer 推理输入, 旧文件名保持不变)
  pump_run_hours.csv      — 各泵连续运行时长 (统计完整 4 天; 中途停机归 0 重新累计)
"""

import os
import sys
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "input_lookback.csv")
DEFAULT_OUTPUT = os.path.join(HERE, "input_lookback_tdengine_test.csv")

# ============================================================================
# 连接配置 (按实际填写; 空值表示待填)
# ============================================================================
CONFIG = dict(
    host="192.168.8.5",            # TDengine 服务器 IP
    port=6041,          # REST 端口 6041 (native 驱动用 6030)
    user="root",            # 用户名 (默认 root)
    password="taosdata",        # 密码 (默认 taosdata)
    database="water_works",        # 数据库名
    table="dynamics_plc_log",     # 超级表名 (含全部点位; plc_ZGYL_0 子表只存压力)
    start_time="2026-02-17 00:00:00",      # 可选: 起始时间 "2026-08-16 00:00:00"; 留空 = 今天-3天 00:00
    end_time="2026-08-18 00:00:00",        # 可选: 结束时间 (左闭右开); 留空 = 今天 00:00
    # start_time="",      # 可选: 起始时间 "2026-08-16 00:00:00"; 留空 = 今天-3天 00:00
    # end_time="",        # 可选: 结束时间 (左闭右开); 留空 = 今天 00:00
)

# data_poi_name -> input_lookback.csv 列名 (语义对应; 不在映射内的 tag 一律丢弃)
TAG_MAP = {
    "total_pressure_set": "170:总管压力",   # 总管压力设定值
    "1_泵运行": "170:1_泵运行",
    "2_泵运行": "170:2_泵运行",
    "3_泵运行": "170:3_泵运行",
    "4_泵运行": "170:4_泵运行",
    "5_泵运行": "170:5_泵运行",
    "6_泵运行": "170:6_泵运行",
    "7_泵运行": "70:7_泵运行",
    "1_瞬时流量": "170:1_瞬时流量",
    "2_瞬时流量": "170:2_瞬时流量",
    "3_瞬时流量": "70:3_瞬时流量",
}

# ============================================================================
# 查询窗口与连续运行时长统计
# ============================================================================
LOOKBACK_DAYS = 4           # 查询窗口: 当前时间之前的 4 个完整自然日
LOOKBACK_OUT_DAYS = 3       # input_lookback 输出只保留前 4 天中的后 3 天
GAP_RESET_SECONDS = 60.0    # 泵状态数据缺失超过该间隔(秒) -> 视为停机, 连续时长归 0

RUN_HOURS_FILE = os.path.join(HERE, "pump_run_hours.csv")   # 输出: 各泵连续运行时长

# P1~P7 的运行状态列 (顺序即 P1..P7, 与 input_lookback.csv 列名一致)
PUMP_RUN_COLS = [TAG_MAP[f"{i}_泵运行"] for i in range(1, 8)]


def default_range():
    """默认查询窗口: 当前时间之前的 4 个完整自然日, 返回 (start_dt, end_dt).

    例: 今天是 19 号 -> 取 15、16、17、18 号全天数据 [15日00:00, 19日00:00)
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return today - timedelta(days=LOOKBACK_DAYS), today


def fetch_day(conn, start_dt, end_dt):
    """查询某一天 [start_dt, end_dt) 的映射 tag 数据, 返回长表 DataFrame (无数据返回 None)."""
    start_time = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    tags = list(TAG_MAP)
    in_list = ",".join(f"'{t.replace(chr(39), chr(39)+chr(39))}'" for t in tags)
    sql = (f"SELECT ts, data_poi_name, data_poi_value "
           f"FROM {CONFIG['database']}.{CONFIG['table']} "
           f"WHERE data_poi_name IN ({in_list})"
           f" AND ts >= '{start_time}' AND ts < '{end_time}'")
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    if not rows:
        return None
    return pd.DataFrame(rows, columns=cols)


def day_to_wide(df, template_cols, placeholder_printed):
    """单天长表 -> 模板列宽表 (与整体版逻辑一致).

    placeholder_printed: set, 记录已打印过"[占位]"提示的列, 避免每天重复打印。
    """
    df = df.copy()
    # tag -> 模板列名; 不在映射内的 (吸水井液位/flowtotal) 变 NaN 直接丢弃
    df["data_poi_name"] = df["data_poi_name"].map(TAG_MAP)
    df = df.dropna(subset=["data_poi_name"])

    # 同一 tag 同一 ts 有多条时取最后一条
    df = df.drop_duplicates(subset=["ts", "data_poi_name"], keep="last")

    # 长表转宽表
    wide = (df.pivot_table(index="ts", columns="data_poi_name",
                           values="data_poi_value", aggfunc="last")
              .reset_index())
    wide = wide.sort_values("ts").reset_index(drop=True)

    # 格式化 ts: 去掉毫秒, 与 input_lookback.csv 一致
    wide["ts"] = pd.to_datetime(wide["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 按模板列建表; 模板有但库里没有的列 -> 空占位 (只提示一次)
    out = pd.DataFrame(index=wide.index)
    out["timestamp"] = wide["ts"]
    for col in template_cols[1:]:
        if col in wide.columns:
            out[col] = pd.to_numeric(wide[col], errors="coerce")
        else:
            out[col] = pd.NA
            if col not in placeholder_printed:
                placeholder_printed.add(col)
                print(f"  [占位] 库中无数据: {col}")
    return out


def continuous_run_hours(ts, states):
    """统计单泵在窗口末的连续运行时长 (小时): 运行则累计, 停机/缺失归 0.

    ts: 时间戳序列 (datetime); states: 泵状态 (0/1), NaN 视为停机。
    与调度器 simulate_run_hours 语义一致: 持续运行累计, 中途停机重新计算。
    """
    ts_sec = pd.to_datetime(ts).astype("int64").to_numpy() // 10**9
    s = np.where(np.asarray(states, dtype=float) > 0.5, 1.0, 0.0)
    n = len(s)
    if n < 2:
        return 0.0
    dt = np.diff(ts_sec) / 3600.0             # 相邻采样间隔 (小时)
    gap = dt > GAP_RESET_SECONDS / 3600.0     # 数据缺失/间隔异常 -> 视为停机
    on = (s[:-1] > 0.5) & (s[1:] > 0.5) & ~gap  # 区间两端都在运行才累计
    # 累计 + 归零: 连续运行段内累加间隔, 停机后的下一个时刻归 0
    seg = np.where(on, dt, 0.0)
    c = np.concatenate([[0.0], np.cumsum(seg)])
    reset = np.zeros_like(c)
    off_idx = np.flatnonzero(~on) + 1
    reset[off_idx] = c[off_idx]
    run = c - np.maximum.accumulate(reset)
    return float(run[-1])


def save_run_hours(path, hours, as_of):
    """写入各泵连续运行时长 (pump, run_hours, as_of), 格式与调度器读写一致."""
    pd.DataFrame({
        "pump": [f"P{i + 1}" for i in range(len(PUMP_RUN_COLS))],
        "run_hours": np.round(hours, 2),
        "as_of": as_of,
    }).to_csv(path, index=False, encoding="utf-8-sig")


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT

    # 读取模板列 (保持与 input_lookback.csv 完全一致的列名与顺序)
    template = pd.read_csv(TEMPLATE, nrows=0)
    template_cols = list(template.columns)
    print("模板列:", template_cols)

    # 起止时间: CONFIG 留空时自动取当前时间之前的 4 个完整自然日
    start_dt, end_dt = default_range()
    if CONFIG["start_time"]:
        start_dt = datetime.strptime(CONFIG["start_time"], "%Y-%m-%d %H:%M:%S")
    if CONFIG["end_time"]:
        end_dt = datetime.strptime(CONFIG["end_time"], "%Y-%m-%d %H:%M:%S")
    print("时间范围:", start_dt, "->", end_dt)

    if not (CONFIG["host"] and CONFIG["database"] and CONFIG["table"]):
        sys.exit("请先在 CONFIG 中填写 host / database / table")

    from taosrest import connect

    conn = connect(url=f"http://{CONFIG['host']}:{CONFIG['port']}",
                   user=CONFIG["user"], password=CONFIG["password"])

    root, ext = os.path.splitext(out_path)
    full_path = root + "_full" + ext

    # 完整 CSV: 先写表头 (含 BOM) 一次, 之后每天追加数据行 (utf-8 不再带 BOM)
    pd.DataFrame(columns=template_cols).to_csv(
        full_path, index=False, encoding="utf-8-sig")
    print("已创建完整文件:", full_path)

    placeholder_printed = set()
    pump_acc = []                                  # 各天泵状态 (timestamp + 7 泵列), 窗口末统一统计
    tail = deque(maxlen=LOOKBACK_OUT_DAYS)         # 只保留最近 3 天宽表, 用于截取输出
    total_rows = 0

    day = start_dt
    while day < end_dt:
        day_end = day + timedelta(days=1)
        if day_end > end_dt:
            day_end = end_dt
        print(f"\n[{day.strftime('%Y-%m-%d')}] 查询 {day} ~ {day_end} ...")

        df = fetch_day(conn, day, day_end)
        if df is None:
            print("  [警告] 当天无数据, 跳过 (CSV 中该天为空缺)")
            day = day_end
            continue
        print("  读取行数:", len(df), " 范围:", df["ts"].min(), "->", df["ts"].max())

        wide = day_to_wide(df, template_cols, placeholder_printed)
        del df                                    # 尽早释放长表内存

        # 完整 CSV 追加当天 (header 已写过, 只追加数据行)
        wide.to_csv(full_path, mode="a", header=False, index=False, encoding="utf-8")
        total_rows += len(wide)
        print(f"  已追加 {len(wide)} 行 -> {os.path.basename(full_path)}")

        # 泵状态累计 (连续时长统计用; 只留 8 列, 约占当天宽表 2/3 内存)
        pump_acc.append(wide[["timestamp"] + PUMP_RUN_COLS].copy())

        # 截取输出: 只保留最近 LOOKBACK_OUT_DAYS 天
        tail.append(wide)
        del wide

        day = day_end

    conn.close()

    if total_rows == 0:
        sys.exit("查询结果为空, 请检查 CONFIG 与表内数据")

    # ── ① 各泵连续运行时长 (全窗口泵状态; 中途停机归 0 重新累计) ──
    pump_all = pd.concat(pump_acc, ignore_index=True)
    ts_series = pd.to_datetime(pump_all["timestamp"])
    hours = [continuous_run_hours(ts_series, pump_all[col]) for col in PUMP_RUN_COLS]
    as_of = end_dt.strftime("%Y-%m-%d %H:%M")
    save_run_hours(RUN_HOURS_FILE, hours, as_of)
    print("\n已写出:", RUN_HOURS_FILE, " (截至", as_of, ")")
    for i, col in enumerate(PUMP_RUN_COLS):
        print(f"  P{i + 1} ({col}) 连续运行 {hours[i]:.2f} h")

    # ── ② input_lookback 输出: 只保留最近 LOOKBACK_OUT_DAYS 天 (旧文件名不变) ──
    out = pd.concat(list(tail), ignore_index=True)
    print(f"截取后 {LOOKBACK_OUT_DAYS} 天: 保留 {len(out)} / {total_rows} 行")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("已写出:", out_path)
    print("完整文件:", full_path, f"({total_rows} 行)")
    print("列结构:", list(out.columns))
    for col in out.columns[1:]:
        n_null = out[col].isna().sum()
        if n_null == len(out):
            print(f"  {col:<20} 全部为空(占位)")
        else:
            print(f"  {col:<20} 非空 {len(out)-n_null:>7}  空 {n_null:>6}  "
                  f"min={out[col].min():>10}  max={out[col].max():>10}")


if __name__ == "__main__":
    main()

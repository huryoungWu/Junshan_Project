#!/usr/bin/env bash
# ============================================================================
# 每日调度任务 — 每天 00:00 (中国时间) 由容器内 cron 调用
#   ① tdengine_to_input_lookback.py       从 TDengine 拉前 4 天数据 → 推理输入
#   ② schedule_daily_strategy_library.py  Transformer 预测 + 策略库匹配 + DP → 当日策略
# 持久化: 挂载卷 /data — 推理输入 / 每日策略产物 / pump_run_hours.csv 状态都在这里,
#         容器重建不丢状态。镜像内置文件只作兜底。
# ============================================================================
set -u

APP_DIR=/app
DATA_DIR="${DATA_DIR:-/data}"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/daily_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$DATA_DIR" "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }
die() { log "[错误] $*"; exit 1; }

log "================ 每日策略调度开始 ================"

# ── ① 恢复上次的泵连续运行状态 (跨天累计, 容器重建不丢) ──
if [ -f "$DATA_DIR/pump_run_hours.csv" ]; then
    cp "$DATA_DIR/pump_run_hours.csv" "$APP_DIR/pump_run_hours.csv"
    log "已从卷恢复状态: pump_run_hours.csv"
fi

# ── ② 选择策略库: 优先卷里最新的 parquet (策略库每日更新时替换),
#       卷里没有则把镜像内置的复制进卷作为兜底 (产物与库放一起, 统一持久化) ──
# 优先卷里最早的 .parquet 文件（排除 _scheduled 结尾的）
STRAT_LIB="$(ls -t "$DATA_DIR"/*.parquet 2>/dev/null | grep -v '_scheduled\.parquet$' | sort | head -n 1)"

if [ -z "$STRAT_LIB" ]; then
    # 兜底：从镜像内置目录取同样规则的最早文件
    FALLBACK="$(ls "$APP_DIR"/*.parquet 2>/dev/null | grep -v '_scheduled\.parquet$' | sort | head -n 1)"
    if [ -z "$FALLBACK" ]; then
        die "卷和镜像内置目录均未找到可用的策略库 parquet 文件"
    fi
    cp "$FALLBACK" "$DATA_DIR/$(basename "$FALLBACK")"
    STRAT_LIB="$DATA_DIR/$(basename "$FALLBACK")"
    log "卷中无策略库, 已复制镜像内置库作兜底: $STRAT_LIB"
else
    log "使用策略库: $STRAT_LIB"
fi

# ── ③ 从 TDengine 拉数 (输入输出都在卷上; 取数失败则跳过调度, 避免用旧数据出方案) ──
cd "$APP_DIR"
log "① TDengine 取数: tdengine_to_input_lookback.py"
if ! python "$APP_DIR/tdengine_to_input_lookback.py" "$DATA_DIR/input_lookback_tdengine.csv" >>"$LOG_FILE" 2>&1; then
    tail -n 20 "$LOG_FILE" >&2
    die "TDengine 取数失败, 跳过今日调度 (详见 $LOG_FILE)"
fi
log "  取数完成 → $DATA_DIR/input_lookback_tdengine.csv"

# ── ④ 策略调度 (预测 + 策略库匹配 + 双向 DP) ──
log "② 策略调度: schedule_daily_strategy_library.py"
if ! python "$APP_DIR/schedule_daily_strategy_library.py" \
        --data "$DATA_DIR/input_lookback.csv" \
        --strategy_lib "$STRAT_LIB" \
        --out "$DATA_DIR/daily_pump_schedule_strategy_lib.csv" \
        >>"$LOG_FILE" 2>&1; then
    tail -n 20 "$LOG_FILE" >&2
    die "策略调度失败 (详见 $LOG_FILE)"
fi
log "  调度完成 → $DATA_DIR/daily_pump_schedule_strategy_lib*.csv"

# ── ⑤ 泵连续运行时长回写卷 (下次运行自动累加) ──
cp "$APP_DIR/pump_run_hours.csv" "$DATA_DIR/pump_run_hours.csv"
log "状态已回写: $DATA_DIR/pump_run_hours.csv"

# ── ⑥ 清理 30 天前的日志 ──
find "$LOG_DIR" -name 'daily_*.log' -mtime +30 -delete 2>/dev/null

log "================ 每日策略调度结束 (成功) ================"

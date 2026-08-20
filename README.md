# 策略库日调度项目(strategy/)

> 泵站每日泵开泵策略调度:从 TDengine 取数 → Transformer 流量预测 → 策略库匹配 + 双向 DP
> 全局选策 → 生成当日调度方案(调度版策略库)→ 转化为客户可用的 MSAI 推理文件。

## 1. 项目流水线

```
TDengine 历史数据(前 4 天 PLC 点位)
   │  ① tdengine_to_input_lookback.py      取数 → 推理输入 CSV + 泵连续运行时长
   ▼
input_lookback_tdengine.csv
   │  ② schedule_daily_strategy_library.py  Transformer 预测 + 策略库匹配 + 双向 DP
   ▼
close7_<日期>_scheduled.parquet(调度版策略库:静态库全量行 + 排班回写)
   │  ③ train_msai.py                      转化封装
   ▼
close7_<日期>_scheduled.msai(客户推理文件,joblib.load 后按数组查表)
```

- **静态策略库** `.parquet`:条件网格候选(流量 50 m³/h 一档、压力 0.01 MPa 一档、
  液位变体展开),无时间维度,`Time_Level` 全为 0、窗口全为 0–2359;
- **调度版策略库** `_scheduled.parquet`:被选中策略填 `Time_Level`(1–50,双向 DP 第 N 优解)
  与 `Suggest_Runtime_Start/End`(30 分钟时段,HHMM 整数,如 0245 = 02:45);未选中策略标记 `Time_Level=1000`
  哨兵(全天兜底)。物理参数列与静态库完全一致,仅这三列被改写;
- **MSAI 文件**:全部字段压缩为 float16/float32/uint8 等数组 + 元数据,`joblib.dump(compress=3)` 打包,
  客户推理端免解析直接查表。

## 2. 快速开始

### Docker 部署(每日 00:00 自动执行)

#### 构建镜像

在 `strategy` 目录下执行(有 Dockerfile 的那个目录):

```bash
# 国内网络加 --build-arg 用清华源 (PIP_INDEX_URL 只作用于 requirements.txt 部分)
docker build -t strategy-daily \
    --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
```

- 构建上下文已由 `.dockerignore` 精简(排除 148MB 的 close7 CSV 等中间产物)
- 主要耗时:下载 torch CPU wheel(约 200MB)+ 其余依赖,一般 5~15 分钟
- 镜像约 1.5~2GB,含:代码 / 模型权重 / 内置兜底策略库 .parquet / cron 定时任务

#### 运行容器

```bash
docker run -d --name strategy-daily 
-v /data/data:/data 
-v /data/data/input_lookback.csv:/app/input_lookback.csv 
-e TDENGINE_HOST=192.168.8.5 strategy-daily
```

可选环境变量(只有与默认不一致时才需要传):

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TDENGINE_HOST` | 192.168.8.5 | TDengine 服务器 IP |
| `TDENGINE_PORT` | 6041 | REST 端口 |
| `TDENGINE_USER` | root | 用户名 |
| `TDENGINE_PASSWORD` | taosdata | 密码 |
| `TDENGINE_DATABASE` | water_works | 数据库 |
| `TDENGINE_TABLE` | dynamics_plc_log | 超级表名 |
| `TZ` | Asia/Shanghai | 已内置, 无需传 |

#### 验证

```bash
docker logs strategy-daily          # 平时无输出正常 (cron 在等零点); 有输出=报错
docker exec strategy-daily ls /app  # 确认代码/策略库/模型进镜像
```

#### 不等零点, 立即手动跑一次

```bash
docker exec strategy-daily /app/run_daily.sh
```

跑完检查:

```bash
docker exec strategy-daily ls /data                          # 应有 daily_pump_schedule_strategy_lib*.csv
docker exec strategy-daily bash -c 'cat /app/logs/daily_*.log'    # 查看当日日志
```

#### 修改定时时间

```bash
crontab -e
# 每天 00:00 执行容器内的调度脚本
0 0 * * * docker exec strategy-daily /app/run_daily.sh >> /var/log/strategy_daily.log 2>&1
```


#### 日常运维

```bash
docker stop strategy-daily            # 停止
docker rm strategy-daily              # 删除容器
docker exec strategy-daily bash -c 'rm -f /app/logs/daily_*.log' # 删除所有 daily_ 开头的日志
docker exec strategy-daily bash -c 'rm -f /app/logs/daily_20260817*.log' # 或只删除某一天的日志(比如 2026年8月17日的)
```

#### 注意事项

- 容器内只跑 cron,**零点自动执行 run_daily.sh**;每天产物 `daily_pump_schedule_strategy_lib*.csv`、
  `_scheduled.parquet/.csv`、`pump_run_hours.csv` 全部保存在卷 `/data`(容器重建不丢);
  日志在镜像内 `/app/logs/`(30 天自动清理);
- 策略库更新:把新 `*.parquet` 放进 `/data` 卷,无需重建镜像;
- 取数失败会跳过当日调度并写入日志,不会用旧数据出方案;

## 3. 目录与关键文件

| 文件 | 作用 |
|---|---|
| `schedule_daily_strategy_library.py` | ② 日调度主脚本(预测 + 匹配 + 双向 DP + 策略库回写) |
| `tdengine_to_input_lookback.py` | ① TDengine 取数 |
| `train_msai.py` | ③ parquet → MSAI 转化 |
| `run_daily.sh` / `schedule-daily` / `Dockerfile` | 容器编排:每日调度流程 / cron 定时 / 镜像定义 |
| `close7_20260810.parquet` | 静态策略库(内置兜底) |
| `close7_20260810_scheduled.parquet` | 调度版策略库(示例产物,当日实际产物在卷 `/data`) |
| `pump_run_hours.csv` | 各泵连续运行时长(跨天累加,停泵归 0) |
| `transformer_pkg/` | Transformer 模型包(含权重 `results/`) |
| `input_lookback.csv` | 取数模板(列结构基准) |
| `docker_build_run.md` | Docker 完整指南(代码解释/镜像包/命令/输入输出/定时任务) |
| `scheduled_vs_static_advantage.md` | 静态库 vs 调度版对比 + parquet 使用方式 |

## 4. 关键数据语义速查

| 概念 | 说明 |
|---|---|
| `Time_Level` | 双向 DP 第 N 优解排名:1 = 全局最优路径,2…50 = 依次降级;**1000 = 未选中哨兵**(非排名) |
| `Suggest_Runtime_Start/End` | 执行时段,**HHMM 整数**(245 = 02:45,前导零被丢弃),真实时长均为 30 分钟 |
| 跨零点时段 | 结束 < 开始(如 2345→0015 = 23:45–次日 00:15),计算时长按跨天处理 |
| 多行同 id | 同一策略被安排到多个时段,每行一个时段,物理参数相同,逐行执行勿按 id 去重 |
| 液位 | 不参与筛选,每个 (时段, 排名) 尽量覆盖全部液位档(缺档借其他泵组真实行) |
| DP 代价 | 节点 = −效率×w_eff + 大泵×w_large + \|预测流量−目标流量\|×w_flow;切换 = 状态翻转×w_state + 频率变化×w_freq |


# 定时任务配置指南

## 小时级交易系统

本系统采用 **小时级信号检测** 架构，实现更及时的交易执行：

```
每天09:00 → 筛选候选池(100个币) → data/daily_scans/candidates_YYYYMMDD.csv
每小时整点 → 检测KDJ信号 → 立即执行交易（并行+5次重试+市价兜底）
```

## 一、清理旧定时任务

**重要**: 安装新任务前，必须先清理所有旧的Quant4Little定时任务，避免冲突。

```bash
cd ~/Quant4LITTLE

# 查看当前定时任务
crontab -l | grep -i quant

# 清理所有旧任务（会自动备份）
bash deploy/clear_crontab.sh

# 验证清理结果
crontab -l
```

**清理脚本会做什么**:
- 自动备份当前crontab到 `/tmp/crontab_backup_YYYYMMDD_HHMMSS.txt`
- 删除所有包含 "quant4little" 或 "Quant4LITTLE" 的任务
- 显示剩余的crontab任务

## 二、安装新定时任务

清理完成后，安装小时级交易系统：

```bash
cd ~/Quant4LITTLE

# 安装新的定时任务
bash deploy/setup_cron_hourly.sh

# 脚本会显示将要安装的任务，输入 y 确认
```

**新定时任务内容**:

```bash
# 每天09:00 (UTC 01:00) - 筛选候选币池
0 1 * * * cd ~/Quant4LITTLE && python3 scripts/daily_candidate_scan.py --top-n 100 >> logs/cron_candidates.log 2>&1

# 每小时整点 - 检测信号并执行交易
0 * * * * cd ~/Quant4LITTLE && bash deploy/cron_hourly_detect.sh >> logs/cron_hourly.log 2>&1
```

**执行时间表（北京时间）**:
- 09:00 - 筛选候选池
- 10:00 - 检测信号+交易
- 11:00 - 检测信号+交易
- 12:00 - 检测信号+交易
- ... 每小时重复 ...

## 三、验证安装

```bash
# 1. 查看已安装的定时任务
crontab -l

# 2. 检查日志目录
ls -lh logs/

# 3. 手动测试候选池筛选
python3 scripts/daily_candidate_scan.py --top-n 100

# 4. 手动测试小时级检测（需要先有候选池）
python3 scripts/hourly_signal_detector.py

# 5. 查看生成的文件
ls -lh data/daily_scans/
ls -lh data/hourly_signals/
```

## 四、监控运行

### 实时监控日志

```bash
# 监控小时级检测日志
tail -f logs/cron_hourly.log

# 监控候选池筛选日志
tail -f logs/cron_candidates.log

# 查看最近的日志
tail -50 logs/cron_hourly.log
```

### 检查生成的信号文件

```bash
# 查看今日候选池
TODAY=$(date +%Y%m%d)
cat data/daily_scans/candidates_${TODAY}.csv

# 查看小时级信号
ls -lt data/hourly_signals/ | head

# 查看当前持仓
python3 scripts/manage_positions.py
```

### 检查系统状态

```bash
# 查看cron服务状态（Linux）
sudo systemctl status cron

# 查看cron日志（系统级）
sudo grep CRON /var/log/syslog | tail -20
```

## 五、常见问题

### Q1: 定时任务没有执行？

**检查步骤**:
```bash
# 1. 确认cron服务运行
sudo systemctl status cron

# 2. 检查crontab语法
crontab -l

# 3. 查看系统日志
sudo grep CRON /var/log/syslog | tail -50

# 4. 检查脚本权限
ls -l deploy/cron_hourly_detect.sh

# 5. 手动运行测试
cd ~/Quant4LITTLE && bash deploy/cron_hourly_detect.sh
```

**常见原因**:
- cron服务未启动: `sudo systemctl start cron`
- 路径不正确: 使用绝对路径
- 环境变量未加载: 检查 `.env` 文件

### Q2: 脚本执行但没有交易？

**可能原因**:
```bash
# 1. 候选池文件不存在
ls -l data/daily_scans/candidates_$(date +%Y%m%d).csv

# 2. 未检测到信号
tail -50 logs/cron_hourly.log | grep "检测到"

# 3. 持仓已满（上限10个）
python3 scripts/manage_positions.py

# 4. API连接问题
python3 -c "from scripts.live_maker_trader import check_exchange; check_exchange()"
```

### Q3: 如何修改执行频率？

编辑crontab:
```bash
crontab -e

# 例如改为每2小时执行:
0 */2 * * * cd ~/Quant4LITTLE && bash deploy/cron_hourly_detect.sh >> logs/cron_hourly.log 2>&1

# 例如改为每30分钟执行:
*/30 * * * * cd ~/Quant4LITTLE && bash deploy/cron_hourly_detect.sh >> logs/cron_hourly.log 2>&1
```

### Q4: 如何暂停交易？

**方法1**: 临时禁用定时任务
```bash
# 注释掉交易任务
crontab -e
# 在交易任务前加 #

# 或直接删除所有任务
bash deploy/clear_crontab.sh
```

**方法2**: 修改持仓限制
```bash
# 编辑 live_maker_trader.py，设置 max_positions=0
# 这样会跳过所有新交易
```

### Q5: 如何恢复旧的定时任务？

```bash
# 清理脚本会自动备份到:
ls -lt /tmp/crontab_backup_*.txt | head -5

# 恢复备份
crontab /tmp/crontab_backup_YYYYMMDD_HHMMSS.txt

# 验证
crontab -l
```

## 六、手动执行命令

**完整工作流程**:

```bash
cd ~/Quant4LITTLE

# Step 1: 筛选候选池
python3 scripts/daily_candidate_scan.py --top-n 100

# Step 2: 检测小时级信号
python3 scripts/hourly_signal_detector.py

# Step 3: 查看检测到的信号
TODAY=$(date +%Y%m%d)
HOUR=$(date +%H)
cat data/hourly_signals/signals_${TODAY}_${HOUR}.csv

# Step 4: 手动执行交易（如果有信号）
python3 scripts/live_maker_trader.py \
  --signals-file data/hourly_signals/signals_${TODAY}_${HOUR}.csv \
  --auto-confirm

# Step 5: 查看持仓
python3 scripts/manage_positions.py
```

**或使用一键检测+交易**:
```bash
# 检测信号并立即执行交易
python3 scripts/hourly_signal_detector.py --execute --auto-confirm
```

## 七、系统架构

```
┌─────────────────────────────────────────────────────────┐
│  每天09:00 - 候选池筛选                                  │
│  daily_candidate_scan.py --top-n 100                   │
│  → data/daily_scans/candidates_YYYYMMDD.csv            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  每小时 - 信号检测                                       │
│  hourly_signal_detector.py                             │
│  读取: data/daily_scans/candidates_YYYYMMDD.csv        │
│  读取: data/hourly_klines/*_1h.csv                     │
│  检测: KDJ > 50 + EMA底部形态 + ATR >= 2%              │
│  输出: data/hourly_signals/signals_YYYYMMDD_HH.csv     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  立即执行 - 并行下单                                     │
│  live_maker_trader.py --signals-file <path>            │
│  策略: Maker限价单                                      │
│  重试: 5次追单 (0.10% → 0.08% → 0.05% → 0.03% → 0.01%)│
│  兜底: 市价单 (offset=0%)                               │
│  风控: 最大10个持仓                                      │
└─────────────────────────────────────────────────────────┘
```

## 八、关键文件说明

| 文件 | 用途 | 执行频率 |
|------|------|----------|
| [scripts/daily_candidate_scan.py](scripts/daily_candidate_scan.py) | 筛选候选币池 | 每天09:00 |
| [scripts/hourly_signal_detector.py](scripts/hourly_signal_detector.py) | 检测KDJ入场信号 | 每小时 |
| [scripts/live_maker_trader.py](scripts/live_maker_trader.py) | 执行实盘交易 | 有信号时 |
| [deploy/cron_hourly_detect.sh](deploy/cron_hourly_detect.sh) | 定时任务包装脚本 | 每小时 |
| [deploy/clear_crontab.sh](deploy/clear_crontab.sh) | 清理旧定时任务 | 手动 |
| [deploy/setup_cron_hourly.sh](deploy/setup_cron_hourly.sh) | 安装新定时任务 | 手动 |

---

**部署完成后的下一步**:
1. 等待第一次定时任务执行（查看日志）
2. 监控交易执行情况
3. 根据实际表现调整参数（KDJ阈值、ATR阈值等）

# 部署小时级交易系统

## 快速部署指南（ECS服务器）

按照以下步骤在ECS服务器上部署小时级交易系统：

### 第一步：拉取最新代码

```bash
cd ~/Quant4LITTLE
git pull origin main
```

**预期输出**:
```
From https://github.com/sakaili/Quant4LITTLE
   12e77d9..e2a77ac  main -> main
Updating 12e77d9..e2a77ac
Fast-forward
 CRON_SETUP.md               | 348 ++++++++++++++++++++++++++++++++++++++++++++
 deploy/clear_crontab.sh     |  45 ++++++
 deploy/setup.sh             |  43 +++---
 deploy/setup_cron_hourly.sh |  88 +++++++++++
 4 files changed, 431 insertions(+), 34 deletions(-)
```

### 第二步：清理旧定时任务

```bash
cd ~/Quant4LITTLE
bash deploy/clear_crontab.sh
```

**这个脚本会做什么**:
- 自动备份当前crontab到 `/tmp/crontab_backup_*.txt`
- 删除所有 Quant4Little 相关的定时任务
- 显示清理后的crontab

**预期输出**:
```
========================================================================
  清理 Quant4Little 定时任务
========================================================================

  备份当前crontab到: /tmp/crontab_backup_20250124_103045.txt

  当前的Quant4Little任务:
  0 1 * * * cd ~/Quant4LITTLE && bash deploy/cron_update_data.sh >> logs/cron_update.log 2>&1
  0 2 * * * cd ~/Quant4LITTLE && bash deploy/cron_generate_signals.sh >> logs/cron_signals.log 2>&1
  30 2 * * * cd ~/Quant4LITTLE && bash deploy/cron_execute_trades.sh >> logs/cron_trades.log 2>&1

  ✓ 已删除所有Quant4Little定时任务

  当前剩余的crontab任务:
  (无)

========================================================================
  清理完成
========================================================================
```

### 第三步：安装小时级定时任务

```bash
cd ~/Quant4LITTLE
bash deploy/setup_cron_hourly.sh
```

**脚本会显示将要安装的任务**:
```
========================================================================
  安装 Quant4Little 小时级定时任务
========================================================================

  项目目录: /root/Quant4LITTLE

  将要安装的定时任务:
  ────────────────────────────────────────────────────────────────
  # Quant4Little 小时级交易系统
  # 每天09:00 (UTC 01:00) - 筛选候选币池
  0 1 * * * cd /root/Quant4LITTLE && python3 scripts/daily_candidate_scan.py --top-n 100 >> logs/cron_candidates.log 2>&1

  # 每小时整点 - 检测信号并执行交易
  0 * * * * cd /root/Quant4LITTLE && bash deploy/cron_hourly_detect.sh >> logs/cron_hourly.log 2>&1
  ────────────────────────────────────────────────────────────────

  确认安装? (y/n)
```

**输入 `y` 确认安装**

**预期输出**:
```
  ✓ 备份当前crontab: /tmp/crontab_backup_20250124_103123.txt
  ✓ 清理旧任务
  ✓ 安装新任务

  当前的crontab任务:
  ────────────────────────────────────────────────────────────────
  # Quant4Little 小时级交易系统
  # 每天09:00 (UTC 01:00) - 筛选候选币池
  0 1 * * * cd /root/Quant4LITTLE && python3 scripts/daily_candidate_scan.py --top-n 100 >> logs/cron_candidates.log 2>&1
  # 每小时整点 - 检测信号并执行交易
  0 * * * * cd /root/Quant4LITTLE && bash deploy/cron_hourly_detect.sh >> logs/cron_hourly.log 2>&1
  ────────────────────────────────────────────────────────────────

  ✅ 定时任务安装成功

  📊 执行时间 (北京时间):
    - 09:00  筛选候选池
    - 每小时  检测信号并交易

  📝 监控命令:
    tail -f logs/cron_hourly.log
    tail -f logs/cron_candidates.log

========================================================================
```

### 第四步：验证安装

```bash
# 查看已安装的定时任务
crontab -l

# 创建日志目录
mkdir -p ~/Quant4LITTLE/logs

# 检查脚本权限
ls -l ~/Quant4LITTLE/deploy/*.sh

# 确保脚本可执行（如果需要）
chmod +x ~/Quant4LITTLE/deploy/*.sh
```

### 第五步：手动测试

在等待定时任务自动执行前，可以手动测试整个流程：

```bash
cd ~/Quant4LITTLE

# 1. 测试候选池筛选（耗时约2-5分钟）
echo "=== 测试候选池筛选 ==="
python3 scripts/daily_candidate_scan.py --top-n 100

# 2. 检查生成的候选池文件
TODAY=$(date +%Y%m%d)
if [ -f "data/daily_scans/candidates_${TODAY}.csv" ]; then
    echo "✓ 候选池文件已生成"
    wc -l "data/daily_scans/candidates_${TODAY}.csv"
    head -5 "data/daily_scans/candidates_${TODAY}.csv"
else
    echo "❌ 候选池文件未生成"
fi

# 3. 测试小时级信号检测（不执行交易）
echo ""
echo "=== 测试信号检测 ==="
python3 scripts/hourly_signal_detector.py

# 4. 查看检测到的信号（如果有）
HOUR=$(date +%H)
if [ -f "data/hourly_signals/signals_${TODAY}_${HOUR}.csv" ]; then
    echo "✓ 检测到信号"
    cat "data/hourly_signals/signals_${TODAY}_${HOUR}.csv"
else
    echo "本小时无信号"
fi
```

### 第六步：监控运行

系统部署后，使用以下命令监控运行状态：

```bash
# 实时监控小时级检测日志
tail -f ~/Quant4LITTLE/logs/cron_hourly.log

# 实时监控候选池筛选日志
tail -f ~/Quant4LITTLE/logs/cron_candidates.log

# 查看最近50行日志
tail -50 ~/Quant4LITTLE/logs/cron_hourly.log

# 查看系统cron日志
sudo grep CRON /var/log/syslog | tail -20
```

## 执行时间表

**北京时间**:

| 时间 | 任务 | 说明 |
|------|------|------|
| 09:00 | 筛选候选池 | 扫描全市场，选出100个底部反弹候选币 |
| 10:00 | 检测信号+交易 | 基于候选池检测KDJ信号，发现信号立即执行 |
| 11:00 | 检测信号+交易 | 同上 |
| 12:00 | 检测信号+交易 | 同上 |
| ... | ... | 每小时重复 |
| 23:00 | 检测信号+交易 | 最后一次 |
| 00:00 | 检测信号+交易 | 新一天开始 |

## 交易逻辑

1. **候选池筛选**（每天09:00）:
   - 扫描所有USDT交易对
   - 筛选条件: 4小时线KDJ_J < 20（超卖）+ 成交量 > 500万USDT
   - 选出100个底部反弹候选币
   - 保存到 `data/daily_scans/candidates_YYYYMMDD.csv`

2. **信号检测**（每小时）:
   - 读取今日候选池
   - 检测小时线信号:
     - KDJ_J > 50（开始反弹）
     - EMA10 < EMA20 < EMA30（底部形态）
     - ATR波动率 >= 2%（有波动空间）
   - 发现信号保存到 `data/hourly_signals/signals_YYYYMMDD_HH.csv`

3. **立即执行交易**（检测到信号时）:
   - 并行下单（最多10个品种）
   - Maker限价单策略（offset: 0.10%）
   - 5次追单重试: 0.10% → 0.08% → 0.05% → 0.03% → 0.01%
   - 最终市价兜底（offset: 0%）
   - 每个品种投入账户总资金的10%

## 风险控制

- **最大持仓**: 10个品种
- **单品种资金**: 10% 账户总资金
- **止损**: 达到持仓上限后不再开新仓
- **止盈**: 需要手动平仓或通过其他脚本管理

## 常见问题

### Q: 第一次执行会在什么时候？

**下一个整点**。例如:
- 如果现在是10:25，下次执行在11:00
- 如果现在是13:58，下次执行在14:00

**候选池筛选**在明天早上09:00（UTC 01:00）首次执行。

### Q: 如果没有候选池文件会怎样？

小时级检测会输出:
```
  候选池文件不存在: data/daily_scans/candidates_YYYYMMDD.csv
  本小时无交易信号
```

需要手动运行一次候选池筛选:
```bash
python3 scripts/daily_candidate_scan.py --top-n 100
```

### Q: 如何确认定时任务真的在运行？

1. **查看系统日志**:
   ```bash
   sudo grep CRON /var/log/syslog | grep Quant4LITTLE | tail -10
   ```

2. **查看应用日志**:
   ```bash
   ls -lth ~/Quant4LITTLE/logs/
   ```

3. **等待下一个整点后检查**:
   ```bash
   # 等到整点后1分钟，例如14:01
   tail -50 ~/Quant4LITTLE/logs/cron_hourly.log
   ```

### Q: 如何临时禁用交易？

**方法1**: 注释掉cron任务
```bash
crontab -e
# 在交易任务前加 # 号
```

**方法2**: 临时删除所有任务
```bash
bash deploy/clear_crontab.sh
```

**方法3**: 修改 `live_maker_trader.py`，设置 `max_positions=0`

### Q: 发现问题如何回滚？

```bash
# 恢复之前的crontab备份
ls -lt /tmp/crontab_backup_*.txt | head -5
crontab /tmp/crontab_backup_YYYYMMDD_HHMMSS.txt

# 验证
crontab -l
```

## 下一步

部署完成后:
1. 等待明天09:00自动生成候选池
2. 从10:00开始每小时自动检测信号
3. 监控交易执行情况
4. 使用 `python3 scripts/manage_positions.py` 查看持仓

---

更多详细信息请查看 [CRON_SETUP.md](CRON_SETUP.md)

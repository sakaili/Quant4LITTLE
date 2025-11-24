# 每小时自动交易设置指南

## 修改内容

### 1. 币池扩大到后200名
- **修改文件**: `scripts/update_latest_data.py`
- **改动**: 从选择Top 100 改为选择后200名（流动性最低的垃圾币）
```python
# 修改前
ticker_df = ticker_df.sort_values("quote_volume", ascending=False)
top_symbols = ticker_df.head(100)["symbol"].tolist()

# 修改后
ticker_df = ticker_df.sort_values("quote_volume", ascending=True)
bottom_symbols = ticker_df.head(200)["symbol"].tolist()
```

### 2. 创建每小时自动运行脚本
- **新文件**: `scripts/hourly_trading.py`
- **功能**:
  1. 更新币池数据（后200名）
  2. 运行Paper Trading生成信号
  3. 记录日志到 `logs/hourly_trading_YYYYMM.log`

---

## Windows 系统设置

### 方法1: 使用任务计划程序（推荐）

1. **打开任务计划程序**
   - 按 `Win + R`
   - 输入 `taskschd.msc`
   - 回车

2. **创建基本任务**
   - 点击右侧 "创建基本任务"
   - 名称: `Quant4Little 每小时交易`
   - 描述: `每小时自动运行做空策略Paper Trading`

3. **设置触发器**
   - 选择 "每天"
   - 开始时间: 选择今天的日期
   - 点击 "下一步"

4. **设置操作**
   - 选择 "启动程序"
   - 程序或脚本: `F:\2025\Quant4Little\run_hourly_trading.bat`
   - 起始于（可选）: `F:\2025\Quant4Little`

5. **高级设置**
   - 完成后，右键点击创建的任务 → "属性"
   - 在 "触发器" 标签页，编辑触发器:
     - 勾选 "重复任务间隔": 选择 `1 小时`
     - 持续时间: `无限期`
   - 在 "设置" 标签页:
     - 勾选 "如果任务运行时间超过以下时间，停止任务": `2 小时`
     - 取消勾选 "仅当计算机使用交流电源时启动此任务"

6. **测试运行**
   - 右键点击任务 → "运行"
   - 查看日志: `logs/hourly_trading_202511.log`

### 方法2: 使用命令行快速设置

以管理员身份运行 PowerShell，执行：

```powershell
# 创建每小时运行的任务
schtasks /create /tn "Quant4Little每小时交易" /tr "F:\2025\Quant4Little\run_hourly_trading.bat" /sc hourly /st 00:00

# 查看任务
schtasks /query /tn "Quant4Little每小时交易"

# 立即运行测试
schtasks /run /tn "Quant4Little每小时交易"

# 删除任务（如需要）
schtasks /delete /tn "Quant4Little每小时交易" /f
```

---

## Linux/服务器设置

### 1. 添加执行权限

```bash
cd /path/to/Quant4Little
chmod +x run_hourly_trading.sh
```

### 2. 编辑 crontab

```bash
crontab -e
```

添加以下行（每小时整点运行）：

```bash
# 每小时整点运行做空策略
0 * * * * /path/to/Quant4Little/run_hourly_trading.sh >> /path/to/Quant4Little/logs/cron.log 2>&1

# 或者每小时的第5分钟运行
5 * * * * /path/to/Quant4Little/run_hourly_trading.sh >> /path/to/Quant4Little/logs/cron.log 2>&1
```

### 3. 查看 crontab 任务

```bash
crontab -l
```

### 4. 测试运行

```bash
cd /path/to/Quant4Little
./run_hourly_trading.sh
```

### 5. 查看日志

```bash
tail -f logs/hourly_trading_202511.log
```

---

## 运行流程

### 每小时执行的步骤

```
[00:00] 定时任务触发
    ↓
[Step 1] 更新币池数据
    - 获取所有USDT永续合约
    - 按交易量排序（升序）
    - 选择后200名低流动性币种
    - 下载最新K线数据
        * 日线: 540天
        * 小时线: 90天
    ↓
[Step 2] 运行Paper Trading
    - 策略筛选候选标的
        * EMA10 < EMA20 < EMA30
        * KDJ_J > 70
        * ATR/Close > 2%
    - 模型打分排序
    - 生成SHORT信号（最多20个）
    ↓
[输出] 保存结果
    - data/paper_trading/candidates_YYYYMMDD.csv
    - data/paper_trading/ranked_YYYYMMDD.csv
    - data/paper_trading/signals_YYYYMMDD.csv
    - data/paper_trading/signals_history.csv
    ↓
[日志] 记录到文件
    - logs/hourly_trading_YYYYMM.log
```

---

## 监控与维护

### 1. 查看实时日志

**Windows:**
```powershell
Get-Content -Path "logs\hourly_trading_202511.log" -Wait
```

**Linux:**
```bash
tail -f logs/hourly_trading_202511.log
```

### 2. 检查最新信号

```bash
# 查看今天的信号
cat data/paper_trading/signals_20251123.csv

# 查看信号历史
tail -20 data/paper_trading/signals_history.csv
```

### 3. 监控币池大小

```bash
# Windows
dir data\daily_klines\*.csv | Measure-Object | Select-Object Count

# Linux
ls data/daily_klines/*.csv | wc -l
```

### 4. 清理旧日志（可选）

```bash
# 只保留最近30天的日志
find logs/ -name "hourly_trading_*.log" -mtime +30 -delete
```

---

## 常见问题

### Q1: 任务没有自动运行？
**Windows:**
- 检查任务计划程序中任务状态
- 确认触发器设置正确（重复间隔1小时）
- 查看 "历史记录" 标签页

**Linux:**
- 检查 cron 服务状态: `systemctl status cron`
- 查看系统日志: `grep CRON /var/log/syslog`

### Q2: 脚本运行失败？
- 查看日志文件: `logs/hourly_trading_YYYYMM.log`
- 确认Python环境正确
- 测试手动运行: `python scripts/hourly_trading.py`

### Q3: 数据更新太慢？
- 200个币种约需要 20-30分钟
- 可以考虑减少币种数量
- 或者调整为每天更新一次数据

### Q4: 想要更改运行频率？
**Windows 任务计划程序:**
- 修改触发器的 "重复任务间隔"

**Linux crontab:**
```bash
# 每30分钟运行一次
*/30 * * * * /path/to/run_hourly_trading.sh

# 每2小时运行一次
0 */2 * * * /path/to/run_hourly_trading.sh

# 只在交易时间运行（8-22点每小时）
0 8-22 * * * /path/to/run_hourly_trading.sh
```

---

## 性能优化建议

### 1. 跳过数据更新（如果数据已是最新）

修改 `scripts/hourly_trading.py`:

```python
def main():
    # 只在每天8点更新数据
    current_hour = datetime.now().hour
    if current_hour == 8:
        update_coin_pool()

    # 其他时间只运行Paper Trading
    run_paper_trading()
```

### 2. 使用缓存加速

- 启用Binance API缓存
- 使用本地数据库存储历史数据

### 3. 减少币种数量

如果性能不足，可以减少到100个币种：

```python
# scripts/update_latest_data.py
bottom_symbols = ticker_df.head(100)["symbol"].tolist()
```

---

## 生产环境注意事项

1. **设置Binance API密钥**（如需实盘）
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

2. **确保网络稳定**
- 使用代理（如需要）
- 设置API重试机制

3. **监控磁盘空间**
- 日志文件会持续增长
- 定期清理旧数据

4. **备份关键数据**
```bash
# 备份模型和配置
tar -czf backup_$(date +%Y%m%d).tar.gz models/ data/paper_trading/
```

---

## 快速启动命令

**Windows (管理员PowerShell):**
```powershell
# 设置每小时任务
schtasks /create /tn "Quant4Little每小时交易" /tr "F:\2025\Quant4Little\run_hourly_trading.bat" /sc hourly /st 00:00

# 立即测试
schtasks /run /tn "Quant4Little每小时交易"

# 查看日志
Get-Content -Path "logs\hourly_trading_202511.log" -Tail 50
```

**Linux:**
```bash
# 添加crontab
echo "0 * * * * $(pwd)/run_hourly_trading.sh" | crontab -

# 立即测试
./run_hourly_trading.sh

# 查看日志
tail -f logs/hourly_trading_202511.log
```

---

祝交易顺利！📈

# 策略配置总结

## ✅ 已完成的修改

### 1. 币池扩大到后200名
- **文件**: `scripts/update_latest_data.py`
- **改动**: 从Top 100 改为后200名低流动性币种
- **逻辑**: 按24h成交量升序排序，选择最垃圾的200个币

### 2. 创建每小时自动运行系统
- **主脚本**: `scripts/hourly_trading.py`
- **Windows脚本**: `run_hourly_trading.bat`
- **Linux脚本**: `run_hourly_trading.sh`
- **详细文档**: `HOURLY_TRADING_SETUP.md`

---

## 📊 当前策略配置

### 币池选择
- **数量**: 200个标的
- **选择标准**: 24h成交量最低的后200名
- **市场**: Binance USDT永续合约
- **更新频率**: 每小时

### 技术指标筛选
- **EMA形态**: EMA10 < EMA20 < EMA30 (底部)
- **KDJ信号**: KDJ_J > 70 (超买)
- **波动率**: ATR/Close > 2%

### 模型排序
- **模型类型**: Transformer
- **输入**: 90根1h K线 + 表格特征
- **输出**: 3分类（差/中/优）
- **选币数量**: 最多20个

### 信号类型
- **方向**: SHORT (做空)
- **状态**: PENDING (待执行)

---

## 🚀 快速启动

### Windows系统

**方法1: 图形界面设置**
1. 按 `Win + R` → 输入 `taskschd.msc`
2. 创建基本任务
3. 触发器: 每天，重复间隔1小时
4. 操作: 启动 `F:\2025\Quant4Little\run_hourly_trading.bat`

**方法2: PowerShell命令（管理员）**
```powershell
# 创建每小时任务
schtasks /create /tn "Quant4Little每小时交易" /tr "F:\2025\Quant4Little\run_hourly_trading.bat" /sc hourly /st 00:00

# 立即测试
schtasks /run /tn "Quant4Little每小时交易"

# 查看日志
Get-Content logs\hourly_trading_202511.log -Tail 50 -Wait
```

### Linux系统

```bash
# 添加执行权限
chmod +x run_hourly_trading.sh

# 添加到crontab（每小时整点运行）
crontab -e
# 添加行: 0 * * * * /path/to/Quant4Little/run_hourly_trading.sh

# 手动测试
./run_hourly_trading.sh

# 查看日志
tail -f logs/hourly_trading_202511.log
```

---

## 📁 输出文件

### 每小时生成
- `data/paper_trading/candidates_YYYYMMDD.csv` - 筛选的候选标的
- `data/paper_trading/ranked_YYYYMMDD.csv` - 模型打分排序
- `data/paper_trading/signals_YYYYMMDD.csv` - 当日最新信号
- `data/paper_trading/signals_history.csv` - 所有信号历史

### 日志文件
- `logs/hourly_trading_YYYYMM.log` - 每月日志文件

---

## 📊 监控命令

### 查看币池大小
```bash
# Windows
dir data\daily_klines\*.csv | Measure-Object | Select-Object Count

# Linux
ls data/daily_klines/*.csv | wc -l
```

### 查看最新信号
```bash
# 查看今天的信号
cat data/paper_trading/signals_20251123.csv

# 查看最近10条信号
tail -10 data/paper_trading/signals_history.csv
```

### 实时监控日志
```bash
# Windows PowerShell
Get-Content -Path "logs\hourly_trading_202511.log" -Wait

# Linux
tail -f logs/hourly_trading_202511.log
```

---

## ⚙️ 性能优化建议

### 1. 减少数据更新频率
只在每天早上8点更新数据，其他时间只运行策略：

修改 `scripts/hourly_trading.py`:
```python
def main():
    current_hour = datetime.now().hour

    # 只在早上8点更新数据
    if current_hour == 8:
        update_coin_pool()

    # 每小时运行策略
    run_paper_trading()
```

### 2. 减少币种数量
如果200个币种太多，可以减少到100-150个：

修改 `scripts/update_latest_data.py`:
```python
bottom_symbols = ticker_df.head(150)["symbol"].tolist()
```

### 3. 并行下载数据
使用多线程加速数据下载（高级用户）

---

## 🎯 下一步计划

### 短期（1-2天）
- [ ] 测试每小时自动运行
- [ ] 验证信号生成正确
- [ ] 监控币池是否完整（200个）

### 中期（1-2周）
- [ ] 收集更多历史信号数据
- [ ] 重新训练模型（更大样本量）
- [ ] 优化筛选条件参数

### 长期（1个月+）
- [ ] 接入实盘交易API
- [ ] 实现自动开平仓
- [ ] 建立监控告警系统

---

## ⚠️ 注意事项

1. **数据更新时间**: 200个币种约需20-30分钟
2. **网络要求**: 需要稳定的网络连接到Binance
3. **磁盘空间**: 预留至少5GB空间
4. **实盘风险**: 当前仅Paper Trading，实盘需谨慎

---

## 📞 故障排除

### 问题1: 任务没有运行
- 检查任务计划程序/crontab配置
- 查看日志文件是否有错误
- 手动运行测试: `python scripts/hourly_trading.py`

### 问题2: 数据下载失败
- 检查网络连接
- 验证Binance API可访问性
- 查看日志中的具体错误信息

### 问题3: 模型加载失败
- 确认 `models/rank_model.pt` 存在
- 确认 `models/rank_model_meta.json` 存在
- 检查PyTorch版本兼容性

---

## 📈 当前测试结果

**日期**: 2025-11-23

**币池**:
- 正在下载后200名低流动性币种
- 包括: CTK, BR, RIF, FORTH, POWR, VIC, SYS, G, QUICK, 1000WHY, 等

**筛选结果** (样例):
- 候选标的: 4个
- 最终信号: 2个 SHORT (DEXE, ORCA)

---

更多详细信息请查看 `HOURLY_TRADING_SETUP.md`

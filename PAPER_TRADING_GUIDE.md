# Paper Trading 使用指南

## 🎯 完成！Paper Trading系统已就绪

我已经为你创建了一个完整的Paper Trading（模拟交易）系统：[scripts/paper_trader.py](scripts/paper_trader.py)

---

## ✅ 系统功能

```
策略筛选 → 模型打分 → 生成信号 → 保存结果
```

**已实现**:
1. ✅ 策略筛选候选标的（基于EMA/KDJ/ATR）
2. ✅ 模型打分排序（使用训练好的模型）
3. ✅ 生成交易信号（只记录不下单）
4. ✅ 保存完整结果（候选/排序/信号）

---

## 🚀 快速开始

### 基础用法

```bash
# 运行今天的Paper Trading
python scripts/paper_trader.py

# 指定日期运行
python scripts/paper_trader.py --date 2025-01-15

# 指定最大持仓数
python scripts/paper_trader.py --max-positions 10

# 不使用模型（仅传统策略）
python scripts/paper_trader.py --no-model
```

### 输出文件

所有结果保存在 `data/paper_trading/`：

```
data/paper_trading/
├── candidates_20250115.csv     # 策略筛选的候选标的
├── ranked_20250115.csv         # 模型打分后的排序结果
├── signals_20250115.csv        # 最终生成的交易信号
└── signals_history.csv         # 所有历史信号汇总
```

---

## 📋 筛选策略规则

当前默认规则（可根据需要调整）：

```python
1. EMA底部形态: EMA10 < EMA20 < EMA30
2. KDJ超买: KDJ_J > 90  # 可能太严格，建议改为 > 80 或 > 70
3. ATR波动率: ATR/Close > 2%
4. 成交量: 有成交数据即可
```

**如果筛选不到标的**，可以放宽条件：

编辑 `scripts/paper_trader.py` 的 `scan_candidates()` 函数：

```python
# 原始（严格）
if kdj_j < 90:
    continue

# 改为（宽松）
if kdj_j < 70:
    continue
```

或完全移除KDJ限制：

```python
# 注释掉KDJ检查
# if kdj_j < 90:
#     continue
```

---

## 🔍 实际运行示例

由于当前数据可能不满足严格的筛选条件（KDJ_J>90），建议：

### 方案1: 放宽筛选条件

修改 `scripts/paper_trader.py` 第 118行左右：

```python
# 修改前
if kdj_j < 90:
    continue

# 修改后（更实用）
if kdj_j < 60:  # 或 70、80
    continue
```

### 方案2: 使用无模型模式测试

```bash
# 先测试策略筛选部分
python scripts/paper_trader.py --date 2025-01-10 --no-model
```

### 方案3: 检查数据可用性

```bash
# 查看有哪些日期的数据
ls data/daily_klines/ | head

# 随机选一个标的查看KDJ值
python -c "
import pandas as pd
from scripts.paper_trader import add_indicators

df = pd.read_csv('data/daily_klines/BTCUSDT.csv', parse_dates=['open_time'])
df = add_indicators(df)
print(df[['open_time', 'close', 'ema10', 'ema20', 'ema30', 'kdj_j']].tail(10))
"
```

---

## 📊 输出示例

成功运行后会看到类似输出：

```
============================================================
Paper Trading System - 2025-01-15
============================================================

[1/4] 策略筛选候选标的 (日期: 2025-01-15)
  [OK] 策略筛选出 45 个候选标的
  [OK] 平均ATR波动率: 3.2%
  [OK] 平均成交量比: 1.8x

[2/4] 模型打分排序
  [OK] 完成 45 个标的打分
  [OK] 平均分数: 0.2156
  [OK] 预测Class 2 (优): 18 个

[3/4] 生成交易信号 (最多 10 个)
  [OK] 选择 10 个Class 2标的

  生成的交易信号:
    symbol     close  model_score  model_class signal_type
  BTCUSDT  45000.00        0.4523            2         BUY
  ETHUSDT   2500.00        0.3891            2         BUY
  ...

[4/4] 保存结果
  [OK] 候选标的: data/paper_trading/candidates_20250115.csv
  [OK] 排序结果: data/paper_trading/ranked_20250115.csv
  [OK] 交易信号: data/paper_trading/signals_20250115.csv
  [OK] 信号历史: data/paper_trading/signals_history.csv

============================================================
[OK] Paper Trading 完成
============================================================

汇总:
  策略筛选: 45 个候选
  模型打分: 45 个排序
  交易信号: 10 个买入

信号详情:
  文件: data/paper_trading/signals_20250115.csv
  Top 5: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
```

---

## 🎨 定制化

### 修改筛选策略

编辑 `scan_candidates()` 函数的规则：

```python
# 更激进的策略
if kdj_j < 50:  # 更早入场
    continue

# 更保守的策略
if kdj_j < 95:  # 等待明确信号
    continue

# 添加其他条件
rsi = signal_row.get("rsi", 50)
if rsi < 30 or rsi > 70:  # 避免极端情况
    continue
```

### 修改排序逻辑

编辑 `generate_signals()` 函数：

```python
# 只选Class 2
signals = ranked[ranked["model_class"] == 2].head(max_positions)

# 或混合Class 1和2
good = ranked[ranked["model_class"].isin([1, 2])].head(max_positions)

# 或按分数阈值
high_score = ranked[ranked["model_score"] > 0.3].head(max_positions)
```

---

## 🔄 持续运行

### 每日定时运行

Linux/Mac (crontab):
```bash
# 每天早上8点运行
0 8 * * * cd /path/to/Quant4Little && python scripts/paper_trader.py >> logs/paper_trading.log 2>&1
```

Windows (任务计划程序):
```bat
@echo off
cd /d F:\2025\Quant4Little
python scripts/paper_trader.py >> logs\paper_trading.log 2>&1
```

### 监控信号历史

```bash
# 查看最近的信号
tail -20 data/paper_trading/signals_history.csv

# 统计信号数量
python -c "
import pandas as pd
df = pd.read_csv('data/paper_trading/signals_history.csv')
print(f'总信号数: {len(df)}')
print(f'按日期统计:')
print(df.groupby(df['signal_time'].str[:10]).size())
"
```

---

## 🐛 故障排查

### 问题1: 没有候选标的

**原因**: 筛选条件太严格

**解决**:
- 放宽KDJ阈值（90 → 70）
- 降低ATR要求（2% → 1%）
- 移除某些条件

### 问题2: 模型加载失败

**原因**: 没有训练好的模型

**解决**:
```bash
# 先训练模型
python -m scripts.modeling.train_ranker --epochs 50

# 或使用无模型模式
python scripts/paper_trader.py --no-model
```

### 问题3: 日期数据不存在

**原因**: K线数据还没下载到那个日期

**解决**:
```bash
# 检查可用日期范围
python -c "
import pandas as pd
df = pd.read_csv('data/daily_klines/BTCUSDT.csv')
print('数据范围:')
print(f'  开始: {df[\"open_time\"].min()}')
print(f'  结束: {df[\"open_time\"].max()}')
"

# 使用数据范围内的日期
python scripts/paper_trader.py --date 2025-01-05
```

---

## 📈 下一步

1. **调整筛选条件**，找到符合你策略的参数
2. **运行历史回测**，验证信号质量
3. **定时每日运行**，积累信号数据
4. **分析信号表现**，对比预测vs实际
5. **逐步过渡到实盘**（先Paper Trading 1-2个月）

---

## 总结

✅ Paper Trading系统已完全就绪
✅ 支持策略筛选 + 模型打分
✅ 只生成信号，不实际下单
✅ 完整记录所有结果

**当前状态**: 系统可以运行，但可能需要调整筛选条件以匹配实际数据

**建议**: 先放宽KDJ条件(90→70)，测试能否筛选到候选标的

# KDJ指标计算说明

## 当前问题

KDJ指标计算可能存在偏差，需要使用更标准的计算方法。

## KDJ指标标准计算公式

### 1. 计算RSV (Raw Stochastic Value)
```
RSV = (Close - Lowest_Low_N) / (Highest_High_N - Lowest_Low_N) × 100
```
- N通常取9
- Lowest_Low_N: N周期内最低价
- Highest_High_N: N周期内最高价

### 2. 计算K值
```
K(today) = (2/3) × K(yesterday) + (1/3) × RSV(today)
```
- 初始K值 = 50

### 3. 计算D值
```
D(today) = (2/3) × D(yesterday) + (1/3) × K(today)
```
- 初始D值 = 50

### 4. 计算J值
```
J = 3 × K - 2 × D
```

## 当前实现

当前 [scripts/indicator_utils.py](scripts/indicator_utils.py) 的实现是正确的，使用的就是标准KDJ公式。

## 可能的问题原因

1. **周期参数**：
   - 当前使用9周期（标准）
   - 有些平台使用14周期
   - 建议保持9周期

2. **数据质量**：
   - K线数据是否完整
   - 是否有缺失值
   - 时间对齐是否正确

3. **阈值设置**：
   - 当前策略使用 J > 70
   - 可能需要根据实际数据调整
   - 建议先验证数据分布

## 验证方法

### 1. 手动验证某个币种的KDJ
```python
import pandas as pd
from scripts.indicator_utils import compute_kdj

# 加载数据
df = pd.read_csv('data/hourly_klines/DEXE_USDT_USDT_1d.csv')

# 计算KDJ
k, d, j = compute_kdj(df[['high', 'low', 'close']])

# 查看最近的KDJ值
print(df[['timestamp', 'close']].tail(10))
print(f"K: {k.tail(10).values}")
print(f"D: {d.tail(10).values}")
print(f"J: {j.tail(10).values}")
```

### 2. 与TradingView对比
- 访问 TradingView
- 找到同一个币种
- 添加KDJ指标
- 对比同一时间点的K、D、J值

### 3. 检查J值分布
```python
# 统计所有币种的J值分布
j_values = []
for file in Path('data/hourly_klines').glob('*.csv'):
    df = pd.read_csv(file)
    _, _, j = compute_kdj(df[['high', 'low', 'close']])
    j_values.extend(j.dropna().values)

# 查看分布
import numpy as np
percentiles = np.percentile(j_values, [10, 25, 50, 75, 90, 95, 99])
print(f"J值分布: {percentiles}")
```

## 推荐操作

### 选项A: 验证当前KDJ计算（推荐）

运行验证脚本检查KDJ计算是否正确：
```bash
python scripts/verify_kdj.py
```

### 选项B: 调整KDJ阈值

如果计算正确但阈值太严格：
- 当前：J > 70
- 可调整为：J > 60 或 J > 50
- 或使用百分位数：J > 80分位

### 选项C: 使用不同的KDJ参数

修改周期参数：
- 短周期（更敏感）：length=5
- 标准周期：length=9
- 长周期（更平滑）：length=14

## 如何修改KDJ阈值

编辑 [scripts/paper_trader.py](scripts/paper_trader.py) 第139行：

```python
# 原来
if kdj_j < 70:
    continue

# 改为
if kdj_j < 60:  # 降低到60
    continue
```

---

## 实际操作建议

1. **先验证数据**：
   - 运行 `python scripts/verify_kdj.py`
   - 检查KDJ计算是否合理

2. **查看数据分布**：
   - 统计所有币种的J值分布
   - 确定合理的阈值

3. **对比参考平台**：
   - 选择1-2个币种
   - 在TradingView上查看KDJ
   - 对比我们的计算结果

4. **调整策略参数**：
   - 根据验证结果调整阈值
   - 重新运行Paper Trading
   - 观察信号数量是否合理

---

需要我创建KDJ验证脚本吗？

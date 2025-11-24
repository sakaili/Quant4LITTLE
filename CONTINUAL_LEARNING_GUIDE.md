# 持续学习系统使用指南

## 概述

持续学习（Continual Learning）系统让模型能够根据实盘交易结果不断进化，持续适应市场变化。

### 核心理念

```
回测数据训练初始模型 → 实盘使用 → 记录实盘结果 → 增量学习 → 模型进化 → 循环
```

### 系统组成

1. **[live_trade_recorder.py](scripts/live_trade_recorder.py)** - 实盘交易记录器
2. **[continual_learner.py](scripts/modeling/continual_learner.py)** - 持续学习引擎
3. **定时任务** - 自动触发模型更新

---

## 一、实盘交易记录

### 1.1 记录开仓

每次开仓时记录交易信息和模型预测：

```bash
python scripts/live_trade_recorder.py \
  --action entry \
  --symbol BTCUSDT \
  --price 45000 \
  --size 0.1 \
  --model-score 0.3456 \
  --model-class 2 \
  --notes "模型推荐，技术形态良好"
```

**输出**:
```
✓ 记录开仓: BTCUSDT_20250120_143022
  Symbol: BTCUSDT
  Entry Price: 45000
  Position Size: 0.1
  Model Score: 0.3456
  Model Class: 2
```

### 1.2 记录平仓

交易结束时记录退出价格：

```bash
python scripts/live_trade_recorder.py \
  --action exit \
  --trade-id BTCUSDT_20250120_143022 \
  --price 46500
```

**输出**:
```
✓ 记录平仓: BTCUSDT_20250120_143022
  Exit Price: 46500
  PnL: 150.00 (3.33%)
  Actual Class: 2
  Model Predicted: 2 (✓)
```

系统会自动：
- 计算实际收益率
- 根据阈值判断实际类别（0/1/2）
- 比较模型预测与实际结果

### 1.3 查看交易记录

```bash
# 查看未平仓位置
python scripts/live_trade_recorder.py --action list

# 查看交易汇总
python scripts/live_trade_recorder.py --action summary

# 查看最近30天模型准确率
python scripts/live_trade_recorder.py --action accuracy --days 30
```

### 1.4 集成到交易系统

在 `live_trader.py` 或 `sim_trader.py` 中集成记录器：

```python
from scripts.live_trade_recorder import LiveTradeRecorder

class Trader:
    def __init__(self):
        self.recorder = LiveTradeRecorder()

    def open_position(self, symbol, price, model_score, model_class):
        # 执行开仓操作...

        # 记录交易
        trade_id = self.recorder.record_entry(
            symbol=symbol,
            entry_price=price,
            position_size=0.1,
            model_score=model_score,
            model_class=model_class
        )

        return trade_id

    def close_position(self, trade_id, exit_price):
        # 执行平仓操作...

        # 记录平仓
        self.recorder.record_exit(trade_id, exit_price)
```

---

## 二、持续学习策略

### 2.1 两种学习模式

#### 增量学习（Incremental Learning）

**触发条件**:
- 有10+个新交易样本
- 距离上次更新 < 30天

**特点**:
- 快速（5-10个epoch）
- 使用小学习率（1e-4）微调
- 保留旧知识，避免灾难性遗忘

**适用场景**:
- 日常模型微调
- 适应短期市场变化

```bash
python scripts/modeling/continual_learner.py --mode incremental --epochs 10
```

#### 完全重训练（Full Retrain）

**触发条件**:
- 距离上次更新 >= 30天
- 或累积样本数显著增加

**特点**:
- 耗时较长（50个epoch）
- 重新划分训练/验证/测试集
- 重新计算归一化统计

**适用场景**:
- 累积大量新数据后
- 模型性能明显下降时

```bash
python scripts/modeling/continual_learner.py --mode retrain --epochs 50
```

### 2.2 自动更新工作流

**推荐方式**: 使用自动模式，系统会智能选择更新策略

```bash
python scripts/modeling/continual_learner.py --mode auto
```

**工作流程**:

```
1. 收集实盘新样本
   ↓
2. 评估当前模型性能（最近7天准确率）
   ↓
3. 决定更新策略
   - 样本数 >= 10 且 时间 < 30天 → 增量学习
   - 时间 >= 30天 → 完全重训练
   - 其他 → 跳过
   ↓
4. 执行模型更新
   - 备份当前模型到 models/backups/
   - 训练新模型
   - 保存更新后的模型
   ↓
5. 评估更新后性能
```

### 2.3 定时任务设置

#### Linux/Mac (cron)

编辑 crontab:
```bash
crontab -e
```

添加每周日凌晨2点自动更新：
```cron
0 2 * * 0 cd /path/to/Quant4Little && python scripts/modeling/continual_learner.py --mode auto >> logs/continual_learning.log 2>&1
```

#### Windows (任务计划程序)

创建批处理文件 `update_model.bat`:
```bat
@echo off
cd /d F:\2025\Quant4Little
python scripts/modeling/continual_learner.py --mode auto >> logs\continual_learning.log 2>&1
```

在任务计划程序中设置每周运行。

---

## 三、数据管理

### 3.1 实盘交易记录文件

**位置**: `data/live_trades.csv`

**字段**:
```
trade_id         - 交易唯一ID
symbol           - 标的符号
entry_time       - 开仓时间
entry_price      - 开仓价格
exit_time        - 平仓时间
exit_price       - 平仓价格
position_size    - 仓位大小
pnl              - 绝对盈亏
pnl_pct          - 收益率
status           - 状态 (open/closed)
model_score      - 模型预测分数
model_class      - 模型预测类别
actual_class     - 实际类别（平仓后计算）
notes            - 备注
```

### 3.2 合并历史数据

为了完全重训练，需要合并回测数据和实盘数据：

```bash
# 创建合并脚本
python -c "
import pandas as pd

# 读取回测交易
backtest = pd.read_csv('data/backtest_trades.csv')

# 读取实盘交易（只保留已平仓）
live = pd.read_csv('data/live_trades.csv')
live = live[live['status'] == 'closed']

# 合并
all_trades = pd.concat([backtest, live], ignore_index=True)
all_trades.to_csv('data/all_trades.csv', index=False)

print(f'合并完成: {len(all_trades)} 条记录')
"
```

### 3.3 模型备份

每次更新前，系统会自动备份模型到 `models/backups/`：

```
models/backups/
  ├── 20250120_143022_before_incremental/
  │   ├── rank_model.pt
  │   └── rank_model_meta.json
  ├── 20250127_020000_before_retrain/
  │   ├── rank_model.pt
  │   └── rank_model_meta.json
  ...
```

**恢复旧模型**:
```bash
# 列出所有备份
ls -lt models/backups/

# 恢复指定备份
cp models/backups/20250120_143022_before_incremental/* models/
```

---

## 四、监控与评估

### 4.1 模型性能追踪

创建监控脚本 `monitor_model.py`:

```python
from scripts.live_trade_recorder import LiveTradeRecorder

recorder = LiveTradeRecorder()

# 评估不同时间窗口
for days in [7, 14, 30]:
    metrics = recorder.get_model_accuracy(days=days)
    print(f"\n最近{days}天:")
    print(f"  准确率: {metrics.get('accuracy', 0):.2%}")
    print(f"  平均收益: {metrics.get('avg_pnl_pct', 0):.2%}")
    print(f"  胜率: {metrics.get('win_rate', 0):.2%}")
```

### 4.2 关键指标

**模型预测准确率**: 实际类别 == 预测类别的比例
```python
accuracy = correct_predictions / total_predictions
```

**Top-K收益率**: 选择模型分数最高的K个标的的平均收益
```python
top_k_return = top_k_trades['pnl_pct'].mean()
```

**预测偏差**: 模型是否系统性偏向某个类别
```python
class_distribution = {
    0: (predictions == 0).mean(),
    1: (predictions == 1).mean(),
    2: (predictions == 2).mean(),
}
```

### 4.3 何时触发重训练

**建议触发条件**:

1. **准确率下降**: 最近7天准确率 < 40%
2. **收益下降**: 最近30天平均收益 < 0
3. **累积样本**: 新样本数 >= 50个
4. **定期更新**: 每30天自动重训练

---

## 五、最佳实践

### 5.1 数据质量

**确保记录的完整性**:
- ✅ 每笔交易都记录开仓和平仓
- ✅ 记录模型预测分数和类别
- ✅ 只使用完整的交易（有entry和exit）进行学习
- ❌ 不要遗漏任何交易记录

### 5.2 学习频率

**推荐策略**:

```
新样本 10-30个 → 每周增量学习
新样本 30-50个 → 每2周增量学习
新样本 50+个   → 完全重训练
距上次更新30天 → 强制完全重训练
```

### 5.3 避免过拟合

**问题**: 模型只学习最近的交易，遗忘历史经验

**解决方案**:
1. **保留历史数据**: 合并回测和实盘数据
2. **使用小学习率**: 增量学习使用1e-4而非1e-3
3. **Early Stopping**: 监控验证损失，防止过拟合
4. **定期完全重训练**: 每30天用所有数据重训练

### 5.4 灾难性遗忘（Catastrophic Forgetting）

**问题**: 学习新数据时完全遗忘旧知识

**解决方案**:

1. **增量学习时保留历史样本**:
```python
# 不只用新样本，混合一些旧样本
old_samples = random.sample(historical_samples, k=50)
training_samples = new_samples + old_samples
```

2. **使用Elastic Weight Consolidation (EWC)**:
给重要参数添加正则化约束（高级技巧，可选）

3. **定期完全重训练**:
用所有历史数据重新训练，重置模型

---

## 六、故障排查

### 问题1: 新样本不足

**现象**:
```
✗ 没有新样本，跳过更新
```

**原因**:
- `live_trades.csv` 中没有已平仓交易
- 或所有交易都已经用于之前的学习

**解决**:
- 确保交易平仓后调用 `record_exit()`
- 检查 `models/rank_model_meta.json` 中的 `last_updated` 时间戳

### 问题2: 模型准确率下降

**现象**:
```
最近7天准确率: 25%  (比初始28.6%还低)
```

**可能原因**:
1. 市场环境变化，模型失效
2. 新样本质量差（标注错误）
3. 过拟合到新数据

**解决**:
1. 检查新交易的质量和标注
2. 恢复到旧版本模型测试
3. 收集更多样本后完全重训练

### 问题3: 训练过慢

**现象**:
完全重训练需要很长时间

**优化**:
1. 减少epoch数: `--epochs 30`
2. 增大batch size: `--batch-size 64`
3. 使用GPU: `--device cuda`

---

## 七、完整使用示例

### 7.1 日常交易流程

```bash
# 1. 早上扫描候选
python scripts/daily_candidate_scan.py

# 2. 使用模型对候选打分
python scripts/use_ranker_model.py --date 20250120 --top-k 20

# 3. 选择Top 5标的开仓（手动或自动）
python scripts/live_trade_recorder.py \
  --action entry --symbol BTCUSDT --price 45000 \
  --model-score 0.4523 --model-class 2

# 4. 5天后平仓
python scripts/live_trade_recorder.py \
  --action exit --trade-id BTCUSDT_20250120_143022 --price 46500

# 5. 查看模型表现
python scripts/live_trade_recorder.py --action accuracy --days 7
```

### 7.2 周末模型更新

```bash
# 每周日凌晨自动运行
python scripts/modeling/continual_learner.py --mode auto
```

**可能的输出**:

```
============================================================
自动持续学习工作流
============================================================

[1/4] 收集实盘新样本...
  距离上次更新: 7 天
✓ 发现 12 笔已平仓交易

[2/4] 评估当前模型性能...
  最近7天: {'samples': 12, 'loss': 0.8234, 'accuracy': 0.583}

[3/4] 决定更新策略...
  → 新样本数 12 >= 10
  → 触发增量更新

[4/4] 执行模型更新...
✓ 模型已备份到: models/backups/20250127_020000_before_incremental
开始增量学习 (12 个新样本)...
  Epoch 1: loss=0.7234, acc=0.667
  Epoch 2: loss=0.6891, acc=0.750
  ...
  Epoch 10: loss=0.5234, acc=0.833
✓ 增量学习完成
✓ 模型已保存: models/rank_model.pt

[5/5] 评估更新后模型...
  更新后性能: {'samples': 12, 'loss': 0.7123, 'accuracy': 0.667}
  准确率变化: +0.084

✓ 持续学习工作流完成
```

---

## 八、进阶配置

### 8.1 自定义学习参数

编辑 `continual_learner.py` 中的配置：

```python
class ContinualLearner:
    def __init__(self, ...):
        # 持续学习配置
        self.min_new_samples = 10        # 触发增量学习的最小新样本数
        self.retrain_interval_days = 30  # 完全重训练的间隔天数
        self.learning_rate = 1e-4        # 增量学习学习率
```

### 8.2 集成到监控系统

将模型性能指标发送到监控平台：

```python
import requests

def report_metrics_to_monitor(metrics):
    # 发送到Prometheus/Grafana
    requests.post('http://monitor.example.com/metrics', json=metrics)
```

### 8.3 A/B测试

同时运行多个模型版本：

```python
# 保留旧模型作为baseline
models = {
    'v1_baseline': load_model('models/baseline/'),
    'v2_continual': load_model('models/'),
}

# 对比预测结果
for symbol in candidates:
    for name, model in models.items():
        score = predict(model, symbol)
        print(f"{name}: {score}")
```

---

## 总结

持续学习系统让模型能够：

✅ **自动适应**: 根据实盘反馈不断进化
✅ **保留记忆**: 避免灾难性遗忘历史知识
✅ **智能更新**: 根据数据量和时间自动选择更新策略
✅ **可追溯**: 完整记录每笔交易和模型版本

**关键成功因素**:
1. 完整记录每笔交易（开仓+平仓）
2. 定期评估模型性能
3. 合理设置更新频率
4. 备份历史模型版本

**下一步**:
- 开始记录实盘交易
- 运行第一次自动更新
- 持续监控模型表现

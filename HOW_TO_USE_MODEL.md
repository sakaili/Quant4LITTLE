# 如何使用排序模型

## 一、模型解释

### 1. 模型功能

这个模型的作用是**对候选标的进行打分和排序**，预测它们未来5日的收益潜力。

**输入**:
- 候选标的的历史K线数据（日线+小时线）
- 技术指标（EMA、ATR、成交量等）

**输出**:
- `expected_value`: 期望收益分数（-0.5到0.5）
- `pred_class`: 预测类别（0=差, 1=中, 2=优）
- `prob_class0/1/2`: 三个类别的概率

### 2. 分类标准

模型将标的分为三类，基于预期的5日收益表现：

```
Class 0 (差): 预期收益 < 0.3%
Class 1 (中): 0.3% ≤ 预期收益 < 1.2%
Class 2 (优): 预期收益 ≥ 1.2%
```

**注意**: 这些阈值是从历史回测数据统计得出的，可能需要根据市场环境调整。

### 3. 当前模型性能

从50轮训练结果来看：

```
验证集准确率: 84.6% (Epoch 5最佳)
测试集准确率: 28.6%
泛化差距: -56%
```

**⚠️ 重要提示**:
- 测试集表现较差，主要因为样本量太小（只有7个测试样本）
- 建议收集更多历史数据重新训练
- 当前模型仅供参考，不建议直接用于实盘交易

---

## 二、使用方法

### 方法1: 使用Python脚本（推荐）

#### 基本用法

```bash
# 对2025年1月20日的候选进行打分
python scripts/use_ranker_model.py --date 20250120

# 输出前20名
python scripts/use_ranker_model.py --date 20250120 --top-k 20

# 保存到指定文件
python scripts/use_ranker_model.py --date 20250120 --output results.csv
```

#### 输出示例

```
============================================================
排序结果 (前20名)
============================================================
    symbol  expected_value  pred_class  prob_class0  prob_class1  prob_class2
0   BTCUSDT          0.4523           2       0.0234       0.1543       0.8223
1   ETHUSDT          0.3891           2       0.0456       0.2134       0.7410
2   BNBUSDT          0.2456           1       0.1234       0.6543       0.2223
...

统计信息:
  总候选数: 80
  成功打分: 75
  预测Class 0 (差): 25
  预测Class 1 (中): 28
  预测Class 2 (优): 22
  平均期望值: 0.0123
  Top20平均期望值: 0.2456
```

### 方法2: 在代码中使用

```python
from pathlib import Path
import torch
import json
from scripts.modeling.model import TransformerRanker

# 1. 加载模型
model_dir = Path("models")
with open(model_dir / "rank_model_meta.json") as f:
    meta = json.load(f)

model = TransformerRanker(**meta["model_kwargs"])
model.load_state_dict(torch.load(model_dir / "rank_model.pt"))
model.eval()

# 2. 准备特征（假设你已经有了归一化后的特征）
# features: shape (1, 15) - 表格特征
# sequences: shape (1, 24, 5) - 序列特征

# 3. 预测
with torch.no_grad():
    logits = model(sequences, features)
    probs = torch.softmax(logits, dim=-1)
    pred_class = logits.argmax(dim=-1).item()

    # 计算期望值
    class_values = torch.tensor([-0.5, 0.0, 0.5])
    expected_value = (probs * class_values).sum().item()

print(f"预测类别: {pred_class}")
print(f"期望收益: {expected_value:.4f}")
print(f"概率分布: {probs}")
```

---

## 三、集成到交易流程

### 1. 替代原有的排序逻辑

原有的选股流程在 `daily_candidate_scan.py` 中：

```python
# 原有逻辑: 按EMA10排序选择底部80名
candidates = all_candidates.nsmallest(80, "ema10")
```

可以改为：

```python
# 新逻辑: 先粗选，再用模型精选
candidates = all_candidates.nsmallest(200, "ema10")  # 粗选200个
candidates = rank_with_model(candidates, date)  # 模型打分
candidates = candidates.nlargest(80, "expected_value")  # 精选80个
```

### 2. 修改后的候选扫描脚本示例

```python
def select_candidates_with_model(df: pd.DataFrame, date: str, model_dir: Path) -> pd.DataFrame:
    """使用模型对候选进行排序选择"""
    from scripts.use_ranker_model import load_model, score_candidates

    # 加载模型
    model, meta = load_model(model_dir)

    # 对候选打分
    scored = score_candidates(
        df, model, meta,
        daily_dir=Path("data/daily_klines"),
        hourly_dir=Path("data/hourly_klines"),
        signal_date=date
    )

    # 按期望值排序
    return scored.nlargest(80, "expected_value")
```

### 3. 集成到live_trader或sim_trader

在选股阶段使用模型：

```python
# 在 live_trader.py 或 sim_trader.py 中
from scripts.use_ranker_model import load_model, score_candidates

class Trader:
    def __init__(self):
        # 加载排序模型
        self.model, self.model_meta = load_model(Path("models"))

    def select_positions(self, candidates: pd.DataFrame, date: str):
        """使用模型选择最优标的"""
        # 对候选打分
        scored = score_candidates(
            candidates, self.model, self.model_meta,
            self.daily_dir, self.hourly_dir, date
        )

        # 选择Top K
        top_candidates = scored.nlargest(self.max_positions, "expected_value")

        # 只选择pred_class=2的标的（预期优质）
        top_candidates = top_candidates[top_candidates["pred_class"] == 2]

        return top_candidates
```

---

## 四、改进建议

### 1. 数据层面

**当前问题**:
- 总样本: 88个（太少）
- 测试集: 7个（不可靠）
- 时间跨度: 仅覆盖26天

**改进方案**:
```bash
# 1. 扩展回测时间范围，收集更多交易数据
python scripts/backtester.py --start 2024-01-01 --end 2025-01-31

# 2. 重新训练模型
python scripts/modeling/train_ranker.py \
  --backtest-csv data/backtest_trades.csv \
  --epochs 50 \
  --batch-size 32
```

**目标**: 至少500-1000个样本，测试集50+样本

### 2. 模型层面

**当前配置**:
```python
TransformerRanker(
    seq_len=24,      # 24小时历史
    seq_dim=5,       # 5维序列特征
    feature_dim=15   # 15维表格特征
)
```

**可尝试的改进**:
- 增加序列长度: `seq_len=48` (48小时)
- 添加更多技术指标
- 尝试不同的模型架构（LSTM、GRU等）
- 改为回归任务（直接预测收益率）

### 3. 特征工程

**当前特征** (从 `rank_model_meta.json`):
```json
"feature_names": [
  "ema10", "ema20", "ema30",
  "ema10_over_ema30", "ema20_over_ema30",
  "atr14", "atr14_over_close",
  "close", "funding_rate",
  "log_quote_volume", "log_market_cap",
  "ret_mean_5d", "ret_std_5d",
  "ema10_slope", "ema20_slope"
]
```

**可添加的特征**:
- RSI、MACD等更多技术指标
- 大盘相关性特征
- 成交量异动特征
- 资金流向特征

### 4. 评估策略

不要只看准确率，更要关注：

```python
# 评估Top K的实际收益
def evaluate_topk_returns(results, k=20):
    top_k = results.nlargest(k, "expected_value")
    actual_returns = calculate_future_returns(top_k["symbol"])

    print(f"Top {k} 平均收益: {actual_returns.mean():.2%}")
    print(f"Top {k} 胜率: {(actual_returns > 0).mean():.2%}")
    print(f"Top {k} 最大收益: {actual_returns.max():.2%}")
    print(f"Top {k} 最大亏损: {actual_returns.min():.2%}")
```

---

## 五、风险提示

1. **模型不可靠**: 当前测试准确率只有28.6%，不建议直接用于实盘
2. **过拟合风险**: 验证/训练损失比2.3，模型严重过拟合
3. **样本偏差**: 88个样本可能不代表所有市场环境
4. **市场变化**: 模型是基于历史数据训练的，市场环境变化后可能失效

**建议**:
- 先在模拟环境测试至少1-2个月
- 持续监控模型预测准确率
- 定期用新数据重新训练模型
- 不要完全依赖模型，结合其他策略和风控

---

## 六、快速测试

如果想测试模型是否工作正常：

```bash
# 1. 检查是否有可用的候选文件
ls data/daily_scans/

# 2. 选择一个日期进行测试（例如2025-01-20）
python scripts/use_ranker_model.py --date 20250120 --top-k 10

# 3. 查看输出的排序结果
cat data/ranked_candidates/ranked_20250120.csv
```

如果成功运行并输出了排序结果，说明模型可以使用。

---

**总结**: 模型已经训练完成并可以使用，但由于数据量太小导致泛化能力较差。建议先收集更多历史数据，重新训练后再考虑实际应用。

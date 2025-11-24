# 修复记录：数据泄露问题（Look-ahead Bias）

## 问题描述

在训练深度学习排序模型时，发现 `data/daily_scans/` 目录下的历史候选扫描文件存在严重的**数据泄露（Look-ahead Bias）**问题：

### 问题表现
```csv
# candidates_20251025.csv (生成于 2025-11-21)
symbol,base,timestamp,quote_volume,market_cap,funding_rate,ema10,ema20,ema30,atr14,latest_close,as_of
G/USDT:USDT,G,2025-11-21 00:00:00+00:00,1603722.978215,,1.551e-05,0.005613,0.006010,0.006415,0.000429,0.005372,2025-10-25
```

**问题点**：
- `timestamp` 列显示 `2025-11-21`（今天）
- `as_of` 列显示 `2025-10-25`（正确的信号日期）
- 这意味着在训练时，模型看到的是"未来"的时间戳信息

### 根本原因

1. **`build_candidates()` 函数**：
   - 调用 `fetch_bulk_history()` 时没有指定截止日期
   - 导致总是获取最新的K线数据
   - `timestamp` 使用了最新K线的时间戳而不是信号日期

2. **`run_scan()` 函数**：
   - `fetch_24h_tickers()` 总是获取当前时刻的最新ticker数据
   - `quote_volume` 和 `market_cap` 也是今天的数据

3. **`latest_kdj_j_above_threshold()` 函数**：
   - 使用 `datetime.now()` 作为结束时间
   - 没有截止到历史的 `as_of_date`

---

## 修复方案

### 1. 修复 `build_candidates()` 函数

**文件**: [scripts/daily_candidate_scan.py](scripts/daily_candidate_scan.py)

**修改内容**：
```python
def build_candidates(
    fetcher: BinanceDataFetcher,
    symbols: Iterable[str],
    meta_map: Dict[str, SymbolMetadata],
    *,
    timeframe: str,
    funding_cooldown: float,
    as_of_date: date,  # 新增参数
) -> List[Candidate]:
    """
    构建候选列表，使用截止到 as_of_date 的历史数据。
    重要：为避免数据泄露，只使用 as_of_date 及之前的数据。
    """
    # 获取历史数据，截止到 as_of_date
    end_dt = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=200)

    histories = fetcher.fetch_bulk_history(
        symbols,
        start=start_dt,
        end=end_dt,  # 明确指定截止日期
        timeframe=timeframe
    )

    for symbol, history in histories.items():
        # 只保留 <= as_of_date 的数据
        history = history[history["timestamp"].dt.date <= as_of_date].copy()

        # ... 其余逻辑

        # 使用 as_of_date 作为信号时间戳
        signal_timestamp = pd.Timestamp(as_of_date, tz=timezone.utc)

        rows.append(
            Candidate(
                symbol=symbol,
                timestamp=signal_timestamp,  # 使用信号日期
                # ...
            )
        )
```

### 2. 修复 `latest_kdj_j_above_threshold()` 函数

**修改内容**：
```python
def latest_kdj_j_above_threshold(
    fetcher: BinanceDataFetcher,
    symbol: str,
    *,
    threshold: float = 90.0,
    hours_lookback: int = 72,
    as_of_date: Optional[date] = None,  # 新增参数
) -> bool:
    """
    如果指定 as_of_date，则只使用该日期及之前的数据（避免数据泄露）。
    """
    if as_of_date is not None:
        # 历史模式：截止到 as_of_date 结束
        end = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)
    else:
        # 实时模式：使用当前时间
        end = datetime.now(timezone.utc)

    start = end - timedelta(hours=hours_lookback)
    frame = fetcher.fetch_klines(symbol, start=start, end=end, timeframe="1h")

    # 如果指定了 as_of_date，再次过滤确保不使用未来数据
    if as_of_date is not None:
        frame = frame[frame["timestamp"].dt.date <= as_of_date]

    # ... 计算 KDJ
```

### 3. 更新 `data_builder.py`

**文件**: [scripts/modeling/data_builder.py](scripts/modeling/data_builder.py)

**修改内容**：
```python
def parse_signal_file(path: Path) -> Tuple[date, pd.DataFrame]:
    """
    解析候选扫描文件，返回信号日期和候选DataFrame。

    注意：使用文件名中的日期作为信号日期，而不是CSV中的timestamp列，
    因为timestamp可能包含数据泄露。
    """
    as_of_str = path.stem.split("_")[1]
    as_of_date = datetime.strptime(as_of_str, "%Y%m%d").date()
    df = pd.read_csv(path)

    # 验证 as_of 字段与文件名一致
    if "as_of" in df.columns and not df.empty:
        csv_date = pd.to_datetime(df["as_of"].iloc[0]).date()
        if csv_date != as_of_date:
            logger.warning(
                f"文件 {path.name} 中的 as_of ({csv_date}) 与文件名日期 ({as_of_date}) 不一致"
            )

    return as_of_date, df
```

---

## 如何重新生成历史数据

### 方法1：使用批量重新生成脚本

```bash
# 重新生成指定日期范围的扫描数据
python scripts/regenerate_historical_scans.py \
  --start 2025-01-01 \
  --end 2025-11-20 \
  --bottom-n 80 \
  --skip-existing  # 跳过已存在的文件
```

### 方法2：手动逐日生成

```bash
# 单独重新生成某一天的数据
python scripts/daily_candidate_scan.py --as-of 2025-10-25 --bottom-n 80
```

---

## 验证修复

### 1. 检查修复后的文件

```bash
# 查看修复后的候选文件
head -2 data/daily_scans/candidates_20251025.csv
```

**期望输出**：
```csv
symbol,base,timestamp,quote_volume,market_cap,funding_rate,ema10,ema20,ema30,atr14,latest_close,as_of
SLP/USDT:USDT,SLP,2025-10-25 00:00:00+00:00,1243796.479696,,5e-05,0.001242,0.001323,0.001394,0.000121,0.001217,2025-10-25
```

✅ **`timestamp` 和 `as_of` 都是 `2025-10-25`**

### 2. 重新训练模型

```bash
# 使用修复后的数据重新训练排序模型
python scripts/modeling/train_ranker.py \
  --candidates-dir data/daily_scans \
  --backtest-csv data/backtest_trades.csv \
  --daily-dir data/daily_klines \
  --hourly-dir data/hourly_klines \
  --output-dir models
```

---

## 影响范围

### ✅ 已修复
- [x] `daily_candidate_scan.py` - 历史扫描时使用正确的时间截止点
- [x] `data_builder.py` - 训练数据构建时使用文件名日期
- [x] `regenerate_historical_scans.py` - 新增批量重新生成脚本

### ⚠️ 需要注意
- 历史的 **资金费率（funding_rate）** 仍然使用当前查询的值
  - 原因：Binance API 不提供历史资金费率查询接口
  - 影响：较小，因为资金费率变化相对缓慢
  - 建议：如需更精确，可从外部数据源补充历史费率

### 📝 后续建议
1. 重新生成所有历史扫描文件（`2025-01-01` 至 `2025-11-20`）
2. 重新运行回测验证一致性
3. 重新训练深度学习模型
4. 对比修复前后的模型性能差异

---

## 关键收获

### 时间序列机器学习的黄金法则
> **永远不要使用未来信息来预测过去**

在构建金融时间序列模型时，必须确保：
1. 特征提取只使用 `t` 时刻及之前的数据
2. 标签（label）对应 `t+1` 或更晚时刻的结果
3. 数据切分必须按时间顺序（train/val/test split by date）
4. 回测时严格模拟真实交易场景（时间延迟、滑点、手续费）

---

**修复日期**: 2025-11-21
**修复者**: Claude Code
**影响版本**: v0.1.0+

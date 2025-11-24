# Quant4LITTLE —— 空气币日线做空策略

## 项目概览
本仓库用于搭建一个完整的量化策略框架，聚焦于 Binance USDT 永续市场中的低流动性“空气币”。策略逻辑：
1. **标的筛选**：成交额/市值倒数的币种，剔除主流币与资金费率过低的合约。
2. **日线趋势判定**：EMA10 < EMA20 < EMA30 且 ATR 未出现极端放大（可按需调整）。
3. **小时级择时**：当日最新一根 1h K 线的 KDJ J 值达到指定阈值（默认 > 90）才触发入场。
4. **风险控制**：开仓后设置 30% 止盈、150% 止损，支持滑点和手续费模拟，最大持仓 30 个币种。

主要模块：
- `scripts/data_fetcher.py`：封装 ccxt 的数据获取，支持主网 / 测试网切换。
- `scripts/daily_candidate_scan.py`：日常候选池筛选脚本。
- `scripts/backtester.py`：事件驱动回测器，包含仓位管理 / 风控。
- `scripts/live_trader.py`：实时信号与下单工具（可纸面模式或实盘）。
- `scripts/sim_trader.py`：生成模拟下单计划，不实际下单。

## 环境配置
1. 安装 Python 3.10+，创建虚拟环境并安装依赖：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. 设置环境变量（主网 / 测试网均可）：
   ```bash
   export BINANCE_API_KEY="你的APIKEY"
   export BINANCE_API_SECRET="你的APISECRET"
   ```
   如果需要代理，设置 `HTTP_PROXY`/`HTTPS_PROXY`。

## 回测
```bash
python scripts/backtester.py \
  --start 2025-01-01 \
  --end 2025-11-01 \
  --max-positions 30 \
  --bottom-n 100
```
输出包含权益曲线、交易明细、Sharpe/回撤等指标。真实逻辑可在此基础上调参。

## 日常信号扫描
```bash
python scripts/daily_candidate_scan.py --as-of 2025-11-19
```
生成 `data/daily_scans/candidates_<date>.csv` 供复盘或回测使用。

## 实时信号 / 下单
### 纸面模式（仅输出信号）
```bash
python scripts/live_trader.py --refresh-pool --bottom-n 100 --as-of 2025-11-19 --paper
```
输出 Step1~3 的长表及最终信号数量，不执行下单。

### 实盘 / 测试网下单示例
```bash
python scripts/live_trader.py \
  --refresh-pool \
  --bottom-n 100 \
  --as-of 2025-11-19 \
  --use-testnet            # 若用测试网
  --per-trade-pct 0.01 \
  --leverage 2 \
  --min-margin 10
```
脚本逻辑：
- 先读取账户净值，按 `max(净值*pct, min_margin) * leverage` 计算名义；若失败则回退 `min_margin * leverage`。
- 市价开空，随后提交 TP/SL 触发单（30% 止盈、150% 止损）。若 reduceOnly 不被支持，会自动降级为非 reduceOnly 触发单。
- `--paper` 保持实时信号，但不触发真实订单。

> **提示**：实盘前请在测试网充分验证；确保 API key 只拥有期货所需权限。可在脚本中加入持仓数检查（例如不足 20 个仓位再补）以及更完善的异常告警。

## 深度学习排序（Top-K 选币）
1. **准备训练数据**：运行 `scripts/daily_candidate_scan.py` 生成历史 `candidates_YYYYMMDD.csv`，并确保 `data/daily_klines/`、`data/hourly_klines/` 与 `data/backtest_trades.csv` 就位。
2. **训练模型**：
   ```bash
   python scripts/modeling/train_ranker.py \
     --candidates-dir data/daily_scans \
     --backtest-csv data/backtest_trades.csv \
     --daily-dir data/daily_klines \
     --hourly-dir data/hourly_klines \
     --output-dir models
   ```
   脚本会生成 `models/rank_model.pt` 与 `models/rank_model_meta.json`。
3. **实时推理**：`live_trader.py` 会在 Step3 后调用 Transformer 排序器，对所有满足 KDJ/EMA 条件的品类做 softmax 归一化评分，并只保留得分最高的 `--max-positions`（默认 20）个标的；其余信号会被丢弃。若模型文件缺失或使用 `--disable-ranker`，则退回原有基于 KDJ 的排序。

## Git / 部署
1. 初始化 Git 并推送到 GitHub：
   ```bash
   git init
   git add .
   git commit -m "初始化量化策略项目"
   git remote add origin https://github.com/xxx/Quant4LITTLE.git
   git push -u origin main
   ```
2. 在 ECS（Ubuntu）上 `git clone` 仓库，配置定时任务（cron 或 systemd timer）每小时执行一次 `live_trader.py --paper` 或实盘版本。
3. 日志与数据会按 `.gitignore` 设置被忽略，确保敏感信息不进仓库。

## 后续工作方向
- 按需实现“持仓不足自动补齐”等实盘逻辑。
- 增加监控 / 告警（如 Telegram/钉钉机器人）。
- 完善资金费率模拟与仓位滑点建模，在回测与实盘间保持一致性。

如有问题或建议，欢迎在仓库提交 Issue / PR。祝交易顺利！

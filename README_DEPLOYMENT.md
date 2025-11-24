# Quant4Little - ECS部署指南

完整的ECS服务器部署文档，适用于2CPU/1GB RAM的低配服务器。

## 📋 目录

- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
- [策略说明](#策略说明)
- [常见问题](#常见问题)

---

## 系统要求

### 硬件要求
- **CPU**: 2核心
- **内存**: 1GB RAM + 2GB Swap
- **存储**: 至少10GB可用空间

### 软件要求
- **操作系统**: Ubuntu 20.04+ / CentOS 7+
- **Python**: 3.8+
- **Git**: 任意版本
- **网络**: 需访问币安API (可能需要代理)

---

## 快速开始

### 方法1: 一键部署 (推荐)

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/Quant4Little.git
cd Quant4Little

# 2. 配置API密钥
cp .env.example .env
nano .env  # 填入 BINANCE_API_KEY 和 BINANCE_API_SECRET

# 3. 运行部署脚本
bash deploy/setup.sh
```

就这么简单！脚本会自动完成所有配置。

---

## 详细步骤

### 步骤1: 创建Swap内存

由于服务器只有1GB RAM，需要创建2GB swap：

```bash
# 创建swap文件（需要root权限）
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 验证
free -h
```

### 步骤2: 安装Python依赖

```bash
# 安装轻量级依赖（使用ONNX，不含PyTorch）
pip3 install -r requirements_onnx.txt
```

**注意**: 使用 `requirements_onnx.txt` 而不是 `requirements.txt`，这样可以避免安装PyTorch（节省800MB+内存）。

### 步骤3: 配置API密钥

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置文件
nano .env
```

填入以下内容：

```bash
# 币安API配置
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# 测试模式 (True=测试网, False=实盘)
USE_TESTNET=False

# 代理配置 (如果需要)
# HTTPS_PROXY=http://your-proxy:port
```

### 步骤4: 转换模型为ONNX

如果您是从本地上传模型文件：

```bash
# 方法A: 在本地转换后上传
python3 scripts/convert_to_onnx.py
# 将 models/rank_model.onnx 和 models/rank_model_meta.json 上传到ECS

# 方法B: 在ECS上转换（需要先上传PyTorch模型）
# 如果有 models/rank_model.pt，运行:
python3 scripts/convert_to_onnx.py
```

### 步骤5: 设置定时任务

```bash
# 编辑crontab
crontab -e

# 添加以下任务（注意修改路径）
# 每日UTC 01:00 更新数据
0 1 * * * cd /path/to/Quant4Little && bash deploy/cron_update_data.sh >> logs/cron_update.log 2>&1

# 每日UTC 02:00 生成信号
0 2 * * * cd /path/to/Quant4Little && bash deploy/cron_generate_signals.sh >> logs/cron_signals.log 2>&1

# 每日UTC 02:30 执行交易
30 2 * * * cd /path/to/Quant4Little && bash deploy/cron_execute_trades.sh >> logs/cron_trades.log 2>&1
```

或者使用自动安装脚本：

```bash
bash deploy/setup.sh  # 会自动配置crontab
```

### 步骤6: 测试运行

```bash
# 测试模型加载
python3 -c "from scripts.lightweight_ranker import LightweightRanker; r = LightweightRanker(); print('✓ 模型加载成功')"

# 生成交易信号
python3 scripts/paper_trader.py --max-positions 5

# 查看生成的信号
ls -lh data/paper_trading/signals_*.csv
cat data/paper_trading/signals_$(date +%Y%m%d).csv

# 执行交易（确保USE_TESTNET=True用于测试）
python3 scripts/live_maker_trader.py
```

---

## 策略说明

### 交易策略

**做空垃圾币策略**

1. **筛选条件**:
   - EMA10 < EMA20 < EMA30 (底部形态)
   - KDJ_J > 50 (超买)
   - ATR波动率 > 2%
   - 成交量放大

2. **AI模型打分**:
   - 使用Transformer模型对候选币种打分
   - 优先选择Class 2 (优质)标的

3. **开仓参数**:
   - **资金管理**: 2%可用余额 × 2倍杠杆 = 每笔约4%
   - **最大持仓**: 10个
   - **订单类型**: Maker限价单（0.1%偏移）

4. **止盈止损**:
   - **止盈**: +30% (自动委托单)
   - **止损**: -200% (自动委托单)
   - **订单类型**: TAKE_PROFIT_MARKET / STOP_MARKET

### 执行流程

```
每日UTC 01:00 (北京09:00)
  └─ 更新数据 (日线+小时线)

每日UTC 02:00 (北京10:00)
  └─ 生成交易信号
      ├─ 策略筛选候选标的
      ├─ AI模型打分排序
      └─ 生成Top 20信号

每日UTC 02:30 (北京10:30)
  └─ 执行交易
      ├─ 读取今日信号
      ├─ 检查持仓数量
      ├─ 下Maker限价单
      ├─ 等待成交
      └─ 自动设置止盈止损
```

---

## 常见问题

### Q1: 如何查看交易日志?

```bash
# 查看最新日志
tail -f logs/cron_trades.log

# 查看历史日志
cat logs/cron_trades.log
```

### Q2: 如何手动执行交易?

```bash
# 生成信号
python3 scripts/paper_trader.py --max-positions 20

# 执行交易（会提示确认）
python3 scripts/live_maker_trader.py

# 自动确认（用于定时任务）
python3 scripts/live_maker_trader.py --auto-confirm
```

### Q3: 如何查看当前持仓?

```bash
# 查看持仓
python3 scripts/manage_positions.py

# 手动管理持仓（检查止盈止损）
python3 scripts/manage_positions.py --take-profit 30 --stop-loss 200
```

### Q4: 内存不足怎么办?

```bash
# 检查内存使用
free -h

# 检查swap
swapon --show

# 如果swap未启用，参考步骤1创建swap

# 清理缓存
sudo sync && sudo sysctl -w vm.drop_caches=3
```

### Q5: 如何更新代码?

```bash
cd /path/to/Quant4Little
git pull origin main

# 重新运行部署脚本（会保留.env配置）
bash deploy/setup.sh
```

### Q6: 如何从测试网切换到实盘?

编辑 `.env` 文件：

```bash
# 测试网
USE_TESTNET=True

# 实盘（⚠️ 真实资金）
USE_TESTNET=False
```

### Q7: 订单成交率低怎么办?

调整Maker偏移比例（`scripts/live_maker_trader.py:384`）：

```python
trader = LiveMakerTrader(
    ...
    maker_offset_pct=0.10,  # 0.10% → 0.05% (更容易成交)
)
```

### Q8: 如何修改止盈止损?

修改 `scripts/live_maker_trader.py:532-533`：

```python
tp_order, sl_order = trader.set_tp_sl_orders(
    ...
    take_profit_pct=30.0,   # 修改止盈比例
    stop_loss_pct=200.0     # 修改止损比例
)
```

### Q9: 如何禁用AI模型（节省资源）?

```bash
# 使用传统策略（不含模型）
python3 scripts/paper_trader.py --no-model
```

### Q10: 如何备份交易数据?

```bash
# 备份所有数据
tar -czf backup_$(date +%Y%m%d).tar.gz data/ logs/ models/

# 备份到远程服务器
scp backup_*.tar.gz user@remote:/path/to/backup/
```

---

## 监控与维护

### 每日检查清单

- [ ] 检查定时任务是否执行：`crontab -l`
- [ ] 查看交易日志：`tail -f logs/cron_trades.log`
- [ ] 检查持仓情况：`python3 scripts/manage_positions.py`
- [ ] 监控内存使用：`free -h`
- [ ] 检查磁盘空间：`df -h`

### 性能优化

```bash
# 1. 减少日志大小
find logs/ -name "*.log" -mtime +7 -delete

# 2. 清理旧数据
find data/paper_trading/ -name "*.csv" -mtime +30 -delete

# 3. 优化swap使用
sudo sysctl vm.swappiness=10
```

---

## 风险提示

⚠️ **重要提示**:

1. **资金风险**: 加密货币交易有极高风险，可能损失全部本金
2. **技术风险**: 策略基于历史数据，不保证未来收益
3. **系统风险**: 网络中断、API故障可能导致交易失败
4. **止损设置**: -200%止损意味着在极端行情下会有巨额亏损

**建议**:
- 从小资金开始测试
- 使用测试网验证策略
- 设置合理的止损比例
- 定期检查持仓和资金

---

## 技术支持

如有问题，请提交Issue或联系开发者。

---

**祝交易顺利！🚀**

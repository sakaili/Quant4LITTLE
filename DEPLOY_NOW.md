# 🚀 立即部署到ECS

> **状态**: ✅ 所有文件已就绪，可以直接部署！

---

## 📦 当前配置

### ✅ 已就绪的文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `models/rank_model.pt` | 479KB | PyTorch模型 |
| `models/rank_model_meta.json` | 1.7KB | 模型元数据 |
| `deploy/setup.sh` | 13KB | 一键部署脚本 |
| `deploy/cron_*.sh` | 3个文件 | 定时任务脚本 |
| `.env.example` | - | API配置模板 |
| `requirements_onnx.txt` | - | Python依赖（PyTorch版） |

### 📊 部署参数

- **模型**: PyTorch CPU模式 (~200-300MB内存)
- **依赖大小**: ~400MB
- **ECS要求**: 2CPU + 1GB RAM + 2GB Swap
- **执行时间**:
  - 09:00 北京时间 - 更新数据
  - 10:00 北京时间 - 生成信号
  - 10:30 北京时间 - 执行交易

---

## 🎯 三步部署

### 步骤1: 本地提交到Git

```bash
# 在本地Windows环境 (当前目录: f:/2025/Quant4Little)
git add .
git commit -m "Ready for ECS deployment - PyTorch model"
git push origin main
```

### 步骤2: ECS上克隆代码

```bash
# SSH登录到ECS服务器
ssh root@your-ecs-ip

# 克隆仓库（替换your-username）
git clone https://github.com/your-username/Quant4Little.git
cd Quant4Little
```

### 步骤3: 配置并部署

```bash
# 1. 配置API密钥
cp .env.example .env
nano .env
# 填入:
#   BINANCE_API_KEY=你的API密钥
#   BINANCE_API_SECRET=你的API密钥
#   USE_TESTNET=True  # 建议先测试

# 2. 一键部署
bash deploy/setup.sh

# 完成！🎉
```

---

## 📝 部署后验证

```bash
# 1. 检查模型加载
python3 -c "from scripts.lightweight_ranker import LightweightRanker; r = LightweightRanker(); print('✓ 模型加载成功')"

# 2. 生成测试信号
python3 scripts/paper_trader.py --max-positions 5

# 3. 查看信号文件
ls -lh data/paper_trading/signals_*.csv

# 4. 检查定时任务
crontab -l

# 5. 查看内存使用
free -h
```

---

## ⚙️ 当前策略配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **筛选条件** | EMA10<EMA20<EMA30, KDJ_J>50, ATR>2% | 底部形态+超买 |
| **资金管理** | 2% × 2x杠杆 | 每笔约4%资金 |
| **最大持仓** | 10个 | 分散风险 |
| **订单类型** | Maker限价单 | 0.1%偏移 |
| **止盈** | +30% | 自动委托单 |
| **止损** | -200% | ⚠️ 极端行情风险高 |

---

## ⏰ 定时任务计划

| 时间(UTC) | 时间(北京) | 任务 | 脚本 |
|----------|----------|------|------|
| 01:00 | 09:00 | 更新数据 | `cron_update_data.sh` |
| 02:00 | 10:00 | 生成信号 | `cron_generate_signals.sh` |
| 02:30 | 10:30 | 执行交易 | `cron_execute_trades.sh` |

**注意**: 止盈止损是自动委托单，无需人工监控！

---

## 📚 完整文档

- **快速指南**: [QUICK_DEPLOY.md](QUICK_DEPLOY.md)
- **完整文档**: [README_DEPLOYMENT.md](README_DEPLOYMENT.md)
- **部署清单**: [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)

---

## ⚠️ 重要提示

1. **测试先行**: 建议先用 `USE_TESTNET=True` 测试
2. **小额开始**: 建议100-500 USDT起步
3. **止损风险**: -200%止损意味着极端行情会有巨额亏损
4. **定期检查**: 每天查看日志和持仓情况

---

## 🔧 常用命令

```bash
# 查看交易日志
tail -f logs/cron_trades.log

# 查看当前持仓
python3 scripts/manage_positions.py

# 手动执行交易（测试用）
python3 scripts/live_maker_trader.py

# 更新代码
git pull origin main
bash deploy/setup.sh
```

---

## 📞 遇到问题？

1. 查看 [README_DEPLOYMENT.md](README_DEPLOYMENT.md) 的"常见问题"章节
2. 检查日志文件: `logs/cron_*.log`
3. 验证环境配置: `cat .env`
4. 检查swap内存: `free -h`

---

**现在就可以部署了！** 🚀

按照上面的三个步骤操作，5分钟内即可完成部署。

祝交易顺利！💰

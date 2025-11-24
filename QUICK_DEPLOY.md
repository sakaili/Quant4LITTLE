# ECS部署快速指南

## 🚀 一键部署（3个命令）

### 步骤1: 上传到GitHub

```bash
# 在本地Windows上
cd f:/2025/Quant4Little

git add .
git commit -m "Ready for ECS deployment"
git push origin main
```

### 步骤2: 在ECS上部署

```bash
# SSH登录ECS
ssh root@your-ecs-ip

# 克隆代码
git clone https://github.com/your-username/Quant4Little.git
cd Quant4Little

# 配置API密钥
cp .env.example .env
nano .env
# 填入: BINANCE_API_KEY, BINANCE_API_SECRET, USE_TESTNET=False

# 一键部署
bash deploy/setup.sh
```

完成！🎉

---

## ⏰ 自动执行时间

| 时间(北京) | 任务 | 说明 |
|-----------|------|------|
| 09:00 | 更新数据 | 下载最新K线数据 |
| 10:00 | 生成信号 | 策略筛选+AI打分 |
| 10:30 | 执行交易 | 下单+自动止盈止损 |

---

## 📊 交易策略参数

- **筛选条件**: EMA10 < EMA20 < EMA30, KDJ_J > 50, ATR > 2%
- **资金管理**: 2% × 2倍杠杆 = 每笔4%
- **最大持仓**: 10个
- **止盈止损**: +30% / -200% (自动委托单)
- **订单类型**: Maker限价单 (0.1%偏移)

---

## 🔍 日常监控

```bash
# 查看交易日志
tail -f logs/cron_trades.log

# 查看当前持仓
python3 scripts/manage_positions.py

# 查看定时任务
crontab -l
```

---

## ⚠️ 重要提示

1. **初次部署**: 建议先用 `USE_TESTNET=True` 测试
2. **小额测试**: 建议100-500 USDT开始
3. **定期检查**: 每天查看日志和持仓
4. **止损风险**: -200%止损意味着极端行情会有巨额亏损

---

## 📝 文件说明

- `deploy/setup.sh` - 一键部署脚本
- `deploy/cron_*.sh` - 定时任务脚本
- `.env.example` - 配置模板
- `README_DEPLOYMENT.md` - 完整文档

---

## 🎯 内存使用

- **PyTorch模型**: ~200-300MB
- **ECS配置**: 1GB RAM + 2GB Swap = 3GB总内存
- **足够使用**: ✅

---

**祝交易顺利！** 🚀

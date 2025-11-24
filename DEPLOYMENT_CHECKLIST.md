# Quant4Little - 部署完成清单

## ✅ 已创建的文件

### 1. 模型优化
- ✅ [scripts/convert_to_onnx.py](scripts/convert_to_onnx.py) - PyTorch → ONNX转换脚本
- ✅ [requirements_onnx.txt](requirements_onnx.txt) - 轻量级依赖（400MB vs 1.2GB）
- ✅ [scripts/lightweight_ranker.py](scripts/lightweight_ranker.py) - 已修复ONNX输入名称

### 2. 部署自动化
- ✅ [deploy/setup.sh](deploy/setup.sh) - 一键部署脚本
- ✅ [deploy/cron_update_data.sh](deploy/cron_update_data.sh) - 每日数据更新
- ✅ [deploy/cron_generate_signals.sh](deploy/cron_generate_signals.sh) - 每日生成信号
- ✅ [deploy/cron_execute_trades.sh](deploy/cron_execute_trades.sh) - 每日执行交易

### 3. 配置文件
- ✅ [.env.example](.env.example) - API密钥模板
- ✅ [.gitignore](.gitignore) - Git忽略规则（已更新）

### 4. 文档
- ✅ [README_DEPLOYMENT.md](README_DEPLOYMENT.md) - 完整部署指南

---

## 🚀 快速部署流程

### 在ECS服务器上执行:

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/Quant4Little.git
cd Quant4Little

# 2. 配置API密钥
cp .env.example .env
nano .env  # 填入 BINANCE_API_KEY 和 BINANCE_API_SECRET

# 3. 运行一键部署
bash deploy/setup.sh
```

就这么简单！🎉

---

## 📝 部署前准备（本地）

在上传到Git之前，需要先转换模型：

```bash
# 转换PyTorch模型为ONNX
python scripts/convert_to_onnx.py

# 验证文件生成
ls -lh models/
# 应该看到:
# - rank_model.pt (PyTorch原始模型)
# - rank_model.onnx (ONNX优化模型)
# - rank_model_meta.json (元数据)
```

---

## 🔧 关键参数确认

### 当前策略配置:

| 参数 | 值 | 文件位置 |
|------|-----|----------|
| KDJ阈值 | 50 | [paper_trader.py:139](scripts/paper_trader.py#L139) |
| 资金管理 | 2% × 2x杠杆 | [live_maker_trader.py:383](scripts/live_maker_trader.py#L383) |
| 最大持仓 | 10个 | [live_maker_trader.py:386](scripts/live_maker_trader.py#L386) |
| Maker偏移 | 0.10% | [live_maker_trader.py:384](scripts/live_maker_trader.py#L384) |
| 止盈 | +30% | [live_maker_trader.py:532](scripts/live_maker_trader.py#L532) |
| 止损 | -200% | [live_maker_trader.py:533](scripts/live_maker_trader.py#L533) |

### 定时任务时间:

| 任务 | 时间 (UTC) | 时间 (北京) | 脚本 |
|------|-----------|------------|------|
| 数据更新 | 01:00 | 09:00 | cron_update_data.sh |
| 生成信号 | 02:00 | 10:00 | cron_generate_signals.sh |
| 执行交易 | 02:30 | 10:30 | cron_execute_trades.sh |

---

## 🎯 ECS服务器要求

- **CPU**: 2核心 ✅
- **内存**: 1GB + 2GB Swap ✅
- **Python**: 3.8+ ✅
- **依赖**: requirements_onnx.txt ✅

---

## ⚠️ 重要提示

1. **转换模型**: 在本地运行 `python scripts/convert_to_onnx.py`
2. **上传模型**: 确保 `models/rank_model.onnx` 和 `models/rank_model_meta.json` 在Git中
3. **配置API**: ECS上运行前必须配置 `.env`
4. **创建Swap**: ECS服务器需要手动创建2GB swap（见README_DEPLOYMENT.md）
5. **测试先行**: 建议先用 `USE_TESTNET=True` 测试

---

## 📊 部署验证清单

### 部署完成后检查:

- [ ] 模型文件存在: `ls models/*.onnx`
- [ ] 环境变量配置: `cat .env`
- [ ] 依赖安装成功: `pip list | grep onnx`
- [ ] Swap内存生效: `free -h`
- [ ] 定时任务已设置: `crontab -l`
- [ ] 模型加载测试: `python3 -c "from scripts.lightweight_ranker import LightweightRanker; LightweightRanker()"`
- [ ] 信号生成测试: `python3 scripts/paper_trader.py --max-positions 5`

---

## 🆘 故障排查

### 模型加载失败?
```bash
# 检查文件是否存在
ls -lh models/rank_model.onnx
ls -lh models/rank_model_meta.json

# 检查onnxruntime是否安装
pip list | grep onnxruntime
```

### 内存不足?
```bash
# 检查swap
free -h
swapon --show

# 创建swap（见README_DEPLOYMENT.md步骤1）
```

### API连接失败?
```bash
# 检查代理配置
echo $HTTPS_PROXY

# 测试API连接
python3 -c "import ccxt; binance = ccxt.binance(); print(binance.fetch_ticker('BTC/USDT'))"
```

---

## 📚 更多文档

- 完整部署指南: [README_DEPLOYMENT.md](README_DEPLOYMENT.md)
- 策略说明: 见README_DEPLOYMENT.md "策略说明"部分
- 常见问题: 见README_DEPLOYMENT.md "常见问题"部分

---

**祝部署顺利！** 🎊

如有问题，请查看 [README_DEPLOYMENT.md](README_DEPLOYMENT.md) 或提交Issue。

# 修复 .env 文件加载问题

## 问题
脚本显示 "请设置环境变量" 错误，即使 `.env` 文件已配置。

## 原因
脚本缺少 `python-dotenv` 库来加载 `.env` 文件。

## 解决方案

在ECS服务器上执行以下命令：

```bash
# 1. 拉取最新代码
cd ~/Quant4LITTLE
git pull origin main

# 2. 安装python-dotenv
pip3 install python-dotenv

# 3. 测试修复是否生效
python3 scripts/manage_positions.py
```

如果看到持仓信息（或"当前无持仓"），说明修复成功！

## 已修复的文件

- ✅ [scripts/manage_positions.py](scripts/manage_positions.py) - 持仓管理
- ✅ [scripts/live_maker_trader.py](scripts/live_maker_trader.py) - 实盘交易
- ✅ [scripts/data_fetcher.py](scripts/data_fetcher.py) - 数据获取
- ✅ [requirements_onnx.txt](requirements_onnx.txt) - 添加python-dotenv依赖

## 验证

```bash
# 检查.env文件配置
cat .env

# 应该看到:
# BINANCE_API_KEY=你的密钥
# BINANCE_API_SECRET=你的密钥
# USE_TESTNET=False

# 测试API连接
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key:', os.getenv('BINANCE_API_KEY')[:10] + '...')"
```

## 继续部署

修复后，可以继续运行部署脚本：

```bash
# 运行修复脚本（之前失败了）
bash fix_setup_pytorch.sh
```

或者手动运行测试：

```bash
# 测试信号生成
python3 scripts/paper_trader.py --max-positions 5

# 查看生成的信号
ls -lh data/paper_trading/signals_*.csv
```

---

**修复完成！现在所有脚本都可以正确读取 .env 文件了。** ✅

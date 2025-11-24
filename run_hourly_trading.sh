#!/bin/bash
# Linux 定时任务脚本 - 每小时运行一次Paper Trading（增强版）
# 添加到 crontab: 0 * * * * /path/to/Quant4Little/run_hourly_trading.sh

cd "$(dirname "$0")"

echo "========================================"
echo "每小时自动交易（增强版 - 含模型自动更新）"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 激活虚拟环境（如果有）
# source venv/bin/activate

# 运行Python脚本（增强版）
python3 scripts/hourly_trading_enhanced.py

echo ""
echo "运行完成"
echo "========================================"

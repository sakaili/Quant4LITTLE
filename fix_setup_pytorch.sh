#!/bin/bash
################################################################################
# 快速修复脚本 - 跳过ONNX转换，直接使用PyTorch模型
################################################################################

echo "════════════════════════════════════════════════════════════════════════════"
echo "  🔧 修复部署 - 使用PyTorch模型"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# 检查PyTorch模型是否存在
if [ ! -f models/rank_model.pt ]; then
    echo "❌ 错误: 未找到 models/rank_model.pt"
    exit 1
fi

if [ ! -f models/rank_model_meta.json ]; then
    echo "❌ 错误: 未找到 models/rank_model_meta.json"
    exit 1
fi

echo "✓ 发现PyTorch模型文件"
echo ""

# 测试模型加载
echo "────────────────────────────────────────────────────────────────────────────"
echo "  测试模型加载..."
echo "────────────────────────────────────────────────────────────────────────────"

python3 -c "
from scripts.lightweight_ranker import LightweightRanker
import sys

try:
    ranker = LightweightRanker()
    print('  ✓ 模型加载成功！')
    print(f'  使用模型: PyTorch (CPU模式)')
    sys.exit(0)
except Exception as e:
    print(f'  ❌ 模型加载失败: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 模型加载测试失败"
    exit 1
fi

echo ""

# 创建必要的目录
echo "────────────────────────────────────────────────────────────────────────────"
echo "  创建必要的目录..."
echo "────────────────────────────────────────────────────────────────────────────"

mkdir -p data/daily_klines
mkdir -p data/hourly_klines
mkdir -p data/paper_trading
mkdir -p logs

echo "  ✓ 目录创建完成"
echo ""

# 设置定时任务
echo "────────────────────────────────────────────────────────────────────────────"
echo "  设置定时任务 (crontab)"
echo "────────────────────────────────────────────────────────────────────────────"

CRON_FILE="/tmp/quant4little_crontab.txt"
PROJECT_DIR=$(pwd)

cat > "$CRON_FILE" <<EOF
# Quant4Little 自动交易定时任务
# 每日UTC 01:00 (北京09:00) - 更新数据
0 1 * * * cd $PROJECT_DIR && bash deploy/cron_update_data.sh >> logs/cron_update.log 2>&1

# 每日UTC 02:00 (北京10:00) - 生成信号
0 2 * * * cd $PROJECT_DIR && bash deploy/cron_generate_signals.sh >> logs/cron_signals.log 2>&1

# 每日UTC 02:30 (北京10:30) - 执行交易
30 2 * * * cd $PROJECT_DIR && bash deploy/cron_execute_trades.sh >> logs/cron_trades.log 2>&1
EOF

crontab "$CRON_FILE"
rm "$CRON_FILE"

echo "  ✓ Crontab已配置"
echo ""

# 显示当前定时任务
echo "  当前定时任务:"
crontab -l | grep -v "^#" | grep -v "^$"
echo ""

# 运行测试
echo "────────────────────────────────────────────────────────────────────────────"
echo "  运行测试..."
echo "────────────────────────────────────────────────────────────────────────────"

echo "  测试信号生成 (生成5个信号)..."
python3 scripts/paper_trader.py --max-positions 5

if [ $? -eq 0 ]; then
    echo ""
    echo "  ✓ 信号生成测试成功！"
    echo ""
    echo "  查看生成的信号文件:"
    ls -lh data/paper_trading/signals_*.csv 2>/dev/null | tail -1
else
    echo ""
    echo "  ⚠️  信号生成测试失败（可能需要先下载数据）"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  ✅ 部署完成！"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  📊 策略配置:"
echo "    - 筛选: EMA10<EMA20<EMA30, KDJ>50, ATR>2%"
echo "    - 资金: 2% × 2x杠杆 = 每笔4%"
echo "    - 持仓: 最多10个"
echo "    - 止盈: +30% (自动)"
echo "    - 止损: -200% (自动) ⚠️"
echo ""
echo "  ⏰ 执行时间 (北京时间):"
echo "    - 09:00 - 更新数据"
echo "    - 10:00 - 生成信号"
echo "    - 10:30 - 执行交易"
echo ""
echo "  📝 常用命令:"
echo "    - 查看日志: tail -f logs/cron_trades.log"
echo "    - 查看持仓: python3 scripts/manage_positions.py"
echo "    - 手动交易: python3 scripts/live_maker_trader.py"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"

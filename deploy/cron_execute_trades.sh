#!/bin/bash
################################################################################
# 定时任务 - 执行交易
# 每日UTC 02:30执行 (北京时间10:30)
################################################################################

set -e

echo "════════════════════════════════════════════════════════════════════════════"
echo "  💰 执行交易任务 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════════════"

# 加载环境变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

cd "$PROJECT_ROOT"

# 检查今日信号文件是否存在
TODAY=$(date '+%Y%m%d')
SIGNAL_FILE="data/paper_trading/signals_${TODAY}.csv"

if [ ! -f "$SIGNAL_FILE" ]; then
    echo "  ❌ 未找到今日信号文件: $SIGNAL_FILE"
    echo "  请先运行: python3 scripts/paper_trader.py"
    exit 1
fi

echo "  ✓ 找到今日信号文件: $SIGNAL_FILE"
echo "  信号数量: $(wc -l < "$SIGNAL_FILE" | xargs)"
echo ""

# 执行交易（自动确认模式）
echo "  执行实盘交易..."
echo ""

python3 scripts/live_maker_trader.py --auto-confirm

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  ✅ 交易执行完成 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════════════"

# 显示当前持仓
echo ""
echo "  查看当前持仓..."
python3 scripts/manage_positions.py 2>&1 | head -50

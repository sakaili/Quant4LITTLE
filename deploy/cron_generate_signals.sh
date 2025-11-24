#!/bin/bash
################################################################################
# 定时任务 - 生成交易信号
# 每日UTC 02:00执行 (北京时间10:00)
################################################################################

set -e

echo "════════════════════════════════════════════════════════════════════════════"
echo "  📊 生成交易信号 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════════════"

# 加载环境变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

cd "$PROJECT_ROOT"

# 生成交易信号
echo "  使用Paper Trading系统生成信号..."
echo ""

python3 scripts/paper_trader.py --max-positions 20

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  ✅ 信号生成完成 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════════════"

# 显示生成的信号文件
TODAY=$(date '+%Y%m%d')
SIGNAL_FILE="data/paper_trading/signals_${TODAY}.csv"

if [ -f "$SIGNAL_FILE" ]; then
    echo ""
    echo "  信号文件: $SIGNAL_FILE"
    echo "  信号数量: $(wc -l < "$SIGNAL_FILE" | xargs)"
    echo ""
    echo "  前5个信号:"
    head -6 "$SIGNAL_FILE" | column -t -s,
fi

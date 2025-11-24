#!/bin/bash
################################################################################
# 安装小时级定时任务
# 每天筛选候选池 + 每小时检测信号并交易
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================================================"
echo "  安装 Quant4Little 小时级定时任务"
echo "========================================================================"
echo ""
echo "  项目目录: $PROJECT_ROOT"
echo ""

# 生成临时cron文件
CRON_FILE="/tmp/quant4little_cron_hourly_$$.txt"

cat > "$CRON_FILE" << CRONEOF
# Quant4Little 小时级交易系统
# 每天09:00 (UTC 01:00) - 筛选候选币池
0 1 * * * cd $PROJECT_ROOT && python3 scripts/daily_candidate_scan.py --bottom-n 300 >> logs/cron_candidates.log 2>&1

# 每小时整点 - 检测信号并执行交易
0 * * * * cd $PROJECT_ROOT && bash deploy/cron_hourly_detect.sh >> logs/cron_hourly.log 2>&1
CRONEOF

echo "  将要安装的定时任务:"
echo "  ────────────────────────────────────────────────────────────────"
cat "$CRON_FILE"
echo "  ────────────────────────────────────────────────────────────────"
echo ""

read -p "  确认安装? (y/n) " -n 1 -r
echo
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 备份当前crontab
    BACKUP_FILE="/tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
    crontab -l > "$BACKUP_FILE" 2>/dev/null || true
    echo "  ✓ 备份当前crontab: $BACKUP_FILE"
    
    # 删除旧的Quant4Little任务
    crontab -l 2>/dev/null | grep -v -i "quant4little\|Quant4LITTLE" | crontab - || true
    echo "  ✓ 清理旧任务"
    
    # 添加新任务
    (crontab -l 2>/dev/null; cat "$CRON_FILE") | crontab -
    echo "  ✓ 安装新任务"
    echo ""
    
    # 创建日志目录
    mkdir -p "$PROJECT_ROOT/logs"
    
    echo "  当前的crontab任务:"
    echo "  ────────────────────────────────────────────────────────────────"
    crontab -l
    echo "  ────────────────────────────────────────────────────────────────"
    echo ""
    echo "  ✅ 定时任务安装成功"
    echo ""
    echo "  📊 执行时间 (北京时间):"
    echo "    - 09:00  筛选候选池"
    echo "    - 每小时  检测信号并交易"
    echo ""
    echo "  📝 监控命令:"
    echo "    tail -f logs/cron_hourly.log"
    echo "    tail -f logs/cron_candidates.log"
else
    echo "  ❌ 取消安装"
fi

rm -f "$CRON_FILE"

echo ""
echo "========================================================================"

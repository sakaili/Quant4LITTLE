#!/bin/bash
################################################################################
# 清理 Quant4Little 定时任务
# 删除所有相关的crontab条目
################################################################################

echo "========================================================================"
echo "  清理 Quant4Little 定时任务"
echo "========================================================================"
echo ""

# 备份当前crontab
BACKUP_FILE="/tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
crontab -l > "$BACKUP_FILE" 2>/dev/null || true

if [ -s "$BACKUP_FILE" ]; then
    echo "  备份当前crontab到: $BACKUP_FILE"
    echo ""
    
    # 显示当前的Quant4Little任务
    echo "  当前的Quant4Little任务:"
    grep -i "quant4little\|Quant4LITTLE" "$BACKUP_FILE" || echo "  (无)"
    echo ""
    
    # 删除所有Quant4Little相关任务
    crontab -l 2>/dev/null | grep -v -i "quant4little\|Quant4LITTLE" | crontab -
    
    echo "  ✓ 已删除所有Quant4Little定时任务"
else
    echo "  当前没有crontab任务"
fi

echo ""
echo "  当前剩余的crontab任务:"
crontab -l 2>/dev/null || echo "  (无)"
echo ""
echo "========================================================================"
echo "  清理完成"
echo "========================================================================"

#!/bin/bash
################################################################################
# ECS服务器一键部署脚本
# 用途: 在2CPU/1GB RAM的ECS服务器上自动部署交易机器人
# 用法: bash deploy/setup.sh
################################################################################

set -e  # 遇到错误立即退出

echo "════════════════════════════════════════════════════════════════════════════"
echo "  🚀 Quant4Little 交易机器人 - ECS自动部署"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "  项目目录: $PROJECT_ROOT"
echo ""

################################################################################
# 1. 检查系统资源
################################################################################
echo "────────────────────────────────────────────────────────────────────────────"
echo "  [1/7] 检查系统资源"
echo "────────────────────────────────────────────────────────────────────────────"

# 检查CPU
CPU_CORES=$(nproc)
echo "  ✓ CPU核心数: $CPU_CORES"

# 检查内存
TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
AVAILABLE_MEM=$(free -m | awk 'NR==2{print $7}')
echo "  ✓ 总内存: ${TOTAL_MEM}MB"
echo "  ✓ 可用内存: ${AVAILABLE_MEM}MB"

# 检查是否需要创建swap
SWAP_SIZE=$(free -m | awk 'NR==3{print $2}')
if [ "$SWAP_SIZE" -lt 2048 ]; then
    echo ""
    echo "  ⚠️  Swap内存不足 (${SWAP_SIZE}MB < 2048MB)"
    echo "  正在创建2GB swap文件..."

    # 创建swap文件（需要root权限）
    if [ -f /swapfile ]; then
        echo "  ℹ️  /swapfile 已存在，跳过创建"
    else
        echo "  注意: 创建swap需要root权限，请手动执行以下命令："
        echo ""
        echo "    sudo fallocate -l 2G /swapfile"
        echo "    sudo chmod 600 /swapfile"
        echo "    sudo mkswap /swapfile"
        echo "    sudo swapon /swapfile"
        echo "    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab"
        echo ""
        read -p "  是否已创建swap? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "  ❌ 请先创建swap后再运行此脚本"
            exit 1
        fi
    fi
else
    echo "  ✓ Swap内存: ${SWAP_SIZE}MB (充足)"
fi

echo ""

################################################################################
# 2. 安装Python依赖
################################################################################
echo "────────────────────────────────────────────────────────────────────────────"
echo "  [2/7] 安装Python依赖"
echo "────────────────────────────────────────────────────────────────────────────"

# 检查Python版本
if ! command -v python3 &> /dev/null; then
    echo "  ❌ Python3未安装，请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "  ✓ Python版本: $PYTHON_VERSION"

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "  ❌ pip3未安装，请先安装pip"
    exit 1
fi

# 安装轻量级依赖（使用ONNX，不含PyTorch）
echo "  正在安装依赖 (requirements_onnx.txt)..."
pip3 install -r requirements_onnx.txt --quiet

echo "  ✓ 依赖安装完成"
echo ""

################################################################################
# 3. 检查环境变量
################################################################################
echo "────────────────────────────────────────────────────────────────────────────"
echo "  [3/7] 检查环境变量"
echo "────────────────────────────────────────────────────────────────────────────"

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "  ⚠️  .env 文件不存在，从 .env.example 复制..."
        cp .env.example .env
        echo ""
        echo "  ⚠️  请编辑 .env 文件，填入您的API密钥:"
        echo ""
        echo "    nano .env"
        echo ""
        echo "  必填项:"
        echo "    - BINANCE_API_KEY"
        echo "    - BINANCE_API_SECRET"
        echo "    - HTTPS_PROXY (如果需要)"
        echo ""
        read -p "  是否已配置.env? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "  ❌ 请先配置.env后再运行此脚本"
            exit 1
        fi
    else
        echo "  ❌ 未找到.env或.env.example文件"
        exit 1
    fi
fi

# 加载环境变量
export $(grep -v '^#' .env | xargs)

# 检查必需的环境变量
if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
    echo "  ❌ BINANCE_API_KEY 或 BINANCE_API_SECRET 未设置"
    exit 1
fi

echo "  ✓ API Key: ${BINANCE_API_KEY:0:10}..."
echo "  ✓ API Secret: ${BINANCE_API_SECRET:0:10}..."
echo "  ✓ 测试模式: ${USE_TESTNET:-False}"

echo ""

################################################################################
# 4. 检查模型文件
################################################################################
echo "────────────────────────────────────────────────────────────────────────────"
echo "  [4/7] 检查模型文件"
echo "────────────────────────────────────────────────────────────────────────────"

if [ ! -f models/rank_model.pt ]; then
    echo "  ❌ PyTorch模型不存在: models/rank_model.pt"
    echo "  请先训练模型或从其他地方复制模型文件到 models/ 目录"
    exit 1
else
    echo "  ✓ PyTorch模型存在: models/rank_model.pt"
fi

if [ ! -f models/rank_model_meta.json ]; then
    echo "  ❌ 模型元数据不存在: models/rank_model_meta.json"
    exit 1
fi

echo "  ✓ 模型元数据存在: models/rank_model_meta.json"
echo ""

################################################################################
# 5. 创建必要的目录
################################################################################
echo "────────────────────────────────────────────────────────────────────────────"
echo "  [5/7] 创建必要的目录"
echo "────────────────────────────────────────────────────────────────────────────"

mkdir -p data/daily_klines
mkdir -p data/hourly_klines
mkdir -p data/paper_trading
mkdir -p logs

echo "  ✓ 目录创建完成"
echo ""

################################################################################
# 6. 设置定时任务
################################################################################
echo "────────────────────────────────────────────────────────────────────────────"
echo "  [6/7] 设置定时任务 (crontab)"
echo "────────────────────────────────────────────────────────────────────────────"

# 生成crontab条目
CRON_FILE="/tmp/quant4little_cron_$$.txt"

cat > "$CRON_FILE" << EOF
# Quant4Little 交易机器人 - 自动执行任务
# 每日UTC 01:00 (北京时间09:00) 更新数据
0 1 * * * cd $PROJECT_ROOT && bash deploy/cron_update_data.sh >> logs/cron_update.log 2>&1

# 每日UTC 02:00 (北京时间10:00) 生成交易信号
0 2 * * * cd $PROJECT_ROOT && bash deploy/cron_generate_signals.sh >> logs/cron_signals.log 2>&1

# 每日UTC 02:30 (北京时间10:30) 执行交易
30 2 * * * cd $PROJECT_ROOT && bash deploy/cron_execute_trades.sh >> logs/cron_trades.log 2>&1
EOF

echo "  生成的定时任务:"
echo ""
cat "$CRON_FILE"
echo ""

read -p "  是否安装定时任务到crontab? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 备份现有crontab
    crontab -l > /tmp/crontab_backup_$$.txt 2>/dev/null || true

    # 删除旧的Quant4Little任务（如果存在）
    (crontab -l 2>/dev/null | grep -v "Quant4Little" || true) | crontab -

    # 添加新任务
    (crontab -l 2>/dev/null; cat "$CRON_FILE") | crontab -

    echo "  ✓ 定时任务已安装"
    echo ""
    echo "  可以使用以下命令查看:"
    echo "    crontab -l"
else
    echo "  ℹ️  跳过定时任务安装"
    echo "  您可以手动执行交易脚本:"
    echo "    bash deploy/cron_execute_trades.sh"
fi

rm -f "$CRON_FILE"

echo ""

################################################################################
# 7. 测试运行
################################################################################
echo "────────────────────────────────────────────────────────────────────────────"
echo "  [7/7] 测试运行"
echo "────────────────────────────────────────────────────────────────────────────"

echo "  正在测试模型加载..."
python3 -c "from scripts.lightweight_ranker import LightweightRanker; r = LightweightRanker(); print('✓ 模型加载成功')"

echo ""
read -p "  是否立即执行一次完整流程测试? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "  [测试1/3] 生成交易信号..."
    python3 scripts/paper_trader.py --max-positions 5

    echo ""
    echo "  [测试2/3] 查看生成的信号..."
    ls -lh data/paper_trading/signals_*.csv | tail -1

    echo ""
    read -p "  是否执行实盘下单测试? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "  ⚠️  注意: 这将执行真实交易!"
        echo "  请确保:"
        echo "    1. USE_TESTNET=True (如果只是测试)"
        echo "    2. 账户有足够的USDT余额"
        echo "    3. 已开启币安合约账户"
        echo ""
        read -p "  确认继续? (输入 yes 确认): " confirm
        if [ "$confirm" = "yes" ]; then
            python3 scripts/live_maker_trader.py
        else
            echo "  ℹ️  已取消实盘测试"
        fi
    fi
else
    echo "  ℹ️  跳过测试"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  ✅ 部署完成!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  📋 使用说明:"
echo ""
echo "  1. 查看日志:"
echo "     tail -f logs/cron_trades.log"
echo ""
echo "  2. 手动执行交易:"
echo "     python3 scripts/live_maker_trader.py --auto-confirm"
echo ""
echo "  3. 查看持仓:"
echo "     python3 scripts/manage_positions.py"
echo ""
echo "  4. 监控定时任务:"
echo "     crontab -l"
echo ""
echo "  5. 查看系统资源:"
echo "     free -h && df -h"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"

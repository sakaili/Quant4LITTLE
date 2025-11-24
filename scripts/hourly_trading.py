#!/usr/bin/env python3
"""
每小时自动运行的交易脚本
1. 更新最新数据
2. 运行Paper Trading生成信号
3. 记录日志
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import logging

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 配置日志
log_dir = ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"hourly_trading_{datetime.now().strftime('%Y%m')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def update_coin_pool():
    """更新币池数据"""
    from scripts.update_latest_data import update_all_klines

    logger.info("=" * 60)
    logger.info("开始更新币池数据（后200名低流动性币种）")
    logger.info("=" * 60)

    try:
        daily_dir = ROOT / "data" / "daily_klines"
        hourly_dir = ROOT / "data" / "hourly_klines"

        update_all_klines(daily_dir, hourly_dir)
        logger.info("✓ 币池数据更新完成")
        return True

    except Exception as e:
        logger.error(f"✗ 币池数据更新失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_paper_trading():
    """运行Paper Trading"""
    from scripts.paper_trader import PaperTrader

    logger.info("=" * 60)
    logger.info("开始运行Paper Trading")
    logger.info("=" * 60)

    try:
        trader = PaperTrader(
            daily_dir=ROOT / "data" / "daily_klines",
            hourly_dir=ROOT / "data" / "hourly_klines",
            model_dir=ROOT / "models",
            output_dir=ROOT / "data" / "paper_trading"
        )

        # 使用当前日期
        current_date = datetime.now().strftime("%Y-%m-%d")
        trader.run(date=current_date, max_positions=20)

        logger.info("✓ Paper Trading 完成")
        return True

    except Exception as e:
        logger.error(f"✗ Paper Trading 失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """主流程"""
    start_time = datetime.now()
    logger.info("\n" + "=" * 80)
    logger.info(f"每小时自动交易 - 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # 步骤1: 更新币池数据
    logger.info("\n[步骤1/2] 更新币池数据")
    data_success = update_coin_pool()

    if not data_success:
        logger.warning("数据更新失败，跳过Paper Trading")
        return

    # 步骤2: 运行Paper Trading
    logger.info("\n[步骤2/2] 运行Paper Trading")
    trading_success = run_paper_trading()

    # 汇总
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 80)
    logger.info("运行完成")
    logger.info("=" * 80)
    logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总耗时: {duration:.1f} 秒")
    logger.info(f"数据更新: {'✓ 成功' if data_success else '✗ 失败'}")
    logger.info(f"Paper Trading: {'✓ 成功' if trading_success else '✗ 失败'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

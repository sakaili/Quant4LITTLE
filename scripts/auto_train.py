#!/usr/bin/env python3
"""
自动化训练脚本：监控数据收集，完成后启动训练并处理错误。
"""
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# 确保项目根目录在路径中
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def monitor_data_collection(target_count: int = 282, check_interval: int = 300):
    """监控数据收集进度"""
    logger.info(f"开始监控数据收集进度，目标: {target_count} 个文件")
    logger.info(f"检查间隔: {check_interval}秒 ({check_interval//60}分钟)")
    logger.info("-" * 60)

    scan_dir = ROOT / "data" / "daily_scans"

    while True:
        scan_files = list(scan_dir.glob("candidates_*.csv"))
        current_count = len(scan_files)
        progress = current_count / target_count * 100

        logger.info(f"进度: {current_count}/{target_count} ({progress:.1f}%)")

        if current_count >= target_count:
            logger.info("✓ 数据收集完成！")
            return True

        time.sleep(check_interval)


def verify_data():
    """验证数据完整性"""
    logger.info("验证数据完整性...")

    scan_dir = ROOT / "data" / "daily_scans"
    backtest_csv = ROOT / "data" / "backtest_trades.csv"
    daily_dir = ROOT / "data" / "daily_klines"
    hourly_dir = ROOT / "data" / "hourly_klines"

    checks = {
        "扫描文件": scan_dir.exists() and len(list(scan_dir.glob("*.csv"))) > 0,
        "回测交易": backtest_csv.exists(),
        "日线数据": daily_dir.exists() and len(list(daily_dir.glob("*.csv"))) > 0,
        "小时线数据": hourly_dir.exists() and len(list(hourly_dir.glob("*.csv"))) > 0,
    }

    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {name}")

    if not all(checks.values()):
        logger.error("数据验证失败！")
        return False

    logger.info("✓ 数据验证通过")
    return True


def train_model():
    """启动模型训练"""
    logger.info("=" * 60)
    logger.info("开始训练模型...")
    logger.info("=" * 60)

    try:
        from scripts.modeling.train_ranker import main as train_main

        # 修改 sys.argv 来传递参数
        original_argv = sys.argv.copy()
        sys.argv = [
            "train_ranker.py",
            "--candidates-dir", "data/daily_scans",
            "--backtest-csv", "data/backtest_trades.csv",
            "--daily-dir", "data/daily_klines",
            "--hourly-dir", "data/hourly_klines",
            "--output-dir", "models",
            "--epochs", "100",
            "--batch-size", "8",
            "--lr", "0.001",
            "--device", "cpu",
        ]

        logger.info("训练参数:")
        for i in range(1, len(sys.argv), 2):
            logger.info(f"  {sys.argv[i]}: {sys.argv[i+1]}")

        # 执行训练
        train_main()

        # 恢复原始参数
        sys.argv = original_argv

        logger.info("✓ 训练完成！")
        return True

    except Exception as exc:
        logger.error(f"训练失败: {exc}", exc_info=True)
        return False


def main():
    """主流程"""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("自动化训练流程启动")
    logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 步骤1: 监控数据收集
    if not monitor_data_collection():
        logger.error("数据收集失败")
        return 1

    # 步骤2: 验证数据
    if not verify_data():
        logger.error("数据验证失败")
        return 1

    # 步骤3: 训练模型
    if not train_model():
        logger.error("模型训练失败")
        return 1

    # 完成
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info("=" * 60)
    logger.info("✓ 所有任务完成！")
    logger.info(f"总耗时: {duration}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

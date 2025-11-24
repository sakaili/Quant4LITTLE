#!/usr/bin/env python3
"""
重新生成历史候选扫描数据的脚本。

该脚本用于修复旧的扫描数据中的时间戳问题（数据泄露），
重新运行历史日期的扫描，确保每个日期使用的都是该日期及之前的数据。

使用方法:
    python scripts/regenerate_historical_scans.py --start 2025-01-01 --end 2025-11-20 --bottom-n 80
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.daily_candidate_scan import run_scan
from scripts.data_fetcher import BinanceDataFetcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="重新生成历史候选扫描数据（修复时间戳泄露问题）"
    )
    parser.add_argument("--start", type=str, required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--bottom-n", type=int, default=80, help="候选池大小（倒数N名）")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/daily_scans"), help="输出目录"
    )
    parser.add_argument("--cooldown", type=float, default=0.2, help="资金费率查询间隔（秒）")
    parser.add_argument(
        "--skip-existing", action="store_true", help="跳过已存在的文件"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    if start_date > end_date:
        logger.error("开始日期必须早于或等于结束日期")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 创建一个共享的 fetcher 以复用连接
    fetcher = BinanceDataFetcher()
    logger.info("开始重新生成从 %s 到 %s 的历史扫描数据", start_date, end_date)

    current = start_date
    success_count = 0
    skip_count = 0
    error_count = 0

    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        output_file = args.output_dir / f"candidates_{date_str}.csv"

        if args.skip_existing and output_file.exists():
            logger.info("跳过已存在的文件: %s", output_file)
            skip_count += 1
            current += timedelta(days=1)
            continue

        try:
            logger.info("正在扫描 %s ...", current)
            candidate_df = run_scan(
                as_of=current,
                bottom_n=args.bottom_n,
                timeframe="1d",
                funding_cooldown=args.cooldown,
                fetcher=fetcher,
            )

            candidate_df.to_csv(output_file, index=False)
            logger.info(
                "✓ 完成 %s: %d 个候选 -> %s",
                current,
                len(candidate_df),
                output_file,
            )
            success_count += 1

        except Exception as exc:
            logger.error("✗ 扫描 %s 失败: %s", current, exc, exc_info=True)
            error_count += 1

        current += timedelta(days=1)

    logger.info(
        "完成！成功: %d, 跳过: %d, 失败: %d",
        success_count,
        skip_count,
        error_count,
    )

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

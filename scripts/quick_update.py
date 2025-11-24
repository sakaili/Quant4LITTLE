#!/usr/bin/env python3
"""
快速更新现有交易对到2025-11-23
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, timezone

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_fetcher import BinanceDataFetcher


def quick_update():
    """快速更新现有文件"""

    daily_dir = Path("data/daily_klines")
    hourly_dir = Path("data/hourly_klines")

    print("=" * 60)
    print("快速更新现有K线数据到最新 (2025-11-23)")
    print("=" * 60)

    # 获取现有文件列表
    existing_daily = list(daily_dir.glob("*_1d.csv"))
    print(f"\n找到 {len(existing_daily)} 个日线文件")

    if len(existing_daily) == 0:
        print("没有现有文件，请先运行完整更新")
        return

    # 初始化获取器
    fetcher = BinanceDataFetcher()

    # 只更新前10个文件作为测试
    test_files = existing_daily[:min(10, len(existing_daily))]

    print(f"\n更新前 {len(test_files)} 个文件...")

    success = 0
    for i, daily_file in enumerate(test_files, 1):
        try:
            # 从文件名提取symbol
            # 例如: BTC_USDT_USDT_1d.csv -> BTC/USDT:USDT
            filename = daily_file.stem  # BTC_USDT_USDT_1d
            parts = filename.rsplit("_", 1)[0]  # BTC_USDT_USDT
            symbol_parts = parts.split("_")

            if len(symbol_parts) >= 3:
                # BTC_USDT_USDT -> BTC/USDT:USDT
                symbol = f"{symbol_parts[0]}/{symbol_parts[1]}:{symbol_parts[2]}"
            else:
                continue

            print(f"\n  [{i}/{len(test_files)}] {symbol}...", end=" ", flush=True)

            # 计算日期范围
            end_date = datetime.now(timezone.utc)
            daily_start = end_date - timedelta(days=540)
            hourly_start = end_date - timedelta(days=90)

            # 获取最新数据
            daily_df = fetcher.fetch_symbol_history_with_indicators(
                symbol=symbol,
                start=daily_start,
                end=end_date,
                timeframe="1d"
            )

            hourly_file = hourly_dir / daily_file.name.replace("_1d.csv", "_1h.csv")
            hourly_df = fetcher.fetch_symbol_history_with_indicators(
                symbol=symbol,
                start=hourly_start,
                end=end_date,
                timeframe="1h"
            )

            if daily_df is not None and len(daily_df) > 0:
                # 保存
                daily_df.to_csv(daily_file, index=False)

                if hourly_df is not None and len(hourly_df) > 0:
                    hourly_df.to_csv(hourly_file, index=False)

                latest = daily_df["timestamp"].max()
                print(f"OK (最新: {latest})")
                success += 1
            else:
                print("SKIP (无数据)")

        except Exception as e:
            print(f"FAIL ({e})")
            continue

    print("\n" + "=" * 60)
    print(f"完成! 成功更新 {success}/{len(test_files)} 个文件")
    print("=" * 60)

    # 检查一个文件的最新日期
    if success > 0:
        sample = pd.read_csv(test_files[0], parse_dates=["timestamp"])
        print(f"\n示例文件最新日期: {sample['timestamp'].max()}")


if __name__ == "__main__":
    quick_update()

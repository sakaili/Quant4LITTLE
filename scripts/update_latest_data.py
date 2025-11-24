#!/usr/bin/env python3
"""
更新所有交易对的K线数据到最新日期
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_fetcher import BinanceDataFetcher


def update_all_klines(output_dir_daily: Path, output_dir_hourly: Path):
    """更新所有K线数据"""

    print("=" * 60)
    print("开始更新K线数据到最新...")
    print("=" * 60)

    # 创建目录
    output_dir_daily.mkdir(parents=True, exist_ok=True)
    output_dir_hourly.mkdir(parents=True, exist_ok=True)

    # 初始化数据获取器
    fetcher = BinanceDataFetcher()

    # 获取所有USDT永续合约
    print("\n[1/3] 获取市场列表...")
    markets = fetcher.fetch_usdt_perp_symbols()
    print(f"  找到 {len(markets)} 个USDT永续合约")

    # 获取24小时行情数据
    print("\n[2/3] 获取24小时行情统计...")
    ticker_df = fetcher.fetch_24h_tickers()
    print(f"  获取 {len(ticker_df)} 个交易对的行情数据")

    # 按交易量排序，选择后200名（流动性最低的垃圾币）
    ticker_df = ticker_df.sort_values("quote_volume", ascending=True)
    bottom_symbols = ticker_df.head(200)["symbol"].tolist()

    print(f"\n[3/3] 更新后 {len(bottom_symbols)} 名低流动性交易对的K线数据...")
    print(f"  日线保存到: {output_dir_daily}")
    print(f"  小时线保存到: {output_dir_hourly}")

    success_count = 0
    failed_symbols = []

    for i, symbol in enumerate(bottom_symbols, 1):
        try:
            print(f"\n  [{i}/{len(bottom_symbols)}] {symbol}...", end=" ", flush=True)

            # 计算日期范围
            from datetime import timezone
            end_date = datetime.now(timezone.utc)
            daily_start = end_date - timedelta(days=540)
            hourly_start = end_date - timedelta(days=90)

            # 获取日线数据（最近540天）并带指标
            daily_df = fetcher.fetch_symbol_history_with_indicators(
                symbol=symbol,
                start=daily_start,
                end=end_date,
                timeframe="1d"
            )

            # 获取小时线数据（最近90天）并带指标
            hourly_df = fetcher.fetch_symbol_history_with_indicators(
                symbol=symbol,
                start=hourly_start,
                end=end_date,
                timeframe="1h"
            )

            if daily_df is not None and len(daily_df) > 0:
                # 保存日线
                symbol_clean = symbol.replace("/", "_").replace(":", "_")
                daily_file = output_dir_daily / f"{symbol_clean}_1d.csv"
                daily_df.to_csv(daily_file, index=False)

                # 保存小时线
                if hourly_df is not None and len(hourly_df) > 0:
                    hourly_file = output_dir_hourly / f"{symbol_clean}_1h.csv"
                    hourly_df.to_csv(hourly_file, index=False)

                latest_date = daily_df["timestamp"].max()
                print(f"OK (最新: {latest_date})")
                success_count += 1
            else:
                print("SKIP (无数据)")
                failed_symbols.append(symbol)

        except Exception as e:
            print(f"FAIL ({e})")
            failed_symbols.append(symbol)
            continue

    print("\n" + "=" * 60)
    print("数据更新完成")
    print("=" * 60)
    print(f"\n统计:")
    print(f"  成功: {success_count}/{len(bottom_symbols)}")
    print(f"  失败: {len(failed_symbols)}")

    if failed_symbols:
        print(f"\n失败列表: {failed_symbols[:10]}")
        if len(failed_symbols) > 10:
            print(f"  ... 还有 {len(failed_symbols) - 10} 个")


def main():
    """主函数"""
    daily_dir = Path("data/daily_klines")
    hourly_dir = Path("data/hourly_klines")

    update_all_klines(daily_dir, hourly_dir)


if __name__ == "__main__":
    main()

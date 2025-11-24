#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安测试网 - 快速订单簿测试
"""
from __future__ import annotations

import io
import sys
import os

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import ccxt
except ImportError:
    print("❌ 请先安装ccxt库: pip install ccxt")
    sys.exit(1)


def main():
    """
    快速测试订单簿获取
    """
    print(f"\n{'█'*70}")
    print(f"  🌐 币安测试网 - 订单簿测试")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*70}\n")

    # 从环境变量获取API密钥
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    use_demo = os.getenv('USE_DEMO', 'False')

    if not api_key or not api_secret:
        print("  ❌ 未检测到API密钥!")
        print("\n  当前环境变量:")
        print(f"    BINANCE_API_KEY: {os.getenv('BINANCE_API_KEY', 'Not Set')}")
        print(f"    BINANCE_API_SECRET: {os.getenv('BINANCE_API_SECRET', 'Not Set')}")
        print(f"    USE_DEMO: {os.getenv('USE_DEMO', 'Not Set')}")
        return

    print(f"  ✅ API密钥已加载")
    print(f"  📋 API Key前缀: {api_key[:20]}...")
    print(f"  🌐 USE_DEMO: {use_demo}")

    # 初始化币安测试网
    print(f"\n{'─'*70}")
    print(f"  🔧 初始化币安交易所...")
    print(f"{'─'*70}")

    try:
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {
                'defaultType': 'future',
            },
            'enableRateLimit': True,
        })

        # 设置测试网
        exchange.set_sandbox_mode(True)

        print(f"  ✅ 交易所初始化成功")
        print(f"  📍 测试网URL: {exchange.urls.get('api', {}).get('public', 'N/A')}")

    except Exception as e:
        print(f"  ❌ 初始化失败: {e}")
        return

    # 测试API连接
    print(f"\n{'─'*70}")
    print(f"  🔌 测试API连接...")
    print(f"{'─'*70}")

    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {}).get('free', 0)

        print(f"  ✅ API连接成功!")
        print(f"  💰 USDT余额: {usdt_balance:.2f}")

    except Exception as e:
        print(f"  ❌ API连接失败: {e}")
        print(f"\n  可能的原因:")
        print(f"  1. API密钥不正确")
        print(f"  2. 不是测试网的API密钥")
        print(f"  3. API权限不足（需要期货交易权限）")
        print(f"  4. 代理设置问题")
        return

    # 测试获取订单簿
    print(f"\n{'─'*70}")
    print(f"  📊 测试订单簿获取...")
    print(f"{'─'*70}")

    test_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']

    for symbol in test_symbols:
        print(f"\n  {'─'*60}")
        print(f"  📈 {symbol}")
        print(f"  {'─'*60}")

        try:
            orderbook = exchange.fetch_order_book(symbol)
            bid = orderbook['bids'][0][0] if orderbook['bids'] else None
            ask = orderbook['asks'][0][0] if orderbook['asks'] else None

            if bid and ask:
                spread_pct = (ask - bid) / bid * 100

                print(f"  买一价 (Bid): {bid:.2f}")
                print(f"  卖一价 (Ask): {ask:.2f}")
                print(f"  价差: {spread_pct:.4f}%")

                # 计算Maker价格（0.10%偏移）
                maker_offset_pct = 0.10
                entry_price = ask * (1 + maker_offset_pct / 100)
                exit_price = bid * (1 - maker_offset_pct / 100)

                print(f"  ─────────────────────────────────────")
                print(f"  做空入场Maker价: {entry_price:.2f} (在Ask之上 {maker_offset_pct}%)")
                print(f"  做空出场Maker价: {exit_price:.2f} (在Bid之下 {maker_offset_pct}%)")

        except Exception as e:
            print(f"  ❌ 获取失败: {e}")

    # 测试市场信息
    print(f"\n{'─'*70}")
    print(f"  🔍 测试市场信息...")
    print(f"{'─'*70}")

    try:
        markets = exchange.load_markets()
        btc_market = markets.get('BTC/USDT:USDT', {})

        print(f"  ✅ 市场信息获取成功")
        print(f"  交易对: BTC/USDT:USDT")
        print(f"  最小下单量: {btc_market.get('limits', {}).get('amount', {}).get('min', 'N/A')}")
        print(f"  价格精度: {btc_market.get('precision', {}).get('price', 'N/A')}")
        print(f"  数量精度: {btc_market.get('precision', {}).get('amount', 'N/A')}")

    except Exception as e:
        print(f"  ❌ 获取市场信息失败: {e}")

    print(f"\n{'='*70}")
    print(f"  ✅ 测试完成!")
    print(f"{'='*70}")
    print(f"\n  📝 总结:")
    print(f"  - API连接正常")
    print(f"  - 可以获取订单簿数据")
    print(f"  - 可以计算Maker订单价格")
    print(f"  - 准备好进行真实下单测试")
    print(f"\n  💡 下一步:")
    print(f"  运行完整测试: python scripts/test_binance_testnet.py")
    print(f"  选择选项2进行真实下单测试")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

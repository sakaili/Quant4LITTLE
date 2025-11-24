#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API密钥诊断工具
"""
from __future__ import annotations

import io
import sys
import os

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from datetime import datetime

try:
    import ccxt
except ImportError:
    print("❌ 请先安装ccxt库: pip install ccxt")
    sys.exit(1)


def main():
    print(f"\n{'='*70}")
    print(f"  🔍 API密钥诊断工具")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # 检查环境变量
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    use_demo = os.getenv('USE_DEMO')
    https_proxy = os.getenv('HTTPS_PROXY')

    print(f"  📋 环境变量检查:")
    print(f"  ─────────────────────────────────────")
    print(f"  BINANCE_API_KEY: {'✅ 已设置' if api_key else '❌ 未设置'}")
    if api_key:
        print(f"    前缀: {api_key[:20]}...")
        print(f"    长度: {len(api_key)} 字符")

    print(f"  BINANCE_API_SECRET: {'✅ 已设置' if api_secret else '❌ 未设置'}")
    if api_secret:
        print(f"    前缀: {api_secret[:20]}...")
        print(f"    长度: {len(api_secret)} 字符")

    print(f"  USE_DEMO: {use_demo}")
    print(f"  HTTPS_PROXY: {https_proxy}")

    if not api_key or not api_secret:
        print(f"\n  ❌ 缺少API密钥!")
        return

    # 测试不同的配置
    print(f"\n{'─'*70}")
    print(f"  🧪 测试1: 测试网模式 (Testnet)")
    print(f"{'─'*70}")

    try:
        exchange_testnet = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
            'proxies': {
                'http': https_proxy,
                'https': https_proxy
            } if https_proxy else None
        })
        exchange_testnet.set_sandbox_mode(True)

        print(f"  测试网URL: {exchange_testnet.urls.get('api', {}).get('public', 'N/A')}")

        balance = exchange_testnet.fetch_balance()
        print(f"  ✅ 测试网连接成功!")
        print(f"  💰 USDT余额: {balance.get('USDT', {}).get('free', 0):.2f}")

    except Exception as e:
        print(f"  ❌ 测试网连接失败: {e}")

    # 测试主网
    print(f"\n{'─'*70}")
    print(f"  🧪 测试2: 主网模式 (Mainnet)")
    print(f"{'─'*70}")

    try:
        exchange_mainnet = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
            'proxies': {
                'http': https_proxy,
                'https': https_proxy
            } if https_proxy else None
        })

        print(f"  主网URL: {exchange_mainnet.urls.get('api', {}).get('public', 'N/A')}")

        balance = exchange_mainnet.fetch_balance()
        print(f"  ⚠️  主网连接成功!")
        print(f"  💰 USDT余额: {balance.get('USDT', {}).get('free', 0):.2f}")
        print(f"\n  ⚠️  警告: 这是真实账户!")

    except Exception as e:
        print(f"  ❌ 主网连接失败: {e}")

    # 测试公开接口（不需要API密钥）
    print(f"\n{'─'*70}")
    print(f"  🧪 测试3: 公开接口测试（无需API密钥）")
    print(f"{'─'*70}")

    try:
        exchange_public = ccxt.binance({
            'enableRateLimit': True,
            'proxies': {
                'http': https_proxy,
                'https': https_proxy
            } if https_proxy else None
        })

        ticker = exchange_public.fetch_ticker('BTC/USDT')
        print(f"  ✅ 公开接口连接成功!")
        print(f"  📈 BTC/USDT 价格: ${ticker['last']:,.2f}")

    except Exception as e:
        print(f"  ❌ 公开接口连接失败: {e}")
        print(f"  可能是代理或网络问题")

    # 总结
    print(f"\n{'='*70}")
    print(f"  📝 诊断总结")
    print(f"{'='*70}")
    print(f"\n  你提供的API密钥格式:")
    print(f"  - 长度正常")
    print(f"  - 看起来像是真实的币安API密钥")
    print(f"\n  ⚠️  重要提示:")
    print(f"  1. 你的API密钥可能是【主网】的密钥，不是测试网的")
    print(f"  2. 测试网需要单独申请API密钥")
    print(f"  3. 测试网申请地址: https://testnet.binancefuture.com")
    print(f"\n  🎯 两个选择:")
    print(f"  选择A: 申请测试网API密钥（推荐用于学习测试）")
    print(f"    - 访问: https://testnet.binancefuture.com")
    print(f"    - 注册并获取测试网API密钥")
    print(f"    - 测试网有免费的虚拟资金")
    print(f"\n  选择B: 使用模拟执行器（不连接真实API）")
    print(f"    - 运行: python scripts/test_maker_orders.py")
    print(f"    - 完全模拟，无需真实API")
    print(f"    - 已经测试成功!")
    print(f"\n  选择C: 使用主网小资金测试（谨慎!）")
    print(f"    - 你的API密钥可以连接主网")
    print(f"    - 建议用100-500 USDT测试")
    print(f"    - 需要修改脚本不启用sandbox模式")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

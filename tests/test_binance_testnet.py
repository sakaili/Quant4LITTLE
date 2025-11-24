#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安测试网 - Maker订单真实测试
使用真实API连接测试网进行下单测试
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
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import ccxt
except ImportError:
    print("❌ 请先安装ccxt库: pip install ccxt")
    sys.exit(1)


class BinanceTestnetMakerExecutor:
    """
    币安测试网Maker订单执行器
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        maker_offset_pct: float = 0.10,
        max_wait_seconds: int = 60,
        check_interval: float = 2.0
    ):
        """
        初始化币安测试网执行器

        Args:
            api_key: 测试网API密钥
            api_secret: 测试网API密钥
            maker_offset_pct: Maker订单离盘口的百分比
            max_wait_seconds: 最长等待时间
            check_interval: 检查间隔
        """
        self.maker_offset_pct = maker_offset_pct
        self.max_wait_seconds = max_wait_seconds
        self.check_interval = check_interval

        # 初始化币安测试网
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {
                'defaultType': 'future',  # 使用永续合约
            },
            'enableRateLimit': True,
        })

        # 设置测试网URL
        self.exchange.set_sandbox_mode(True)

        print(f"\n{'='*70}")
        print(f"  🌐 币安测试网Maker执行器已初始化")
        print(f"{'='*70}")
        print(f"  Maker偏移: {maker_offset_pct}%")
        print(f"  最长等待: {max_wait_seconds}秒")
        print(f"  检查间隔: {check_interval}秒")
        print(f"  测试网模式: ✅")
        print(f"{'='*70}\n")

    def test_connection(self) -> bool:
        """
        测试API连接
        """
        try:
            print(f"  🔌 测试API连接...")
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']

            print(f"  ✅ API连接成功!")
            print(f"  💰 USDT余额: {usdt_balance:.2f}")
            return True

        except Exception as e:
            print(f"  ❌ API连接失败: {e}")
            return False

    def get_orderbook(self, symbol: str) -> dict | None:
        """
        获取实时订单簿
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol)
            bid = orderbook['bids'][0][0] if orderbook['bids'] else None
            ask = orderbook['asks'][0][0] if orderbook['asks'] else None

            if bid and ask:
                spread_pct = (ask - bid) / bid * 100
                return {
                    'bid': bid,
                    'ask': ask,
                    'spread_pct': spread_pct
                }
            else:
                return None

        except Exception as e:
            print(f"  ❌ 获取订单簿失败: {e}")
            return None

    def calculate_maker_price(self, side: str, orderbook: dict) -> float:
        """
        计算Maker订单价格
        """
        bid = orderbook['bid']
        ask = orderbook['ask']

        if side == 'short_entry':
            # 做空入场: 在ask之上挂卖单
            price = ask * (1 + self.maker_offset_pct / 100)
        elif side == 'short_exit':
            # 做空出场: 在bid之下挂买单
            price = bid * (1 - self.maker_offset_pct / 100)
        else:
            raise ValueError(f"Unknown side: {side}")

        return price

    def get_price_precision(self, symbol: str) -> int:
        """
        获取价格精度
        """
        try:
            market = self.exchange.market(symbol)
            return market['precision']['price']
        except:
            return 4  # 默认4位小数

    def place_maker_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        reduce_only: bool = False
    ) -> dict | None:
        """
        下真实的Maker限价单

        Args:
            symbol: 交易对 (例如: 'BTC/USDT:USDT')
            side: 'short_entry' 或 'short_exit'
            amount: 数量
            reduce_only: 是否仅减仓

        Returns:
            订单信息
        """
        # 获取订单簿
        orderbook = self.get_orderbook(symbol)
        if not orderbook:
            print(f"  ❌ 无法获取订单簿")
            return None

        # 计算限价
        limit_price = self.calculate_maker_price(side, orderbook)

        # 获取价格精度
        price_precision = self.get_price_precision(symbol)
        limit_price = round(limit_price, price_precision)

        # 确定订单方向
        order_side = 'sell' if side == 'short_entry' else 'buy'

        # 下单参数
        params = {}
        if reduce_only:
            params['reduceOnly'] = True

        try:
            print(f"\n{'─'*70}")
            print(f"  📝 正在下限价单...")
            print(f"{'─'*70}")
            print(f"  交易对: {symbol}")
            print(f"  方向: {order_side.upper()} ({side})")
            print(f"  限价: {limit_price}")
            print(f"  数量: {amount}")
            print(f"  盘口: Bid={orderbook['bid']:.6f}, Ask={orderbook['ask']:.6f}, Spread={orderbook['spread_pct']:.3f}%")
            print(f"  仅减仓: {reduce_only}")

            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=order_side,
                amount=amount,
                price=limit_price,
                params=params
            )

            print(f"  ✅ 限价单已下达!")
            print(f"  订单ID: {order['id']}")
            print(f"{'─'*70}")

            return order

        except Exception as e:
            print(f"  ❌ 下单失败: {e}")
            return None

    def wait_for_fill(
        self,
        symbol: str,
        order_id: str,
        cancel_if_timeout: bool = True
    ) -> tuple[bool, dict | None]:
        """
        等待订单成交
        """
        start_time = time.time()

        print(f"\n  ⏳ 等待订单成交...")

        while True:
            elapsed = time.time() - start_time

            if elapsed > self.max_wait_seconds:
                print(f"  ⏰ 订单超时 ({self.max_wait_seconds}秒)")

                if cancel_if_timeout:
                    try:
                        self.exchange.cancel_order(order_id, symbol)
                        print(f"  ❌ 订单已取消")
                        return False, None
                    except Exception as e:
                        print(f"  ⚠️  取消订单失败: {e}")
                        return False, None
                else:
                    return False, None

            # 查询订单状态
            try:
                order = self.exchange.fetch_order(order_id, symbol)
                status = order['status']
                filled = order.get('filled', 0)
                remaining = order.get('remaining', 0)

                if status == 'closed':
                    print(f"  ✅ 订单完全成交!")
                    print(f"  成交价: {order.get('average', 0):.6f}")
                    print(f"  成交量: {filled:.6f}")
                    print(f"  等待时间: {elapsed:.0f}秒")
                    return True, order

                elif status == 'canceled':
                    print(f"  ❌ 订单已被取消")
                    return False, order

                elif status == 'open':
                    print(f"  ⏱️  等待中... 已成交: {filled:.6f}/{filled + remaining:.6f} ({elapsed:.0f}秒)")
                    time.sleep(self.check_interval)

                else:
                    print(f"  ⚠️  未知状态: {status}")
                    time.sleep(self.check_interval)

            except Exception as e:
                print(f"  ⚠️  查询订单失败: {e}")
                time.sleep(self.check_interval)


def test_single_trade():
    """
    测试单笔交易流程
    """
    print(f"\n{'#'*70}")
    print(f"  🧪 测试场景: 单笔做空交易（真实API）")
    print(f"{'#'*70}\n")

    # 从环境变量获取API密钥
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    use_demo = os.getenv('USE_DEMO', 'False').lower() == 'true'

    if not api_key or not api_secret:
        print("  ❌ 请设置环境变量:")
        print("     $env:BINANCE_API_KEY = 'your_key'")
        print("     $env:BINANCE_API_SECRET = 'your_secret'")
        print("     $env:USE_DEMO = 'True'")
        return

    print(f"  📋 配置信息:")
    print(f"  API Key: {api_key[:10]}...{api_key[-10:]}")
    print(f"  测试网模式: {'✅' if use_demo else '❌'}")

    # 创建执行器
    executor = BinanceTestnetMakerExecutor(
        api_key=api_key,
        api_secret=api_secret,
        maker_offset_pct=0.10,
        max_wait_seconds=60,
        check_interval=2.0
    )

    # 测试连接
    if not executor.test_connection():
        print("\n  ❌ API连接失败，请检查配置")
        return

    # 选择交易对（使用BTC测试，因为流动性好）
    symbol = 'BTC/USDT:USDT'
    amount = 0.001  # 0.001 BTC

    print(f"\n{'='*70}")
    print(f"  📊 交易参数")
    print(f"{'='*70}")
    print(f"  交易对: {symbol}")
    print(f"  数量: {amount} BTC")
    print(f"{'='*70}\n")

    # 步骤1: 做空入场
    print(f"\n{'='*70}")
    print(f"  🔽 步骤1: 做空入场 (SHORT ENTRY)")
    print(f"{'='*70}")

    entry_order = executor.place_maker_order(
        symbol=symbol,
        side='short_entry',
        amount=amount,
        reduce_only=False
    )

    if not entry_order:
        print("\n  ❌ 入场失败")
        return

    # 等待入场成交
    is_filled, filled_entry = executor.wait_for_fill(
        symbol=symbol,
        order_id=entry_order['id'],
        cancel_if_timeout=True
    )

    if not is_filled:
        print("\n  ❌ 入场订单未成交")
        return

    entry_price = filled_entry['average']
    print(f"\n  ✅ 做空入场成功! 价格: {entry_price:.2f}")

    # 步骤2: 模拟持仓一段时间
    print(f"\n{'='*70}")
    print(f"  ⏰ 步骤2: 模拟持仓（等待5秒）")
    print(f"{'='*70}")

    for i in range(5):
        print(f"  ⏱️  持仓中... {i+1}/5秒")
        time.sleep(1)

    # 步骤3: 做空出场
    print(f"\n{'='*70}")
    print(f"  🔼 步骤3: 做空出场 (SHORT EXIT)")
    print(f"{'='*70}")

    exit_order = executor.place_maker_order(
        symbol=symbol,
        side='short_exit',
        amount=amount,
        reduce_only=True  # 仅减仓
    )

    if not exit_order:
        print("\n  ⚠️  出场下单失败，使用市价单紧急平仓")
        try:
            emergency_order = executor.exchange.create_market_order(
                symbol=symbol,
                side='buy',
                amount=amount,
                params={'reduceOnly': True}
            )
            print(f"  ✅ 市价单平仓成功")
            return
        except Exception as e:
            print(f"  ❌ 紧急平仓失败: {e}")
            return

    # 等待出场成交
    is_filled, filled_exit = executor.wait_for_fill(
        symbol=symbol,
        order_id=exit_order['id'],
        cancel_if_timeout=False  # 不自动取消，手动处理
    )

    if not is_filled:
        print("\n  ⚠️  出场订单未成交，使用市价单紧急平仓")
        try:
            # 先取消未成交的限价单
            executor.exchange.cancel_order(exit_order['id'], symbol)

            # 市价单平仓
            emergency_order = executor.exchange.create_market_order(
                symbol=symbol,
                side='buy',
                amount=amount,
                params={'reduceOnly': True}
            )
            print(f"  ✅ 市价单平仓成功")

            # 获取市价单成交信息
            filled_exit = executor.exchange.fetch_order(emergency_order['id'], symbol)

        except Exception as e:
            print(f"  ❌ 紧急平仓失败: {e}")
            print(f"  ⚠️  请手动平仓!")
            return

    exit_price = filled_exit['average']
    print(f"\n  ✅ 做空出场成功! 价格: {exit_price:.2f}")

    # 步骤4: 计算盈亏
    print(f"\n{'='*70}")
    print(f"  💰 交易结果")
    print(f"{'='*70}")

    entry_avg = entry_price
    exit_avg = exit_price

    # 做空收益 = (入场价 - 出场价) / 入场价
    pnl_pct = (entry_avg - exit_avg) / entry_avg
    pnl_usdt = (entry_avg - exit_avg) * amount

    # 费用（需要从订单详情获取，这里简化）
    entry_fee = filled_entry.get('fee', {}).get('cost', 0)
    exit_fee = filled_exit.get('fee', {}).get('cost', 0)
    total_fee = entry_fee + exit_fee

    net_pnl_usdt = pnl_usdt - abs(total_fee)
    net_pnl_pct = net_pnl_usdt / (entry_avg * amount)

    print(f"  入场价: {entry_avg:.2f}")
    print(f"  出场价: {exit_avg:.2f}")
    print(f"  价格变动: {(exit_avg - entry_avg) / entry_avg * 100:.2f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  做空收益: {pnl_pct * 100:.2f}% ({pnl_usdt:.2f} USDT)")
    print(f"  交易费用: {abs(total_fee):.4f} USDT")
    print(f"  净收益: {net_pnl_pct * 100:.2f}% ({net_pnl_usdt:.2f} USDT)")
    print(f"{'='*70}\n")

    if pnl_usdt > 0:
        print(f"  🎉 交易盈利!")
    else:
        print(f"  📉 交易亏损")


def test_orderbook_only():
    """
    仅测试订单簿获取（不下单）
    """
    print(f"\n{'#'*70}")
    print(f"  🧪 测试场景: 订单簿获取测试")
    print(f"{'#'*70}\n")

    # 从环境变量获取API密钥
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        print("  ❌ 请设置环境变量")
        return

    # 创建执行器
    executor = BinanceTestnetMakerExecutor(
        api_key=api_key,
        api_secret=api_secret,
        maker_offset_pct=0.10,
        max_wait_seconds=60
    )

    # 测试连接
    if not executor.test_connection():
        return

    # 测试获取订单簿
    test_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'DOGE/USDT:USDT']

    for symbol in test_symbols:
        print(f"\n{'─'*70}")
        print(f"  📊 {symbol}")
        print(f"{'─'*70}")

        orderbook = executor.get_orderbook(symbol)

        if orderbook:
            print(f"  买一价 (Bid): {orderbook['bid']:.6f}")
            print(f"  卖一价 (Ask): {orderbook['ask']:.6f}")
            print(f"  价差: {orderbook['spread_pct']:.3f}%")

            # 计算Maker价格
            entry_price = executor.calculate_maker_price('short_entry', orderbook)
            exit_price = executor.calculate_maker_price('short_exit', orderbook)

            print(f"  做空入场价: {entry_price:.6f}")
            print(f"  做空出场价: {exit_price:.6f}")
        else:
            print(f"  ❌ 获取失败")


def main():
    """
    主函数
    """
    print(f"\n{'█'*70}")
    print(f"  🌐 币安测试网 - Maker订单真实测试")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*70}\n")

    # 检查环境变量
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    use_demo = os.getenv('USE_DEMO', 'False')

    if not api_key or not api_secret:
        print("  ❌ 未检测到API密钥!")
        print("\n  请在PowerShell中设置:")
        print("  $env:BINANCE_API_KEY = 'your_key'")
        print("  $env:BINANCE_API_SECRET = 'your_secret'")
        print("  $env:USE_DEMO = 'True'")
        print("\n  然后重新运行: python scripts/test_binance_testnet.py")
        return

    print(f"  ✅ API密钥已加载")
    print(f"  📋 API Key: {api_key[:10]}...{api_key[-10:]}")
    print(f"  🌐 测试网模式: {use_demo}")

    # 选择测试模式
    print(f"\n{'─'*70}")
    print(f"  请选择测试模式:")
    print(f"  1. 仅测试订单簿获取（不下单）")
    print(f"  2. 完整交易测试（真实下单）")
    print(f"{'─'*70}")

    choice = input("  请输入选项 (1/2): ").strip()

    if choice == '1':
        test_orderbook_only()
    elif choice == '2':
        test_single_trade()
    else:
        print("  ❌ 无效选项")

    print(f"\n{'█'*70}")
    print(f"  ✅ 测试完成!")
    print(f"{'█'*70}\n")


if __name__ == "__main__":
    main()

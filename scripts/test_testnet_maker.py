#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安测试网 - Maker订单真实下单测试
使用测试网API进行完整的做空交易测试
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


class BinanceTestnetMaker:
    """
    币安测试网Maker订单执行器
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        use_testnet: bool = True,
        maker_offset_pct: float = 0.10,
        max_wait_seconds: int = 60,
        check_interval: float = 2.0
    ):
        self.maker_offset_pct = maker_offset_pct
        self.max_wait_seconds = max_wait_seconds
        self.check_interval = check_interval
        self.use_testnet = use_testnet

        # 获取代理设置
        https_proxy = os.getenv('HTTPS_PROXY')

        # 初始化币安
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {
                'defaultType': 'future',
            },
            'enableRateLimit': True,
            'proxies': {
                'http': https_proxy,
                'https': https_proxy
            } if https_proxy else None
        })

        # 设置测试网或主网
        if use_testnet:
            self.exchange.set_sandbox_mode(True)

        print(f"\n{'='*70}")
        print(f"  🌐 币安{'测试网' if use_testnet else '主网'}Maker执行器")
        print(f"{'='*70}")
        print(f"  模式: {'🧪 测试网 (虚拟资金)' if use_testnet else '⚠️  主网 (真实资金)'}")
        print(f"  Maker偏移: {maker_offset_pct}%")
        print(f"  最长等待: {max_wait_seconds}秒")
        print(f"  代理: {https_proxy if https_proxy else '无'}")
        print(f"{'='*70}\n")

    def test_connection(self) -> bool:
        """测试连接并显示余额"""
        try:
            print(f"  🔌 测试API连接...")
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)

            print(f"  ✅ 连接成功!")
            print(f"  💰 USDT余额: {usdt_balance:.2f}")

            if usdt_balance < 10:
                print(f"  ⚠️  余额不足，建议至少10 USDT")
                return False

            return True

        except Exception as e:
            print(f"  ❌ 连接失败: {e}")
            return False

    def get_orderbook(self, symbol: str) -> dict | None:
        """获取订单簿"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol)
            bid = orderbook['bids'][0][0] if orderbook['bids'] else None
            ask = orderbook['asks'][0][0] if orderbook['asks'] else None

            if bid and ask:
                spread_pct = (ask - bid) / bid * 100
                return {'bid': bid, 'ask': ask, 'spread_pct': spread_pct}
            return None

        except Exception as e:
            print(f"  ❌ 获取订单簿失败: {e}")
            return None

    def calculate_maker_price(self, side: str, orderbook: dict) -> float:
        """计算Maker价格"""
        bid, ask = orderbook['bid'], orderbook['ask']

        if side == 'short_entry':
            # 做空入场: 在ask之上挂卖单
            return ask * (1 + self.maker_offset_pct / 100)
        elif side == 'short_exit':
            # 做空出场: 在bid之下挂买单
            return bid * (1 - self.maker_offset_pct / 100)
        else:
            raise ValueError(f"Unknown side: {side}")

    def format_price(self, symbol: str, price: float) -> float:
        """格式化价格到正确精度"""
        try:
            market = self.exchange.market(symbol)
            precision = market['precision']['price']

            if isinstance(precision, int):
                return round(price, precision)
            else:
                # 如果precision是tick size
                return round(price / precision) * precision
        except:
            return round(price, 2)

    def format_amount(self, symbol: str, amount: float) -> float:
        """格式化数量到正确精度"""
        try:
            market = self.exchange.market(symbol)
            precision = market['precision']['amount']

            if isinstance(precision, int):
                return round(amount, precision)
            else:
                return round(amount / precision) * precision
        except:
            return round(amount, 3)

    def place_maker_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        reduce_only: bool = False
    ) -> dict | None:
        """下Maker限价单"""
        orderbook = self.get_orderbook(symbol)
        if not orderbook:
            return None

        # 计算限价
        limit_price = self.calculate_maker_price(side, orderbook)
        limit_price = self.format_price(symbol, limit_price)
        amount = self.format_amount(symbol, amount)

        order_side = 'sell' if side == 'short_entry' else 'buy'
        params = {'reduceOnly': True} if reduce_only else {}

        try:
            print(f"\n{'─'*70}")
            print(f"  📝 下{'做空入场' if side == 'short_entry' else '做空出场'}限价单")
            print(f"{'─'*70}")
            print(f"  交易对: {symbol}")
            print(f"  方向: {order_side.upper()}")
            print(f"  限价: {limit_price:.2f}")
            print(f"  数量: {amount:.3f}")
            print(f"  盘口: Bid={orderbook['bid']:.2f}, Ask={orderbook['ask']:.2f}")
            print(f"  价差: {orderbook['spread_pct']:.4f}%")

            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=order_side,
                amount=amount,
                price=limit_price,
                params=params
            )

            print(f"  ✅ 订单已下达! ID: {order['id']}")
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
        """等待订单成交"""
        start_time = time.time()
        print(f"\n  ⏳ 等待订单成交 (最多{self.max_wait_seconds}秒)...")

        while True:
            elapsed = time.time() - start_time

            if elapsed > self.max_wait_seconds:
                print(f"  ⏰ 超时 ({self.max_wait_seconds}秒)")

                if cancel_if_timeout:
                    try:
                        self.exchange.cancel_order(order_id, symbol)
                        print(f"  ❌ 订单已取消")
                    except Exception as e:
                        print(f"  ⚠️  取消失败: {e}")
                return False, None

            try:
                order = self.exchange.fetch_order(order_id, symbol)
                status = order['status']
                filled = order.get('filled', 0)
                remaining = order.get('remaining', 0)

                if status == 'closed':
                    print(f"  ✅ 完全成交!")
                    print(f"    成交价: {order.get('average', 0):.2f}")
                    print(f"    成交量: {filled:.4f}")
                    print(f"    耗时: {elapsed:.0f}秒")
                    return True, order

                elif status == 'canceled':
                    print(f"  ❌ 订单已取消")
                    return False, order

                elif status == 'open':
                    progress = filled / (filled + remaining) * 100 if (filled + remaining) > 0 else 0
                    print(f"  ⏱️  等待中... {progress:.0f}% ({elapsed:.0f}秒)", end='\r')
                    time.sleep(self.check_interval)

            except Exception as e:
                print(f"  ⚠️  查询失败: {e}")
                time.sleep(self.check_interval)

    def market_close_position(self, symbol: str, amount: float) -> bool:
        """紧急市价平仓"""
        try:
            print(f"\n  🚨 紧急市价平仓...")
            order = self.exchange.create_market_order(
                symbol=symbol,
                side='buy',
                amount=amount,
                params={'reduceOnly': True}
            )
            print(f"  ✅ 市价平仓成功! ID: {order['id']}")
            return True
        except Exception as e:
            print(f"  ❌ 市价平仓失败: {e}")
            return False


def test_short_trade_full():
    """
    完整做空交易测试
    """
    print(f"\n{'█'*70}")
    print(f"  🧪 币安测试网 - Maker订单做空交易测试")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*70}\n")

    # 从环境变量获取配置
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    use_testnet = os.getenv('USE_TESTNET', 'True').lower() == 'true'

    if not api_key or not api_secret:
        print("  ❌ 请设置环境变量:")
        print("     $env:BINANCE_API_KEY = 'your_key'")
        print("     $env:BINANCE_API_SECRET = 'your_secret'")
        print("     $env:USE_TESTNET = 'True'  # True=测试网, False=主网")
        return

    # 创建执行器
    executor = BinanceTestnetMaker(
        api_key=api_key,
        api_secret=api_secret,
        use_testnet=use_testnet,
        maker_offset_pct=0.10,
        max_wait_seconds=60
    )

    # 测试连接
    if not executor.test_connection():
        return

    # 交易参数（使用BTC，流动性好）
    symbol = 'BTC/USDT:USDT'
    amount = 0.001  # 0.001 BTC，约80-100 USDT

    print(f"\n{'='*70}")
    print(f"  📊 交易参数")
    print(f"{'='*70}")
    print(f"  交易对: {symbol}")
    print(f"  数量: {amount} BTC")
    print(f"  策略: 做空 (SHORT)")
    print(f"  订单类型: Maker限价单")
    print(f"{'='*70}")

    # 确认
    print(f"\n  ⚠️  确认信息:")
    print(f"  - 这将在{'测试网' if use_testnet else '⚠️ 主网'}进行真实下单")
    print(f"  - 使用{'虚拟资金' if use_testnet else '⚠️ 真实资金'}")

    input(f"\n  按回车键继续，或Ctrl+C取消...")

    # ==================== 步骤1: 做空入场 ====================
    print(f"\n{'='*70}")
    print(f"  🔽 步骤1/3: 做空入场")
    print(f"{'='*70}")

    entry_order = executor.place_maker_order(
        symbol=symbol,
        side='short_entry',
        amount=amount,
        reduce_only=False
    )

    if not entry_order:
        print("\n  ❌ 入场失败!")
        return

    is_filled, filled_entry = executor.wait_for_fill(
        symbol=symbol,
        order_id=entry_order['id'],
        cancel_if_timeout=True
    )

    if not is_filled:
        print("\n  ❌ 入场订单未成交，测试结束")
        return

    entry_price = filled_entry['average']
    print(f"\n  ✅ 做空入场成功! 价格: ${entry_price:,.2f}")

    # ==================== 步骤2: 模拟持仓 ====================
    print(f"\n{'='*70}")
    print(f"  ⏰ 步骤2/3: 模拟持仓")
    print(f"{'='*70}")
    print(f"  在真实场景中，会持仓5天")
    print(f"  现在模拟持仓10秒...")

    for i in range(10):
        # 查询当前价格
        ticker = executor.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        pnl_pct = (entry_price - current_price) / entry_price * 100
        pnl_usdt = (entry_price - current_price) * amount

        print(f"  ⏱️  持仓中... {i+1}/10秒 | "
              f"当前价: ${current_price:,.2f} | "
              f"浮动盈亏: {pnl_pct:+.2f}% ({pnl_usdt:+.2f} USDT)", end='\r')
        time.sleep(1)

    print()  # 换行

    # ==================== 步骤3: 做空出场 ====================
    print(f"\n{'='*70}")
    print(f"  🔼 步骤3/3: 做空出场")
    print(f"{'='*70}")

    exit_order = executor.place_maker_order(
        symbol=symbol,
        side='short_exit',
        amount=amount,
        reduce_only=True
    )

    if not exit_order:
        print("\n  ⚠️  出场下单失败，尝试市价平仓...")
        if executor.market_close_position(symbol, amount):
            print("\n  ✅ 市价平仓成功")
        else:
            print("\n  ❌ 平仓失败，请手动平仓!")
        return

    is_filled, filled_exit = executor.wait_for_fill(
        symbol=symbol,
        order_id=exit_order['id'],
        cancel_if_timeout=False
    )

    if not is_filled:
        print("\n  ⏰ 出场订单超时，尝试市价平仓...")
        try:
            executor.exchange.cancel_order(exit_order['id'], symbol)
        except:
            pass

        if executor.market_close_position(symbol, amount):
            # 重新获取成交信息
            time.sleep(1)
            trades = executor.exchange.fetch_my_trades(symbol, limit=1)
            if trades:
                exit_price = trades[0]['price']
                print(f"  ✅ 市价平仓成功! 价格: ${exit_price:,.2f}")
            else:
                print("\n  ⚠️  请手动检查持仓!")
                return
        else:
            print("\n  ❌ 平仓失败，请手动平仓!")
            return
    else:
        exit_price = filled_exit['average']
        print(f"\n  ✅ 做空出场成功! 价格: ${exit_price:,.2f}")

    # ==================== 结果汇总 ====================
    print(f"\n{'='*70}")
    print(f"  💰 交易结果")
    print(f"{'='*70}")

    pnl_pct = (entry_price - exit_price) / entry_price
    pnl_usdt = (entry_price - exit_price) * amount

    # 费用（从订单中获取）
    entry_fee = abs(filled_entry.get('fee', {}).get('cost', 0)) if filled_entry else 0
    exit_fee = abs(filled_exit.get('fee', {}).get('cost', 0)) if filled_exit else 0
    total_fee = entry_fee + exit_fee

    net_pnl_usdt = pnl_usdt - total_fee
    net_pnl_pct = net_pnl_usdt / (entry_price * amount)

    print(f"  入场价: ${entry_price:,.2f}")
    print(f"  出场价: ${exit_price:,.2f}")
    print(f"  价格变动: {(exit_price - entry_price) / entry_price * 100:+.2f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  做空毛收益: {pnl_pct * 100:+.2f}% (${pnl_usdt:+.2f})")
    print(f"  交易费用: -${total_fee:.4f}")
    print(f"  净收益: {net_pnl_pct * 100:+.2f}% (${net_pnl_usdt:+.2f})")
    print(f"{'='*70}")

    if net_pnl_usdt > 0:
        print(f"\n  🎉 交易盈利 ${net_pnl_usdt:.2f} USDT!")
    elif net_pnl_usdt < 0:
        print(f"\n  📉 交易亏损 ${abs(net_pnl_usdt):.2f} USDT")
    else:
        print(f"\n  ⚖️  盈亏平衡")

    print(f"\n{'='*70}")
    print(f"  ✅ 测试完成!")
    print(f"{'='*70}\n")


def main():
    """主函数"""
    try:
        test_short_trade_full()
    except KeyboardInterrupt:
        print(f"\n\n  ⚠️  用户取消测试")
    except Exception as e:
        print(f"\n\n  ❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

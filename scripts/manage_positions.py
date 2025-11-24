#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持仓管理 - 止盈止损
自动监控持仓，执行止盈止损
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


class PositionManager:
    """持仓管理器"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        use_testnet: bool = False,
        take_profit_pct: float = 30.0,  # 止盈30%
        stop_loss_pct: float = 200.0,   # 止损200%
    ):
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.use_testnet = use_testnet

        # 获取代理
        https_proxy = os.getenv('HTTPS_PROXY')

        # 初始化交易所
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
            'proxies': {
                'http': https_proxy,
                'https': https_proxy
            } if https_proxy else None
        })

        if use_testnet:
            self.exchange.set_sandbox_mode(True)

        print(f"\n{'='*70}")
        print(f"  💼 持仓管理器")
        print(f"{'='*70}")
        print(f"  模式: {'🧪 测试网' if use_testnet else '⚠️  实盘'}")
        print(f"  止盈: {take_profit_pct}%")
        print(f"  止损: {stop_loss_pct}%")
        print(f"{'='*70}\n")

    def get_positions(self) -> list:
        """获取当前持仓"""
        try:
            positions = self.exchange.fetch_positions()
            active = [p for p in positions if float(p.get('contracts', 0)) != 0]
            return active
        except Exception as e:
            print(f"  ❌ 获取持仓失败: {e}")
            return []

    def close_position(self, symbol: str, amount: float, side: str, position_side: str) -> bool:
        """平仓"""
        try:
            # 做空平仓 = 买入, 做多平仓 = 卖出
            close_side = 'buy' if side == 'short' else 'sell'

            print(f"    平仓: {symbol}")
            print(f"    方向: {close_side}")
            print(f"    数量: {abs(amount):.4f}")
            print(f"    持仓侧: {position_side}")

            # 尝试使用市价单平仓
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=abs(amount),
                params={
                    'positionSide': position_side  # 指定持仓侧
                }
            )

            print(f"    ✅ 平仓成功! 订单ID: {order['id']}")
            return True

        except Exception as e:
            print(f"    ❌ 平仓失败: {e}")
            return False

    def check_and_manage_positions(self):
        """检查并管理所有持仓"""
        print(f"{'─'*70}")
        print(f"  📊 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─'*70}\n")

        positions = self.get_positions()

        if len(positions) == 0:
            print("  ℹ️  无持仓")
            return

        print(f"  持仓数量: {len(positions)}")
        print(f"{'─'*70}\n")

        closed_count = 0

        for pos in positions:
            symbol = pos['symbol']
            contracts = float(pos.get('contracts', 0))
            entry_price = float(pos.get('entryPrice', 0))
            mark_price = float(pos.get('markPrice', 0))
            unrealized_pnl = float(pos.get('unrealizedPnl', 0))
            notional = float(pos.get('notional', 0))  # 名义价值
            position_side = pos.get('side', 'both').upper()  # LONG, SHORT, or BOTH

            # 使用CCXT返回的percentage字段（已经是百分比）
            pnl_pct = float(pos.get('percentage', 0))

            # 如果没有percentage字段，手动计算
            if pnl_pct == 0 and notional != 0:
                pnl_pct = (unrealized_pnl / abs(notional)) * 100

            # 根据position_side判断方向（不是contracts）
            side = position_side.lower()

            print(f"  {'📉' if side == 'short' else '📈'} {symbol}")
            print(f"    仓位: {abs(contracts):.4f} ({side.upper()})")
            print(f"    入场价: {entry_price:.4f}")
            print(f"    当前价: {mark_price:.4f}")
            print(f"    浮动盈亏: {unrealized_pnl:+.2f} USDT ({pnl_pct:+.2f}%)")

            # 检查止盈
            if pnl_pct >= self.take_profit_pct:
                print(f"    ✅ 触发止盈! ({pnl_pct:+.2f}% >= {self.take_profit_pct}%)")
                if self.close_position(symbol, contracts, side, position_side):
                    closed_count += 1
                print()
                continue

            # 检查止损
            if pnl_pct <= -self.stop_loss_pct:
                print(f"    🛑 触发止损! ({pnl_pct:+.2f}% <= -{self.stop_loss_pct}%)")
                if self.close_position(symbol, contracts, side, position_side):
                    closed_count += 1
                print()
                continue

            print()

        if closed_count > 0:
            print(f"{'='*70}")
            print(f"  📊 平仓汇总")
            print(f"{'='*70}")
            print(f"  平仓数量: {closed_count} 个")
            print(f"{'='*70}\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="持仓管理 - 止盈止损")
    parser.add_argument("--loop", action="store_true", help="循环监控模式")
    parser.add_argument("--interval", type=int, default=60, help="监控间隔(秒)")
    parser.add_argument("--take-profit", type=float, default=30.0, help="止盈百分比")
    parser.add_argument("--stop-loss", type=float, default=200.0, help="止损百分比")
    args = parser.parse_args()

    print(f"\n{'█'*70}")
    print(f"  💼 持仓管理 - 止盈止损")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*70}\n")

    # 环境变量
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    use_testnet = os.getenv('USE_TESTNET', 'False').lower() == 'true'

    if not api_key or not api_secret:
        print("  ❌ 请设置环境变量:")
        print("     $env:BINANCE_API_KEY = 'your_key'")
        print("     $env:BINANCE_API_SECRET = 'your_secret'")
        return

    # 创建管理器
    manager = PositionManager(
        api_key=api_key,
        api_secret=api_secret,
        use_testnet=use_testnet,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss
    )

    if args.loop:
        print(f"  🔄 循环监控模式 (每{args.interval}秒检查一次)")
        print(f"  按 Ctrl+C 停止\n")

        try:
            while True:
                manager.check_and_manage_positions()
                print(f"  💤 等待{args.interval}秒...")
                print(f"{'─'*70}\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n\n  ⚠️  用户中断")
    else:
        manager.check_and_manage_positions()

    print(f"{'='*70}")
    print(f"  ✅ 完成!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  ⚠️  用户中断")
    except Exception as e:
        print(f"\n\n  ❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

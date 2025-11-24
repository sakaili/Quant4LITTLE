#!/usr/bin/env python3
"""
Maker订单执行工具
用于将Paper Trading信号转换为实际的限价单(maker)执行策略
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple
from datetime import datetime
import ccxt


class MakerOrderExecutor:
    """
    Maker订单执行器
    用于执行限价单策略，避免市价单的高费用
    """

    def __init__(
        self,
        exchange: ccxt.Exchange,
        maker_offset_pct: float = 0.05,  # 离盘口的距离(百分比)
        max_wait_seconds: int = 60,      # 最长等待时间
        check_interval: float = 2.0      # 检查间隔(秒)
    ):
        """
        初始化

        Args:
            exchange: CCXT交易所对象
            maker_offset_pct: Maker订单离盘口的百分比距离(默认0.05%)
            max_wait_seconds: 最长等待订单成交时间(秒)
            check_interval: 检查订单状态的间隔(秒)
        """
        self.exchange = exchange
        self.maker_offset_pct = maker_offset_pct
        self.max_wait_seconds = max_wait_seconds
        self.check_interval = check_interval

    def get_orderbook(self, symbol: str) -> Dict:
        """
        获取订单簿

        Returns:
            {'bid': float, 'ask': float, 'spread_pct': float}
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
            print(f"[ERROR] 获取订单簿失败: {e}")
            return None

    def calculate_maker_price(
        self,
        side: str,  # 'short_entry', 'short_exit'
        orderbook: Dict
    ) -> Optional[float]:
        """
        计算Maker订单价格

        Args:
            side: 订单类型
                - 'short_entry': 做空入场 (卖出开仓)
                - 'short_exit': 做空出场 (买入平仓)
            orderbook: 订单簿数据

        Returns:
            限价单价格
        """
        if not orderbook:
            return None

        bid = orderbook['bid']
        ask = orderbook['ask']

        if side == 'short_entry':
            # 做空入场: 卖出开仓
            # 在ask之上挂单，等待买单吃我们的卖单
            # 我们是maker，对手是taker
            price = ask * (1 + self.maker_offset_pct / 100)
            return round(price, 8)

        elif side == 'short_exit':
            # 做空出场: 买入平仓
            # 在bid之下挂单，等待卖单吃我们的买单
            # 我们是maker，对手是taker
            price = bid * (1 - self.maker_offset_pct / 100)
            return round(price, 8)

        else:
            raise ValueError(f"Unknown side: {side}")

    def place_maker_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """
        下maker限价单

        Args:
            symbol: 交易对 (例如: 'DEXE/USDT:USDT')
            side: 'short_entry' 或 'short_exit'
            amount: 数量
            reduce_only: 是否仅减仓(平仓单设为True)

        Returns:
            订单信息字典
        """
        # 获取订单簿
        orderbook = self.get_orderbook(symbol)
        if not orderbook:
            print(f"[ERROR] 无法获取 {symbol} 订单簿")
            return None

        # 计算限价
        limit_price = self.calculate_maker_price(side, orderbook)
        if not limit_price:
            print(f"[ERROR] 无法计算限价")
            return None

        # 确定订单方向
        order_side = 'sell' if side == 'short_entry' else 'buy'

        # 下单参数
        params = {}
        if reduce_only:
            params['reduceOnly'] = True

        try:
            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=order_side,
                amount=amount,
                price=limit_price,
                params=params
            )

            print(f"[OK] 限价单已下达:")
            print(f"  Symbol: {symbol}")
            print(f"  Side: {order_side.upper()} ({side})")
            print(f"  Price: {limit_price}")
            print(f"  Amount: {amount}")
            print(f"  Order ID: {order['id']}")
            print(f"  盘口: Bid={orderbook['bid']:.6f}, Ask={orderbook['ask']:.6f}, Spread={orderbook['spread_pct']:.3f}%")

            return order

        except Exception as e:
            print(f"[ERROR] 下单失败: {e}")
            return None

    def wait_for_fill(
        self,
        symbol: str,
        order_id: str,
        cancel_if_timeout: bool = True
    ) -> Tuple[bool, Optional[Dict]]:
        """
        等待订单成交

        Args:
            symbol: 交易对
            order_id: 订单ID
            cancel_if_timeout: 超时是否取消订单

        Returns:
            (is_filled: bool, order_info: Dict)
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > self.max_wait_seconds:
                print(f"[TIMEOUT] 订单 {order_id} 超时未成交 ({self.max_wait_seconds}秒)")

                if cancel_if_timeout:
                    try:
                        self.exchange.cancel_order(order_id, symbol)
                        print(f"[OK] 订单已取消")
                        return False, None
                    except Exception as e:
                        print(f"[ERROR] 取消订单失败: {e}")
                        return False, None
                else:
                    return False, None

            # 查询订单状态
            try:
                order = self.exchange.fetch_order(order_id, symbol)
                status = order['status']
                filled = order['filled']
                remaining = order['remaining']

                if status == 'closed' and filled > 0:
                    print(f"[OK] 订单完全成交: {filled}")
                    return True, order

                elif status == 'canceled':
                    print(f"[CANCELED] 订单已被取消")
                    return False, order

                elif status == 'open':
                    print(f"[WAITING] 订单等待中... 已成交: {filled}/{filled + remaining} ({elapsed:.0f}s)")
                    time.sleep(self.check_interval)

                else:
                    print(f"[UNKNOWN] 未知订单状态: {status}")
                    time.sleep(self.check_interval)

            except Exception as e:
                print(f"[ERROR] 查询订单失败: {e}")
                time.sleep(self.check_interval)


def execute_short_entry_maker(
    executor: MakerOrderExecutor,
    symbol: str,
    amount: float
) -> Optional[Dict]:
    """
    执行做空入场(Maker方式)

    Args:
        executor: Maker订单执行器
        symbol: 交易对
        amount: 数量

    Returns:
        成交订单信息
    """
    print(f"\n{'='*60}")
    print(f"[SHORT ENTRY] {symbol} - Amount: {amount}")
    print(f"{'='*60}")

    # 下限价卖单
    order = executor.place_maker_order(
        symbol=symbol,
        side='short_entry',
        amount=amount,
        reduce_only=False
    )

    if not order:
        print(f"[FAILED] 下单失败")
        return None

    # 等待成交
    is_filled, filled_order = executor.wait_for_fill(
        symbol=symbol,
        order_id=order['id'],
        cancel_if_timeout=True
    )

    if is_filled:
        print(f"[SUCCESS] 做空入场成功")
        print(f"  成交价: {filled_order['average']}")
        print(f"  成交量: {filled_order['filled']}")
        return filled_order
    else:
        print(f"[FAILED] 订单未成交")
        return None


def execute_short_exit_maker(
    executor: MakerOrderExecutor,
    symbol: str,
    amount: float
) -> Optional[Dict]:
    """
    执行做空出场(Maker方式)

    Args:
        executor: Maker订单执行器
        symbol: 交易对
        amount: 数量

    Returns:
        成交订单信息
    """
    print(f"\n{'='*60}")
    print(f"[SHORT EXIT] {symbol} - Amount: {amount}")
    print(f"{'='*60}")

    # 下限价买单
    order = executor.place_maker_order(
        symbol=symbol,
        side='short_exit',
        amount=amount,
        reduce_only=True  # 平仓单
    )

    if not order:
        print(f"[FAILED] 下单失败")
        return None

    # 等待成交
    is_filled, filled_order = executor.wait_for_fill(
        symbol=symbol,
        order_id=order['id'],
        cancel_if_timeout=True
    )

    if is_filled:
        print(f"[SUCCESS] 做空出场成功")
        print(f"  成交价: {filled_order['average']}")
        print(f"  成交量: {filled_order['filled']}")
        return filled_order
    else:
        print(f"[FAILED] 订单未成交")
        return None


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    """
    示例: 如何使用Maker订单执行器
    """

    # 初始化交易所(需要API密钥)
    exchange = ccxt.binance({
        'apiKey': 'YOUR_API_KEY',
        'secret': 'YOUR_SECRET',
        'options': {
            'defaultType': 'future',  # 使用永续合约
        }
    })
    exchange.set_sandbox_mode(True)  # 使用测试网

    # 创建Maker执行器
    executor = MakerOrderExecutor(
        exchange=exchange,
        maker_offset_pct=0.05,  # 离盘口0.05%
        max_wait_seconds=60,    # 最多等待60秒
        check_interval=2.0      # 每2秒检查一次
    )

    # 执行做空入场
    symbol = 'DEXE/USDT:USDT'
    amount = 1.0  # 数量

    entry_order = execute_short_entry_maker(
        executor=executor,
        symbol=symbol,
        amount=amount
    )

    if entry_order:
        print(f"\n[持仓] 等待5天后平仓...")
        # 实际使用中应该等待5天

        # 执行做空出场
        exit_order = execute_short_exit_maker(
            executor=executor,
            symbol=symbol,
            amount=amount
        )

        if exit_order:
            # 计算盈亏
            entry_price = entry_order['average']
            exit_price = exit_order['average']
            pnl_pct = (entry_price - exit_price) / entry_price * 100

            print(f"\n{'='*60}")
            print(f"[交易完成]")
            print(f"  入场价: {entry_price:.6f}")
            print(f"  出场价: {exit_price:.6f}")
            print(f"  收益率: {pnl_pct:.2f}%")
            print(f"{'='*60}")

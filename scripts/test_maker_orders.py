#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maker订单模拟盘测试脚本
使用币安测试网进行完整的做空流程测试
"""
from __future__ import annotations

import io
import sys

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import sys
from pathlib import Path
import time
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 模拟执行器（不需要真实API）
class MockMakerOrderExecutor:
    """
    模拟Maker订单执行器
    用于测试逻辑，不实际连接交易所
    """

    def __init__(
        self,
        maker_offset_pct: float = 0.10,
        max_wait_seconds: int = 60,
        check_interval: float = 2.0,
        simulate_fill_rate: float = 0.8  # 模拟80%的成交率
    ):
        self.maker_offset_pct = maker_offset_pct
        self.max_wait_seconds = max_wait_seconds
        self.check_interval = check_interval
        self.simulate_fill_rate = simulate_fill_rate

        print(f"\n{'='*70}")
        print(f"  📊 模拟Maker订单执行器已初始化")
        print(f"{'='*70}")
        print(f"  Maker偏移: {maker_offset_pct}%")
        print(f"  最长等待: {max_wait_seconds}秒")
        print(f"  检查间隔: {check_interval}秒")
        print(f"  模拟成交率: {simulate_fill_rate*100:.0f}%")
        print(f"{'='*70}\n")

    def get_simulated_orderbook(self, symbol: str, current_price: float):
        """
        模拟获取订单簿
        """
        # 模拟0.1%的买卖价差
        spread_pct = 0.1
        bid = current_price * (1 - spread_pct / 200)
        ask = current_price * (1 + spread_pct / 200)

        return {
            'bid': bid,
            'ask': ask,
            'spread_pct': spread_pct
        }

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

        return round(price, 6)

    def place_maker_order(
        self,
        symbol: str,
        side: str,
        current_price: float,
        amount: float
    ) -> dict:
        """
        模拟下Maker限价单
        """
        # 获取模拟订单簿
        orderbook = self.get_simulated_orderbook(symbol, current_price)

        # 计算限价
        limit_price = self.calculate_maker_price(side, orderbook)

        # 确定订单方向
        order_side = 'sell' if side == 'short_entry' else 'buy'

        order_id = f"MOCK_{int(time.time() * 1000)}"

        print(f"\n{'─'*70}")
        print(f"  📝 限价单已下达 (模拟)")
        print(f"{'─'*70}")
        print(f"  交易对: {symbol}")
        print(f"  方向: {order_side.upper()} ({side})")
        print(f"  当前价: {current_price:.6f}")
        print(f"  限价: {limit_price:.6f}")
        print(f"  数量: {amount:.4f}")
        print(f"  订单ID: {order_id}")
        print(f"  盘口: Bid={orderbook['bid']:.6f}, Ask={orderbook['ask']:.6f}, Spread={orderbook['spread_pct']:.3f}%")
        print(f"{'─'*70}")

        return {
            'id': order_id,
            'symbol': symbol,
            'side': order_side,
            'price': limit_price,
            'amount': amount,
            'status': 'open',
            'timestamp': time.time()
        }

    def wait_for_fill(
        self,
        order: dict,
        simulate_success: bool = None
    ) -> tuple[bool, dict]:
        """
        模拟等待订单成交
        """
        if simulate_success is None:
            import random
            simulate_success = random.random() < self.simulate_fill_rate

        print(f"\n  ⏳ 等待订单成交...")

        # 模拟等待过程
        wait_intervals = 5
        for i in range(wait_intervals):
            elapsed = (i + 1) * (self.max_wait_seconds / wait_intervals)
            filled_pct = (i + 1) / wait_intervals * 100

            if simulate_success and i >= wait_intervals - 2:
                # 模拟成交
                order['status'] = 'closed'
                order['filled'] = order['amount']
                order['average'] = order['price']
                order['fee'] = -order['price'] * order['amount'] * 0.0002  # -0.02% Maker返佣

                print(f"  ✅ 订单完全成交!")
                print(f"  成交价: {order['average']:.6f}")
                print(f"  成交量: {order['filled']:.4f}")
                print(f"  费用: {order['fee']:.4f} USDT (Maker返佣)")
                print(f"  等待时间: {elapsed:.0f}秒")

                return True, order
            else:
                print(f"  ⏱️  等待中... {filled_pct:.0f}% ({elapsed:.0f}秒)")
                time.sleep(0.5)  # 实际等待0.5秒模拟

        if not simulate_success:
            # 模拟超时未成交
            print(f"  ⏰ 订单超时未成交 ({self.max_wait_seconds}秒)")
            print(f"  ❌ 订单已取消")
            order['status'] = 'canceled'
            return False, order

        return False, order


def simulate_short_entry(
    executor: MockMakerOrderExecutor,
    symbol: str,
    current_price: float,
    amount: float
) -> dict | None:
    """
    模拟做空入场
    """
    print(f"\n{'='*70}")
    print(f"  🔽 做空入场 (SHORT ENTRY)")
    print(f"{'='*70}")

    # 下限价卖单
    order = executor.place_maker_order(
        symbol=symbol,
        side='short_entry',
        current_price=current_price,
        amount=amount
    )

    # 等待成交
    is_filled, filled_order = executor.wait_for_fill(order)

    if is_filled:
        print(f"\n  ✅ 做空入场成功!")
        return filled_order
    else:
        print(f"\n  ❌ 做空入场失败!")
        return None


def simulate_short_exit(
    executor: MockMakerOrderExecutor,
    symbol: str,
    current_price: float,
    amount: float
) -> dict | None:
    """
    模拟做空出场
    """
    print(f"\n{'='*70}")
    print(f"  🔼 做空出场 (SHORT EXIT)")
    print(f"{'='*70}")

    # 下限价买单
    order = executor.place_maker_order(
        symbol=symbol,
        side='short_exit',
        current_price=current_price,
        amount=amount
    )

    # 等待成交
    is_filled, filled_order = executor.wait_for_fill(order)

    if is_filled:
        print(f"\n  ✅ 做空出场成功!")
        return filled_order
    else:
        print(f"\n  ❌ 做空出场失败!")
        return None


def run_single_trade_simulation():
    """
    运行单笔交易模拟
    """
    print(f"\n{'#'*70}")
    print(f"  🧪 测试场景1: 单笔做空交易模拟")
    print(f"{'#'*70}")

    # 创建模拟执行器
    executor = MockMakerOrderExecutor(
        maker_offset_pct=0.10,   # 0.10% 偏移
        max_wait_seconds=60,
        check_interval=2.0,
        simulate_fill_rate=0.8   # 80%成交率
    )

    # 模拟信号
    symbol = "DEXE/USDT:USDT"
    entry_price = 7.321
    amount = 1.0

    print(f"  标的: {symbol}")
    print(f"  入场价: {entry_price:.6f}")
    print(f"  数量: {amount:.4f}")

    # 执行入场
    entry_order = simulate_short_entry(
        executor=executor,
        symbol=symbol,
        current_price=entry_price,
        amount=amount
    )

    if not entry_order:
        print(f"\n  ❌ 入场失败，测试结束")
        return None

    # 模拟5天后价格下跌
    print(f"\n  ⏰ 模拟持仓5天...")
    time.sleep(1)

    exit_price = 6.500  # 下跌11.2%
    print(f"  5天后价格: {exit_price:.6f} (下跌 {(entry_price - exit_price) / entry_price * 100:.1f}%)")

    # 执行出场
    exit_order = simulate_short_exit(
        executor=executor,
        symbol=symbol,
        current_price=exit_price,
        amount=amount
    )

    if not exit_order:
        print(f"\n  ❌ 出场失败")
        return None

    # 计算盈亏
    entry_avg = entry_order['average']
    exit_avg = exit_order['average']
    entry_fee = entry_order['fee']
    exit_fee = exit_order['fee']

    # 做空收益 = (入场价 - 出场价) / 入场价
    pnl_pct = (entry_avg - exit_avg) / entry_avg
    pnl_usdt = (entry_avg - exit_avg) * amount
    total_fee = entry_fee + exit_fee
    net_pnl_usdt = pnl_usdt + total_fee  # 费用是负数（返佣是正）
    net_pnl_pct = net_pnl_usdt / (entry_avg * amount)

    print(f"\n{'='*70}")
    print(f"  💰 交易结果汇总")
    print(f"{'='*70}")
    print(f"  入场价: {entry_avg:.6f}")
    print(f"  出场价: {exit_avg:.6f}")
    print(f"  价格变动: {(exit_price - entry_price) / entry_price * 100:.2f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  做空收益: {pnl_pct * 100:.2f}% ({pnl_usdt:.2f} USDT)")
    print(f"  Maker返佣: {total_fee:.4f} USDT")
    print(f"  净收益: {net_pnl_pct * 100:.2f}% ({net_pnl_usdt:.2f} USDT)")
    print(f"{'='*70}\n")

    return {
        'symbol': symbol,
        'entry_price': entry_avg,
        'exit_price': exit_avg,
        'pnl_pct': pnl_pct,
        'pnl_usdt': pnl_usdt,
        'fee': total_fee,
        'net_pnl_pct': net_pnl_pct,
        'net_pnl_usdt': net_pnl_usdt
    }


def run_batch_simulation():
    """
    运行批量交易模拟（模拟今日Paper Trading信号）
    """
    print(f"\n{'#'*70}")
    print(f"  🧪 测试场景2: 批量信号模拟（读取Paper Trading信号）")
    print(f"{'#'*70}")

    # 读取最新的Paper Trading信号
    signals_dir = ROOT / "data" / "paper_trading"
    history_file = signals_dir / "signals_history.csv"

    if not history_file.exists():
        print(f"\n  ❌ 未找到信号历史文件: {history_file}")
        print(f"  请先运行Paper Trading生成信号")
        return

    # 读取信号
    signals = pd.read_csv(history_file)
    signals['signal_time'] = pd.to_datetime(signals['signal_time'])

    # 只取今天的信号
    today = datetime.now().date()
    today_signals = signals[signals['signal_time'].dt.date == today]

    if len(today_signals) == 0:
        print(f"\n  ⚠️  今天还没有信号，使用最近5个信号进行测试")
        today_signals = signals.tail(5)

    print(f"\n  找到 {len(today_signals)} 个信号")
    print(f"\n{'─'*70}")
    print(today_signals[['symbol', 'close', 'model_score', 'model_class', 'signal_type']].to_string(index=False))
    print(f"{'─'*70}\n")

    # 创建模拟执行器
    executor = MockMakerOrderExecutor(
        maker_offset_pct=0.10,
        max_wait_seconds=60,
        simulate_fill_rate=0.85  # 85%成交率
    )

    # 执行每个信号
    results = []

    for idx, signal in today_signals.iterrows():
        print(f"\n{'─'*70}")
        print(f"  信号 {len(results) + 1}/{len(today_signals)}")
        print(f"{'─'*70}")

        symbol = signal['symbol']
        entry_price = signal['close']
        amount = 1.0  # 固定数量

        # 执行入场
        entry_order = simulate_short_entry(
            executor=executor,
            symbol=symbol,
            current_price=entry_price,
            amount=amount
        )

        if not entry_order:
            print(f"  ⏭️  跳过此信号")
            continue

        # 模拟5天后价格（随机下跌0-15%）
        import random
        price_change_pct = random.uniform(-0.15, 0.05)  # -15%到+5%
        exit_price = entry_price * (1 + price_change_pct)

        # 执行出场
        exit_order = simulate_short_exit(
            executor=executor,
            symbol=symbol,
            current_price=exit_price,
            amount=amount
        )

        if not exit_order:
            print(f"  ⏭️  跳过此信号")
            continue

        # 计算盈亏
        entry_avg = entry_order['average']
        exit_avg = exit_order['average']
        pnl_pct = (entry_avg - exit_avg) / entry_avg
        pnl_usdt = (entry_avg - exit_avg) * amount
        total_fee = entry_order['fee'] + exit_order['fee']
        net_pnl_usdt = pnl_usdt + total_fee

        result = {
            'symbol': symbol,
            'entry_price': entry_avg,
            'exit_price': exit_avg,
            'pnl_pct': pnl_pct,
            'fee': total_fee,
            'net_pnl_usdt': net_pnl_usdt
        }
        results.append(result)

        print(f"  📊 收益: {pnl_pct * 100:.2f}%, 费用: {total_fee:.4f} USDT, 净收益: {net_pnl_usdt:.2f} USDT")

    # 汇总统计
    if len(results) > 0:
        results_df = pd.DataFrame(results)

        print(f"\n{'='*70}")
        print(f"  📈 批量交易统计")
        print(f"{'='*70}")
        print(f"  总交易数: {len(results)}")
        print(f"  成交率: {len(results) / len(today_signals) * 100:.0f}%")
        print(f"  ─────────────────────────────────────")
        print(f"  平均收益: {results_df['pnl_pct'].mean() * 100:.2f}%")
        print(f"  中位收益: {results_df['pnl_pct'].median() * 100:.2f}%")
        print(f"  胜率: {(results_df['pnl_pct'] > 0).sum() / len(results) * 100:.1f}%")
        print(f"  ─────────────────────────────────────")
        print(f"  总收益: {results_df['net_pnl_usdt'].sum():.2f} USDT")
        print(f"  总费用返佣: {results_df['fee'].sum():.4f} USDT")
        print(f"  ─────────────────────────────────────")
        print(f"  最佳交易: {results_df['pnl_pct'].max() * 100:.2f}%")
        print(f"  最差交易: {results_df['pnl_pct'].min() * 100:.2f}%")
        print(f"{'='*70}\n")


def run_fee_comparison():
    """
    运行费用对比测试
    """
    print(f"\n{'#'*70}")
    print(f"  🧪 测试场景3: Maker vs Taker 费用对比")
    print(f"{'#'*70}\n")

    # 测试参数
    entry_price = 7.321
    exit_price = 6.500
    amount = 1.0

    # Taker费用（市价单）
    taker_fee_rate = 0.0005  # 0.05%
    entry_fee_taker = entry_price * amount * taker_fee_rate
    exit_fee_taker = exit_price * amount * taker_fee_rate
    total_fee_taker = entry_fee_taker + exit_fee_taker

    # Maker费用（限价单）
    maker_fee_rate = -0.0002  # -0.02% 返佣
    entry_fee_maker = entry_price * amount * maker_fee_rate
    exit_fee_maker = exit_price * amount * maker_fee_rate
    total_fee_maker = entry_fee_maker + exit_fee_maker

    # 收益计算
    gross_pnl = (entry_price - exit_price) * amount
    net_pnl_taker = gross_pnl - total_fee_taker
    net_pnl_maker = gross_pnl - total_fee_maker

    print(f"  交易参数:")
    print(f"  ─────────────────────────────────────")
    print(f"  入场价: {entry_price:.6f}")
    print(f"  出场价: {exit_price:.6f}")
    print(f"  数量: {amount:.4f}")
    print(f"  毛收益: {gross_pnl:.2f} USDT ({gross_pnl / (entry_price * amount) * 100:.2f}%)")

    print(f"\n{'─'*70}")
    print(f"  💸 Taker (市价单) - 0.05% 费率")
    print(f"{'─'*70}")
    print(f"  入场费用: {entry_fee_taker:.4f} USDT")
    print(f"  出场费用: {exit_fee_taker:.4f} USDT")
    print(f"  总费用: {total_fee_taker:.4f} USDT")
    print(f"  净收益: {net_pnl_taker:.2f} USDT ({net_pnl_taker / (entry_price * amount) * 100:.2f}%)")

    print(f"\n{'─'*70}")
    print(f"  ✨ Maker (限价单) - 0.02% 返佣")
    print(f"{'─'*70}")
    print(f"  入场费用: {entry_fee_maker:.4f} USDT (返佣)")
    print(f"  出场费用: {exit_fee_maker:.4f} USDT (返佣)")
    print(f"  总费用: {total_fee_maker:.4f} USDT (返佣)")
    print(f"  净收益: {net_pnl_maker:.2f} USDT ({net_pnl_maker / (entry_price * amount) * 100:.2f}%)")

    print(f"\n{'='*70}")
    print(f"  📊 对比总结")
    print(f"{'='*70}")
    print(f"  费用差距: {total_fee_taker - total_fee_maker:.4f} USDT")
    print(f"  收益提升: {net_pnl_maker - net_pnl_taker:.2f} USDT")
    print(f"  收益率提升: {(net_pnl_maker - net_pnl_taker) / (entry_price * amount) * 100:.2f}%")
    print(f"{'='*70}\n")

    # 每月累计效果
    trades_per_day = 10
    days = 30
    total_trades = trades_per_day * days

    monthly_fee_taker = total_fee_taker * total_trades
    monthly_fee_maker = total_fee_maker * total_trades
    monthly_saving = monthly_fee_taker - monthly_fee_maker

    print(f"  📅 每月累计效果 (每天{trades_per_day}笔)")
    print(f"  ─────────────────────────────────────")
    print(f"  Taker总费用: {monthly_fee_taker:.2f} USDT")
    print(f"  Maker总返佣: {abs(monthly_fee_maker):.2f} USDT")
    print(f"  每月节省: {monthly_saving:.2f} USDT")
    print(f"{'='*70}\n")


def main():
    """
    主函数
    """
    print(f"\n{'█'*70}")
    print(f"  🧪 Maker订单策略 - 模拟盘测试")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*70}\n")

    # 场景1: 单笔交易模拟
    print(f"\n")
    result = run_single_trade_simulation()

    # 场景2: 批量信号模拟
    print(f"\n")
    run_batch_simulation()

    # 场景3: 费用对比
    print(f"\n")
    run_fee_comparison()

    print(f"\n{'█'*70}")
    print(f"  ✅ 所有测试完成!")
    print(f"{'█'*70}\n")


if __name__ == "__main__":
    main()

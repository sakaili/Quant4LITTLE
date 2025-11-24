#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘Maker订单执行 - 1%资金管理
读取Paper Trading信号，按1%可用资金下单
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
from datetime import datetime, date
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv未安装，将使用系统环境变量")

try:
    import ccxt
except ImportError:
    print("❌ 请先安装ccxt库: pip install ccxt")
    sys.exit(1)


class LiveMakerTrader:
    """
    实盘Maker交易执行器
    1%资金管理策略
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        use_testnet: bool = False,
        position_pct: float = 0.01,  # 每笔1%资金
        maker_offset_pct: float = 0.10,
        max_wait_seconds: int = 60,
        max_positions: int = 10  # 最多同时10个仓位
    ):
        self.position_pct = position_pct
        self.maker_offset_pct = maker_offset_pct
        self.max_wait_seconds = max_wait_seconds
        self.max_positions = max_positions
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
        print(f"  🤖 实盘Maker交易执行器")
        print(f"{'='*70}")
        print(f"  模式: {'🧪 测试网' if use_testnet else '⚠️  实盘'}")
        print(f"  资金管理: 每笔 {position_pct*100:.1f}% 可用资金")
        print(f"  最大持仓: {max_positions} 个")
        print(f"  Maker偏移: {maker_offset_pct}%")
        print(f"{'='*70}\n")

    def get_account_equity(self) -> float:
        """获取账户净值 (钱包余额)"""
        try:
            balance = self.exchange.fetch_balance()
            # 使用钱包余额而不是total（避免未实现盈亏影响）
            usdt_balance = balance.get('USDT', {})

            # 尝试获取钱包余额，如果没有则使用free
            wallet_balance = usdt_balance.get('free', 0) + usdt_balance.get('used', 0)

            return wallet_balance
        except Exception as e:
            print(f"  ❌ 获取账户净值失败: {e}")
            return 0

    def get_available_balance(self) -> float:
        """获取可用USDT余额"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_free = balance.get('USDT', {}).get('free', 0)
            return usdt_free
        except Exception as e:
            print(f"  ❌ 获取余额失败: {e}")
            return 0

    def get_current_positions(self) -> list:
        """获取当前持仓"""
        try:
            positions = self.exchange.fetch_positions()
            active = [p for p in positions if float(p.get('contracts', 0)) != 0]
            return active
        except Exception as e:
            print(f"  ❌ 获取持仓失败: {e}")
            return []

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        account_equity: float,
        leverage: float = 2.0
    ) -> float:
        """
        计算开仓数量

        Args:
            symbol: 交易对
            current_price: 当前价格
            account_equity: 账户净值
            leverage: 杠杆倍数

        Returns:
            开仓数量
        """
        # 账户净值的1% × 杠杆
        position_value = account_equity * self.position_pct * leverage

        # 转换为币数量
        amount = position_value / current_price

        # 格式化精度
        try:
            market = self.exchange.market(symbol)
            precision = market['precision']['amount']

            if isinstance(precision, int):
                amount = round(amount, precision)
            else:
                amount = round(amount / precision) * precision

        except:
            amount = round(amount, 3)

        return amount

    def get_orderbook(self, symbol: str) -> dict | None:
        """获取订单簿"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol)
            bid = orderbook['bids'][0][0] if orderbook['bids'] else None
            ask = orderbook['asks'][0][0] if orderbook['asks'] else None

            if bid and ask:
                return {'bid': bid, 'ask': ask, 'spread_pct': (ask-bid)/bid*100}
            return None
        except Exception as e:
            print(f"  ❌ 获取订单簿失败: {e}")
            return None

    def place_short_entry(
        self,
        symbol: str,
        amount: float
    ) -> dict | None:
        """
        做空入场
        """
        # 确保数量为正数
        amount = abs(amount)

        if amount <= 0:
            print(f"  ❌ 开仓数量无效: {amount}")
            return None

        orderbook = self.get_orderbook(symbol)
        if not orderbook:
            return None

        # 计算Maker价格（在ask之上）
        limit_price = orderbook['ask'] * (1 + self.maker_offset_pct / 100)

        # 格式化价格
        try:
            market = self.exchange.market(symbol)
            price_precision = market['precision']['price']
            if isinstance(price_precision, int):
                limit_price = round(limit_price, price_precision)
            else:
                limit_price = round(limit_price / price_precision) * price_precision
        except:
            limit_price = round(limit_price, 2)

        try:
            print(f"  📝 做空入场: {symbol}")
            print(f"    限价: {limit_price:.4f}")
            print(f"    数量: {amount:.4f}")
            print(f"    盘口: Bid={orderbook['bid']:.4f}, Ask={orderbook['ask']:.4f}")

            order = self.exchange.create_limit_order(
                symbol=symbol,
                side='sell',
                amount=amount,
                price=limit_price,
                params={
                    'positionSide': 'SHORT'  # 指定为做空方向（双向持仓模式）
                }
            )

            print(f"  ✅ 订单已下达! ID: {order['id']}")
            return order

        except Exception as e:
            print(f"  ❌ 下单失败: {e}")
            return None

    def set_tp_sl_orders(
        self,
        symbol: str,
        amount: float,
        entry_price: float,
        take_profit_pct: float = 30.0,
        stop_loss_pct: float = 200.0
    ) -> tuple[dict | None, dict | None]:
        """
        设置止盈止损委托单

        对于做空仓位:
        - 止盈价格 = 入场价 × (1 - 止盈%)
        - 止损价格 = 入场价 × (1 + 止损%)
        """
        try:
            # 计算止盈止损价格
            tp_price = entry_price * (1 - take_profit_pct / 100)  # 做空止盈=价格下跌
            sl_price = entry_price * (1 + stop_loss_pct / 100)    # 做空止损=价格上涨

            # 格式化价格精度
            try:
                market = self.exchange.market(symbol)
                price_precision = market['precision']['price']
                if isinstance(price_precision, int):
                    tp_price = round(tp_price, price_precision)
                    sl_price = round(sl_price, price_precision)
                else:
                    tp_price = round(tp_price / price_precision) * price_precision
                    sl_price = round(sl_price / price_precision) * price_precision
            except:
                tp_price = round(tp_price, 2)
                sl_price = round(sl_price, 2)

            print(f"  📝 设置止盈止损:")
            print(f"    入场价: {entry_price:.4f}")
            print(f"    止盈价: {tp_price:.4f} (-{take_profit_pct}%)")
            print(f"    止损价: {sl_price:.4f} (+{stop_loss_pct}%)")

            # 下止盈单（做空止盈=买入平仓）
            tp_order = self.exchange.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side='buy',
                amount=abs(amount),
                params={
                    'stopPrice': tp_price,
                    'positionSide': 'SHORT',
                    'workingType': 'MARK_PRICE'
                }
            )
            print(f"    ✅ 止盈单已设置! ID: {tp_order['id']}")

            # 下止损单（做空止损=买入平仓）
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side='buy',
                amount=abs(amount),
                params={
                    'stopPrice': sl_price,
                    'positionSide': 'SHORT',
                    'workingType': 'MARK_PRICE'
                }
            )
            print(f"    ✅ 止损单已设置! ID: {sl_order['id']}")

            return tp_order, sl_order

        except Exception as e:
            print(f"    ❌ 设置止盈止损失败: {e}")
            return None, None

    def wait_for_fill(self, symbol: str, order_id: str) -> tuple[bool, dict | None]:
        """等待订单成交"""
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > self.max_wait_seconds:
                print(f"  ⏰ 超时 ({self.max_wait_seconds}秒)")
                try:
                    self.exchange.cancel_order(order_id, symbol)
                    print(f"  ❌ 订单已取消")
                except:
                    pass
                return False, None

            try:
                order = self.exchange.fetch_order(order_id, symbol)
                status = order['status']

                if status == 'closed':
                    print(f"  ✅ 成交! 价格: {order.get('average', 0):.4f}")
                    return True, order
                elif status == 'canceled':
                    print(f"  ❌ 已取消")
                    return False, None
                elif status == 'open':
                    filled_pct = (order.get('filled', 0) / order.get('amount', 1)) * 100
                    print(f"  ⏱️  等待... {filled_pct:.0f}% ({elapsed:.0f}s)", end='\r')
                    time.sleep(2)

            except Exception as e:
                print(f"  ⚠️  查询失败: {e}")
                time.sleep(2)


def load_today_signals() -> pd.DataFrame:
    """加载今日Paper Trading信号"""
    signals_dir = ROOT / "data" / "paper_trading"
    today = date.today().strftime('%Y%m%d')
    today_file = signals_dir / f"signals_{today}.csv"

    if today_file.exists():
        signals = pd.read_csv(today_file)
        print(f"  ✅ 加载今日信号: {len(signals)} 个")
        return signals
    else:
        print(f"  ⚠️  未找到今日信号文件: {today_file}")
        print(f"  提示: 请先运行 python scripts/hourly_trading_enhanced.py 生成信号")
        return pd.DataFrame()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="实盘Maker交易执行")
    parser.add_argument("--auto-confirm", action="store_true", help="自动确认，跳过yes输入")
    args = parser.parse_args()

    print(f"\n{'█'*70}")
    print(f"  🤖 实盘Maker交易执行 - 2%资金管理 × 2倍杠杆")
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
        print("     $env:USE_TESTNET = 'False'  # False=实盘, True=测试网")
        return

    # 创建交易器
    trader = LiveMakerTrader(
        api_key=api_key,
        api_secret=api_secret,
        use_testnet=use_testnet,
        position_pct=0.02,      # 2%资金
        maker_offset_pct=0.10,  # 0.10%偏移
        max_wait_seconds=60,
        max_positions=10
    )

    # 获取账户信息
    print(f"\n{'─'*70}")
    print(f"  💰 账户信息")
    print(f"{'─'*70}")

    # 使用可用余额（更安全）
    available_balance = trader.get_available_balance()
    if available_balance == 0:
        print(f"  ❌ 无法获取余额或余额为0")
        return

    account_equity = trader.get_account_equity()
    leverage = 2.0  # 2倍杠杆

    print(f"  账户净值: {account_equity:.2f} USDT")
    print(f"  可用余额: {available_balance:.2f} USDT")
    print(f"  杠杆倍数: {leverage}x")
    # 使用可用余额而不是净值
    print(f"  每笔金额: {available_balance * 0.02 * leverage:.2f} USDT (余额2% × {leverage}x)")

    # 检查当前持仓
    current_positions = trader.get_current_positions()
    print(f"  当前持仓: {len(current_positions)} 个")

    if len(current_positions) >= trader.max_positions:
        print(f"  ⚠️  已达最大持仓数量 ({trader.max_positions})，跳过新开仓")
        return

    # 加载今日信号
    print(f"\n{'─'*70}")
    print(f"  📊 加载交易信号")
    print(f"{'─'*70}")

    signals = load_today_signals()
    if len(signals) == 0:
        print(f"  ❌ 无可用信号")
        return

    print(f"\n  前5个信号:")
    print(signals[['symbol', 'close', 'model_score', 'model_class', 'signal_type']].head().to_string(index=False))

    # 确认执行
    remaining_slots = trader.max_positions - len(current_positions)
    signals_to_execute = min(len(signals), remaining_slots)

    print(f"\n{'='*70}")
    print(f"  ⚠️  确认信息")
    print(f"{'='*70}")
    print(f"  模式: {'测试网 (虚拟资金)' if use_testnet else '⚠️ 实盘 (真实资金)'}")
    print(f"  可用余额: {available_balance:.2f} USDT")
    print(f"  杠杆倍数: {leverage}x")
    print(f"  每笔金额: {available_balance * 0.02 * leverage:.2f} USDT (余额2% × {leverage}x)")
    print(f"  计划开仓: {signals_to_execute} 个")
    print(f"  可用槽位: {remaining_slots} 个")
    print(f"{'='*70}")

    if not args.auto_confirm:
        confirm = input(f"\n  是否继续? (输入 'yes' 确认): ")
        if confirm.lower() != 'yes':
            print(f"\n  ❌ 用户取消")
            return
    else:
        print(f"\n  ✅ 自动确认模式，跳过手动确认")

    # 执行交易
    print(f"\n{'='*70}")
    print(f"  🚀 开始执行交易")
    print(f"{'='*70}\n")

    success_count = 0
    failed_count = 0

    for i, (idx, signal) in enumerate(signals.head(signals_to_execute).iterrows()):
        print(f"\n{'─'*70}")
        print(f"  信号 {i+1}/{signals_to_execute}")
        print(f"{'─'*70}")

        # 转换交易对名称: DEXE_USDT_USDT_1d -> DEXEUSDT
        raw_symbol = signal['symbol']
        if '_USDT_USDT_' in raw_symbol:
            # 格式: XXX_USDT_USDT_1d -> XXXUSDT
            symbol = raw_symbol.split('_USDT_USDT_')[0] + 'USDT'
        elif raw_symbol.endswith('_1d') or raw_symbol.endswith('_1h'):
            # 格式: XXXUSDT_1d -> XXXUSDT
            symbol = raw_symbol.rsplit('_', 1)[0]
        else:
            symbol = raw_symbol

        # 获取实时价格（从订单簿）
        orderbook = trader.get_orderbook(symbol)
        if not orderbook:
            print(f"  ❌ 无法获取{symbol}实时价格，跳过")
            failed_count += 1
            continue

        # 使用实时价格计算开仓数量
        current_price = orderbook['ask']  # 使用卖一价

        # 计算开仓数量 (使用可用余额 × 杠杆 × 实时价格)
        amount = trader.calculate_position_size(
            symbol=symbol,
            current_price=current_price,
            account_equity=available_balance,  # 使用可用余额
            leverage=leverage
        )

        position_value = amount * current_price

        print(f"  标的: {raw_symbol}")
        print(f"  交易对: {symbol}")
        print(f"  信号价: {signal['close']:.4f} (历史)")
        print(f"  实时价: {current_price:.4f} (当前)")
        print(f"  开仓数量: {amount:.4f}")
        print(f"  开仓金额: {position_value:.2f} USDT")
        print(f"  杠杆: {leverage}x")
        print(f"  模型分类: Class {signal.get('model_class', 'N/A')}")

        # 下单
        order = trader.place_short_entry(symbol=symbol, amount=amount)

        if not order:
            print(f"  ❌ 下单失败，跳过")
            failed_count += 1
            continue

        # 等待成交
        is_filled, filled_order = trader.wait_for_fill(
            symbol=symbol,
            order_id=order['id']
        )

        if is_filled:
            success_count += 1
            print(f"  ✅ 做空入场成功!")

            # 获取成交价格
            entry_price = filled_order.get('average', current_price)

            # 立即设置止盈止损委托单
            tp_order, sl_order = trader.set_tp_sl_orders(
                symbol=symbol,
                amount=amount,
                entry_price=entry_price,
                take_profit_pct=30.0,   # 30%止盈
                stop_loss_pct=200.0     # 200%止损
            )

            if tp_order and sl_order:
                print(f"  ✅ 止盈止损已自动设置!")
            else:
                print(f"  ⚠️  止盈止损设置失败，请手动设置!")

        else:
            failed_count += 1
            print(f"  ❌ 订单未成交")

        # 避免频繁请求
        time.sleep(1)

    # 汇总
    print(f"\n{'='*70}")
    print(f"  📊 执行结果")
    print(f"{'='*70}")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {failed_count} 个")
    print(f"  成交率: {success_count / signals_to_execute * 100:.1f}%")

    if success_count > 0:
        total_used = available_balance * 0.02 * leverage * success_count
        print(f"  使用资金: {total_used:.2f} USDT (余额{success_count*2}% × {leverage}x)")

    print(f"{'='*70}\n")

    # 显示当前持仓
    print(f"{'─'*70}")
    print(f"  💼 当前持仓")
    print(f"{'─'*70}")

    positions = trader.get_current_positions()
    if len(positions) > 0:
        for pos in positions:
            symbol = pos['symbol']
            contracts = float(pos.get('contracts', 0))
            entry_price = float(pos.get('entryPrice', 0))
            mark_price = float(pos.get('markPrice', 0))
            unrealized_pnl = float(pos.get('unrealizedPnl', 0))

            if contracts < 0:  # 空头
                print(f"  📉 {symbol}")
                print(f"     数量: {abs(contracts):.4f}")
                print(f"     入场价: {entry_price:.4f}")
                print(f"     标记价: {mark_price:.4f}")
                print(f"     浮动盈亏: {unrealized_pnl:+.2f} USDT")
    else:
        print(f"  无持仓")

    print(f"\n{'='*70}")
    print(f"  ✅ 执行完成!")
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

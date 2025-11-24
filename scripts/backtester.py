#!/usr/bin/env python3
"""
Module 3: Event-driven backtester for the Binance “air coin” short strategy.

Key features:
    * Daily scan of the lowest-liquidity perpetuals (static pool derived from
      the current 24h ticker snapshot).
    * Entries occur on the next session's open after EMA10 < EMA20 < EMA30 signal.
    * Risk controls: 30% take-profit, ATR-based stop (entry + 3*ATR14),
      and EMA20 technical stop (close > EMA20 -> exit next open).
    * Position sizing: net worth * per_trade_pct * leverage (e.g. 1% * 2x).
    * Transaction costs: 5 bps per side + 10 bps adverse slippage.
    * Metrics: final equity, Sharpe ratio, max drawdown, win rate, trade log,
      and an equity curve plot.

Usage:
    python backtester.py --start 2022-01-01 --end 2025-01-01
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Ensure project root on sys.path when run as a script.
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas_ta as ta
import pandas as pd

from scripts.data_fetcher import BinanceDataFetcher
from scripts.daily_candidate_scan import (
    filter_out_majors,
    pick_air_coin_pool,
)

logger = logging.getLogger("backtester")

FEE_RATE = 0.0005
SLIPPAGE = 0.001
TAKE_PROFIT_PCT = 0.50  # 止盈 50%
ATR_STOP_MULTIPLIER = 3.0
MAX_POSITIONS = 30  # 仍保留最大并行仓位上限
INITIAL_EQUITY = 400.0  # 初始资金改为 400 USDT
PER_TRADE_PCT = 0.01  # 每单占用净值 1% 作为保证金
LEVERAGE = 2.0  # 2 倍杠杆开仓
FUNDING_RATE_FLOOR = -0.01
BOTTOM_N_DEFAULT = 100  # 成交额/市值倒数 100 个


@dataclass
class EntrySignal:
    symbol: str
    signal_date: date
    entry_date: date
    atr14: float


@dataclass
class ShortPosition:
    symbol: str
    entry_date: date
    entry_price: float
    qty: float
    atr_stop: float
    take_profit: float
    entry_atr: float
    entry_fee: float
    exit_next_open_reason: Optional[str] = None

    def unrealized_pnl(self, mark_price: float) -> float:
        return (self.entry_price - mark_price) * self.qty


@dataclass
class ClosedTrade:
    symbol: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: List[ClosedTrade]
    summary: Dict[str, float]
    trades_path: Path
    equity_plot_path: Path


class ShortAirCoinBacktester:
    def __init__(
        self,
        *,
        start: date,
        end: date,
        max_positions: int = MAX_POSITIONS,
        initial_equity: float = INITIAL_EQUITY,
        fee_rate: float = FEE_RATE,
        slippage: float = SLIPPAGE,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        atr_stop_multiplier: float = ATR_STOP_MULTIPLIER,
        per_trade_pct: float = PER_TRADE_PCT,
        leverage: float = LEVERAGE,
        bottom_n: int = BOTTOM_N_DEFAULT,
        timeframe: str = "1d",
        data_dir: Path = Path("data"),
        fetcher: Optional[BinanceDataFetcher] = None,
    ) -> None:
        self.start = start
        self.end = end
        self.max_positions = max_positions
        self.initial_equity = initial_equity
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.take_profit_pct = take_profit_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        self.per_trade_pct = per_trade_pct
        self.leverage = leverage
        self.bottom_n = bottom_n
        self.timeframe = timeframe
        self.data_dir = data_dir
        self.fetcher = fetcher or BinanceDataFetcher()

        self.positions: Dict[str, ShortPosition] = {}
        self.pending_entries: List[EntrySignal] = []
        self.trades: List[ClosedTrade] = []
        self.realized_pnl = 0.0

        self.histories: Dict[str, pd.DataFrame] = {}
        self.hourly_histories: Dict[str, pd.DataFrame] = {}
        self.calendar: List[date] = []
        self.air_coin_pool: List[str] = []
        self.funding_rates: Dict[str, Optional[float]] = {}
        self.pool_schedule: List[Tuple[date, List[str]]] = []

    # ------------------------------------------------------------------
    def _symbol_to_filename(self, symbol: str) -> str:
        """Sanitize symbol for filesystem use."""
        return symbol.replace("/", "_").replace(":", "_")

    # Data preparation
    # ------------------------------------------------------------------
    def prepare(self) -> None:
        metas = self.fetcher.fetch_usdt_perp_symbols()
        meta_map = {meta.symbol: meta for meta in metas}
        pools_dir = self.data_dir / "monthly_pools"
        pools_dir.mkdir(parents=True, exist_ok=True)

        # 每个月首日滚动获取成交额/市值倒数的池子
        pool_union: Set[str] = set()
        checkpoint = date(self.start.year, self.start.month, 1)
        while checkpoint <= self.end:
            tickers = self.fetcher.fetch_24h_tickers([meta.symbol for meta in metas])
            tickers = filter_out_majors(tickers, meta_map)
            pool = pick_air_coin_pool(tickers, bottom_n=self.bottom_n)
            if not pool:
                checkpoint = self._next_month(checkpoint)
                continue
            filtered_pool: List[str] = []
            for symbol in sorted(pool):
                rate = self._fetch_funding_rate(symbol)
                self.funding_rates[symbol] = rate
                if rate is not None and rate < FUNDING_RATE_FLOOR:
                    continue
                filtered_pool.append(symbol)
            if filtered_pool:
                self.pool_schedule.append((checkpoint, filtered_pool))
                pool_union.update(filtered_pool)
                snapshot = tickers[tickers["symbol"].isin(filtered_pool)].copy()
                snapshot["funding_rate"] = snapshot["symbol"].map(
                    lambda s: self.funding_rates.get(s)
                )
                snapshot.to_csv(
                    pools_dir / f"pool_{checkpoint.isoformat()}.csv", index=False
                )
            checkpoint = self._next_month(checkpoint)

        if not self.pool_schedule:
            raise RuntimeError("无法构建任何月度候选池")

        logger.info("Backtest pool size (union): %s symbols", len(pool_union))
        start_buffer = self.start - timedelta(days=120)
        histories = self.fetcher.fetch_bulk_history(
            pool_union,
            start=datetime.combine(start_buffer, datetime.min.time(), tzinfo=timezone.utc),
            end=datetime.combine(self.end, datetime.min.time(), tzinfo=timezone.utc),
            timeframe=self.timeframe,
        )
        hourly_histories = self.fetcher.fetch_bulk_history(
            pool_union,
            start=datetime.combine(start_buffer, datetime.min.time(), tzinfo=timezone.utc),
            end=datetime.combine(self.end, datetime.min.time(), tzinfo=timezone.utc),
            timeframe="1h",
        )
        # 落盘日线/小时线，便于审计/复现（近似：使用当前可获取的历史）
        daily_dir = self.data_dir / "daily_klines"
        hourly_dir = self.data_dir / "hourly_klines"
        daily_dir.mkdir(parents=True, exist_ok=True)
        hourly_dir.mkdir(parents=True, exist_ok=True)
        calendar: Set[date] = set()
        for symbol, frame in histories.items():
            if frame.empty:
                continue
            frame = frame.copy()
            frame["ema10"] = ta.ema(frame["close"], length=10)
            frame["ema20"] = ta.ema(frame["close"], length=20)
            frame["ema30"] = ta.ema(frame["close"], length=30)
            mask = (frame["timestamp"].dt.date >= self.start) & (
                frame["timestamp"].dt.date <= self.end
            )
            trimmed = frame.loc[mask].copy()
            if trimmed.empty:
                continue
            trimmed["date"] = trimmed["timestamp"].dt.date
            trimmed.set_index("date", inplace=True)
            self.histories[symbol] = trimmed
            calendar.update(trimmed.index.unique())
            safe_name = self._symbol_to_filename(symbol)
            trimmed.to_csv(daily_dir / f"{safe_name}_1d.csv", index=False)
            if symbol in hourly_histories:
                hframe = hourly_histories[symbol]
                hframe["date"] = hframe["timestamp"].dt.date
                self.hourly_histories[symbol] = hframe
                hframe.to_csv(hourly_dir / f"{safe_name}_1h.csv", index=False)
        if not self.histories:
            raise RuntimeError("无可用历史数据，检查网络或时间范围。")
        self.calendar = sorted(calendar)
        logger.info("Calendar size: %s days", len(self.calendar))

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        equity_records = []
        for current_date in self.calendar:
            day_rows = self._collect_day_rows(current_date)
            if not day_rows:
                continue
            self._process_exit_next_open(current_date, day_rows)
            self._process_intraday_exits(current_date, day_rows)
            self._enter_new_positions(current_date, day_rows)
            self._schedule_future_entries(current_date, day_rows)
            equity = self._mark_to_market(day_rows)
            equity_records.append({"date": current_date, "equity": equity})
        equity_df = pd.DataFrame(equity_records)
        trades_path = self.data_dir / "backtest_trades.csv"
        equity_plot_path = self.data_dir / "equity_curve.png"
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        if self.trades:
            pd.DataFrame([t.__dict__ for t in self.trades]).to_csv(
                trades_path, index=False
            )
        self._plot_equity_curve(equity_df, equity_plot_path)
        summary = self._compute_summary(equity_df)
        return BacktestResult(
            equity_curve=equity_df,
            trades=self.trades,
            summary=summary,
            trades_path=trades_path,
            equity_plot_path=equity_plot_path,
        )

    # ------------------------------------------------------------------
    def _collect_day_rows(self, current_date: date) -> Dict[str, pd.Series]:
        rows = {}
        active = self._active_pool_symbols(current_date)
        for symbol in active:
            frame = self.histories.get(symbol)
            if frame is not None and current_date in frame.index:
                rows[symbol] = frame.loc[current_date]
        return rows

    def _process_exit_next_open(
        self, current_date: date, day_rows: Dict[str, pd.Series]
    ) -> None:
        exits = []
        for symbol, position in list(self.positions.items()):
            if position.exit_next_open_reason and symbol in day_rows:
                row = day_rows[symbol]
                fill_price = float(row["open"]) * (1 + self.slippage)
                self._close_position(
                    symbol,
                    position,
                    exit_date=current_date,
                    exit_price=fill_price,
                    reason=position.exit_next_open_reason,
                )
                exits.append(symbol)
        for symbol in exits:
            self.positions.pop(symbol, None)

    def _process_intraday_exits(
        self, current_date: date, day_rows: Dict[str, pd.Series]
    ) -> None:
        exits = []
        for symbol, position in list(self.positions.items()):
            row = day_rows.get(symbol)
            if row is None:
                continue
            high = float(row["high"])
            low = float(row["low"])
            open_price = float(row["open"])
            exit_reason = None
            exit_price = None

            if open_price >= position.atr_stop:
                exit_reason = "hard_stop_gap"
                exit_price = open_price * (1 + self.slippage)
            elif open_price <= position.take_profit:
                exit_reason = "take_profit_gap"
                exit_price = open_price * (1 + self.slippage)
            elif high >= position.atr_stop:
                exit_reason = "hard_stop"
                exit_price = position.atr_stop * (1 + self.slippage)
            elif low <= position.take_profit:
                exit_reason = "take_profit"
                exit_price = position.take_profit * (1 + self.slippage)

            if exit_reason and exit_price is not None:
                self._close_position(
                    symbol,
                    position,
                    exit_date=current_date,
                    exit_price=exit_price,
                    reason=exit_reason,
                )
                exits.append(symbol)
        for symbol in exits:
            self.positions.pop(symbol, None)

    def _enter_new_positions(
        self, current_date: date, day_rows: Dict[str, pd.Series]
    ) -> None:
        active = self._active_pool_symbols(current_date)
        entries_today = [e for e in self.pending_entries if e.entry_date == current_date]
        if not entries_today:
            return
        pending_after = [e for e in self.pending_entries if e.entry_date != current_date]
        self.pending_entries = pending_after
        for signal in entries_today:
            if signal.symbol not in active:
                continue
            if signal.symbol not in day_rows:
                continue
            if signal.symbol in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                continue
            row = day_rows[signal.symbol]
            open_price = float(row["open"])
            fill_price = open_price * (1 - self.slippage)
            if math.isnan(fill_price) or math.isnan(signal.atr14):
                continue
            position_notional = self._current_equity() * self.per_trade_pct * self.leverage
            if position_notional <= 0:
                continue
            qty = position_notional / fill_price
            # 止损：价格上涨 150%（亏损 150%）强制平仓；止盈 50%
            atr_stop = fill_price * 2.5
            take_profit = fill_price * (1 - self.take_profit_pct)
            entry_fee = position_notional * self.fee_rate
            self.positions[signal.symbol] = ShortPosition(
                symbol=signal.symbol,
                entry_date=current_date,
                entry_price=fill_price,
                qty=qty,
                atr_stop=atr_stop,
                take_profit=take_profit,
                entry_atr=signal.atr14,
                entry_fee=entry_fee,
            )

    def _schedule_future_entries(
        self, current_date: date, day_rows: Dict[str, pd.Series]
    ) -> None:
        next_date = self._next_trading_date(current_date)
        if next_date is None:
            return
        active_next = self._active_pool_symbols(next_date)
        queued_symbols = {e.symbol for e in self.pending_entries}
        for symbol, row in day_rows.items():
            if symbol not in active_next:
                continue
            if symbol in self.positions or symbol in queued_symbols:
                continue
            funding = self.funding_rates.get(symbol)
            if funding is not None and funding < FUNDING_RATE_FLOOR:
                continue
            ema10 = float(row.get("ema10", float("nan")))
            ema20 = float(row.get("ema20", float("nan")))
            ema30 = float(row.get("ema30", float("nan")))
            atr14 = float(row["atr14"])
            if any(math.isnan(val) for val in (ema10, ema20, ema30, atr14)):
                continue
            if not (ema10 < ema20 < ema30):
                continue
            if not self._hourly_kdj_j_ok(symbol, current_date, threshold=80.0):
                continue
            self.pending_entries.append(
                EntrySignal(
                    symbol=symbol,
                    signal_date=current_date,
                    entry_date=next_date,
                    atr14=atr14,
                )
            )

    def _atr_spike_ok(self, symbol: str, current_date: date) -> bool:
        history = self.histories[symbol]
        if current_date not in history.index:
            return False
        idx = history.index.get_loc(current_date)
        if isinstance(idx, slice):
            idx = idx.start
        if idx < 3:
            return True
        atr_series = history["atr14"].iloc[: idx + 1].dropna()
        if len(atr_series) < 4:
            return True
        recent = atr_series.iloc[-1]
        prev_mean = atr_series.iloc[-4:-1].mean()
        if prev_mean <= 0:
            return True
        return recent <= 3 * prev_mean

    def _close_position(
        self,
        symbol: str,
        position: ShortPosition,
        *,
        exit_date: date,
        exit_price: float,
        reason: str,
    ) -> None:
        qty = position.qty
        exit_notional = exit_price * qty
        exit_fee = exit_notional * self.fee_rate
        pnl = (position.entry_price - exit_price) * qty - exit_fee - position.entry_fee
        self.realized_pnl += pnl
        return_pct = pnl / (position.entry_price * qty)
        self.trades.append(
            ClosedTrade(
                symbol=symbol,
                entry_date=position.entry_date,
                exit_date=exit_date,
                entry_price=position.entry_price,
                exit_price=exit_price,
                qty=qty,
                pnl=pnl,
                return_pct=return_pct,
                exit_reason=reason,
            )
        )

    def _mark_to_market(self, day_rows: Dict[str, pd.Series]) -> float:
        unrealized = 0.0
        for symbol, position in self.positions.items():
            row = day_rows.get(symbol)
            if row is None:
                continue
            close_price = float(row["close"])
            unrealized += position.unrealized_pnl(close_price)
        return self.initial_equity + self.realized_pnl + unrealized

    def _current_equity(self) -> float:
        return self.initial_equity + self.realized_pnl

    def _next_trading_date(self, current_date: date) -> Optional[date]:
        idx = self.calendar.index(current_date)
        if idx + 1 < len(self.calendar):
            return self.calendar[idx + 1]
        return None

    def _next_month(self, current: date) -> date:
        if current.month == 12:
            return date(current.year + 1, 1, 1)
        return date(current.year, current.month + 1, 1)

    def _active_pool_symbols(self, current_date: date) -> Set[str]:
        active: Optional[List[str]] = None
        for checkpoint, pool in self.pool_schedule:
            if checkpoint <= current_date:
                active = pool
            else:
                break
        return set(active or [])

    def _plot_equity_curve(self, equity_df: pd.DataFrame, output_path: Path) -> None:
        if equity_df.empty:
            return
        plt.figure(figsize=(10, 5))
        plt.plot(equity_df["date"], equity_df["equity"], label="Equity")
        plt.title("Air Coin Short Strategy Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity (USDT)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    def _compute_summary(self, equity_df: pd.DataFrame) -> Dict[str, float]:
        if equity_df.empty:
            return {}
        returns = equity_df["equity"].pct_change().dropna()
        if len(returns) > 1:
            sharpe = (
                (returns.mean() / returns.std()) * math.sqrt(252)
                if returns.std() != 0
                else 0.0
            )
        else:
            sharpe = 0.0
        running_max = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] / running_max) - 1.0
        max_dd = drawdown.min()
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl > 0)
        win_rate = wins / total_trades if total_trades else 0.0
        return {
            "final_equity": equity_df["equity"].iloc[-1],
            "total_return_pct": equity_df["equity"].iloc[-1] / self.initial_equity - 1,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_trades": total_trades,
        }

    def _fetch_funding_rate(self, symbol: str) -> Optional[float]:
        try:
            rate = self.fetcher.exchange.fetch_funding_rate(symbol)
        except Exception as exc:  # pragma: no cover - network call
            logger.warning("Funding rate fetch failed for %s: %s", symbol, exc)
            return None
        return rate.get("fundingRate")

    def _hourly_kdj_j_ok(self, symbol: str, current_date: date, threshold: float) -> bool:
        """Check if any 1h KDJ J on the given date exceeds threshold."""
        hframe = self.hourly_histories.get(symbol)
        if hframe is None or hframe.empty:
            return False
        daily_mask = hframe["date"] == current_date
        subset = hframe.loc[daily_mask]
        if subset.empty:
            return False
        stoch = ta.stoch(
            high=subset["high"],
            low=subset["low"],
            close=subset["close"],
            k=9,
            d=3,
            smooth_k=3,
        )
        if stoch is None or stoch.empty:
            return False
        k = stoch.iloc[:, 0]
        d = stoch.iloc[:, 1]
        j = 3 * k - 2 * d
        if j.dropna().empty:
            return False
        return j.dropna().max() > threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the Binance air-coins short strategy."
    )
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    parser.add_argument("--bottom-n", type=int, default=BOTTOM_N_DEFAULT)
    parser.add_argument("--initial-equity", type=float, default=INITIAL_EQUITY)
    parser.add_argument(
        "--per-trade-pct",
        type=float,
        default=PER_TRADE_PCT,
        help="保证金占当前净值的比例（例如 0.01 表示 1%）。",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=LEVERAGE,
        help="开仓杠杆倍数（例如 2 表示 2 倍杠杆）。",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    backtester = ShortAirCoinBacktester(
        start=start,
        end=end,
        max_positions=args.max_positions,
        bottom_n=args.bottom_n,
        initial_equity=args.initial_equity,
        per_trade_pct=args.per_trade_pct,
        leverage=args.leverage,
    )
    backtester.prepare()
    result = backtester.run()
    print("Backtest summary:")
    for key, value in result.summary.items():
        print(f"- {key}: {value:.4f}")
    print(f"Trades saved to: {result.trades_path}")
    print(f"Equity curve saved to: {result.equity_plot_path}")


if __name__ == "__main__":
    main()

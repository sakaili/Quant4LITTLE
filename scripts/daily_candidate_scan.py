#!/usr/bin/env python3
"""
Module 2: Daily scanner that identifies the weakest Binance USDT perpetuals.

筛选步骤（必须按顺序执行）：
1. 过滤主流币：直接剔除 BTC/ETH/BNB/SOL/XRP 等大市值资产。
2. 流动性 + “伪市值”过滤：先找出 24h 成交额最低的 N 个，再与
   24h ticker 里提供的市值倒数 N 个求交集 -> “空气币池”。
3. 入场信号：候选池中，最新日线满足 EMA20 < EMA30 的标的。
4. 风险控制：资金费率不低于 -1%，且 ATR14 没有在最近 3 天暴涨
   （今天 ATR > 3 * 前 3 天均值）。

脚本会生成 data/daily_scans/candidates_YYYYMMDD.csv，供回测/实时交易使用。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

# Ensure project root is on PYTHONPATH when invoked as a script.
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pandas_ta as ta

from scripts.data_fetcher import BinanceDataFetcher, SymbolMetadata

logger = logging.getLogger("daily_scan")

MAJOR_BASES = {
    "BTC",
    "ETH",
    "BNB",
    "SOL",
    "XRP",
    "ADA",
    "DOGE",
    "TON",
    "TRX",
    "LINK",
    "DOT",
    "AVAX",
    "ATOM",
    "MATIC",
    "LTC",
    "SHIB",
    "UNI",
    "XLM",
    "ETC",
}

FUNDING_RATE_FLOOR = -0.01  # -1%
ATR_SPIKE_LOOKBACK = 3
ATR_SPIKE_MULTIPLIER = 3.0
BOTTOM_N = 50
OUTPUT_DIR = Path("data/daily_scans")


@dataclass(frozen=True)
class Candidate:
    symbol: str
    base: str
    timestamp: pd.Timestamp
    quote_volume: float
    market_cap: Optional[float]
    funding_rate: Optional[float]
    ema20: float
    ema30: float
    atr14: float
    latest_close: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Binance USDT perpetuals for short candidates."
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="交易日（UTC）。默认今天。",
    )
    parser.add_argument(
        "--bottom-n",
        type=int,
        default=BOTTOM_N,
        help="倒数 N 名成交额/市值。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="候选列表输出目录。",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        help="K 线周期（默认 1d）。",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.2,
        help="Funding rate 查询之间的延迟秒数，避免触发限频。",
    )
    return parser.parse_args()


def filter_out_majors(
    tickers: pd.DataFrame, meta_map: Dict[str, SymbolMetadata]
) -> pd.DataFrame:
    bases = []
    for symbol in tickers["symbol"]:
        base = meta_map.get(symbol).base if symbol in meta_map else symbol.split("/")[0]
        bases.append(base)
    tickers = tickers.copy()
    tickers["base"] = bases
    filtered = tickers[~tickers["base"].isin(MAJOR_BASES)].copy()
    return filtered


def pick_air_coin_pool(
    tickers: pd.DataFrame,
    *,
    bottom_n: int,
) -> Set[str]:
    liquidity_bottom = (
        tickers.sort_values("quote_volume", ascending=True).head(bottom_n)["symbol"]
    )
    market_cap_bottom = (
        tickers.dropna(subset=["market_cap"])
        .sort_values("market_cap", ascending=True)
        .head(bottom_n)["symbol"]
    )
    pool = set(liquidity_bottom) & set(market_cap_bottom)
    if pool:
        logger.info(
            "Air coin pool size %s (intersection of liquidity & market cap bottom %s)",
            len(pool),
            bottom_n,
        )
        return pool
    logger.warning(
        "Market cap data不足，退化为仅按成交额倒数 %s 选取候选池。", bottom_n
    )
    return set(liquidity_bottom)


def ema_cross_filter(history: pd.DataFrame) -> bool:
    last = history.iloc[-1]
    ema20 = last["ema20"]
    ema30 = last["ema30"]
    if pd.isna(ema20) or pd.isna(ema30):
        return False
    return ema20 < ema30


def atr_spike_filter(history: pd.DataFrame) -> bool:
    if len(history) < ATR_SPIKE_LOOKBACK + 1:
        return True
    atr_series = history["atr14"].dropna()
    if len(atr_series) < ATR_SPIKE_LOOKBACK + 1:
        return True
    recent = atr_series.iloc[-1]
    prev_mean = atr_series.iloc[-(ATR_SPIKE_LOOKBACK + 1) : -1].mean()
    if prev_mean <= 0:
        return True
    return recent <= ATR_SPIKE_MULTIPLIER * prev_mean


def fetch_funding_rate(
    fetcher: BinanceDataFetcher, symbol: str
) -> Optional[float]:
    try:
        rate = fetcher.exchange.fetch_funding_rate(symbol)
    except Exception as exc:  # pragma: no cover - ccxt runtime
        logger.warning("Funding rate query failed for %s: %s", symbol, exc)
        return None
    return rate.get("fundingRate")


def latest_kdj_j_above_threshold(
    fetcher: BinanceDataFetcher,
    symbol: str,
    *,
    threshold: float = 90.0,
    hours_lookback: int = 72,
) -> bool:
    """
    拉取近若干小时（默认 72h）的 1h K 线，计算 KDJ，并判断最新 J 是否高于阈值。
    使用 pandas-ta 的 stoch 生成 K/D，再计算 J = 3K - 2D。
    """
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours_lookback)
        frame = fetcher.fetch_klines(
            symbol, start=start, end=end, timeframe="1h", limit=500
        )
    except Exception as exc:  # pragma: no cover - 网络调用
        logger.warning("1h KDJ 拉取失败 %s: %s", symbol, exc)
        return False
    if frame.empty:
        return False
    stoch = ta.stoch(
        high=frame["high"],
        low=frame["low"],
        close=frame["close"],
        k=9,
        d=3,
        smooth_k=3,
    )
    if stoch is None or stoch.empty:
        return False
    k = stoch.iloc[:, 0]
    d = stoch.iloc[:, 1]
    j = 3 * k - 2 * d
    latest_j = j.dropna().iloc[-1] if not j.dropna().empty else None
    if latest_j is None:
        return False
    return latest_j > threshold


def build_candidates(
    fetcher: BinanceDataFetcher,
    symbols: Iterable[str],
    meta_map: Dict[str, SymbolMetadata],
    *,
    timeframe: str,
    funding_cooldown: float,
) -> List[Candidate]:
    histories = fetcher.fetch_bulk_history(symbols, timeframe=timeframe)
    rows: List[Candidate] = []
    for symbol, history in histories.items():
        if history.empty or not ema_cross_filter(history):
            continue
        if not atr_spike_filter(history):
            continue
        if not latest_kdj_j_above_threshold(fetcher, symbol, threshold=90.0):
            continue
        funding = fetch_funding_rate(fetcher, symbol)
        if funding is not None and funding < FUNDING_RATE_FLOOR:
            logger.debug("%s rejected: funding %.4f", symbol, funding)
            time.sleep(funding_cooldown)
            continue

        last = history.iloc[-1]
        meta = meta_map[symbol]
        rows.append(
            Candidate(
                symbol=symbol,
                base=meta.base,
                timestamp=pd.Timestamp(last["timestamp"]).tz_convert("UTC"),
                quote_volume=float("nan"),
                market_cap=None,  # will fill later
                funding_rate=funding,
                ema20=float(last["ema20"]),
                ema30=float(last["ema30"]),
                atr14=float(last["atr14"]),
                latest_close=float(last["close"]),
            )
        )
        time.sleep(funding_cooldown)
    return rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    candidate_df = run_scan(
        as_of=as_of,
        bottom_n=args.bottom_n,
        timeframe=args.timeframe,
        funding_cooldown=args.cooldown,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"candidates_{as_of.strftime('%Y%m%d')}.csv"
    candidate_df.to_csv(output_path, index=False)
    print(f"Wrote {len(candidate_df)} candidates -> {output_path}")


def run_scan(
    *,
    as_of: date,
    bottom_n: int,
    timeframe: str,
    funding_cooldown: float,
    fetcher: Optional[BinanceDataFetcher] = None,
) -> pd.DataFrame:
    fetcher = fetcher or BinanceDataFetcher()
    metas = fetcher.fetch_usdt_perp_symbols()
    meta_map = {meta.symbol: meta for meta in metas}
    ticker_symbols = [meta.symbol for meta in metas]
    tickers = fetcher.fetch_24h_tickers(ticker_symbols)
    tickers = filter_out_majors(tickers, meta_map)
    air_pool = pick_air_coin_pool(tickers, bottom_n=bottom_n)
    if not air_pool:
        logger.error("没有符合要求的空气币池，终止。")
        sys.exit(1)

    candidates = build_candidates(
        fetcher,
        air_pool,
        meta_map,
        timeframe=timeframe,
        funding_cooldown=funding_cooldown,
    )
    if not candidates:
        logger.warning("今天没有符合策略条件的标的。")
    candidate_df = pd.DataFrame([c.__dict__ for c in candidates])
    if not candidate_df.empty:
        # 回填 ticker 中的报价 & 市值信息
        candidate_df = candidate_df.merge(
            tickers[["symbol", "quote_volume", "market_cap"]].rename(
                columns={
                    "quote_volume": "ticker_quote_volume",
                    "market_cap": "ticker_market_cap",
                }
            ),
            on="symbol",
            how="left",
        )
        candidate_df["quote_volume"] = candidate_df["ticker_quote_volume"]
        candidate_df["market_cap"] = candidate_df["ticker_market_cap"]
        candidate_df.drop(
            columns=["ticker_quote_volume", "ticker_market_cap"], inplace=True
        )
    candidate_df["as_of"] = as_of.strftime("%Y-%m-%d")
    return candidate_df


if __name__ == "__main__":
    main()

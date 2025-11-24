#!/usr/bin/env python3
"""
Module 2: Daily scanner that identifies the weakest Binance USDT perpetuals.

筛选步骤（必须按顺序执行）：
1. 过滤主流币：直接剔除 BTC/ETH/BNB/SOL/XRP 等大市值资产。
2. 流动性过滤：按24小时成交额倒数排序，选取底部100个标的作为"空气币池"。
3. 入场信号：候选池中，最新日线满足 EMA10 < EMA20 < EMA30 的标的。
4. 风险控制：资金费率不低于 -1%，且 ATR14 没有在最近 3 天暴涨
   （今天 ATR > 3 * 前 3 天均值）。

脚本会生成 data/daily_scans/candidates_YYYYMMDD.csv，供回测/实时交易使用。
"""
from __future__ import annotations

import argparse
import logging
import sys
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

from scripts.data_fetcher import BinanceDataFetcher, SymbolMetadata
from scripts.indicator_utils import compute_kdj

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
BOTTOM_N = 100
OUTPUT_DIR = Path("data/daily_scans")


@dataclass(frozen=True)
class Candidate:
    symbol: str
    base: str
    timestamp: pd.Timestamp
    quote_volume: float
    market_cap: Optional[float]
    funding_rate: Optional[float]
    ema10: float
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
    """
    按照24小时成交额倒数排序，选取底部N个标的作为"空气币池"。
    """
    liquidity_bottom = (
        tickers.sort_values("quote_volume", ascending=True).head(bottom_n)["symbol"]
    )
    logger.info(
        "按成交额倒数选取候选池，共 %s 个标的（bottom-%s）",
        len(liquidity_bottom),
        bottom_n,
    )
    return set(liquidity_bottom)


def ema_cross_filter(history: pd.DataFrame) -> bool:
    last = history.iloc[-1]
    ema10 = last["ema10"]
    ema20 = last["ema20"]
    ema30 = last["ema30"]
    if pd.isna(ema10) or pd.isna(ema20) or pd.isna(ema30):
        return False
    return ema10 < ema20 < ema30


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
    as_of_date: Optional[date] = None,
) -> bool:
    """
    拉取近若干小时（默认 72h）的 1h K 线，计算 KDJ，并判断最新 J 是否高于阈值。
    使用 pandas-ta 的 stoch 生成 K/D，再计算 J = 3K - 2D。

    如果指定 as_of_date，则只使用该日期及之前的数据（避免数据泄露）。
    """
    try:
        if as_of_date is not None:
            # 历史模式：截止到 as_of_date 结束
            end = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)
        else:
            # 实时模式：使用当前时间
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

    # 如果指定了 as_of_date，再次过滤确保不使用未来数据
    if as_of_date is not None:
        frame = frame[frame["timestamp"].dt.date <= as_of_date]
        if frame.empty:
            return False

    _, _, j = compute_kdj(frame[["high", "low", "close"]])
    cleaned = j.dropna()
    if cleaned.empty:
        return False
    return cleaned.iloc[-1] > threshold


def build_candidates(
    fetcher: BinanceDataFetcher,
    symbols: Iterable[str],
    meta_map: Dict[str, SymbolMetadata],
    *,
    timeframe: str,
    funding_cooldown: float,
    as_of_date: date,
) -> List[Candidate]:
    """
    构建候选列表，使用截止到 as_of_date 的历史数据。

    重要：为避免数据泄露，只使用 as_of_date 及之前的数据。
    """
    # 获取历史数据，截止到 as_of_date
    end_dt = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=200)  # 足够计算EMA30和ATR14

    histories = fetcher.fetch_bulk_history(
        symbols,
        start=start_dt,
        end=end_dt,
        timeframe=timeframe
    )
    rows: List[Candidate] = []

    for symbol, history in histories.items():
        if history.empty:
            continue

        # 只保留 <= as_of_date 的数据
        history = history[history["timestamp"].dt.date <= as_of_date].copy()
        if history.empty:
            continue

        # 检查 EMA 条件
        if not ema_cross_filter(history):
            continue
        if not atr_spike_filter(history):
            continue

        # 候选池不再使用KDJ过滤，由小时级信号检测负责
        # 注释原因：扩大候选池规模，让更多标的进入观察范围
        # if not latest_kdj_j_above_threshold(
        #     fetcher, symbol, threshold=80.0, as_of_date=as_of_date
        # ):
        #     continue

        # 获取资金费率（注意：这里获取的是当前的费率，历史费率难以获取）
        funding = fetch_funding_rate(fetcher, symbol)
        if funding is not None and funding < FUNDING_RATE_FLOOR:
            logger.debug("%s rejected: funding %.4f", symbol, funding)
            # time.sleep(funding_cooldown)  # 已禁用：提升数据抓取速度
            continue

        last = history.iloc[-1]
        meta = meta_map[symbol]

        # 使用 as_of_date 作为信号时间戳，避免使用未来数据
        signal_timestamp = pd.Timestamp(as_of_date, tz=timezone.utc)

        rows.append(
            Candidate(
                symbol=symbol,
                base=meta.base,
                timestamp=signal_timestamp,  # 使用信号日期而非K线时间戳
                quote_volume=float("nan"),  # 将在 run_scan 中回填
                market_cap=None,
                funding_rate=funding,
                ema10=float(last["ema10"]),
                ema20=float(last["ema20"]),
                ema30=float(last["ema30"]),
                atr14=float(last["atr14"]),
                latest_close=float(last["close"]),
            )
        )
        # time.sleep(funding_cooldown)  # 已禁用：提升数据抓取速度
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
        as_of_date=as_of,
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

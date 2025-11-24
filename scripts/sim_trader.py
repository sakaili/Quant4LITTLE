#!/usr/bin/env python3
"""
模拟盘下单规划脚本（基于 live_trader 信号逻辑，默认只生成计划不实际下单）�?
流程�?1) 每月池：加载或刷新成交额/市值倒数 N（默�?100），剔除资金费率为负的币�?2) 日线信号：EMA10 < EMA20 < EMA30�?3) 小时信号：最�?30 �?1h K 线，取最新一�?KDJ J 值，要求 J>90�?4) 生成做空计划（maker 价：买一或买五），落盘到 data/sim_orders/sim_orders_<timestamp>.csv�?
注意：仅输出计划，不实际下单。若要实盘，可在 live_trader 基础上接�?API key 并去�?paper 限制�?"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# 确保脚本模式下可以找到项目内模块
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.data_fetcher import BinanceDataFetcher
from scripts.daily_candidate_scan import filter_out_majors, pick_air_coin_pool
from scripts.live_trader import (
    apply_proxies,
    hourly_kdj_levels,
    fetch_indicator_frames,
    refresh_monthly_pool,
    load_last_pool,
    maker_price_from_orderbook,
    fetch_funding_rate_safe,
)

logger = logging.getLogger("sim_trader")
OUTPUT_DIR = Path("data/sim_orders")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulated order planner for the short strategy.")
    parser.add_argument("--as-of", type=str, default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--bottom-n", type=int, default=100, help="倒数 N 个成交额/市值的池子大小")
    parser.add_argument("--refresh-pool", action="store_true", help="刷新当月池子")
    parser.add_argument("--use-bid5", action="store_true", help="maker 价使用买五价，否则买一�?)
    parser.add_argument("--notional", type=float, default=5.0, help="单笔名义仓位（USDT�?)
    parser.add_argument("--proxy-http", type=str, default=None, help="HTTP 代理")
    parser.add_argument("--proxy-https", type=str, default=None, help="HTTPS 代理")
    parser.add_argument("--kdj-threshold", type=float, default=90.0, help="最�?1h KDJ J 阈�?)
    parser.add_argument("--use-testnet", action="store_true", help="使用币安期货测试网（模拟盘）")
    return parser.parse_args()


def filter_bottom_with_funding(
    fetcher: BinanceDataFetcher,
    metas: List,
    bottom_n: int,
) -> pd.DataFrame:
    available = {m.symbol for m in metas}
    tickers = fetcher.fetch_24h_tickers([m.symbol for m in metas])
    tickers = filter_out_majors(tickers, {m.symbol: m for m in metas})
    bottom_df = (
        tickers.sort_values("quote_volume", ascending=True)
        .head(bottom_n)
        .loc[:, ["symbol", "quote_volume", "market_cap"]]
    ).copy()
    bottom_df["funding_rate"] = [
        fetch_funding_rate_safe(fetcher.exchange, sym) if sym in available else None
        for sym in bottom_df["symbol"]
    ]
    # 剔除 funding < 0
    bottom_df = bottom_df[
        bottom_df["funding_rate"].isna() | (bottom_df["funding_rate"] >= 0)
    ].reset_index(drop=True)
    return bottom_df


def build_signals(
    fetcher: BinanceDataFetcher,
    symbols: List[str],
    as_of: date,
    *,
    kdj_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_frames, hourly_frames = fetch_indicator_frames(fetcher, symbols, as_of=as_of)

    ema_rows = []
    kdj_rows = []
    signal_rows = []
    levels = [50.0, 60.0, 70.0, 80.0, 90.0]

    for sym in symbols:
        dframe = daily_frames.get(sym)
        if dframe is None or dframe.empty:
            continue
        matches = dframe[dframe["timestamp"].dt.date == as_of]
        if matches.empty:
            continue
        row = matches.iloc[-1]
        ema10 = row.get("ema10_alt")
        ema20 = row.get("ema20_alt")
        ema30 = row.get("ema30_alt")
        if any(pd.isna(x) for x in (ema10, ema20, ema30)):
            continue
        if not (ema10 < ema20 < ema30):
            continue
        ema_rows.append({"symbol": sym, "ema10": float(ema10), "ema20": float(ema20), "ema30": float(ema30)})

        hframe = hourly_frames.get(sym)
        if hframe is None or hframe.empty:
            continue
        recent = hframe.tail(30)
        if recent.empty:
            continue
        last_ts = recent["timestamp"].iloc[-1]
        kdj_map, latest_j = hourly_kdj_levels(recent, levels)
        kdj_row = {
            "symbol": sym,
            "timestamp": last_ts,
            "latest_J": latest_j,
            **{f"J>{int(lvl)}": ("Y" if kdj_map[lvl] else "N") for lvl in levels},
        }
        kdj_rows.append(kdj_row)
        if kdj_map.get(kdj_threshold, False):
            signal_rows.append(
                {
                    **kdj_row,
                    "ema10": float(ema10),
                    "ema20": float(ema20),
                    "ema30": float(ema30),
                }
            )

    ema_df = pd.DataFrame(ema_rows)
    kdj_df = pd.DataFrame(kdj_rows)
    if not kdj_df.empty and "latest_J" in kdj_df.columns:
        kdj_df.sort_values(by="latest_J", ascending=False, inplace=True)
    signal_df = pd.DataFrame(signal_rows)
    return ema_df, kdj_df, signal_df


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    as_of = date.fromisoformat(args.as_of)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fetcher = BinanceDataFetcher(use_testnet=args.use_testnet)
    apply_proxies(fetcher.exchange, args.proxy_http, args.proxy_https)

    if args.refresh_pool:
        pool = refresh_monthly_pool(fetcher, bottom_n=args.bottom_n, as_of=as_of)
    else:
        pool = load_last_pool(as_of)
        if not pool:
            pool = refresh_monthly_pool(fetcher, bottom_n=args.bottom_n, as_of=as_of)

    metas = fetcher.fetch_usdt_perp_symbols()
    bottom_df = filter_bottom_with_funding(fetcher, metas, args.bottom_n)
    logger.info("Step 1 - 成交额倒数�?funding>=0�?s �?, len(bottom_df))
    print(bottom_df.to_string(index=False))

    symbols = bottom_df["symbol"].tolist()
    ema_df, kdj_df, signal_df = build_signals(
        fetcher, symbols, as_of=as_of, kdj_threshold=args.kdj_threshold
    )

    logger.info("Step 2 - EMA10 < EMA20 < EMA30�?s �?, len(ema_df))
    if not ema_df.empty:
        print(ema_df.to_string(index=False))

    logger.info("Step 3 - 最�?1h KDJ（按 J 降序）：%s �?, len(kdj_df))
    if not kdj_df.empty:
        print(kdj_df.to_string(index=False))

    if signal_df.empty:
        logger.info("无符�?J>%.1f + EMA 条件的信号�?, args.kdj_threshold)
        return

    # 生成“计划下单”表
    plans = []
    for _, row in signal_df.iterrows():
        sym = row["symbol"]
        price = maker_price_from_orderbook(fetcher.exchange, sym, args.use_bid5)
        if price is None or price <= 0:
            continue
        qty = args.notional / price
        plans.append(
            {
                "symbol": sym,
                "price": price,
                "qty": qty,
                "notional": args.notional,
                "bid_level": "bid5" if args.use_bid5 else "bid1",
                "latest_J": row.get("latest_J"),
                "ema10": row.get("ema10"),
                "ema20": row.get("ema20"),
                "ema30": row.get("ema30"),
            }
        )

    if not plans:
        logger.info("无可生成的模拟下单计划（可能缺少 orderbook 价格）�?)
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"sim_orders_{ts}.csv"
    pd.DataFrame(plans).to_csv(output_path, index=False)
    logger.info("生成模拟下单计划 %s �?-> %s", len(plans), output_path)
    print(pd.DataFrame(plans).to_string(index=False))


if __name__ == "__main__":
    main()

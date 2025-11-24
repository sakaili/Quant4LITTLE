#!/usr/bin/env python3
"""
Prototype live-trading helper for the short strategy.

Features (maker-only intent):
- 每月首日刷新成交额/市值倒数 100 的池子并落盘：data/monthly_pools/pool_YYYY-MM-DD.csv
- 实时拉取日线 / 小时线，筛选入场信号：日线 EMA10 < EMA20 < EMA30 + ATR 正常，且最近一日的 1h 任意 KDJ J > 90
- 仅生成候选与建议下单价格（买一价或买五价），下单使用 postOnly 保证 maker

用法举例（仅生成信号，不下单）:
    python scripts/live_trader.py --refresh-pool --bottom-n 100 --as-of 2025-11-17
    python scripts/live_trader.py --paper --notional 5 --use-bid5

⚠️ 默认不会自动下单；想要发单请去掉 --paper，并确保 exchange API key 已配置且允许 postOnly。
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Ensure project root is on sys.path when invoked as a script.
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ccxt  # type: ignore
import pandas as pd
import pandas_ta as ta

from scripts.data_fetcher import BinanceDataFetcher, SymbolMetadata
from scripts.daily_candidate_scan import filter_out_majors, pick_air_coin_pool
from scripts.modeling.predictor import RankerPredictor, prepare_candidate_input
from scripts.indicator_utils import compute_kdj

logger = logging.getLogger("live_trader")

FUNDING_RATE_FLOOR = -0.01
BOTTOM_N_DEFAULT = 100
POOL_DIR = Path("data/monthly_pools")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live / paper signal generator (maker-only).")
    parser.add_argument("--as-of", type=str, default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--bottom-n", type=int, default=BOTTOM_N_DEFAULT)
    parser.add_argument("--refresh-pool", action="store_true", help="刷新当月池子")
    parser.add_argument("--use-bid5", action="store_true", help="maker 价使用买五价，否则买一价")
    parser.add_argument("--notional", type=float, default=None, help="固定单笔名义仓位（USDT），为空则按净值百分比计算")
    parser.add_argument("--per-trade-pct", type=float, default=0.01, help="每单占用净值比例（默认1%）")
    parser.add_argument("--leverage", type=float, default=2.0, help="杠杆倍数（默认 2x）")
    parser.add_argument("--min-margin", type=float, default=10.0, help="若无法获取净值或净值过低时的最低初始保证金（USDT）")
    parser.add_argument("--paper", action="store_true", help="仅打印信号，不实际下单")
    parser.add_argument("--proxy-http", type=str, default=None, help="HTTP 代理（如 http://127.0.0.1:7890）")
    parser.add_argument("--proxy-https", type=str, default=None, help="HTTPS 代理")
    parser.add_argument("--use-testnet", action="store_true", help="使用币安期货测试网")
    parser.add_argument("--max-positions", type=int, default=20, help="最大并发新仓位数量（按模型排序截断）")
    parser.add_argument("--ranker-model", type=Path, default=Path("models/rank_model.pt"), help="深度学习排序模型路径")
    parser.add_argument("--ranker-meta", type=Path, default=Path("models/rank_model_meta.json"), help="排序模型元数据")
    parser.add_argument("--ranker-device", type=str, default="cpu", help="模型推理设备（cpu/cuda）")
    parser.add_argument("--disable-ranker", action="store_true", help="跳过深度学习排序阶段")
    return parser.parse_args()


def apply_proxies(exchange: ccxt.Exchange, http: Optional[str], https: Optional[str]) -> None:
    proxies = {}
    if http:
        proxies["http"] = http
    if https:
        proxies["https"] = https
    if proxies:
        exchange.session.proxies.update(proxies)


def refresh_monthly_pool(fetcher: BinanceDataFetcher, bottom_n: int, as_of: date) -> List[str]:
    metas = fetcher.fetch_usdt_perp_symbols()
    meta_map = {m.symbol: m for m in metas}
    available = {m.symbol for m in metas}
    tickers = fetcher.fetch_24h_tickers([m.symbol for m in metas])
    tickers = filter_out_majors(tickers, meta_map)
    pool = pick_air_coin_pool(tickers, bottom_n=bottom_n)
    filtered: List[str] = []
    for symbol in sorted(pool):
        if symbol not in available:
            continue
        rate = fetch_funding_rate_safe(fetcher.exchange, symbol)
        if rate is not None and rate < FUNDING_RATE_FLOOR:
            continue
        filtered.append(symbol)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = tickers[tickers["symbol"].isin(filtered)].copy()
    snapshot["funding_rate"] = snapshot["symbol"].map(
        lambda s: fetch_funding_rate_safe(fetcher.exchange, s)
    )
    snapshot.to_csv(POOL_DIR / f"pool_{as_of.isoformat()}.csv", index=False)
    logger.info("Refreshed pool for %s, size=%s", as_of, len(filtered))
    return filtered


def load_last_pool(as_of: date) -> List[str]:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(POOL_DIR.glob("pool_*.csv"), reverse=True)
    for path in candidates:
        pool_date = date.fromisoformat(path.stem.split("_")[1])
        if pool_date <= as_of:
            df = pd.read_csv(path)
            return df["symbol"].tolist()
    return []


def fetch_indicator_frames(
    fetcher: BinanceDataFetcher,
    symbols: Iterable[str],
    *,
    as_of: date,
    daily_lookback_days: int = 200,
    hourly_lookback_hours: int = 168,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    day_start = datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc)
    prev_day_start = day_start - timedelta(days=1)
    daily_end_dt = prev_day_start + timedelta(days=1)
    now_utc = datetime.now(timezone.utc)
    hourly_end_dt = min(now_utc, day_start + timedelta(days=1))
    start_daily = daily_end_dt - timedelta(days=daily_lookback_days)
    start_hourly = hourly_end_dt - timedelta(hours=hourly_lookback_hours)
    daily = {}
    hourly = {}
    for sym in symbols:
        try:
            d = fetcher.fetch_symbol_history_with_indicators(
                sym,
                start=start_daily,
                end=daily_end_dt,
                timeframe="1d",
                indicators_kwargs={"add_rsi": False, "add_return_90d": False},
            )
            d["ema10_alt"] = ta.ema(d["close"], length=10)
            d["ema20_alt"] = ta.ema(d["close"], length=20)
            d["ema30_alt"] = ta.ema(d["close"], length=30)
            daily[sym] = d
        except Exception as exc:
            logger.warning("Daily fetch failed %s: %s", sym, exc)
        try:
            h = fetcher.fetch_klines(sym, start=start_hourly, end=hourly_end_dt, timeframe="1h")
            hourly[sym] = h
        except Exception as exc:
            logger.warning("Hourly fetch failed %s: %s", sym, exc)
    return daily, hourly


def fetch_funding_rate_safe(exchange: ccxt.Exchange, symbol: str) -> Optional[float]:
    try:
        data = exchange.fetch_funding_rate(symbol)
        rate = data.get("fundingRate")
        return float(rate) if rate is not None else None
    except Exception as exc:  # pragma: no cover - network call
        logger.warning("Funding rate fetch failed for %s: %s", symbol, exc)
        return None


def hourly_kdj_levels(frame: pd.DataFrame, levels: Iterable[float]) -> Tuple[Dict[float, bool], Optional[float]]:
    results = {lvl: False for lvl in levels}
    if frame.empty:
        return results, None
    _, _, j = compute_kdj(frame[["high", "low", "close"]])
    cleaned = j.dropna()
    if cleaned.empty:
        return results, None
    latest_j = cleaned.iloc[-1]
    for lvl in levels:
        results[lvl] = latest_j > lvl
    return results, latest_j


def print_step(label: str, df: pd.DataFrame) -> None:
    if df.empty:
        logger.info("%s: none", label)
    else:
        logger.info("%s:", label)
        print(df.to_string(index=False))


def maker_price_from_orderbook(exchange: ccxt.Exchange, symbol: str, use_bid5: bool) -> Optional[float]:
    ob = exchange.fetch_order_book(symbol)
    bids = ob.get("bids") or []
    if not bids:
        return None
    level = 4 if use_bid5 and len(bids) > 4 else 0
    return float(bids[level][0])


def get_market_price(exchange: ccxt.Exchange, symbol: str) -> Optional[float]:
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker.get("mark") or ticker.get("last") or ticker.get("info", {}).get("markPrice")
        return float(price) if price not in (None, "") else None
    except Exception as exc:  # pragma: no cover - network
        logger.warning("fetch_ticker failed for %s: %s", symbol, exc)
        return None


def apply_ranking(
    signals: List[Tuple[str, Dict[float, bool]]],
    *,
    ranker: Optional[RankerPredictor],
    daily_frames: Dict[str, pd.DataFrame],
    hourly_frames: Dict[str, pd.DataFrame],
    symbol_meta: Dict[str, Dict[str, float]],
    signal_date: date,
    entry_date: date,
    max_positions: int,
) -> Tuple[List[Tuple[str, Dict[float, bool]]], Dict[str, Dict[str, object]]]:
    if ranker is None or not signals:
        return signals, {}
    entry_cutoff = datetime.combine(entry_date, datetime.min.time(), tzinfo=timezone.utc)
    candidates = []
    for sym, _ in signals:
        dframe = daily_frames.get(sym)
        hframe = hourly_frames.get(sym)
        if dframe is None or hframe is None or dframe.empty or hframe.empty:
            continue
        meta = symbol_meta.get(sym, {})
        candidate = prepare_candidate_input(
            symbol=sym,
            signal_date=signal_date,
            entry_cutoff=entry_cutoff,
            daily_history=dframe,
            hourly_history=hframe,
            funding_rate=float(meta.get("funding_rate") or 0.0),
            quote_volume=float(meta.get("quote_volume") or 0.0),
            market_cap=float(meta.get("market_cap") or 0.0),
            seq_len=ranker.seq_len,
        )
        if candidate:
            candidates.append(candidate)
    if not candidates:
        return signals, {}
    ranked = ranker.score(candidates)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    top_symbols = [entry["symbol"] for entry in ranked[:max_positions]]
    ranking_map = {entry["symbol"]: entry for entry in ranked}
    filtered = [item for item in signals if item[0] in top_symbols]
    filtered.sort(key=lambda item: ranking_map[item[0]]["score"], reverse=True)
    return filtered, ranking_map


def place_market_short(
    exchange: ccxt.Exchange,
    symbol: str,
    notional: float,
    position_side: str = "SHORT",
) -> Optional[Dict]:
    price = get_market_price(exchange, symbol)
    if price is None or price <= 0:
        logger.warning("No market price for %s", symbol)
        return None
    qty = notional / price
    params = {"positionSide": position_side}
    try:
        order = exchange.create_order(
            symbol=symbol,
            type="market",
            side="sell",
            amount=qty,
            price=None,
            params=params,
        )
        order["reference_price"] = price
        return order
    except Exception as exc:  # pragma: no cover - live order
        logger.error("Order failed %s: %s", symbol, exc)
        return None


def place_tp_sl(
    exchange: ccxt.Exchange,
    symbol: str,
    qty: float,
    tp_price: float,
    sl_price: float,
    position_side: str = "SHORT",
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Submit reduce-only TP/SL market orders for a short position."""
    take_profit = None
    stop_loss = None

    def submit(order_type: str, stop_price: float) -> Optional[Dict]:
        params = {
            "positionSide": position_side,
            "reduceOnly": True,
            "workingType": "CONTRACT_PRICE",
            "stopPrice": stop_price,
        }
        try:
            return exchange.create_order(
                symbol=symbol,
                type=order_type,
                side="buy",
                amount=qty,
                price=None,
                params=params,
            )
        except Exception as exc:
            if "reduceonly" in str(exc).lower():
                logger.warning("reduceOnly unsupported for %s %s, retrying without it", symbol, order_type)
                params.pop("reduceOnly", None)
                try:
                    return exchange.create_order(
                        symbol=symbol,
                        type=order_type,
                        side="buy",
                        amount=qty,
                        price=None,
                        params=params,
                    )
                except Exception as exc2:
                    logger.error("%s order failed %s: %s", order_type, symbol, exc2)
                    return None
            logger.error("%s order failed %s: %s", order_type, symbol, exc)
            return None

    take_profit = submit("TAKE_PROFIT_MARKET", tp_price)
    stop_loss = submit("STOP_MARKET", sl_price)
    return take_profit, stop_loss


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    as_of = date.fromisoformat(args.as_of)
    daily_signal_date = as_of - timedelta(days=1)
    fetcher = BinanceDataFetcher(use_testnet=args.use_testnet)
    apply_proxies(fetcher.exchange, args.proxy_http, args.proxy_https)
    ranker: Optional[RankerPredictor] = None
    if not args.disable_ranker:
        model_path = Path(args.ranker_model)
        meta_path = Path(args.ranker_meta)
        if model_path.exists() and meta_path.exists():
            try:
                ranker = RankerPredictor(model_path, meta_path, device=args.ranker_device)
                logger.info("Loaded ranking model from %s", model_path)
            except Exception as exc:
                logger.warning("Failed to load ranking model: %s", exc)
        else:
            logger.info("Ranking model not found at %s or %s; skipping ranking stage.", model_path, meta_path)

    if args.refresh_pool:
        pool = refresh_monthly_pool(fetcher, bottom_n=args.bottom_n, as_of=as_of)
    else:
        pool = load_last_pool(as_of)
        if not pool:
            pool = refresh_monthly_pool(fetcher, bottom_n=args.bottom_n, as_of=as_of)

    ticker_snapshot = fetcher.fetch_24h_tickers(pool)
    bottom_df = (
        ticker_snapshot.sort_values("quote_volume", ascending=True)
        .head(args.bottom_n)
        .loc[:, ["symbol", "quote_volume", "market_cap"]]
    ).copy()
    bottom_df["funding_rate"] = [
        fetch_funding_rate_safe(fetcher.exchange, sym) for sym in bottom_df["symbol"]
    ]
    filtered_bottom = bottom_df[
        bottom_df["funding_rate"].isna() | (bottom_df["funding_rate"] >= 0)
    ].reset_index(drop=True)
    print_step("Step 1 - 成交额最低的 100 名（去负费率）", filtered_bottom)
    symbol_meta = filtered_bottom.set_index("symbol").to_dict("index")

    symbols = filtered_bottom["symbol"].tolist()
    daily_frames, hourly_frames = fetch_indicator_frames(fetcher, symbols, as_of=as_of)

    ema_pass_rows = []
    kdj_rows = []
    signals: List[Tuple[str, Dict[float, bool]]] = []
    levels = [50.0, 60.0, 70.0, 80.0, 90.0]

    for sym in symbols:
        dframe = daily_frames.get(sym)
        if dframe is None or dframe.empty:
            continue
        matches = dframe[dframe["timestamp"].dt.date == daily_signal_date]
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
        ema_pass_rows.append(
            {
                "symbol": sym,
                "ema10": float(ema10),
                "ema20": float(ema20),
                "ema30": float(ema30),
            }
        )

        hframe = hourly_frames.get(sym)
        if hframe is None or hframe.empty:
            continue
        recent = hframe.tail(30)
        if recent.empty:
            continue
        last_ts = recent["timestamp"].iloc[-1]
        kdj_map, latest_j = hourly_kdj_levels(recent, levels)
        kdj_rows.append(
            {
                "symbol": sym,
                "timestamp": last_ts,
                "latest_J": latest_j,
                **{f"J>{int(lvl)}": ("Y" if kdj_map[lvl] else "N") for lvl in levels},
            }
        )
        if kdj_map[90.0]:
            signals.append((sym, kdj_map))

    print_step("Step 2 - EMA10 < EMA20 < EMA30", pd.DataFrame(ema_pass_rows))
    kdj_df = pd.DataFrame(kdj_rows)
    if not kdj_df.empty and "latest_J" in kdj_df.columns:
        kdj_df.sort_values(by="latest_J", ascending=False, inplace=True)
    print_step("Step 3 - 小时 KDJ 阈值（按最新 J 排序）", kdj_df)

    signals, ranking_scores = apply_ranking(
        signals,
        ranker=ranker,
        daily_frames=daily_frames,
        hourly_frames=hourly_frames,
        symbol_meta=symbol_meta,
        signal_date=daily_signal_date,
        entry_date=as_of,
        max_positions=args.max_positions,
    )
    if ranking_scores:
        preview = ", ".join(
            f"{sym}:{ranking_scores[sym]['score']:.3f}"
            for sym, _ in signals[: min(len(signals), 5)]
            if sym in ranking_scores
        )
        if preview:
            logger.info("Ranker top picks: %s", preview)

    logger.info("Signals on %s (J>90 + EMA条件): %s", as_of, len(signals))
    if not signals:
        return
    exchange = fetcher.exchange
    # 计算动态名义：净值 * pct * leverage，如指定固定 notional 则使用固定值
    equity = None
    try:
        balance = exchange.fetch_balance()
        equity = balance.get("total", {}).get("USDT") or balance.get("USDT", {}).get("free")
    except Exception as exc:
        logger.warning("无法获取账户净值，使用固定名义或跳过动态名义: %s", exc)

    for sym, levels_map in signals:
        level_info = ", ".join(f"J>{int(lvl)}={'Y' if hit else 'N'}" for lvl, hit in levels_map.items())
        price = get_market_price(exchange, sym)
        if price is None or price <= 0:
            logger.warning("无有效价格，跳过 %s", sym)
            continue
        # 动态名义
        notional = args.notional
        if notional is None and equity:
            margin = float(equity) * args.per_trade_pct
            if margin < args.min_margin:
                margin = args.min_margin
            notional = margin * args.leverage
        if notional is None:
            notional = args.min_margin * args.leverage
            logger.info("使用回退名义 %.2f (%sUSDT * leverage)", notional, args.min_margin)
        if notional < 5:
            logger.warning("名义 %.2f 低于 5 USDT，跳过 %s", notional, sym)
            continue

        tp_price = price * (1 - 0.30)
        sl_price = price * 2.5  # 150% 不利止损
        score_info = ""
        rank_entry = ranking_scores.get(sym)
        if rank_entry:
            score_info = f" score={rank_entry['score']:.3f}"
        msg = (
            f"{sym} price@{'bid5' if args.use_bid5 else 'bid1'}={price} "
            f"[{level_info}] notional={notional:.4f} tp={tp_price:.6f} sl={sl_price:.6f}{score_info}"
        )
        if args.paper:
            logger.info("[PAPER] %s qty=%.6f", msg, notional / price)
        else:
            qty = notional / price
            try:
                fetcher.exchange.set_leverage(int(args.leverage), sym, params={"marginMode": "ISOLATED"})
            except Exception as exc:
                logger.warning("set_leverage failed for %s: %s", sym, exc)
            order = place_market_short(exchange, sym, notional, position_side="SHORT")
            if order is None:
                continue
            tp_order, sl_order = place_tp_sl(
                exchange,
                sym,
                qty,
                tp_price,
                sl_price,
                position_side="SHORT",
            )
            logger.info("[LIVE] %s order=%s tp=%s sl=%s", msg, order, tp_order, sl_order)


if __name__ == "__main__":
    main()

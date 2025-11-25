#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小时级信号检测器 - Hourly Signal Detector

功能:
1. 读取今日候选币池
2. 基于小时线数据检测入场信号
3. 发现信号立即执行交易
4. 支持Transformer模型排序（可选）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.indicator_utils import compute_kdj
from scripts.modeling.features import build_tabular_vector, build_hourly_sequence
from scripts.data_fetcher import BinanceDataFetcher


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加技术指标"""
    df = df.copy()
    
    # EMA
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema30"] = df["close"].ewm(span=30, adjust=False).mean()
    
    # ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(window=14).mean()
    df["atr_pct"] = (df["atr14"] / df["close"]) * 100
    
    # KDJ
    k, d, j = compute_kdj(df)
    df["kdj_k"] = k
    df["kdj_d"] = d
    df["kdj_j"] = j
    
    return df


def detect_signals(
    candidates_file: Path,
    hourly_dir: Path,
    daily_dir: Optional[Path] = None,
    use_model: bool = False,
    model_dir: Optional[Path] = None,
    top_k: int = 10,
    detect_date: date = None,
    auto_fetch: bool = True
) -> pd.DataFrame:
    """检测候选池中的入场信号"""
    if not candidates_file.exists():
        print(f"  候选池文件不存在: {candidates_file}")
        return pd.DataFrame()

    candidates = pd.read_csv(candidates_file)
    print(f"  候选池大小: {len(candidates)}")

    # 自动抓取缺失的数据
    if auto_fetch:
        fetcher = BinanceDataFetcher()
        missing_symbols = []

        for _, row in candidates.iterrows():
            symbol = row["symbol"]
            # 清理文件名：移除/和:字符
            clean_symbol = symbol.replace("/", "").replace(":", "")
            hourly_file = hourly_dir / f"{clean_symbol}_1h.csv"
            if not hourly_file.exists():
                missing_symbols.append(symbol)

        if missing_symbols:
            print(f"  检测到{len(missing_symbols)}个标的缺少小时线数据，开始自动下载...")
            hourly_dir.mkdir(parents=True, exist_ok=True)

            from datetime import timedelta, timezone
            import time
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=30)  # 下载最近30天数据

            for i, symbol in enumerate(missing_symbols):
                try:
                    print(f"    [{i+1}/{len(missing_symbols)}] 下载 {symbol} 小时线数据...", end="")
                    df = fetcher.fetch_klines(symbol, start=start, end=end, timeframe="1h")
                    if not df.empty:
                        # 清理文件名：移除/和:字符
                        clean_symbol = symbol.replace("/", "").replace(":", "")
                        hourly_file = hourly_dir / f"{clean_symbol}_1h.csv"
                        df.to_csv(hourly_file, index=False)
                        print(f" ✓ ({len(df)}根K线)")
                    else:
                        print(f" ✗ (无数据)")
                    # time.sleep(1)  # 已禁用：提升数据抓取速度
                except Exception as e:
                    print(f" ✗ (错误: {e})")
                    # time.sleep(1)  # 已禁用：提升数据抓取速度
                    continue

            print(f"  数据下载完成\n")

    print(f"  开始检测信号...\n")

    signals = []
    stats = {
        "total": len(candidates),
        "no_hourly_file": 0,
        "insufficient_data": 0,
        "kdj_fail": 0,
        "pass": 0,
        "error": 0
    }

    for idx, row in candidates.iterrows():
        symbol = row["symbol"]

        try:
            # 清理文件名：移除/和:字符
            clean_symbol = symbol.replace("/", "").replace(":", "")
            hourly_file = hourly_dir / f"{clean_symbol}_1h.csv"
            if not hourly_file.exists():
                stats["no_hourly_file"] += 1
                continue

            df_hourly = pd.read_csv(hourly_file)
            if len(df_hourly) < 50:
                stats["insufficient_data"] += 1
                continue

            df_hourly = add_indicators(df_hourly)
            latest = df_hourly.iloc[-1]

            # 使用模型模式：跳过KDJ筛选，让模型评估所有标的
            if use_model:
                # 模型模式：收集所有标的数据
                signal_data = {
                    "symbol": symbol,
                    "close": latest["close"],
                    "kdj_j": latest["kdj_j"],
                    "ema10": latest["ema10"],
                    "ema20": latest["ema20"],
                    "ema30": latest["ema30"],
                    "atr_pct": latest["atr_pct"],
                    "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "signal_type": "MODEL"
                }
                signal_data["_hourly_df"] = df_hourly
                # 日线文件名格式：SYMBOL_USDT_USDT_1d.csv (保留下划线)
                daily_filename = symbol.replace("/", "_").replace(":", "_") + "_1d.csv"
                signal_data["_daily_file"] = daily_dir / daily_filename
                signals.append(signal_data)

                if idx < 20:
                    print(f"  [{idx+1:3d}] {symbol:20s} | KDJ={latest['kdj_j']:6.2f} (待模型评分)")
                stats["pass"] += 1
            else:
                # 非模型模式：使用KDJ>70筛选
                kdj_ok = latest["kdj_j"] > 70

                if idx < 20 or kdj_ok:
                    kdj_status = "✓" if kdj_ok else "✗"
                    print(f"  [{idx+1:3d}] {symbol:20s} | KDJ={latest['kdj_j']:6.2f} {kdj_status}")

                if not kdj_ok:
                    stats["kdj_fail"] += 1
                    continue

                stats["pass"] += 1
                signal_data = {
                    "symbol": symbol,
                    "close": latest["close"],
                    "kdj_j": latest["kdj_j"],
                    "ema10": latest["ema10"],
                    "ema20": latest["ema20"],
                    "ema30": latest["ema30"],
                    "atr_pct": latest["atr_pct"],
                    "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "signal_type": "HOURLY"
                }
                signals.append(signal_data)
                print(f"  ✓ 发现信号: {symbol} (KDJ={latest['kdj_j']:.1f})")

        except Exception as e:
            stats["error"] += 1
            continue

    # 打印统计摘要
    print(f"\n  ========== 检测统计 ==========")
    print(f"  总候选标的:     {stats['total']:3d}")
    print(f"  缺少小时数据:   {stats['no_hourly_file']:3d}")
    print(f"  数据不足(<50): {stats['insufficient_data']:3d}")
    if not use_model:
        print(f"  KDJ未达标(<70): {stats['kdj_fail']:3d}")
    print(f"  通过筛选:       {stats['pass']:3d}")
    print(f"  处理错误:       {stats['error']:3d}")
    print(f"  ==============================\n")

    if len(signals) == 0:
        print(f"\n  无可用标的进行评估")
        return pd.DataFrame()

    df_signals = pd.DataFrame(signals)

    # 如果使用模型，进行排序
    if use_model and model_dir and daily_dir and len(df_signals) > 0:
        print(f"\n  使用Transformer模型排序...")
        df_signals = rank_with_model(
            df_signals,
            daily_dir,
            model_dir,
            detect_date or date.today(),
            top_k
        )
    else:
        # 不使用模型，按KDJ降序排列
        df_signals = df_signals.sort_values("kdj_j", ascending=False)
        if len(df_signals) > top_k:
            df_signals = df_signals.head(top_k)

    # 清理临时字段
    df_signals = df_signals.drop(columns=[c for c in df_signals.columns if c.startswith("_")], errors="ignore")

    print(f"\n  最终信号数量: {len(df_signals)}")
    return df_signals


def rank_with_model(
    signals: pd.DataFrame,
    daily_dir: Path,
    model_dir: Path,
    signal_date: date,
    top_k: int
) -> pd.DataFrame:
    """使用Transformer模型对信号进行排序"""
    try:
        from scripts.lightweight_ranker import LightweightRanker

        # 加载模型
        ranker = LightweightRanker(model_dir)
        print(f"  ✓ 模型加载成功")

        scores = []

        for idx, row in signals.iterrows():
            symbol = row["symbol"]

            try:
                # 读取日线数据
                daily_file = row["_daily_file"]
                if not daily_file.exists():
                    print(f"  ⚠️  {symbol}: 缺少日线数据")
                    scores.append({
                        "symbol": symbol,
                        "model_score": -999.0,  # 标记为无效
                        "pred_class": -1,
                        "prob_class0": 0.0,
                        "prob_class1": 0.0,
                        "prob_class2": 0.0,
                    })
                    continue

                df_daily = pd.read_csv(daily_file, parse_dates=["timestamp"])
                df_daily = add_indicators(df_daily)

                # 构建表格特征
                funding_rate = row.get("funding_rate", 0.0)
                quote_volume = row.get("quote_volume", 0.0)
                market_cap = row.get("market_cap", 0.0)

                tabular = build_tabular_vector(
                    df_daily,
                    signal_date,
                    funding_rate=funding_rate,
                    quote_volume=quote_volume,
                    market_cap=market_cap
                )

                if tabular is None:
                    print(f"  ⚠️  {symbol}: 特征构建失败")
                    scores.append({
                        "symbol": symbol,
                        "model_score": -999.0,
                        "pred_class": -1,
                        "prob_class0": 0.0,
                        "prob_class1": 0.0,
                        "prob_class2": 0.0,
                    })
                    continue

                # 构建序列特征 (使用小时线最后24根K线)
                df_hourly = row["_hourly_df"]
                df_hourly = pd.DataFrame(df_hourly) if not isinstance(df_hourly, pd.DataFrame) else df_hourly
                df_hourly["timestamp"] = pd.to_datetime(df_hourly["timestamp"], utc=True)

                cutoff = pd.Timestamp(signal_date, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
                sequence = build_hourly_sequence(df_hourly, cutoff, seq_len=24)

                # 模型推理
                expected_value, pred_class, probs = ranker.predict(tabular, sequence)

                scores.append({
                    "symbol": symbol,
                    "model_score": float(expected_value),
                    "pred_class": int(pred_class),
                    "prob_class0": float(probs[0]),
                    "prob_class1": float(probs[1]),
                    "prob_class2": float(probs[2]),
                })

                print(f"  ✓ {symbol}: score={expected_value:.4f}, class={pred_class}")

            except Exception as e:
                print(f"  ⚠️  {symbol}: 推理失败 - {e}")
                scores.append({
                    "symbol": symbol,
                    "model_score": -999.0,
                    "pred_class": -1,
                    "prob_class0": 0.0,
                    "prob_class1": 0.0,
                    "prob_class2": 0.0,
                })

        # 合并分数
        df_scores = pd.DataFrame(scores)
        signals = signals.merge(df_scores, on="symbol", how="left")

        # 过滤无效信号并排序
        signals = signals[signals["model_score"] > -900]
        signals = signals.sort_values("model_score", ascending=False)

        # 截断Top-K
        if len(signals) > top_k:
            signals = signals.head(top_k)

        print(f"  ✓ 模型排序完成，保留Top-{len(signals)}个信号")
        return signals

    except Exception as e:
        print(f"  ⚠️  模型加载或推理失败: {e}")
        print(f"  回退到传统排序（按KDJ降序）")
        return signals.sort_values("kdj_j", ascending=False).head(top_k)


def main():
    parser = argparse.ArgumentParser(description="Hourly Signal Detector")
    parser.add_argument("--date", type=str, help="检测日期(YYYY-MM-DD)")
    parser.add_argument("--candidates-dir", type=Path, default=Path("data/daily_scans"))
    parser.add_argument("--hourly-dir", type=Path, default=Path("data/hourly_klines"))
    parser.add_argument("--daily-dir", type=Path, default=Path("data/daily_klines"), help="日线数据目录（模型需要）")
    parser.add_argument("--output-dir", type=Path, default=Path("data/signals"), help="信号输出目录")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="模型目录")
    parser.add_argument("--use-model", action="store_true", help="使用Transformer模型排序")
    parser.add_argument("--top-k", type=int, default=10, help="保留Top-K个信号（默认10）")
    parser.add_argument("--execute", action="store_true", help="检测到信号后立即执行交易")
    parser.add_argument("--auto-confirm", action="store_true", help="自动确认交易")

    args = parser.parse_args()

    detect_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    detect_hour = datetime.now().hour

    print(f"{'='*80}")
    print(f"  小时级信号检测")
    print(f"  日期: {detect_date} {detect_hour}:00")
    if args.use_model:
        print(f"  模式: Transformer模型排序 (Top-{args.top_k})")
    else:
        print(f"  模式: 传统KDJ排序 (Top-{args.top_k})")
    print(f"{'='*80}\n")

    # 读取候选池
    candidates_file = args.candidates_dir / f"candidates_{detect_date.strftime('%Y%m%d')}.csv"

    # 检测信号
    signals = detect_signals(
        candidates_file=candidates_file,
        hourly_dir=args.hourly_dir,
        daily_dir=args.daily_dir if args.use_model else None,
        use_model=args.use_model,
        model_dir=args.model_dir if args.use_model else None,
        top_k=args.top_k,
        detect_date=detect_date
    )

    if len(signals) == 0:
        print(f"\n  本小时无交易信号")
        return

    # 保存信号
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signals_filename = f"signals_{detect_date.strftime('%Y%m%d')}_{detect_hour:02d}.csv"
    signals_file = args.output_dir / signals_filename
    signals.to_csv(signals_file, index=False)

    print(f"\n  信号已保存: {signals_file}")
    print(f"  数量: {len(signals)}")

    # 显示信号概要
    if args.use_model and "model_score" in signals.columns:
        print(f"\n  Top-3信号:")
        for idx, row in signals.head(3).iterrows():
            print(f"    {row['symbol']}: score={row['model_score']:.4f}, KDJ={row['kdj_j']:.1f}")
    else:
        print(f"\n  Top-3信号:")
        for idx, row in signals.head(3).iterrows():
            print(f"    {row['symbol']}: KDJ={row['kdj_j']:.1f}")
    print()

    # 是否立即执行交易
    if args.execute:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "live_maker_trader.py"),
            "--signals-file", str(signals_file),
            "--auto-confirm"
        ]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()

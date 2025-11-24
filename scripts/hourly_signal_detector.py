#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小时级信号检测器 - Hourly Signal Detector

功能:
1. 读取今日候选币池
2. 基于小时线数据检测入场信号  
3. 发现信号立即执行交易
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.indicator_utils import compute_kdj


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
    hourly_dir: Path
) -> pd.DataFrame:
    """检测候选池中的入场信号"""
    if not candidates_file.exists():
        print(f"  候选池文件不存在: {candidates_file}")
        return pd.DataFrame()
    
    candidates = pd.read_csv(candidates_file)
    print(f"  候选池大小: {len(candidates)}")
    
    signals = []
    
    for idx, row in candidates.iterrows():
        symbol = row["symbol"]
        
        try:
            hourly_file = hourly_dir / f"{symbol}_1h.csv"
            if not hourly_file.exists():
                continue
            
            df_hourly = pd.read_csv(hourly_file)
            if len(df_hourly) < 50:
                continue
            
            df_hourly = add_indicators(df_hourly)
            latest = df_hourly.iloc[-1]
            
            # 信号条件: KDJ超买 + EMA底部形态 + ATR波动
            kdj_ok = latest["kdj_j"] > 50
            ema_ok = latest["ema10"] < latest["ema20"] < latest["ema30"]
            atr_ok = latest["atr_pct"] >= 2.0
            
            if kdj_ok and ema_ok and atr_ok:
                signals.append({
                    "symbol": symbol,
                    "close": latest["close"],
                    "kdj_j": latest["kdj_j"],
                    "ema10": latest["ema10"],
                    "ema20": latest["ema20"],
                    "ema30": latest["ema30"],
                    "atr_pct": latest["atr_pct"],
                    "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "signal_type": "HOURLY"
                })
                print(f"  发现信号: {symbol} (KDJ={latest['kdj_j']:.1f})")
        
        except Exception as e:
            continue
    
    if len(signals) == 0:
        return pd.DataFrame()
    
    df_signals = pd.DataFrame(signals)
    df_signals = df_signals.sort_values("kdj_j", ascending=False)
    
    print(f"\n  检测到 {len(df_signals)} 个入场信号")
    return df_signals


def main():
    parser = argparse.ArgumentParser(description="Hourly Signal Detector")
    parser.add_argument("--date", type=str, help="检测日期(YYYY-MM-DD)")
    parser.add_argument("--candidates-dir", type=Path, default=Path("data/daily_scans"))
    parser.add_argument("--hourly-dir", type=Path, default=Path("data/hourly_klines"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hourly_signals"))
    parser.add_argument("--execute", action="store_true", help="检测到信号后立即执行交易")
    parser.add_argument("--auto-confirm", action="store_true", help="自动确认交易")
    
    args = parser.parse_args()
    
    detect_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    detect_hour = datetime.now().hour
    
    print(f"{'='*80}")
    print(f"  小时级信号检测")
    print(f"  日期: {detect_date} {detect_hour}:00")
    print(f"{'='*80}\n")
    
    # 读取候选池
    candidates_file = args.candidates_dir / f"candidates_{detect_date.strftime('%Y%m%d')}.csv"
    
    # 检测信号
    signals = detect_signals(
        candidates_file=candidates_file,
        hourly_dir=args.hourly_dir
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
    print(f"  数量: {len(signals)}\n")
    
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

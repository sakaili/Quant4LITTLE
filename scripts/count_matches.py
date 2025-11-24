#!/usr/bin/env python3
"""
统计有多少个扫描文件的候选与回测交易匹配
"""
import sys
from pathlib import Path
from datetime import timedelta
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load backtest
bt = pd.read_csv(ROOT / "data" / "backtest_trades.csv")
bt["entry_date"] = pd.to_datetime(bt["entry_date"]).dt.date
trade_map = {}
for _, row in bt.iterrows():
    key = (row["symbol"], row["entry_date"])
    trade_map[key] = row

print(f"Total backtest trades: {len(bt)}")
print(f"Unique (symbol, entry_date) pairs: {len(trade_map)}")

# Check scan files
scan_dir = ROOT / "data" / "daily_scans"
total_scan_files = 0
non_empty_scan_files = 0
files_with_matches = 0
total_matches = 0

for path in sorted(scan_dir.glob("candidates_*.csv")):
    total_scan_files += 1
    as_of_str = path.stem.split("_")[1]
    as_of = pd.to_datetime(as_of_str).date()
    entry_date = as_of + timedelta(days=1)

    df = pd.read_csv(path)
    if len(df) > 0:
        non_empty_scan_files += 1

    matches = 0
    for _, row in df.iterrows():
        symbol = row["symbol"]
        if (symbol, entry_date) in trade_map:
            matches += 1

    if matches > 0:
        files_with_matches += 1
        total_matches += matches
        print(f"  {path.name}: {len(df)} candidates, {matches} matches")

print(f"\nSummary:")
print(f"  Total scan files: {total_scan_files}")
print(f"  Non-empty scan files: {non_empty_scan_files}")
print(f"  Files with matches: {files_with_matches}")
print(f"  Total matches (training samples): {total_matches}")

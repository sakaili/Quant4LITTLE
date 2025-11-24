#!/usr/bin/env python3
"""
Debug script to understand why paper_trader.py finds no candidates
"""
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.indicator_utils import compute_kdj


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加技术指标到数据框"""
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

    # KDJ
    k, d, j = compute_kdj(df)
    df["kdj_k"] = k
    df["kdj_d"] = d
    df["kdj_j"] = j

    return df


def main():
    date = "2025-11-01"
    daily_dir = Path("data/daily_klines")

    print(f"[DEBUG] Analyzing data for date: {date}")
    print("=" * 60)

    target_date = pd.to_datetime(date)
    stats = {
        "total_files": 0,
        "enough_data": 0,
        "ema_condition": 0,
        "kdj_condition": 0,
        "atr_condition": 0,
        "all_passed": 0,
    }

    failed_details = []
    passed_symbols = []

    # Check first 20 files
    for i, daily_file in enumerate(daily_dir.glob("*.csv")):
        if i >= 20:  # Only check first 20 for debugging
            break

        stats["total_files"] += 1
        symbol = daily_file.stem

        try:
            # Read data
            daily_df = pd.read_csv(daily_file, parse_dates=["timestamp"])
            daily_df = add_indicators(daily_df)

            # Filter to target date (handle timezone)
            daily_mask = daily_df["timestamp"] <= target_date.tz_localize("UTC")

            if daily_mask.sum() < 30:
                failed_details.append((symbol, "not enough data", daily_mask.sum()))
                continue

            stats["enough_data"] += 1

            signal_row = daily_df[daily_mask].iloc[-1]
            history = daily_df[daily_mask]

            # Check EMA condition
            ema10 = signal_row.get("ema10", 0)
            ema20 = signal_row.get("ema20", 0)
            ema30 = signal_row.get("ema30", 0)

            ema_pass = ema10 < ema20 < ema30
            if ema_pass:
                stats["ema_condition"] += 1

            # Check KDJ condition
            kdj_j = signal_row.get("kdj_j", 0)
            kdj_pass = kdj_j >= 70
            if kdj_pass:
                stats["kdj_condition"] += 1

            # Check ATR condition
            atr14 = signal_row.get("atr14", 0)
            close = signal_row.get("close", 1)
            atr_ratio = atr14 / close if close > 0 else 0
            atr_pass = atr_ratio >= 0.02

            if atr_pass:
                stats["atr_condition"] += 1

            # All conditions
            if ema_pass and kdj_pass and atr_pass:
                stats["all_passed"] += 1
                passed_symbols.append(symbol)
                print(f"[PASS] {symbol}: EMA={ema10:.2f}<{ema20:.2f}<{ema30:.2f}, KDJ_J={kdj_j:.1f}, ATR={atr_ratio:.2%}")
            else:
                failed_reason = []
                if not ema_pass:
                    failed_reason.append(f"EMA({ema10:.2f},{ema20:.2f},{ema30:.2f})")
                if not kdj_pass:
                    failed_reason.append(f"KDJ_J={kdj_j:.1f}<70")
                if not atr_pass:
                    failed_reason.append(f"ATR={atr_ratio:.2%}<2%")

                failed_details.append((symbol, " | ".join(failed_reason), None))

        except Exception as e:
            failed_details.append((symbol, f"error: {e}", None))
            continue

    print("\n" + "=" * 60)
    print("STATISTICS:")
    print(f"  Total files checked: {stats['total_files']}")
    print(f"  Enough data (>=30): {stats['enough_data']}")
    print(f"  Passed EMA condition: {stats['ema_condition']}")
    print(f"  Passed KDJ condition (J>=70): {stats['kdj_condition']}")
    print(f"  Passed ATR condition (>=2%): {stats['atr_condition']}")
    print(f"  PASSED ALL: {stats['all_passed']}")

    if passed_symbols:
        print(f"\nPassed symbols: {passed_symbols}")
    else:
        print("\n[!] NO SYMBOLS PASSED ALL CONDITIONS")
        print("\nFailed details (sample):")
        for symbol, reason, data_count in failed_details[:10]:
            if data_count is not None:
                print(f"  {symbol}: {reason} (only {data_count} rows)")
            else:
                print(f"  {symbol}: {reason}")

    # Suggest parameter adjustments
    print("\n" + "=" * 60)
    print("SUGGESTIONS:")

    if stats["kdj_condition"] == 0:
        print("  [!] KDJ_J>=70 is too strict - consider lowering to 60 or 50")

    if stats["ema_condition"] == 0:
        print("  [!] EMA10<EMA20<EMA30 (bottom pattern) - no symbols match")
        print("      Consider inverting: EMA10>EMA20>EMA30 (trending up)")

    if stats["atr_condition"] == 0:
        print("  [!] ATR>=2% is too strict - consider lowering to 1%")


if __name__ == "__main__":
    main()

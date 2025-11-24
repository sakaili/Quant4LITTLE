from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.modeling.features import (
    SEQ_COLUMNS,
    TABULAR_FEATURE_NAMES,
    build_hourly_sequence,
    build_tabular_vector,
)


@dataclass
class Sample:
    symbol: str
    signal_date: date
    entry_date: date
    tabular: np.ndarray
    sequence: np.ndarray
    class_label: int
    value_score: float


def load_backtest_labels(backtest_csv: Path) -> Dict[Tuple[str, date], Dict[str, object]]:
    df = pd.read_csv(backtest_csv)
    df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.date
    df["exit_date"] = pd.to_datetime(df["exit_date"]).dt.date
    mapping: Dict[Tuple[str, date], Dict[str, object]] = {}
    for _, row in df.iterrows():
        key = (row["symbol"], row["entry_date"])
        mapping[key] = row.to_dict()
    return mapping


def parse_signal_file(path: Path) -> Tuple[date, pd.DataFrame]:
    """
    解析候选扫描文件，返回信号日期和候选DataFrame。

    注意：使用文件名中的日期作为信号日期，而不是CSV中的timestamp列，
    因为timestamp可能包含数据泄露。
    """
    as_of_str = path.stem.split("_")[1]
    as_of_date = datetime.strptime(as_of_str, "%Y%m%d").date()
    df = pd.read_csv(path)

    # 如果CSV中有as_of字段，验证其与文件名一致
    if "as_of" in df.columns and not df.empty:
        csv_date = pd.to_datetime(df["as_of"].iloc[0]).date()
        if csv_date != as_of_date:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"文件 {path.name} 中的 as_of ({csv_date}) 与文件名日期 ({as_of_date}) 不一致，使用文件名日期"
            )

    return as_of_date, df


def compute_value_score(trade_row: Dict[str, object]) -> float:
    ret = float(trade_row.get("return_pct", 0.0))
    entry_date = trade_row["entry_date"]
    exit_date = trade_row["exit_date"]
    duration_days = max((exit_date - entry_date).days, 1)
    return ret / (1.0 + duration_days)


def classify_value(value_score: float, thresholds: Tuple[float, float]) -> int:
    low, high = thresholds
    if value_score <= low:
        return 0
    if value_score >= high:
        return 2
    return 1


class OfflineDataBuilder:
    def __init__(
        self,
        *,
        daily_dir: Path,
        hourly_dir: Path,
        seq_len: int = 24,
    ):
        self.daily_dir = daily_dir
        self.hourly_dir = hourly_dir
        self.seq_len = seq_len
        self.daily_cache: Dict[str, pd.DataFrame] = {}
        self.hourly_cache: Dict[str, pd.DataFrame] = {}

    def _load_daily(self, symbol: str) -> Optional[pd.DataFrame]:
        path = self.daily_dir / f"{symbol.replace('/', '_').replace(':', '_')}_1d.csv"
        if symbol not in self.daily_cache:
            if not path.exists():
                return None
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            self.daily_cache[symbol] = df
        return self.daily_cache.get(symbol)

    def _load_hourly(self, symbol: str) -> Optional[pd.DataFrame]:
        path = self.hourly_dir / f"{symbol.replace('/', '_').replace(':', '_')}_1h.csv"
        if symbol not in self.hourly_cache:
            if not path.exists():
                return None
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            self.hourly_cache[symbol] = df
        return self.hourly_cache.get(symbol)

    def build_sample(
        self,
        *,
        symbol: str,
        signal_date: date,
        entry_date: date,
        funding_rate: float,
        quote_volume: float,
        market_cap: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        daily_df = self._load_daily(symbol)
        hourly_df = self._load_hourly(symbol)
        if daily_df is None or hourly_df is None:
            return None
        tabular = build_tabular_vector(
            daily_df,
            signal_date,
            funding_rate=funding_rate,
            quote_volume=quote_volume,
            market_cap=market_cap,
        )
        if tabular is None:
            return None
        cutoff_dt = datetime.combine(entry_date, datetime.min.time(), tzinfo=timezone.utc)
        sequence = build_hourly_sequence(
            hourly_df,
            cutoff_dt,
            seq_len=self.seq_len,
            columns=SEQ_COLUMNS,
        )
        return tabular, sequence


def build_samples(
    *,
    candidates_dir: Path,
    backtest_csv: Path,
    daily_dir: Path,
    hourly_dir: Path,
    seq_len: int = 24,
    value_thresholds: Tuple[float, float] = (-0.2, 0.2),
) -> List[Sample]:
    trade_map = load_backtest_labels(backtest_csv)
    builder = OfflineDataBuilder(daily_dir=daily_dir, hourly_dir=hourly_dir, seq_len=seq_len)
    samples: List[Sample] = []
    for path in sorted(candidates_dir.glob("candidates_*.csv")):
        as_of_date, df = parse_signal_file(path)
        entry_date = as_of_date + timedelta(days=1)
        for _, row in df.iterrows():
            symbol = row["symbol"]
            trade = trade_map.get((symbol, entry_date))
            if trade is None:
                continue
            features = builder.build_sample(
                symbol=symbol,
                signal_date=as_of_date,
                entry_date=entry_date,
                funding_rate=float(row.get("funding_rate") or 0.0),
                quote_volume=float(row.get("quote_volume") or 0.0),
                market_cap=float(row.get("market_cap") or 0.0),
            )
            if features is None:
                continue
            tabular, sequence = features
            value_score = compute_value_score(trade)
            class_label = classify_value(value_score, value_thresholds)
            samples.append(
                Sample(
                    symbol=symbol,
                    signal_date=as_of_date,
                    entry_date=entry_date,
                    tabular=tabular,
                    sequence=sequence,
                    class_label=class_label,
                    value_score=value_score,
                )
            )
    return samples


__all__ = [
    "Sample",
    "build_samples",
    "TABULAR_FEATURE_NAMES",
    "SEQ_COLUMNS",
]

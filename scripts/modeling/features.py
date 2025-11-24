from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SEQ_COLUMNS = ["close", "volume", "ema20", "ema30", "atr14"]
TABULAR_FEATURE_NAMES = [
    "ema10",
    "ema20",
    "ema30",
    "ema10_over_ema30",
    "ema20_over_ema30",
    "atr14",
    "atr14_over_close",
    "close",
    "funding_rate",
    "log_quote_volume",
    "log_market_cap",
    "ret_mean_5d",
    "ret_std_5d",
    "ema10_slope",
    "ema20_slope",
]


def _ensure_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """确保 timestamp 列是 datetime 类型"""
    if "timestamp" in frame.columns:
        # 使用 pd.api.types.is_datetime64_any_dtype 来检查所有 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
            frame = frame.copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _attach_ema(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    if "ema10" not in history.columns:
        history["ema10"] = history["close"].ewm(span=10, adjust=False).mean()
    if "ema20" not in history.columns:
        history["ema20"] = history["close"].ewm(span=20, adjust=False).mean()
    if "ema30" not in history.columns:
        history["ema30"] = history["close"].ewm(span=30, adjust=False).mean()
    if "atr14" not in history.columns:
        high = history["high"]
        low = history["low"]
        close = history["close"]
        tr = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        history["atr14"] = tr.rolling(window=14, min_periods=1).mean()
    return history


def build_tabular_vector(
    history: pd.DataFrame,
    signal_date: date,
    *,
    funding_rate: float = 0.0,
    quote_volume: float = 0.0,
    market_cap: float = 0.0,
) -> Optional[np.ndarray]:
    history = _attach_ema(_ensure_timestamp(history))
    snapshot = history[history["timestamp"].dt.date <= signal_date]
    if snapshot.empty:
        return None
    snapshot = snapshot.iloc[-1]
    prev = history[history["timestamp"].dt.date < signal_date]
    prev = prev.iloc[-1] if not prev.empty else snapshot
    recent = history[history["timestamp"].dt.date <= signal_date].tail(6)
    returns = recent["close"].pct_change().dropna()

    ema10 = float(snapshot.get("ema10", np.nan))
    ema20 = float(snapshot.get("ema20", np.nan))
    ema30 = float(snapshot.get("ema30", np.nan))
    atr14 = float(snapshot.get("atr14", np.nan))
    close = float(snapshot.get("close", np.nan))

    ema10_prev = float(prev.get("ema10", np.nan))
    ema20_prev = float(prev.get("ema20", np.nan))

    vector = np.array(
        [
            ema10,
            ema20,
            ema30,
            _safe_ratio(ema10, ema30) - 1.0,
            _safe_ratio(ema20, ema30) - 1.0,
            atr14,
            _safe_ratio(atr14, close),
            close,
            funding_rate or 0.0,
            np.log1p(max(quote_volume, 0.0)),
            np.log1p(max(market_cap, 0.0)),
            returns.mean() if not returns.empty else 0.0,
            returns.std() if len(returns) > 1 else 0.0,
            ema10 - ema10_prev,
            ema20 - ema20_prev,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_hourly_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "ema20" not in frame.columns:
        frame["ema20"] = frame["close"].ewm(span=20, adjust=False).mean()
    if "ema30" not in frame.columns:
        frame["ema30"] = frame["close"].ewm(span=30, adjust=False).mean()
    if "atr14" not in frame.columns:
        high = frame["high"]
        low = frame["low"]
        close = frame["close"]
        tr = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        frame["atr14"] = tr.rolling(window=14, min_periods=1).mean()
    return frame


def build_hourly_sequence(
    hourly_frame: pd.DataFrame,
    cutoff: datetime,
    *,
    seq_len: int = 24,
    columns: Sequence[str] = SEQ_COLUMNS,
) -> np.ndarray:
    hourly_frame = _ensure_hourly_indicators(_ensure_timestamp(hourly_frame))
    subset = hourly_frame[hourly_frame["timestamp"] < cutoff]
    if subset.empty:
        return np.zeros((seq_len, len(columns)), dtype=np.float32)
    existing_cols = [col for col in columns if col in subset.columns]
    missing_cols = [col for col in columns if col not in subset.columns]
    data = subset[existing_cols].tail(seq_len).copy()
    for col in missing_cols:
        data[col] = 0.0
    data = data[columns]
    arr = data.to_numpy(dtype=np.float32)
    if arr.shape[0] < seq_len:
        pad = np.zeros((seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.vstack([pad, arr])
    return arr


def _safe_ratio(num: float, denom: float) -> float:
    if denom == 0 or np.isnan(num) or np.isnan(denom):
        return 0.0
    return num / denom


__all__ = [
    "SEQ_COLUMNS",
    "TABULAR_FEATURE_NAMES",
    "build_tabular_vector",
    "build_hourly_sequence",
]

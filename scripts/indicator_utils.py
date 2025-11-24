#!/usr/bin/env python3
"""
Indicator helpers shared across modules.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def compute_kdj(
    frame: pd.DataFrame,
    *,
    length: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return (K, D, J) series for the provided OHLC dataframe.

    Implements the classical RSV-based smoothing:
        RSV = (close - lowest_low_n) / (highest_high_n - lowest_low_n) * 100
        K_t = 2/3 * K_{t-1} + 1/3 * RSV_t   (K_0 = 50)
        D_t = 2/3 * D_{t-1} + 1/3 * K_t     (D_0 = 50)
        J_t = 3 * K_t - 2 * D_t
    """
    if frame.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    highest = frame["high"].rolling(window=length, min_periods=1).max()
    lowest = frame["low"].rolling(window=length, min_periods=1).min()
    denom = (highest - lowest).replace(0, pd.NA)
    rsv = ((frame["close"] - lowest) / denom) * 100
    rsv = rsv.fillna(0.0)

    k_vals = []
    d_vals = []
    k_prev = 50.0
    d_prev = 50.0
    alpha = 1.0 / 3.0
    beta = 1.0 - alpha
    for value in rsv:
        k_prev = beta * k_prev + alpha * value
        d_prev = beta * d_prev + alpha * k_prev
        k_vals.append(k_prev)
        d_vals.append(d_prev)

    k = pd.Series(k_vals, index=frame.index, dtype=float)
    d = pd.Series(d_vals, index=frame.index, dtype=float)
    j = 3 * k - 2 * d
    return k, d, j


__all__ = ["compute_kdj"]

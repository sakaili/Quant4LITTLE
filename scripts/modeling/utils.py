from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

SYMBOL_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9]+")


def symbol_to_filename(symbol: str, timeframe: str) -> str:
    """Convert trading pair symbol to a filesystem-safe filename."""
    normalized = SYMBOL_SANITIZE_PATTERN.sub("_", symbol).strip("_")
    return f"{normalized}_{timeframe}.csv"


def resolve_kline_path(base_dir: Path, symbol: str, timeframe: str) -> Path:
    return base_dir / symbol_to_filename(symbol, timeframe)


def load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


__all__ = ["symbol_to_filename", "resolve_kline_path", "load_csv_if_exists"]

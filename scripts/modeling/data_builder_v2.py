"""
改进的数据构建器 v2

直接从回测交易记录生成训练样本，不依赖候选扫描文件。
这样可以确保所有88个回测交易都能生成训练样本。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.modeling.features import (
    SEQ_COLUMNS,
    TABULAR_FEATURE_NAMES,
    build_hourly_sequence,
    build_tabular_vector,
)

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    symbol: str
    signal_date: date
    entry_date: date
    tabular: np.ndarray
    sequence: np.ndarray
    class_label: int
    value_score: float


def compute_value_score(trade_row: Dict[str, object]) -> float:
    """计算价值分数 = 收益率 / (1 + 持仓天数)"""
    ret = float(trade_row.get("return_pct", 0.0))
    entry_date = trade_row["entry_date"]
    exit_date = trade_row["exit_date"]
    duration_days = max((exit_date - entry_date).days, 1)
    return ret / (1.0 + duration_days)


def classify_value(value_score: float, thresholds: Tuple[float, float]) -> int:
    """
    将价值分数分类为3类:
    0 = 差 (value_score <= low)
    1 = 中等 (low < value_score < high)
    2 = 优 (value_score >= high)
    """
    low, high = thresholds
    if value_score <= low:
        return 0
    if value_score >= high:
        return 2
    return 1


class OfflineDataBuilder:
    """离线数据构建器，从本地K线文件构建特征"""

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
        """加载日线数据"""
        path = self.daily_dir / f"{symbol.replace('/', '_').replace(':', '_')}_1d.csv"
        if symbol not in self.daily_cache:
            if not path.exists():
                return None
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            self.daily_cache[symbol] = df
        return self.daily_cache.get(symbol)

    def _load_hourly(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载小时线数据"""
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
        """
        构建单个样本的特征

        Returns:
            (tabular_features, sequence_features) 或 None（如果数据不足）
        """
        daily_df = self._load_daily(symbol)
        hourly_df = self._load_hourly(symbol)

        if daily_df is None or hourly_df is None:
            return None

        # 构建表格特征（基于信号日期）
        tabular = build_tabular_vector(
            daily_df,
            signal_date,
            funding_rate=funding_rate,
            quote_volume=quote_volume,
            market_cap=market_cap,
        )
        if tabular is None:
            return None

        # 构建序列特征（截止到entry_date的凌晨）
        cutoff_dt = datetime.combine(entry_date, datetime.min.time(), tzinfo=timezone.utc)
        sequence = build_hourly_sequence(
            hourly_df,
            cutoff_dt,
            seq_len=self.seq_len,
            columns=SEQ_COLUMNS,
        )

        return tabular, sequence


def build_samples_from_backtest(
    *,
    backtest_csv: Path,
    daily_dir: Path,
    hourly_dir: Path,
    seq_len: int = 24,
    value_thresholds: Tuple[float, float] = (-0.2, 0.2),
    default_funding_rate: float = 0.0,
    default_quote_volume: float = 0.0,
    default_market_cap: float = 0.0,
) -> List[Sample]:
    """
    直接从回测交易记录生成训练样本

    Args:
        backtest_csv: 回测交易CSV文件路径
        daily_dir: 日线K线数据目录
        hourly_dir: 小时线K线数据目录
        seq_len: 序列长度
        value_thresholds: (low, high) 分类阈值
        default_funding_rate: 默认资金费率（如无法获取历史数据）
        default_quote_volume: 默认成交量
        default_market_cap: 默认市值

    Returns:
        训练样本列表
    """
    # 加载回测交易
    bt_df = pd.read_csv(backtest_csv)
    bt_df["entry_date"] = pd.to_datetime(bt_df["entry_date"]).dt.date
    bt_df["exit_date"] = pd.to_datetime(bt_df["exit_date"]).dt.date

    logger.info(f"加载 {len(bt_df)} 条回测交易")

    builder = OfflineDataBuilder(daily_dir=daily_dir, hourly_dir=hourly_dir, seq_len=seq_len)
    samples: List[Sample] = []

    for _, trade in bt_df.iterrows():
        symbol = trade["symbol"]
        entry_date = trade["entry_date"]
        signal_date = entry_date - timedelta(days=1)

        # 构建特征
        features = builder.build_sample(
            symbol=symbol,
            signal_date=signal_date,
            entry_date=entry_date,
            funding_rate=default_funding_rate,
            quote_volume=default_quote_volume,
            market_cap=default_market_cap,
        )

        if features is None:
            logger.warning(
                f"无法为 {symbol} 构建特征 (signal_date={signal_date}, entry_date={entry_date})"
            )
            continue

        tabular, sequence = features

        # 计算标签
        value_score = compute_value_score(trade.to_dict())
        class_label = classify_value(value_score, value_thresholds)

        samples.append(
            Sample(
                symbol=symbol,
                signal_date=signal_date,
                entry_date=entry_date,
                tabular=tabular,
                sequence=sequence,
                class_label=class_label,
                value_score=value_score,
            )
        )

    logger.info(f"成功构建 {len(samples)}/{len(bt_df)} 个训练样本")
    return samples


__all__ = [
    "Sample",
    "build_samples_from_backtest",
    "TABULAR_FEATURE_NAMES",
    "SEQ_COLUMNS",
]

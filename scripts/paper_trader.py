#!/usr/bin/env python3
"""
Paper Trading System - 模拟交易系统

流程:
1. 策略筛选候选标的
2. 模型打分排序
3. 生成交易信号
4. 记录但不实际下单
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.indicator_utils import compute_kdj
from scripts.modeling.features import SEQ_COLUMNS, build_tabular_vector


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


class PaperTrader:
    """模拟交易器"""

    def __init__(
        self,
        daily_dir: Path = Path("data/daily_klines"),
        hourly_dir: Path = Path("data/hourly_klines"),
        model_dir: Path = Path("models"),
        output_dir: Path = Path("data/paper_trading"),
        use_model: bool = True,
    ):
        self.daily_dir = daily_dir
        self.hourly_dir = hourly_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.use_model = use_model

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载模型（如果需要）
        if self.use_model:
            self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            from scripts.lightweight_ranker import LightweightRanker

            self.ranker = LightweightRanker(self.model_dir)
            print("[OK] 模型加载成功")
        except Exception as e:
            print(f"[!]  模型加载失败: {e}")
            print("   将使用传统策略（不含模型）")
            self.use_model = False
            self.ranker = None

    def scan_candidates(self, date: str = None) -> pd.DataFrame:
        """
        策略筛选候选标的

        策略规则:
        1. EMA10 < EMA20 < EMA30 (底部形态)
        2. KDJ_J > 90 (超买反转)
        3. ATR14 相对波动率 > 阈值
        4. 成交量放大
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        print(f"\n[1/4] 策略筛选候选标的 (日期: {date})")

        candidates = []

        # 遍历所有标的
        for daily_file in self.daily_dir.glob("*.csv"):
            try:
                symbol = daily_file.stem

                # 读取日线数据
                daily_df = pd.read_csv(daily_file, parse_dates=["timestamp"])
                daily_df = add_indicators(daily_df)

                # 过滤到指定日期 (处理时区)
                target_date = pd.to_datetime(date).tz_localize("UTC")
                daily_mask = daily_df["timestamp"] <= target_date

                if daily_mask.sum() < 30:
                    continue

                signal_row = daily_df[daily_mask].iloc[-1]
                history = daily_df[daily_mask]

                # 策略规则
                # 1. EMA底部形态
                ema10 = signal_row.get("ema10", 0)
                ema20 = signal_row.get("ema20", 0)
                ema30 = signal_row.get("ema30", 0)

                if not (ema10 < ema20 < ema30):
                    continue

                # 2. KDJ_J超买
                kdj_j = signal_row.get("kdj_j", 0)
                if kdj_j < 50:
                    continue

                # 3. ATR波动率
                atr14 = signal_row.get("atr14", 0)
                close = signal_row.get("close", 1)
                atr_ratio = atr14 / close if close > 0 else 0

                if atr_ratio < 0.02:  # 至少2%的波动率
                    continue

                # 4. 成交量放大（可选）
                volume = signal_row.get("volume", 0)
                avg_volume = history.tail(20)["volume"].mean()
                volume_ratio = volume / avg_volume if avg_volume > 0 else 0

                # 通过筛选
                candidates.append(
                    {
                        "symbol": symbol,
                        "close": close,
                        "ema10": ema10,
                        "ema20": ema20,
                        "ema30": ema30,
                        "kdj_j": kdj_j,
                        "atr14": atr14,
                        "atr_ratio": atr_ratio,
                        "volume": volume,
                        "volume_ratio": volume_ratio,
                        "ema10_rank": ema10,  # 用于传统排序
                    }
                )

            except Exception as e:
                continue

        df = pd.DataFrame(candidates)

        if len(df) == 0:
            print("  [X] 没有符合条件的候选")
            return df

        print(f"  [OK] 策略筛选出 {len(df)} 个候选标的")
        print(f"  [OK] 平均ATR波动率: {df['atr_ratio'].mean():.2%}")
        print(f"  [OK] 平均成交量比: {df['volume_ratio'].mean():.2f}x")

        return df

    def rank_with_model(self, candidates: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        使用模型对候选标的打分排序
        """
        if not self.use_model or self.ranker is None:
            print("\n[2/4] 跳过模型打分（使用传统排序）")
            # 传统排序：按EMA10从低到高
            candidates = candidates.sort_values("ema10_rank", ascending=True)
            candidates["model_score"] = 0.0
            candidates["model_class"] = -1
            return candidates

        print(f"\n[2/4] 模型打分排序")

        scores = []
        seq_len = self.ranker.meta["seq_len"]

        for idx, row in candidates.iterrows():
            symbol = row["symbol"]

            try:
                # 准备特征
                tabular, sequence = self._prepare_features(symbol, date, seq_len)

                if tabular is None:
                    continue

                # 模型推理
                score, pred_class, probs = self.ranker.predict(tabular, sequence)

                scores.append(
                    {
                        "symbol": symbol,
                        "model_score": score,
                        "model_class": pred_class,
                        "prob_class0": probs[0],
                        "prob_class1": probs[1],
                        "prob_class2": probs[2],
                    }
                )

            except Exception as e:
                print(f"  [!]  {symbol} 打分失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not scores:
            print("  [X] 没有成功打分的标的")
            return candidates

        # 合并打分结果
        scores_df = pd.DataFrame(scores)
        candidates = candidates.merge(scores_df, on="symbol", how="left")

        # 按模型分数排序
        candidates = candidates.sort_values("model_score", ascending=False)

        print(f"  [OK] 完成 {len(scores)} 个标的打分")
        print(f"  [OK] 平均分数: {candidates['model_score'].mean():.4f}")
        print(f"  [OK] 预测Class 2 (优): {(candidates['model_class'] == 2).sum()} 个")

        return candidates

    def _prepare_features(self, symbol: str, date: str, seq_len: int):
        """准备模型输入特征"""
        daily_file = self.daily_dir / f"{symbol}.csv"
        hourly_file = self.hourly_dir / f"{symbol.replace('_1d', '_1h')}.csv"

        if not daily_file.exists():
            print(f"  [!]  日线文件不存在: {daily_file}")
            return None, None

        if not hourly_file.exists():
            print(f"  [!]  小时线文件不存在: {hourly_file}")
            return None, None

        # 读取数据
        daily_df = pd.read_csv(daily_file, parse_dates=["timestamp"])
        hourly_df = pd.read_csv(hourly_file, parse_dates=["timestamp"])

        daily_df = add_indicators(daily_df)
        hourly_df = add_indicators(hourly_df)

        # 过滤到指定日期 (处理时区)
        target_date = pd.to_datetime(date).tz_localize("UTC")
        daily_mask = daily_df["timestamp"] <= target_date
        hourly_mask = hourly_df["timestamp"] <= target_date

        if daily_mask.sum() < 30 or hourly_mask.sum() < seq_len:
            return None, None

        signal_row = daily_df[daily_mask].iloc[-1]
        history_daily = daily_df[daily_mask]
        history_hourly = hourly_df[hourly_mask].iloc[-seq_len:]

        # 构建特征
        import numpy as np
        from datetime import datetime

        signal_date = pd.to_datetime(date).date()
        tabular = build_tabular_vector(history_daily, signal_date)
        sequence = history_hourly[SEQ_COLUMNS].values

        return tabular, sequence

    def generate_signals(self, ranked: pd.DataFrame, max_positions: int = 20) -> pd.DataFrame:
        """
        生成交易信号

        规则:
        1. 选择Top K个标的
        2. 如果使用模型，只选择预测Class 2的
        3. 生成买入信号
        """
        print(f"\n[3/4] 生成交易信号 (最多 {max_positions} 个)")

        if len(ranked) == 0:
            print("  [X] 没有候选标的")
            return pd.DataFrame()

        # 过滤条件
        if self.use_model and "model_class" in ranked.columns:
            # 优先选择Class 2（优质）
            class2 = ranked[ranked["model_class"] == 2]

            if len(class2) >= max_positions:
                signals = class2.head(max_positions).copy()
                print(f"  [OK] 选择 {len(signals)} 个Class 2标的")
            else:
                # Class 2不够，补充Class 1
                class1 = ranked[ranked["model_class"] == 1]
                signals = pd.concat([class2, class1]).head(max_positions).copy()
                print(f"  [OK] 选择 {len(class2)} 个Class 2 + {len(signals)-len(class2)} 个Class 1")
        else:
            # 传统策略：直接选Top K
            signals = ranked.head(max_positions).copy()
            print(f"  [OK] 选择Top {len(signals)} 个标的（传统策略）")

        # 添加信号时间
        signals["signal_time"] = datetime.now().isoformat()
        signals["signal_type"] = "SHORT"  # 做空垃圾币策略
        signals["status"] = "PENDING"

        print(f"\n  生成的交易信号:")
        if self.use_model and "model_score" in signals.columns:
            display_cols = ["symbol", "close", "model_score", "model_class", "signal_type"]
            print(signals[display_cols].to_string(index=False))
        else:
            print(signals[["symbol", "close", "ema10", "kdj_j", "signal_type"]].to_string(index=False))

        return signals

    def save_results(self, candidates: pd.DataFrame, ranked: pd.DataFrame, signals: pd.DataFrame, date: str):
        """保存结果"""
        import time

        print(f"\n[4/4] 保存结果")

        date_str = date.replace("-", "")

        # 保存候选标的
        if len(candidates) > 0:
            candidates_file = self.output_dir / f"candidates_{date_str}.csv"
            candidates.to_csv(candidates_file, index=False)
            print(f"  [OK] 候选标的: {candidates_file}")

        # 保存排序结果
        if len(ranked) > 0:
            ranked_file = self.output_dir / f"ranked_{date_str}.csv"
            ranked.to_csv(ranked_file, index=False)
            print(f"  [OK] 排序结果: {ranked_file}")

        # 保存交易信号
        if len(signals) > 0:
            signals_file = self.output_dir / f"signals_{date_str}.csv"
            signals.to_csv(signals_file, index=False)
            print(f"  [OK] 交易信号: {signals_file}")

            # 追加到信号历史（带重试机制）
            history_file = self.output_dir / "signals_history.csv"
            if history_file.exists():
                history = pd.read_csv(history_file)
                history = pd.concat([history, signals], ignore_index=True)
            else:
                history = signals

            # 重试保存（最多3次）
            for attempt in range(3):
                try:
                    history.to_csv(history_file, index=False)
                    print(f"  [OK] 信号历史: {history_file}")
                    break
                except PermissionError as e:
                    if attempt < 2:
                        print(f"  [!]  文件被占用，等待1秒后重试 ({attempt + 1}/3)...")
                        time.sleep(1)
                    else:
                        print(f"  [X] 无法保存信号历史: 文件被其他程序占用")
                        print(f"      请关闭打开 {history_file} 的程序（如Excel）")
                except Exception as e:
                    print(f"  [X] 保存信号历史失败: {e}")
                    break

    def run(self, date: str = None, max_positions: int = 20):
        """运行完整流程"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        print("=" * 60)
        print(f"Paper Trading System - {date}")
        print("=" * 60)

        # 1. 策略筛选
        candidates = self.scan_candidates(date)

        if len(candidates) == 0:
            print("\n[X] 没有候选标的，结束运行")
            return

        # 2. 模型打分
        ranked = self.rank_with_model(candidates, date)

        # 3. 生成信号
        signals = self.generate_signals(ranked, max_positions)

        # 4. 保存结果
        self.save_results(candidates, ranked, signals, date)

        print("\n" + "=" * 60)
        print("[OK] Paper Trading 完成")
        print("=" * 60)

        # 汇总
        print(f"\n汇总:")
        print(f"  策略筛选: {len(candidates)} 个候选")
        print(f"  模型打分: {len(ranked)} 个排序")
        print(f"  交易信号: {len(signals)} 个买入")

        if len(signals) > 0:
            print(f"\n信号详情:")
            print(f"  文件: {self.output_dir}/signals_{date.replace('-', '')}.csv")
            print(f"  Top 5: {signals.head(5)['symbol'].tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Paper Trading System")
    parser.add_argument("--date", type=str, help="交易日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--max-positions", type=int, default=20, help="最大持仓数")
    parser.add_argument("--no-model", action="store_true", help="不使用模型，仅传统策略")
    parser.add_argument("--daily-dir", type=Path, default=Path("data/daily_klines"))
    parser.add_argument("--hourly-dir", type=Path, default=Path("data/hourly_klines"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_trading"))

    args = parser.parse_args()

    trader = PaperTrader(
        daily_dir=args.daily_dir,
        hourly_dir=args.hourly_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        use_model=not args.no_model,
    )

    trader.run(date=args.date, max_positions=args.max_positions)


if __name__ == "__main__":
    main()

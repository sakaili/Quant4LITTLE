#!/usr/bin/env python3
"""
实盘交易记录追踪器

功能：
1. 记录每笔实盘交易的入场/出场
2. 计算实际收益用于持续学习
3. 维护完整的交易历史记录
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


class LiveTradeRecorder:
    """实盘交易记录器"""

    def __init__(self, live_trades_csv: Path = Path("data/live_trades.csv")):
        self.live_trades_csv = live_trades_csv
        self.live_trades_csv.parent.mkdir(parents=True, exist_ok=True)

        # 初始化交易记录文件
        if not self.live_trades_csv.exists():
            self._init_trades_file()

    def _init_trades_file(self):
        """初始化交易记录文件"""
        df = pd.DataFrame(
            columns=[
                "trade_id",
                "symbol",
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                "position_size",
                "pnl",
                "pnl_pct",
                "status",  # open, closed
                "model_score",  # 模型预测分数
                "model_class",  # 模型预测类别
                "actual_class",  # 实际类别（平仓后计算）
                "notes",
            ]
        )
        df.to_csv(self.live_trades_csv, index=False)
        print(f"✓ 初始化交易记录文件: {self.live_trades_csv}")

    def record_entry(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        model_score: float = None,
        model_class: int = None,
        notes: str = "",
    ) -> str:
        """记录开仓"""
        df = pd.read_csv(self.live_trades_csv)

        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        entry_time = datetime.now().isoformat()

        new_trade = {
            "trade_id": trade_id,
            "symbol": symbol,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": None,
            "exit_price": None,
            "position_size": position_size,
            "pnl": None,
            "pnl_pct": None,
            "status": "open",
            "model_score": model_score,
            "model_class": model_class,
            "actual_class": None,
            "notes": notes,
        }

        df = pd.concat([df, pd.DataFrame([new_trade])], ignore_index=True)
        df.to_csv(self.live_trades_csv, index=False)

        print(f"✓ 记录开仓: {trade_id}")
        print(f"  Symbol: {symbol}")
        print(f"  Entry Price: {entry_price}")
        print(f"  Position Size: {position_size}")
        if model_score is not None:
            print(f"  Model Score: {model_score:.4f}")
            print(f"  Model Class: {model_class}")

        return trade_id

    def record_exit(self, trade_id: str, exit_price: float, value_thresholds: tuple = (0.003, 0.012)):
        """记录平仓"""
        df = pd.read_csv(self.live_trades_csv)

        # 找到对应的交易
        idx = df[df["trade_id"] == trade_id].index
        if len(idx) == 0:
            print(f"✗ 未找到交易: {trade_id}")
            return

        idx = idx[0]
        entry_price = df.loc[idx, "entry_price"]
        position_size = df.loc[idx, "position_size"]

        # 计算收益
        pnl = (exit_price - entry_price) * position_size
        pnl_pct = (exit_price - entry_price) / entry_price

        # 根据收益计算实际类别
        low_threshold, high_threshold = value_thresholds
        if pnl_pct < low_threshold:
            actual_class = 0  # 差
        elif pnl_pct < high_threshold:
            actual_class = 1  # 中
        else:
            actual_class = 2  # 优

        # 更新记录
        df.loc[idx, "exit_time"] = datetime.now().isoformat()
        df.loc[idx, "exit_price"] = exit_price
        df.loc[idx, "pnl"] = pnl
        df.loc[idx, "pnl_pct"] = pnl_pct
        df.loc[idx, "status"] = "closed"
        df.loc[idx, "actual_class"] = actual_class

        df.to_csv(self.live_trades_csv, index=False)

        print(f"✓ 记录平仓: {trade_id}")
        print(f"  Exit Price: {exit_price}")
        print(f"  PnL: {pnl:.2f} ({pnl_pct:.2%})")
        print(f"  Actual Class: {actual_class}")

        # 比较模型预测
        model_class = df.loc[idx, "model_class"]
        if pd.notna(model_class):
            model_class = int(model_class)
            correct = model_class == actual_class
            print(f"  Model Predicted: {model_class} ({'✓' if correct else '✗'})")

    def get_open_positions(self) -> pd.DataFrame:
        """获取所有未平仓位置"""
        df = pd.read_csv(self.live_trades_csv)
        return df[df["status"] == "open"]

    def get_closed_trades(self, days: int = None) -> pd.DataFrame:
        """获取已平仓交易"""
        df = pd.read_csv(self.live_trades_csv)
        closed = df[df["status"] == "closed"].copy()

        if days and len(closed) > 0:
            closed["entry_time"] = pd.to_datetime(closed["entry_time"])
            cutoff = datetime.now() - pd.Timedelta(days=days)
            closed = closed[closed["entry_time"] >= cutoff]

        return closed

    def get_model_accuracy(self, days: int = 30) -> dict:
        """计算模型预测准确率"""
        closed = self.get_closed_trades(days=days)

        if len(closed) == 0:
            return {"error": "没有已平仓交易"}

        # 过滤有模型预测的交易
        with_prediction = closed[closed["model_class"].notna()].copy()

        if len(with_prediction) == 0:
            return {"error": "没有模型预测记录"}

        correct = (with_prediction["model_class"] == with_prediction["actual_class"]).sum()
        total = len(with_prediction)
        accuracy = correct / total

        return {
            "period_days": days,
            "total_trades": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "avg_pnl_pct": with_prediction["pnl_pct"].mean(),
            "win_rate": (with_prediction["pnl_pct"] > 0).mean(),
        }

    def summary(self):
        """打印交易汇总"""
        df = pd.read_csv(self.live_trades_csv)

        print("=" * 60)
        print("实盘交易汇总")
        print("=" * 60)

        print(f"\n总交易数: {len(df)}")
        print(f"  未平仓: {(df['status'] == 'open').sum()}")
        print(f"  已平仓: {(df['status'] == 'closed').sum()}")

        closed = df[df["status"] == "closed"]
        if len(closed) > 0:
            print(f"\n已平仓交易统计:")
            print(f"  总盈亏: {closed['pnl'].sum():.2f}")
            print(f"  平均收益率: {closed['pnl_pct'].mean():.2%}")
            print(f"  胜率: {(closed['pnl_pct'] > 0).mean():.2%}")
            print(f"  最大盈利: {closed['pnl_pct'].max():.2%}")
            print(f"  最大亏损: {closed['pnl_pct'].min():.2%}")

            # 模型准确率
            with_pred = closed[closed["model_class"].notna()]
            if len(with_pred) > 0:
                correct = (with_pred["model_class"] == with_pred["actual_class"]).sum()
                print(f"\n模型预测准确率:")
                print(f"  预测交易数: {len(with_pred)}")
                print(f"  正确预测: {correct}")
                print(f"  准确率: {correct / len(with_pred):.2%}")


def main():
    parser = argparse.ArgumentParser(description="实盘交易记录追踪器")
    parser.add_argument("--action", choices=["entry", "exit", "list", "summary", "accuracy"], required=True)
    parser.add_argument("--symbol", type=str, help="交易标的")
    parser.add_argument("--price", type=float, help="价格")
    parser.add_argument("--size", type=float, help="仓位大小", default=1.0)
    parser.add_argument("--trade-id", type=str, help="交易ID（用于平仓）")
    parser.add_argument("--model-score", type=float, help="模型预测分数")
    parser.add_argument("--model-class", type=int, help="模型预测类别")
    parser.add_argument("--notes", type=str, default="", help="备注")
    parser.add_argument("--days", type=int, default=30, help="统计天数")
    args = parser.parse_args()

    recorder = LiveTradeRecorder()

    if args.action == "entry":
        if not args.symbol or not args.price:
            print("错误: 开仓需要 --symbol 和 --price")
            return 1
        recorder.record_entry(
            symbol=args.symbol,
            entry_price=args.price,
            position_size=args.size,
            model_score=args.model_score,
            model_class=args.model_class,
            notes=args.notes,
        )

    elif args.action == "exit":
        if not args.trade_id or not args.price:
            print("错误: 平仓需要 --trade-id 和 --price")
            return 1
        recorder.record_exit(trade_id=args.trade_id, exit_price=args.price)

    elif args.action == "list":
        print("\n未平仓位置:")
        print(recorder.get_open_positions())

        print("\n最近已平仓交易:")
        print(recorder.get_closed_trades(days=args.days))

    elif args.action == "summary":
        recorder.summary()

    elif args.action == "accuracy":
        metrics = recorder.get_model_accuracy(days=args.days)
        print(f"\n最近{args.days}天模型准确率:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    import sys

    sys.exit(main())
